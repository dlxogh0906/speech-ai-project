#!/usr/bin/env python
"""
Classifier-decoupling sweep on cached (frozen) XLS-R features.

Compares five classifier heads on the SAME frozen representation:
  1. linear     - plain instance-balanced linear softmax (your current baseline)
  2. tau_norm   - take the trained linear weights, rescale rows by 1/||w||^tau
  3. cosine     - L2-normalize features AND weights (cosine classifier)
  4. crt        - retrain the head only, with class-balanced sampling
  5. ncm        - nearest class mean (NO training): class-mean prototypes + cosine

For each method it reports top-1 / top-5 / macro-F1, OVERALL and split into
head / mid / few-shot buckets (by per-class train count). Method and tau are
selected on the validation split, final numbers reported on test.

Inputs: per-split .npz caches (train/valid/test) each holding a 2D float
feature matrix and a 1D label vector. Array keys are auto-detected; override
with --feat_key / --label_key if detection picks the wrong one.

Usage:
  python decouple_eval.py \
      --features_dir runs/features_62class_12s \
      --out runs/decouple_62class.json \
      --linear_epochs 40 --device cuda

This does NOT touch the encoder and is independent of any running fine-tune job.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import json
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("This script needs torch (use the fairseq_xlsr env).")

try:
    from sklearn.metrics import f1_score
except ImportError:
    sys.exit("This script needs scikit-learn for macro-F1.")


# ----------------------------------------------------------------------------- IO
def _autodetect(npz, feat_key, label_key):
    keys = list(npz.files)
    fk = feat_key
    lk = label_key
    if fk is None:
        for k in keys:
            a = npz[k]
            if a.ndim == 2 and np.issubdtype(a.dtype, np.floating):
                fk = k
                break
    if lk is None:
        # prefer an integer 1D array, else any key that looks like a label
        for k in keys:
            a = npz[k]
            if a.ndim == 1 and np.issubdtype(a.dtype, np.integer):
                lk = k
                break
        if lk is None:
            for k in keys:
                if "lab" in k.lower():
                    lk = k
                    break
    if fk is None or lk is None:
        raise ValueError(f"Could not auto-detect feature/label arrays. keys={keys}. "
                         f"Pass --feat_key / --label_key explicitly.")
    return fk, lk


def load_split(path, feat_key, label_key):
    npz = np.load(path, allow_pickle=True)
    fk, lk = _autodetect(npz, feat_key, label_key)
    X = np.asarray(npz[fk], dtype=np.float32)
    y_raw = np.asarray(npz[lk])
    return X, y_raw, (fk, lk)


def build_label_vocab(*label_arrays):
    uniq = sorted(set().union(*[set(np.asarray(a).tolist()) for a in label_arrays]))
    return {lab: i for i, lab in enumerate(uniq)}, uniq


def to_ids(y_raw, vocab):
    return np.array([vocab[v] for v in np.asarray(y_raw).tolist()], dtype=np.int64)


# -------------------------------------------------------------------------- metrics
def topk_hits(logits, y, k):
    k = min(k, logits.shape[1])
    top = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return np.array([yi in row for yi, row in zip(y, top)])


def evaluate(logits, y, buckets):
    """logits: (N, C) numpy. buckets: dict bucket_name -> set(class_id)."""
    preds = logits.argmax(1)
    present = sorted(set(y.tolist()))
    out = {
        "top1": float((preds == y).mean()),
        "top5": float(topk_hits(logits, y, 5).mean()),
        "macro_f1": float(f1_score(y, preds, labels=present, average="macro", zero_division=0)),
        "n_classes_present": len(present),
    }
    for name, cls in buckets.items():
        labs = sorted(cls & set(present))
        out[f"macro_f1_{name}"] = (
            float(f1_score(y, preds, labels=labs, average="macro", zero_division=0))
            if labs else None
        )
        out[f"n_{name}"] = len(labs)
    return out


# ------------------------------------------------------------------------- training
def train_linear(Xtr, ytr, C, device, epochs, lr, wd, batch, class_balanced=False, seed=42):
    torch.manual_seed(seed)
    dev = torch.device(device)
    d = Xtr.shape[1]
    head = nn.Linear(d, C).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr)
    n = len(ytr)

    if class_balanced:
        counts = np.bincount(ytr, minlength=C).astype(np.float64)
        w = 1.0 / np.clip(counts[ytr], 1, None)
        p = w / w.sum()
    else:
        p = None

    steps = max(1, n // batch)
    for ep in range(epochs):
        if p is None:
            perm = torch.randperm(n)
            for s in range(steps):
                idx = perm[s * batch:(s + 1) * batch]
                xb = Xt[idx].to(dev)
                yb = yt[idx].to(dev)
                opt.zero_grad()
                lossf(head(xb), yb).backward()
                opt.step()
        else:
            for _ in range(steps):
                idx = np.random.choice(n, size=batch, replace=True, p=p)
                xb = Xt[idx].to(dev)
                yb = yt[idx].to(dev)
                opt.zero_grad()
                lossf(head(xb), yb).backward()
                opt.step()
    W = head.weight.detach().cpu().numpy()   # (C, d)
    b = head.bias.detach().cpu().numpy()      # (C,)
    return W, b


def ncm_prototypes(Xtr, ytr, C):
    Xn = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)
    d = Xtr.shape[1]
    proto = np.zeros((C, d), dtype=np.float32)
    for c in range(C):
        m = ytr == c
        if m.any():
            proto[c] = Xn[m].mean(0)
    proto = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    return proto


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True,
                    help="dir containing train.npz / valid.npz / test.npz")
    ap.add_argument("--train", default="train.npz")
    ap.add_argument("--valid", default="valid.npz")
    ap.add_argument("--test", default="test.npz")
    ap.add_argument("--feat_key", default=None)
    ap.add_argument("--label_key", default=None)
    ap.add_argument("--out", default="decouple_results.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--linear_epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4096)
    # head/mid/few buckets by per-class TRAIN sample count (proxy for speaker scarcity)
    ap.add_argument("--few_max", type=int, default=100)
    ap.add_argument("--mid_max", type=int, default=10000)
    ap.add_argument("--taus", default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    args = ap.parse_args()

    import os
    Xtr, ytr_raw, kt = load_split(os.path.join(args.features_dir, args.train), args.feat_key, args.label_key)
    Xva, yva_raw, _ = load_split(os.path.join(args.features_dir, args.valid), args.feat_key, args.label_key)
    Xte, yte_raw, _ = load_split(os.path.join(args.features_dir, args.test), args.feat_key, args.label_key)
    print(f"[keys] feature='{kt[0]}' label='{kt[1]}'")

    vocab, names = build_label_vocab(ytr_raw, yva_raw, yte_raw)
    C = len(vocab)
    ytr = to_ids(ytr_raw, vocab)
    yva = to_ids(yva_raw, vocab)
    yte = to_ids(yte_raw, vocab)
    print(f"[data] train={len(ytr)} valid={len(yva)} test={len(yte)} classes={C}")

    counts = np.bincount(ytr, minlength=C)
    buckets = {
        "few":  {c for c in range(C) if counts[c] <= args.few_max},
        "mid":  {c for c in range(C) if args.few_max < counts[c] <= args.mid_max},
        "head": {c for c in range(C) if counts[c] > args.mid_max},
    }
    print(f"[buckets] head={len(buckets['head'])} mid={len(buckets['mid'])} few={len(buckets['few'])}")

    results = {}

    def logits_linear(X, W, b):
        return X @ W.T + b

    def logits_taunorm(X, W, tau):
        Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) ** tau + 1e-12)
        return X @ Wn.T

    def logits_cosine(X, W):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
        return Xn @ Wn.T

    # --- 1) linear baseline (instance-balanced) -----------------------------
    print("[train] linear (instance-balanced)")
    W, b = train_linear(Xtr, ytr, C, args.device, args.linear_epochs, args.lr, args.wd, args.batch)
    results["linear"] = {"test": evaluate(logits_linear(Xte, W, b), yte, buckets)}

    # --- 2) tau-normalization (reuse W; pick tau on valid by macro-F1) -------
    taus = [float(t) for t in args.taus.split(",")]
    best = (None, -1.0)
    for t in taus:
        m = evaluate(logits_taunorm(Xva, W, t), yva, buckets)["macro_f1"]
        if m > best[1]:
            best = (t, m)
    results["tau_norm"] = {
        "best_tau": best[0],
        "test": evaluate(logits_taunorm(Xte, W, best[0]), yte, buckets),
    }
    print(f"[tau_norm] best tau={best[0]} (valid macroF1={best[1]:.4f})")

    # --- 3) cosine classifier (reuse W) -------------------------------------
    results["cosine"] = {"test": evaluate(logits_cosine(Xte, W), yte, buckets)}

    # --- 4) cRT (retrain head, class-balanced sampling) ---------------------
    print("[train] cRT (class-balanced)")
    Wc, bc = train_linear(Xtr, ytr, C, args.device, args.linear_epochs, args.lr, args.wd,
                          args.batch, class_balanced=True)
    results["crt"] = {"test": evaluate(logits_linear(Xte, Wc, bc), yte, buckets)}

    # --- 5) NCM (no training) -----------------------------------------------
    print("[compute] NCM prototypes")
    proto = ncm_prototypes(Xtr, ytr, C)
    Xte_n = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    results["ncm"] = {"test": evaluate(Xte_n @ proto.T, yte, buckets)}

    # --- report -------------------------------------------------------------
    cols = ["top1", "top5", "macro_f1", "macro_f1_head", "macro_f1_mid", "macro_f1_few"]
    print("\n=== TEST ===")
    print(f"{'method':<10} " + " ".join(f"{c:>13}" for c in cols))
    for name, r in results.items():
        t = r["test"]
        row = " ".join(
            f"{(t[c] if t[c] is not None else float('nan')):>13.4f}" for c in cols
        )
        print(f"{name:<10} {row}")

    with open(args.out, "w") as f:
        json.dump({"label_names": [str(n) for n in names],
                   "class_counts": counts.tolist(),
                   "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
