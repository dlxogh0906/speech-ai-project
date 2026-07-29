#!/usr/bin/env python
"""
Classifier-decoupling sweep, EXACTLY matched to train_62class_duration_sweep.py.

It imports train_62class_duration_sweep and reuses its data pipeline verbatim, so
the features/labels/split are byte-identical to your duration sweep:
  - foreign features per duration  : duration_sweep_policy31/sec{08,12,16}/features
  - foreign 61-class labels        : restored via speaker -> original label
  - speaker-disjoint split         : forced from target_counts/speaker_assignment.json
  - native Korean as class 61      : 62-class problem
  - z-score normalization          : mean/std from train (same as your script)

On top of those normalized features it compares 5 classifier heads:
  linear   - AdamW CE, best epoch by val top5  (reproduces your foreign_only_61way)
  tau_norm - reuse linear W, rescale rows by 1/||w||^tau (tau picked on val foreign macro-F1)
  cosine   - L2-normalize features AND weights
  crt      - retrain head with class-balanced sampling
  ncm      - nearest class mean (no training)

Reports, per method: 62-class OVERALL and FOREIGN_ONLY_61WAY (foreign test rows,
logits[:, :61]) with head/mid/few buckets by per-class foreign TRAIN count.

Run (one or more durations):
  python decouple_eval_62class.py --durations 12s \
      --out runs/decouple62_12s.json --few_max 2000 --mid_max 20000

This only reads cached features and trains tiny heads. It does NOT touch the
encoder and is independent of any running fine-tune job.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_OUTPUTS

import argparse
import json
import os
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

# Reuse the exact data pipeline of the duration sweep.
import train_62class_duration_sweep as sweep  # noqa: E402

KOREAN_CLASS_ID = 61  # foreign = 0..60, Korean = 61  (62-class total)


# ---------------------------------------------------------------- data (== sweep)
def build_splits(duration, label_source_dir, split_source_dir):
    """Return {split: (X, y)} and labels[62], identical to train_62class_duration_sweep."""
    spk2id, labels = sweep.build_speaker_to_label_id(label_source_dir)
    target_assignment = sweep.load_target_split_assignment(split_source_dir)
    foreign_dir = sweep.DEFAULT_FOREIGN_FEATURES[duration]
    korean_dir = sweep.DEFAULT_KOREAN_FEATURES[duration]

    all_x, all_y, all_spk, _ = sweep.load_foreign_all(foreign_dir, spk2id)
    missing = sorted(set(target_assignment) - set(all_spk))
    if missing:
        raise RuntimeError(f"{duration}: split speakers missing from features: {missing[:10]}")

    out = {}
    for split in ("train", "valid", "test"):
        mask = np.fromiter(
            (target_assignment[s] == split for s in all_spk), dtype=bool, count=len(all_spk)
        )
        fx, fy = all_x[mask], all_y[mask]
        kx, ky, _ = sweep.load_korean_split(korean_dir, split)
        x = np.concatenate([fx, kx]).astype(np.float32)
        y = np.concatenate([fy, ky]).astype(np.int64)
        out[split] = (x, y)
        print(f"{duration} {split}: X={x.shape} foreign={len(fy)} korean={len(ky)} "
              f"classes={len(np.unique(y))}", flush=True)
    return out, labels


def zscore(train_x, *others):
    mean = train_x.mean(0, keepdims=True)
    std = train_x.std(0, keepdims=True) + 1e-6
    return [(train_x - mean) / std] + [(o - mean) / std for o in others]


# ---------------------------------------------------------------------- metrics
def topk_hits(logits, y, k):
    k = min(k, logits.shape[1])
    top = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return np.array([yi in row for yi, row in zip(y, top)])


def eval_overall(logits, y):
    preds = logits.argmax(1)
    present = sorted(set(y.tolist()))
    return {
        "n": int(len(y)),
        "top1": float((preds == y).mean()),
        "top5": float(topk_hits(logits, y, 5).mean()),
        "macro_f1": float(f1_score(y, preds, labels=present, average="macro", zero_division=0)),
        "classes_present": len(present),
    }


def eval_foreign(logits, y, buckets, target_ids=None):
    """Foreign rows only, first 61 logit columns (== sweep's foreign_only_61way)."""
    m = y != KOREAN_CLASS_ID
    yf = y[m]
    lf = logits[m][:, :KOREAN_CLASS_ID]
    preds = lf.argmax(1)
    present = sorted(set(yf.tolist()))
    out = {
        "n": int(len(yf)),
        "top1": float((preds == yf).mean()),
        "top5": float(topk_hits(lf, yf, 5).mean()),
        "macro_f1": float(f1_score(yf, preds, labels=present, average="macro", zero_division=0)),
        "classes_present": len(present),
    }
    for name, cls in buckets.items():
        labs = sorted(cls & set(present))
        out[f"macro_f1_{name}"] = (
            float(f1_score(yf, preds, labels=labs, average="macro", zero_division=0))
            if labs else None
        )
        out[f"n_{name}"] = len(labs)
    if target_ids:
        tt = {}
        for c in target_ids:
            mk = yf == c
            tt[int(c)] = float((preds[mk] == c).mean()) if mk.any() else None
        out["target_top1"] = tt
    return out


# --------------------------------------------------------------------- training
def _predict(model, X, dev, batch=16384):
    out = []
    model.eval()
    with torch.inference_mode():
        for s in range(0, len(X), batch):
            xb = torch.from_numpy(X[s:s + batch]).to(dev)
            out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out)


def train_head(Xtr, ytr, C, dev, epochs, lr, wd, batch, Xva, yva,
               class_balanced=False, seed=42, tag="linear"):
    """Linear head via AdamW CE; keep weights from the best epoch by valid top5.

    Matches train_62class_duration_sweep (AdamW, lr 1e-3, wd 1e-4, batch 4096,
    epochs 60, best-by-val-top5) so the 'linear' row reproduces foreign_only_61way.
    """
    import time
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt = torch.from_numpy(Xtr)        # kept on CPU; batches moved to GPU (no VRAM clash)
    yt = torch.from_numpy(ytr)
    n = len(ytr)
    model = nn.Linear(Xtr.shape[1], C).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()

    if class_balanced:
        counts = np.bincount(ytr, minlength=C).astype(np.float64)
        p = (1.0 / np.clip(counts[ytr], 1, None))
        p = p / p.sum()
    else:
        p = None

    best = {"top5": -1.0, "W": None, "b": None}
    for ep in range(1, epochs + 1):
        started = time.time()
        model.train()
        loss_sum = 0.0
        if p is None:
            order = torch.randperm(n)
        else:
            order = torch.from_numpy(np.random.choice(n, size=n, replace=True, p=p))
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = Xt[idx].to(dev)
            yb = yt[idx].to(dev)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(idx)
        vlog = _predict(model, Xva, dev)
        t5 = float(topk_hits(vlog, yva, 5).mean())
        t1 = float((vlog.argmax(1) == yva).mean())
        if t5 > best["top5"]:
            best = {
                "top5": t5,
                "W": model.weight.detach().cpu().numpy().copy(),
                "b": model.bias.detach().cpu().numpy().copy(),
            }
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"  [{tag}] epoch={ep:>2}/{epochs} loss={loss_sum/n:.4f} "
                  f"val_top1={t1:.4f} val_top5={t5:.4f} ({time.time()-started:.1f}s)",
                  flush=True)
    return best["W"], best["b"]


def ncm_prototypes(Xtr, ytr, C):
    Xn = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)
    proto = np.zeros((C, Xtr.shape[1]), dtype=np.float32)
    for c in range(C):
        m = ytr == c
        if m.any():
            proto[c] = Xn[m].mean(0)
    return proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)


# ------------------------------------------------------------------------- main
def run_duration(duration, args, dev):
    splits, labels = build_splits(duration, args.label_source_dir, args.split_source_dir)
    C = len(labels)  # 62
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits["train"], splits["valid"], splits["test"]

    # buckets by per-class FOREIGN train count, computed on the ORIGINAL (pre-aug)
    # train so head/mid/few membership is identical with and without augmentation.
    ftr = ytr[ytr != KOREAN_CLASS_ID]
    counts = np.bincount(ftr, minlength=KOREAN_CLASS_ID)
    buckets = {
        "few":  {c for c in range(KOREAN_CLASS_ID) if counts[c] <= args.few_max},
        "mid":  {c for c in range(KOREAN_CLASS_ID) if args.few_max < counts[c] <= args.mid_max},
        "head": {c for c in range(KOREAN_CLASS_ID) if counts[c] > args.mid_max},
    }
    print(f"[buckets foreign] head={len(buckets['head'])} mid={len(buckets['mid'])} "
          f"few={len(buckets['few'])}", flush=True)

    target_ids = [int(t) for t in args.target_ids.split(",")] if args.target_ids else None

    # append augmented features to TRAIN ONLY (valid/test untouched, speaker-disjoint kept)
    if args.extra_train_npz:
        ex_x, ex_y = [], []
        for p in args.extra_train_npz:
            z = np.load(p, allow_pickle=True)
            ex_x.append(z["X"].astype(np.float32))
            ex_y.append(z["y"].astype(np.int64))
        ex_x = np.concatenate(ex_x); ex_y = np.concatenate(ex_y)
        uniq, cnt = np.unique(ex_y, return_counts=True)
        print(f"[+aug] +{len(ex_y)} train rows from {len(args.extra_train_npz)} npz "
              f"({dict(zip(uniq.tolist(), cnt.tolist()))})", flush=True)
        Xtr = np.concatenate([Xtr, ex_x]); ytr = np.concatenate([ytr, ex_y])

    Xtr, Xva, Xte = zscore(Xtr, Xva, Xte)

    def lin(X, W, b):
        return X @ W.T + b

    def taunorm(X, W, tau):
        Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) ** tau + 1e-12)
        return X @ Wn.T

    def cos(X, W):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
        return Xn @ Wn.T

    results = {}

    # 1) linear (reproduces foreign_only_61way)
    print(f"[{duration}] train linear", flush=True)
    W, b = train_head(Xtr, ytr, C, dev, args.epochs, args.lr, args.wd, args.batch, Xva, yva)
    te = lin(Xte, W, b)
    results["linear"] = {"overall": eval_overall(te, yte),
                         "foreign": eval_foreign(te, yte, buckets, target_ids)}

    # 2) tau_norm: pick tau on valid FOREIGN macro-F1
    taus = [float(t) for t in args.taus.split(",")]
    va = None
    best = (None, -1.0)
    for t in taus:
        m = eval_foreign(taunorm(Xva, W, t), yva, buckets)["macro_f1"]
        if m > best[1]:
            best = (t, m)
    results["tau_norm"] = {"best_tau": best[0], "val_foreign_macro_f1": best[1],
                           "overall": eval_overall(taunorm(Xte, W, best[0]), yte),
                           "foreign": eval_foreign(taunorm(Xte, W, best[0]), yte, buckets, target_ids)}
    print(f"[{duration}] tau_norm best tau={best[0]} (val foreign macroF1={best[1]:.4f})", flush=True)

    # 3) cosine
    results["cosine"] = {"overall": eval_overall(cos(Xte, W), yte),
                         "foreign": eval_foreign(cos(Xte, W), yte, buckets, target_ids)}

    # 4) cRT (class-balanced retrain)
    print(f"[{duration}] train cRT (class-balanced)", flush=True)
    Wc, bc = train_head(Xtr, ytr, C, dev, args.epochs, args.lr, args.wd, args.batch,
                        Xva, yva, class_balanced=True, tag="crt")
    te = lin(Xte, Wc, bc)
    results["crt"] = {"overall": eval_overall(te, yte),
                      "foreign": eval_foreign(te, yte, buckets, target_ids)}

    # 5) NCM (no training)
    print(f"[{duration}] NCM prototypes", flush=True)
    proto = ncm_prototypes(Xtr, ytr, C)
    Xte_n = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    te = Xte_n @ proto.T
    results["ncm"] = {"overall": eval_overall(te, yte),
                      "foreign": eval_foreign(te, yte, buckets, target_ids)}

    # report ----------------------------------------------------------------
    cols = ["top1", "top5", "macro_f1", "macro_f1_head", "macro_f1_mid", "macro_f1_few"]
    print(f"\n=== {duration} FOREIGN_ONLY_61WAY (test) ===")
    print(f"{'method':<10} " + " ".join(f"{c:>13}" for c in cols))
    for name, r in results.items():
        t = r["foreign"]
        print(f"{name:<10} " + " ".join(
            f"{(t[c] if t.get(c) is not None else float('nan')):>13.4f}" for c in cols))
    print(f"\n=== {duration} OVERALL 62-class (test) ===")
    for name, r in results.items():
        o = r["overall"]
        print(f"{name:<10} top1={o['top1']:.4f} top5={o['top5']:.4f} macroF1={o['macro_f1']:.4f}")

    if target_ids:
        print(f"\n=== {duration} TARGET-class top1(%) (test) ===")
        print(f"{'method':<10} " + " ".join(f"{labels[c][:9]:>11}" for c in target_ids))
        for name, r in results.items():
            tt = r["foreign"].get("target_top1", {})
            row = " ".join(
                f"{(tt.get(c) * 100 if tt.get(c) is not None else float('nan')):>11.1f}"
                for c in target_ids)
            print(f"{name:<10} {row}")

    return {"duration": duration, "labels": labels,
            "foreign_train_counts": counts.tolist(), "results": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--durations", nargs="+", default=["12s"], choices=["8s", "12s", "16s"])
    ap.add_argument("--label-source-dir", default=str(sweep.LABEL_SOURCE_DIR))
    ap.add_argument("--split-source-dir",
                    default=f"{NAT_OUTPUTS}/feature_resplit_61_merged6_seed42_target_counts")
    ap.add_argument("--out", default="runs/decouple62.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--few_max", type=int, default=2000)
    ap.add_argument("--mid_max", type=int, default=20000)
    ap.add_argument("--taus", default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--extra_train_npz", nargs="*", default=[],
                    help="extra .npz (X,y) appended to TRAIN only, e.g. augmented features")
    ap.add_argument("--target_ids", default="8,9,11,13,15",
                    help="foreign class ids to report per-class top1 (default = 5 aug targets)")
    args = ap.parse_args()

    from pathlib import Path
    args.label_source_dir = Path(args.label_source_dir)
    args.split_source_dir = Path(args.split_source_dir)
    dev = torch.device(args.device)

    all_out = []
    for d in args.durations:
        all_out.append(run_duration(d, args, dev))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"durations": args.durations,
                   "label_source_dir": str(args.label_source_dir),
                   "split_source_dir": str(args.split_source_dir),
                   "few_max": args.few_max, "mid_max": args.mid_max,
                   "by_duration": all_out}, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
