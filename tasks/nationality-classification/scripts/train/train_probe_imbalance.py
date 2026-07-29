"""Imbalance-handling sweep on cached 61-class XLS-R features.

All variants train the SAME linear head on the SAME frozen features (runs/features_fine_811),
so differences are purely from the imbalance technique. Reports test top1/top5/macro-F1,
plus macro-F1 broken down by speaker-count tier (the real bottleneck under speaker-disjoint
eval). Variants:
  baseline      plain cross-entropy
  posthoc_la    baseline logits, post-hoc logit adjustment (Menon ICLR'21) at inference
  la_train      logit-adjusted training loss
  class_balanced effective-number reweighting (Cui CVPR'19)
  oversample    class-balanced sampling (minority oversampled per batch)
  cb_plus_la    class_balanced weights + la_train
Best epoch per variant chosen by VALID macro-F1.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_RUNS


import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load(features_dir, split):
    d = np.load(Path(features_dir) / f"{split}.npz", allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.int64), d["speaker"]


def topk(logits, y, k):
    k = min(k, logits.shape[1])
    idx = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([yi in row for yi, row in zip(y, idx)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default=f"{NAT_RUNS}/features_fine_811")
    ap.add_argument("--out_dir", default=f"{NAT_RUNS}/probe_imbalance_fine_811")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=1.0, help="logit-adjustment strength")
    ap.add_argument("--beta", type=float, default=0.9999, help="class-balanced effective-number")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import f1_score

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    id2label = json.loads((Path(args.features_dir) / "id2label.json").read_text(encoding="utf-8"))
    C = len(id2label); labels = [id2label[str(i)] for i in range(C)]

    Xtr, ytr, spk_tr = load(args.features_dir, "train")
    Xva, yva, _ = load(args.features_dir, "valid")
    Xte, yte, _ = load(args.features_dir, "test")
    dim = Xtr.shape[1]

    # standardize
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd; Xte = (Xte - mu) / sd

    # class priors / counts (train)
    cnt = np.bincount(ytr, minlength=C).astype(np.float64)
    prior = cnt / cnt.sum()
    log_prior = torch.tensor(np.log(np.clip(prior, 1e-12, None)), dtype=torch.float32, device=dev)

    # effective-number class-balanced weights (Cui CVPR'19)
    eff = 1.0 - np.power(args.beta, cnt)
    cb_w = (1.0 - args.beta) / np.clip(eff, 1e-12, None)
    cb_w = cb_w / cb_w.sum() * C  # normalize to mean 1
    cb_w_t = torch.tensor(cb_w, dtype=torch.float32, device=dev)

    # speaker-count tiers (train) for per-tier macro-F1
    spk_per_class = Counter()
    seen = {}
    for c, s in zip(ytr, spk_tr):
        seen.setdefault(int(c), set()).add(str(s))
    nspk = {c: len(s) for c, s in seen.items()}
    def tier(c):
        n = nspk.get(c, 0)
        return "A(>=20)" if n >= 20 else "B(5-19)" if n >= 5 else "C(2-4)" if n >= 2 else "D(1)"

    Xtr_t = torch.from_numpy(Xtr).to(dev); ytr_t = torch.from_numpy(ytr).to(dev)
    n = Xtr_t.shape[0]
    samp_w = torch.tensor(1.0 / cnt[ytr], dtype=torch.float64, device=dev)  # for oversampling

    def make_head():
        torch.manual_seed(args.seed)
        return nn.Sequential(nn.Dropout(0.0), nn.Linear(dim, C)).to(dev)

    def logits_all(head, Xnp):
        head.eval()
        out = []
        with torch.no_grad():
            for i in range(0, Xnp.shape[0], 16384):
                out.append(head(torch.from_numpy(Xnp[i:i + 16384]).to(dev)).cpu())
        return torch.cat(out).numpy()

    def macro_f1(logits, y):
        return f1_score(y, logits.argmax(1), average="macro", zero_division=0)

    def train_variant(name):
        head = make_head()
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        use_cb = name in ("class_balanced", "cb_plus_la")
        use_la = name in ("la_train", "cb_plus_la")
        oversample = name == "oversample"
        weight = cb_w_t if use_cb else None
        best_f1, best_logits_te = -1, None
        for ep in range(1, args.epochs + 1):
            head.train()
            if oversample:
                idx_all = torch.multinomial(samp_w, n, replacement=True)
            else:
                idx_all = torch.randperm(n, device=dev)
            for i in range(0, n, args.batch_size):
                idx = idx_all[i:i + args.batch_size]
                logits = head(Xtr_t[idx])
                if use_la:
                    logits = logits + args.tau * log_prior
                loss = F.cross_entropy(logits, ytr_t[idx], weight=weight)
                opt.zero_grad(); loss.backward(); opt.step()
            vlog = logits_all(head, Xva)
            if use_la:
                vlog = vlog - args.tau * np.log(np.clip(prior, 1e-12, None))
            f1 = macro_f1(vlog, yva)
            if f1 > best_f1:
                best_f1 = f1
                tlog = logits_all(head, Xte)
                if use_la:
                    tlog = tlog - args.tau * np.log(np.clip(prior, 1e-12, None))
                best_logits_te = tlog
        return best_logits_te, head

    variants = ["baseline", "posthoc_la", "la_train", "class_balanced", "oversample", "cb_plus_la"]
    rows = []
    base_te_logits = None
    for name in variants:
        if name == "posthoc_la":
            tlog = base_te_logits - args.tau * np.log(np.clip(prior, 1e-12, None))
        else:
            tlog, _ = train_variant(name)
            if name == "baseline":
                base_te_logits = tlog
        t1, t5, mf1 = topk(tlog, yte, 1), topk(tlog, yte, 5), macro_f1(tlog, yte)
        # per-tier macro-F1 (only classes present in test)
        present = sorted(set(yte.tolist()))
        preds = tlog.argmax(1)
        tier_f1 = {}
        for T in ["A(>=20)", "B(5-19)", "C(2-4)", "D(1)"]:
            cls = [c for c in present if tier(c) == T]
            if cls:
                f1s = f1_score(yte, preds, labels=cls, average="macro", zero_division=0)
                tier_f1[T] = (len(cls), round(float(f1s), 3))
        rows.append({"variant": name, "top1": round(t1, 4), "top5": round(t5, 4),
                     "macro_f1": round(mf1, 4), "tier_f1": tier_f1})
        print(f"[{name:14s}] top1={t1:.4f} top5={t5:.4f} macroF1={mf1:.4f}  tier={tier_f1}", flush=True)

    (out_dir / "imbalance_results.json").write_text(
        json.dumps({"variants": rows, "tau": args.tau, "beta": args.beta,
                    "test_present_classes": len(set(yte.tolist())), "num_classes": C},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== summary (test) ===")
    print(f"{'variant':16s}{'top1':>8s}{'top5':>8s}{'macroF1':>9s}")
    for r in rows:
        print(f"{r['variant']:16s}{r['top1']:>8.4f}{r['top5']:>8.4f}{r['macro_f1']:>9.4f}")
    print(f"\nsaved -> {out_dir/'imbalance_results.json'}")


if __name__ == "__main__":
    main()
