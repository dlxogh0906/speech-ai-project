#!/usr/bin/env python
"""Weighted-sampling (class-rebalancing) sweep on the linear head — keeps ALL data.

Per-sample draw prob p ∝ (1/class_count)^beta:
  beta=0  → natural distribution (≈ baseline)
  beta=1  → full class balance (= cRT)
Sweeps beta, reports 62-class overall + foreign 61-way (top1/top5/macro-F1 + mid/few).
Includes a no-sampling baseline (randperm).

  python weight_sample_sweep.py --betas 0.25 0.5 0.75 1.0 --out runs/wsample_sweep_12s.json
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_OUTPUTS

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import decouple_eval_62class as de

KOR = 61
TC = f"{NAT_OUTPUTS}/feature_resplit_61_merged6_seed42_target_counts"


def train_beta(Xtr, ytr, C, dev, epochs, lr, wd, batch, Xva, yva, beta, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr)
    n = len(ytr)
    model = nn.Linear(Xtr.shape[1], C).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    if beta is None:
        p = None
    else:
        cnt = np.bincount(ytr, minlength=C).astype(np.float64)
        w = (1.0 / np.clip(cnt[ytr], 1, None)) ** beta
        p = w / w.sum()
    best = {"top5": -1.0, "W": None, "b": None}
    for ep in range(1, epochs + 1):
        model.train()
        order = (torch.randperm(n) if p is None
                 else torch.from_numpy(np.random.choice(n, size=n, replace=True, p=p)))
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            opt.zero_grad(set_to_none=True)
            lossf(model(Xt[idx].to(dev)), yt[idx].to(dev)).backward()
            opt.step()
        vlog = de._predict(model, Xva, dev)
        t5 = float(de.topk_hits(vlog, yva, 5).mean())
        if t5 > best["top5"]:
            best = {"top5": t5,
                    "W": model.weight.detach().cpu().numpy().copy(),
                    "b": model.bias.detach().cpu().numpy().copy()}
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"  [beta={beta}] ep{ep}/{epochs} val_top5={t5:.4f}", flush=True)
    return best["W"], best["b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--duration", default="12s")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--few_max", type=int, default=2000)
    ap.add_argument("--mid_max", type=int, default=20000)
    ap.add_argument("--out", default="runs/wsample_sweep_12s.json")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits, labels = de.build_splits(args.duration, Path(TC), Path(TC))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits["train"], splits["valid"], splits["test"]
    C = len(labels)
    Xz, Xvz, Xtz = de.zscore(Xtr, Xva, Xte)
    ftr = ytr[ytr != KOR]
    counts = np.bincount(ftr, minlength=KOR)
    buckets = {
        "few":  {c for c in range(KOR) if counts[c] <= args.few_max},
        "mid":  {c for c in range(KOR) if args.few_max < counts[c] <= args.mid_max},
        "head": {c for c in range(KOR) if counts[c] > args.mid_max},
    }

    results = []
    for beta in [None] + args.betas:        # None = no-sampling baseline (randperm)
        W, b = train_beta(Xz, ytr, C, dev, args.epochs, 1e-3, 1e-4, 4096, Xvz, yva, beta)
        te = Xtz @ W.T + b
        o = de.eval_overall(te, yte)
        f = de.eval_foreign(te, yte, buckets)
        results.append({"beta": beta,
                        "O_top1": o["top1"], "O_top5": o["top5"], "O_mF1": o["macro_f1"],
                        "F_top1": f["top1"], "F_top5": f["top5"], "F_mF1": f["macro_f1"],
                        "F_mid": f["macro_f1_mid"], "F_few": f["macro_f1_few"]})
        print(f"[beta={beta}] done", flush=True)

    print("\n=== WEIGHTED-SAMPLING SWEEP (O=62-class overall, F=foreign 61-way) ===")
    print(f"{'beta':>6} | {'O.top1':>7} {'O.top5':>7} {'O.mF1':>6} | "
          f"{'F.top1':>7} {'F.top5':>7} {'F.mF1':>6} {'F.mid':>6} {'F.few':>6}")
    for r in results:
        beta = "none" if r["beta"] is None else r["beta"]
        print(f"{str(beta):>6} | {r['O_top1']:>7.4f} {r['O_top5']:>7.4f} {r['O_mF1']:>6.4f} | "
              f"{r['F_top1']:>7.4f} {r['F_top5']:>7.4f} {r['F_mF1']:>6.4f} "
              f"{(r['F_mid'] or 0):>6.4f} {(r['F_few'] or 0):>6.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
