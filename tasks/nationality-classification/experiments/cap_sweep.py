#!/usr/bin/env python
"""Train-downsampling (cap-per-class) sweep on the linear head.

For each cap, downsample TRAIN to at most `cap` utts per class (keep all if fewer),
z-score, train linear (best-by-val-top5), evaluate 62-class overall + foreign 61-way
(top1/top5/macro-F1 + mid/few buckets). Includes a no-cap baseline row.

  python cap_sweep.py --caps 1000 2000 3000 4000 5000 --out runs/cap_sweep_12s.json
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
from pathlib import Path

import numpy as np
import torch
import decouple_eval_62class as de  # build_splits/zscore/train_head/eval_*

KOR = 61
TC = f"{NAT_OUTPUTS}/feature_resplit_61_merged6_seed42_target_counts"


def cap_train(X, y, cap, seed=42):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if cap and len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", nargs="+", type=int, default=[1000, 2000, 3000, 4000, 5000])
    ap.add_argument("--duration", default="12s")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--few_max", type=int, default=2000)
    ap.add_argument("--mid_max", type=int, default=20000)
    ap.add_argument("--out", default="runs/cap_sweep_12s.json")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits, labels = de.build_splits(args.duration, Path(TC), Path(TC))
    (Xtr0, ytr0), (Xva, yva), (Xte, yte) = splits["train"], splits["valid"], splits["test"]
    C = len(labels)
    ftr = ytr0[ytr0 != KOR]
    counts = np.bincount(ftr, minlength=KOR)
    buckets = {
        "few":  {c for c in range(KOR) if counts[c] <= args.few_max},
        "mid":  {c for c in range(KOR) if args.few_max < counts[c] <= args.mid_max},
        "head": {c for c in range(KOR) if counts[c] > args.mid_max},
    }

    results = []
    for cap in [None] + args.caps:          # None = no-cap baseline
        Xtr, ytr = (Xtr0, ytr0) if cap is None else cap_train(Xtr0, ytr0, cap)
        Xz, Xvz, Xtz = de.zscore(Xtr, Xva, Xte)
        W, b = de.train_head(Xz, ytr, C, dev, args.epochs, 1e-3, 1e-4, 4096, Xvz, yva,
                             tag=f"cap{cap}")
        te = Xtz @ W.T + b
        o = de.eval_overall(te, yte)
        f = de.eval_foreign(te, yte, buckets)
        results.append({"cap": cap, "n_train": int(len(ytr)),
                        "O_top1": o["top1"], "O_top5": o["top5"], "O_mF1": o["macro_f1"],
                        "F_top1": f["top1"], "F_top5": f["top5"], "F_mF1": f["macro_f1"],
                        "F_mid": f["macro_f1_mid"], "F_few": f["macro_f1_few"]})
        print(f"[cap={cap}] done (n_train={len(ytr)})", flush=True)

    print("\n=== CAP SWEEP (O=62-class overall, F=foreign 61-way) ===")
    print(f"{'cap':>6} {'n_train':>9} | {'O.top1':>7} {'O.top5':>7} {'O.mF1':>6} | "
          f"{'F.top1':>7} {'F.top5':>7} {'F.mF1':>6} {'F.mid':>6} {'F.few':>6}")
    for r in results:
        cap = "none" if r["cap"] is None else r["cap"]
        print(f"{str(cap):>6} {r['n_train']:>9} | {r['O_top1']:>7.4f} {r['O_top5']:>7.4f} {r['O_mF1']:>6.4f} | "
              f"{r['F_top1']:>7.4f} {r['F_top5']:>7.4f} {r['F_mF1']:>6.4f} "
              f"{(r['F_mid'] or 0):>6.4f} {(r['F_few'] or 0):>6.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
