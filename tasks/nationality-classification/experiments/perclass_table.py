#!/usr/bin/env python
"""Per-class table (class, test_n, train_spk, test_spk, top1, top5) for the LINEAR head.

Reuses decouple_eval_62class's data pipeline (62-class, target_counts split, z-score),
trains the linear head (optionally with --extra_train_npz augmented features appended to
TRAIN), then dumps per-class foreign top1/top5 + Korean, sorted high->low.

  # AFTER (augmented):
  python perclass_table.py --extra_train_npz runs/features_aug_12s/train.npz
  # BEFORE (no aug):
  python perclass_table.py
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_OUTPUTS

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import decouple_eval_62class as de  # reuse build_splits/zscore/train_head/topk_hits
import torch

KOR = 61
TC = f"{NAT_OUTPUTS}/feature_resplit_61_merged6_seed42_target_counts"
KFEAT = f"{DATA_ROOT}/taeho/korean_native_features/sec12"


def speaker_counts():
    """train/test unique-speaker counts per foreign class + Korean (target_counts split)."""
    id2label = json.load(open(os.path.join(TC, "id2label.json"), encoding="utf-8"))
    def per(split):
        z = np.load(os.path.join(TC, f"{split}.npz"), allow_pickle=True)
        y = z["y"]; spk = z["speaker"].astype(str)
        return {int(c): len(np.unique(spk[y == c])) for c in np.unique(y)}
    tr, te = per("train"), per("test")
    ktr = len(np.unique(np.load(os.path.join(KFEAT, "train.npz"), allow_pickle=True)["speaker"].astype(str)))
    kte = len(np.unique(np.load(os.path.join(KFEAT, "test.npz"), allow_pickle=True)["speaker"].astype(str)))
    return id2label, tr, te, ktr, kte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", default="12s")
    ap.add_argument("--extra_train_npz", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits, labels = de.build_splits(args.duration, Path(TC), Path(TC))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits["train"], splits["valid"], splits["test"]
    if args.extra_train_npz:
        xs, ys = [Xtr], [ytr]
        for p in args.extra_train_npz:
            z = np.load(p, allow_pickle=True)
            xs.append(z["X"].astype(np.float32)); ys.append(z["y"].astype(np.int64))
        Xtr = np.concatenate(xs); ytr = np.concatenate(ys)
        print(f"[+aug] train now {len(ytr)} rows", flush=True)
    Xtr, Xva, Xte = de.zscore(Xtr, Xva, Xte)

    W, b = de.train_head(Xtr, ytr, len(labels), dev, args.epochs, 1e-3, 1e-4, 4096, Xva, yva, tag="linear")
    logits = Xte @ W.T + b

    # foreign per-class (61-way: foreign rows, first 61 cols) + Korean (62-way)
    id2label, tr_spk, te_spk, ktr, kte = speaker_counts()
    m = yte != KOR
    yf = yte[m]; lf = logits[m][:, :KOR]
    pf = lf.argmax(1)
    t5 = de.topk_hits(lf, yf, 5)

    rows = []
    for c in range(KOR):
        mk = yf == c
        n = int(mk.sum())
        rows.append({
            "class": id2label[str(c)], "test_n": n,
            "train_spk": tr_spk.get(c, 0), "test_spk": te_spk.get(c, 0),
            "top1": float(pf[mk].__eq__(c).mean()) if n else None,
            "top5": float(t5[mk].mean()) if n else None,
        })
    # Korean (62-way)
    mk = yte == KOR
    rows.append({
        "class": "한국어_모국어(★)", "test_n": int(mk.sum()),
        "train_spk": ktr, "test_spk": kte,
        "top1": float(logits[mk].argmax(1).__eq__(KOR).mean()),
        "top5": float(de.topk_hits(logits[mk], yte[mk], 5).mean()),
    })

    present = [r for r in rows if r["test_n"] > 0]
    absent = [r for r in rows if r["test_n"] == 0]
    present.sort(key=lambda r: r["top1"], reverse=True)

    tag = "AFTER(aug)" if args.extra_train_npz else "BEFORE(base)"
    print(f"\n=== {args.duration} LINEAR per-class [{tag}] ===")
    print(f"{'#':>3} {'class':<14} {'test_n':>7} {'train_spk':>9} {'test_spk':>8} {'top1%':>7} {'top5%':>7}")
    for i, r in enumerate(present, 1):
        print(f"{i:>3} {r['class']:<14} {r['test_n']:>7} {r['train_spk']:>9} {r['test_spk']:>8} "
              f"{r['top1']*100:>7.1f} {r['top5']*100:>7.1f}")
    print(f"\n--- test 없음 (train-only) {len(absent)}개 ---")
    print(", ".join(f"{r['class']}(tr_spk={r['train_spk']})" for r in absent))

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["class", "test_n", "train_spk", "test_spk", "top1", "top5"])
            w.writeheader()
            for r in present + absent:
                w.writerow(r)
        print(f"\n[saved] {args.out_csv}")


if __name__ == "__main__":
    main()
