"""40-country + native Korean = 41-class linear probe (cached layer-31 features).

Adds the Korean class (from extract_korean_layer31.py) to the 40-country scheme, re-splits
speaker-disjoint, trains a linear head, and reports top-1/top-5 plus Korean detection
(precision/recall/F1) and foreign<->Korean leakage.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_DATA


import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from country_classifier.make_country40_probe import SPLIT, LANG2C
from country_classifier.make_speaker_split import assign_speakers

FEAT = Path(f"{DATA_ROOT}/taeho_layers_full")
KOREAN = Path(f"{DATA_ROOT}/taeho_layers_korean/korean.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--korean_cap_speakers", type=int, default=0, help="0=use all korean speakers")
    args = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F

    # foreign: speaker -> (motherTongue, majority country)
    rows = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
    spk_mt = {}; spk_cc = defaultdict(Counter)
    for r in rows:
        spk_mt[r["speaker_id"]] = r["mother_tongue"]
        spk_cc[r["speaker_id"]][(r["country"] or "").strip()] += 1
    spk_c = {s: c.most_common(1)[0][0] for s, c in spk_cc.items()}

    def country40(spk):
        mt = spk_mt.get(spk)
        return SPLIT[mt].get(spk_c.get(spk)) if mt in SPLIT else LANG2C.get(mt)

    Xs, sps = [], []
    for split in ["train", "valid", "test"]:
        d = np.load(FEAT / f"{split}.npz", allow_pickle=True)
        Xs.append(np.clip(np.nan_to_num(d["X"].astype(np.float32)), -50, 50)[:, 0, :])
        sps.extend([str(s) for s in d["speaker"]])
    X = np.concatenate(Xs, 0); sps = np.array(sps)
    lab = np.array([country40(s) for s in sps], dtype=object)
    keep = lab != None  # noqa: E711
    X, sps, lab = X[keep], sps[keep], lab[keep]

    # --- add Korean ---
    kd = np.load(KOREAN, allow_pickle=True)
    kX = np.clip(np.nan_to_num(kd["X"].astype(np.float32)), -50, 50)[:, 0, :]
    kspk = np.array(["KOR:" + str(s) for s in kd["speaker"]])  # prefix avoids any id clash
    if args.korean_cap_speakers:
        uniq = sorted(set(kspk.tolist()))
        rng = np.random.default_rng(42)
        sel = set(rng.choice(uniq, min(args.korean_cap_speakers, len(uniq)), replace=False).tolist())
        m = np.array([s in sel for s in kspk])
        kX, kspk = kX[m], kspk[m]
    klab = np.array(["한국어"] * len(kspk), dtype=object)
    X = np.concatenate([X, kX]); sps = np.concatenate([sps, kspk]); lab = np.concatenate([lab, klab])
    print(f"foreign {len(lab)-len(kspk):,} + korean {len(kspk):,}({len(set(kspk))}spk) = {len(lab):,}", flush=True)

    classes = sorted(set(lab.tolist()))
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[c] for c in lab], np.int64)
    KOR = c2i["한국어"]

    items = [{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l in zip(sps, lab)]
    assign = assign_speakers(items, [8, 1, 1], seed=42)
    split = np.array([assign[s] for s in sps])
    idx = {s: np.where(split == s)[0] for s in ["train", "valid", "test"]}
    print(f"{len(classes)} classes | " + " ".join(
        f"{s}:{len(idx[s]):,}({len(set(sps[idx[s]]))}spk)" for s in ["train", "valid", "test"]), flush=True)

    dev = "cuda"
    Xtr, ytr = X[idx["train"]], y[idx["train"]]
    Xva, yva = X[idx["valid"]], y[idx["valid"]]
    Xte, yte = X[idx["test"]], y[idx["test"]]
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    torch.manual_seed(42)
    head = nn.Linear(1920, len(classes)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    At = torch.from_numpy(Xtr).to(dev); Yt = torch.from_numpy(ytr).to(dev); n = len(Yt)

    def logits(Xn):
        head.eval()
        with torch.no_grad():
            return head(torch.from_numpy(Xn).to(dev)).cpu().numpy()

    def mami(L, yy):
        i5 = np.argpartition(-L, 4, axis=1)[:, :5]; h5 = np.array([a in r for a, r in zip(yy, i5)])
        t1 = (L.argmax(1) == yy); pres = sorted(set(yy.tolist()))
        return t1.mean(), h5.mean(), np.mean([t1[yy == c].mean() for c in pres]), np.mean([h5[yy == c].mean() for c in pres])

    best = -1; bw = None
    for ep in range(60):
        head.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 4096):
            j = perm[i:i + 4096]
            F.cross_entropy(head(At[j]), Yt[j]).backward(); opt.step(); opt.zero_grad()
        _, _, _, vma = mami(logits(Xva), yva)
        if vma > best:
            best = vma; bw = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(bw)
    tl = logits(Xte); pred = tl.argmax(1)
    t1, t5, m1, m5 = mami(tl, yte)
    print(f"\n=== 41-class (40국가 + 한국어) linear probe (test) ===")
    print(f"  top1(micro)={t1:.4f}  top5(micro)={t5:.4f}  |  top1(macro)={m1:.4f}  top5(macro)={m5:.4f}")
    print(f"  (참고 40국가만: micro top5 0.956 / macro 0.606)")
    # Korean detection
    tp = int(((pred == KOR) & (yte == KOR)).sum())
    fp = int(((pred == KOR) & (yte != KOR)).sum())
    fn = int(((pred != KOR) & (yte == KOR)).sum())
    nk = int((yte == KOR).sum())
    prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"\n  [한국어 탐지]  test 한국어 {nk:,}개")
    print(f"    precision={prec:.4f}  recall={rec:.4f}  F1={f1:.4f}")
    print(f"    외국인을 한국어로 오분류(FP): {fp:,} / 외국 {int((yte!=KOR).sum()):,}  ({fp/max(1,int((yte!=KOR).sum()))*100:.2f}%)")
    print(f"    한국인을 외국으로 오분류(FN): {fn:,} / {nk:,}  ({fn/max(1,nk)*100:.2f}%)")


if __name__ == "__main__":
    main()
