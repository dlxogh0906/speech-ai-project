#!/usr/bin/env python
"""Per-class top1/top5: baseline(none) vs weighted-sampling beta — same code, with Δ.

  python perclass_beta.py --beta 0.75 --out_csv runs/perclass_beta075_12s.csv
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_OUTPUTS

import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
import decouple_eval_62class as de

KOR = 61
TC = f"{NAT_OUTPUTS}/feature_resplit_61_merged6_seed42_target_counts"
KF = f"{DATA_ROOT}/taeho/korean_native_features/sec12"


def train_beta(Xtr, ytr, C, dev, epochs, Xva, yva, beta, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr); n = len(ytr)
    model = nn.Linear(Xtr.shape[1], C).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    if beta is None:
        p = None
    else:
        cnt = np.bincount(ytr, minlength=C).astype(np.float64)
        w = (1.0 / np.clip(cnt[ytr], 1, None)) ** beta
        p = w / w.sum()
    best = {"top5": -1, "W": None, "b": None}
    for ep in range(1, epochs + 1):
        model.train()
        order = (torch.randperm(n) if p is None
                 else torch.from_numpy(np.random.choice(n, size=n, replace=True, p=p)))
        for s in range(0, n, 4096):
            idx = order[s:s + 4096]
            opt.zero_grad(set_to_none=True)
            lossf(model(Xt[idx].to(dev)), yt[idx].to(dev)).backward(); opt.step()
        vlog = de._predict(model, Xva, dev); t5 = float(de.topk_hits(vlog, yva, 5).mean())
        if t5 > best["top5"]:
            best = {"top5": t5, "W": model.weight.detach().cpu().numpy().copy(),
                    "b": model.bias.detach().cpu().numpy().copy()}
        if ep == 1 or ep % 20 == 0 or ep == epochs:
            print(f"  [beta={beta}] ep{ep} val_top5={t5:.4f}", flush=True)
    return best["W"], best["b"]


def perclass(logits, y, C):
    preds = logits.argmax(1)
    top5 = np.argpartition(-logits, 4, axis=1)[:, :5]
    out = {}
    for c in range(C):
        m = y == c; n = int(m.sum())
        if n == 0:
            out[c] = (0, None, None)
        else:
            t1 = float((preds[m] == c).mean())
            t5 = float(np.any(top5[m] == c, axis=1).mean())
            out[c] = (n, t1, t5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits, labels = de.build_splits("12s", Path(TC), Path(TC))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits["train"], splits["valid"], splits["test"]
    C = len(labels)
    Xz, Xvz, Xtz = de.zscore(Xtr, Xva, Xte)

    print("[train baseline none]")
    Wb, bb = train_beta(Xz, ytr, C, dev, args.epochs, Xvz, yva, None)
    print(f"[train beta={args.beta}]")
    Ww, bw = train_beta(Xz, ytr, C, dev, args.epochs, Xvz, yva, args.beta)
    base = perclass(Xtz @ Wb.T + bb, yte, C)
    bet = perclass(Xtz @ Ww.T + bw, yte, C)

    # speaker counts
    id2label = json.load(open(os.path.join(TC, "id2label.json"), encoding="utf-8"))
    def spk(d, sp):
        z = np.load(os.path.join(d, f"{sp}.npz"), allow_pickle=True); return z["y"], z["speaker"].astype(str)
    trs = {};
    yy, ss = spk(TC, "train")
    for c in np.unique(yy): trs[int(c)] = len(np.unique(ss[yy == c]))
    ktr = len(np.unique(spk(KF, "train")[1]))
    def name(c): return "한국어_모국어" if c == KOR else id2label[str(c)]
    def trspk(c): return ktr if c == KOR else trs.get(c, 0)

    rows = []
    for c in range(C):
        n, t1b, t5b = base[c]
        _, t1w, t5w = bet[c]
        if n == 0: continue
        rows.append((name(c), n, trspk(c), t1b, t1w, t5b, t5w))
    rows.sort(key=lambda r: r[3], reverse=True)

    print(f"\n=== per-class: baseline → beta={args.beta} (62-class overall) ===")
    print(f"{'#':>2} {'클래스':<13}{'test_n':>7}{'tr화자':>6} | {'top1':>5}→{'top1β':>5} {'Δ':>6} | {'top5':>5}→{'top5β':>5} {'Δ':>6}")
    for i, (nm, n, ts, t1b, t1w, t5b, t5w) in enumerate(rows, 1):
        d1 = (t1w - t1b) * 100; d5 = (t5w - t5b) * 100
        print(f"{i:>2} {nm:<13}{n:>7}{ts:>6} | {t1b*100:>5.1f}→{t1w*100:>5.1f} {d1:>+6.1f} | "
              f"{t5b*100:>5.1f}→{t5w*100:>5.1f} {d5:>+6.1f}")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["class","test_n","train_spk","top1_base","top1_beta","top5_base","top5_beta"])
            for nm,n,ts,t1b,t1w,t5b,t5w in rows:
                w.writerow([nm,n,ts,round(t1b,4),round(t1w,4),round(t5b,4),round(t5w,4)])
        print(f"\n[saved] {args.out_csv}")


if __name__ == "__main__":
    main()
