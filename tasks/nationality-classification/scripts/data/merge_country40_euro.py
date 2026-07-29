"""Merge Asian(505) + European(041) into the existing 40-class probe, with a
domain-leak diagnostic.

- Asian features: $SPEECH_AI_DATA/taeho_layers_full (relabel by speaker -> 40 via fine_labels)
- Euro features:  $SPEECH_AI_DATA/taeho_layers_euro  (relabel by speaker -> 40 via euro manifest)
Speaker-disjoint 8:1:1 (seed 42) over the COMBINED speaker set, linear probe on layer 31.
Reports per-class top1/top5 + each class's euro fraction, and for MIXED-domain classes
checks asia-test vs euro-test accuracy separately (domain confound probe).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_DATA


import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from country_classifier.make_country40_probe import SPLIT, LANG2C
from country_classifier.make_speaker_split import assign_speakers

ASIA = Path(f"{DATA_ROOT}/taeho_layers_full")
EURO = Path(f"{DATA_ROOT}/taeho_layers_euro/train.npz")
EURO_MANIFEST = f"{NAT_DATA}/euro_country40_manifest.csv"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["asia", "euro"],
                    help="which domains to include: asia / euro / both")
    ap.add_argument("--domain_norm", default="global", choices=["global", "perdomain", "center", "whiten"],
                    help="global=one mu/sd; perdomain=per-domain mu/sd; center=per-domain mean; whiten=per-domain ZCA")
    args = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F

    # ---- Asian: speaker -> 40-country ----
    rows = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
    spk_mt = {}; spk_cc = defaultdict(Counter)
    for r in rows:
        spk_mt[r["speaker_id"]] = r["mother_tongue"]; spk_cc[r["speaker_id"]][(r["country"] or "").strip()] += 1
    spk_c = {s: c.most_common(1)[0][0] for s, c in spk_cc.items()}
    def c40(s):
        mt = spk_mt.get(s); return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)

    X, sps, lab, dom = [], [], [], []
    for sp in ["train", "valid", "test"]:
        d = np.load(ASIA / f"{sp}.npz", allow_pickle=True)
        Xa = np.clip(np.nan_to_num(d["X"].astype(np.float32)), -50, 50)[:, 0, :]
        sa = [str(s) for s in d["speaker"]]
        la = [c40(s) for s in sa]
        for i, l in enumerate(la):
            if l is None:
                continue
            X.append(Xa[i]); sps.append(sa[i]); lab.append(l); dom.append("asia")

    # ---- Euro: speaker -> 40 via manifest ----
    spk2lab = {}
    for r in csv.DictReader(open(EURO_MANIFEST, encoding="utf-8")):
        spk2lab[r["speaker_id"]] = r["label_name"]
    de = np.load(EURO, allow_pickle=True)
    Xe = np.clip(np.nan_to_num(de["X"].astype(np.float32)), -50, 50)[:, 0, :]
    se = [str(s) for s in de["speaker"]]
    for i, s in enumerate(se):
        l = spk2lab.get(s)
        if l is None:
            continue
        X.append(Xe[i]); sps.append(s); lab.append(l); dom.append("euro")

    X = np.stack(X); sps = np.array(sps); lab = np.array(lab, dtype=object); dom = np.array(dom)
    # --- feature scale sanity (must be comparable across domains) ---
    for dd in ["asia", "euro"]:
        mk = dom == dd
        if mk.sum():
            xs = X[mk]
            print(f"  [scale:{dd}] mean={xs.mean():.3f} std={xs.std():.3f} "
                  f"min={xs.min():.1f} max={xs.max():.1f} %|x|>=50={(np.abs(xs)>=49.9).mean()*100:.2f} n={mk.sum():,}", flush=True)
    # --- optional domain filter (for baseline reproduction) ---
    if set(args.domains) != {"asia", "euro"}:
        keepd = np.isin(dom, args.domains)
        X, sps, lab, dom = X[keepd], sps[keepd], lab[keepd], dom[keepd]
    classes = sorted(set(lab.tolist())); c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[c] for c in lab], np.int64)
    print(f"[domains={args.domains}] 합계 {len(y):,} utts | {len(classes)} classes | "
          f"asia {int((dom=='asia').sum()):,} / euro {int((dom=='euro').sum()):,}", flush=True)

    items = [{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l in zip(sps, lab)]
    assign = assign_speakers(items, [8, 1, 1], seed=42)
    split = np.array([assign[s] for s in sps])
    idx = {s: np.where(split == s)[0] for s in ["train", "valid", "test"]}

    dev = "cuda"
    tr, va, te = idx["train"], idx["valid"], idx["test"]
    if args.domain_norm == "global":
        mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
        Xtr = (X[tr] - mu) / sd; Xva = (X[va] - mu) / sd; Xte = (X[te] - mu) / sd
    elif args.domain_norm == "whiten":
        # per-domain ZCA whitening (align 1st+2nd order marginals across domains)
        W = {}
        for dd in sorted(set(dom[tr].tolist())):
            Xd = X[tr][dom[tr] == dd]; m = Xd.mean(0, keepdims=True)
            cov = np.cov((Xd - m).T) + 1e-2 * np.eye(X.shape[1])
            ev, V = np.linalg.eigh(cov)
            Wd = V @ np.diag(1.0 / np.sqrt(np.maximum(ev, 1e-6))) @ V.T
            W[dd] = (m, Wd.astype(np.float32))
            print(f"  [whiten:{dd}] done", flush=True)
        def dn(Xs, ds):
            out = Xs.copy()
            for dd, (m, Wd) in W.items():
                mk = ds == dd
                if mk.any(): out[mk] = (Xs[mk] - m) @ Wd
            return out
        Xtr = dn(X[tr], dom[tr]); Xva = dn(X[va], dom[va]); Xte = dn(X[te], dom[te])
    else:
        # per-domain stats from TRAIN; applied per row by its domain (deployment=online uses euro stats)
        stats = {}
        for dd in sorted(set(dom[tr].tolist())):
            mk = dom[tr] == dd
            m = X[tr][mk].mean(0, keepdims=True)
            s = (X[tr][mk].std(0, keepdims=True) + 1e-6) if args.domain_norm == "perdomain" else 1.0
            stats[dd] = (m, s)
            print(f"  [norm:{dd}] mean|{float(np.abs(m).mean()):.3f}| {'sd '+format(float(np.mean(s)),'.3f') if args.domain_norm=='perdomain' else '(center)'}", flush=True)

        def dn(Xs, ds):
            out = Xs.copy()
            for dd, (m, s) in stats.items():
                mk = ds == dd
                if mk.any():
                    out[mk] = (Xs[mk] - m) / s
            return out
        Xtr = dn(X[tr], dom[tr]); Xva = dn(X[va], dom[va]); Xte = dn(X[te], dom[te])
    ytr, yva, yte = y[tr], y[va], y[te]
    torch.manual_seed(42); head = nn.Linear(X.shape[1], len(classes)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    At = torch.from_numpy(Xtr).to(dev); Yt = torch.from_numpy(ytr).to(dev); n = len(Yt)
    def logits(Xn):
        head.eval()
        with torch.no_grad(): return head(torch.from_numpy(Xn).to(dev)).cpu().numpy()
    def macro5(L, yy):
        i5 = np.argpartition(-L, 4, axis=1)[:, :5]; h = np.array([a in r for a, r in zip(yy, i5)])
        return np.mean([h[yy == c].mean() for c in sorted(set(yy.tolist()))])
    best = -1; bw = None
    for ep in range(60):
        head.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 4096):
            j = perm[i:i+4096]; F.cross_entropy(head(At[j]), Yt[j]).backward(); opt.step(); opt.zero_grad()
        m = macro5(logits(Xva), yva)
        if m > best: best = m; bw = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(bw)

    tl = logits(Xte); pred = tl.argmax(1); t5 = np.argpartition(-tl, 4, axis=1)[:, :5]
    hit1 = (pred == yte); hit5 = np.array([yi in r for yi, r in zip(yte, t5)])
    dte = dom[te]; ste = sps[te]
    present = sorted(set(yte.tolist()), key=lambda c: -int((yte == c).sum()))

    print(f"\n=== 합친 40-class (test) ===")
    print(f"  top1(micro)={hit1.mean():.4f}  top5(micro)={hit5.mean():.4f}  |  "
          f"top1(macro)={np.mean([hit1[yte==c].mean() for c in present]):.4f}  "
          f"top5(macro)={np.mean([hit5[yte==c].mean() for c in present]):.4f}")
    print(f"  (참고 아시아만 40-country: micro top5 0.956 / macro 0.606)")

    print(f"\n{'클래스':<12}{'test':>7}{'화자':>6}{'euro%':>7}{'top1':>7}{'top5':>7}   도메인별 top1")
    print("-" * 78)
    for c in present:
        m = yte == c
        nspk = len(set(ste[m])); ef = (dte[m] == "euro").mean()
        a_m = m & (dom[te] == "asia"); e_m = m & (dom[te] == "euro")
        a1 = hit1[a_m].mean() if a_m.sum() else float("nan")
        e1 = hit1[e_m].mean() if e_m.sum() else float("nan")
        mixed = a_m.sum() > 0 and e_m.sum() > 0
        dstr = f"asia {a1:.2f}({int(a_m.sum())}) | euro {e1:.2f}({int(e_m.sum())})" if mixed else (
               f"euro만 {e1:.2f}" if e_m.sum() else f"asia만 {a1:.2f}")
        flag = "  <<섞임" if mixed else ("  [euro전용]" if ef > 0.9 else "")
        print(f"{classes[c]:<12}{int(m.sum()):>7}{nspk:>6}{ef*100:>6.0f}%{hit1[m].mean():>7.2f}{hit5[m].mean():>7.2f}   {dstr}{flag}")

    # domain-leak summary
    print(f"\n=== 도메인 누수 진단 ===")
    am = dte == "asia"; em = dte == "euro"
    print(f"  asia test top1={hit1[am].mean():.3f} (n={am.sum():,}) | euro test top1={hit1[em].mean():.3f} (n={em.sum():,})")
    print(f"  euro전용(>90%) 클래스 평균 top1={np.mean([hit1[yte==c].mean() for c in present if (dte[yte==c]=='euro').mean()>0.9]):.3f}  ← 수상하게 높으면 도메인누수 의심")


if __name__ == "__main__":
    main()
