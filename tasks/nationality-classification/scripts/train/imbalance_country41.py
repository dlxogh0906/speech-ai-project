"""Keep 41-class COUNTRY output; try to recover macro WITHOUT merging, on cached features.
Compares: layer31+CE (baseline repro) / multilayer+CE / +logit-adjust / +class-balanced /
multilayer+CE then tau-normalized classifier. Encoder NOT re-run.
Data assembly + split identical to merge_country41_euro.py (euro=0).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import DATA_ROOT, NAT_DATA, NAT_OUTPUTS

import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from country_classifier.make_country40_probe import SPLIT, LANG2C
from country_classifier.make_speaker_split import assign_speakers

ASIA = Path(f"{DATA_ROOT}/taeho_layers_full")
KOR = Path(f"{DATA_ROOT}/taeho_layers_korean/korean.npz")
OUT = Path(f"{NAT_OUTPUTS}/0701_country41_imbalance"); OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50


def assemble():
    rows = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
    spk_mt = {}; spk_cc = defaultdict(Counter)
    for r in rows:
        spk_mt[r["speaker_id"]] = r["mother_tongue"]
        spk_cc[r["speaker_id"]][(r["country"] or "").strip()] += 1
    spk_c = {s: c.most_common(1)[0][0] for s, c in spk_cc.items()}
    def c40(s):
        mt = spk_mt.get(s)
        return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)

    plan = [(ASIA / f"{sp}.npz", c40) for sp in ["train", "valid", "test"]]
    plan.append((KOR, lambda s: "한국어"))
    # pass1: labels/speakers
    sps, lab = [], []
    for path, fn in plan:
        z = np.load(path, allow_pickle=True)
        for s in z["speaker"]:
            l = fn(str(s))
            if l:
                sps.append(("KOR:" + str(s)) if path == KOR else str(s)); lab.append(l)
        del z
    sps = np.array(sps); lab = np.array(lab, dtype=object)
    classes = sorted(set(lab.tolist())); c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[c] for c in lab], np.int64)
    assign = assign_speakers([{"speaker_id": s, "label_name": l, "num_frames": 0}
                              for s, l in zip(sps, lab)], [8, 1, 1], seed=42)
    split = np.array([assign[s] for s in sps])
    # pass2: features (3 layers) preallocated
    N = len(y); Xall = np.empty((N, 3, 1920), np.float16); g = 0
    for path, fn in plan:
        z = np.load(path, allow_pickle=True)
        Xs = np.clip(np.nan_to_num(z["X"].astype(np.float32)), -50, 50).astype(np.float16)
        for i, s in enumerate(z["speaker"]):
            if fn(str(s)):
                Xall[g] = Xs[i]; g += 1
        del z, Xs
    assert g == N
    return Xall, y, split, classes, c2i


def run(Xall, y, split, classes, multilayer):
    idx = {s: np.where(split == s)[0] for s in ["train", "valid", "test"]}
    tr, va, te = idx["train"], idx["valid"], idx["test"]
    if multilayer:
        X = torch.from_numpy(Xall.reshape(len(y), -1)).to(DEV)        # (N,5760) f16
    else:
        X = torch.from_numpy(np.ascontiguousarray(Xall[:, 0, :])).to(DEV)  # layer31
    yt = torch.from_numpy(y).to(DEV)
    tr_t = torch.from_numpy(tr).to(DEV)
    mu = X[tr_t].float().mean(0, keepdim=True); sd = X[tr_t].float().std(0, keepdim=True) + 1e-6
    D = X.shape[1]; C = len(classes)
    cnt = np.bincount(y[tr], minlength=C).astype(np.float64)
    prior = torch.tensor(np.log(np.clip(cnt / cnt.sum(), 1e-12, None)), dtype=torch.float32, device=DEV)
    # class-balanced (effective number) weights
    beta = 0.9999
    cbw = (1 - beta) / (1 - np.power(beta, np.clip(cnt, 1, None)))
    cbw = torch.tensor(cbw / cbw.sum() * C, dtype=torch.float32, device=DEV)

    def std_batch(ix):
        return (X[ix].float() - mu) / sd

    def evaluate(head, tau=None, W=None, b=None):
        head_eval = head
        with torch.no_grad():
            outs = []
            src = te
            for i in range(0, len(src), 16384):
                ix = torch.from_numpy(src[i:i+16384]).to(DEV)
                z = std_batch(ix)
                if W is not None:
                    lg = z @ W.T + (b if b is not None else 0)
                else:
                    lg = head_eval(z)
                outs.append(lg.cpu())
            L = torch.cat(outs).numpy()
        yy = y[te]
        p1 = L.argmax(1); t5 = np.argpartition(-L, 4, 1)[:, :5]
        h1 = (p1 == yy); h5 = np.array([a in r for a, r in zip(yy, t5)])
        pres = sorted(set(yy.tolist()))
        return {"micro1": float(h1.mean()), "micro5": float(h5.mean()),
                "macro1": float(np.mean([h1[yy == c].mean() for c in pres])),
                "macro5": float(np.mean([h5[yy == c].mean() for c in pres]))}, L, p1

    def train(mode):
        torch.manual_seed(42)
        head = nn.Linear(D, C).to(DEV)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        best = -1; bw = None
        for ep in range(EPOCHS):
            head.train(); perm = tr_t[torch.randperm(len(tr_t), device=DEV)]
            for i in range(0, len(perm), 4096):
                ix = perm[i:i+4096]; z = std_batch(ix); lg = head(z)
                if mode == "logit_adjust":
                    loss = F.cross_entropy(lg + prior, yt[ix])
                elif mode == "class_balanced":
                    loss = F.cross_entropy(lg, yt[ix], weight=cbw)
                else:
                    loss = F.cross_entropy(lg, yt[ix])
                loss.backward(); opt.step(); opt.zero_grad()
            # valid macro5 for selection
            with torch.no_grad():
                head.eval(); vix = torch.from_numpy(va).to(DEV)
                Lv = []
                for i in range(0, len(va), 16384):
                    jx = torch.from_numpy(va[i:i+16384]).to(DEV)
                    Lv.append(head(std_batch(jx)).cpu())
                Lv = torch.cat(Lv).numpy(); yv = y[va]
                i5 = np.argpartition(-Lv, 4, 1)[:, :5]
                hv = np.array([a in r for a, r in zip(yv, i5)])
                m = np.mean([hv[yv == c].mean() for c in sorted(set(yv.tolist()))])
            if m > best: best = m; bw = {k: v.detach().clone() for k, v in head.state_dict().items()}
        head.load_state_dict(bw); return head

    results = {}
    # plain CE
    h_ce = train("ce"); results["CE"], _, _ = evaluate(h_ce)
    # logit-adjust
    h_la = train("logit_adjust"); results["logit_adjust"], _, _ = evaluate(h_la)
    # class-balanced
    h_cb = train("class_balanced"); results["class_balanced"], _, _ = evaluate(h_cb)
    # tau-normalize the plain-CE classifier (decouple): sweep tau on valid
    W0 = h_ce.state_dict()["weight"].clone(); b0 = h_ce.state_dict()["bias"].clone()
    norms = W0.norm(dim=1, keepdim=True)
    best_tau, best_vm, bestW = 0.0, -1, None
    for tau in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
        Wt = W0 / (norms ** tau)
        with torch.no_grad():
            vix = va; Lv = []
            for i in range(0, len(vix), 16384):
                jx = torch.from_numpy(vix[i:i+16384]).to(DEV); z = std_batch(jx)
                Lv.append((z @ Wt.T).cpu())   # bias dropped for tau-norm
            Lv = torch.cat(Lv).numpy(); yv = y[va]
            i5 = np.argpartition(-Lv, 4, 1)[:, :5]; hv = np.array([a in r for a, r in zip(yv, i5)])
            vm = np.mean([hv[yv == c].mean() for c in sorted(set(yv.tolist()))])
        if vm > best_vm: best_vm, best_tau, bestW = vm, tau, Wt.clone()
    res_tau, _, _ = evaluate(h_ce, W=bestW, b=None)
    results[f"CE+tau_norm(tau={best_tau})"] = res_tau
    return results


def main():
    print("assembling cached features ...", flush=True)
    Xall, y, split, classes, c2i = assemble()
    print(f"{len(y):,} utts | {len(classes)} country classes | dev={DEV}", flush=True)

    print("\n--- layer31 only (baseline repro) ---", flush=True)
    r31 = run(Xall, y, split, classes, multilayer=False)
    print("\n--- multilayer 31+35+39 ---", flush=True)
    rml = run(Xall, y, split, classes, multilayer=True)

    print("\n================  RESULTS (41-class country, test)  ================")
    print(f"{'method':<28}{'micro1':>9}{'micro5':>9}{'macro1':>9}{'macro5':>9}")
    print(f"{'baseline(metrics.json)':<28}{0.8326:>9.4f}{0.9631:>9.4f}{0.3560:>9.4f}{0.5775:>9.4f}")
    for tag, R in [("L31", r31), ("ML", rml)]:
        for k, v in R.items():
            print(f"{tag+' '+k:<28}{v['micro1']:>9.4f}{v['micro5']:>9.4f}{v['macro1']:>9.4f}{v['macro5']:>9.4f}")
    json.dump({"layer31": r31, "multilayer": rml}, open(OUT / "results.json", "w"),
              ensure_ascii=False, indent=2)
    print(f"\n-> {OUT/'results.json'}")


if __name__ == "__main__":
    main()
