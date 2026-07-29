# -*- coding: utf-8 -*-
"""XLS-R 300M 층 sweep — 505+한국어+041문장+043 병합(0704와 동일 데이터/split).

캐시를 1회 로드 후 5개 층(13/15/17/19/21)을 순회.
- 층 선택은 **valid macro top5** (test 재사용 금지, ⑦ 원칙).
- best 층만 test 리포트 + head 저장.
동일 조건: max_sec 12, clamp[-50,50], mean-pool(캐시), speaker-disjoint seed42, macro5 early-stop, 60ep lr1e-3.
다른 것: 인코더(2B→300M), dim(1920→1024), layer(31→sweep).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_DATA, NAT_OUTPUTS, NAT_RUNS

import argparse, csv, json, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from country_classifier.make_country40_probe import SPLIT, LANG2C
from country_classifier.make_speaker_split import assign_speakers

LAYERS = [13, 15, 17, 19, 21]
FULL = Path(f"{DATA_ROOT}/taeho300_layers_full")
KOR = Path(f"{DATA_ROOT}/taeho300_layers_korean/train.npz")
EURO = Path(f"{DATA_ROOT}/taeho300_layers_euro_sent/train.npz")
ASIA043 = Path(f"{DATA_ROOT}/taeho300_layers_asia/train.npz")
EURO_MAN = f"{NAT_DATA}/euro_country40_manifest.csv"
ASIA_MAN = f"{NAT_DATA}/asian_country40_manifest.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_head", default=f"{NAT_OUTPUTS}/0705_country41_sent043_300m")
    ap.add_argument("--out", default=f"{NAT_RUNS}/sweep300_layers.csv")
    ap.add_argument("--labels_en", type=int, default=1)
    args = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    dev = "cuda"

    # ---- label 매핑 ----
    rows = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
    spk_mt = {}; spk_cc = defaultdict(Counter)
    for r in rows:
        spk_mt[r["speaker_id"]] = r["mother_tongue"]; spk_cc[r["speaker_id"]][(r["country"] or "").strip()] += 1
    spk_c = {s: c.most_common(1)[0][0] for s, c in spk_cc.items()}
    def c40(s):
        mt = spk_mt.get(s); return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)
    e2l = {r["speaker_id"]: r["label_name"] for r in csv.DictReader(open(EURO_MAN, encoding="utf-8"))}
    a2l = {}
    for r in csv.DictReader(open(ASIA_MAN, encoding="utf-8")):
        a2l.setdefault(r["speaker_id"], r["label_name"])

    # ---- 화자/라벨/도메인 + (소스,행번호) 인덱스만 먼저 구성 (feature는 층별로 나중에) ----
    # sources: (npz경로, 그 npz 내 원본 행번호) — 층 슬라이스는 sweep 루프에서 층 하나씩만 로드
    src_path, src_row = [], []
    sps, lab, dom = [], [], []
    def add(path, speaker_iter, labeler, domain):
        for i, s in enumerate(speaker_iter):
            l = labeler(str(s))
            if l is None: continue
            src_path.append(path); src_row.append(i)
            sps.append(str(s)); lab.append(l); dom.append(domain)
    # 505 (train/valid/test 순)
    for sp in ["train", "valid", "test"]:
        p = str(FULL / f"{sp}.npz"); d = np.load(p, allow_pickle=True)
        add(p, d["speaker"], lambda s: c40(s), "asia")
    # korean (라벨 항상 '한국어', speaker는 KOR: prefix)
    p = str(KOR); kd = np.load(p, allow_pickle=True)
    for i, s in enumerate(kd["speaker"]):
        src_path.append(p); src_row.append(i); sps.append("KOR:" + str(s)); lab.append("한국어"); dom.append("korean")
    # euro (문장전용 캐시)
    p = str(EURO); de = np.load(p, allow_pickle=True)
    add(p, de["speaker"], lambda s: e2l.get(s), "euro")
    # asia043
    p = str(ASIA043); da = np.load(p, allow_pickle=True)
    add(p, da["speaker"], lambda s: a2l.get(s), "asia043")
    src_path = np.array(src_path); src_row = np.array(src_row)
    sps = np.array(sps); lab = np.array(lab, dtype=object); dom = np.array(dom)
    N = len(sps)
    uniq_paths = list(dict.fromkeys(src_path.tolist()))
    print(f"[index] {N:,} utts across {len(uniq_paths)} npz (feature는 층별 로드)", flush=True)

    def load_layer(li):
        """li 층만 모든 소스에서 잘라 (N,1024) fp32 로 조립 (피크 ~1층분)."""
        out = np.empty((N, 1024), np.float32)
        for p in uniq_paths:
            Xp = np.load(p, allow_pickle=True)["X"][:, li, :].astype(np.float32)   # (n_p,1024)
            m = src_path == p
            out[m] = Xp[src_row[m]]
            del Xp
        return np.clip(np.nan_to_num(out), -50, 50)

    if args.labels_en:
        ko2en = json.load(open("configs/country41_ko2en.json", encoding="utf-8"))
        lab = np.array([ko2en[l] for l in lab], dtype=object)
    classes = sorted(set(lab.tolist())); c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[c] for c in lab], np.int64)
    items = [{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l in zip(sps, lab)]
    assign = assign_speakers(items, [8, 1, 1], seed=42); split = np.array([assign[s] for s in sps])
    tr, va, te = (np.where(split == s)[0] for s in ["train", "valid", "test"])
    ytr, yva, yte = y[tr], y[va], y[te]
    dte = dom[te]

    def macro5(L, yy):
        i5 = np.argpartition(-L, 4, axis=1)[:, :5]; h = np.array([a in r for a, r in zip(yy, i5)])
        return np.mean([h[yy == c].mean() for c in sorted(set(yy.tolist()))])

    def train_probe(Xn_tr, Xn_va):
        torch.manual_seed(42); head = nn.Linear(Xn_tr.shape[1], len(classes)).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        At = torch.from_numpy(Xn_tr).to(dev); Yt = torch.from_numpy(ytr).to(dev); nn_ = len(Yt)
        def logits(Z):
            head.eval()
            with torch.no_grad(): return head(torch.from_numpy(Z).to(dev)).cpu().numpy()
        best = -1; bw = None
        for ep in range(60):
            head.train(); perm = torch.randperm(nn_, device=dev)
            for i in range(0, nn_, 4096):
                j = perm[i:i+4096]; F.cross_entropy(head(At[j]), Yt[j]).backward(); opt.step(); opt.zero_grad()
            m = macro5(logits(Xn_va), yva)
            if m > best: best = m; bw = {k: v.detach().clone() for k, v in head.state_dict().items()}
        head.load_state_dict(bw)
        return head, best

    # ---- 층 sweep (선택은 valid macro5) ----
    results = []
    for li, layer in enumerate(LAYERS):
        Xl = load_layer(li)                                # (N,1024) fp32, 이 층만
        mu = Xl[tr].mean(0, keepdims=True); sd = Xl[tr].std(0, keepdims=True) + 1e-6
        Xtr, Xva, Xte = (Xl[tr]-mu)/sd, (Xl[va]-mu)/sd, (Xl[te]-mu)/sd
        head, vmac5 = train_probe(Xtr, Xva)
        with torch.no_grad():
            vl = head(torch.from_numpy(Xva).to(dev)).cpu().numpy()
            tl = head(torch.from_numpy(Xte).to(dev)).cpu().numpy()
        vmic5 = np.mean([yy in r for yy, r in zip(yva, np.argpartition(-vl, 4, 1)[:, :5])])
        pred = tl.argmax(1); t5 = np.argpartition(-tl, 4, 1)[:, :5]
        h1 = pred == yte; h5 = np.array([yy in r for yy, r in zip(yte, t5)])
        pres = sorted(set(yte.tolist()))
        tmic5 = h5.mean(); tmac5 = np.mean([h5[yte == c].mean() for c in pres]); tmic1 = h1.mean()
        results.append({"layer": layer, "li": li, "valid_macro5": round(float(vmac5), 4),
                        "valid_micro5": round(float(vmic5), 4), "test_micro5": round(float(tmic5), 4),
                        "test_macro5": round(float(tmac5), 4), "test_micro1": round(float(tmic1), 4),
                        "_mu": mu, "_sd": sd, "_head": head, "_h1": h1, "_h5": h5})
        print(f"[layer {layer}] VALID macro5={vmac5:.4f} micro5={vmic5:.4f} | (참고 test) micro5={tmic5:.4f} macro5={tmac5:.4f} micro1={tmic1:.4f}", flush=True)

    best = max(results, key=lambda r: r["valid_macro5"])
    print(f"\n[SELECT] valid macro5 최고 = layer {best['layer']} (valid {best['valid_macro5']:.4f})", flush=True)
    print(f"[TEST @ layer {best['layer']}] micro5={best['test_micro5']:.4f} macro5={best['test_macro5']:.4f} micro1={best['test_micro1']:.4f}", flush=True)
    # 도메인 블록
    h1, h5 = best["_h1"], best["_h5"]
    for dd, nm in [("asia", "505"), ("euro", "041"), ("asia043", "043"), ("korean", "korean")]:
        mk = dte == dd
        if mk.sum(): print(f"    [{nm:6}] n={int(mk.sum()):>7,} top1={h1[mk].mean():.4f} top5={h5[mk].mean():.4f}", flush=True)

    # ---- sweep 요약 CSV ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "valid_macro5", "valid_micro5", "test_micro5", "test_macro5", "test_micro1", "selected"])
        for r in results:
            w.writerow([r["layer"], r["valid_macro5"], r["valid_micro5"], r["test_micro5"], r["test_macro5"], r["test_micro1"], "*" if r is best else ""])
    print(f"[saved] {args.out}")

    # ---- best head 저장 ----
    if args.save_head:
        os.makedirs(args.save_head, exist_ok=True)
        pres = sorted(set(yte.tolist())); ste = sps[te]
        with open(os.path.join(args.save_head, "per_class.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["class", "test_n", "test_spk", "top1", "top5"])
            for c in pres:
                m = yte == c
                w.writerow([classes[c], int(m.sum()), len(set(ste[m])), round(float(h1[m].mean()), 4), round(float(h5[m].mean()), 4)])
        ko2en = json.load(open("configs/country41_ko2en.json", encoding="utf-8"))
        en = classes if args.labels_en else [ko2en[c] for c in classes]
        torch.save({"head": best["_head"].state_dict(), "mu": best["_mu"], "sd": best["_sd"],
                    "labels": classes, "labels_en": en, "c2i": c2i,
                    "layer": best["layer"], "feat_dim": 1024, "max_sec": 12,
                    "backbone": "xlsr_300m", "euro_sent_only": True, "asia043": True,
                    "kind": "country41_euro_sent_asia043_300m"},
                   os.path.join(args.save_head, "head.pt"))
        dmn = {}
        for dd, nm in [("asia", "505"), ("euro", "041"), ("asia043", "043"), ("korean", "korean")]:
            mk = dte == dd
            if mk.sum(): dmn[nm] = {"n": int(mk.sum()), "top1": round(float(h1[mk].mean()), 4), "top5": round(float(h5[mk].mean()), 4)}
        json.dump({"backbone": "xlsr_300m", "layer": best["layer"],
                   "valid_macro5": best["valid_macro5"],
                   "overall": {"test_micro5": best["test_micro5"], "test_macro5": best["test_macro5"], "test_micro1": best["test_micro1"]},
                   "by_domain": dmn, "sweep": [{k: r[k] for k in ("layer", "valid_macro5", "test_micro5", "test_macro5")} for r in results]},
                  open(os.path.join(args.save_head, "metrics.json"), "w"), ensure_ascii=False, indent=2)
        print(f"[save_head] -> {args.save_head}/head.pt (layer {best['layer']}, +metrics.json, per_class.csv)")


if __name__ == "__main__":
    main()
