"""41-class (40 country + 한국어) probe, with/without the European(041) merge.
Writes per-class test metrics to CSV so baseline vs merged can be diffed.

  --euro 0  -> baseline: asia(40) + korean        (기존 41)
  --euro 1  -> merged:   asia(40) + korean + euro  (유럽 합친 41)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_DATA, NAT_TASK

import argparse, csv
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from country_classifier.make_country40_probe import SPLIT, LANG2C
from country_classifier.make_speaker_split import assign_speakers

ASIA = Path(f"{DATA_ROOT}/taeho_layers_full")
EURO = Path(f"{DATA_ROOT}/taeho_layers_euro/train.npz")
KOR = Path(f"{DATA_ROOT}/taeho_layers_korean/korean.npz")
EURO_MANIFEST = f"{NAT_DATA}/euro_country40_manifest.csv"
ASIA043 = Path(f"{DATA_ROOT}/taeho_layers_asia/train.npz")
ASIA043_MANIFEST = f"{NAT_DATA}/asian_country40_manifest.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--euro", type=int, default=1)
    ap.add_argument("--euro_sent_only", type=int, default=0, help="1=041을 문장(≥4단어)만 사용")
    ap.add_argument("--asia043", type=int, default=0, help="1=043(아시아, 선별 14클래스 문장 prompt>=4) 추가")
    ap.add_argument("--labels_en", type=int, default=0, help="1=클래스명을 영어로 변환해 학습/저장 (configs/country41_ko2en.json)")
    ap.add_argument("--feat16", type=int, default=0, help="1=16s feature 캐시 사용 (euro는 문장전용 캐시라 sent mask 불필요)")
    ap.add_argument("--feat300", type=int, default=0, help="1=XLS-R 300M 캐시 사용 (layers[13,15,17,19,21], dim 1024)")
    ap.add_argument("--layer_idx", type=int, default=0, help="캐시의 몇 번째 층을 쓸지 (0-base). 2B=[31,35,39], 300M=[13,15,17,19,21]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_head", default="", help="지정 시 head.pt+mu/sd+labels 저장할 디렉토리")
    args = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F

    LI = args.layer_idx                                   # 캐시 내 층 인덱스
    if args.feat300:
        assert args.euro_sent_only, "--feat300은 euro 문장전용 캐시라 --euro_sent_only 1 필수"
        asia_dir = Path(f"{DATA_ROOT}/taeho300_layers_full")
        kor_p = Path(f"{DATA_ROOT}/taeho300_layers_korean/train.npz")
        euro_p = Path(f"{DATA_ROOT}/taeho300_layers_euro_sent/train.npz")     # 문장(prompt>=4)만 추출됨
        asia043_p = Path(f"{DATA_ROOT}/taeho300_layers_asia/train.npz")
        LAYER_LIST = [13, 15, 17, 19, 21]
    elif args.feat16:
        assert args.euro_sent_only, "--feat16은 euro 문장전용 캐시라 --euro_sent_only 1 필수"
        asia_dir = Path(f"{DATA_ROOT}/taeho_layers_full_16s")
        kor_p = Path(f"{DATA_ROOT}/taeho_layers_korean_16s/train.npz")
        euro_p = Path(f"{DATA_ROOT}/taeho_layers_euro_sent_16s/train.npz")   # 문장(prompt>=4)만 추출됨
        asia043_p = Path(f"{DATA_ROOT}/taeho_layers_asia_16s/train.npz")
        LAYER_LIST = [31, 35, 39]
    else:
        asia_dir, kor_p, euro_p, asia043_p = ASIA, KOR, EURO, ASIA043
        LAYER_LIST = [31, 35, 39]
    print(f"[cfg] feat300={args.feat300} feat16={args.feat16} layer_idx={LI} -> layer {LAYER_LIST[LI]}", flush=True)

    rows = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
    spk_mt = {}; spk_cc = defaultdict(Counter)
    for r in rows:
        spk_mt[r["speaker_id"]] = r["mother_tongue"]; spk_cc[r["speaker_id"]][(r["country"] or "").strip()] += 1
    spk_c = {s: c.most_common(1)[0][0] for s, c in spk_cc.items()}
    def c40(s):
        mt = spk_mt.get(s); return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)

    X, sps, lab, dom = [], [], [], []
    for sp in ["train", "valid", "test"]:
        d = np.load(asia_dir / f"{sp}.npz", allow_pickle=True)
        Xa = np.clip(np.nan_to_num(d["X"].astype(np.float32)), -50, 50)[:, LI, :]
        for i, s in enumerate(d["speaker"]):
            l = c40(str(s))
            if l: X.append(Xa[i]); sps.append(str(s)); lab.append(l); dom.append("asia")
    # korean
    kd = np.load(kor_p, allow_pickle=True)
    kX = np.clip(np.nan_to_num(kd["X"].astype(np.float32)), -50, 50)[:, LI, :]
    for i, s in enumerate(kd["speaker"]):
        X.append(kX[i]); sps.append("KOR:" + str(s)); lab.append("한국어"); dom.append("korean")
    # euro
    if args.euro:
        spk2lab = {r["speaker_id"]: r["label_name"] for r in csv.DictReader(open(EURO_MANIFEST, encoding="utf-8"))}
        de = np.load(euro_p, allow_pickle=True)
        Xe = np.clip(np.nan_to_num(de["X"].astype(np.float32)), -50, 50)[:, LI, :]
        # 16s 캐시는 문장전용 매니페스트로 추출돼 마스크 불필요
        sent = np.load(f"{DATA_ROOT}/taeho_layers_euro/sent_mask_ge4.npy") if (args.euro_sent_only and not args.feat16) else None
        for i, s in enumerate(de["speaker"]):
            if sent is not None and not sent[i]:
                continue
            l = spk2lab.get(str(s))
            if l: X.append(Xe[i]); sps.append(str(s)); lab.append(l); dom.append("euro")
    # asia 043 (매니페스트가 이미 선별 14클래스 + prompt>=4 문장만)
    if args.asia043:
        a2l = {}
        for r in csv.DictReader(open(ASIA043_MANIFEST, encoding="utf-8")):
            a2l.setdefault(r["speaker_id"], r["label_name"])
        da = np.load(asia043_p, allow_pickle=True)
        Xa4 = np.clip(np.nan_to_num(da["X"].astype(np.float32)), -50, 50)[:, LI, :]
        for i, s in enumerate(da["speaker"]):
            l = a2l.get(str(s))
            if l: X.append(Xa4[i]); sps.append(str(s)); lab.append(l); dom.append("asia043")

    X = np.stack(X); sps = np.array(sps); lab = np.array(lab, dtype=object); dom = np.array(dom)
    if args.labels_en:
        import json as _json
        ko2en = _json.load(open(NAT_TASK / "configs/country41_ko2en.json", encoding="utf-8"))
        lab = np.array([ko2en[l] for l in lab], dtype=object)
    classes = sorted(set(lab.tolist())); c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[c] for c in lab], np.int64)
    items = [{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l in zip(sps, lab)]
    assign = assign_speakers(items, [8, 1, 1], seed=42); split = np.array([assign[s] for s in sps])
    idx = {s: np.where(split == s)[0] for s in ["train", "valid", "test"]}
    print(f"euro={args.euro} | {len(y):,} utts | {len(classes)} classes", flush=True)

    dev = "cuda"; tr, va, te = idx["train"], idx["valid"], idx["test"]
    mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
    Xtr = (X[tr]-mu)/sd; Xva = (X[va]-mu)/sd; Xte = (X[te]-mu)/sd
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
    present = sorted(set(yte.tolist()))
    o = {"micro1": hit1.mean(), "micro5": hit5.mean(),
         "macro1": np.mean([hit1[yte == c].mean() for c in present]),
         "macro5": np.mean([hit5[yte == c].mean() for c in present]), "nclass": len(present)}
    print(f"  micro top1={o['micro1']:.4f} top5={o['micro5']:.4f} | macro top1={o['macro1']:.4f} top5={o['macro5']:.4f} | {o['nclass']}/{len(classes)}", flush=True)
    # --- test micro split BY DOMAIN (핵심: 아시아 부분이 진짜 나빠졌나?) ---
    dte = dom[te]
    for dd in ["asia", "euro", "asia043", "korean"]:
        mk = dte == dd
        if mk.sum():
            print(f"    [{dd:6}] n={int(mk.sum()):>7,}  micro top1={hit1[mk].mean():.4f}  top5={hit5[mk].mean():.4f}", flush=True)
    ste = sps[te]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["class", "test_n", "test_spk", "top1", "top5"])
        for c in present:
            m = yte == c
            w.writerow([classes[c], int(m.sum()), len(set(ste[m])), round(float(hit1[m].mean()), 4), round(float(hit5[m].mean()), 4)])
        w.writerow(["__OVERALL__", len(yte), len(set(ste)), round(float(o["micro1"]), 4), round(float(o["micro5"]), 4)])
        w.writerow(["__MACRO__", "", "", round(float(o["macro1"]), 4), round(float(o["macro5"]), 4)])
    print(f"  -> {args.out}")
    # --- per-class 출처(505/041/043) 분해 (섹션 8용) ---
    if args.euro or args.asia043:
        dpath = args.out.replace(".csv", "_domain.csv")
        with open(dpath, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["class", "n505", "top1_505", "top5_505",
                                           "n041", "top1_041", "top5_041",
                                           "n043", "top1_043", "top5_043"])
            for c in present:
                cm = yte == c
                a = cm & (dte == "asia"); e = cm & (dte == "euro"); a4 = cm & (dte == "asia043")
                def m(mask, h): return round(float(h[mask].mean()), 4) if mask.sum() else ""
                if a.sum() or e.sum() or a4.sum():
                    w.writerow([classes[c], int(a.sum()), m(a, hit1), m(a, hit5),
                                int(e.sum()), m(e, hit1), m(e, hit5),
                                int(a4.sum()), m(a4, hit1), m(a4, hit5)])
        print(f"  -> {dpath}")

    if args.save_head:
        import json, os
        os.makedirs(args.save_head, exist_ok=True)
        dmn = {}
        for dd, nm in [("asia", "505"), ("euro", "041"), ("asia043", "043"), ("korean", "korean")]:
            mk = dte == dd
            if mk.sum():
                dmn[nm] = {"n": int(mk.sum()), "top1": round(float(hit1[mk].mean()), 4), "top5": round(float(hit5[mk].mean()), 4)}
        kind = "country41_euro_sent" if args.euro_sent_only else "country41_euro_mixed"
        if args.asia043:
            kind += "_asia043"
        if args.feat16:
            kind += "_16s"
        torch.save({"head": head.state_dict(), "mu": mu, "sd": sd, "labels": classes, "c2i": c2i,
                    "layer": 31, "feat_dim": int(X.shape[1]), "max_sec": 16 if args.feat16 else 12,
                    "euro_sent_only": bool(args.euro_sent_only), "asia043": bool(args.asia043),
                    "kind": kind},
                   os.path.join(args.save_head, "head.pt"))
        json.dump({"overall": {"micro_top1": round(float(o["micro1"]), 4), "micro_top5": round(float(o["micro5"]), 4),
                               "macro_top1": round(float(o["macro1"]), 4), "macro_top5": round(float(o["macro5"]), 4),
                               "test_n": int(len(yte)), "classes_present": int(o["nclass"])},
                   "by_domain": dmn},
                  open(os.path.join(args.save_head, "metrics.json"), "w"), ensure_ascii=False, indent=2)
        json.dump(c2i, open(os.path.join(args.save_head, "label2id.json"), "w"), ensure_ascii=False, indent=2)
        print(f"  [save_head] -> {args.save_head}/head.pt (+ metrics.json, label2id.json)")


if __name__ == "__main__":
    main()
