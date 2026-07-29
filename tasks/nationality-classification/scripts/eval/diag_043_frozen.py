# -*- coding: utf-8 -*-
"""⑨ 043 병합 진단 1: frozen head-vs-head 비교 (CPU only).

같은 test 행에 head A(0702 best)와 head B(0704 +043)를 모두 얹어
'겉보기 하락'을 head효과 vs split(화자교체)효과로 분해한다.
  - 시험지 1: A-run의 test 행 (505+041문장+한국어; 043 없음)
  - 시험지 2: B-run의 test 중 043 블록 (2,609행)
출력: $NATIONALITY_RUNS/diag_043/frozen.json + stdout(tee로 로그 저장 권장)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import DATA_ROOT, NAT_DATA, NAT_OUTPUTS, NAT_RUNS

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from country_classifier.make_country40_probe import SPLIT, LANG2C
from country_classifier.make_speaker_split import assign_speakers

HEAD_A = f"{NAT_OUTPUTS}/0702_country41_sent_best/head.pt"
HEAD_B = f"{NAT_OUTPUTS}/0704_country41_sent043/head.pt"
OUT = Path(f"{NAT_RUNS}/diag_043/frozen.json")


def main():
    import torch, torch.nn as nn
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # ---- (merge_country41_euro.py 와 동일 순서/규격으로 아이템+feature 구성) ----
    rows = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
    spk_mt = {}; spk_cc = defaultdict(Counter)
    for r in rows:
        spk_mt[r["speaker_id"]] = r["mother_tongue"]; spk_cc[r["speaker_id"]][(r["country"] or "").strip()] += 1
    spk_c = {s: c.most_common(1)[0][0] for s, c in spk_cc.items()}
    def c40(s):
        mt = spk_mt.get(s); return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)

    X, sps, lab, dom = [], [], [], []
    for sp in ["train", "valid", "test"]:
        d = np.load(f"{DATA_ROOT}/taeho_layers_full/{sp}.npz", allow_pickle=True)
        Xa = np.clip(np.nan_to_num(d["X"].astype(np.float32)), -50, 50)[:, 0, :]
        for i, s in enumerate(d["speaker"]):
            l = c40(str(s))
            if l: X.append(Xa[i]); sps.append(str(s)); lab.append(l); dom.append("asia")
    kd = np.load(f"{DATA_ROOT}/taeho_layers_korean/korean.npz", allow_pickle=True)
    kX = np.clip(np.nan_to_num(kd["X"].astype(np.float32)), -50, 50)[:, 0, :]
    for i, s in enumerate(kd["speaker"]):
        X.append(kX[i]); sps.append("KOR:" + str(s)); lab.append("한국어"); dom.append("korean")
    sent = np.load(f"{DATA_ROOT}/taeho_layers_euro/sent_mask_ge4.npy")
    e2l = {r["speaker_id"]: r["label_name"] for r in csv.DictReader(open(f"{NAT_DATA}/euro_country40_manifest.csv", encoding="utf-8"))}
    de = np.load(f"{DATA_ROOT}/taeho_layers_euro/train.npz", allow_pickle=True)
    Xe = np.clip(np.nan_to_num(de["X"].astype(np.float32)), -50, 50)[:, 0, :]
    for i, s in enumerate(de["speaker"]):
        if sent[i] and e2l.get(str(s)):
            X.append(Xe[i]); sps.append(str(s)); lab.append(e2l[str(s)]); dom.append("euro")
    X = np.stack(X); sps = np.array(sps); lab = np.array(lab, dtype=object)
    print(f"[pool A] {len(X):,} rows", flush=True)

    a2l = {}
    for r in csv.DictReader(open(f"{NAT_DATA}/asian_country40_manifest.csv", encoding="utf-8")):
        a2l.setdefault(r["speaker_id"], r["label_name"])
    da = np.load(f"{DATA_ROOT}/taeho_layers_asia/train.npz", allow_pickle=True)
    X43 = np.clip(np.nan_to_num(da["X"].astype(np.float32)), -50, 50)[:, 0, :]
    s43 = np.array([str(s) for s in da["speaker"]])
    l43 = np.array([a2l.get(s, "") for s in s43], dtype=object)

    # ---- split 재구성 (A: 옛 아이템만 / B: +043) ----
    items_A = [{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l in zip(sps, lab)]
    asg_A = assign_speakers(items_A, [8, 1, 1], seed=42)
    te_A = np.array([asg_A[s] == "test" for s in sps])
    items_B = items_A + [{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l in zip(s43, l43) if l]
    asg_B = assign_speakers(items_B, [8, 1, 1], seed=42)
    te43 = np.array([bool(l) and asg_B[s] == "test" for s, l in zip(s43, l43)])
    print(f"[A-test] {int(te_A.sum()):,} rows | [043-test] {int(te43.sum()):,} rows", flush=True)

    # ---- head 평가 ----
    def evalhead(path, Xn, labs):
        d = torch.load(path, map_location="cpu")
        head = nn.Linear(d["feat_dim"], len(d["labels"])); head.load_state_dict(d["head"]); head.eval()
        c2i = d["c2i"]; y = np.array([c2i[l] for l in labs])
        h1 = np.zeros(len(y), bool); h5 = np.zeros(len(y), bool)
        with torch.no_grad():
            for i in range(0, len(Xn), 65536):
                xb = (Xn[i:i+65536] - d["mu"]) / d["sd"]
                L = head(torch.from_numpy(xb).float()).numpy()
                t5 = np.argpartition(-L, 4, axis=1)[:, :5]
                h1[i:i+65536] = L.argmax(1) == y[i:i+65536]
                h5[i:i+65536] = [yy in r for yy, r in zip(y[i:i+65536], t5)]
        return h1, h5

    res = {}
    for exam, Xt, lt in [("old_test", X[te_A], lab[te_A]), ("asia043_test", X43[te43], l43[te43])]:
        res[exam] = {}
        for nm, hp in [("A_head", HEAD_A), ("B_head", HEAD_B)]:
            h1, h5 = evalhead(hp, Xt, lt)
            per = {}
            for c in sorted(set(lt.tolist())):
                m = lt == c
                per[c] = {"n": int(m.sum()), "top1": round(float(h1[m].mean()), 4), "top5": round(float(h5[m].mean()), 4)}
            pres = sorted(set(lt.tolist()))
            res[exam][nm] = {"micro_top1": round(float(h1.mean()), 4), "micro_top5": round(float(h5.mean()), 4),
                             "macro_top5": round(float(np.mean([per[c]["top5"] for c in pres])), 4), "per_class": per}
            print(f"[{exam}][{nm}] micro top1={h1.mean():.4f} top5={h5.mean():.4f} macro5={res[exam][nm]['macro_top5']:.4f}", flush=True)

    # ---- per-class Δ 표 ----
    for exam in res:
        A = res[exam]["A_head"]["per_class"]; B = res[exam]["B_head"]["per_class"]
        print(f"\n=== {exam}: per-class top5 (A_head → B_head, 같은 행) ===")
        for c in sorted(A):
            print(f"  {c:10} n={A[c]['n']:>6}  {A[c]['top5']:.4f} → {B[c]['top5']:.4f}  Δ{B[c]['top5']-A[c]['top5']:+.4f}")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
