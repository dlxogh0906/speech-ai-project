# -*- coding: utf-8 -*-
"""⑨ 043 병합 진단 2: 043 테스트블록(top5 0.518) 해부 (CPU only).

가설: H1 학습화자수 병목 / H2 발음과제(짧은 발화)가 약함 / H3 혼동 패턴 / H4 화자단위 붕괴.
npz 행순서는 extract_layer_features.cap_csv가 label_id로 그룹핑한 순서 → 매니페스트를
같은 방식으로 그룹핑해 정렬 복원(화자 시퀀스 일치 assert).
출력: $NATIONALITY_RUNS/diag_043/anatomy.json + stdout
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

HEAD_B = f"{NAT_OUTPUTS}/0704_country41_sent043/head.pt"
MAN = f"{NAT_DATA}/asian_country40_manifest.csv"
OUT = Path(f"{NAT_RUNS}/diag_043/anatomy.json")


def main():
    import torch, torch.nn as nn
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # ---- npz 행순서 복원: manifest를 label_id로 안정 그룹핑 ----
    man = list(csv.DictReader(open(MAN, encoding="utf-8")))
    groups = defaultdict(list)
    for r in man:
        groups[r["label_id"]].append(r)
    ordered = [r for k in groups for r in groups[k]]          # dict 삽입순 = 등장순
    da = np.load(f"{DATA_ROOT}/taeho_layers_asia/train.npz", allow_pickle=True)
    s_npz = [str(s) for s in da["speaker"]]
    assert [r["speaker_id"] for r in ordered] == s_npz, "정렬 복원 실패"
    print(f"[align] manifest {len(man):,} rows == npz (label_id 그룹핑 순서) OK", flush=True)

    X = np.clip(np.nan_to_num(da["X"].astype(np.float32)), -50, 50)[:, 0, :]
    labs = np.array([r["label_name"] for r in ordered], dtype=object)
    spk = np.array([r["speaker_id"] for r in ordered])
    task = np.array(["speech" if "/selected_speech/" in r["audio_path"] else "pron" for r in ordered])
    dur = np.array([float(r["duration_sec"] or 0) for r in ordered])

    # ---- B split 재구성(옛 아이템은 speaker/label만 필요) ----
    rows = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
    spk_mt = {}; spk_cc = defaultdict(Counter)
    for r in rows:
        spk_mt[r["speaker_id"]] = r["mother_tongue"]; spk_cc[r["speaker_id"]][(r["country"] or "").strip()] += 1
    spk_c = {s: c.most_common(1)[0][0] for s, c in spk_cc.items()}
    def c40(s):
        mt = spk_mt.get(s); return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)
    items = []
    for sp in ["train", "valid", "test"]:
        d = np.load(f"{DATA_ROOT}/taeho_layers_full/{sp}.npz", allow_pickle=True)
        for s in d["speaker"]:
            l = c40(str(s))
            if l: items.append((str(s), l))
    kd = np.load(f"{DATA_ROOT}/taeho_layers_korean/korean.npz", allow_pickle=True)
    for s in kd["speaker"]: items.append(("KOR:" + str(s), "한국어"))
    sent = np.load(f"{DATA_ROOT}/taeho_layers_euro/sent_mask_ge4.npy")
    e2l = {r["speaker_id"]: r["label_name"] for r in csv.DictReader(open(f"{NAT_DATA}/euro_country40_manifest.csv", encoding="utf-8"))}
    de = np.load(f"{DATA_ROOT}/taeho_layers_euro/train.npz", allow_pickle=True)
    for i, s in enumerate(de["speaker"]):
        if sent[i] and e2l.get(str(s)): items.append((str(s), e2l[str(s)]))
    items += list(zip(spk.tolist(), labs.tolist()))
    asg = assign_speakers([{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l in items], [8, 1, 1], seed=42)
    te = np.array([asg[s] == "test" for s in spk])
    # 클래스별 train 화자수(병합 전체 풀)
    tr_spk = defaultdict(set)
    for s, l in items:
        if asg[s] == "train": tr_spk[l].add(s)
    print(f"[B-test 043] {int(te.sum()):,} rows / {len(set(spk[te]))} speakers", flush=True)

    # ---- head B 평가 ----
    d = torch.load(HEAD_B, map_location="cpu")
    head = nn.Linear(d["feat_dim"], len(d["labels"])); head.load_state_dict(d["head"]); head.eval()
    c2i = d["c2i"]; i2c = {v: k for k, v in c2i.items()}
    Xt = (X[te] - d["mu"]) / d["sd"]
    with torch.no_grad():
        L = head(torch.from_numpy(Xt).float()).numpy()
    yt = np.array([c2i[l] for l in labs[te]])
    pred = L.argmax(1); t5 = np.argpartition(-L, 4, axis=1)[:, :5]
    h1 = pred == yt; h5 = np.array([yy in r for yy, r in zip(yt, t5)])
    lt, st, kt, dt = labs[te], spk[te], task[te], dur[te]
    print(f"[head B] 043-test micro top1={h1.mean():.4f} top5={h5.mean():.4f} (학습로그 0.2054/0.5178 재현확인)", flush=True)

    res = {"micro": {"top1": round(float(h1.mean()), 4), "top5": round(float(h5.mean()), 4)}}
    # H1
    per = {}
    print(f"\n=== H1: per-class vs 병합 train 화자수 ===")
    for c in sorted(set(lt.tolist()), key=lambda c: -int((lt == c).sum())):
        m = lt == c
        per[c] = {"n": int(m.sum()), "test_spk": len(set(st[m])), "top1": round(float(h1[m].mean()), 4),
                  "top5": round(float(h5[m].mean()), 4), "train_spk": len(tr_spk[c])}
        print(f"  {c:10} n={per[c]['n']:>5} tspk={per[c]['test_spk']:>3} top5={per[c]['top5']:.3f} train_spk={per[c]['train_spk']}")
    from scipy.stats import spearmanr
    cs = [c for c in per]
    rho = spearmanr([per[c]["train_spk"] for c in cs], [per[c]["top5"] for c in cs]).correlation
    res["H1_per_class"] = per; res["H1_spearman_trainspk_top5"] = round(float(rho), 3)
    print(f"  Spearman(train화자수, top5) = {rho:+.3f}")
    # H2
    res["H2_task"] = {}
    print(f"\n=== H2: 과제별 ===")
    for tk in ["speech", "pron"]:
        m = kt == tk
        res["H2_task"][tk] = {"n": int(m.sum()), "top1": round(float(h1[m].mean()), 4), "top5": round(float(h5[m].mean()), 4)}
        print(f"  {tk:7} n={int(m.sum()):>5} top1={h1[m].mean():.3f} top5={h5[m].mean():.3f}")
    res["H2_duration"] = {}
    for lo, hi, nm in [(0, 4, "<4s"), (4, 8, "4-8s"), (8, 12, "8-12s"), (12, 1e9, ">=12s")]:
        m = (dt >= lo) & (dt < hi)
        if m.sum():
            res["H2_duration"][nm] = {"n": int(m.sum()), "top1": round(float(h1[m].mean()), 4), "top5": round(float(h5[m].mean()), 4)}
            print(f"  {nm:6} n={int(m.sum()):>5} top1={h1[m].mean():.3f} top5={h5[m].mean():.3f}")
    # H3
    res["H3_confusion"] = {}
    print(f"\n=== H3: 약한 클래스 top-1 혼동 상위3 ===")
    for c in ["이란", "파키스탄", "방글라데시", "네팔", "캄보디아", "말레이시아", "타지키스탄"]:
        m = lt == c
        if not m.sum(): continue
        cc = Counter(i2c[p] for p in pred[m]).most_common(3)
        res["H3_confusion"][c] = [[k, round(v / int(m.sum()), 3)] for k, v in cc]
        print(f"  {c:10} → " + ", ".join(f"{k} {v/int(m.sum()):.0%}" for k, v in cc))
    # H4
    ps = {}
    for s in set(st.tolist()):
        m = st == s
        ps[s] = {"label": lt[m][0], "n": int(m.sum()), "top5": round(float(h5[m].mean()), 4)}
    fails = {s: v for s, v in ps.items() if v["top5"] < 0.2}
    goods = {s: v for s, v in ps.items() if v["top5"] >= 0.8}
    res["H4_speakers"] = {"n_speakers": len(ps), "median_top5": round(float(np.median([v["top5"] for v in ps.values()])), 3),
                          "n_fail_lt0.2": len(fails), "fail_by_class": dict(Counter(v["label"] for v in fails.values())),
                          "n_good_ge0.8": len(goods), "good_by_class": dict(Counter(v["label"] for v in goods.values()))}
    print(f"\n=== H4: 화자단위 ===\n  {len(ps)}명 | top5<0.2 = {len(fails)}명 {res['H4_speakers']['fail_by_class']} | >=0.8 = {len(goods)}명 {res['H4_speakers']['good_by_class']}")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
