# -*- coding: utf-8 -*-
"""GOP+보정: 동결 MDD 모델의 특징 6종 위에 화자 단위 선형 회귀(Ridge, ~7파라미터).

fit  = train 화자 (MDD와 동일 화자분리 split의 train쪽) 의 lab 사람 점수
eval = test 화자 (fit에 없음) → 차원별 PCC. 목표 ≥ 0.60 (점수 팩 저자 보고 0.7228).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import json, collections, argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--train_cache", default=f"{MDD_OUTPUTS}/gop_feats_train.jsonl")
ap.add_argument("--test_cache", default=f"{MDD_OUTPUTS}/gop_feats_test.jsonl")
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/eval_gop_calibrated.json")
ap.add_argument("--max_utts", type=int, default=30, help="화자당 집계 발화 수 상한")
A = ap.parse_args()

FEATS = ["gop", "neg_nll", "neg_per", "rate", "blank_frac", "max_blank"]


def load(path):
    spk = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line); spk[r["stem"][:5]].append(r)
    return spk


import random as _rnd


def rows(spk, dim, min_utts=3):
    X, y, ids = [], [], []
    for s, rs in spk.items():
        rs2 = [r for r in rs if r.get(dim, 0) and r[dim] > 0]
        if len(rs2) > A.max_utts:                     # 캐시가 길이순이라 앞자르기 금지 → 무작위 표본
            _rnd.Random(hash(s) & 0xffff).shuffle(rs2)
            rs2 = rs2[: A.max_utts]
        if len(rs2) < min_utts:
            continue
        X.append([np.mean([r[f] for r in rs2]) for f in FEATS])
        y.append(np.mean([r[dim] for r in rs2])); ids.append(s)
    return np.array(X), np.array(y), ids


tr, te = load(A.train_cache), load(A.test_cache)
overlap = set(tr) & set(te)
print(f"[화자] train {len(tr):,} · test {len(te):,} · 겹침 {len(overlap)} (0이어야 함)")

res = {"feats": FEATS, "pcc_test": {}, "coef": {}}
print(f"\n========== GOP+보정 (화자 단위, 특징 {len(FEATS)}개 · Ridge) ==========")
print(f"{'차원':8}{'train화자':>9}{'test화자':>9}{'단독최고(−PER)':>15}{'보정후 PCC':>12}")
for dim, name in (("pron", "발음숙달"), ("comp", "이해가능"), ("flu", "유창성")):
    Xtr, ytr, _ = rows(tr, dim)
    Xte, yte, _ = rows(te, dim)
    if len(ytr) < 50 or len(yte) < 50:
        print(f"{name:8}  데이터 부족"); continue
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    pred = m.predict(sc.transform(Xte))
    pcc = float(np.corrcoef(pred, yte)[0, 1])
    solo = float(np.corrcoef(Xte[:, FEATS.index("neg_per")], yte)[0, 1])
    res["pcc_test"][dim] = {"n_train_spk": len(ytr), "n_test_spk": len(yte),
                            "pcc_calibrated": round(pcc, 4), "pcc_negper_only": round(solo, 4)}
    res["coef"][dim] = {f: round(float(c), 3) for f, c in zip(FEATS, m.coef_)}
    print(f"{name:8}{len(ytr):>9,}{len(yte):>9,}{solo:>15.3f}{pcc:>12.3f}")
print("(목표 PCC ≥ 0.60 · 점수 팩 저자 보고 0.7228)")
json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[saved] {A.out}")
for dim in res["coef"]:
    print(f"  {dim} 가중치:", res["coef"][dim])
