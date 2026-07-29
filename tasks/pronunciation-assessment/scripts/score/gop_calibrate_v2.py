# -*- coding: utf-8 -*-
"""점수 3종 보정·평가 v2 — 확장 특징 + GBM.

인코더는 동결. 학습되는 건 화자 단위 회귀기 하나뿐(트리 또는 선형).
게이트 판정은 **화자 단위 PCC** (사람 점수가 화자에 지배되므로 발화 단위는 참고용).

비교하는 4가지:
  v1-ridge   기존 6특징 + Ridge          (재현 기준선: 유창성 0.709 / 이해 0.598 / 발음 0.561)
  v2-ridge   확장 특징 + Ridge           (특징만 늘린 효과)
  v2-gbm     확장 특징 + GBM             (비선형 효과)
  v2-gbm-utt 발화 단위로 fit → 화자 평균  (표본 수를 늘린 효과)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import json, argparse, collections
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--train", default=f"{MDD_OUTPUTS}/gop_feats_v2_train.jsonl")
ap.add_argument("--test", default=f"{MDD_OUTPUTS}/gop_feats_v2_test.jsonl")
ap.add_argument("--v1_train", default=f"{MDD_OUTPUTS}/gop_feats_train.jsonl")
ap.add_argument("--v1_test", default=f"{MDD_OUTPUTS}/gop_feats_test.jsonl")
ap.add_argument("--max_utts", type=int, default=30, help="화자당 집계 발화 수 상한(제품 시나리오)")
ap.add_argument("--min_utts", type=int, default=3)
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/eval_score_v2.json")
A = ap.parse_args()

DROP = {"stem", "pron", "comp", "flu", "pred", "gops"}
V1 = ["gop", "neg_nll", "neg_per", "rate", "blank_frac", "max_blank"]
# 화자 내 변동성도 신호다(들쭉날쭉한 화자) — 이 특징들은 평균과 표준편차를 함께 쓴다
STD_OF = ["neg_per", "gop_p10", "gop_bad", "r_onset", "r_coda", "r_vowel", "err_clump", "artic_rate"]
DIMS = (("pron", "발음숙달"), ("comp", "이해가능"), ("flu", "유창성"))


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        rows.append(json.loads(line))
    return rows


def feat_names(rows, allow=None):
    ks = [k for k in rows[0] if k not in DROP and isinstance(rows[0][k], (int, float))]
    if allow is not None:
        ks = [k for k in ks if k in allow]
    return sorted(ks)


def by_speaker(rows, dim, names, max_utts, min_utts, rng):
    """화자별로 라벨 있는 발화를 무작위 표집 → 특징 평균(+일부 표준편차)과 라벨 평균."""
    spk = collections.defaultdict(list)
    for r in rows:
        if r.get(dim, 0) and r[dim] > 0:
            spk[r["stem"][:5]].append(r)
    X, y, ids = [], [], []
    stds = [k for k in STD_OF if k in names]
    for s, rs in spk.items():
        if len(rs) > max_utts:
            rng.shuffle(rs); rs = rs[:max_utts]
        if len(rs) < min_utts:
            continue
        v = [np.mean([r[k] for r in rs]) for k in names]
        v += [np.std([r[k] for r in rs]) for k in stds]
        X.append(v); y.append(np.mean([r[dim] for r in rs])); ids.append(s)
    return np.array(X), np.array(y), ids


def pcc(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def fit_eval(Xtr, ytr, Xte, yte, kind):
    if kind == "ridge":
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
        return pcc(m.predict(sc.transform(Xte)), yte)
    m = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=4,
        min_samples_leaf=40, l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
    return pcc(m.predict(Xte), yte)


tr2, te2 = load(A.train), load(A.test)
tr1, te1 = load(A.v1_train), load(A.v1_test)
n2, n1 = feat_names(tr2), feat_names(tr1, allow=set(V1))
print(f"[data] v2 train {len(tr2):,} / test {len(te2):,} · 특징 {len(n2)}개")
print(f"       v1 train {len(tr1):,} / test {len(te1):,} · 특징 {len(n1)}개")
print(f"[집계] 화자당 최대 {A.max_utts}발화 · 최소 {A.min_utts}발화\n")

res = {}
print(f"{'차원':8}{'화자(tr/te)':>16}{'v1-ridge':>10}{'v2-ridge':>10}{'v2-gbm':>10}{'v2-gbm-utt':>12}")
for dim, name in DIMS:
    rng = np.random.default_rng(0)
    X1a, y1a, _ = by_speaker(tr1, dim, n1, A.max_utts, A.min_utts, rng)
    X1b, y1b, _ = by_speaker(te1, dim, n1, A.max_utts, A.min_utts, rng)
    X2a, y2a, ida = by_speaker(tr2, dim, n2, A.max_utts, A.min_utts, rng)
    X2b, y2b, idb = by_speaker(te2, dim, n2, A.max_utts, A.min_utts, rng)
    assert not (set(ida) & set(idb)), "화자 누출!"
    if min(len(y1a), len(y1b), len(y2a), len(y2b)) < 50:
        print(f"{name:8}  데이터 부족"); continue

    r = {"n_train_spk": len(y2a), "n_test_spk": len(y2b),
         "v1_ridge": round(fit_eval(X1a, y1a, X1b, y1b, "ridge"), 4),
         "v2_ridge": round(fit_eval(X2a, y2a, X2b, y2b, "ridge"), 4),
         "v2_gbm": round(fit_eval(X2a, y2a, X2b, y2b, "gbm"), 4)}

    # 발화 단위 fit → 화자 평균 (표본 수를 늘리는 효과 확인)
    utr = [x for x in tr2 if x.get(dim, 0) and x[dim] > 0]
    ute = [x for x in te2 if x.get(dim, 0) and x[dim] > 0]
    Xu = np.array([[x[k] for k in n2] for x in utr]); yu = np.array([x[dim] for x in utr])
    Xv = np.array([[x[k] for k in n2] for x in ute])
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=4,
                                      min_samples_leaf=200, l2_regularization=1.0,
                                      random_state=0).fit(Xu, yu)
    pv = m.predict(Xv)
    agg = collections.defaultdict(list); lab = {}
    for x, p in zip(ute, pv):
        agg[x["stem"][:5]].append(p); lab[x["stem"][:5]] = lab.get(x["stem"][:5], []) + [x[dim]]
    ks = [s for s in agg if len(agg[s]) >= A.min_utts]
    r["v2_gbm_utt"] = round(pcc([np.mean(agg[s]) for s in ks], [np.mean(lab[s]) for s in ks]), 4)
    r["n_utt_train"] = len(utr)
    res[dim] = r
    print(f"{name:8}{len(y2a):>7,}/{len(y2b):<8,}"
          f"{r['v1_ridge']:>10}{r['v2_ridge']:>10}{r['v2_gbm']:>10}{r['v2_gbm_utt']:>12}")

print("\n(게이트 PCC ≥ 0.60 · 공식 회귀 팩 저자 보고 0.7228)")
json.dump({"max_utts": A.max_utts, "n_feats_v2": len(n2), "dims": res},
          open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[saved] {A.out}")
