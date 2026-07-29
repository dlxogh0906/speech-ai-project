# -*- coding: utf-8 -*-
"""점수 3종 회귀기를 학습해 **파일로 저장**한다 (지금까지는 평가할 때마다 즉석 학습했다).

채택 구성 = v2-gbm-utt: 발화 단위로 GBM을 학습하고, 서비스에서는 화자(세션) 단위로 평균낸다.
  · 발화 단위 학습을 쓰는 이유: train 화자당 12.9발화 / test 148.9발화로 집계 표본 수가 크게
    달라, 화자 단위로 fit 하면 학습·서비스 분포가 어긋난다. 발화 단위는 그 비대칭을 우회한다.
  · 서비스에서 발화 하나만으로 점수를 내면 안 된다(PCC 0.43). 20~30발화 모으면 포화(0.65~0.76).

저장물: 모델 3개 + 특징 이름 순서 + 재현에 필요한 메타. 특징 순서가 어긋나면 조용히 틀리므로
       API 쪽에서 feature_names 를 그대로 읽어 쓴다.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import json, argparse, collections
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

ap = argparse.ArgumentParser()
ap.add_argument("--train", default=f"{MDD_OUTPUTS}/gop_feats_v2_train.jsonl")
ap.add_argument("--test", default=f"{MDD_OUTPUTS}/gop_feats_v2_test.jsonl")
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/score_models_v2.joblib")
ap.add_argument("--min_utts", type=int, default=3)
A = ap.parse_args()

DROP = {"stem", "pron", "comp", "flu", "pred", "gops"}
DIMS = (("pron", "발음숙달"), ("comp", "이해가능"), ("flu", "유창성"))
load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]

tr, te = load(A.train), load(A.test)
names = sorted(k for k in tr[0] if k not in DROP and isinstance(tr[0][k], (int, float)))
print(f"[data] train {len(tr):,} · test {len(te):,} · 특징 {len(names)}개")

# 화자 분리 확인 — 여기서 새는 순간 아래 수치가 전부 거짓이 된다
s_tr = {r["stem"][:5] for r in tr}; s_te = {r["stem"][:5] for r in te}
assert not (s_tr & s_te), f"화자 누출 {len(s_tr & s_te)}명"
print(f"[화자] train {len(s_tr):,} · test {len(s_te):,} · 겹침 0 확인\n")

pcc = lambda a, b: float(np.corrcoef(a, b)[0, 1])
bundle = {"feature_names": names, "min_utts": A.min_utts, "models": {}, "metrics": {}}
print(f"{'차원':9}{'학습 발화':>11}{'평가 화자':>10}{'발화단위':>10}{'화자단위':>10}")
for dim, name in DIMS:
    a = [x for x in tr if x.get(dim, 0) and x[dim] > 0]
    b = [x for x in te if x.get(dim, 0) and x[dim] > 0]
    X = np.array([[x[k] for k in names] for x in a]); y = np.array([x[dim] for x in a])
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=4,
                                      min_samples_leaf=200, l2_regularization=1.0,
                                      random_state=0).fit(X, y)
    p = m.predict(np.array([[x[k] for k in names] for x in b]))
    t = np.array([x[dim] for x in b])
    ag, al = collections.defaultdict(list), collections.defaultdict(list)
    for x, q in zip(b, p):
        ag[x["stem"][:5]].append(q); al[x["stem"][:5]].append(x[dim])
    ks = [s for s in ag if len(ag[s]) >= A.min_utts]
    u, sp = pcc(p, t), pcc([np.mean(ag[s]) for s in ks], [np.mean(al[s]) for s in ks])
    bundle["models"][dim] = m
    bundle["metrics"][dim] = {"name": name, "pcc_utt": round(u, 4), "pcc_spk": round(sp, 4),
                              "n_train": len(a), "n_test_spk": len(ks)}
    print(f"{name:9}{len(a):>11,}{len(ks):>10,}{u:>10.4f}{sp:>10.4f}")

joblib.dump(bundle, A.out, compress=3)
print(f"\n[saved] {A.out}")
print("  서비스에서는 발화별 예측을 세션(20~30발화) 평균해 쓸 것 — 발화 하나는 신뢰 불가")
