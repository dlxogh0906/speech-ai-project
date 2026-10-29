# -*- coding: utf-8 -*-
"""문장 점수 눈금 — 예측값을 **학습자 분포에서의 백분위**로 편다.

왜 필요한가(실측)
    사람 채점은 외국인 학습자 기준이라 평균이 3.3/5 이고, 회귀 모델은 확신이 없으면 평균 쪽으로 붙는다.
    실제 5점짜리 발화에도 3.82 를 준다 → 5점 만점 비례로 환산하면 **잘 읽어도 76점이 천장**이다.
    순서(상관 0.43~0.61)는 맞는데 눈금만 좁다.

무엇을 하나
    학습 표본에서 모델 예측값의 분위수를 뽑아 둔다. 서비스에서는 그 표를 찾아
    "이 예측값은 학습자 중 상위 몇 %" 를 점수로 준다. **순서는 그대로, 눈금만 펴진다.**
    80점 = 학습자 상위 20% 라는 뜻이 되어 발음 점수(틀렸을 확률 기반)와 읽는 법이 같아진다.

    $PY .../calibrate_utt_percentile.py --feats $MDD_OUTPUTS/score_phoneme/gop_margin_test.jsonl \\
        --scorer $MDD_OUTPUTS/score_v2_w2vbert/score_models_v2_w2vbert.joblib --out $MDD_OUTPUTS/score_phoneme
산출: utt_percentile.json
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os

import numpy as np

KEYS = ("pron", "comp", "flu")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import joblib
    B = joblib.load(a.scorer)
    F, M = B["feature_names"], B["models"]
    rows = [json.loads(l) for l in open(a.feats, encoding="utf-8")]
    X = np.array([[float(d.get(k, 0.0)) for k in F] for d in rows])
    cal, L = {}, ["# 문장 점수 눈금 — 예측값 → 학습자 백분위", "",
                  f"- 표본 {len(X):,}발화", "",
                  "| 차원 | 예측 1분위 | 중앙 | 9분위 | 지금 환산(중앙) | 바뀐 점수(중앙) |", "|---|--:|--:|--:|--:|--:|"]
    for k in KEYS:
        m = np.array([bool(d.get(k)) for d in rows])       # 라벨 있는 발화만 (유창성은 문장에만 있다)
        p = M[k].predict(X[m])
        q = np.percentile(p, np.arange(0, 101, 2))
        cal[k] = {"q": [round(float(v), 4) for v in q], "n": int(m.sum())}
        med = float(np.median(p))
        L.append("| %s | %.2f | %.2f | %.2f | %d | %d |" % (
            k, np.percentile(p, 10), med, np.percentile(p, 90),
            round(med / 5 * 100), round(float(np.interp(med, q, np.arange(0, 101, 2))))))
    os.makedirs(a.out, exist_ok=True)
    json.dump(cal, open(os.path.join(a.out, "utt_percentile.json"), "w", encoding="utf-8"), ensure_ascii=False)
    open(os.path.join(a.out, "utt_percentile_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
