# -*- coding: utf-8 -*-
"""문장 점수 보정 — 1~5 예측값을 **"사람이 좋게 볼 발음일 확률"** 로 바꾼다.

왜 바꾸나
    지금은 예측한 1~5 점을 그대로 0~100 으로 늘린다(3점 = 50). 사람 채점 평균이 3점대라
    보통 읽으면 50~65 가 나오고, 보정된 글자 점수(맞으면 90~100)와 자가 달라 화면이 어긋난다.

무엇으로 바꾸나
    사람 점수 >= GOOD 을 '좋음' 으로 두고, 모델 예측값에서 그 확률을 맞춘다(로지스틱 1변수).
    그러면 네 칸이 같은 기준(80↑ = 좋음)으로 읽힌다.
    ⚠️ 발화 1개의 정확도(상관 0.43)는 그대로다 — 눈금만 바뀐다.

    $PY .../calibrate_utt_scores.py --feats $MDD_OUTPUTS/score_phoneme/gop_margin_test.jsonl \\
        --scorer $MDD_OUTPUTS/score_v2_w2vbert/score_models_v2_w2vbert.joblib \\
        --out $MDD_OUTPUTS/score_phoneme
산출: utt_calibration.json · utt_calibration_report.md
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

GOOD = 3.5          # 사람 점수가 이 이상이면 '좋음'
KEYS = ("pron", "comp", "flu")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--good", type=float, default=GOOD)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import joblib
    from sklearn.linear_model import LogisticRegression
    B = joblib.load(a.scorer)
    F, M = B["feature_names"], B["models"]

    X, Y = [], {k: [] for k in KEYS}
    for line in open(a.feats, encoding="utf-8"):
        d = json.loads(line)
        if not d.get("pron"):
            continue
        X.append([float(d.get(k, 0.0)) for k in F])
        for k in KEYS:
            Y[k].append(float(d.get(k, 0)))
    X = np.asarray(X)
    rng = np.random.RandomState(a.seed)
    tr = rng.rand(len(X)) < 0.7
    cal, L = {"_good": a.good}, ["# 문장 점수 보정 — 예측 1~5 → '좋음' 확률", "",
                                 f"- 발화 {len(X):,} · 기준: 사람 점수 ≥ {a.good} 이면 좋음", "",
                                 "| 차원 | 좋음 비율 | 보정 전(중앙) | 보정 후(중앙) | 좋은 발화 80↑ | 나쁜 발화 80↓ |",
                                 "|---|--:|--:|--:|--:|--:|"]
    for k in KEYS:
        y = np.asarray(Y[k]); p = M[k].predict(X)
        good = (y >= a.good).astype(int)
        m = LogisticRegression(max_iter=1000).fit(p[tr].reshape(-1, 1), good[tr])
        A, b = float(m.coef_[0][0]), float(m.intercept_[0])
        cal[k] = {"a": A, "b": b}
        ev = ~tr
        new = 100.0 / (1.0 + np.exp(-(A * p[ev] + b)))
        old = np.clip((p[ev] - 1) / 4 * 100, 0, 100)
        g = good[ev] == 1
        L.append("| %s | %.0f%% | %.0f | %.0f | %.1f%% | %.1f%% |"
                 % (k, 100 * good.mean(), np.median(old), np.median(new),
                    100 * (new[g] >= 80).mean(), 100 * (new[~g] < 80).mean()))
    os.makedirs(a.out, exist_ok=True)
    json.dump(cal, open(os.path.join(a.out, "utt_calibration.json"), "w", encoding="utf-8"), ensure_ascii=False)
    open(os.path.join(a.out, "utt_calibration_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
