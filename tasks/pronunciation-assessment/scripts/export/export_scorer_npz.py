# -*- coding: utf-8 -*-
"""점수 모델 joblib(HistGradientBoostingRegressor 3종) → sklearn 없이 읽는 .npz (`mdd.hgbr_np`).

sklearn 버전이 다른 곳(예: numpy 2 · py3.12 인 NPU 서버)에서도 같은 점수를 내려고 만든다.
저장 뒤 원본 predict 와 numpy 예측을 비교해 최대 오차를 출력한다 — 0 이 아니면 실패로 끝난다.

    $PY tasks/pronunciation-assessment/scripts/export/export_scorer_npz.py --scorer <score_models.joblib> --out <out.npz> \
        [--check <X 를 담은 npz>]
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorer", required=True, help="train_score_models.py 의 joblib")
    ap.add_argument("--out", required=True, help="출력 .npz")
    ap.add_argument("--check", default=None, help="비교용 특징 행렬(키 X)을 담은 npz. 없으면 무작위 5,000행만 쓴다")
    a = ap.parse_args()
    import joblib
    from mdd.hgbr_np import save_scorer_npz, load_scorer_npz
    B = joblib.load(a.scorer)
    save_scorer_npz(a.out, B)
    N = load_scorer_npz(a.out)
    nf = len(B["feature_names"])
    rng = np.random.default_rng(0)
    X = [rng.normal(0, 3, size=(5000, nf))]
    if a.check:
        X.append(np.load(a.check)["X"])
    X = np.vstack(X)
    X[rng.random(X.shape) < 0.01] = np.nan                       # 결측 방향도 확인
    worst = 0.0
    for k, m in B["models"].items():
        d = float(np.max(np.abs(m.predict(X) - N["models"][k].predict(X))))
        worst = max(worst, d)
        print(f"{k}: 최대 |Δ| {d:.3e} ({len(X)}행)")
    print(f"저장 {a.out} · 특징 {nf} · 최대 |Δ| {worst:.3e}")
    if worst != 0.0:
        raise SystemExit("원본과 예측이 다르다")


if __name__ == "__main__":
    main()
