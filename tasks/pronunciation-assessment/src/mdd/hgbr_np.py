# -*- coding: utf-8 -*-
"""점수 모델(HistGradientBoostingRegressor 3종)을 scikit-learn 없이 numpy 만으로 예측한다.

왜: `score_models_v2_w2vbert.joblib` 은 scikit-learn 1.3.2(py3.8, numpy 1.24)로 저장됐다. numpy 2 가 깔린 곳
(NPU 서버 py3.12)에서는 1.3.2 를 설치할 수 없고, 새 버전(1.9.1)은 이 pickle 을
`Can't get attribute '__pyx_unpickle_CyHalfSquaredError'` 로 열지 못한다(2026-09-27 실측).
그래서 트리 노드 값(분기 특징·임계값·결측 방향·자식·잎 값)만 꺼내 .npz 로 옮기고, 예측을 여기서 다시 한다.

예측 규칙은 sklearn `_predictor.pyx` 와 같다 — 결측(NaN)이면 missing_go_to_left, 아니면 x <= num_threshold 면 왼쪽.
합산도 sklearn 과 같은 순서(기준값 + 1번 트리 + 2번 트리 …, float64)라 결과가 비트 단위로 같다.
지원 범위: 제곱·절대오차 회귀(항등 링크) · 반복당 트리 1개 · 범주형 특징 없음. 벗어나면 export 에서 멈춘다.

    export:  scripts/export/export_scorer_npz.py (sklearn 이 있는 환경에서)
    사용:    mdd.session.SessionScorer 가 경로가 .npz 면 이걸로 읽는다.
"""
import json

import numpy as np

_FIELDS = ("feature_idx", "num_threshold", "missing_go_to_left", "left", "right", "is_leaf", "value")


def export_hgbr(m):
    """학습된 HistGradientBoostingRegressor → 배열 dict (sklearn 이 있는 환경에서 부른다)."""
    assert type(m._loss.link).__name__ == "IdentityLink", "항등 링크 회귀만 지원한다"
    assert getattr(m, "is_categorical_", None) is None and getattr(m, "_preprocessor", None) is None, "범주형 특징은 지원하지 않는다"
    assert all(len(it) == 1 for it in m._predictors), "반복당 트리 1개(회귀)만 지원한다"
    nodes = [it[0].nodes for it in m._predictors]
    sizes = [len(n) for n in nodes]
    roots = np.cumsum([0] + sizes)[:-1].astype(np.int64)
    cat = np.concatenate(nodes)
    shift = np.repeat(roots, sizes)
    leaf = cat["is_leaf"].astype(bool)
    return {"feature_idx": cat["feature_idx"].astype(np.int64),
            "num_threshold": cat["num_threshold"].astype(np.float64),
            "missing_go_to_left": cat["missing_go_to_left"].astype(bool),
            "left": np.where(leaf, 0, cat["left"].astype(np.int64) + shift),
            "right": np.where(leaf, 0, cat["right"].astype(np.int64) + shift),
            "is_leaf": leaf, "value": cat["value"].astype(np.float64), "roots": roots,
            "baseline": np.asarray(m._baseline_prediction, np.float64).ravel()[:1],
            "n_features": np.array([m.n_features_in_], np.int64)}


class NpHGBR:
    """export_hgbr 배열로 predict(X) 하는 예측기. X: [N, n_features] float."""

    def __init__(self, a):
        self.a = {k: np.asarray(v) for k, v in a.items()}
        self.n_features = int(self.a["n_features"][0])

    def predict(self, X):
        a = self.a
        X = np.asarray(X, np.float64)
        assert X.ndim == 2 and X.shape[1] == self.n_features, f"특징 수가 다르다: {X.shape} vs {self.n_features}"
        rows = np.arange(X.shape[0])[:, None]
        node = np.broadcast_to(a["roots"], (X.shape[0], len(a["roots"]))).copy()
        while True:
            leaf = a["is_leaf"][node]
            if leaf.all():
                break
            x = X[rows, a["feature_idx"][node]]
            left = np.where(np.isnan(x), a["missing_go_to_left"][node], x <= a["num_threshold"][node])
            node = np.where(leaf, node, np.where(left, a["left"][node], a["right"][node]))
        vals = a["value"][node]
        out = np.full(X.shape[0], a["baseline"][0])
        for j in range(vals.shape[1]):                 # sklearn 과 같은 합산 순서
            out += vals[:, j]
        return out


def save_scorer_npz(path, bundle):
    """joblib 점수 묶음({"models": {pron, comp, flu}, "feature_names", "min_utts", "metrics"}) → .npz"""
    arrs = {"feature_names": np.array(bundle["feature_names"], dtype=str),
            "meta": np.array(json.dumps({"min_utts": bundle.get("min_utts"), "keys": list(bundle["models"]),
                                         "metrics": bundle.get("metrics")}, ensure_ascii=False, default=float))}
    for k, m in bundle["models"].items():
        for f, v in export_hgbr(m).items():
            arrs[f"{k}__{f}"] = v
    np.savez(path, **arrs)


def load_scorer_npz(path):
    """save_scorer_npz 산출물 → joblib 묶음과 같은 모양의 dict (models 값이 NpHGBR)."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    models = {k: NpHGBR({f: z[f"{k}__{f}"] for f in _FIELDS + ("roots", "baseline", "n_features")}) for k in meta["keys"]}
    return {"models": models, "feature_names": [str(s) for s in z["feature_names"]],
            "min_utts": meta.get("min_utts"), "metrics": meta.get("metrics")}
