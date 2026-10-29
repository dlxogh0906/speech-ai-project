# -*- coding: utf-8 -*-
"""영어 문장 점수기 학습·평가 — 문장 특징 → 사람 문장 점수(정확도·유창성·운율·종합·완결성, speechocean762).

입력: `en_fit_phone_scorer.py` 가 남긴 음소 특징표($MDD_OUTPUTS/english/scorer/<모델>/feats_*.npz)와 음소 점수기.
train 발화의 음소 점수는 **화자 단위 5겹 교차 예측**으로 만든다 — 같은 train 으로 학습한 음소 점수기를 그대로 쓰면
train 쪽 점수가 실제보다 좋게 나와 문장 회귀기가 엉뚱하게 배운다.

평가: speechocean762 test PCC (공개 수치: GOPT 정확도 0.714 · 유창성 0.753 · 운율 0.760 · 종합 0.742)

    $PY .../en_fit_sentence_scorer.py --model lv60-espeak
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, json, os

import numpy as np

TARGETS = ("accuracy", "fluency", "prosodic", "total", "completeness")
PUBLISHED = {"GOPT": {"accuracy": 0.714, "fluency": 0.753, "prosodic": 0.760, "total": 0.742, "completeness": 0.155}}


def utt_table(D, phone_scores):
    from mdd_en.scorer import COLS
    from mdd_en.utterance import utt_features
    mi = COLS.index("mismatch")
    rows = []
    for u in range(len(D["utts"])):
        m = D["utt"] == u
        rows.append(utt_features(phone_scores[m], D["word"][m].tolist(),
                                 list(zip(D["span_s"][m].tolist(), D["span_e"][m].tolist())),
                                 int(D["n_frames"][u]), deleted=D["deleted"][m], mismatch=D["X"][m, mi]))
    return np.stack(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scorer", default=os.path.join(str(MDD_OUTPUTS), "english", "scorer"))
    ap.add_argument("--so762", default=os.path.join(str(DATA_ROOT), "english", "speechocean762"))
    a = ap.parse_args()
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    from mdd_en.scorer import COLS
    from mdd_en.speechocean import load_split
    from mdd_en.utterance import UTT_COLS
    d = os.path.join(a.scorer, a.model)
    tr, te = [dict(np.load(os.path.join(d, "feats_%s.npz" % k))) for k in ("so762train", "so762")]
    if "span_s" not in tr:
        raise SystemExit("feats 에 정렬 구간이 없다 — en_fit_phone_scorer.py 를 --reuse 없이 다시 돌릴 것")
    ps = joblib.load(os.path.join(d, "phone_scorer.joblib"))
    pi = COLS.index("phone")
    meta = {"train": dict((r["utt"], r) for r in load_split(a.so762, "train")),
            "test": dict((r["utt"], r) for r in load_split(a.so762, "test"))}

    # train 음소 점수: 화자 단위 5겹 교차 예측
    spk = np.array([meta["train"][tr["utts"][u]]["spk"] for u in tr["utt"]])
    s_tr = np.zeros(len(tr["acc"]), np.float32)
    for fi, (i_fit, i_out) in enumerate(GroupKFold(5).split(tr["X"], groups=spk)):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, categorical_features=[pi],
                                          random_state=0).fit(tr["X"][i_fit], tr["acc"][i_fit])
        s_tr[i_out] = np.clip(m.predict(tr["X"][i_out]), 0, 2) * 50
    s_te = np.clip(ps["reg"].predict(te["X"]), 0, 2) * 50
    U_tr, U_te = utt_table(tr, s_tr), utt_table(te, s_te)
    y_tr = dict((t, np.array([meta["train"][u][t] for u in tr["utts"]], np.float32)) for t in TARGETS)
    y_te = dict((t, np.array([meta["test"][u][t] for u in te["utts"]], np.float32)) for t in TARGETS)

    rep = {"model": a.model, "published": PUBLISHED, "pcc": {}}
    regs = {}
    for t in TARGETS:
        r = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
                                          random_state=0).fit(U_tr, y_tr[t])
        p = r.predict(U_te)
        rep["pcc"][t] = round(float(np.corrcoef(p, y_te[t])[0, 1]), 4)
        regs[t] = r
    # 사람 점수 분포(눈금 참고): test 평균·표준편차
    rep["human_test_mean"] = dict((t, round(float(y_te[t].mean()), 3)) for t in TARGETS)
    rep["pred_test_mean"] = dict((t, round(float(regs[t].predict(U_te).mean()), 3)) for t in TARGETS)
    print("[%s] PCC %s  (GOPT %s)" % (a.model, rep["pcc"], PUBLISHED["GOPT"]), flush=True)
    joblib.dump({"regs": regs, "cols": UTT_COLS, "model": a.model}, os.path.join(d, "sent_scorer.joblib"))
    json.dump(rep, open(os.path.join(d, "sent_scorer.json"), "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(d, "utt_feats.npz"), U_tr=U_tr, U_te=U_te, utts_tr=tr["utts"], utts_te=te["utts"])
    print("[saved] %s" % d)


if __name__ == "__main__":
    main()
