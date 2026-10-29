# -*- coding: utf-8 -*-
"""영어 문장 점수를 한국인 학습자(AIHub 037) 사람 점수에 맞춘다 — 평가 + 최종 학습 + 백분위 눈금.

입력: `en_extract_037_feats.py` 가 쓴 $MDD_OUTPUTS/english/score037/<model>/feats.*.jsonl
목표: 사람 조음(art) → pronunciation · 사람 운율(pros) → rhythm (1~5, 평가자 2명 평균)
      fluency 는 037 에 라벨이 없어 speechocean762 유창성 예측(so_fluency, 0~10)을 그대로 ×10
      overall = 0.6·발음 + 0.2·유창 + 0.2·운율 — 한국어 API 와 같은 비율
회귀기: 부스팅(HGB) · 선형(Ridge) · 둘의 평균 중 화자 5겹 발화 상관이 가장 높은 것을 고른다
눈금: 기본 human(예측을 사람 점수 분포의 평균·폭으로 되편 뒤 ×20). 백분위(pct)는 --scale pct
평가 (비교 기준 = 지금 응답 4칸, speechocean762 기준)
  A. 화자 5겹 — 모든 발화의 겹밖 예측. 발화 PCC · 화자 PCC(화자 전체 평균 · 무작위 5발화 평균)
  B. 화자도 문장도 처음 — 화자 20% · 문장 20% 를 빼고 학습하고, 둘 다 처음인 발화에서만 잰다
     (앱 사용자는 처음 보는 문장을 읽는다. 037 은 2천여 문장을 여러 화자가 반복해 읽었다)
산출: <out>/ = 기존 점수기 파일 사본 + sent_scorer_ko.joblib · sent_scorer_ko.json · sent_scorer_ko.md
      서버는 `--scorer-dir <out>` 으로 띄우면 새 점수를 쓴다(`mdd_en.sent_score`).

    $PY .../en_fit_sentence_scorer_037.py --model synth-d
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, glob, json, os, shutil

import numpy as np

TARGETS = (("pronunciation", "art"), ("rhythm", "pros"))
HGB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
           random_state=0)
BASE_FILES = ("phone_scorer.joblib", "phone_scorer.json", "sent_scorer.joblib", "sent_scorer.json", "flag_policy.json")


def pcc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def spk_pcc(pred, y, spk, k=None, reps=20, seed=0):
    """화자 단위 상관. k=None 이면 화자의 모든 발화 평균, k 면 무작위 k 발화 평균(reps 번 평균)."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    idx = {}
    for i, s in enumerate(spk):
        idx.setdefault(s, []).append(i)
    groups = [np.array(v) for v in idx.values() if k is None or len(v) >= k]
    if k is None:
        return pcc([pred[g].mean() for g in groups], [y[g].mean() for g in groups])
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(reps):
        sel = [rng.choice(g, k, replace=False) for g in groups]
        out.append(pcc([pred[g].mean() for g in sel], [y[g].mean() for g in sel]))
    return float(np.mean(out))


def describe(v):
    v = np.asarray(v, float)
    return "%.0f ± %.0f (p10 %.0f · p50 %.0f · p90 %.0f)" % (v.mean(), v.std(), *np.percentile(v, [10, 50, 90]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="synth-d")
    ap.add_argument("--feats", default=None, help="기본: $MDD_OUTPUTS/english/score037/<model>")
    ap.add_argument("--base", default=None, help="기존 점수기 폴더(기본: $MDD_OUTPUTS/english/scorer/<model>)")
    ap.add_argument("--out", default=None, help="기본: $MDD_OUTPUTS/english/scorer/<model>-ko037")
    ap.add_argument("--scale", default="human", choices=["human", "pct"])
    a = ap.parse_args()
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from mdd_en.sent_score import FEAT_COLS, PCT_STEPS, W_OVERALL, to_display

    fd = a.feats or os.path.join(str(MDD_OUTPUTS), "english", "score037", a.model)
    R, seen = [], set()
    for p in sorted(glob.glob(os.path.join(fd, "feats.*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] not in seen:
                seen.add(r["id"]); R.append(r)
    R.sort(key=lambda r: r["id"])
    X = np.array([[float(r["feats"][c]) for c in FEAT_COLS] for r in R], np.float32)
    Y = {"art": np.array([r["art"] for r in R], float), "pros": np.array([r["pros"] for r in R], float)}
    Y["avg"] = (Y["art"] + Y["pros"]) / 2
    spk = np.array([r["spk"] for r in R])
    tk = np.array([r["text_key"] for r in R])
    api = dict((k, np.array([r["api"][k] or 0 for r in R], float)) for k in ("overall", "pronunciation", "fluency", "rhythm"))
    fl = X[:, FEAT_COLS.index("so_fluency")]
    print("[데이터] 발화 %d · 화자 %d · 문장 %d" % (len(R), len(set(spk)), len(set(tk))), flush=True)

    # ── A. 화자 5겹 겹밖 예측 ──
    makers = {"hgb": lambda: [HistGradientBoostingRegressor(**HGB)],
              "ridge": lambda: [make_pipeline(StandardScaler(), Ridge(alpha=1.0))],
              "ens": lambda: [HistGradientBoostingRegressor(**HGB), make_pipeline(StandardScaler(), Ridge(alpha=1.0))]}

    def fit_predict(kind, t, i_fit, i_out):
        ms = [m.fit(X[i_fit], Y[t][i_fit]) for m in makers[kind]()]
        return np.clip(np.mean([m.predict(X[i_out]) for m in ms], 0), 1, 5), ms

    cand = dict(((k, kind), np.zeros(len(R))) for k, _ in TARGETS for kind in makers)
    for tr, te in GroupKFold(5).split(X, groups=spk):
        for k, t in TARGETS:
            for kind in ("hgb", "ridge"):
                cand[(k, kind)][te] = fit_predict(kind, t, tr, te)[0]
    for k, _ in TARGETS:
        cand[(k, "ens")] = (cand[(k, "hgb")] + cand[(k, "ridge")]) / 2     # 같은 겹 · 같은 모델의 평균 = ens 와 같다
    choice = dict((k, max(makers, key=lambda kind: pcc(cand[(k, kind)], Y[t]))) for k, t in TARGETS)
    oof = dict((k, cand[(k, choice[k])]) for k, _ in TARGETS)
    print("[회귀기] " + " · ".join("%s: %s" % (k, " ".join("%s %.4f" % (kind, pcc(cand[(k, kind)], Y[t])) for kind in makers))
                                   for k, t in TARGETS) + " → %s" % choice, flush=True)

    def scale_of(k, t, kind):
        if kind == "pct":
            return {"kind": "pct", "q": np.percentile(oof[k], PCT_STEPS).tolist()}
        return {"kind": "human", "mu_p": float(oof[k].mean()), "sd_p": float(oof[k].std()),
                "mu_h": float(Y[t].mean()), "sd_h": float(Y[t].std()), "top": 5.0}

    scales = dict((s_, dict((k, scale_of(k, t, s_)) for k, t in TARGETS)) for s_ in ("human", "pct"))
    scales["human"]["fluency"] = {"kind": "x10"}
    scales["pct"]["fluency"] = {"kind": "pct", "q": np.percentile(fl, PCT_STEPS).tolist()}

    def display(pr, pros, flu, sc):
        new = {"pronunciation": np.array([to_display(v, sc["pronunciation"]) for v in pr], float),
               "rhythm": np.array([to_display(v, sc["rhythm"]) for v in pros], float),
               "fluency": np.array([to_display(v, sc["fluency"]) for v in flu], float)}
        new["overall"] = sum(w * new[k] for k, w in W_OVERALL) / sum(w for _, w in W_OVERALL)
        return new

    new = display(oof["pronunciation"], oof["rhythm"], fl, scales[a.scale])
    alt = display(oof["pronunciation"], oof["rhythm"], fl, scales["pct" if a.scale == "human" else "human"])
    rows = [("지금 overall (so762 종합)", api["overall"]), ("지금 pronunciation (음소 점수 평균)", api["pronunciation"]),
            ("지금 rhythm (so762 운율)", api["rhythm"]), ("지금 fluency (so762 유창)", api["fluency"]),
            ("새 pronunciation (037 조음 예측)", new["pronunciation"]), ("새 rhythm (037 운율 예측)", new["rhythm"]),
            ("새 fluency (so762 유창)", new["fluency"]), ("새 overall (0.6·0.2·0.2)", new["overall"])]
    rows += [("(참고) 조음 예측 · %s" % kind, cand[("pronunciation", kind)]) for kind in makers]
    rep = {"model": a.model, "n_utts": len(R), "n_spk": int(len(set(spk))), "n_sent": int(len(set(tk))),
           "choice": choice, "scale": a.scale,
           "A_speaker5fold": {}, "B_unseen_speaker_sentence": {}, "distribution": {}}
    lines = ["## A. 화자 5겹 (처음 보는 화자, 문장은 겹칠 수 있음) — 발화 %d · 화자 %d" % (len(R), len(set(spk))), "",
             "| 점수 | 발화: 조음 | 발화: 운율 | 발화: 평균 | 화자 5발화: 조음 | 화자 전체: 조음 | 화자 5발화: 평균 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, v in rows:
        m = {"art": pcc(v, Y["art"]), "pros": pcc(v, Y["pros"]), "avg": pcc(v, Y["avg"]),
             "spk5_art": spk_pcc(v, Y["art"], spk, 5), "spk_all_art": spk_pcc(v, Y["art"], spk),
             "spk5_avg": spk_pcc(v, Y["avg"], spk, 5)}
        rep["A_speaker5fold"][name] = dict((k, round(x, 4)) for k, x in m.items())
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (name, m["art"], m["pros"], m["avg"],
                                                                         m["spk5_art"], m["spk_all_art"], m["spk5_avg"]))
    lines += ["", "화면 점수 분포(0~100, 평균 ± 표준편차) — 지금 → 새(%s) · 다른 눈금(%s):" % (
        a.scale, "pct" if a.scale == "human" else "human"), ""]
    for k in ("overall", "pronunciation", "fluency", "rhythm"):
        rep["distribution"][k] = {"now": describe(api[k]), "new": describe(new[k]), "alt": describe(alt[k])}
        lines.append("- %s: 지금 %s → 새 %s · 다른 눈금 %s" % (k, describe(api[k]), describe(new[k]), describe(alt[k])))
    lines += ["", "사람 조음 점수별 새 pronunciation 평균(눈금 %s):" % a.scale, ""]
    for h in sorted(set(Y["art"].tolist())):
        sel = Y["art"] == h
        lines.append("- 사람 %.1f점(%d발화): 새 %.0f · 지금 %.0f" % (h, sel.sum(), new["pronunciation"][sel].mean(),
                                                           api["pronunciation"][sel].mean()))

    # ── B. 화자도 문장도 처음 ──
    rng = np.random.RandomState(0)
    us, ut = np.array(sorted(set(spk))), np.array(sorted(set(tk)))
    s_te = set(rng.choice(us, int(len(us) * 0.2), replace=False))
    t_te = set(rng.choice(ut, int(len(ut) * 0.2), replace=False))
    in_s = np.array([s in s_te for s in spk]); in_t = np.array([t in t_te for t in tk])
    tr, te_new, te_seen = (~in_s) & (~in_t), in_s & in_t, in_s & (~in_t)
    pb = {}
    for k, t in TARGETS:
        pb[k] = fit_predict(choice[k], t, np.flatnonzero(tr), np.arange(len(R)))[0]
    nb = display(pb["pronunciation"], pb["rhythm"], fl, scales[a.scale])
    lines += ["", "## B. 처음 보는 화자 × 처음 보는 문장 — 학습 %d 발화, 평가 %d 발화(문장도 처음) / %d 발화(문장은 학습에 있음)"
              % (tr.sum(), te_new.sum(), te_seen.sum()), "",
              "| 점수 | 문장도 처음: 조음 | 문장도 처음: 평균 | 문장은 본 것: 조음 | 문장은 본 것: 평균 |", "|---|---:|---:|---:|---:|"]
    for name, v in (("지금 overall", api["overall"]), ("지금 pronunciation", api["pronunciation"]),
                    ("새 pronunciation", nb["pronunciation"]), ("새 overall", nb["overall"])):
        m = {"new_art": pcc(v[te_new], Y["art"][te_new]), "new_avg": pcc(v[te_new], Y["avg"][te_new]),
             "seen_art": pcc(v[te_seen], Y["art"][te_seen]), "seen_avg": pcc(v[te_seen], Y["avg"][te_seen])}
        rep["B_unseen_speaker_sentence"][name] = dict((k, round(x, 4)) for k, x in m.items())
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (name, m["new_art"], m["new_avg"], m["seen_art"], m["seen_avg"]))
    rep["B_sizes"] = {"train": int(tr.sum()), "test_new_sentence": int(te_new.sum()), "test_seen_sentence": int(te_seen.sum())}

    # ── 최종: 전부로 학습 ──
    regs = dict((k, fit_predict(choice[k], t, np.arange(len(R)), np.arange(1))[1]) for k, t in TARGETS)
    out = a.out or os.path.join(str(MDD_OUTPUTS), "english", "scorer", a.model + "-ko037")
    base = a.base or os.path.join(str(MDD_OUTPUTS), "english", "scorer", a.model)
    os.makedirs(out, exist_ok=True)
    for f in BASE_FILES:
        if os.path.exists(os.path.join(base, f)):
            shutil.copy2(os.path.join(base, f), os.path.join(out, f))
    info = {"targets": {"pronunciation": "037 articulationScore(1~5)", "rhythm": "037 prosodyScore(1~5)",
                        "fluency": "so762 유창성 예측(0~10) 그대로 — 037 에 유창성 라벨 없음"},
            "scale": a.scale, "scale_note": ("예측을 사람 점수 분포의 평균·폭으로 되편 뒤 ×20(사람 4점 ≈ 80)"
                                             if a.scale == "human" else "037 학습자 분포 백분위(80 = 상위 20%)"),
            "display_note": ("발음·운율 = 한국인 학습자(AIHub 037) 사람 점수(조음·운율, 1~5) 예측을 사람 점수 분포의 평균·폭으로"
                             " 되편 뒤 ×20(사람 4점 ≈ 80) · 유창 = 속도·쉼 기반 유창성 예측(speechocean762, 0~10) ×10"
                             if a.scale == "human" else
                             "발음·운율·유창 = 한국인 학습자(AIHub 037) 분포의 백분위(80 = 상위 20%) — 발음 ← 사람 조음 점수 예측"
                             " · 운율 ← 사람 운율 점수 예측 · 유창 ← speechocean762 유창성 예측"),
            "regressor": choice, "weights": dict(W_OVERALL),
            "n_utts": len(R), "n_spk": int(len(set(spk))), "base_scorer": base, "model": a.model}
    joblib.dump({"regs": regs, "direct": {"fluency": "so_fluency"}, "cols": FEAT_COLS, "scale": scales[a.scale],
                 "scale_alt": scales["pct" if a.scale == "human" else "human"], "info": info},
                os.path.join(out, "sent_scorer_ko.joblib"))
    rep["info"] = info
    json.dump(rep, open(os.path.join(out, "sent_scorer_ko.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(out, "sent_scorer_ko.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n[저장] %s" % out, flush=True)


if __name__ == "__main__":
    main()
