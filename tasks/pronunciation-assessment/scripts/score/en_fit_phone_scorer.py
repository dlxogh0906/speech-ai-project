# -*- coding: utf-8 -*-
"""영어 음소 점수기 학습·평가 — GOP 특징 → "이 소리가 틀렸을 확률". speechocean762 train 으로 학습한다.

입력: `en_phone_bakeoff.py` 가 남긴 모델별 캐시(발화별 음소 사후확률)
    $MDD_OUTPUTS/english/bakeoff/<모델>/{so762train,so762,l2arctic}.pkl

점수기: 사람 음소 점수(speechocean762, 5명 평균 0~2)를 바로 회귀 → ×50 = 0~100 (GOPT 와 같은 학습 목표).
    오류 확률 분류기(라벨 <1.0 · <1.5)는 비교용으로 같이 잰다.
평가
    so762 test   음소 PCC · 단어 PCC(음소 점수 평균 vs 단어 정확도) · 오류 판별 AUC   ← 공개 수치와 비교 가능
    l2arctic     한국인 4명 · 사람이 단 실제 오류에 대한 AUC 와, train 에서 정한 문턱의 정밀도·재현율
                 (CC BY-NC — 평가에만 쓴다. 학습·문턱 결정에 안 쓴다)

산출물: $MDD_OUTPUTS/english/scorer/<모델>/
    feats_<데이터>.npz   음소 특징표(문장 점수 학습에 다시 쓴다)
    phone_scorer.joblib  · phone_scorer.json(지표)

    $PY .../en_fit_phone_scorer.py --model lv60-espeak
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, json, os, pickle, re, time

import numpy as np



def strip(p):
    return re.sub(r"\d", "", p)


def build(ds, cache, meta, kind, cmu, use_variants, device):
    """캐시 + 메타 → 음소 특징표·라벨. 단어별 CMUdict 발음 후보 중 소리에 맞는 것을 고른다(길이가 같은 후보만 —
    라벨이 표준 음소 자리에 붙어 있어서)."""
    import torch
    from mdd_en.align import align
    from mdd_en.gop import AlignError, choose_variants, gop_features
    from mdd_en.phones import ARPA39, VOWELS, fold, same_sound
    from mdd_en.recognizer import PHONE_NAMES
    from mdd_en.scorer import phone_table
    idx = dict((p, i) for i, p in enumerate(PHONE_NAMES))
    to_ids = lambda ph: [idx[p] for p in fold(ph, kind)]
    alt = sorted(set(to_ids(ARPA39)))
    Xs, acc, err, utt_i, word_i, n_var = [], [], [], [], [], 0
    span_s, span_e, best_sub, deleted, n_frames, kept, n_skip = [], [], [], [], [], [], 0
    for c in cache:
        r = meta[c["utt"]]
        if ds == "l2arctic":
            ph = [p for p in r["phones"] if p["canon"] is not None]
            canon = [p["canon"] for p in ph]
            stressed = [p["stressed"] for p in ph]
            word_of = [p["w"] for p in ph]
            words = r["words"]
            a_ = [0.0 if (p["type"] in ("s", "d") and p["actual"] != p["canon"]) else 2.0 for p in ph]
        else:
            canon, stressed, word_of, a_ = [], [], [], []
            words = [w["text"] for w in r["words"]]
            for wi, w in enumerate(r["words"]):
                for p, s in zip(w["phones"], w["phones_acc"]):
                    canon.append(strip(p))
                    dg = re.search(r"\d", p)
                    stressed.append(None if dg is None else dg.group(0) != "0")
                    word_of.append(wi)
                    a_.append(s)
        post = torch.tensor(c["post"].astype(np.float32), device=device)
        eff, eff_st = list(canon), list(stressed)
        if use_variants and cmu is not None:
            # 서버처럼 CMUdict 후보 중 소리에 맞는 발음을 고르고, **강세도 고른 후보에서** 가져온다.
            # 라벨이 표준 음소 자리에 붙어 있으므로 길이가 같고 자리마다 모음/자음이 같은 후보만 받는다.
            groups = []
            for wi in range(len(words)):
                pos = [i for i, w in enumerate(word_of) if w == wi]
                if not pos:
                    continue
                base = [(canon[i], stressed[i]) for i in pos]
                vs = [base]
                for v in cmu.get(re.sub(r"[^a-z']", "", words[wi].lower()), []):
                    v = [(strip(x), (None if not re.search(r"\d", x) else re.search(r"\d", x).group(0) != "0"))
                         for x in v]
                    if (len(v) == len(base) and [p for p, _ in v] != [p for p, _ in base]
                            and all((p in VOWELS) == (q in VOWELS) for (p, _), (q, _) in zip(v, base))
                            and v not in vs):
                        vs.append(v)
                groups.append((pos, vs))
            pick = choose_variants(post, [[[p for p, _ in v] for v in g[1]] for g in groups], to_ids)
            for (pos, vs), k in zip(groups, pick):
                if k:
                    n_var += 1
                for i, (p, st) in zip(pos, vs[k]):
                    eff[i] = p
                    if k:
                        eff_st[i] = st
        stressed = eff_st
        try:
            G, bsub, spans = gop_features(post, to_ids(eff), alt)
        except AlignError:
            n_skip += 1                                   # 녹음이 문장보다 짧다 — 서버라면 400
            continue
        u = len(kept)
        kept.append(c["utt"])
        sites, _ = align(fold(eff, kind), fold(c["hyp"], kind))
        span_s += [a for a, _ in spans]
        span_e += [b for _, b in spans]
        best_sub += bsub.tolist()
        deleted += [s["actual"] is None for s in sites]
        n_frames.append(post.shape[0])
        mism = [0.0 if (s["actual"] is not None and same_sound(fold([eff[i]], kind)[0], s["actual"], stressed[i]))
                else 1.0 for i, s in enumerate(sites)]
        Xs.append(phone_table(G, eff, stressed, word_of, words, mism))
        acc += a_
        err += [x < 1.0 for x in a_]
        utt_i += [u] * len(canon)
        word_i += word_of
    print("[%s] 발화 %d(정렬 불가로 뺌 %d) · 음소 %d · 발음 후보 바꾼 단어 %d"
          % (ds, len(kept), n_skip, len(acc), n_var), flush=True)
    return {"X": np.concatenate(Xs), "acc": np.array(acc, np.float32), "err": np.array(err, bool),
            "utt": np.array(utt_i), "word": np.array([-1 if w is None else w for w in word_i]),
            "utts": np.array(kept), "span_s": np.array(span_s), "span_e": np.array(span_e),
            "best_sub": np.array(best_sub), "deleted": np.array(deleted, bool), "n_frames": np.array(n_frames)}


def pcc(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="mdd_en.models 의 이름(lv60-espeak · en037-v1 …)")
    ap.add_argument("--bakeoff", default=os.path.join(str(MDD_OUTPUTS), "english", "bakeoff"))
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "english", "scorer"))
    ap.add_argument("--so762", default=os.path.join(str(DATA_ROOT), "english", "speechocean762"))
    ap.add_argument("--l2arctic", default=os.path.join(str(DATA_ROOT), "english", "l2arctic_v5"))
    ap.add_argument("--no-variants", dest="variants", action="store_false")
    ap.add_argument("--reuse", action="store_true", help="feats_*.npz 가 있으면 다시 계산하지 않는다")
    a = ap.parse_args()
    import joblib
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from mdd_en.l2arctic import load_speakers
    from mdd_en.scorer import COLS
    from mdd_en.speechocean import load_split
    from mdd_en.models import MODELS
    kind = MODELS[a.model]["kind"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join(a.out, a.model + ("" if a.variants else "_novar"))
    os.makedirs(out, exist_ok=True)
    cmu = None
    if a.variants:
        from mdd_en.g2p import EnglishG2P
        cmu = EnglishG2P().g.cmu

    D = {}
    for ds in ("so762train", "so762", "l2arctic"):
        fp = os.path.join(out, "feats_%s.npz" % ds)
        if a.reuse and os.path.exists(fp):
            D[ds] = dict(np.load(fp))
            continue
        cache = pickle.load(open(os.path.join(a.bakeoff, a.model, ds + ".pkl"), "rb"))
        if ds == "l2arctic":
            meta = dict((r["utt"], r) for r in load_speakers(a.l2arctic))
        else:
            meta = dict((r["utt"], r) for r in load_split(a.so762, "train" if ds == "so762train" else "test"))
        t0 = time.time()
        D[ds] = build(ds, cache, meta, kind, cmu, a.variants, device)
        print("   %.0f초" % (time.time() - t0), flush=True)
        np.savez_compressed(fp, **D[ds])

    tr, te, l2 = D["so762train"], D["so762"], D["l2arctic"]
    pi = COLS.index("phone")
    hgb = dict(max_iter=300, learning_rate=0.05, categorical_features=[pi], random_state=0)
    onehot = lambda X: np.concatenate([np.delete(X, pi, 1), np.eye(40)[np.clip(X[:, pi].astype(int), 0, 39)]], 1)
    # 점수(0~100, 높을수록 잘 발음) 를 내는 방법들. 기본은 reg — 사람 음소 점수(0~2)를 바로 회귀(GOPT 와 같은 학습 목표).
    # 오류 확률 분류기는 비교용: 라벨 문턱(<1.0 · <1.5)에 따라 한국인 화자 판별력이 크게 달라졌다(결과 문서 참고).
    def fit(name):
        if name == "reg":
            m = HistGradientBoostingRegressor(**hgb).fit(tr["X"], tr["acc"])
            return lambda X: np.clip(m.predict(X), 0, 2) * 50.0, m
        if name == "logistic1.0":
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(onehot(tr["X"]), tr["err"])
            return lambda X: (1 - m.predict_proba(onehot(X))[:, 1]) * 100.0, m
        thr = float(name[3:])
        m = HistGradientBoostingClassifier(**hgb).fit(tr["X"], tr["acc"] < thr)
        return lambda X: (1 - m.predict_proba(X)[:, 1]) * 100.0, m

    so_te = dict((r["utt"], r) for r in load_split(a.so762, "test"))
    wk = te["utt"] * 1000 + te["word"]
    keys, inv = np.unique(wk, return_inverse=True)
    wacc = np.array([so_te[te["utts"][k // 1000]]["words"][k % 1000]["accuracy"] for k in keys])
    rep = {"model": a.model, "variants": a.variants, "n_train_phones": int(len(tr["acc"])),
           "label_note": "so762 오류 = 5명 평균 < 1.0. l2arctic 오류 = 사람 주석 대치·삭제(억양 수준 대치도 포함)"}
    keep = {}
    for name in ("reg", "cls1.0", "cls1.5", "logistic1.0"):
        score, m = fit(name)
        s_te, s_l2 = score(te["X"]), score(l2["X"])
        wscore = np.bincount(inv, s_te) / np.bincount(inv)
        rep[name] = {"so762_phone_pcc": round(pcc(s_te, te["acc"]), 4),
                     "so762_phone_auc": round(float(roc_auc_score(te["err"], -s_te)), 4),
                     "so762_word_pcc": round(pcc(wscore, wacc), 4),
                     "l2arctic_auc": round(float(roc_auc_score(l2["err"], -s_l2)), 4)}
        keep[name] = (score, m)
        print("[%s] %s" % (name, json.dumps(rep[name], ensure_ascii=False)), flush=True)
    k = COLS.index("p_af")
    rep["p_af_only"] = {"so762_phone_pcc": round(pcc(te["X"][:, k], te["acc"]), 4),
                        "l2arctic_auc": round(float(roc_auc_score(l2["err"], -l2["X"][:, k])), 4)}
    print("[p_af only] %s" % rep["p_af_only"], flush=True)

    # 지적 규칙 — 문턱은 so762 **train** 에서만 정한다(맞게 발음한 자리 acc=2.0 의 하위 q 분위).
    # train 점수는 **화자 단위 5겹 교차 예측**으로 낸다 — 학습에 쓴 자리의 점수는 실제보다 좋게 나와서,
    # 그대로 분위를 잡으면 test 의 실제 오지적이 목표(3%)보다 40% 가량 많았다(리뷰 2026-10-04).
    # 서버는 '점수 < T' 만 안다 → 저장하는 규칙도 그 꼴만. 받아쓰기 불일치를 섞은 규칙은 비교로만 남긴다.
    from sklearn.model_selection import GroupKFold
    score, m = keep["reg"]
    so_tr = dict((r["utt"], r) for r in load_split(a.so762, "train"))
    spk = np.array([so_tr[tr["utts"][u]]["spk"] for u in tr["utt"]])
    s_tr = np.zeros(len(tr["acc"]), np.float32)
    for i_fit, i_out in GroupKFold(5).split(tr["X"], groups=spk):
        mf = HistGradientBoostingRegressor(**hgb).fit(tr["X"][i_fit], tr["acc"][i_fit])
        s_tr[i_out] = np.clip(mf.predict(tr["X"][i_out]), 0, 2) * 50.0
    s_te, s_l2 = score(te["X"]), score(l2["X"])
    mi = COLS.index("mismatch")
    good = tr["acc"] >= 2.0
    pol = {}
    for q in (0.03, 0.05, 0.10, 0.20):
        T = float(np.quantile(s_tr[good], q))
        pol["score<T(frr%d%%)" % (q * 100)] = (T, q, lambda s, X, T=T: s < T)
        pol["mismatch&score<T(frr%d%%)" % (q * 100)] = (T, q, lambda s, X, T=T: (X[:, mi] > 0) & (s < T))
    pol["mismatch"] = (None, None, lambda s, X: X[:, mi] > 0)
    pol["mismatch&score<60"] = (60.0, None, lambda s, X: (X[:, mi] > 0) & (s < 60))
    rep["flag_policies"] = {}
    for name, (T, q, f) in pol.items():
        row = {"T": None if T is None else round(T, 2)}
        for dsn, s, X, y in (("so762train_oof", s_tr, tr["X"], tr["acc"] < 1.5), ("so762", s_te, te["X"], te["acc"] < 1.5),
                             ("l2arctic", s_l2, l2["X"], l2["err"])):
            pr = f(s, X)
            tp = int((pr & y).sum()); P = tp / max(pr.sum(), 1); R = tp / max(y.sum(), 1)
            row[dsn] = {"precision": round(P, 3), "recall": round(R, 3), "f1": round(2 * P * R / max(P + R, 1e-9), 3),
                        "frr": round(float((pr & ~y).sum() / max((~y).sum(), 1)), 3)}
        row["so762_frr_on_acc2"] = round(float(f(s_te, te["X"])[te["acc"] >= 2.0].mean()), 4)
        rep["flag_policies"][name] = row
        print("  [지적] %-28s T=%-6s so762 %s · acc2 오지적 %.3f | L2 %s" % (name, row["T"], row["so762"],
              row["so762_frr_on_acc2"], row["l2arctic"]), flush=True)
    # 기본 지적 규칙: '점수 < T' 중 train(교차 예측) F1(라벨 acc<1.5) 이 가장 높은 것
    cand = [k for k in rep["flag_policies"] if k.startswith("score<T")]
    best = max(cand, key=lambda k: rep["flag_policies"][k]["so762train_oof"]["f1"])
    rep["flag_rule"] = {"name": best, "T": rep["flag_policies"][best]["T"],
                        "frr_target": "%d%%" % round(pol[best][1] * 100)}
    print("[지적 규칙] %s" % rep["flag_rule"], flush=True)
    joblib.dump({"reg": m, "cols": COLS, "model": a.model, "flag_rule": rep["flag_rule"]},
                os.path.join(out, "phone_scorer.joblib"))
    json.dump(rep, open(os.path.join(out, "phone_scorer.json"), "w"), ensure_ascii=False, indent=1)
    print("[saved] %s" % out)


if __name__ == "__main__":
    main()
