# -*- coding: utf-8 -*-
"""자리 단위 오발음 **분류기 탐침** — 인코더를 다시 학습하지 않고 얼마나 올라가나?

왜: 현재 지적은 「전사가 표준발음과 다르면 오류」다(MDD F1 0.4964). margin GOP 한 개로 임계값만
잡아도 F1 0.486 이 나온다 — 두 신호가 비슷한 수준이다. 그러면 **구조를 바꿔야 하는지, 아니면
후처리·보정만으로 여유가 남았는지**를 먼저 알아야 한다. 이 스크립트는 캐시된 특징
(margin GOP · GOP · 자리 문맥 · 발화 수준 41종)만으로 분류기를 학습해 그 상한을 잰다.
GPU 없이 수 분. **모델 파일을 배포하지 않는다 — 판단용 수치만 낸다.**

정답은 TAG(사람이 들은 실제 발음)와 canonical 의 정렬 결과다. 화자를 나눠 학습/평가한다.

    $PY .../probe_site_classifier.py --gop $MDD_OUTPUTS/score_phoneme/gop_margin_test.jsonl \\
        --manifest $MDD_OUTPUTS/manifest_tag_rebuilt --split test \\
        --out $MDD_OUTPUTS/analysis/mdd_error_2026-09-27/site_classifier
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, json, os
import numpy as np
import edlib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

VOWELS = set(u"ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
UTT_FEATS = ["gop", "gop_p10", "gop_p25", "gop_med", "gop_min", "gop_std", "gop_bad", "neg_nll", "neg_per",
             "dur", "n_canon", "rate", "speech_rate", "artic_rate", "blank_frac", "lead_sil", "trail_sil",
             "speech_span", "n_pause", "pause_frac", "max_pause"]


def proj(canon, seq):
    c, s = u"".join(canon), u"".join(seq)
    if not s:
        return [None] * len(canon)
    a = edlib.getNiceAlignment(edlib.align(c, s, task="path"), c, s)
    out = []
    for qc, ts in zip(a["query_aligned"], a["target_aligned"]):
        if qc == "-":
            continue
        out.append(None if ts == "-" else ts)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gop", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True, help="산출 접두어")
    ap.add_argument("--train-frac", dest="train_frac", type=float, default=0.7, help="화자 기준 학습 비율")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tsv = open(os.path.join(a.manifest, "%s.tsv" % a.split), encoding="utf-8").read().splitlines()[1:]
    ltr = open(os.path.join(a.manifest, "%s.ltr" % a.split), encoding="utf-8").read().splitlines()
    TAG = {}
    for i, row in enumerate(tsv):
        TAG[os.path.basename(row.split("\t")[0])[:-4]] = ltr[i].split()

    jamo = sorted(set(u"ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ") | VOWELS |
                  set(u"ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ"))
    jid = dict((c, i + 1) for i, c in enumerate(jamo))
    X, y, spk, base_pred = [], [], [], []
    rng = np.random.RandomState(a.seed)
    n_utt = 0
    with open(a.gop, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            canon, gm, gp = d.get("canon"), d.get("gops_m"), d.get("gops")
            tag = TAG.get(d["stem"])
            if not canon or not gm or tag is None or len(gm) != len(canon):
                continue
            canon = list(canon)
            ptag = proj(canon, tag)
            ppred = proj(canon, list(d.get("pred") or ""))
            if len(ptag) != len(canon) or len(ppred) != len(canon):
                continue
            n_utt += 1
            uf = [float(d.get(k) or 0.0) for k in UTT_FEATS]
            for i, c in enumerate(canon):
                pos = 1 if c in VOWELS else (3 if (c == u"ㅇ" or not (i + 1 < len(canon) and canon[i + 1] in VOWELS)) else 2)
                X.append([float(gm[i]), float(gp[i] if gp else 0.0), jid.get(c, 0), pos, i, len(canon),
                          jid.get(canon[i - 1], 0) if i else 0,
                          jid.get(canon[i + 1], 0) if i + 1 < len(canon) else 0] + uf)
                y.append(1 if ptag[i] != c else 0)
                spk.append(d["stem"][:5])
                base_pred.append(1 if ppred[i] != c else 0)
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.int8)
    base_pred = np.asarray(base_pred, dtype=np.int8); spk = np.asarray(spk)
    uspk = np.unique(spk); rng.shuffle(uspk)
    tr_spk = set(uspk[:int(len(uspk) * a.train_frac)])
    m_tr = np.array([s in tr_spk for s in spk])
    print("[data] 발화 %d · 자리 %d (오류 %.2f%%) · 화자 %d → 학습 %d / 평가 %d"
          % (n_utt, len(y), 100.0 * y.mean(), len(uspk), m_tr.sum(), (~m_tr).sum()), flush=True)

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         random_state=a.seed).fit(X[m_tr], y[m_tr])
    p = clf.predict_proba(X[~m_tr])[:, 1]
    yt, bt, mt = y[~m_tr], base_pred[~m_tr], X[~m_tr][:, 0]

    def prf(flag):
        tp = int((flag & (yt == 1)).sum()); fp = int((flag & (yt == 0)).sum()); fn = int(((~flag) & (yt == 1)).sum())
        pr = tp / float(tp + fp) if tp + fp else 0.0
        rc = tp / float(tp + fn) if tp + fn else 0.0
        return {"flagged": int(flag.sum()), "TP": tp, "FP": fp, "FN": fn, "precision": round(pr, 4),
                "recall": round(rc, 4), "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0}

    res = {"gop": os.path.abspath(a.gop), "manifest": os.path.abspath(a.manifest), "split": a.split,
           "sites_eval": int((~m_tr).sum()), "error_rate": round(float(yt.mean()), 4),
           "auc": round(float(roc_auc_score(yt, p)), 4), "ap": round(float(average_precision_score(yt, p)), 4),
           "baselines": {"transcription_mismatch": prf(bt == 1),
                         "margin<1": prf(mt < 1.0), "margin<0": prf(mt < 0.0)},
           "classifier": {}, "combined": {},
           "note": "인코더 재학습 없이 캐시 특징만으로 낸 상한 추정. 화자 분리 평가"}
    # 전사 기준과 같은 recall 에서의 정밀도, 그리고 F1 최대점
    order = np.argsort(-p)
    want = int((bt == 1).sum())
    thr_same_n = p[order[min(want, len(order) - 1)]]
    res["classifier"]["same_flag_count"] = dict(prf(p >= thr_same_n), threshold=round(float(thr_same_n), 4))
    best = None
    for thr in np.arange(0.05, 0.96, 0.05):
        r = prf(p >= thr)
        if best is None or r["f1"] > best[1]["f1"]:
            best = (float(thr), r)
    res["classifier"]["best_f1"] = dict(best[1], threshold=round(best[0], 3))
    res["combined"]["transcription_and_p>=0.5"] = prf((bt == 1) & (p >= 0.5))
    res["combined"]["transcription_or_p>=0.8"] = prf((bt == 1) | (p >= 0.8))
    imp = None
    try:
        from sklearn.inspection import permutation_importance
        sub = np.random.RandomState(0).choice(np.arange((~m_tr).sum()), size=min(30000, (~m_tr).sum()), replace=False)
        pi = permutation_importance(clf, X[~m_tr][sub], yt[sub], n_repeats=3, random_state=0, scoring="average_precision")
        names = ["margin_gop", "gop", "phone", "position", "index", "n_canon", "prev_phone", "next_phone"] + UTT_FEATS
        imp = sorted(zip(names, [round(float(v), 4) for v in pi.importances_mean]), key=lambda kv: -kv[1])[:10]
    except Exception as e:
        imp = [("실패", str(e)[:60])]
    res["top_features_permutation_ap"] = imp
    json.dump(res, open(a.out + ".json", "w"), ensure_ascii=False, indent=1)
    print("[base] 전사기준 %s" % res["baselines"]["transcription_mismatch"])
    print("[clf ] 같은 지적 수: %s" % res["classifier"]["same_flag_count"])
    print("[clf ] F1 최대     : %s" % res["classifier"]["best_f1"])
    print("[comb] 전사 AND p>=.5: %s" % res["combined"]["transcription_and_p>=0.5"])
    print("[saved] %s.json" % (a.out))


if __name__ == "__main__":
    main()
