# -*- coding: utf-8 -*-
"""유창성을 **속도·쉼만** 보는 규칙으로 계산하기 위한 기준 구간을 뽑는다.

왜 필요한가 — 사람이 매긴 세 점수는 서로 상관 **0.84**(화자 단위 0.90) 다.
발음 4.5점 이상 발화에서 유창성 3점 미만은 **0.0%**, 발음 2.0 이하에서 유창성 4점 이상은 3.5% 뿐이다.
즉 AIHub 채점 기준에서는 **발음이 틀리면 유창성도 낮게 매긴다.** 그 라벨을 예측하는 회귀기는
「또박또박 읽었지만 발음을 틀린 녹음」에 낮은 유창성을 준다(특징에서 발음 관련 항목을 전부 빼도
예측 평균 2.88 → 2.80, 거의 그대로다). 모델 결함이 아니라 라벨 정의를 따른 결과다.

외부 API(SpeechSuper)는 유창성을 **속도·끊김**으로만 매겨 같은 녹음에 92~98점을 준다.
칸마다 다른 정보를 보여주려면 우리도 그 정의로 따로 계산해야 한다 → 이 스크립트가 그 기준을 만든다.

기준 = 사람이 유창성 4.5점 이상을 준 발화들의 조음속도·쉼 분포.
점수 = 100 − 벌점(느리면 · 쉼이 길고 잦으면). **빠른 것은 벌하지 않는다**(빠르다고 안 유창한 게 아니다).

    $PY .../fit_fluency_rule.py --train $MDD_OUTPUTS/score_v2_w2vbert/gop_feats_v2_train.jsonl \\
        --test $MDD_OUTPUTS/score_v2_w2vbert/gop_feats_v2_test.jsonl \\
        --out $MDD_OUTPUTS/score_phoneme/fluency_rule.json
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, collections, json, os
import numpy as np
from mdd.fluency import REF_FLU, RULES, band_of as _band_of, fluency_rule   # 점수 식은 서비스와 한 곳에서


def band_of(rows):
    return _band_of(rows, lambda v, q: float(np.percentile(np.asarray(v, dtype=float), q)))


def pcc(a, b):
    return float(np.corrcoef(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=os.path.join(str(MDD_OUTPUTS), "score_v2_w2vbert", "gop_feats_v2_train.jsonl"))
    ap.add_argument("--test", default=os.path.join(str(MDD_OUTPUTS), "score_v2_w2vbert", "gop_feats_v2_test.jsonl"))
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "score_phoneme", "fluency_rule.json"))
    ap.add_argument("--min-utts", dest="min_utts", type=int, default=3)
    a = ap.parse_args()

    load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
    tr, te = load(a.train), load(a.test)
    ref = [r for r in tr if r.get("flu") and r["flu"] >= REF_FLU]
    band = band_of(ref)
    print("[기준] 유창성 %.1f점 이상 %d발화" % (REF_FLU, len(ref)))
    for k, _, _ in RULES:
        print("   %-11s p10 %.3f · 중위 %.3f · p90 %.3f" % (k, band[k]["p10"], band[k]["med"], band[k]["p90"]))

    b = [r for r in te if r.get("flu") and r["flu"] > 0]
    s = [fluency_rule(r, band) for r in b]
    y = [r["flu"] for r in b]
    ag, al = collections.defaultdict(list), collections.defaultdict(list)
    for r, q in zip(b, s):
        ag[r["stem"][:5]].append(q); al[r["stem"][:5]].append(r["flu"])
    ks = [k for k in ag if len(ag[k]) >= a.min_utts]
    u = pcc(s, y); sp = pcc([np.mean(ag[k]) for k in ks], [np.mean(al[k]) for k in ks])
    pr = np.array([r.get("pron") or 0 for r in b], dtype=float)
    arr = np.array(s)
    lo = float(arr[pr <= 2.0].mean()) if (pr <= 2.0).any() else None
    hi = float(arr[pr >= 4.5].mean()) if (pr >= 4.5).any() else None
    print("[검증] 사람 유창성 라벨 상관 — 발화 %.3f · 화자 %.3f (%d명)" % (u, sp, len(ks)))
    print("       발음 2.0 이하 발화 평균 %.1f · 발음 4.5+ 평균 %.1f" % (lo or -1, hi or -1))

    json.dump({"ref_flu": REF_FLU, "rules": [[k, w, d] for k, w, d in RULES], "band": band,
               "n_ref": len(ref), "pcc_utt": round(u, 4), "pcc_spk": round(sp, 4),
               "mean_pron_low": round(lo, 1) if lo is not None else None,
               "mean_pron_high": round(hi, 1) if hi is not None else None,
               "note": "속도·쉼만 보는 유창성. 사람 라벨 예측(상관 0.610/0.762)보다 상관은 낮지만 발음과 독립"},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[saved] %s" % a.out)


if __name__ == "__main__":
    main()
