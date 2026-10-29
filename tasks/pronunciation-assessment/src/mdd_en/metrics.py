# -*- coding: utf-8 -*-
"""음소 단위 오류 판별 지표 — 사람이 적은 실제 발음(L2-ARCTIC 등)과 모델 받아쓰기를 표준 자리마다 대조한다.

    rows = site_eval(canon, human_actual, human_err, stressed, hyp, kind)
    prf(rows, 1)   # 엄격(기호가 같아야 맞음) · prf(rows, 2) 관대(미국 영어 변이 허용)

정밀도 = 모델이 틀렸다고 한 자리 중 사람도 틀렸다고 한 비율 · 재현율 = 사람 오류 중 모델이 잡은 비율
오지적(frr) = 사람이 맞다고 한 자리를 모델이 틀렸다고 한 비율. 정확 진단(cd) = 틀린 자리에서 들린 소리까지 맞힘.
"""

import numpy as np


def site_eval(canon, human_actual, human_err, stressed, hyp, kind):
    """표준 자리별 (사람 오류, 모델 오류[엄격], 모델 오류[관대], 정확 진단) 목록."""
    from mdd_en.align import align
    from mdd_en.phones import fold, same_sound
    c = fold(canon, kind)
    sites, _ = align(c, fold(hyp, kind))
    out = []
    for i, s in enumerate(sites):
        a = s["actual"]
        strict = not (a is not None and (a == c[i] or (c[i], a) in {("T", "DX"), ("D", "DX"), ("T", "Q")}))
        lenient = not (a is not None and same_sound(c[i], a, stressed[i]))
        ha = human_actual[i] if human_actual is not None else None
        cd = None
        if human_err[i] and strict and human_actual is not None:
            hf = fold([ha], kind)[0] if ha else None
            cd = (a is None and hf is None) or (a is not None and hf is not None and a == hf)
        out.append((bool(human_err[i]), strict, lenient, cd))
    return out


def prf(rows, k):
    h = np.array([r[0] for r in rows]); p = np.array([r[k] for r in rows])
    tr, fr, fa = int((h & p).sum()), int((~h & p).sum()), int((h & ~p).sum())
    ta = int((~h & ~p).sum())
    P = tr / max(tr + fr, 1); R = tr / max(tr + fa, 1)
    return {"precision": round(P, 4), "recall": round(R, 4), "f1": round(2 * P * R / max(P + R, 1e-9), 4),
            "frr": round(fr / max(fr + ta, 1), 4), "n_human_err": int(h.sum()), "n_sites": len(rows)}


# 한국인 대표 오류 짝(표준 → 한국인이 흔히 내는 소리). 음성학 문헌의 L1 전이 오류 목록 — 평가용 집계 기준.
KO_PAIRS = [("DH", "D"), ("Z", "S"), ("Z", "JH"), ("TH", "S"), ("TH", "T"), ("F", "P"), ("V", "B"), ("R", "L"),
            ("L", "R"), ("S", "SH"), ("D", "T"), ("B", "P"), ("G", "K"), ("IH", "IY"), ("IY", "IH"), ("EH", "AE"),
            ("AE", "EH"), ("OW", "AO"), ("UH", "UW"), ("ER", "AH")]


def pair_sites(canon, human_actual, hyp, kind):
    """표준 자리마다 (표준, 사람이 들은 소리, 모델이 적은 소리). 소리 없음은 None. fold 를 적용한 기호."""
    from mdd_en.align import align
    from mdd_en.phones import fold
    c = fold(canon, kind)
    sites, _ = align(c, fold(hyp, kind))
    h = [fold([x], kind)[0] if x else None for x in human_actual]
    return [(c[i], h[i], s["actual"]) for i, s in enumerate(sites)]


def pair_report(rows, pairs=KO_PAIRS):
    """한국인 대표 오류 짝마다: 사람이 그 오류를 들은 자리에서 모델도 그 소리로 적은 비율(재현) ·
    사람이 표준대로 들은 자리에서 모델이 그 오류 소리로 적은 비율(헛지적). 전체 합계도."""
    out, tot = {}, {"err": 0, "hit": 0, "ok": 0, "fa": 0}
    for c, s in pairs:
        err = [m for cc, h, m in rows if cc == c and h == s]
        ok = [m for cc, h, m in rows if cc == c and h == c]
        hit, fa = sum(m == s for m in err), sum(m == s for m in ok)
        out["%s>%s" % (c, s)] = {"n_err": len(err), "recall": round(hit / max(len(err), 1), 3),
                                 "n_ok": len(ok), "false_alarm": round(fa / max(len(ok), 1), 4)}
        tot["err"] += len(err); tot["hit"] += hit; tot["ok"] += len(ok); tot["fa"] += fa
    out["_all"] = {"n_err": tot["err"], "recall": round(tot["hit"] / max(tot["err"], 1), 3),
                   "n_ok": tot["ok"], "false_alarm": round(tot["fa"] / max(tot["ok"], 1), 4)}
    return out


def map_positions(ours, ref):
    """우리 표준 음소열 자리 → 기준(사람 주석) 표준 음소열 자리. 정렬해서 **같은 음소로 맞은 자리만** 잇는다
    (G2P 차이로 표준발음이 다른 자리는 비교에서 뺀다)."""
    from mdd_en.align import align
    sites, _ = align(ours, ref)
    out, j = {}, 0
    for i, s in enumerate(sites):
        if s["actual"] is not None:
            if s["actual"] == ours[i]:
                out[i] = j
            j += 1
        j += len(s["inserted"])
    return out
