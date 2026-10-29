# -*- coding: utf-8 -*-
"""표준발음 음소열 ↔ 인식한 음소열 정렬(편집거리). 음소마다 글자 하나로 바꿔 edlib 로 정렬한다."""

import edlib


def _codes(*seqs):
    sym = {}
    for s in seqs:
        for p in s:
            if p not in sym:
                sym[p] = chr(0x4E00 + len(sym))          # '-'(틈) 와 겹치지 않는 글자
    return sym


def align(canon, hyp):
    """→ (sites, dist). sites[i] = {"actual": 인식한 음소 또는 None(빠뜨림), "inserted": [덧붙은 음소]}.

    덧붙은 음소는 바로 **앞** 표준 자리에 붙인다(문장 맨 앞이면 첫 자리).
    """
    sites = [{"actual": None, "inserted": []} for _ in canon]
    if not canon:
        return sites, len(hyp)
    if not hyp:
        return sites, len(canon)
    sym = _codes(canon, hyp)
    a, b = "".join(sym[p] for p in canon), "".join(sym[p] for p in hyp)
    r = edlib.align(b, a, task="path")
    nice = edlib.getNiceAlignment(r, b, a)
    ci, hi = -1, 0
    for q, t in zip(nice["query_aligned"], nice["target_aligned"]):
        if t != "-":
            ci += 1
            if q != "-":
                sites[ci]["actual"] = hyp[hi]
                hi += 1
        else:                                             # 표준에 없는 소리
            sites[max(ci, 0)]["inserted"].append(hyp[hi])
            hi += 1
    return sites, r["editDistance"]
