# -*- coding: utf-8 -*-
"""한국인 대표 오류 자리의 의사라벨 — "표준대로 냈나, 한국인식으로 냈나"를 CTC 우도로 비교해 확실한 것만 고른다.

받아쓰기(greedy)는 문맥으로 단어를 짐작해 사전 발음을 적는 버릇이 있어서, 자리마다 **두 후보만 놓고**
우도를 비교한다(나머지 소리는 표준발음으로 고정):
    log P(… s …) − log P(… c …)  >  margin   →  한국인식 s 로 냈다
    log P(… c …) − log P(… s …)  >  margin   →  표준대로 c
    그 사이                                  →  모름(이 자리가 있는 발화는 학습에서 뺀다)
단어 끝 모음 삽입(book → 부커)·끝소리 탈락도 같은 방식(삽입/탈락한 열 vs 표준 열).
"""

import numpy as np

from mdd_en.gop import _ctc_loglik
from mdd_en.l1_ko_errors import RULES, _ok, strip


def candidates(words):
    """단어 목록(ARPAbet+강세) → [(전체 자리 번호, 규칙 이름, 표준, 대안)] — 대안 None = 탈락, '+AH' = 뒤 삽입."""
    out, k = [], 0
    for w in words:
        ph = w["ph"]
        for i, p in enumerate(ph):
            b = strip(p)
            seen = set()
            for name, c, sub, cond, _ in RULES:
                if (c == b or (c == "*" and sub == "+AH0")) and _ok(cond, ph, i):
                    alt = "+AH" if sub == "+AH0" else sub
                    if alt in seen:
                        continue
                    seen.add(alt)
                    out.append((k + i, name, b, alt))
        k += len(ph)
    return out


def site_llrs(post, canon, cands, to_ids):
    """대안마다 LLR = log P(대안 열) − log P(표준 열). 정렬이 안 되면(녹음이 너무 짧음) None."""
    seqs = [to_ids(canon)]
    for pos, name, c, alt in cands:
        if alt is None:
            s = canon[:pos] + canon[pos + 1:]
        elif alt == "+AH":
            s = canon[: pos + 1] + ["AH"] + canon[pos + 1:]
        else:
            s = canon[:pos] + [alt] + canon[pos + 1:]
        seqs.append(to_ids(s))
    ll = _ctc_loglik(post.float(), seqs)
    if not np.isfinite(ll[0]) or ll[0] < -1e3:
        return None
    return [float(v - ll[0]) for v in ll[1:]]


def decide(post, canon, cands, to_ids, margin=1.0):
    """→ [(자리, 규칙, 표준, 대안, 판정)] 판정: 'alt'(한국인식) · 'canon'(표준) · None(모름).

    post: (T, K) torch 로그확률(blank=0) · canon: 강세 뗀 표준 음소열 · to_ids: 음소열 → id 열
    margin: 숫자 하나(모든 규칙: 사이 구간은 모름) 또는 {규칙 이름: 문턱}(규칙별 — 문턱보다 크면 한국인식,
            아니면 표준. 표에 없는 규칙은 모름)
    """
    llr = site_llrs(post, canon, cands, to_ids)
    if llr is None:
        return [(p, n, c, a, None) for p, n, c, a in cands]
    out = []
    for (pos, name, c, alt), d in zip(cands, llr):
        if isinstance(margin, dict):
            m = margin.get(name)
            out.append((pos, name, c, alt, None if m is None else ("alt" if d > m else "canon")))
        else:
            out.append((pos, name, c, alt, "alt" if d > margin else ("canon" if d < -margin else None)))
    return out


def apply(canon, decisions):
    """판정 중 'alt' 만 표준열에 반영 → 의사라벨 음소열. 한 자리에 두 대안이 'alt' 면 더 앞의 것 하나만."""
    sub, ins, dele = {}, set(), set()
    for pos, name, c, alt, d in decisions:
        if d != "alt" or pos in sub or pos in dele:
            continue
        if alt is None:
            dele.add(pos)
        elif alt == "+AH":
            ins.add(pos)
        else:
            sub[pos] = alt
    out = []
    for i, p in enumerate(canon):
        if i in dele:
            continue
        out.append(sub.get(i, p))
        if i in ins:
            out.append("AH")
    return out
