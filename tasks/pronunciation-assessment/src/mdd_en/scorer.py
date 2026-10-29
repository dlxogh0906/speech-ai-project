# -*- coding: utf-8 -*-
"""음소 특징표 — GOP 특징에 자리 정보(모음·강세·단어 안 위치·기능어)를 붙인다. 학습(`en_fit_phone_scorer.py`)과
서버가 **같은 함수**로 만들어야 점수가 같다.

음소 점수 = (1 − 그 소리가 틀렸을 확률) × 100 — 한국어 엔진과 같은 정의.
"""

import re

import numpy as np

from mdd_en.gop import FEATS
from mdd_en.phones import ARPA39, VOWELS

# 이어 말하면 약하게 발음하는 게 정상인 단어(to → tə, and → ən …) — 약모음을 틀렸다고 하지 않게
FUNC_WORDS = set("""a an the to of and for in on at from with as but or nor that than then them his her he she we
you your i is are was were be been am has have had do does did can could would should will shall there some just us
him me my by so not its it our their they this these those what when who whom which if into onto up out over""".split())

COLS = list(FEATS) + ["mismatch", "is_vowel", "stressed", "w_init", "w_final", "w_nph", "func", "phone"]
PHONE_CODE = dict((p, i) for i, p in enumerate(ARPA39))


def phone_table(gop, canon, stressed, word_of, word_text, mismatch):
    """→ (N, len(COLS)) float32. 마지막 열 phone 은 음소 번호(범주형).

    gop: gop_features 의 (N, 7) · canon: 강세 뗀 ARPAbet · stressed: True/False/None · word_of: 음소 → 단어 번호
    word_text: 단어 목록 · mismatch: 인식 음소열이 이 자리에서 표준과 달랐는가(0/1)
    """
    N = len(canon)
    X = np.zeros((N, len(COLS)), np.float32)
    X[:, : len(FEATS)] = gop
    for i, p in enumerate(canon):
        w = word_of[i]
        same = [j for j in range(N) if word_of[j] == w]
        wt = re.sub(r"[^a-z']", "", (word_text[w] if w is not None and w < len(word_text) else "").lower())
        X[i, len(FEATS):] = [mismatch[i], p in VOWELS, 1.0 if stressed[i] else 0.0,
                             same and i == same[0], same and i == same[-1], len(same),
                             wt in FUNC_WORDS, PHONE_CODE.get(p, -1)]
    return X
