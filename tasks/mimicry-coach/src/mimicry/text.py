# -*- coding: utf-8 -*-
"""대사 텍스트 → 음절 목록 + 정렬용 자모 토큰.

정렬 모델(w2vbert_stage1)은 `KoG2P → 자모` 표기로 학습됐다
(`tasks/pronunciation-assessment/scripts/data/build_kspon_native_v1.py --method kog2p`).
그래서 정렬 입력도 같은 변환을 쓴다 — 음소→자모 표는 `mdd.jamo_to_kog2p.INV` 를 그대로 가져온다.
한국어 음운 규칙(연음·비음화·경음화…)은 음절 수를 바꾸지 않으므로 KoG2P 결과를 음절로 다시 묶으면
원문 음절과 1:1 로 맞는다. 안 맞거나 KoG2P 를 못 쓰면 음절을 직접 분해한다(초성 ㅇ 은 뺀다 — 학습 표기와 같다).
"""
import re

L_ = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
V_ = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
T_ = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ",
      "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
VOWELS = set(V_)
COMPOUND = {"ㄳ", "ㄵ", "ㄶ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅄ"}
# 학습 표기 관례 — ㅙ·ㅞ 는 두 자모로 적는다(build_kspon_native_v1.TAG_VOWEL_SPLIT 와 같다)
VOWEL_SPLIT = {"ㅙ": ["ㅗ", "ㅐ"], "ㅞ": ["ㅜ", "ㅔ"]}


def is_hangul(ch):
    return 0 <= ord(ch) - 0xAC00 < 11172


def has_batchim(ch):
    o = ord(ch) - 0xAC00
    return 0 <= o < 11172 and o % 28 != 0


def decompose(ch):
    o = ord(ch) - 0xAC00
    return L_[o // 588], V_[(o % 588) // 28], T_[o % 28]


def parse_line(text):
    """→ (어절 목록, 음절 목록, 음절별 어절 번호). 문장부호는 버리고, 숫자·영문은 받지 않는다."""
    if re.search(r"[0-9A-Za-z]", text):
        raise ValueError("숫자·영문은 한글로 적어 주세요: %r" % text)
    words = re.sub(r"[^가-힣\s]", " ", text).split()
    if not words:
        raise ValueError("한글 대사가 없습니다: %r" % text)
    syllables, word_of = [], []
    for wi, w in enumerate(words):
        for ch in w:
            syllables.append(ch)
            word_of.append(wi)
    return words, syllables, word_of


_G2P = {}


def _kog2p():
    if not _G2P:
        import sys
        from shared.paths import G2P_TOOLS, ensure_kog2p
        ensure_kog2p()
        argv, sys.argv = sys.argv, sys.argv[:1]          # g2p.py 가 import 시 argv 를 파싱한다
        try:
            from g2p import readRules, graph2prono, ver_info
        finally:
            sys.argv = argv
        from mdd.jamo_to_kog2p import INV                 # 정렬 모델 학습 표기와 같은 음소→자모 표
        _G2P.update(rules=readRules(ver_info[0], str(G2P_TOOLS / "KoG2P" / "rulebook.txt")),
                    g2p=graph2prono, inv=INV)
    return _G2P


def syllabify(toks):
    """자모 토큰열 → 음절별 토큰 묶음(실패하면 None).

    규칙은 `mdd.jamo_to_kog2p.jamo_to_phones` 와 같다: 모음 앞 자음은 초성, 아니면 종성.
    ㅇ·겹받침은 위치와 무관하게 종성(초성 ㅇ 은 표기에 없다)."""
    out, cur = [], None
    n = len(toks)
    for i, t in enumerate(toks):
        if t in VOWELS:
            if cur is not None and cur[1] is None:
                cur[1] = t
            else:
                cur = [None, t, []]
                out.append(cur)
        elif i + 1 < n and toks[i + 1] in VOWELS and t != "ㅇ" and t not in COMPOUND:
            cur = [t, None, []]
            out.append(cur)
        elif cur is None:
            return None
        else:
            cur[2].append(t)
    if any(c[1] is None for c in out):
        return None
    return [([c[0]] if c[0] else []) + [c[1]] + c[2] for c in out]


def direct_tokens(ch):
    l, v, t = decompose(ch)
    return ([] if l == "ㅇ" else [l]) + [v] + ([t] if t else [])


def _split_vowels(syl_toks):
    return [[x for t in s for x in VOWEL_SPLIT.get(t, [t])] for s in syl_toks]


def align_tokens(words, syllables):
    """→ (음절별 정렬 토큰 목록, 'kog2p' | 'direct:<사유>')."""
    try:
        g = _kog2p()
        text = re.sub(r"\s+", " ", " ".join(words)).strip()   # KoG2P 는 공백 2칸에서 음절을 깨뜨린다
        phs = [p for p in g["g2p"](text, *g["rules"]).split() if p in g["inv"]]
        syl = syllabify([g["inv"][p] for p in phs])
        if syl is not None and len(syl) == len(syllables):
            return _split_vowels(syl), "kog2p"
        why = "음절 수 불일치(%s≠%d)" % (None if syl is None else len(syl), len(syllables))
    except Exception as e:                                     # KoG2P 미설치·경로 미설정 등
        why = "%s: %s" % (type(e).__name__, e)
    return _split_vowels([direct_tokens(ch) for ch in syllables]), "direct:" + why
