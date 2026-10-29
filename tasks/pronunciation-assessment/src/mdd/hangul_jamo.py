# -*- coding: utf-8 -*-
"""한글 ↔ 발음 자모열 — TAG · 표준발음(KoG2P) 자모열과 같은 표기 관례로.

사람 전사자는 들린 소리를 **한글로** 적는다(자모를 하나씩 치는 것보다 훨씬 빠르다). 그 한글을 모델·TAG 와
같은 자모열로 바꾸고(`decompose`), 반대로 자모열을 읽기 쉬운 한글로 보여준다(`compose`).

표기 관례(`deploy/pronunciation-demo/app.py` 의 canonical 과 같다):
- 첫소리 ㅇ 은 소리가 없어 자모열에 넣지 않는다 — "아" → ㅏ, "영어" → ㅕㅇㅓ (받침 ㅇ 은 넣는다)
- ㅙ → ㅗㅐ, ㅞ → ㅜㅔ 로 나눈다(TAG/LAB 관례). 다른 겹모음(ㅘ ㅝ ㅚ ㅟ ㅢ)은 한 자모로 둔다
- 전사에서 낱자모(ㄱ, ㅏ …)를 직접 쳐도 그대로 받는다 — 한글 음절로 못 적는 소리용
- 음절 뒤 `?` 는 "확신 없음" 표시 — 그 음절의 자모 번호를 따로 돌려준다
"""

CHO = u"ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = u"ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = [u""] + list(u"ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
SPLIT_V = {u"ㅙ": [u"ㅗ", u"ㅐ"], u"ㅞ": [u"ㅜ", u"ㅔ"]}
MERGE_V = {(u"ㅗ", u"ㅐ"): u"ㅙ", (u"ㅜ", u"ㅔ"): u"ㅞ"}


def is_syllable(ch):
    return u"가" <= ch <= u"힣"


def decompose(text):
    """한글 전사 → (자모 목록, 확신 없음 표시된 자모 번호 집합)."""
    out, unsure = [], set()
    chars = list(text or u"")
    for k, ch in enumerate(chars):
        start = len(out)
        if is_syllable(ch):
            o = ord(ch) - 0xAC00
            cho, jung, jong = CHO[o // 588], JUNG[(o % 588) // 28], JONG[o % 28]
            if cho != u"ㅇ":
                out.append(cho)
            out += SPLIT_V.get(jung, [jung])
            if jong:
                out.append(jong)
        elif ch in CHO or ch in JUNG or ch in JONG[1:]:
            out += SPLIT_V.get(ch, [ch])
        else:
            continue
        if k + 1 < len(chars) and chars[k + 1] == u"?":
            unsure.update(range(start, len(out)))
    return out, unsure


def compose(jamo):
    """발음 자모열 → 한글(보여주기용). 첫소리 ㅇ 이 없는 관례라 ㅇ 은 늘 받침으로 본다."""
    a, i = [], 0
    while i < len(jamo):
        pair = (jamo[i], jamo[i + 1]) if i + 1 < len(jamo) else None
        if pair in MERGE_V:
            a.append(MERGE_V[pair]); i += 2
        else:
            a.append(jamo[i]); i += 1
    isv = lambda c: c is not None and c in JUNG
    at = lambda k: a[k] if k < len(a) else None
    out, i = [], 0
    while i < len(a):
        if a[i] in CHO and a[i] != u"ㅇ" and isv(at(i + 1)):
            on, vi = CHO.index(a[i]), i + 1
        elif isv(a[i]):
            on, vi = CHO.index(u"ㅇ"), i
        else:
            out.append(a[i]); i += 1
            continue
        k, t = vi + 1, 0
        c = at(k)
        if c is not None and c in JONG[1:] and (c == u"ㅇ" or not isv(at(k + 1))):
            t = JONG.index(c); k += 1
        out.append(chr(0xAC00 + (on * 21 + JUNG.index(a[vi])) * 28 + t))
        i = k
    return u"".join(out)
