# -*- coding: utf-8 -*-
"""
s0_symbol_map.py
~~~~~~~~~~~~~~~~
pronunciation_lab 의 `phones` 티어(호환자모, U+3131~U+3163)를
KoG2P 음소기호로 변환하는 매핑표.  (작업지시서 5절-1 / 3-B-4)

- 자음은 초성/종성 위치에 따라 기호가 다르다(ㄱ: 초성 k0 / 종성 kf).
  pronunciation_lab phones 티어는 자모를 순서대로 나열하므로,
  '자음-모음-자음...' 패턴으로 초/중/종성 위치를 추론해 매핑한다.
- UI 역매핑(기호->한글)도 함께 제공한다.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]


# 호환자모 초성 자음 -> KoG2P onset
ONSET_MAP = {
    'ㄱ': 'k0', 'ㄲ': 'kk', 'ㄴ': 'nn', 'ㄷ': 't0', 'ㄸ': 'tt',
    'ㄹ': 'rr', 'ㅁ': 'mm', 'ㅂ': 'p0', 'ㅃ': 'pp', 'ㅅ': 's0',
    'ㅆ': 'ss', 'ㅇ': 'oh', 'ㅈ': 'c0', 'ㅉ': 'cc', 'ㅊ': 'ch',
    'ㅋ': 'kh', 'ㅌ': 'th', 'ㅍ': 'ph', 'ㅎ': 'h0',
}

# 호환자모 종성 자음 -> KoG2P coda
CODA_MAP = {
    'ㄱ': 'kf', 'ㄲ': 'kk', 'ㄳ': 'ks', 'ㄴ': 'nf', 'ㄵ': 'nc',
    'ㄶ': 'nh', 'ㄷ': 'tf', 'ㄹ': 'll', 'ㄺ': 'lk', 'ㄻ': 'lm',
    'ㄼ': 'lb', 'ㄽ': 'ls', 'ㄾ': 'lt', 'ㄿ': 'lp', 'ㅀ': 'lh',
    'ㅁ': 'mf', 'ㅂ': 'pf', 'ㅄ': 'ps', 'ㅅ': 's0', 'ㅆ': 'ss',
    'ㅇ': 'ng', 'ㅈ': 'c0', 'ㅊ': 'ch', 'ㅋ': 'kh', 'ㅌ': 'th',
    'ㅍ': 'ph', 'ㅎ': 'h0',
}

# 호환자모 모음 -> KoG2P nucleus
VOWEL_MAP = {
    'ㅏ': 'aa', 'ㅐ': 'qq', 'ㅑ': 'ya', 'ㅒ': 'yq', 'ㅓ': 'vv',
    'ㅔ': 'ee', 'ㅕ': 'yv', 'ㅖ': 'ye', 'ㅗ': 'oo', 'ㅘ': 'wa',
    'ㅙ': 'wq', 'ㅚ': 'wo', 'ㅛ': 'yo', 'ㅜ': 'uu', 'ㅝ': 'wv',
    'ㅞ': 'we', 'ㅟ': 'wi', 'ㅠ': 'yu', 'ㅡ': 'xx', 'ㅢ': 'xi',
    'ㅣ': 'ii',
}

VOWELS = set(VOWEL_MAP.keys())
CONSONANTS = set(ONSET_MAP.keys()) | set(CODA_MAP.keys())

# UI 역매핑: KoG2P 기호 -> 대표 한글 자모 (기호가 초/종성 겸용이면 대표값)
SYMBOL_TO_JAMO = {}
for j, s in VOWEL_MAP.items():
    SYMBOL_TO_JAMO.setdefault(s, j)
for j, s in ONSET_MAP.items():
    SYMBOL_TO_JAMO.setdefault(s, j)
for j, s in CODA_MAP.items():
    SYMBOL_TO_JAMO.setdefault(s, j)


def jamo_seq_to_kog2p(jamo_list):
    """호환자모 리스트(phones 티어)를 KoG2P 기호열로 변환.
    자음의 초/종성 위치는 앞뒤 문맥으로 추론:
      - 자음 다음에 모음이 오면 초성, 아니면(자음/끝) 종성.
    변환 불가 문자는 ('?', 원문) 로 표시해 손실을 드러낸다.
    return: list[(kog2p_symbol, source_jamo, role)]
    """
    out = []
    n = len(jamo_list)
    for i, ch in enumerate(jamo_list):
        nxt = jamo_list[i + 1] if i + 1 < n else None
        if ch in VOWELS:
            out.append((VOWEL_MAP[ch], ch, 'nucleus'))
        elif ch in CONSONANTS:
            is_onset = (nxt in VOWELS)
            if is_onset and ch in ONSET_MAP:
                out.append((ONSET_MAP[ch], ch, 'onset'))
            elif ch in CODA_MAP:
                out.append((CODA_MAP[ch], ch, 'coda'))
            elif ch in ONSET_MAP:
                out.append((ONSET_MAP[ch], ch, 'onset'))
            else:
                out.append(('?', ch, 'unknown'))
        else:
            out.append(('?', ch, 'unknown'))
    return out


if __name__ == "__main__":
    print("[symbol map] onset:", len(ONSET_MAP), "coda:", len(CODA_MAP), "vowel:", len(VOWEL_MAP))
    demo = list("ㅎㅐㄱㅅㅣㅁ")  # 핵심 -> h0 qq kf ss ii mf 기대
    print("demo jamo:", demo)
    for sym, src, role in jamo_seq_to_kog2p(demo):
        print(f"  {src} -> {sym} ({role})")
