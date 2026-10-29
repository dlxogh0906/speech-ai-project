# -*- coding: utf-8 -*-
"""
s0_kog2p_wrapper.py
~~~~~~~~~~~~~~~~~~~~
KoG2P(scarletcho/KoG2P)를 import 가능한 모듈로 래핑한다.

- 한글 텍스트 -> KoG2P 음소기호 리스트 (예: '핵심' -> ['h0','qq','kf','ss','ii','mf'])
- KoG2P 표준 음운규칙(연음·격음화·경음화 등) 전체 적용 (작업지시서 3-A: 규칙 전체 적용)
- Py3.13 SyntaxWarning 억제 + import 시 argv 보호(g2p.py가 import 시 parse_args를 호출함)
- lru_cache 로 문장 단위 캐싱

사용:
    from s0_kog2p_wrapper import text_to_phones
    text_to_phones('오늘 아침에 몇 시에 일어났어요?')
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import os
import sys
import re
import warnings
from functools import lru_cache

from shared.paths import G2P_TOOLS

_KOG2P_DIR = str(G2P_TOOLS / "KoG2P")   # 옛 scripts/_g2p_tools 배치는 이 저장소에 없다 → .env 의 G2P_TOOLS
_RULEBOOK = os.path.join(_KOG2P_DIR, "rulebook.txt")

# --- KoG2P 안전 import (argv 보호 + 경고 억제) ---
_saved_argv = sys.argv
sys.argv = [sys.argv[0]]          # g2p.py 의 module-level parse_args() 보호
if _KOG2P_DIR not in sys.path:
    sys.path.insert(0, _KOG2P_DIR)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")           # invalid escape SyntaxWarning 억제
    import g2p as _kog2p                        # noqa: E402
sys.argv = _saved_argv

# 규칙 테이블은 1회만 로드 (rule_in, rule_out)
_RULE_IN, _RULE_OUT = _kog2p.readRules(3, _RULEBOOK)

# KoG2P 전체 음소 인벤토리 (graph2phone 기준)
ONSET = ['k0', 'kk', 'nn', 't0', 'tt', 'rr', 'mm', 'p0', 'pp',
         's0', 'ss', 'oh', 'c0', 'cc', 'ch', 'kh', 'th', 'ph', 'h0']
NUCLEUS = ['aa', 'qq', 'ya', 'yq', 'vv', 'ee', 'yv', 'ye', 'oo', 'wa',
           'wq', 'wo', 'yo', 'uu', 'wv', 'we', 'wi', 'yu', 'xx', 'xi', 'ii']
CODA = ['kf', 'ks', 'nf', 'nc', 'nh', 'tf', 'll', 'lk', 'lm', 'lb',
        'ls', 'lt', 'lp', 'lh', 'mf', 'pf', 'ps', 'ng']  # 실현 종성(빈값 제외, oh->ng)
# 전체 기호 집합(중복 제거) — 검증용
ALL_SYMBOLS = sorted(set(ONSET + NUCLEUS + CODA + ['kh', 'th', 'ph', 'ch', 'ss', 's0', 'c0', 'h0']))


@lru_cache(maxsize=200000)
def text_to_phones(text):
    """한글 텍스트 -> KoG2P 음소기호 tuple. 규칙 전체 적용. 실패 시 빈 tuple."""
    if not text or not text.strip():
        return tuple()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prono = _kog2p.graph2prono(text, _RULE_IN, _RULE_OUT)
    if not prono:
        return tuple()
    # '#'(어절경계) 및 공백 정리 -> 음소 토큰만
    toks = [t for t in re.split(r"[#\s]+", prono.strip()) if t]
    return tuple(toks)


def text_to_phones_by_word(text):
    """어절 경계(#)를 보존해 [[어절1 음소...], [어절2 음소...]] 로 반환.
    MDD 의 '어절 붕괴 필터'용."""
    if not text or not text.strip():
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prono = _kog2p.graph2prono(text, _RULE_IN, _RULE_OUT)
    if not prono:
        return []
    words = []
    for w in re.split(r"#", prono):
        toks = [t for t in w.strip().split() if t]
        if toks:
            words.append(toks)
    return words


if __name__ == "__main__":
    # 자가 검증
    samples = [
        "오늘 아침에 몇 시에 일어났어요?",
        "나는 집 내부 공사를 끝냈다.",
        "핵심",
        "집 내붕 뿟살이",   # orthographic 예시(들린 대로)
    ]
    print("[KoG2P wrapper self-test]")
    print("rulebook:", _RULEBOOK, "| rules:", len(_RULE_IN))
    print("symbol inventory size:", len(ALL_SYMBOLS))
    for s in samples:
        print(f"  {s!r}")
        print(f"    flat : {list(text_to_phones(s))}")
        print(f"    words: {text_to_phones_by_word(s)}")
