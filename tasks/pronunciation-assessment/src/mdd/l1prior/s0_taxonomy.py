# -*- coding: utf-8 -*-
"""
s0_taxonomy.py
~~~~~~~~~~~~~~
error_type taxonomy — pronunciation_lab 의 gold 태그(17종)를
KoG2P 음소 자질 편차 규칙으로 정의한다. (작업지시서 4절)

핵심: pronunciation_lab 이 정의한 유형 기준을, Speech_lab 의 MDD 대치/삭제/삽입
대치쌍에 자동 적용해 error_type 을 붙인다 -> 두 데이터가 같은 taxonomy 로 통일.

classify_op(op) -> (error_type, confidence, rationale)
  op = {"kind": "sub|del|ins", "target": <phone or None>, "actual": <phone or None>,
        "target_next": <phone or None>}   # 문맥(구개음화·연음 판정용)

주의: 강세/경계억양/반복/간투사 = 초분절·유창성 -> 음소 MDD로 판정 불가.
     이들은 error_type='(suprasegmental/fluency)'로 남기고 auto=False.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]


# ── 음소 자질 테이블 ──────────────────────────────────────────
PLAIN = {'k0', 't0', 'p0', 'c0', 's0'}          # 평음
TENSE = {'kk', 'tt', 'pp', 'cc', 'ss'}          # 경음(된소리)
ASPIR = {'kh', 'th', 'ph', 'ch'}                # 격음(유기음)
NASAL = {'nn', 'nf', 'mm', 'mf', 'ng'}          # 비음
LIQUID = {'rr', 'll'}                            # 유음
CODA = {'kf', 'ks', 'nf', 'nc', 'nh', 'tf', 'll', 'lk', 'lm', 'lb',
        'ls', 'lt', 'lp', 'lh', 'mf', 'pf', 'ps', 'ng'}
NUCLEUS = {'aa', 'qq', 'ya', 'yq', 'vv', 'ee', 'yv', 'ye', 'oo', 'wa',
           'wq', 'wo', 'yo', 'uu', 'wv', 'we', 'wi', 'yu', 'xx', 'xi', 'ii'}
VOWEL_I = {'ii', 'ya', 'yv', 'yo', 'yu', 'ye', 'yq', 'xi'}  # 전설/i계(구개음화 유발)

# 평/경/격 시리즈 상호 대응(같은 조음위치)
LARYNGEAL_SERIES = {
    'k': {'plain': 'k0', 'tense': 'kk', 'aspir': 'kh'},
    't': {'plain': 't0', 'tense': 'tt', 'aspir': 'th'},
    'p': {'plain': 'p0', 'tense': 'pp', 'aspir': 'ph'},
    'c': {'plain': 'c0', 'tense': 'cc', 'aspir': 'ch'},
    's': {'plain': 's0', 'tense': 'ss'},
}
_SYM2SERIES = {}
for k, v in LARYNGEAL_SERIES.items():
    for lar, sym in v.items():
        _SYM2SERIES[sym] = (k, lar)

# 폐쇄음 종성 -> 대응 비음(비음화 판정)
OBSTRUENT_CODA_TO_NASAL = {'kf': 'ng', 'tf': 'nf', 'pf': 'mf'}


def _laryngeal(sym):
    return _SYM2SERIES.get(sym, (None, None))


def is_nucleus(p):
    return p in NUCLEUS


def classify_op(op):
    """MDD 단일 연산(op)을 error_type 으로 자동 분류."""
    kind = op["kind"]
    t = op.get("target")
    a = op.get("actual")
    tn = op.get("target_next")

    # ── 삭제 ─────────────────────────────────────────
    if kind == "del":
        if t in NUCLEUS:
            return ("음운탈락", "high", f"모음 {t} 탈락")
        return ("음운탈락", "high", f"자음 {t} 탈락")

    # ── 삽입 ─────────────────────────────────────────
    if kind == "ins":
        if a in NUCLEUS:
            return ("모음삽입", "high", f"모음 {a} 첨가")
        if a in ('nn', 'nf'):
            return ("ㄴ-첨가", "medium", f"{a} 첨가")
        return ("음운탈락", "low", f"자음 {a} 삽입(미분류)")  # 드묾

    # ── 대치(sub) ───────────────────────────────────
    tk, tl = _laryngeal(t)   # target series/laryngeal
    ak, al = _laryngeal(a)

    # 모음오류: target·actual 둘 다 모음, 다름
    if t in NUCLEUS and a in NUCLEUS:
        return ("모음오류", "high", f"{t}->{a}")

    # 격음화: 격음<->평음 (같은 조음위치)
    if tk and ak and tk == ak:
        if tl == 'aspir' and al == 'plain':
            return ("격음화", "high", f"격음 {t}->평음 {a} (유기성 소실)")
        if tl == 'plain' and al == 'aspir':
            return ("격음화", "medium", f"평음 {t}->격음 {a} (과유기)")
        # 경음화: 평/격 <-> 경음
        if tl == 'tense' or al == 'tense':
            return ("경음화", "high", f"{t}->{a} (긴장성 편차)")

    # 비음화: 폐쇄 종성 -> 대응 비음
    if OBSTRUENT_CODA_TO_NASAL.get(t) == a:
        return ("비음화", "high", f"폐쇄종성 {t}->비음 {a}")
    if t in CODA and a in NASAL and t not in NASAL:
        return ("비음화", "medium", f"종성 {t}->비음 {a}")

    # 유음화: ㄴ<->ㄹ
    if {t, a} <= {'nf', 'll'} or {t, a} <= {'nn', 'rr'}:
        return ("유음화", "high", f"{t}<->{a} (ㄴ/ㄹ 교체)")

    # 구개음화: t계 종성/초성 + 다음 i계 모음 -> c계
    if t in ('tf', 't0', 'th') and a in ('c0', 'ch', 'cc') and (tn in VOWEL_I):
        return ("구개음화", "medium", f"{t}->{a} before i계 {tn}")

    # 종성 오류: target 이 종성 위치
    if t in CODA:
        return ("종성", "medium", f"종성 {t}->{a}")

    # 초성 오류: target 이 초성 자음(비모음·비종성)
    if t not in NUCLEUS and t not in CODA:
        return ("초성", "medium", f"초성 {t}->{a}")

    return ("기타", "low", f"{t}->{a} (미분류)")


# 초분절/유창성 태그(음소 MDD 불가) — pronunciation_lab 전용
SUPRASEG_FLUENCY = {"강세 오류", "경계억양 오류", "반복", "간투사"}
# core 통계 대상(기타·간투사·반복 제외) — 기존 작업과 동일 기준
CORE_TAGS = ["초성", "종성", "모음오류", "경음화", "비음화", "격음화", "유음화",
             "구개음화", "연음규칙", "ㄴ-첨가", "모음삽입", "음운탈락",
             "강세 오류", "경계억양 오류"]


if __name__ == "__main__":
    tests = [
        {"kind": "sub", "target": "kh", "actual": "k0"},         # 격음화
        {"kind": "sub", "target": "k0", "actual": "kk"},         # 경음화
        {"kind": "sub", "target": "aa", "actual": "vv"},         # 모음오류
        {"kind": "sub", "target": "kf", "actual": "ng"},         # 비음화
        {"kind": "sub", "target": "nf", "actual": "ll"},         # 유음화
        {"kind": "sub", "target": "tf", "actual": "c0", "target_next": "ii"},  # 구개음화
        {"kind": "del", "target": "nf", "actual": None},         # 음운탈락(종성)
        {"kind": "ins", "target": None, "actual": "xx"},         # 모음삽입
        {"kind": "sub", "target": "nf", "actual": "mf"},         # 종성
    ]
    print("[taxonomy self-test]")
    for op in tests:
        et, conf, why = classify_op(op)
        print(f"  {op} -> {et} [{conf}] {why}")
