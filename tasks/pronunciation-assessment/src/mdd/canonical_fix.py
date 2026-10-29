# -*- coding: utf-8 -*-
"""KoG2P 표준발음 보정 — 발음 API 가 실행 중에 만드는 표준발음(KoG2P)이 자주 틀리는 곳을 고친다.

확인한 KoG2P 오류(2026-10-11, AIHub 라벨 표준발음 224문항 대조 + 단어 시험)
  1. 관형형 -(으)ㄹ 뒤 된소리 누락   먹을 수→[머글수] (표준 [머글쑤]) · 볼게요→[볼게요] ([볼께요]) · 많을 줄 · 드실 분 · 있을 정도
  2. 한자어 된소리 누락             사건→[사건] ([사껀]) · 결정 · 실제 · 절대 · 열심히 · 장점 · 성격 · 평가 · 여권 …
  3. ㄴ 첨가 과다·누락              중요→[중뇨] ([중요]) · 필요→[필료] · 공연 … / 별일→[벼릴] ([별릴]) · 웬일 · 볼일 · 교육열
     (합성어의 ㄴ 첨가 — 한여름[한녀름] · 식용유[시굥뉴] — 는 맞게 낸다. 그래서 단어 목록으로만 막는다)
  4. 자음 뒤 'ㅢ'                   희망→[희망] ([히망]) · 무늬 (표준발음법 제5항 다만 3)
  5. '져·쪄·쳐'                     빠졌네요→[빠젼네요] ([빠전네요]) (제5항 다만 1)
  6. **단어 경계를 넘는 동화**       KoG2P 는 띄어쓰기를 무시하고 문장 전체에 규칙을 건다.
       일을 누가→[이를루가] · 이것 좀→[이걷쫌] · 잘 붙이세요(…록 잘)→[짤] · 빵 달콤한→[빵딸콤] · 딱 하루→[따카루]
     그런데 라벨은 구절마다 다르다 — 생각 나세요[생강나세요] · 못 보셨어요[몯뽀셔써요] · 오늘 날씨[오늘랄씨] 는
     경계 너머로 동화하고, 밥 같이[밥가치] · 직접 가서[직쩝가서] · 이것 좀[이걷좀] 은 안 한다. 둘 다 허용 발음이라
     하나로 못 박을 수 없다 → **녹음에 더 맞는 쪽을 고른다**(`variants` + `choose`).
방식
  - 1~5 는 KoG2P 에 넣기 전에 **글자를 고쳐 쓴다**(먹을 수 → 먹을 쑤 · 희망 → 히망). KoG2P 는 그대로 읽는다.
    ㄴ 첨가 과다·된소리 과잉(갈비→[갈삐])은 KoG2P 결과(자모열)에서 그 단어의 잘못된 자모열을 맞는 것으로 바꾼다
    (단어 단독 KoG2P 결과로 만든 표).
  - 6 은 **허용 변이**로 다룬다. 기본은 문장 전체 KoG2P(지금 API 와 같다) + 1~5 보정. 변이 후보는
    단어별 KoG2P(경계 동화 없음) · 경계 ㄴ 첨가 · 조사 '의'→[에] · 비어두 '의'→[이] · 같은 자리 된소리 앞 받침 ㄷ 탈락 ·
    어절 안 공명음 뒤 ㅎ 탈락. 후보마다 CTC 로그우도가
    기본보다 높으면 그 구간을 바꾼다(`choose`). 허용 발음 사이의 선택이라 실제 오류를 감추지 않는다 — 다만
    '경계 비음화를 안 한 것'(생각 나세요 → [생각나세요])처럼 라벨이 오류로 적는 것도 받아들이게 되므로 재현율을 같이 잰다.
목록은 실측으로 확인한 것과 흔한 단어만 넣었다 — 모든 경우를 덮지 않는다. 새 단어는 `LEXICAL_RESPELL`·`NO_N_INSERT` 에 더한다.
**고친 뒤에는 회귀 점검을 돌린다**: `scripts/eval/check_canonical_fix.py`(evalsets/g2p_regression_v1.tsv — 시험 224문항 밖 문장).
2026-10-15 리뷰로 고친 것: 표를 낱말이 있을 때만(동료→[동요] 버그) · 어미는 어절 끝에서만(줄거리→[줄꺼리]) ·
관형형 판정(서울 거예요→[꺼]) · 2음절 낱말은 어절 첫머리에서만(시장점유율→[쩜]) · 거의 등은 의→에 제외 · ㄷ 탈락 범위.
"""

import re

TENSE = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
CONSONANTS = set("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
TENSE_SET = set("ㄲㄸㅃㅆㅉ")

# 2·3. 단어 철자 → 발음대로 고쳐 쓴 철자. 3음절 이상은 어절 안 어디에 있어도 바꾸고(결정적 → 결쩡적),
#      **2음절은 어절 첫머리에서만** 바꾼다 — 시장점유율 ⊃ '장점', 살인사건 은 놓치더라도 엉뚱한 낱말을 건드리지 않게.
#      다른 낱말 속에 들어가 틀리는 것(한자 ⊂ 한자리 · 안과 ⊂ 교실 안과 밖)은 넣지 않았다.
#      성적(成績)[성적] · 물건[물건] 은 사전 발음이 예사소리라 뺐다(2026-10-15 리뷰).
LEXICAL_RESPELL = {
    # 한자어·합성어 된소리
    "사건": "사껀", "조건": "조껀", "문자": "문짜", "인기": "인끼", "결정": "결쩡",
    "일정": "일쩡", "실제": "실쩨", "절대": "절때", "열심": "열씸", "발달": "발딸", "갈등": "갈뜽", "물질": "물찔",
    "출석": "출썩", "발전": "발쩐", "실수": "실쑤", "결석": "결썩", "물가": "물까", "활동": "활똥",
    "열정": "열쩡", "발생": "발쌩", "결심": "결씸", "실시": "실씨", "설정": "설쩡", "결제": "결쩨", "몰상식": "몰쌍식",
    "절실": "절씰", "일상": "일쌍", "실적": "실쩍", "결점": "결쩜", "발상": "발쌍", "출신": "출씬", "출장": "출짱",
    "여권": "여꿘", "용돈": "용똔", "발등": "발뜽", "손등": "손뜽", "염증": "염쯩", "차이점": "차이쩜", "장점": "장쩜",
    "단점": "단쩜", "관점": "관쩜", "시점": "시쩜", "초점": "초쩜", "요점": "요쩜", "인권": "인꿘", "정권": "정꿘",
    "글자": "글짜", "성격": "성껵", "인격": "인껵", "평가": "평까", "산불": "산뿔", "빵집": "빵찝", "밤길": "밤낄",
    "산길": "산낄", "일자리": "일짜리",
    "필수": "필쑤", "가능성": "가능썽", "중요성": "중요썽", "필요성": "필요썽", "문제점": "문제쩜", "공통점": "공통쩜",
    "장단점": "장단쩜", "우울증": "우울쯩", "신분증": "신분쯩", "승차권": "승차꿘", "입장권": "입장꿘", "상품권": "상품꿘",
    "사용법": "사용뻡", "치과": "치꽈", "성과": "성꽈", "내과": "내꽈", "손가락": "손까락", "발가락": "발까락",
    # ㄴ 첨가·유음화
    "공권력": "공꿘녁", "별일": "별릴", "웬일": "웬닐", "볼일": "볼릴", "교육열": "교융녈", "의견란": "의견난",
}
LEXICAL_TENSE = LEXICAL_RESPELL                  # 예전 이름(평가 스크립트 호환)
# 1. 관형형 -(으)ㄹ 뒤 의존명사(띄어 쓴 경우): **앞 어절이 관형형으로 보일 때만**(`_adnominal`) 뒤 어절 첫소리를 된소리로.
#    '거의'·'수건' 같은 낱말이 걸리지 않게 뒤 어절 전체 꼴을 좁게 잡는다.
#    앞 어절이 ㄹ 받침이기만 하면 관형형으로 보던 것은 서울 거예요 · 이틀 정도 · 열 분 까지 된소리로 만들었다(2026-10-15 리뷰).
_PARTICLE = r"(가|도|는|를|을|이|야|예요|에요|요|다|라|로|니까|죠|지만|면|든지|밖에|뿐|은|에|이에요|입니다)?[.?!,]?$"
_DEP = re.compile(r"^(수|거|것|걸|게|데|분|적|바|정도)" + _PARTICLE)
_DEP_JUL = re.compile(r"^줄" + _PARTICLE)                  # '줄'은 뒤가 알다·모르다일 때만(물을 줄 거야 — '주다')
_JUL_NEXT = re.compile(r"^(알|몰|모르)")
# 관형형 -(으)ㄹ 로 보는 앞 어절: '…을/…를'(먹을 · 있을 · 부를 — 조사 을/를 뒤에 의존명사가 오는 문장은 비문이다),
# 또는 동사 활용형으로 흔한 끝 음절. 명사·수사와 겹치는 음절(열 · 일 · 날 · 둘 · 설 · 들 · 물 · 말 · 발 · 길 …)은 한 음절로는 안 받는다.
_ADN_ONE = set("할 갈 볼 올 될 칠 쓸 탈 잘 낼 쉴 뵐 살 팔 놀 울 알 빌 걸 밀 풀 줄 끌 클 질 들".split())
_ADN_LAST = set("할 갈 볼 올 될 칠 쓸 탈 잘 낼 쉴 뵐 팔 놀 알 빌 걸 밀 풀 줄 끌 클 실 릴 닐 킬 날 를 을 들".split())
# 울 은 뺐다(서울). 들 은 복수(학생들 거예요)와 겹치지만 힘들 · 만들 · 어려울… 쪽이 훨씬 흔해 넣었다


def _adnominal(prev):
    if len(prev) >= 2:
        return prev[-1] in _ADN_LAST or prev.endswith("만들")
    return prev in _ADN_ONE


# 1'. 붙여 쓴 어미 — **어절 끝에서만**(줄거리 · 팔걸이 · 살거든요 를 건드리지 않게).
#     -(으)ㄹ걸(요) · -(으)ㄹ거(야·예요·에요·다·라·니·지·죠) · -(으)ㄹ게요 · 문장 끝 -(으)ㄹ게 · -(으)ㄹ지(도·는·를·요·가·에·만)
_END_STEM = "(을|할|볼|갈|줄|올|될|칠|쓸|팔|살)"
_END_WORD = re.compile(r"^(.*?)" + _END_STEM + r"(걸|거|게|지)(요|야|예요|에요|다|라|니|죠|도|는|를|가|에|만)?([.?!,]*)$")
_END_ANY = re.compile(r"^(.*?)([가-힣])(게요|걸요|거예요|거에요|거야)([.?!,]*)$")      # 만들게요 · 놀 거야 … ㄹ 받침이면 다
# -(으)ㄹ수록 은 늘 어미다 — ㄹ 받침 음절이면 다 된다(빠를수록 · 갈수록)
_SUROK = re.compile(r"([가-힣])수록")


def _end_word(w, last):
    """한 어절의 어미 된소리. last = 문장 마지막 어절인가('할게' 는 문장 끝에서만 — '살게 하다' 부사형)."""
    m = _END_WORD.match(w)
    if m:
        head, stem, e, tail, punct = m.groups()
        tail = tail or ""
        ok = ((e == "걸" and tail in ("", "요")) or
              (e == "거" and tail in ("", "야", "예요", "에요", "다", "라", "니", "죠")) or
              (e == "게" and (tail == "요" or (tail == "" and last))) or
              (e == "지" and stem not in ("살", "팔", "줄") and tail in ("", "도", "는", "를", "요", "가", "에", "만")))
        # -(으)ㄹ지 는 된소리(먹을지[머글찌]) 지만 ㄹ 받침 용언 + -지(어디 살지 · 안 팔지 · 줄지 않다)는 예사소리라 살·팔·줄은 뺐다
        if ok:
            return head + stem + _tense_onset(e[0]) + e[1:] + tail + punct
    m = _END_ANY.match(w)
    if m and _coda(m.group(2)) == 8:
        head, syl, e, punct = m.groups()
        return head + syl + _tense_onset(e[0]) + e[1:] + punct
    return w


# 3. ㄴ 첨가를 하지 않는 단어(한자어 등). KoG2P 단독 결과의 'ㅇ/ㄴ/ㄹ 받침 + ㄴ/ㄹ + 이계 모음' 을 원래대로 돌린다.
NO_N_INSERT = ["중요", "필요", "공연", "강요", "명예", "공유", "경유", "상영", "동영상", "방영", "공용", "공예", "전용",
               "형용사", "동요", "등용", "정유", "상용", "영양", "선열", "금요일", "일요일", "목요일", "토요일", "공영", "경영",
               "운영", "본연", "관용", "상연", "선약", "인연", "전염", "준열", "점유"]
# KoG2P 가 된소리로 잘못 내는 낱말(사전 오류·과잉 규칙) — 단어 단독 KoG2P 결과에서 철자에 없는 된소리를 예사소리로
NO_TENSE = ["갈비", "무엇인지"]
_N_INSERT_ONSET = {"이": "니", "야": "냐", "여": "녀", "요": "뇨", "유": "뉴", "예": "녜", "얘": "냬"}
_N_INSERT_SKIP = re.compile(r"^(이|이다|이에요|입니다|이야|예요|요|야)[.?!,]?$")  # 서술격 조사 등 — 첨가 대상 아님


def _coda(ch):
    o = ord(ch) - 0xAC00
    return (o % 28) if 0 <= o < 11172 else None


def _is_hangul(ch):
    return 0 <= ord(ch) - 0xAC00 < 11172


def _tense_onset(ch):
    o = ord(ch) - 0xAC00
    if not (0 <= o < 11172):
        return ch
    c = CHO[o // 588]
    if c not in TENSE:
        return ch
    return chr(0xAC00 + CHO.index(TENSE[c]) * 588 + o % 588)


def _vowel_rules(ch):
    """4·5: 자음 첫소리 + ㅢ → ㅣ · ㅈㅉㅊ + ㅕ → ㅓ (글자 단위)."""
    o = ord(ch) - 0xAC00
    if not (0 <= o < 11172):
        return ch
    on, nu, co = o // 588, (o % 588) // 28, o % 28
    if nu == 19 and on != 11:                       # ㅢ, 첫소리 ㅇ 아님(의는 그대로)
        nu = 20
    elif nu == 6 and on in (12, 13, 14):            # 져·쪄·쳐 → 저·쩌·처
        nu = 4
    return chr(0xAC00 + on * 588 + nu * 28 + co)


def _respell(w):
    for k, v in LEXICAL_RESPELL.items():
        if len(k) >= 3:
            w = w.replace(k, v)
        elif w.startswith(k):
            w = v + w[len(k):]
    return w


def rewrite_text(text):
    """KoG2P 에 넣기 전 철자 보정(1·2·4·5 + 낱말 고쳐쓰기). 바뀐 글을 돌려준다."""
    text = "".join(_vowel_rules(c) for c in text)
    words = [_respell(w) for w in text.split(" ")]
    words = [_end_word(w, i == len(words) - 1) for i, w in enumerate(words)]
    text = _SUROK.sub(lambda m: m.group(0) if _coda(m.group(1)) != 8 else m.group(1) + "쑤록", " ".join(words))
    words = text.split(" ")
    for i in range(1, len(words)):
        prev = re.sub(r"[.?!,]+$", "", words[i - 1])
        if not prev or _coda(prev[-1]) != 8 or not _adnominal(prev):          # 8 = ㄹ 받침
            continue
        w, nxt = words[i], (words[i + 1] if i + 1 < len(words) else "")
        if _DEP.match(w) or (_DEP_JUL.match(w) and _JUL_NEXT.match(nxt)):
            words[i] = _tense_onset(w[0]) + w[1:]
    return " ".join(words)


def build_n_insert_table(kog2p_word):
    """NO_N_INSERT 단어마다 (단어, KoG2P 가 낸 자모열, 맞는 자모열) — kog2p_word(단어) → 자모 문자열."""
    table = []
    for w in NO_N_INSERT:
        got = kog2p_word(w)
        fixed = re.sub(r"([ㅇㄴㄹㅁ])[ㄴㄹ]([ㅑㅕㅛㅠㅖㅣ])", r"\1\2", got)
        if fixed != got:                     # 필료(ㅍㅣㄹㄹㅛ) → 피료(ㅍㅣㄹㅛ) 도 같은 식으로 고쳐진다
            table.append((w, got, fixed))
    return table


_LENIS = dict((v, k) for k, v in TENSE.items())


def _spelling_jamo(word):
    out = []
    for ch in word:
        o = ord(ch) - 0xAC00
        if 0 <= o < 11172:
            out.append(CHO[o // 588])
    return set(out)


def build_no_tense_table(kog2p_word):
    """NO_TENSE 단어마다 (KoG2P 자모열, 철자에 없는 된소리를 예사소리로 돌린 자모열)."""
    table = []
    for w in NO_TENSE:
        got = kog2p_word(w)
        sp = _spelling_jamo(w)
        fixed = "".join(_LENIS[c] if (c in _LENIS and c not in sp) else c for c in got)
        if fixed != got:
            table.append((w, got, fixed))
    return table


def fix_jamo(jamo, n_table, text):
    """KoG2P 결과(자모 리스트)에 표 보정(ㄴ 첨가 과다 · 된소리 과잉). **그 낱말이 글(text)에 있을 때만, 나온 횟수만큼만**
    바꾼다 — 자모열만 보고 바꾸면 동료[동뇨] · 정류장[정뉴장] · 공룡[공뇽] 처럼 같은 자모열을 가진 다른 낱말의 맞는 ㄴ 첨가까지
    지운다(2026-10-15 리뷰). (자음 뒤 ㅢ 는 rewrite_text 가 글자 단위로 한다.)"""
    s = "".join(jamo)
    for w, got, fixed in n_table:
        n = text.count(w)
        if n:
            s = s.replace(got, fixed, n)
    return list(s)


# ── 허용 변이(6) ─────────────────────────────────────────────────────────────

def _norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text or "")).strip()


def _n_insert_text(text):
    """경계 ㄴ 첨가 후보: 받침으로 끝난 어절 + 이·야·여·요·유·예로 시작하는 어절 → 뒤 어절 첫소리를 ㄴ(ㄹ 받침 뒤는 ㄹ)으로."""
    words = text.split(" ")
    changed = False
    for i in range(1, len(words)):
        prev = re.sub(r"[.?!,]+$", "", words[i - 1])
        w = words[i]
        if not prev or not w or not _is_hangul(prev[-1]) or _coda(prev[-1]) == 0 or _N_INSERT_SKIP.match(w):
            continue
        o = ord(w[0]) - 0xAC00
        if not (0 <= o < 11172) or o // 588 != 11:
            continue
        head = chr(0xAC00 + 11 * 588 + (o % 588) - (o % 28))           # 받침 뺀 첫 음절(이·여 …)
        if head not in _N_INSERT_ONSET:
            continue
        on = CHO.index("ㄹ") if _coda(prev[-1]) == 8 else CHO.index("ㄴ")
        words[i] = chr(0xAC00 + on * 588 + o % 588) + w[1:]
        changed = True
    return " ".join(words) if changed else None


# 어절 끝 '의'가 조사가 아니라 낱말의 일부인 것(거의 · 회의 · 주의 …) — 의→에 를 주지 않고 비어두 의→[이](허용)만 준다
_UI_WORDS = ("거의", "회의", "주의", "강의", "예의", "정의", "동의", "합의", "논의", "협의", "문의", "결의", "토의", "편의",
             "의의", "탈의", "내의", "민의", "성의", "고의", "호의", "창의", "건의", "심의", "질의", "모의", "신의", "하의")


def _ui_to_e_text(text):
    """조사 '의' → [에] 후보: 두 음절 이상 어절 끝의 '의'(감사의 · 공권력의). 거의 · 회의 같은 낱말은 뺀다."""
    words = text.split(" ")
    out = []
    for w in words:
        core = re.sub(r"[.?!,]+$", "", w)
        if len(core) >= 2 and core.endswith("의") and not core.endswith(_UI_WORDS):
            w = core[:-1] + "에" + w[len(core):]
        out.append(w)
    return " ".join(out) if out != words else None


def _ui_to_i_text(text):
    """비어두 '의' → [이] 후보(표준발음법 제5항 다만 4: 주의[주이] · 회의[회이] · 거의[거이]). 조사 '의'는 빼고 어절 첫 '의'도 뺀다."""
    words = text.split(" ")
    out = []
    for w in words:
        core = re.sub(r"[.?!,]+$", "", w)
        cs = list(w)
        for k in range(1, len(core)):
            if cs[k] != "의":
                continue
            if k == len(core) - 1 and not core.endswith(_UI_WORDS):       # 어절 끝 조사 '의' — 의→에 쪽에서 다룬다
                continue
            cs[k] = "이"
        out.append("".join(cs))
    return " ".join(out) if out != words else None


def _regions(base, alt, kind):
    """base 와 alt 자모열을 맞춰 다른 구간마다 (i, j, 바꿀 자모열, 종류) — base[i:j] 를 바꾼다(i == j 면 끼워 넣기)."""
    import edlib
    b, a = "".join(base), "".join(alt)
    if b == a or not b or not a:
        return []
    al = edlib.getNiceAlignment(edlib.align(a, b, task="path"), a, b)   # query = alt, target = base
    out, i, cur = [], 0, None
    for q, t in zip(al["query_aligned"], al["target_aligned"]):
        same = q == t
        if same:
            if cur is not None:
                out.append(cur); cur = None
            i += 1
            continue
        if cur is None:
            cur = [i, i, ""]
        if t != "-":
            i += 1; cur[1] = i
        if q != "-":
            cur[2] += q
    if cur is not None:
        out.append(cur)
    return [(s, e, list(r), _kind(base[s:e], list(r), kind)) for s, e, r in out]


_NASAL = {"ㄱ": "ㅇ", "ㄲ": "ㅇ", "ㅋ": "ㅇ", "ㄷ": "ㄴ", "ㅌ": "ㄴ", "ㅅ": "ㄴ", "ㅆ": "ㄴ", "ㅈ": "ㄴ", "ㅊ": "ㄴ",
          "ㅂ": "ㅁ", "ㅍ": "ㅁ"}
_FAMILY = {}
for _f in ("ㄱㄲㅋ", "ㄷㄸㅌ", "ㅂㅃㅍ", "ㅈㅉㅊ", "ㅅㅆ"):
    for _c in _f:
        _FAMILY[_c] = _f


def _kind(b, r, kind):
    """경계 변이를 동화 종류로 나눈다(재현율을 종류별로 재려고)."""
    if kind != "경계":
        return kind
    if len(b) == len(r):
        d = [(x, y) for x, y in zip(b, r) if x != y]
        if d and all(_FAMILY.get(x) and _FAMILY.get(x) == _FAMILY.get(y) for x, y in d):
            return "경계 경음·격음"
        if d and all(_NASAL.get(x) == y or _NASAL.get(y) == x for x, y in d):
            return "경계 비음화"
        if d and all({x, y} == {"ㄴ", "ㄹ"} for x, y in d):
            return "경계 유음화"
    if "ㅎ" in b + r:
        return "경계 격음화"
    return "경계 기타"


def _coda_t_drop(base):
    """같은 자리 된소리 앞 받침 ㄷ 탈락 후보(갔다 [갇따]~[가따] · 했습니다 [핻씀니다]~[해씀니다] · 갔지 [갇찌]~[가찌]).
    자모열에서 자음 앞 자음은 받침이다. ㄲ·ㅃ 앞(듣고[듣꼬] · 꽃밭[꼳빧])은 받침을 빠뜨리는 학습자 오류일 수 있어 뺐다."""
    return [(k, k + 1, [], "ㄷ 탈락") for k in range(len(base) - 1) if base[k] == "ㄷ" and base[k + 1] in "ㄸㅆㅉ"]


def _ss_drop_text(text):
    """과거·추측 ㅆ 받침이 예사소리 앞에서 빠지는 발음 후보(했구나 [핻꾸나]~[해꾸나] · 있구나 · 갔고 · 했지) — 어절 안에서만.
    ㅆ 받침을 지우고 뒤 첫소리를 된소리로 바꿔 쓴 글을 돌려준다(없으면 None). 줄기 받침(듣고 · 꽃밭)은 학습자의 받침 빠뜨리기와
    구별이 안 돼 넣지 않는다(2026-10-15 리뷰)."""
    words = text.split(" ")
    out = []
    for w in words:
        cs = list(w)
        for k in range(len(cs) - 1):
            o, n = ord(cs[k]) - 0xAC00, ord(cs[k + 1]) - 0xAC00
            if 0 <= o < 11172 and 0 <= n < 11172 and o % 28 == 20 and CHO[n // 588] in "ㄱㄷㅂㅅㅈ":   # 20 = ㅆ 받침
                cs[k] = chr(0xAC00 + o - 20)
                cs[k + 1] = _tense_onset(cs[k + 1])
        out.append("".join(cs))
    return " ".join(out) if out != words else None


_VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")


def _h_drop_text(text):
    """공명음 받침 뒤 ㅎ 탈락 후보(전화 [전화]~[저놔] · 물하고 [무라고] · 결혼) — **어절 안**에서만. 어절 첫 ㅎ
    (저는 한국어)은 그대로 둔다. 뒤 음절 첫소리 ㅎ 을 ㅇ 으로 바꿔 쓴 글을 돌려준다(없으면 None)."""
    words = text.split(" ")
    out = []
    for w in words:
        cs = list(w)
        for k in range(1, len(cs)):
            o, p = ord(cs[k]) - 0xAC00, ord(cs[k - 1]) - 0xAC00
            if 0 <= o < 11172 and 0 <= p < 11172 and o // 588 == 18 and p % 28 in (4, 8, 16, 21):   # ㅎ · ㄴㄹㅁㅇ 받침
                cs[k] = chr(0xAC00 + 11 * 588 + o % 588)
        out.append("".join(cs))
    return " ".join(out) if out != words else None


# 기본으로 끄는 변이 — 2026-10-11 측정(eval_canonical_source.py, 6,720발화): 경계 비음화는 고른 변이가 라벨에서
# 멀어지는 쪽이 많고(26:45) 재현율만 깎는다(생각 나세요 → [생각나세요] 를 받아 줘 버린다). ㄴ 첨가는 효과 0 이나 해는 없다.
EXCLUDE_KINDS = ("경계 비음화",)


class CanonicalBuilder:
    """문장 → 보정 표준발음 / 허용 변이. kog2p(문장) → 자모 리스트(발음 API canonical() 과 같은 함수)."""

    def __init__(self, kog2p):
        self.kog2p = kog2p
        kw = lambda w: "".join(kog2p(w))
        self.table = build_n_insert_table(kw) + build_no_tense_table(kw)

    def _g2p(self, text, per_word=False):
        if per_word:
            out = []
            for w in _norm(text).split(" "):
                out += fix_jamo(self.kog2p(w), self.table, w)
            return out
        return fix_jamo(self.kog2p(text), self.table, text)

    def fixed(self, text):
        """결정적 보정(1~5)만 — 문장 전체 KoG2P."""
        return self._g2p(rewrite_text(text))

    def variants(self, text, kinds=None, exclude=EXCLUDE_KINDS):
        """→ (기본 자모열, 변이 [(i, j, 자모열, 종류)]). kinds 를 주면 그 종류만(종류 이름의 앞부분으로 거른다).
        exclude 의 종류는 뺀다(기본 EXCLUDE_KINDS — 모든 종류를 보려면 exclude=())."""
        t = rewrite_text(text)
        base = self._g2p(t)
        alts = _regions(base, self._g2p(t, per_word=True), "경계")
        t2 = _n_insert_text(t)
        if t2:
            alts += _regions(base, self._g2p(t2), "ㄴ 첨가")
        t3 = _ui_to_e_text(t)
        if t3:
            alts += _regions(base, self._g2p(t3), "의→에")
        t5 = _ui_to_i_text(t)
        if t5:
            alts += _regions(base, self._g2p(t5), "의→이")
        t4 = _h_drop_text(t)
        if t4:
            alts += _regions(base, self._g2p(t4), "ㅎ 탈락")
        t6 = _ss_drop_text(t)
        if t6:
            alts += _regions(base, self._g2p(t6), "ㄷ 탈락")
        alts += _coda_t_drop(base)
        seen, out = set(), []
        for s, e, r, k in alts:
            key = (s, e, "".join(r))
            if key in seen or (kinds is not None and not any(k.startswith(x) for x in kinds)) or k in exclude:
                continue
            seen.add(key)
            out.append((s, e, r, k))
        return base, out


def choose(lprobs, base, alts, to_id, blank=0, margin=1.0):
    """변이마다 CTC 로그우도를 재서 기본보다 margin 넘게 높은 것을 고른다. 겹치는 변이끼리는 가장 높은 하나.

    margin 기본 1.0(2026-10-15 측정, 6,720발화 · 보정이 안 바꾸는 196문항): 0 이면 헛지적 37.1→17.2% · 재현율 28.7→27.0%,
    1 이면 18.2% · 27.6%, 3 이면 20.0% · 27.8%. 재현율 손해를 줄이는 쪽으로 1 을 기본으로 둔다.

    lprobs: (T, V) 로그확률(numpy 나 torch) · to_id(자모) → id
    → (고른 자모열, [(i, j, 자모열, 종류, 우도 차)]). 표준발음 자체가 정렬되지 않으면 (base, []).
    """
    import torch
    import torch.nn.functional as F
    if not alts:
        return list(base), []
    seqs = [list(base)] + [list(base[:s]) + list(r) + list(base[e:]) for s, e, r, _k in alts]
    if any(len(q) == 0 for q in seqs):
        return list(base), []
    lp = torch.as_tensor(lprobs).float()
    T, V = int(lp.shape[0]), int(lp.shape[1])
    N = len(seqs)
    tg = torch.tensor([to_id(c) for q in seqs for c in q], dtype=torch.long)
    lens = torch.tensor([len(q) for q in seqs], dtype=torch.long)
    with torch.no_grad():
        nll = F.ctc_loss(lp.unsqueeze(1).expand(T, N, V).contiguous(), tg, torch.full((N,), T, dtype=torch.long), lens,
                         blank=blank, reduction="none", zero_infinity=False)
    ll = (-nll).tolist()
    if ll[0] != ll[0] or ll[0] == float("-inf"):
        return list(base), []
    gain = [(v - ll[0]) if v == v else float("-inf") for v in ll[1:]]
    # 겹침(끼워 넣기는 그 자리 양옆과 겹친다고 본다) 무리마다 가장 큰 것 하나
    order = sorted(range(len(alts)), key=lambda k: -gain[k])
    taken, picked = [], []
    for k in order:
        if not (gain[k] > margin):
            break
        s, e = alts[k][0], alts[k][1]
        if any(s <= e2 and s2 <= e for s2, e2 in taken):
            continue
        taken.append((s, e))
        picked.append(k)
    out = list(base)
    for k in sorted(picked, key=lambda k: -alts[k][0]):
        s, e, r, _kd = alts[k]
        out[s:e] = list(r)
    return out, [tuple(alts[k]) + (gain[k],) for k in sorted(picked, key=lambda k: alts[k][0])]
