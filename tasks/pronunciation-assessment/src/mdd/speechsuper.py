# -*- coding: utf-8 -*-
"""SpeechSuper `sent.eval.kr` 호환 응답 만들기 — 우리 진단 결과를 그 스키마로 옮긴다.

호환 필드
    result.{overall, pronunciation, fluency, rhythm}
    result.words[].{word, readType, score, phonemes[]}
    phonemes[].{alpha, pronunciation, position}

우리가 더 얹는 필드(그쪽 클라이언트는 모르는 키라 무시된다)
    phonemes[].{alpha_jamo, actual, actual_jamo, errorType}   ← **무엇을 무엇으로 냈는지**
    words[].inserted                                          ← 없는 소리를 넣은 경우
    _meta                                                     ← 신뢰도 경고·기준

글자↔소리 대응이 1:1 이 아닌 경우가 있어 **철자 자모열과 표준발음 자모열을 정렬해서** 묶는다.
    · 음운규칙   독립문 → [동님문] : 글자 ㄱ 자리의 소리는 ㅇ
    · 연음       옷이 → [오시]     : 앞 글자 받침이 뒷 글자 첫소리로 이동
    · 개수 불일치 닭 → [닥](겹받침 1소리) · ㅙ → ㅗ+ㅐ(우리 내부 표기) · 초성 ㅇ 은 소리 없음
정렬로 각 표준발음 자리에 **글자 번호**를 붙이고, 자리에 대응하는 소리가 없으면(삭제) `actual=None`,
표준 자리가 없는 소리(삽입)는 **앞 글자**의 `inserted` 로 보낸다.

점수는 **(1 − 그 소리가 틀렸을 확률) × 100** 이다(`scripts/score/calibrate_phone_scores.py` 의 `calibration.json`).
검증셋 실측: 맞게 발음한 자리의 **97.3% 가 80점 이상**, 90점 이상 구간의 실제 오류율 1.2%.
보정 파일이 없으면 예전 방식(백분위 `norms.json`) 으로 물러난다 — 그때는 50 이 평균이라 화면 기준선이 달라진다.
"""

import json
import os

import numpy as np

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

READ_NORMAL, READ_MISS, READ_SUB, READ_BAD, READ_INS = 0, 1, 2, 3, 4

# 문장 점수 눈금 맞추기 ─────────────────────────────────────────────
# 네 칸이 서로 다른 자를 쓰고 있었다: overall 은 음소 점수 평균(맞으면 90+),
# pronunciation/fluency/rhythm 은 사람 점수 1~5 를 그대로 늘린 값(3점=50).
# 그래서 "overall 89 인데 pronunciation 50" 이 나왔다.
#   · pronunciation = 음소 기반 값으로 바꾼다(= overall 과 같은 자)
#   · fluency·이해도 = **학습자 분포에서의 백분위**(`utt_percentile.json` 이 있을 때).
#     사람 채점이 외국인 기준이라 평균 3.3/5 이고 회귀 모델은 평균 쪽으로 붙어(실제 5점 발화도 예측 3.82)
#     5점 만점 비례로는 **잘 읽어도 76점이 천장**이었다. 백분위로 펴면 순서는 그대로 두고 눈금만 넓어진다.
#     → 80점 = 학습자 상위 20%. 보정 파일이 없으면 예전처럼 5점 만점 대비 비율로 물러난다.
#   · overall = 0.6·발음 + 0.2·유창 + 0.2·이해 (없는 값은 빼고 남은 것끼리 비율 유지)
# ⚠️ 눈금만 맞춘 것이다. 발화 1개의 정확도(상관 0.43·유창성 −0.03)는 그대로다.
# 전사(CTC 받아쓰기)와 점수(강제정렬 확신도)는 따로 계산돼서 어긋날 수 있다 —
# 「ㅕ 를 ㅓ 로 들었다」고 표시하면서 그 자리 점수가 90 인 일이 실제로 나왔다.
# 학습자에게 보여주는 지적은 **둘이 같이 말할 때만** 한다. 1,787,938 자리 실측:
#   전사만        정밀도 0.474 · 재현율 0.518 · F1 0.4948 · 지적 74,340
#   전사+점수<60  정밀도 0.557 · 재현율 0.319 · F1 0.4058 · 지적 38,970  ← 채택
# F1 은 내려가지만 **틀렸다고 잘못 말하는 횟수가 절반으로 줄고**, 화면의 모순이 사라진다.
# 들린 소리(actual·actual_jamo)는 그대로 실어 보내므로 정보는 잃지 않는다.
FLAG_MAX = 60
# 예외(2026-10-18): 첫소리 예사소리(ㄱ ㄷ ㅂ ㅅ ㅈ)를 짝 된소리·거센소리로 들은 자리는 점수가 FLAG_MAX 를 넘어도 지적한다.
# 이 방향은 자리 점수가 오류와 정상을 못 가른다(학습자 902자리 점수 중앙값 오류 75 · 정상 78) — FLAG_MAX 에 막혀
# 모델이 들은 오류의 86% 가 '애매'로 숨었다(일부러 틀린 합성 300개: 뚜부 2/20 · 싸과 2/20 · 한쿡 0/20 지적, 20/20 인
# 거피·도끼·보도와 비대칭). 들은 소리의 확률(`gop_feats_v2.pred_probs`)이 STRONG_MIN_PROB 이상일 때만 연다.
STRONG_OF = {"ㄱ": "ㄲㅋ", "ㄷ": "ㄸㅌ", "ㅂ": "ㅃㅍ", "ㅅ": "ㅆ", "ㅈ": "ㅉㅊ"}
STRONG_MIN_PROB = 0.6
# 예외 2(2026-10-19 서비스 점검): 첫소리·받침 ㄴ↔ㄹ 도 같은 확률 기준으로 연다. 모델은 받침 ㄹ→ㄴ 을 77% 듣는데 화면 지적은
# 21% 였다. 풀었을 때 정밀도 50~65%(학습자 6,720발화) — 서비스 평균(40%)보다 높다.
LIQUID_NASAL_PAIRS = {"ㄹ": "ㄴ", "ㄴ": "ㄹ"}
LIQUID_NASAL = {}            # 기본은 끔(예전과 같다) — app.py --liquid-rule 이 LIQUID_NASAL_PAIRS 로 켠다
# 빠진 소리(음운탈락)는 점수가 비어 FLAG_MAX 를 안 거치고 늘 표시됐다. 학습자에게서도 맞을 확률 22%(진짜 247 · 헛것 850)이고
# 원어민 낭독 헛지적의 절반이 이것이다. 강제정렬 점수로는 진짜와 헛것이 안 갈린다(중앙값 55 vs 54).
# → 연속으로 DEL_RUN_MIN 개 이상 빠졌을 때(글자·낱말을 통째로 빼먹음)만 지적하고, 홀로 빠진 소리는 '애매'로 둔다(2026-10-19).
DEL_RUN_MIN = 1             # 기본 1 = 예전처럼 빠진 소리를 다 지적. app.py --del-run-min 3 으로 바꾼다
# 고침(2026-10-19 저녁): 연속 길이는 나쁜 기준이었다 — 학습자에게서 남긴 빠짐 지적은 18% 만 맞고 지운 것은 40% 가 맞았다
# (ㄴ 첨가를 안 한 별일[별릴]→[벼릴] · 받침 빠뜨림을 거의 다 숨김). 중앙값은 같아도 강제정렬 점수가 낮은 쪽은 진짜가 많다
# (20 미만 43% · 전체 32%). → 빠진 소리는 강제정렬 점수(_score)가 DEL_SCORE_MAX 미만일 때만 지적한다.
# 100% 모델 · 50 에서 학습자 진짜 빠짐 지적 50 → 107(연속 3 대비) · 원어민 100음절당 낭독 1.3 → 1.6.
DEL_SCORE_MAX = None        # 기본 None = 끔. app.py --del-score-max 50 으로 켠다(DEL_RUN_MIN 은 1 로 둘 것)
# 예외 3(2026-10-19): ㅅ↔ㅆ. 일부러 또렷하게 틀리면 모델 확률이 뒤집힌다(사과 ㅆ 0% → 싸과 ㅆ 70%, 같은 목소리 짝 71%).
#   ㅅ→ㅆ: 들은 소리 확률 SS_LOOSE_PROB(0.5) 이상이면 표시(예외 1 의 0.6 대신).
#   ㅆ→ㅅ: 첫소리·모음 뒤 자리에서만 STRONG_MIN_PROB 이상이면 표시. 받침 뒤 ㅆ(할 수[할쑤] · 인상류)는 원어민도
#          37~71% 가 ㅅ 에 가깝게 내서(학습에 안 쓴 원어민 낭독) 넣지 않는다. 첫소리·모음 뒤는 일부러 ㅅ 로 읽은 합성 71% 를 듣는다.
SS_RULE = False             # 기본 끔 — app.py --ss-rule 이 켠다
SS_LOOSE_PROB = 0.5
# 예외 4(2026-10-22): ㅅㅆ·ㅈㅉㅊ 첫소리 판별기(mdd.sz_detector) — 받아쓰기가 못 듣는 세기 오류를 모음 시작 목소리 결·음높이·길이로
# 다시 본다. sites[i]["sz"] 가 있을 때만 쓴다. 기본 끔 — app.py --sz-detector <json> 이 켠다.
SZ_RULE = False

HUMAN_MAX = 5.0
UTT_PCT = None          # {"pron": {"q": [...]}, ...} — SpeechSuperFormatter 가 채운다
W_OVERALL = {"pron": 0.6, "flu": 0.2, "comp": 0.2}


def human_to_100(v, key=None, pct=None):
    """사람 점수(1~5) 예측값 → 0~100.

    백분위 표가 있으면 **학습자 중 상위 몇 %** 로, 없으면 5점 만점 대비 비율로 바꾼다.
    """
    if v is None:
        return None
    if pct and key in pct:
        q = pct[key]["q"]
        return int(round(float(np.clip(np.interp(float(v), q, np.arange(0, 101, 2)), 0, 100))))
    return int(round(float(np.clip(float(v) / HUMAN_MAX * 100.0, 0, 100))))
BAD_SCORE = 20          # 글자 점수가 이보다 낮으면 오독(readType 3) 으로 본다


def nz(v):
    """0 점은 숫자로 내보내지 않는다 — 0 과 "판정 못함"을 한 칸(null)으로 합쳐 달라는 요청(2026-09-26).

    **평균은 0 을 그대로 넣어 구한 뒤 마지막에만 바꾼다.** 0 을 평균에서 빼면 점수가 부풀려진다
    (SpeechSuper 비교에서 실측: 빠뜨린 소리를 0 으로 세도, 평균에서 빼도 둘 다 나빠졌다).
    """
    return None if v == 0 else v


def decompose(text):
    """한글 문장 → [(글자번호, 글자, 자모)]. 소리 없는 초성 ㅇ 은 뺀다(표준발음열과 맞추기 위해)."""
    out, wi = [], 0
    for ch in text:
        if not ("가" <= ch <= "힣"):
            continue
        s = ord(ch) - 0xAC00
        cho, jung, jong = CHO[s // 588], JUNG[(s % 588) // 28], JONG[s % 28]
        if cho != "ㅇ":
            out.append((wi, ch, cho))
        out.append((wi, ch, jung))
        if jong != " ":
            out.append((wi, ch, jong))
        wi += 1
    return out


def map_to_words(text, canon):
    """표준발음 자리 i → 글자 번호. 철자 자모열과 정렬해서 잇는다.

    ⚠️ 2026-09-25 조사: SpeechSuper 와 글자 묶음이 22.5% 어긋나기에 **소리 음절 기준**(연음된 초성을
    다음 글자로 보내는 방식)으로 바꿔 봤으나, 저장된 30건으로 재측정하니 일치율이 77.5% → 67.6% 로
    **떨어져 되돌렸다.** 이유는 두 가지다.
      · SpeechSuper 자체가 일관되지 않다 — 값이는 소리 기준(값=ㄱㅏㅂ·이=ㅆㅣ),
        꽃이·밥을은 철자 기준(꽃=ㄲㅗㅊ·밥=ㅂㅏㅂ) 이다.
      · 그쪽 `alpha` 에는 **실제로 들린 소리**가 섞여 있다(학=ㅏㄱ·먹=ㅁㅓㅇ·공=ㄲㅗㅇ).
        우리 `alpha`(내야 할 소리)와 같은 종류의 값이 아니므로 일치율 자체가 기준이 되지 못한다.
    학습자 화면에서는 글자에 보이는 철자와 맞추는 편이 읽기 쉬워 **철자 기준을 유지한다.**
    """
    import edlib
    spell = decompose(text)
    if not spell or not canon:
        return [0] * len(canon), [c for _, c, _ in dict((w, (w, c, None)) for w, c, _ in spell).values()]
    words = []
    for wi, ch, _ in spell:
        if not words or words[-1][0] != wi:
            words.append((wi, ch))
    a = edlib.getNiceAlignment(edlib.align("".join(canon), "".join(j for _, _, j in spell), task="path"),
                               "".join(canon), "".join(j for _, _, j in spell))
    owner, si, last = [], 0, 0
    for x, y in zip(a["query_aligned"], a["target_aligned"]):
        if y != "-":
            last = spell[si][0]; si += 1
        if x != "-":
            owner.append(last)
    while len(owner) < len(canon):
        owner.append(last)
    return owner, [c for _, c in words]


class SpeechSuperFormatter:
    """진단 결과 → SpeechSuper 호환 JSON. norms.json 이 없으면 점수는 None 으로 둔다."""

    def __init__(self, norms_path=None, jamo_to_phones=None, phone_context=None, ctype=None,
                 calib_path=None, utt_pct_path=None, flu_rule_path=None, utt_comp=False):
        self.norms = json.load(open(norms_path, encoding="utf-8")) if (norms_path and os.path.exists(norms_path)) else {}
        self.cal = json.load(open(calib_path, encoding="utf-8")) if (calib_path and os.path.exists(calib_path)) else None
        self.pct = (json.load(open(utt_pct_path, encoding="utf-8"))
                    if (utt_pct_path and os.path.exists(utt_pct_path)) else None)
        self.flu = (json.load(open(flu_rule_path, encoding="utf-8"))
                    if (flu_rule_path and os.path.exists(flu_rule_path)) else None)
        # 이해 가능성은 발화 1개 기준 사람 채점과 상관 0.433 뿐이다(화자 단위 0.691).
        # 백분위 눈금에서는 예측이 평균(3.3/5) 근처에 몰려 있어 조금만 낮아도 한 자릿수로 떨어진다
        # — 정확히 읽은 녹음이 이해 4점으로 나왔다. 발화 1개로는 **보내지 않고** 세션(5발화+)에서만 쓴다.
        self.utt_comp = bool(utt_comp)
        self.j2p = jamo_to_phones
        self.ctx = phone_context
        self.ctype = ctype

    def _score(self, key, margin):
        """(1 − 오류확률) × 100. 보정 파일이 없으면 백분위로 물러난다."""
        if margin is None:
            return None
        if self.cal:
            c = self.cal.get(key) or self.cal.get("_global")
            if c:
                p = 1.0 / (1.0 + np.exp(-(c["a"] * float(margin) + c["b"])))
                return int(round(float(np.clip((1.0 - p) * 100.0, 0, 100))))
        if key not in self.norms:
            return None
        q = self.norms[key]["q"]
        return int(round(float(np.clip(np.interp(margin, q, np.arange(0, 101, 2)), 0, 100))))

    def build(self, text, canon, sites, margins, utt_scores=None, record_id=None, vowels=None,
              feats=None):
        """canon: 표준발음 자모열 · sites: 자리별 판정 · margins: 자리별 margin GOP."""
        owner, words = map_to_words(text, canon)
        canon_ph = self.j2p(canon) if (self.j2p and canon) else [None] * len(canon)
        pred = [s.get("pred") for s in sites]
        pred_ph = [s.get("pred_ph") for s in sites]
        buckets = [[] for _ in words]
        ins = [[] for _ in words]
        del_run = [0] * len(sites)                   # 자리마다 '이어서 빠진 소리' 길이
        k = 0
        while k < len(sites):
            if pred[k] is None:
                j = k
                while j < len(sites) and pred[j] is None:
                    j += 1
                for q in range(k, j):
                    del_run[q] = j - k
                k = j
            else:
                k += 1
        # 글자 하나가 통째로 안 들렸으면 그 글자의 빠진 소리는 **한꺼번에** 정한다(2026-10-22). 자모마다 따로 정하면
        # 일부만 '애매'가 되어 글자 판정이 '빠짐'이 아니라 '다르게 읽음'으로 나갔다(이상한 입력 103개: 통째로 안 들린
        # 글자 694개 중 다르게 읽음 170 · 정상 18). DEL_SCORE_MAX 가 켜져 있으면 글자의 강제정렬 점수 평균이 그 미만일 때
        # 전부 지적(→ 빠짐), 아니면 전부 애매. 무조건 다 지적하면 처음 보는 문단 헛지적이 44% 늘었다.
        whole_flag = [None] * len(words)            # 통째로 안 들린 글자만: True = 다 지적 · False = 다 애매
        if words:
            heard_w, ds_w = [False] * len(words), [[] for _ in words]
            seen_w = [False] * len(words)
            for i in range(len(sites)):
                wi = min(owner[i] if i < len(owner) else (len(words) - 1), len(words) - 1)
                seen_w[wi] = True
                heard_w[wi] = heard_w[wi] or pred[i] is not None
                if pred[i] is None and DEL_SCORE_MAX is not None:
                    key_i = self.ctx(canon, i, vowels) if (self.ctx and vowels) else None
                    v = self._score(key_i, margins[i] if i < len(margins) else None)
                    if v is not None:
                        ds_w[wi].append(v)
            for wi in range(len(words)):
                if seen_w[wi] and not heard_w[wi]:
                    whole_flag[wi] = (DEL_SCORE_MAX is None or not ds_w[wi]
                                      or sum(ds_w[wi]) / len(ds_w[wi]) < DEL_SCORE_MAX)
        for i, s in enumerate(sites):
            wi = owner[i] if i < len(owner) else (len(words) - 1)
            wi = min(wi, len(words) - 1)
            pos = "중성" if (vowels and canon[i] in vowels) else None
            key = self.ctx(canon, i, vowels) if (self.ctx and vowels) else None
            if pos is None and key:
                pos = "종성" if key.startswith("받침") else "초성"
            p = {"alpha": canon_ph[i], "alpha_jamo": canon[i], "position": pos,
                 # 인식하지 못한 소리는 강제정렬 margin 으로 0점 처리하지 않는다.
                 "pronunciation": (self._score(key, margins[i] if i < len(margins) else None)
                                   if pred[i] is not None else None),
                 "actual": pred_ph[i], "actual_jamo": pred[i]}
            if pred[i] and pred[i] != canon[i] and self.ctype:
                is_coda = pos == "종성"
                p["errorType"] = self.ctype(canon[i], pred[i], "sub", is_coda)
            elif pred[i] is None:
                ds = (self._score(key, margins[i] if i < len(margins) else None)
                      if DEL_SCORE_MAX is not None else None)
                if whole_flag[wi] is not None:
                    flag = whole_flag[wi]                # 통째로 안 들린 글자 — 위 주석
                else:
                    flag = del_run[i] >= DEL_RUN_MIN and (ds is None or ds < DEL_SCORE_MAX)
                if flag:
                    p["errorType"] = "음운탈락"
                else:
                    p["uncertain"] = True                # 홀로 빠진 소리 · 소리 흔적이 있는 곳 — 위 DEL_RUN_MIN 주석
            # 전사는 다르다는데 점수가 높으면 지적하지 않는다(FLAG_MAX 주석의 실측 근거). 예외 둘은 위 주석.
            pr_i = s.get("prob") or 0
            sure = bool(pred[i]) and pr_i >= STRONG_MIN_PROB
            strong = sure and ((pos == "초성" and pred[i] in STRONG_OF.get(canon[i], ""))
                               or (pos in ("초성", "종성") and LIQUID_NASAL.get(canon[i]) == pred[i]))
            if SS_RULE and pos == "초성" and pred[i]:
                if canon[i] == "ㅅ" and pred[i] == "ㅆ" and pr_i >= SS_LOOSE_PROB:
                    strong = True
                elif canon[i] == "ㅆ" and pred[i] == "ㅅ" and pr_i >= STRONG_MIN_PROB and (i == 0 or canon[i - 1] in (vowels or ())):
                    strong = True
            # 판별기(SZ_RULE · mdd.sz_detector): ㅅㅆ·ㅈㅉㅊ 첫소리에서 소리 단서로 본 오류 확률이 기준선 이상이면 지적한다.
            # 받아쓰기가 표준음으로 적었으면 들린 소리를 판별기가 고른 자모로 바꾼다(detector 칸에 확률을 남긴다).
            szr = s.get("sz") if SZ_RULE else None
            if szr and pos == "초성" and szr.get("th") is not None and szr["p"] >= szr["th"]:
                if pred[i] == canon[i] and self.ctype:
                    alt = szr["alt"]
                    p["actual_jamo"] = alt
                    try:
                        seq = list(canon); seq[i] = alt
                        p["actual"] = self.j2p(seq)[i] if self.j2p else None
                    except Exception:
                        p["actual"] = None
                    p["errorType"] = self.ctype(canon[i], alt, "sub", False)
                    p.pop("uncertain", None)
                    p["detector"] = szr["p"]
                    strong = True
                elif pred[i] and pred[i] != canon[i]:
                    p["detector"] = szr["p"]
                    strong = True
            if p.get("errorType") and p["pronunciation"] is not None and p["pronunciation"] >= FLAG_MAX and not strong:
                p.pop("errorType")
                p["uncertain"] = True
            buckets[wi].append(p)
            for t in (s.get("inserted") or ""):
                ins[wi].append(t)
        out_words = []
        for wi, w in enumerate(words):
            ph = buckets[wi]
            got = [q["pronunciation"] for q in ph if q["pronunciation"] is not None]
            score = int(round(sum(got) / len(got))) if got else None
            n_sub = sum(1 for q in ph if q["actual_jamo"] and q["actual_jamo"] != q["alpha_jamo"]
                        and not q.get("uncertain"))
            n_del = sum(1 for q in ph if q["actual_jamo"] is None and not q.get("uncertain"))
            if ph and n_del == len(ph):
                rt = READ_MISS
            elif score is not None and score < BAD_SCORE and (n_sub or n_del):
                rt = READ_BAD
            elif n_sub or n_del:
                rt = READ_SUB
            elif ins[wi]:
                rt = READ_INS
            else:
                rt = READ_NORMAL
            row = {"word": w, "readType": rt, "score": 0 if rt == READ_BAD else score, "phonemes": ph}
            if ins[wi]:
                row["inserted"] = [{"actual_jamo": t} for t in ins[wi]]
            out_words.append(row)

        got = [w["score"] for w in out_words if w["score"] is not None]
        phone_based = int(round(sum(got) / len(got))) if got else None
        for row in out_words:                    # 평균을 구한 뒤에 0 → null
            row["score"] = nz(row["score"])
            for q in row["phonemes"]:
                q["pronunciation"] = nz(q["pronunciation"])
        # 전체 인식 실패 시 회귀기가 낸 값도 유효한 발화 점수가 아니다.
        us = (utt_scores or {}) if any(p is not None for p in pred) else {}
        pron = phone_based                                  # 음소 기반 — overall 과 같은 자
        flu_human = human_to_100(us.get("flu"), "flu", self.pct)
        flu = flu_human
        if self.flu and feats:                    # 속도·쉼만 보는 유창성(발음과 독립) — mdd.fluency
            from mdd.fluency import fluency_rule
            flu = int(round(fluency_rule(feats, self.flu["band"])))
        comp_human = human_to_100(us.get("comp"), "comp", self.pct)
        comp = comp_human if self.utt_comp else None    # 발화 1개로는 신뢰도가 낮다(위 주석)
        parts = [(W_OVERALL[k], v) for k, v in (("pron", pron), ("flu", flu), ("comp", comp)) if v is not None]
        overall = int(round(sum(w * v for w, v in parts) / sum(w for w, _ in parts))) if parts else None
        overall, pron, flu, comp = nz(overall), nz(pron), nz(flu), nz(comp)
        return {
            "refText": text, "recordId": record_id,
            # rhythm 자리에는 **이해 가능성**을 넣는다 — 우리에게 리듬 점수는 없고,
            # 앱은 네 칸(overall·pronunciation·fluency·rhythm)을 그리므로 빈 칸을 남기지 않는다.
            # 같은 값을 comprehensibility 로도 보내, 의미를 아는 쪽은 그 이름으로 읽으면 된다.
            "result": {"overall": overall, "pronunciation": pron, "fluency": flu,
                       "rhythm": comp, "comprehensibility": comp, "words": out_words},
            "_meta": {
                "engine": "beavertalk-mdd",
                "scale": "overall = 0.6·발음 + 0.2·유창 + 0.2·이해. 발음은 음소 판정 기반, "
                         + ("유창·이해는 학습자 분포 백분위(80 = 상위 20%)" if self.pct
                            else "유창·이해는 사람 점수(5점 만점) 대비 비율"),
                "pronunciation_src": "음소 점수 평균(= 글자 점수 평균). 사람 점수 예측이 아니다",
                "fluency_src": ("속도·쉼만 보는 규칙(발음과 독립. 사람 라벨 상관 0.44/0.68)" if (self.flu and feats)
                                else "사람 유창성 점수 예측(발음과 함께 움직인다. 상관 0.61/0.76)"),
                "fluency_human": flu_human,      # 사람 라벨 예측값도 같이 보낸다(눈금은 위와 같다)
                "comprehensibility_utt": comp_human,   # 발화 1개 값 — 참고용으로만(상관 0.433)
                "comprehensibility_note": ("발화 1개 이해도는 상관 0.433 뿐이라 result 에 넣지 않는다"
                                           " — /session 으로 5발화 이상 모아 쓸 것" if not self.utt_comp
                                           else "발화 1개 이해도를 그대로 내보내는 설정(utt_comp=True)"),
                "flagged_rule": "전사가 표준발음과 달라도 그 자리 점수가 %d 이상이면 오류로 표시하지 않는다"
                                " — uncertain:true 로만 남긴다" % FLAG_MAX,
                "raw_human_1to5": {k: (round(float(us[k]), 2) if us.get(k) is not None else None)
                                   for k in ("pron", "comp", "flu")},
                "phoneme_score": "(1 − 그 소리가 틀렸을 확률) × 100. 검증셋에서 맞게 발음한 자리의 97.3% 가 80 이상",
                "fluency_warning": "유창성은 발화 1개 기준 상관 0.61 · 화자 단위 0.76. 발음이 흐리면 같이 내려간다",
                "sentence_score": "발화 1개 기준은 사람 채점과 상관 0.43~0.61 로 낮다 — 10발화 이상 모아 쓸 것",
                "rhythm": "리듬 점수는 없다. 이 칸에는 **이해 가능성(comprehensibility)** 을 넣었다 — 화면 라벨을 '이해도'로 바꿀 것",
                "comprehensibility": "AIHub 3종 중 하나(이해 가능성). SpeechSuper 규격엔 없는 추가 항목",
                "score_pcc_utt": {"pronunciation": 0.433, "comprehensibility": 0.433, "fluency": 0.610},
                "score_pcc_spk": {"pronunciation": 0.674, "comprehensibility": 0.691, "fluency": 0.763},
                "zero_as_null": "0 점은 null 로 보낸다(요청). null = 0 점이거나 그 자리를 판정하지 못한 것",
                "unmatched": "표준발음 자리에 대응하는 소리가 없으면 actual=null, 표준 자리가 없는 소리는 words[].inserted",
            },
        }
