# -*- coding: utf-8 -*-
"""대치 쌍별 조음 설명 — "표준 ㄲ 을 ㄱ 처럼 냈다"를 "무엇을 어떻게 바꾸면 되는가"로 옮긴다.

판별기(`mdd.phone_detector`)는 자리마다 "오발음인가"와 "무엇으로 냈나"(오발음 자리 top-1 79%)를 낸다.
여기서는 (표준음, 들린 소리) 쌍의 **조음 자질 차이**로 문장을 만든다 — 쌍을 일일이 나열하지 않으므로 모든 조합에서 돈다.
    자음  위치(양순·치조·경구개·연구개·후두) · 방법(파열·파찰·마찰·비음·유음) · 발성(평음·경음·격음)
    모음  혀 높이 · 혀 앞뒤 · 입술 둥글림 · 반모음(y/w)
근거가 검증된 설명만 한다. 복원 조음 궤적(SPARC)은 개인 단위에서 오발음을 가리지 못해(results/15 §6-2) 쓰지 않는다.

    explain("ㄲ", "ㄱ", "onset") → {"title": "경음을 평음처럼", "tips": ["목에 힘을 주고 …"], "diff": ["발성"]}
    syllabify(canon) → 음절별 자모 묶음(표시용, 발음형 한글 포함)
"""

PLACE = {**{c: "양순" for c in "ㅂㅃㅍㅁ"}, **{c: "치조" for c in "ㄷㄸㅌㄴㄹㅅㅆ"}, **{c: "경구개" for c in "ㅈㅉㅊ"},
         **{c: "연구개" for c in "ㄱㄲㅋㅇ"}, "ㅎ": "후두"}
MANNER = {**{c: "파열" for c in "ㅂㅃㅍㄷㄸㅌㄱㄲㅋ"}, **{c: "파찰" for c in "ㅈㅉㅊ"}, **{c: "마찰" for c in "ㅅㅆㅎ"},
          **{c: "비음" for c in "ㅁㄴㅇ"}, "ㄹ": "유음"}
PHON = {**{c: "평음" for c in "ㅂㄷㄱㅈㅅ"}, **{c: "경음" for c in "ㅃㄸㄲㅉㅆ"}, **{c: "격음" for c in "ㅍㅌㅋㅊ"}}
# 모음: (높이 0고~2저, 앞뒤 0전~2후, 원순, 반모음)
VOWEL = {"ㅣ": (0, 0, 0, ""), "ㅔ": (1, 0, 0, ""), "ㅐ": (1, 0, 0, ""), "ㅡ": (0, 2, 0, ""), "ㅓ": (1, 2, 0, ""), "ㅏ": (2, 1, 0, ""),
         "ㅜ": (0, 2, 1, ""), "ㅗ": (1, 2, 1, ""), "ㅟ": (0, 0, 1, "w"), "ㅚ": (1, 0, 1, "w"),
         "ㅑ": (2, 1, 0, "y"), "ㅕ": (1, 2, 0, "y"), "ㅛ": (1, 2, 1, "y"), "ㅠ": (0, 2, 1, "y"), "ㅖ": (1, 0, 0, "y"), "ㅒ": (1, 0, 0, "y"),
         "ㅘ": (2, 1, 0, "w"), "ㅝ": (1, 2, 0, "w"), "ㅙ": (1, 0, 0, "w"), "ㅞ": (1, 0, 0, "w"), "ㅢ": (0, 2, 0, "ɰ")}
SAME_FOR_NATIVES = [{"ㅐ", "ㅔ"}, {"ㅙ", "ㅞ", "ㅚ"}, {"ㅒ", "ㅖ"}]      # 현대 서울말에서 구별하지 않는 쌍 — 지적하지 않는 편이 옳다

PLACE_TIP = {"양순": "두 입술을 붙였다 떼세요", "치조": "혀끝을 윗잇몸에 대세요", "경구개": "혓바닥 앞쪽을 입천장에 넓게 대세요",
             "연구개": "혀 뒤쪽을 입천장 뒤(여린입천장)에 대세요", "후두": "목에서 숨만 내보내세요"}
MANNER_TIP = {"파열": "완전히 막았다가 한 번에 터뜨리세요", "파찰": "막았다가 좁은 틈으로 마찰을 내며 떼세요", "마찰": "막지 말고 좁은 틈으로 바람을 계속 내보내세요",
              "비음": "입은 막고 **코로** 소리를 울리세요", "유음": "혀끝을 댄 채 **혀 옆으로** 공기를 내보내세요"}
PHON_TIP = {("경음", "평음"): "목에 힘을 주고 숨 없이 짧고 단단하게 내세요. 지금은 힘이 빠져 평음처럼 들려요.",
            ("경음", "격음"): "숨(기식)이 너무 많이 나와요. 숨을 막은 채 단단하게 터뜨리세요.",
            ("격음", "평음"): "터뜨린 뒤 숨을 **강하게** 내보내세요. 손바닥에 바람이 느껴져야 해요.",
            ("격음", "경음"): "목의 힘을 풀고, 터뜨린 뒤 숨을 길게 내보내세요.",
            ("평음", "격음"): "숨을 약하게 하세요. 첫소리를 부드럽게 시작하면 됩니다.",
            ("평음", "경음"): "목의 힘을 빼고 부드럽게 시작하세요."}


def is_native_merger(canon, heard):
    return any(canon in s and heard in s for s in SAME_FOR_NATIVES)


def explain(canon, heard, pos):
    """pos: onset | coda | V. heard 는 자모 또는 '∅'(탈락). 반환 dict(title, tips[], diff[], native_merger)."""
    out = {"canon": canon, "heard": heard, "pos": pos, "diff": [], "tips": [], "native_merger": False}
    if heard == "∅":
        out["title"] = f"{canon} 소리가 들리지 않았어요"
        if pos == "coda":
            tip = {"ㄹ": "혀끝을 윗잇몸에 붙인 채로 끝내세요.", "ㄴ": "혀끝을 윗잇몸에 붙이고 코로 울리며 끝내세요.", "ㅁ": "입술을 다물고 코로 울리며 끝내세요.",
                   "ㅇ": "혀 뒤쪽을 올려 코로 울리며 끝내세요."}.get(canon, "받침을 삼키지 말고, 그 자리를 막은 채로 끝내세요.")
        elif pos == "V":
            tip = "모음을 생략하지 말고 한 박자 분명히 내세요."
        else:
            tip = {"ㅎ": "ㅎ 은 약해도 숨소리가 들려야 해요.", "ㄴ": "혀끝을 윗잇몸에 분명히 댔다 떼세요."}.get(canon, "첫소리를 분명히 내세요.")
        out["diff"] = ["탈락"]; out["tips"] = [tip]
        return out
    if canon in VOWEL and heard in VOWEL:
        if is_native_merger(canon, heard):
            out.update(title=f"{canon} 와 {heard} 는 원어민도 구별하지 않아요", native_merger=True, tips=["이 차이는 신경 쓰지 않아도 됩니다."], diff=["원어민 비구별"])
            return out
        h, b, r, g = VOWEL[canon]; h2, b2, r2, g2 = VOWEL[heard]
        if r != r2:
            out["diff"].append("입술"); out["tips"].append("입술을 **둥글게 모으세요**." if r else "입술을 둥글게 하지 말고 **양옆으로 푸세요**.")
        if h != h2:
            out["diff"].append("혀 높이"); out["tips"].append("입을 **더 벌리세요**(혀를 낮게)." if h > h2 else "입을 **덜 벌리세요**(혀를 높게).")
        if b != b2:
            out["diff"].append("혀 앞뒤"); out["tips"].append("혀를 **뒤로** 당기세요." if b > b2 else "혀를 **앞으로** 내미세요.")
        if g != g2:
            out["diff"].append("반모음"); out["tips"].append({"y": "앞에 짧은 'ㅣ' 를 붙여 시작하세요.", "w": "앞에 짧은 'ㅜ/ㅗ' 를 붙여 시작하세요.", "ɰ": "'ㅡ' 에서 'ㅣ' 로 빠르게 미끄러지세요.",
                                                         "": "앞에 다른 소리를 붙이지 말고 바로 시작하세요."}[g])
        out["title"] = f"{canon} 를 {heard} 처럼 냈어요"
        return out
    if canon in MANNER and heard in MANNER:
        if PHON.get(canon) and PHON.get(heard) and PHON[canon] != PHON[heard]:
            out["diff"].append("발성"); out["tips"].append(PHON_TIP[(PHON[canon], PHON[heard])])
        if MANNER[canon] != MANNER[heard]:
            out["diff"].append("방법"); out["tips"].append(MANNER_TIP[MANNER[canon]] + ".")
        if PLACE[canon] != PLACE[heard]:
            out["diff"].append("위치"); out["tips"].append(PLACE_TIP[PLACE[canon]] + f" (지금은 {PLACE[heard]} 소리로 들려요).")
        if MANNER[canon] == MANNER[heard] == "비음" or {canon, heard} == {"ㄴ", "ㄹ"}:
            pass
        elif not out["diff"]:
            out["tips"].append("소리는 비슷하지만 또렷하지 않았어요. 천천히 한 번 더 내 보세요.")
        kind = PHON.get(canon)
        out["title"] = (f"{kind}을 {PHON[heard]}처럼 냈어요" if out["diff"] == ["발성"] else f"{canon} 를 {heard} 처럼 냈어요")
        if pos == "coda" and "위치" not in out["diff"] and "방법" in out["diff"]:
            out["tips"].insert(0, "혀(입술) 위치는 맞아요.")
        return out
    out["title"] = f"{canon} 자리에 {heard} 가 들렸어요"
    out["diff"] = ["자음↔모음"]; out["tips"] = ["이 자리의 소리를 천천히 분리해서 다시 내 보세요."]
    return out


# ─────────────────────────── 표시용 음절 묶기 ───────────────────────────
_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")


def positions(canon):
    """자리별 onset | coda | V. 받침 ㅇ 은 항상 coda(초성 ㅇ 은 발음열에 없다)."""
    n = len(canon)
    return ["V" if x in VOWEL else ("coda" if (x == "ㅇ" or j + 1 == n or canon[j + 1] not in VOWEL) else "onset") for j, x in enumerate(canon)]


def syllabify(canon):
    """발음 자모열 → [{"text": 발음형 한글 1글자, "idx": [자리 번호...]}]. 모음 1개 = 음절 1개."""
    pos = positions(canon)
    syl, cur = [], None
    for j, (x, p) in enumerate(zip(canon, pos)):
        if p == "onset" or (p == "V" and (cur is None or cur["v"] is not None)):
            cur = {"o": None, "v": None, "c": None, "idx": []}; syl.append(cur)
        if cur is None:
            cur = {"o": None, "v": None, "c": None, "idx": []}; syl.append(cur)
        cur["idx"].append(j)
        if p == "onset":
            cur["o"] = x
        elif p == "V":
            cur["v"] = x
        elif cur["c"] is None:
            cur["c"] = x
    out = []
    for s in syl:
        if s["v"] in _JUNG:
            o = _CHO.index(s["o"]) if s["o"] and s["o"] in _CHO else _CHO.index("ㅇ")      # 초성 없는 음절은 ㅇ 으로 적는다
            c = _JONG.index(s["c"]) if s["c"] and s["c"] in _JONG else 0
            text = chr(0xAC00 + (o * 21 + _JUNG.index(s["v"])) * 28 + c)
        else:
            text = "".join(canon[j] for j in s["idx"])
        out.append({"text": text, "idx": s["idx"]})
    return out
