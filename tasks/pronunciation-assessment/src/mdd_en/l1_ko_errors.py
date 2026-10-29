# -*- coding: utf-8 -*-
"""한국인 영어 학습자의 대표 발음 오류 — 합성 학습 데이터(일부러 틀리게 읽힌 TTS)를 만드는 규칙.

왜: 공개 음소 인식기와 AIHub 037(자동 라벨)로 학습한 모델은 문맥으로 단어를 짐작해 **사전 발음을 적는다**
(think 문맥이면 /s/ 로 읽어도 TH). "문맥은 think 인데 소리는 sink" 인 예를 정답과 함께 보여 줘서 들린 소리를
적게 만든다. 규칙은 음성학 문헌의 한국어 L1 전이 오류 목록(평가셋 L2-ARCTIC 으로 맞추지 않았다).

    words = [{"w": "think", "ph": ["TH", "IH1", "NG", "K"]}, …]        # mdd_en.g2p 결과(강세 포함)
    new_words, errs = apply_errors(words, rng, rate=1.0)
    text = to_kokoro(new_words)                                         # Kokoro(misaki US) 음소 문자열
    label = [strip(p) for w in new_words for p in w["ph"]]              # 실제로 읽힌 소리 = CTC 정답

규칙마다 적용 자리 조건(단어 첫소리 · 모음 앞 · 강세 모음 · 단어 끝)과 기본 확률이 있다.
한 자리에는 규칙 하나만. rate 로 발화 단위 오류 밀도를 조절한다(0 = 표준대로).
"""

import re

VOWEL = set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
STOPS_FRIC = set("P B T D K G CH JH S Z SH F V TH DH".split())

# (이름, 표준, 바꿀 소리 또는 None(탈락) 또는 "+AH0"(뒤에 모음 삽입), 조건, 기본 확률)
RULES = [
    ("dh_d", "DH", "D", "any", 0.40),
    ("z_s", "Z", "S", "not_initial", 0.35),
    ("z_jh", "Z", "JH", "word_initial", 0.30),
    ("th_s", "TH", "S", "any", 0.25),
    ("th_t", "TH", "T", "any", 0.12),
    ("f_p", "F", "P", "any", 0.30),
    ("v_b", "V", "B", "any", 0.30),
    ("r_l", "R", "L", "prevocalic", 0.15),
    ("l_r", "L", "R", "prevocalic", 0.10),
    ("s_sh", "S", "SH", "before_high_front", 0.25),
    ("d_t", "D", "T", "word_initial", 0.10),
    ("b_p", "B", "P", "word_initial", 0.10),
    ("g_k", "G", "K", "word_initial", 0.10),
    ("ih_iy", "IH", "IY", "stressed", 0.25),
    ("iy_ih", "IY", "IH", "stressed", 0.12),
    ("eh_ae", "EH", "AE", "stressed", 0.15),
    ("ae_eh", "AE", "EH", "stressed", 0.15),
    ("ow_ao", "OW", "AO", "any", 0.20),
    ("uh_uw", "UH", "UW", "any", 0.20),
    ("er_ah", "ER", "AH", "any", 0.10),
    ("final_del", "T", None, "word_final_after_cons", 0.10),
    ("final_del_d", "D", None, "word_final_after_cons", 0.10),
    ("epenthesis", "*", "+AH0", "word_final_stop_fric", 0.10),
]

# ARPAbet(강세 숫자) → Kokoro(misaki 미국식) 음소
K_CONS = {"B": "b", "CH": "ʧ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "ʤ", "K": "k", "L": "l",
          "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
          "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"}
K_VOW = {"AA": "ɑ", "AE": "æ", "AO": "ɔ", "AW": "W", "AY": "I", "EH": "ɛ", "EY": "A", "IH": "ɪ", "IY": "i",
         "OW": "O", "OY": "Y", "UH": "ʊ", "UW": "u"}


def strip(p):
    return re.sub(r"\d", "", p)


def stress_of(p):
    m = re.search(r"\d", p)
    return int(m.group(0)) if m else None


def _ok(cond, ph, i):
    """규칙 조건 — ph 는 단어 하나의 음소(강세 포함), i 는 자리."""
    base = [strip(x) for x in ph]
    nxt = base[i + 1] if i + 1 < len(base) else None
    if cond == "any":
        return True
    if cond == "word_initial":
        return i == 0
    if cond == "not_initial":
        return i > 0
    if cond == "prevocalic":
        return nxt in VOWEL
    if cond == "before_high_front":
        return nxt in ("IY", "IH")
    if cond == "stressed":
        return stress_of(ph[i]) in (1, 2)
    if cond == "word_final_after_cons":
        return i == len(base) - 1 and i > 0 and base[i - 1] not in VOWEL
    if cond == "word_final_stop_fric":
        return i == len(base) - 1 and base[i] in STOPS_FRIC
    return False


def apply_errors(words, rng, rate=1.0):
    """→ (오류를 넣은 단어 목록, [(단어 번호, 자리, 규칙 이름, 표준, 바뀐 소리)]). 원본은 바꾸지 않는다."""
    out, errs = [], []
    for wi, w in enumerate(words):
        ph = list(w["ph"])
        new = []
        for i, p in enumerate(ph):
            b = strip(p)
            cands = [r for r in RULES if (r[1] == b or (r[1] == "*" and r[2] == "+AH0")) and _ok(r[3], ph, i)]
            done = False
            rng.shuffle(cands)
            for name, c, sub, cond, prob in cands:
                if rng.random() < min(1.0, prob * rate):
                    if sub is None:                                  # 탈락
                        errs.append((wi, i, name, b, None))
                    elif sub == "+AH0":                              # 뒤에 약모음 삽입(book → 부커)
                        new.append(p)
                        new.append("AH0")
                        errs.append((wi, i, name, b, "+AH"))
                    else:
                        s = stress_of(p)
                        new.append(sub + (str(s) if (sub in VOWEL and s is not None) else ""))
                        errs.append((wi, i, name, b, sub))
                    done = True
                    break
            if not done:
                new.append(p)
        out.append({"w": w["w"], "ph": new})
    return out, errs


def to_kokoro(words, final="."):
    """단어 목록(ARPAbet+강세) → Kokoro 음소 문자열. 강세 표시는 모음 바로 앞(ˈ 1 · ˌ 2)."""
    toks = []
    for w in words:
        s = ""
        for p in w["ph"]:
            b, st = strip(p), stress_of(p)
            if b in K_CONS:
                s += K_CONS[b]
                continue
            mark = "ˈ" if st == 1 else ("ˌ" if st == 2 else "")
            if b == "AH":
                s += mark + ("ʌ" if st in (1, 2) else "ə")
            elif b == "ER":
                s += mark + ("ɜɹ" if st in (1, 2) else "ɚ")
            elif b in K_VOW:
                s += mark + K_VOW[b]
        if s:
            toks.append(s)
    return " ".join(toks) + (final or "")
