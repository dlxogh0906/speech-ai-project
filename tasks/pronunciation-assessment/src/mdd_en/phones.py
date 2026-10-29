# -*- coding: utf-8 -*-
"""영어 음소 체계 — ARPAbet 39(CMUdict) 를 기준으로, 모델마다 다른 출력 기호를 여기로 옮긴다.

모델 출력 기호 → ARPAbet 은 **여러 음소가 될 수 있다**(espeak 'ɑːɹ' → AA R, 'aɪɚ' → AY ER).
ARPAbet 에 없는 소리는 따로 이름을 둔다 — 표준발음과 절대 같아지지 않으므로 그대로 "다른 소리"로 잡힌다.
    DX  탄설음 [ɾ] (water 의 t)          → T·D 자리면 맞은 것으로 본다(미국식 발음)
    Q   성문 파열음 [ʔ] (button 의 t)     → T 자리면 맞은 것으로 본다
    EU  한국어 ㅡ 같은 끼워 넣은 모음(slplab 'eu')

"맞은 것으로 볼 짝"(`EQUIV`)은 미국 영어에서 둘 다 표준으로 통하는 경우만 둔다.
"""

ARPA39 = ("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH "
          "UH UW V W Y Z ZH").split()
VOWELS = set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())

# espeak-ng IPA(facebook/wav2vec2-*-espeak-cv-ft) → ARPAbet. 여기 없는 기호는 영어가 아니다 → 디코딩에서 막는다.
ESPEAK_EN = {
    "n": "N", "s": "S", "t": "T", "ə": "AH", "l": "L", "a": "AA", "i": "IY", "k": "K", "d": "D", "m": "M",
    "ɛ": "EH", "ɾ": "DX", "e": "EY", "ɪ": "IH", "p": "P", "o": "OW", "ɐ": "AH", "z": "Z", "ð": "DH",
    "f": "F", "j": "Y", "v": "V", "b": "B", "ɹ": "R", "r": "R", "ʊ": "UH", "iː": "IY", "w": "W", "ʌ": "AH",
    "u": "UW", "ɡ": "G", "æ": "AE", "aɪ": "AY", "ʃ": "SH", "h": "HH", "ɔ": "AO", "ɑː": "AA", "ŋ": "NG",
    "ɚ": "ER", "eɪ": "EY", "uː": "UW", "oʊ": "OW", "ᵻ": "IH", "eː": "EY", "θ": "TH", "aʊ": "AW",
    "oː": "OW", "ɜ": "ER", "ɑ": "AA", "dʒ": "JH", "əl": "AH L", "ɜː": "ER", "ʒ": "ZH", "tʃ": "CH",
    "ɔː": "AO", "ɑːɹ": "AA R", "ɔːɹ": "AO R", "aː": "AA", "oːɹ": "AO R", "ʔ": "Q", "iə": "IY AH",
    "ɛɹ": "EH R", "ɪɹ": "IH R", "ɔɪ": "OY", "ʊɹ": "UH R", "aɪɚ": "AY ER", "aɪə": "AY AH", "n̩": "AH N",
    "l̩": "AH L", "ɒ": "AA", "ɫ": "L", "əʊ": "OW", "eə": "EH R", "ʊə": "UH R", "ɛː": "EH", "ɪː": "IH",
    "ts": "T S", "ju": "Y UW", "ɑ:": "AA", "u:": "UW", "i:": "IY", "o:": "OW", "ɛɪ": "EY", "oɪ": "OY",
    "ər1": "ER", "ər2": "ER", "ər4": "ER", "ər5": "ER", "ərɜ": "ER", "au": "AW", "ai": "AY",
}

# TIMIT 39 묶음(HuPER 등): AO 가 없다(AA 로 합쳐짐) · DX 가 있다
TIMIT39_FOLD = {"AO": "AA"}

# slplab(한국인 학습자 영어): ARPAbet + '_err' 쌍둥이 + 한국어식 소리.
# 'x_err' 는 "x 자리를 틀리게 냈다"는 뜻이라 'X*'(표준과 절대 같지 않은 기호)로 둔다.
# 한국어식 소리(ㅡ 삽입 eu · 단모음 o · ts · 된소리 b*/d*/g*)도 영어 표준과 다른 소리로 센다.
SLPLAB_EXTRA = {"eu": "EU", "o": "O*", "ts": "TS", "ax": "AH", "b*": "B*", "d*": "D*", "g*": "G*"}

# 미국 영어에서 둘 다 표준으로 통하는 짝 — 이 경우는 틀린 소리로 세지 않는다
EQUIV = {("T", "DX"), ("D", "DX"), ("T", "Q"), ("AO", "AA"), ("AH", "IH"), ("IH", "IY"), ("ER", "AH")}
# ↑ ("AH","IH")·("ER","AH")·("IH","IY") 는 **강세 없는 모음**(약모음)에서만 쓴다 — `same_sound` 참고.
#   강세 있는 IH→IY(ship→sheep, live→leave)는 한국인의 대표 오류라 봐주지 않는다.
WEAK_ONLY = {("AH", "IH"), ("ER", "AH"), ("IH", "IY")}


def same_sound(canon, actual, stressed=True):
    """표준 음소(강세 뗀 것)와 들린 음소가 같은 소리로 통하는가."""
    if canon == actual:
        return True
    if (canon, actual) in EQUIV:
        return stressed is False or (canon, actual) not in WEAK_ONLY
    return False


def token_map(vocab, kind):
    """모델 어휘 {토큰: id} → {id: [ARPAbet …]}. 영어가 아닌 토큰은 빼고(디코딩에서 막는다), 특수 토큰도 뺀다.

    kind: espeak · arpa39(TIMIT 묶음, HuPER) · arpa(CMU 39, charsiu) · slplab
    """
    out = {}
    for tok, i in vocab.items():
        if tok.startswith("<") or tok.startswith("[") or tok in ("|", " "):
            continue
        if kind == "espeak":
            ph = ESPEAK_EN.get(tok)
        elif kind == "slplab":
            err = tok.endswith("_err")
            t = tok[:-4] if err else tok
            ph = SLPLAB_EXTRA.get(t) or (t.upper() if t.upper() in ARPA39 else None)
            if ph and err:
                ph = " ".join(p.rstrip("*") + "*" for p in ph.split())
        else:
            t = tok.upper()
            ph = t if (t in ARPA39 or t == "DX") else None
        if ph:
            out[i] = ph.split()
    return out


def fold(phones, kind):
    """모델이 구별하지 못하는 소리는 표준발음 쪽도 같이 묶는다(TIMIT 39 의 AO→AA)."""
    if kind == "arpa39":
        return [TIMIT39_FOLD.get(p, p) for p in phones]
    return list(phones)

# 화면 표시용 IPA(미국 영어). AH·ER 은 강세에 따라 다르게 적는다.
ARPA2IPA = {"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d",
            "DH": "ð", "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
            "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ", "P": "p",
            "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w", "Y": "j",
            "Z": "z", "ZH": "ʒ", "DX": "ɾ", "Q": "ʔ", "EU": "ɨ"}


def to_ipa(p, stressed=None):
    if p is None:
        return None
    if p == "AH" and stressed is False:
        return "ə"
    if p == "ER" and stressed is False:
        return "ɚ"
    return ARPA2IPA.get(p.rstrip("*"), p.lower())


# AIHub 037(교육용 한국인의 영어 음성) 라벨의 IPA(한 글자씩, 이중모음을 쪼개 적는다) → ARPAbet 39.
# 실측(2026-10-05): a·o·e 뒤는 ɪ/ʊ 가 압도적 — 이중모음으로 묶는다. 'ə r' 는 CMUdict 처럼 ER 하나로.
AIHUB037_PAIR = {("a", "ɪ"): "AY", ("a", "ʊ"): "AW", ("o", "ʊ"): "OW", ("e", "ɪ"): "EY", ("ɔ", "ɪ"): "OY",
                 ("ə", "r"): "ER"}
AIHUB037_ONE = {"ə": "AH", "ʌ": "AH", "ɪ": "IH", "i": "IY", "u": "UW", "ʊ": "UH", "ɛ": "EH", "æ": "AE", "ɑ": "AA",
                "ɔ": "AO", "a": "AA", "o": "OW", "e": "EH", "r": "R", "n": "N", "t": "T", "s": "S", "d": "D", "l": "L",
                "m": "M", "k": "K", "z": "Z", "ð": "DH", "p": "P", "b": "B", "v": "V", "w": "W", "f": "F", "h": "HH",
                "ŋ": "NG", "ʃ": "SH", "j": "Y", "g": "G", "ɡ": "G", "ʤ": "JH", "θ": "TH", "ʧ": "CH", "ʒ": "ZH"}


def aihub037_to_arpa(tokens):
    """037 의 phonemeText/transAnsw 토큰 → ARPAbet. '***'(빈자리)·'null' 은 버린다. 모르는 기호는 None 으로 알린다.

    단독 'o' 는 뒤가 r 이면 AO(more = M AO R, CMUdict 관례), 아니면 OW.
    """
    t = [x for x in tokens if x not in ("***", "null")]
    out, i = [], 0
    while i < len(t):
        pair = (t[i], t[i + 1]) if i + 1 < len(t) else None
        if pair in AIHUB037_PAIR:
            out.append(AIHUB037_PAIR[pair]); i += 2
            continue
        if t[i] == "o" and i + 1 < len(t) and t[i + 1] == "r":
            out.append("AO"); i += 1
            continue
        p = AIHUB037_ONE.get(t[i])
        if p is None:
            return None
        out.append(p); i += 1
    return out
