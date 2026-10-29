# -*- coding: utf-8 -*-
"""부스 문장(booth_sentences.json) 에 표준 발음(pron)과 발음 안내(guide: 로마자·키릴·가타카나)를 채운다.
    $PY deploy/voice-collect/make_booth_guides.py
단어별 KoG2P(문장 전체로 돌리면 단어 경계를 넘는 연음·ㄴ첨가가 잘못 들어간다) + FIX 수동 교정 + MERGE(이어 읽는 말).
"""
import sys, pathlib, json, re
R = pathlib.Path(__file__).resolve().parents[2]   # 저장소 루트
sys.path[:0] = [str(R), *(str(p) for p in sorted(R.glob("tasks/*/src")))]
from shared.paths import G2P_TOOLS, ensure_kog2p
from mdd.jamo_to_kog2p import phones_to_jamo, INV, VOWEL
ensure_kog2p()
_argv, sys.argv = sys.argv, sys.argv[:1]
from g2p import readRules, graph2prono, ver_info
sys.argv = _argv
ri, ro = readRules(ver_info[0], f"{G2P_TOOLS}/KoG2P/rulebook.txt")
CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"); JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"); JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
V = set(JUNG)
def syllabify(jamo):
    vi = [i for i, j in enumerate(jamo) if j in V]; syl = []
    for k, i in enumerate(vi):
        prev_end = vi[k - 1] + 1 if k else 0
        cons = jamo[prev_end:i]
        onset = ""
        if k == 0:
            onset = cons[-1] if cons else ""
        else:
            if len(cons) == 1:
                if cons[0] == "ㅇ": syl[-1][2] = "ㅇ"
                else: onset = cons[0]
            elif len(cons) == 2:
                syl[-1][2] = cons[0]; onset = cons[1]
            elif len(cons) > 2:
                raise ValueError(f"자음 {cons}")
        syl.append([onset, jamo[i], ""])
    tail = jamo[vi[-1] + 1:]
    if len(tail) == 1: syl[-1][2] = tail[0]
    elif len(tail) > 1: raise ValueError(f"끝 자음 {tail}")
    return syl
def compose(o, v, c):
    return chr(0xAC00 + (CHO.index(o if o else "ㅇ") * 21 + JUNG.index(v)) * 28 + JONG.index(c))
def pron_words(sentence):
    phs = [p for p in graph2prono(sentence, ri, ro).split() if p in INV]
    syl = syllabify(phones_to_jamo(phs))
    words = sentence.split(); counts = [len(re.findall(r"[가-힣]", w)) for w in words]
    assert sum(counts) == len(syl), (sentence, sum(counts), len(syl))
    out, k = [], 0
    for w, n in zip(words, counts):
        punct = re.sub(r"[가-힣]", "", w)
        out.append(("".join(compose(*s) for s in syl[k:k + n]), syl[k:k + n], punct)); k += n
    return out

P = str(R / "deploy/voice-collect/booth_sentences.json")
FIX = {  # (세트, 문장): {발음 단어: 교정} — KoG2P 가 놓치거나 틀리는 것
    (1, 5): {"친구의": "친구에"}, (2, 9): {"펴늬저미": "펴니저미"}, (2, 10): {"녀니": "려니"}, (3, 2): {"거예요.": "꺼예요."},
    (3, 6): {"시대의": "시대에"}, (4, 2): {"열심히": "열씸히"}, (4, 3): {"놓쳐써요.": "녿처써요."}, (4, 4): {"깜짝": "깜짱"},
    (4, 9): {"다른": "따른"}, (5, 3): {"십": "심"}, (5, 5): {"선쌩님께": "선생님께"}, (5, 8): {"선쌩니미": "선생니미"},
}
MERGE = {(2, 10): [("일", "려니")], (5, 3): [("심", "년")], (4, 9): [("옫", "따른")]}   # 이어 읽는 말은 한 덩어리로
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"; JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
NEUT = {"ㄲ": "ㄱ", "ㅋ": "ㄱ", "ㅅ": "ㄷ", "ㅆ": "ㄷ", "ㅈ": "ㄷ", "ㅊ": "ㄷ", "ㅌ": "ㄷ", "ㅎ": "ㄷ", "ㅍ": "ㅂ"}
def dec(ch):
    c = ord(ch) - 0xAC00; o, v, k = CHO[c // 588], JUNG[(c % 588) // 28], JONG[c % 28]
    if o in "ㅈㅉㅊ" and v == "ㅕ": v = "ㅓ"            # 져·쪄·쳐 → [저·쩌·처]
    return o, v, NEUT.get(k, k)
def norm_word(w):
    out = ""
    for ch in w:
        o, v, k = dec(ch); out += chr(0xAC00 + (CHO.index(o) * 21 + JUNG.index(v)) * 28 + JONG.index(k))
    return out
R_ON = dict(zip(CHO, ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]))
R_V = dict(zip(JUNG, ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]))
R_CO = {"": "", "ㄱ": "k", "ㄴ": "n", "ㄷ": "t", "ㄹ": "l", "ㅁ": "m", "ㅂ": "p", "ㅇ": "ng"}
def roman(word):
    syl = [dec(c) for c in word]; out = []
    for i, (o, v, k) in enumerate(syl):
        on = "l" if (o == "ㄹ" and i and syl[i - 1][2] == "ㄹ") else R_ON[o]
        out.append(on + R_V[v] + R_CO[k])
    return "-".join(out)
C_ON = {"ㄱ": ("к", "г"), "ㄲ": ("кк", "кк"), "ㄴ": ("н", "н"), "ㄷ": ("т", "д"), "ㄸ": ("тт", "тт"), "ㄹ": ("р", "р"), "ㅁ": ("м", "м"),
        "ㅂ": ("п", "б"), "ㅃ": ("пп", "пп"), "ㅅ": ("с", "с"), "ㅆ": ("сс", "сс"), "ㅇ": ("", ""), "ㅈ": ("ч", "дж"), "ㅉ": ("чч", "чч"),
        "ㅊ": ("чх", "чх"), "ㅋ": ("кх", "кх"), "ㅌ": ("тх", "тх"), "ㅍ": ("пх", "пх"), "ㅎ": ("х", "х")}
C_V = dict(zip(JUNG, ["а", "э", "я", "йе", "о", "е", "ё", "е", "о", "ва", "вэ", "вэ", "ё", "у", "во", "вэ", "ви", "ю", "ы", "ый", "и"]))
C_CO = {"": "", "ㄱ": "к", "ㄴ": "н", "ㄷ": "т", "ㄹ": "ль", "ㅁ": "м", "ㅂ": "п", "ㅇ": "н"}
def cyr(word):
    syl = [dec(c) for c in word]; out = []
    for i, (o, v, k) in enumerate(syl):
        if o == "ㄹ" and i and syl[i - 1][2] == "ㄹ": on = "л"
        else: on = C_ON[o][0 if i == 0 else 1]
        vv = C_V[v]
        if o == "ㅇ" and v == "ㅔ": vv = "э"
        if o == "ㅇ" and v in "ㅒㅖ": vv = "йе"
        co = C_CO[k]
        if k == "ㄹ" and i + 1 < len(syl) and syl[i + 1][0] == "ㄹ": co = "л"
        out.append(on + vv + co)
    return "-".join(out)
K_ON = {"ㄱ": ("k", "g"), "ㄲ": ("k", "k"), "ㅋ": ("k", "k"), "ㄴ": ("n", "n"), "ㄷ": ("t", "d"), "ㄸ": ("t", "t"), "ㅌ": ("t", "t"),
        "ㄹ": ("r", "r"), "ㅁ": ("m", "m"), "ㅂ": ("p", "b"), "ㅃ": ("p", "p"), "ㅍ": ("p", "p"), "ㅅ": ("s", "s"), "ㅆ": ("s", "s"),
        "ㅇ": ("", ""), "ㅈ": ("ch", "j"), "ㅉ": ("ch", "ch"), "ㅊ": ("ch", "ch"), "ㅎ": ("h", "h")}
K_V = dict(zip(JUNG, ["a", "e", "ya", "ye", "o", "e", "yo", "ye", "o", "wa", "we", "we", "yo", "u", "wo", "we", "wi", "yu", "u", "ui", "i"]))
BASE = {  # 자음행 × 모음 → 가타카나
    "": dict(a="ア", i="イ", u="ウ", e="エ", o="オ", ya="ヤ", yu="ユ", yo="ヨ", ye="イェ", wa="ワ", wi="ウィ", we="ウェ", wo="ウォ", ui="ウイ"),
    "k": dict(a="カ", i="キ", u="ク", e="ケ", o="コ", ya="キャ", yu="キュ", yo="キョ", ye="キェ", wa="クァ", wi="クィ", we="クェ", wo="クォ", ui="クイ"),
    "g": dict(a="ガ", i="ギ", u="グ", e="ゲ", o="ゴ", ya="ギャ", yu="ギュ", yo="ギョ", ye="ギェ", wa="グァ", wi="グィ", we="グェ", wo="グォ", ui="グイ"),
    "s": dict(a="サ", i="シ", u="ス", e="セ", o="ソ", ya="シャ", yu="シュ", yo="ショ", ye="シェ", wa="スァ", wi="スィ", we="スェ", wo="スォ", ui="スイ"),
    "t": dict(a="タ", i="ティ", u="トゥ", e="テ", o="ト", ya="ティャ", yu="テュ", yo="ティョ", ye="ティェ", wa="トァ", wi="トゥィ", we="トゥェ", wo="トォ", ui="ティ"),
    "d": dict(a="ダ", i="ディ", u="ドゥ", e="デ", o="ド", ya="ディャ", yu="デュ", yo="ディョ", ye="ディェ", wa="ドァ", wi="ドゥィ", we="ドゥェ", wo="ドォ", ui="ディ"),
    "n": dict(a="ナ", i="ニ", u="ヌ", e="ネ", o="ノ", ya="ニャ", yu="ニュ", yo="ニョ", ye="ニェ", wa="ヌァ", wi="ヌィ", we="ヌェ", wo="ヌォ", ui="ヌイ"),
    "h": dict(a="ハ", i="ヒ", u="フ", e="ヘ", o="ホ", ya="ヒャ", yu="ヒュ", yo="ヒョ", ye="ヒェ", wa="ファ", wi="フィ", we="フェ", wo="フォ", ui="フイ"),
    "p": dict(a="パ", i="ピ", u="プ", e="ペ", o="ポ", ya="ピャ", yu="ピュ", yo="ピョ", ye="ピェ", wa="プァ", wi="プィ", we="プェ", wo="プォ", ui="プイ"),
    "b": dict(a="バ", i="ビ", u="ブ", e="ベ", o="ボ", ya="ビャ", yu="ビュ", yo="ビョ", ye="ビェ", wa="ブァ", wi="ブィ", we="ブェ", wo="ブォ", ui="ブイ"),
    "m": dict(a="マ", i="ミ", u="ム", e="メ", o="モ", ya="ミャ", yu="ミュ", yo="ミョ", ye="ミェ", wa="ムァ", wi="ムィ", we="ムェ", wo="ムォ", ui="ムイ"),
    "r": dict(a="ラ", i="リ", u="ル", e="レ", o="ロ", ya="リャ", yu="リュ", yo="リョ", ye="リェ", wa="ルァ", wi="ルィ", we="ルェ", wo="ルォ", ui="ルイ"),
    "ch": dict(a="チャ", i="チ", u="チュ", e="チェ", o="チョ", ya="チャ", yu="チュ", yo="チョ", ye="チェ", wa="チュァ", wi="チュィ", we="チュェ", wo="チュォ", ui="チュイ"),
    "j": dict(a="ジャ", i="ジ", u="ジュ", e="ジェ", o="ジョ", ya="ジャ", yu="ジュ", yo="ジョ", ye="ジェ", wa="ジュァ", wi="ジュィ", we="ジュェ", wo="ジュォ", ui="ジュイ"),
}
TENSE = set("ㄲㄸㅃㅆㅉ"); OBSTRUENT = set("ㄱㄲㅋㄷㄸㅌㅂㅃㅍㅅㅆㅈㅉㅊ")
def kana(word):
    syl = [dec(c) for c in word]; out = []
    for i, (o, v, k) in enumerate(syl):
        s = ""
        if o in TENSE and i and syl[i - 1][2] == "": s += "ッ"          # 빠 → ッパ
        s += BASE[K_ON[o][0 if i == 0 else 1]][K_V[v]]
        nxt = syl[i + 1][0] if i + 1 < len(syl) else ""
        if k in ("ㄴ", "ㅇ"): s += "ン"
        elif k == "ㅁ": s += "ム"                                # 작은 가나(ㇺ·ㇽ·ㇰ)는 기기 글꼴에서 깨질 수 있어 일반 가나로
        elif k == "ㄹ": s += "ル"
        elif k in ("ㄱ", "ㄷ", "ㅂ"):
            s += "ッ" if nxt in OBSTRUENT else {"ㄱ": "ク", "ㄷ": "ッ", "ㅂ": "プ"}[k]
        out.append(s)
    return "".join(out)
D = json.load(open(P, encoding="utf-8"))
for _st in D["sets"]:
    for _it in _st["items"]: _it.pop("pron", None); _it.pop("guide", None)
D["note"] = D["note"].split(" pron = ")[0]
for st in D["sets"]:
    for it in st["items"]:
        words = pron_words_per_word(it["text"]); fx = FIX.get((st["id"], it["id"]), {})
        pw = []
        for w, _, p in words:
            key = w + p
            if key in fx: w = re.sub(r"[^가-힣]", "", fx[key])
            elif w in fx: w = fx[w]
            pw.append((w, p))
        for x, y in MERGE.get((st["id"], it["id"]), []):
            for i in range(len(pw) - 1):
                if pw[i][0] == x and pw[i + 1][0] == y and not pw[i][1]:
                    pw[i:i + 2] = [(x + y, pw[i + 1][1])]; break
        pw = [(norm_word(w), p) for w, p in pw]
        it["pron"] = " ".join(w + p for w, p in pw)
        it["guide"] = {"roman": " ".join(roman(w) + p for w, p in pw), "cyrl": " ".join(cyr(w) + p for w, p in pw),
                       "kana": "　".join(kana(w) + p.replace(",", "、").replace(".", "。").replace("?", "？") for w, p in pw)}
        print(f"{st['id']}-{it['id']:02d} {it['text']}\n   [{it['pron']}]\n   {it['guide']['roman']}\n   {it['guide']['cyrl']}\n   {it['guide']['kana']}")
D["note"] += " pron = 표준 발음(단어별 KoG2P + 교정), guide = 발음 안내(로마자·키릴·가타카나, 발음 기준)."
json.dump(D, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
