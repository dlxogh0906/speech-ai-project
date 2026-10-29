# -*- coding: utf-8 -*-
"""표준발음 보정(`mdd.canonical_fix`) 회귀 점검 — 시험 224문항 **밖** 문장으로, 보정이 맞는 발음을 망가뜨리지 않는지 본다.

점검표 `evalsets/g2p_regression_v1.tsv`(id · kind · text · expected · note)
  guard  KoG2P 가 원래 맞는 것 — 보정 뒤에도 그대로여야 한다(동료[동뇨] · 서울 거예요 · 줄거리 …)
  fix    KoG2P 가 틀리는 것 — 보정 뒤에 맞아야 한다(먹을 수[머글쑤] · 필요[피료] · 희망[히망] …)
  known  알고 놓치는 것(정밀도를 위해 감수) — 실패해도 실패로 세지 않고 따로 보여 준다
expected = 발음대로 쓴 한글(띄어쓰기 무시, 허용 발음은 | 로 여럿). 자모로 풀어 비교한다(첫소리 ㅇ 은 없앰, ㅙ·ㅞ 는 둘로 —
발음 API 표기와 같다). 비교 대상은 결정적 보정 결과(`CanonicalBuilder.fixed`) — 허용 변이는 녹음이 있어야 고르므로 여기서 보지 않는다.

    $PY .../check_canonical_fix.py            # 실패만 자세히, 끝에 종류별 합계 · 실패가 있으면 종료 코드 1
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import G2P_TOOLS, ensure_kog2p

import argparse, collections, csv, re

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ",
        "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
SPLIT = {"ㅙ": "ㅗㅐ", "ㅞ": "ㅜㅔ"}


def to_jamo(pron):
    """발음대로 쓴 한글 → 발음 API 자모열(첫소리 ㅇ 없음 · ㅙ·ㅞ 둘로)."""
    out = []
    for ch in re.sub(r"[^가-힣]", "", pron):
        o = ord(ch) - 0xAC00
        on, nu, co = CHO[o // 588], JUNG[(o % 588) // 28], JONG[o % 28]
        if on != "ㅇ":
            out.append(on)
        out.append(SPLIT.get(nu, nu))
        if co:
            out.append(co)
    return "".join(out)


def to_hangul(jamo):
    """자모열 → 읽기용 한글(실패 보고용 · 받침 ㅇ 과 첫소리 ㅇ 이 겹치는 곳은 대충)."""
    j, out, i = list(jamo), [], 0
    V = set(JUNG)
    while i < len(j):
        on = "ㅇ"
        if j[i] not in V:
            if i + 1 < len(j) and j[i + 1] in V:
                on = j[i]; i += 1
            else:
                out.append(j[i]); i += 1; continue
        v = j[i]; i += 1
        if v == "ㅗ" and i < len(j) and j[i] == "ㅐ":
            v = "ㅙ"; i += 1
        elif v == "ㅜ" and i < len(j) and j[i] == "ㅔ":
            v = "ㅞ"; i += 1
        co = ""
        if i < len(j) and j[i] not in V and not (i + 1 < len(j) and j[i + 1] in V) and j[i] in JONG:
            co = j[i]; i += 1
        out.append(chr(0xAC00 + CHO.index(on) * 588 + JUNG.index(v) * 28 + JONG.index(co)))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(_r / "tasks" / "pronunciation-assessment" / "evalsets" / "g2p_regression_v1.tsv"))
    ap.add_argument("--all", action="store_true", help="통과한 것도 다 보여 준다")
    a = ap.parse_args()
    ensure_kog2p()
    argv, _sys.argv = _sys.argv, _sys.argv[:1]
    try:
        from g2p import readRules, graph2prono, ver_info
    finally:
        _sys.argv = argv
    from mdd.jamo_to_kog2p import phones_to_jamo, INV
    from mdd.canonical_fix import CanonicalBuilder
    ri, ro = readRules(ver_info[0], str(G2P_TOOLS / "KoG2P" / "rulebook.txt"))

    def kog2p(s):                                    # = 발음 API canonical()
        text = re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z가-힣\s]", " ", s or "")).strip()
        out = []
        for j in phones_to_jamo([p for p in graph2prono(text, ri, ro).split() if p in INV]):
            out += {"ㅙ": ["ㅗ", "ㅐ"], "ㅞ": ["ㅜ", "ㅔ"]}.get(j, [j])
        return out
    B = CanonicalBuilder(kog2p)

    rows = list(csv.DictReader(open(a.cases, encoding="utf-8"), delimiter="\t"))
    cnt = collections.Counter()
    fails = []
    for r in rows:
        ok = set(to_jamo(e) for e in r["expected"].split("|"))
        api = "".join(kog2p(r["text"]))
        fx = "".join(B.fixed(r["text"]))
        k = r["kind"]
        cnt[k + "_n"] += 1
        cnt[k + "_api_ok"] += api in ok
        cnt[k + "_fix_ok"] += fx in ok
        if fx not in ok or a.all:
            fails.append((r, api, fx, fx in ok))
    for r, api, fx, good in fails:
        mark = "통과" if good else ("알고 놓침" if r["kind"] == "known" else "실패")
        print("[%s] %s %-6s %s\n      기대 [%s] · KoG2P [%s] · 보정 [%s]%s" % (
            mark, r["id"], r["kind"], r["text"], r["expected"], to_hangul(api), to_hangul(fx),
            ("  — " + r["note"]) if r["note"] else ""))
    print("\n| 종류 | 문장 | KoG2P 맞음 | 보정 뒤 맞음 |\n|---|---:|---:|---:|")
    for k in ("guard", "fix", "known"):
        if cnt[k + "_n"]:
            print("| %s | %d | %d | %d |" % (k, cnt[k + "_n"], cnt[k + "_api_ok"], cnt[k + "_fix_ok"]))
    bad = (cnt["guard_n"] - cnt["guard_fix_ok"]) + (cnt["fix_n"] - cnt["fix_fix_ok"])
    print("\n실패(guard·fix) %d건" % bad)
    _sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
