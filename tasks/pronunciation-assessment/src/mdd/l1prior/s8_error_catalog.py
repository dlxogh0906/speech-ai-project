# -*- coding: utf-8 -*-
"""
s8_error_catalog.py
~~~~~~~~~~~~~~~~~~~
KoG2P MDD 로 나온 **국적(L1)별 발음오류 패턴을 음소 수준까지 상세 정리**한다.
"어떤 발음오류(예: 초성 k0(ㄱ)→kh(ㅋ))가 나타나는지"를 위치(초성/중성/종성)·
구체 음소·오류유형·빈도로 명확히 카탈로그화.

입력(Speech_lab MDD 산출물, 기존):
  _mdd_analysis/<tag>/substitution_pairs.csv   L1 × target>actual × count (op=sub)
  _mdd_analysis/<tag>/l1_summary.csv           L1별 usable/오류수
  _mdd_analysis/<tag>/l1_errortype_counts.csv  L1 × error_type (삭제/삽입 포함)
  _mdd_analysis/<tag>/l1_prior/l1_prior_residuals.csv  잔차(대표 약점 순)

출력(_mdd_analysis/<tag>/error_catalog/):
  l1_error_catalog.csv           enriched 롱포맷(위치·한글·유형 병기)
  L1_발음오류_상세카탈로그.md     사람이 읽는 국적별 상세 카탈로그
  심볼_범례.csv                  KoG2P 기호 ↔ 한글 ↔ 위치

실행:
  python s8_error_catalog.py --tag 040_041_042_043_both --min-usable 500 --topn 12
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import os
import csv
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 하위 -> 데이터 루트는 부모
from mdd.l1prior.s0_kog2p_wrapper import ONSET, NUCLEUS, CODA          # 위치 판정
from mdd.l1prior.s0_symbol_map import SYMBOL_TO_JAMO                    # 기호→한글
from mdd.l1prior.s0_taxonomy import classify_op, CORE_TAGS             # 유형 분류

ONSET_S, NUC_S, CODA_S = set(ONSET), set(NUCLEUS), set(CODA)

# 기호 읽기(대략 발음) — 범례용
SOUND = {
    'k0': 'ㄱ', 'kk': 'ㄲ', 'nn': 'ㄴ', 't0': 'ㄷ', 'tt': 'ㄸ', 'rr': 'ㄹ',
    'mm': 'ㅁ', 'p0': 'ㅂ', 'pp': 'ㅃ', 's0': 'ㅅ', 'ss': 'ㅆ', 'oh': 'ㅇ(무음)',
    'c0': 'ㅈ', 'cc': 'ㅉ', 'ch': 'ㅊ', 'kh': 'ㅋ', 'th': 'ㅌ', 'ph': 'ㅍ', 'h0': 'ㅎ',
    'aa': 'ㅏ', 'qq': 'ㅐ', 'ya': 'ㅑ', 'yq': 'ㅒ', 'vv': 'ㅓ', 'ee': 'ㅔ', 'yv': 'ㅕ',
    'ye': 'ㅖ', 'oo': 'ㅗ', 'wa': 'ㅘ', 'wq': 'ㅙ', 'wo': 'ㅚ', 'yo': 'ㅛ', 'uu': 'ㅜ',
    'wv': 'ㅝ', 'we': 'ㅞ', 'wi': 'ㅟ', 'yu': 'ㅠ', 'xx': 'ㅡ', 'xi': 'ㅢ', 'ii': 'ㅣ',
    'kf': 'ㄱ받침', 'nf': 'ㄴ받침', 'tf': 'ㄷ받침', 'll': 'ㄹ받침', 'mf': 'ㅁ받침',
    'pf': 'ㅂ받침', 'ng': 'ㅇ받침', 'ks': 'ㄳ', 'nc': 'ㄵ', 'nh': 'ㄶ', 'lk': 'ㄺ',
    'lm': 'ㄻ', 'lb': 'ㄼ', 'ls': 'ㄽ', 'lt': 'ㄾ', 'lp': 'ㄿ', 'lh': 'ㅀ', 'ps': 'ㅄ',
}
# 국적 한글(주요국)
KO = {
    'Indonesia': '인도네시아', 'Vietnam': '베트남', 'China': '중국', 'Philippines': '필리핀',
    'Japan': '일본', 'United States': '미국', 'Thailand': '태국', 'Mongolia': '몽골',
    'Russian Federation': '러시아', 'Myanmar': '미얀마', 'Taiwan': '대만', 'Spain': '스페인',
    'France': '프랑스', 'Germany': '독일', 'United Kingdom': '영국', 'Uzbekistan': '우즈베키스탄',
    'India': '인도', 'Nepal': '네팔', 'Cambodia': '캄보디아', 'Malaysia': '말레이시아',
    'Italy': '이탈리아', 'Brazil': '브라질', 'Mexico': '멕시코', 'Canada': '캐나다',
    'Australia': '호주', 'Singapore': '싱가포르', 'Turkey': '튀르키예', 'Poland': '폴란드',
    'Hungary': '헝가리', 'Romania': '루마니아', 'Portugal': '포르투갈', 'Kazakhstan': '카자흐스탄',
    'Kyrgyzstan': '키르기스스탄', 'Ukraine': '우크라이나', 'Egypt': '이집트', 'Bangladesh': '방글라데시',
    'Pakistan': '파키스탄', 'Sri Lanka': '스리랑카', 'Hong Kong': '홍콩', 'Austria': '오스트리아',
    'Czech Republic': '체코', 'Netherlands': '네덜란드', 'South Africa': '남아공',
}


def role(sym):
    if sym in NUC_S:
        return "중성"
    if sym in CODA_S:
        return "종성"
    if sym in ONSET_S:
        return "초성"
    return "기타"


def ko(sym):
    return SOUND.get(sym, SYMBOL_TO_JAMO.get(sym, sym))


def label(sym):
    """예: k0 -> 'k0=ㄱ(초성)'"""
    return f"{sym}={ko(sym)}({role(sym)})"


def etype(t, a):
    try:
        return classify_op({"kind": "sub", "target": t, "actual": a, "target_next": None})[0]
    except Exception:
        return "기타"


def main(tag, min_usable, topn):
    d = os.path.join(ROOT, "_mdd_analysis", tag)
    outdir = os.path.join(d, "error_catalog")
    os.makedirs(outdir, exist_ok=True)

    subs = pd.read_csv(os.path.join(d, "substitution_pairs.csv"), encoding="utf-8-sig")
    summ = pd.read_csv(os.path.join(d, "l1_summary.csv"), encoding="utf-8-sig").set_index("L1")
    etc = pd.read_csv(os.path.join(d, "l1_errortype_counts.csv"),
                      encoding="utf-8-sig").set_index("L1")
    resid_p = os.path.join(d, "l1_prior", "l1_prior_residuals.csv")
    resid = pd.read_csv(resid_p, encoding="utf-8-sig", index_col=0) if os.path.exists(resid_p) else None

    # 대상 국적: usable >= min_usable, 물량 순
    keep = summ.index[summ["usable"] >= min_usable]
    order = summ.loc[keep].sort_values("usable", ascending=False).index.tolist()

    # ── 심볼 범례 CSV ──
    with open(os.path.join(outdir, "심볼_범례.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["symbol", "hangul", "position"])
        for s in ONSET + NUCLEUS + CODA:
            w.writerow([s, ko(s), role(s)])

    # ── enriched 롱포맷 ──
    cat_rows = []
    for L1 in order:
        g = subs[subs.L1 == L1].copy()
        tot = g["count"].sum()
        g = g.sort_values("count", ascending=False)
        for rk, (_, r) in enumerate(g.iterrows(), 1):
            t, a = str(r.target_phone), str(r.actual_phone)
            cat_rows.append([
                L1, rk, role(t), t, ko(t), a, ko(a), etype(t, a),
                int(r["count"]), round(r["count"] / tot * 100, 2)])
    pd.DataFrame(cat_rows, columns=[
        "L1", "rank", "target_position", "target_sym", "target_ko",
        "actual_sym", "actual_ko", "error_type", "count", "share_pct"
    ]).to_csv(os.path.join(outdir, "l1_error_catalog.csv"),
              index=False, encoding="utf-8-sig")

    # ── 전역 개요: 가장 흔한 대치(국적 무관) ──
    allp = subs.groupby(["target_phone", "actual_phone"])["count"].sum().reset_index()
    allp = allp.sort_values("count", ascending=False).head(20)

    # ── 상세 마크다운 ──
    L = []
    L.append("# 국적(L1)별 한국어 발음오류 상세 카탈로그 — KoG2P MDD\n")
    L.append(f"_생성: s8_error_catalog.py · 대상 tag `{tag}` · usable≥{min_usable} → {len(order)}개국_\n")
    L.append("## 0. 읽는 법\n")
    L.append("- **표기**: `k0=ㄱ(초성)` 처럼 `KoG2P기호=한글(위치)`. 위치=초성/중성(모음)/종성(받침).")
    L.append("- **대치 표기**: `정답 → 실현`. 예: `k0=ㄱ(초성) → kh=ㅋ(초성)` = 초성 ㄱ을 ㅋ으로 발음.")
    L.append("- **출처**: Speech_lab '들리는 대로의 한글 전사'를 KoG2P로 음소화 → 정답열과 MDD(edit-distance) 정렬한 대치쌍.")
    L.append("- **share%** = 그 국적의 전체 대치오류 중 비중. **잔차** = gold 기준 통계적 유의(양수=약점).")
    L.append("- 삭제(음운탈락)·삽입(모음삽입/ㄴ첨가)은 대치쌍이 아니므로 유형 카운트로 별도 표기.\n")

    L.append("## 1. 전체 개요 — 국적 무관 최빈 대치 20\n")
    L.append("| 순위 | 정답음소 | → | 실현음소 | 유형 | 총빈도 |")
    L.append("|--:|---|:-:|---|---|--:|")
    for i, (_, r) in enumerate(allp.iterrows(), 1):
        t, a = str(r.target_phone), str(r.actual_phone)
        L.append(f"| {i} | {label(t)} | → | {label(a)} | {etype(t,a)} | {int(r['count']):,} |")
    L.append("")

    L.append("## 2. 국적별 상세\n")
    for L1 in order:
        g = subs[subs.L1 == L1].copy()
        tot = int(g["count"].sum())
        usable = int(summ.loc[L1, "usable"])
        nm = f"{L1} ({KO.get(L1, '')})".strip()
        L.append(f"### {nm}")
        L.append(f"- usable 발화 **{usable:,}** · 대치오류 총 **{tot:,}**건")
        # 대표 오류유형(잔차 상위)
        if resid is not None and L1 in resid.index:
            rr = resid.loc[L1].sort_values(ascending=False)
            top_r = [f"{k} ({v:+.0f})" for k, v in rr.head(4).items() if v > 0]
            if top_r:
                L.append(f"- **대표 약점(잔차순)**: {', '.join(top_r)}")
        # 삭제/삽입 유형 카운트
        if L1 in etc.index:
            row = etc.loc[L1]
            di = []
            for t in ["음운탈락", "모음삽입", "ㄴ-첨가"]:
                if t in row and row[t] > 0:
                    di.append(f"{t} {int(row[t]):,}")
            if di:
                L.append(f"- **삭제/삽입**: {' · '.join(di)}")
        # 위치별 top 대치
        g["pos"] = g.target_phone.map(role)
        g = g.sort_values("count", ascending=False)
        for pos in ["초성", "중성", "종성"]:
            gp = g[g.pos == pos].head(topn)
            if gp.empty:
                continue
            L.append(f"\n**{pos} 오류 (상위 {min(topn,len(gp))})**")
            L.append("| 정답 → 실현 | 유형 | 빈도 | 비중 |")
            L.append("|---|---|--:|--:|")
            for _, r in gp.iterrows():
                t, a = str(r.target_phone), str(r.actual_phone)
                L.append(f"| {label(t)} → {label(a)} | {etype(t,a)} | "
                         f"{int(r['count']):,} | {r['count']/tot*100:.1f}% |")
        L.append("")

    open(os.path.join(outdir, "L1_발음오류_상세카탈로그.md"), "w",
         encoding="utf-8").write("\n".join(L))

    print(f"[완료] {outdir}")
    print(f"  대상 {len(order)}개국 · 카탈로그 행 {len(cat_rows):,}")
    print("  파일: L1_발음오류_상세카탈로그.md, l1_error_catalog.csv, 심볼_범례.csv")
    # 콘솔 미리보기(주요국 초성 top3)
    for L1 in ["Vietnam", "China", "Japan", "Philippines"]:
        if L1 not in order:
            continue
        g = subs[subs.L1 == L1].copy(); g["pos"] = g.target_phone.map(role)
        gp = g[g.pos == "초성"].sort_values("count", ascending=False).head(3)
        s = ", ".join(f"{label(str(r.target_phone))}→{label(str(r.actual_phone))}({int(r['count'])})"
                      for _, r in gp.iterrows())
        print(f"    [{L1} 초성] {s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="040_041_042_043_both")
    ap.add_argument("--min-usable", type=int, default=500)
    ap.add_argument("--topn", type=int, default=12)
    a = ap.parse_args()
    main(a.tag, a.min_usable, a.topn)
