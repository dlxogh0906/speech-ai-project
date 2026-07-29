# -*- coding: utf-8 -*-
"""
숙련도(proficiency) 보정 분석  ── 국적차가 L1 간섭인지 실력차인지 분리
------------------------------------------------------------------
입력 : _error_tag_analysis/combined/agg_full.csv  (Validation+Training 합본)

핵심 아이디어
  국적별 오류율이 높은 게 '모국어 간섭' 때문인지 '초급이 많아서'인지 구분하려면,
  같은 숙련도 그룹 안에서 국적을 비교해야 한다(층화 분석).
  → 특정 패턴(예: 중국=종성)이 모든 숙련도대에서 유지되면 = 순수 L1 효과로 확정.

출력 : _error_tag_analysis/combined/proficiency_adjusted_analysis.xlsx
  - rate_by_band   : 국적 x 숙련도  →  100음소당 오류율
  - files_by_band  : 국적 x 숙련도  →  파일수(표본)
  - confound       : 국적 평균숙련도 vs 오류율 상관 (혼입 정도 수치화)
  - resid_<band>   : 숙련도대별 표준화 잔차(core 태그) → 층화 검증
  - key_patterns   : 대표 패턴 태그의 국적x숙련도 오류율(지속성 확인)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]


import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 하위 -> 데이터 루트는 부모
BASE = os.path.join(HERE, "_error_tag_analysis", "combined")
SRC = os.path.join(BASE, "agg_full.csv")
OUT = os.path.join(BASE, "proficiency_adjusted_analysis.xlsx")

META = ["dataset", "split", "nationality", "language", "proficiency",
        "topik_level", "n_files", "n_err_files", "total_phones",
        "total_words", "total_errors"]
EXCLUDE = ["기타", "간투사", "반복"]
PROF_ORDER = ["Beginner", "Intermediate", "Advance", "Fluent"]
PROF_SCORE = {"Beginner": 1, "Intermediate": 2, "Advance": 3, "Fluent": 4}
MIN_TOTAL = 500     # 전체 국적 필터
MIN_BAND = 200      # 층화 잔차용 밴드내 최소 표본
# 대표 패턴 (국적, 특징 태그)
KEY = [("China", "종성"), ("Japan", "종성"), ("Vietnam", "종성"),
       ("Philippines", "모음오류"), ("United States", "모음오류"),
       ("Indonesia", "경음화"), ("Thailand", "강세 오류")]


def resid_table(sub, tags):
    """nationality x tag 카운트 → 표준화 잔차 DataFrame"""
    # 합계 0인 열(태그) 제거
    keep = [t for t in tags if sub[t].sum() > 0]
    # 합계 0인 행(국적) 제거 → 기대빈도 0 방지
    sub = sub[sub[keep].sum(axis=1) > 0].copy()
    obs = sub[keep].to_numpy(dtype=float)
    chi2, p, dof, exp = chi2_contingency(obs)
    r = (obs - exp) / np.sqrt(exp)
    order = sub[keep].sum().sort_values(ascending=False).index.tolist()
    df = pd.DataFrame(r, columns=keep, index=sub["nationality"])[order].round(2)
    return df.reset_index(), chi2, p, dof


def main():
    if not os.path.exists(SRC):
        print("입력 없음:", SRC, "\n먼저 combine_excel.py 를 실행하세요.")
        return

    df = pd.read_csv(SRC, encoding="utf-8-sig")
    tag_cols = [c for c in df.columns if c not in META]
    core = [c for c in tag_cols if c not in EXCLUDE]
    df = df[df["proficiency"].isin(PROF_ORDER)].copy()

    # (국적,숙련도) 집계 -----------------------------------------------------
    g = df.groupby(["nationality", "proficiency"], as_index=False).agg(
        {**{c: "sum" for c in ["n_files", "total_phones", "total_errors"]},
         **{c: "sum" for c in tag_cols}})
    g["err_per100"] = (g["total_errors"] / g["total_phones"].replace(0, 1)
                       * 100).round(3)

    # 국적 전체 표본으로 상위 국적만 -----------------------------------------
    nat_tot = df.groupby("nationality")["n_files"].sum()
    big = nat_tot[nat_tot >= MIN_TOTAL].index.tolist()
    gb = g[g["nationality"].isin(big)]

    # rate_by_band / files_by_band 피벗 --------------------------------------
    rate_piv = gb.pivot(index="nationality", columns="proficiency",
                        values="err_per100").reindex(columns=PROF_ORDER)
    rate_piv = rate_piv.reindex(nat_tot[big].sort_values(ascending=False).index)
    files_piv = gb.pivot(index="nationality", columns="proficiency",
                         values="n_files").reindex(columns=PROF_ORDER).fillna(0).astype(int)
    files_piv = files_piv.reindex(rate_piv.index)

    # 혼입 정도: 국적 평균 숙련도점수 vs 전체 오류율 --------------------------
    rows = []
    for nat in big:
        sub = df[df["nationality"] == nat]
        w = sub["n_files"].to_numpy(dtype=float)
        score = np.average([PROF_SCORE[p] for p in sub["proficiency"]], weights=w)
        rate = sub["total_errors"].sum() / max(sub["total_phones"].sum(), 1) * 100
        rows.append((nat, round(score, 3), round(rate, 3), int(w.sum())))
    conf = pd.DataFrame(rows, columns=["nationality", "mean_prof_score",
                                       "err_per100", "n_files"])
    r_pear, p_pear = pearsonr(conf["mean_prof_score"], conf["err_per100"])

    # 숙련도대별 층화 잔차 ----------------------------------------------------
    band_resid = {}
    for band in PROF_ORDER:
        sub = g[(g["proficiency"] == band) &
                (g["n_files"] >= MIN_BAND) &
                (g["nationality"].isin(big))].copy()
        sub = sub.sort_values("n_files", ascending=False)
        if len(sub) < 3:
            continue
        rt, chi2, p, dof = resid_table(sub, core)
        rt.insert(0, "proficiency", band)
        band_resid[band] = (rt, chi2, p, dof, len(sub))

    # key_patterns : 대표 태그 오류율(국적 x 숙련도) --------------------------
    kp_rows = []
    for nat, tag in KEY:
        for band in PROF_ORDER:
            row = g[(g["nationality"] == nat) & (g["proficiency"] == band)]
            if len(row):
                rr = row.iloc[0]
                rate = rr[tag] / max(rr["total_phones"], 1) * 100
                kp_rows.append((nat, tag, band, round(rate, 3), int(rr["n_files"])))
    key_df = pd.DataFrame(kp_rows, columns=["nationality", "tag",
                                            "proficiency", "tag_per100", "n_files"])

    # ---- 저장 --------------------------------------------------------------
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        rate_piv.round(3).to_excel(xw, sheet_name="rate_by_band")
        files_piv.to_excel(xw, sheet_name="files_by_band")
        conf.sort_values("mean_prof_score").to_excel(
            xw, sheet_name="confound", index=False)
        pd.DataFrame({
            "항목": ["평균숙련도 vs 오류율 Pearson r", "p_value", "해석"],
            "값": [round(r_pear, 3), f"{p_pear:.3e}",
                   "음(-)이면 숙련도↑→오류↓ (실력효과 존재). "
                   "그래도 국적패턴이 밴드내 유지되면 L1효과도 독립적으로 존재"],
        }).to_excel(xw, sheet_name="confound", index=False, startrow=len(conf) + 3)
        for band, (rt, chi2, p, dof, nn) in band_resid.items():
            rt.to_excel(xw, sheet_name=f"resid_{band}", index=False)
        key_df.to_excel(xw, sheet_name="key_patterns", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"

    # ---- 콘솔 요약 ----------------------------------------------------------
    print(f"[혼입] 평균숙련도 vs 오류율  Pearson r = {r_pear:.3f} (p={p_pear:.2e})")
    print("   → 음수면 '숙련도 높을수록 오류 적음'(실력효과) 존재\n")
    print("[층화 검정] 각 숙련도대에서 국적x오류 카이제곱 (유의하면 L1효과 독립 존재)")
    for band, (rt, chi2, p, dof, nn) in band_resid.items():
        print(f"   {band:13s} 국적{nn}개  chi2={chi2:,.0f} p={p:.1e}")
    print("\n[대표 패턴 지속성] 특징 태그의 100음소당 오류율 (숙련도대별)")
    for nat, tag in KEY:
        d = key_df[(key_df.nationality == nat) & (key_df.tag == tag)]
        seq = "  ".join(f"{b[:3]}={d[d.proficiency==b]['tag_per100'].iloc[0]:.2f}"
                        for b in PROF_ORDER if (d.proficiency == b).any())
        print(f"   {nat:15s} [{tag:6s}]  {seq}")
    print("\n[출력]", OUT)


if __name__ == "__main__":
    main()
