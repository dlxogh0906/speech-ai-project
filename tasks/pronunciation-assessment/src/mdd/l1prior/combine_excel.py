# -*- coding: utf-8 -*-
"""
Validation + Training 통합 엑셀 생성
------------------------------------
입력:
  _error_tag_analysis/agg_full.csv          (Validation)
  _error_tag_analysis/training/agg_full.csv  (Training)

출력:
  _error_tag_analysis/combined_validation_training.xlsx
    - agg_full            : 두 split 원본 집계 합본(raw)
    - by_nationality      : 국적 x 오류태그 카운트(합산) + 오류율
    - summary             : 국적별 요약(파일수/오류율/상위3)
    - residuals_core      : 표준화 잔차(기타·간투사·반복 제외, n>=500)
    - rate_per100_core    : 100음소당 오류율(정제)
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
from scipy.stats import chi2_contingency

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 하위 -> 데이터 루트는 부모
BASE = os.path.join(HERE, "_error_tag_analysis")
VAL = os.path.join(BASE, "validation", "agg_full.csv")
TRN = os.path.join(BASE, "training", "agg_full.csv")
OUT = os.path.join(BASE, "combined", "combined_validation_training.xlsx")

META = ["dataset", "split", "nationality", "language", "proficiency",
        "topik_level", "n_files", "n_err_files", "total_phones",
        "total_words", "total_errors"]
EXCLUDE = ["기타", "간투사", "반복"]
MIN_FILES = 500


def load(path, split_name):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "split" not in df.columns:
        df.insert(1, "split", split_name)
    return df


def main():
    for p in (VAL, TRN):
        if not os.path.exists(p):
            print("입력 없음:", p)
            return

    val = load(VAL, "Validation")
    trn = load(TRN, "Training")

    # 컬럼 정렬 통일 후 합본 (없는 태그열은 0 채움)
    combined = pd.concat([val, trn], ignore_index=True, sort=False).fillna(0)
    tag_cols = [c for c in combined.columns if c not in META]
    # 정수형 보정
    for c in ["n_files", "n_err_files", "total_phones", "total_words",
              "total_errors"] + tag_cols:
        if c in combined:
            combined[c] = combined[c].astype(np.int64)

    # ---- by_nationality : 국적 단위 합산 --------------------------------
    grp = combined.groupby("nationality", as_index=False).agg(
        {**{c: "sum" for c in ["n_files", "n_err_files", "total_phones",
                               "total_words", "total_errors"]},
         **{c: "sum" for c in tag_cols}})
    grp = grp.sort_values("n_files", ascending=False).reset_index(drop=True)

    # 오류율(100음소당) 컬럼 추가본
    by_nat = grp.copy()
    ph = by_nat["total_phones"].replace(0, 1)
    for c in tag_cols:
        by_nat[c + "_per100"] = (by_nat[c] / ph * 100).round(4)

    # ---- summary --------------------------------------------------------
    def top3(row):
        s = row[tag_cols].sort_values(ascending=False)
        s = s[s > 0]
        return pd.Series({
            "top1": f"{s.index[0]}:{int(s.iloc[0])}" if len(s) > 0 else "",
            "top2": f"{s.index[1]}:{int(s.iloc[1])}" if len(s) > 1 else "",
            "top3": f"{s.index[2]}:{int(s.iloc[2])}" if len(s) > 2 else "",
        })

    summary = grp[["nationality", "n_files", "n_err_files",
                   "total_phones", "total_errors"]].copy()
    summary["err_file_pct"] = (grp["n_err_files"] / grp["n_files"].replace(0, 1)
                               * 100).round(2)
    summary["errors_per_100phones"] = (grp["total_errors"] /
                                       grp["total_phones"].replace(0, 1)
                                       * 100).round(4)
    summary = pd.concat([summary, grp.apply(top3, axis=1)], axis=1)

    # ---- residuals_core (통계) ------------------------------------------
    core_tags = [c for c in tag_cols if c not in EXCLUDE]
    sub = grp[grp["n_files"] >= MIN_FILES].copy()
    obs = sub[core_tags].to_numpy(dtype=float)
    chi2, p, dof, expected = chi2_contingency(obs)
    resid = (obs - expected) / np.sqrt(expected)
    # 태그를 전체 빈도순 정렬
    order = sub[core_tags].sum().sort_values(ascending=False).index.tolist()
    resid_df = pd.DataFrame(resid, columns=core_tags, index=sub["nationality"])[order]
    resid_df = resid_df.round(2).reset_index()

    rate = obs / sub["total_phones"].to_numpy(dtype=float).reshape(-1, 1) * 100
    rate_df = pd.DataFrame(rate, columns=core_tags, index=sub["nationality"])[order]
    rate_df = rate_df.round(3).reset_index()

    print(f"[통계] chi2={chi2:,.1f} dof={dof} p={p:.2e} "
          f"국적={len(sub)} 태그={len(core_tags)}")

    # ---- 엑셀 저장 ------------------------------------------------------
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        combined.to_excel(xw, sheet_name="agg_full", index=False)
        by_nat.to_excel(xw, sheet_name="by_nationality", index=False)
        summary.to_excel(xw, sheet_name="summary", index=False)
        resid_df.to_excel(xw, sheet_name="residuals_core", index=False)
        rate_df.to_excel(xw, sheet_name="rate_per100_core", index=False)
        # 검정 결과 메모 시트
        pd.DataFrame({
            "항목": ["chi2", "dof", "p_value", "국적수(n>=500)",
                     "오류태그수(core)", "제외태그", "합산 split"],
            "값": [round(chi2, 2), dof, f"{p:.3e}", len(sub),
                   len(core_tags), ", ".join(EXCLUDE), "Validation+Training"],
        }).to_excel(xw, sheet_name="chi2_info", index=False)

        # 헤더 틀고정
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"

    # ---- 통합 pivot CSV (stats_heatmap 입력용) --------------------------
    comb_dir = os.path.join(BASE, "combined")
    os.makedirs(comb_dir, exist_ok=True)
    pivot_cols = ["nationality", "n_files", "total_phones", "total_errors"] + tag_cols
    grp[pivot_cols].to_csv(os.path.join(comb_dir, "by_nationality_pivot.csv"),
                           index=False, encoding="utf-8-sig")
    combined.to_csv(os.path.join(comb_dir, "agg_full.csv"),
                    index=False, encoding="utf-8-sig")

    print("[출력]", OUT)
    print("  시트: agg_full / by_nationality / summary / "
          "residuals_core / rate_per100_core / chi2_info")
    print(f"  합본 행수: {len(combined):,}  국적수: {len(grp)}")
    print("[출력] combined/by_nationality_pivot.csv, combined/agg_full.csv")


if __name__ == "__main__":
    main()
