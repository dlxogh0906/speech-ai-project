# -*- coding: utf-8 -*-
"""
s4_l1_prior.py
~~~~~~~~~~~~~~
Speech_lab MDD 집계(s1_run_batch 산출물)로부터 L1 prior 를 만들고,
기존 pronunciation_lab gold 잔차와 교차검증한다. (작업지시서 6절)

입력:
  _mdd_analysis/<tag>/l1_errortype_counts.csv   (L1 × error_type, usable)
  _mdd_analysis/<tag>/l1_summary.csv            (L1별 usable 등)
  _mdd_analysis/<tag>/substitution_pairs.csv    (L1 × target>actual)
  _error_tag_analysis/combined/standardized_residuals_core.csv  (gold, 교차검증용)

출력(같은 <tag> 폴더 하위 l1_prior/):
  l1_prior_residuals.csv        L1 × error_type 표준화잔차 (Speech_lab)  = L1 prior 본체
  l1_prior_top_substitutions.csv L1별 상위 대치쌍(정답>실현, 한글 병기)
  crossval_by_errortype.csv     오류유형별 Speech vs gold 잔차 상관(피어슨)
  l1_prior_report.md            요약 보고서

실행:
  python s4_l1_prior.py --tag 040_041_042_043_both --min-usable 500
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from mdd.l1prior.s0_symbol_map import SYMBOL_TO_JAMO
from mdd.l1prior.s0_taxonomy import CORE_TAGS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 하위 -> 데이터 루트는 부모
GOLD = os.path.join(ROOT, "_error_tag_analysis", "combined",
                    "standardized_residuals_core.csv")
EXCLUDE = {"기타"}   # 미분류 제외(기존 core 기준과 정합)


def std_residuals(counts_df):
    """L1×error_type 카운트 -> 표준화(피어슨) 잔차 df. r=(O-E)/sqrt(E)."""
    M = counts_df.values.astype(float)
    N = M.sum()
    row = M.sum(1, keepdims=True)
    col = M.sum(0, keepdims=True)
    E = row @ col / N
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (M - E) / np.sqrt(E)
    R = np.nan_to_num(R)
    return pd.DataFrame(R, index=counts_df.index, columns=counts_df.columns)


def gloss(sym):
    return SYMBOL_TO_JAMO.get(sym, sym)


def main(tag, min_usable, topn):
    d = os.path.join(ROOT, "_mdd_analysis", tag)
    outdir = os.path.join(d, "l1_prior")
    os.makedirs(outdir, exist_ok=True)

    # ── 로드 ──
    counts = pd.read_csv(os.path.join(d, "l1_errortype_counts.csv"),
                         encoding="utf-8-sig").set_index("L1")
    summ = pd.read_csv(os.path.join(d, "l1_summary.csv"),
                       encoding="utf-8-sig").set_index("L1")
    subs = pd.read_csv(os.path.join(d, "substitution_pairs.csv"),
                       encoding="utf-8-sig")

    # 유형 열 정리(기타 제외)
    et_cols = [c for c in counts.columns if c not in EXCLUDE]
    counts = counts[et_cols]

    # usable 임계로 L1 필터
    keep = summ.index[summ["usable"] >= min_usable]
    counts = counts.loc[counts.index.intersection(keep)]
    print(f"[*] L1 필터: usable>={min_usable} -> {len(counts)}개국 (전체 {len(summ)})")

    # ── L1 prior 잔차 ──
    resid = std_residuals(counts).round(3)
    resid.to_csv(os.path.join(outdir, "l1_prior_residuals.csv"),
                 encoding="utf-8-sig")

    # ── L1별 상위 대치쌍(정답>실현) ──
    subs = subs[subs["L1"].isin(counts.index)].copy()
    subs["pair"] = subs["target_phone"].astype(str) + ">" + subs["actual_phone"].astype(str)
    rows = []
    for L1, g in subs.groupby("L1"):
        tot = g["count"].sum()
        for _, r in g.sort_values("count", ascending=False).head(topn).iterrows():
            t, a = str(r["target_phone"]), str(r["actual_phone"])
            rows.append([L1, t, a, f"{gloss(t)}>{gloss(a)}", int(r["count"]),
                         round(r["count"] / tot, 4)])
    pd.DataFrame(rows, columns=["L1", "target", "actual", "gloss", "count", "share"]
                 ).to_csv(os.path.join(outdir, "l1_prior_top_substitutions.csv"),
                          index=False, encoding="utf-8-sig")

    # ── 교차검증: Speech_lab 잔차 vs pronunciation_lab gold 잔차 ──
    gold = pd.read_csv(GOLD, encoding="utf-8-sig", index_col=0)
    common_L1 = sorted(set(resid.index) & set(gold.index))
    common_et = [c for c in resid.columns if c in gold.columns]
    cross_rows = []
    for et in common_et:
        x = resid.loc[common_L1, et].values
        y = gold.loc[common_L1, et].values
        if np.std(x) > 0 and np.std(y) > 0 and len(common_L1) >= 3:
            r, p = pearsonr(x, y)
            cross_rows.append([et, len(common_L1), round(r, 3), f"{p:.1e}"])
        else:
            cross_rows.append([et, len(common_L1), None, None])
    # 전체(flatten) 상관
    X = resid.loc[common_L1, common_et].values.flatten()
    Y = gold.loc[common_L1, common_et].values.flatten()
    r_all, p_all = pearsonr(X, Y)
    cross_rows.append(["__ALL__", len(common_L1) * len(common_et),
                       round(r_all, 3), f"{p_all:.1e}"])
    cross = pd.DataFrame(cross_rows, columns=["error_type", "n", "pearson_r", "p"])
    cross.to_csv(os.path.join(outdir, "crossval_by_errortype.csv"),
                 index=False, encoding="utf-8-sig")

    # ── 보고서 ──
    lines = []
    lines.append("# L1 prior (Speech_lab MDD) + gold 교차검증\n")
    lines.append(f"- 대상 tag: `{tag}` · L1 {len(counts)}개국(usable≥{min_usable})")
    lines.append(f"- 교차검증 공통: L1 {len(common_L1)}개국 × 오류유형 {len(common_et)}종")
    lines.append(f"- **전체 잔차 상관(Speech vs gold): Pearson r = {r_all:.3f}, p = {p_all:.1e}**\n")
    lines.append("## 오류유형별 Speech_lab↔gold 상관")
    lines.append("| 오류유형 | r | p |")
    lines.append("|---|---|---|")
    for et, n, r, p in cross_rows:
        if et == "__ALL__":
            continue
        lines.append(f"| {et} | {r} | {p} |")
    lines.append("\n## L1 prior 대표 신호 (표준화잔차 |r|≥5 상위)")
    sig = resid.stack().reset_index()
    sig.columns = ["L1", "error_type", "resid"]
    sig = sig.reindex(sig["resid"].abs().sort_values(ascending=False).index)
    lines.append("| L1 | 오류유형 | 잔차 |")
    lines.append("|---|---|---|")
    for _, r in sig.head(25).iterrows():
        lines.append(f"| {r['L1']} | {r['error_type']} | {r['resid']:+.1f} |")
    open(os.path.join(outdir, "l1_prior_report.md"), "w",
         encoding="utf-8").write("\n".join(lines))

    # ── 콘솔 ──
    print(f"[완료] {outdir}")
    print(f"  교차검증 전체 상관: r={r_all:.3f} (p={p_all:.1e}), 공통 {len(common_L1)}개국×{len(common_et)}유형")
    print("  유형별 상관:")
    for et, n, r, p in cross_rows:
        if et != "__ALL__":
            print(f"    {et:<10} r={r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="040_041_042_043_both")
    ap.add_argument("--min-usable", type=int, default=500)
    ap.add_argument("--topn", type=int, default=15)
    a = ap.parse_args()
    main(a.tag, a.min_usable, a.topn)
