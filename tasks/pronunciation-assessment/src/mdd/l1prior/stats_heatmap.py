# -*- coding: utf-8 -*-
"""
국적 x 발음오류(error_tag) 통계 분석 + 히트맵
------------------------------------------------
입력 : _error_tag_analysis/by_nationality_pivot.csv  (analyze_error_tags.py 산출물)

분석 :
  1) 소표본 국적 필터 (--min-files, 기본 500)
  2) 국적 x 오류유형 분할표에 대한 카이제곱 독립성 검정
  3) 표준화(피어슨) 잔차 계산 → 어떤 국적이 어떤 오류를 유의하게 과다/과소 산출하는지
  4) 히트맵 2종 저장
     - heatmap_residuals.png : 표준화 잔차 (빨강=과다 / 파랑=과소)  ★핵심
     - heatmap_rate.png      : 100음소당 오류율 (절대 난이도 비교)

산출물(입력 폴더에 저장):
  - standardized_residuals.csv
  - chi2_result.txt
  - heatmap_residuals.png
  - heatmap_rate.png

사용:
  python stats_heatmap.py
  python stats_heatmap.py --min-files 300
  python stats_heatmap.py --pivot "경로/by_nationality_pivot.csv"
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]


import os
import sys
import argparse

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import matplotlib
matplotlib.use("Agg")               # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt

# 한글 폰트 (Windows: 맑은 고딕)
for _f in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
    try:
        matplotlib.rcParams["font.family"] = _f
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 통계 대상에서 제외할 메타성 컬럼
META_COLS = ["nationality", "n_files", "total_phones", "total_errors"]


def draw_heatmap(mat, row_labels, col_labels, title, out_path,
                 cmap, center_zero=False, fmt="{:.1f}", cbar_label=""):
    n_rows, n_cols = mat.shape
    fig_w = max(8, n_cols * 0.62 + 3)
    fig_h = max(5, n_rows * 0.42 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if center_zero:
        vmax = np.nanmax(np.abs(mat)) or 1.0
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(mat, cmap=cmap, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=13, pad=12)

    # 셀 값 주석
    thr = (np.nanmax(np.abs(mat)) or 1.0) * 0.55
    for i in range(n_rows):
        for j in range(n_cols):
            v = mat[i, j]
            if np.isnan(v):
                continue
            color = "white" if (center_zero and abs(v) > thr) or \
                               (not center_zero and v > thr) else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 하위 -> 데이터 루트는 부모
    default_pivot = os.path.join(here, "_error_tag_analysis", "by_nationality_pivot.csv")

    ap = argparse.ArgumentParser()
    ap.add_argument("--pivot", default=default_pivot)
    ap.add_argument("--min-files", type=int, default=500,
                    help="이 파일수 미만 국적은 제외 (기본 500)")
    ap.add_argument("--top-tags", type=int, default=0,
                    help=">0 이면 빈도 상위 N개 태그만 사용")
    ap.add_argument("--exclude-tags", nargs="*", default=[],
                    help="분석에서 제외할 오류태그 (예: 기타 간투사 반복)")
    ap.add_argument("--suffix", default="",
                    help="출력 파일명 뒤에 붙일 접미사 (기존 결과 보존용)")
    args = ap.parse_args()
    sfx = args.suffix

    if not os.path.exists(args.pivot):
        print("입력 파일을 찾을 수 없습니다:", args.pivot)
        return
    out_dir = os.path.dirname(args.pivot)

    df = pd.read_csv(args.pivot, encoding="utf-8-sig")
    tag_cols = [c for c in df.columns if c not in META_COLS]

    # 제외 태그 반영
    if args.exclude_tags:
        excl = [t for t in args.exclude_tags if t in tag_cols]
        tag_cols = [c for c in tag_cols if c not in excl]
        print(f"[제외] 태그 {excl} 제외 → 분석 태그 {len(tag_cols)}개")

    # 1) 소표본 필터
    n_before = len(df)
    df = df[df["n_files"] >= args.min_files].copy()
    df = df.sort_values("n_files", ascending=False).reset_index(drop=True)
    print(f"[필터] 국적 {n_before} → {len(df)}개 (n_files >= {args.min_files})")

    # 태그 상위 N개만(옵션)
    if args.top_tags and args.top_tags < len(tag_cols):
        order = df[tag_cols].sum().sort_values(ascending=False).index.tolist()
        tag_cols = order[:args.top_tags]
    # 전부 0인 태그 제거
    tag_cols = [c for c in tag_cols if df[c].sum() > 0]

    nats = df["nationality"].tolist()
    obs = df[tag_cols].to_numpy(dtype=float)   # 국적 x 태그 (관측 카운트)

    # 2) 카이제곱 검정
    chi2, p, dof, expected = chi2_contingency(obs)
    n_total = obs.sum()
    cramers_v = np.sqrt((chi2 / n_total) / (min(obs.shape) - 1))
    with open(os.path.join(out_dir, f"chi2_result{sfx}.txt"), "w", encoding="utf-8") as f:
        f.write("국적 x 오류유형 카이제곱 독립성 검정\n")
        f.write(f"대상 국적 수 : {len(nats)}\n")
        f.write(f"오류유형 수  : {len(tag_cols)}\n")
        f.write(f"총 오류 카운트: {int(n_total):,}\n")
        f.write(f"chi2 = {chi2:,.2f}\n")
        f.write(f"dof  = {dof}\n")
        f.write(f"p-value = {p:.3e}\n")
        f.write(f"Cramer's V = {cramers_v:.4f}\n")
    print(f"[카이제곱] chi2={chi2:,.1f}, dof={dof}, p={p:.3e}, Cramer's V={cramers_v:.3f}")

    # 3) 표준화(피어슨) 잔차 = (관측 - 기대) / sqrt(기대)
    resid = (obs - expected) / np.sqrt(expected)
    resid_df = pd.DataFrame(resid, index=nats, columns=tag_cols)
    resid_df.to_csv(os.path.join(out_dir, f"standardized_residuals{sfx}.csv"),
                    encoding="utf-8-sig")

    # 태그를 전체 빈도순으로 정렬(보기 좋게)
    tag_order = df[tag_cols].sum().sort_values(ascending=False).index.tolist()
    resid_ord = resid_df[tag_order]

    # 4-a) 잔차 히트맵 (핵심)
    draw_heatmap(
        resid_ord.to_numpy(), nats, tag_order,
        f"국적별 특징적 발음오류 (표준화 잔차)  |  Cramer's V={cramers_v:.3f}",
        os.path.join(out_dir, f"heatmap_residuals{sfx}.png"),
        cmap="RdBu_r", center_zero=True, fmt="{:+.1f}",
        cbar_label="표준화 잔차 (+과다 / -과소)")

    # 4-b) 100음소당 오류율 히트맵
    phones = df["total_phones"].to_numpy(dtype=float).reshape(-1, 1)
    rate = obs / phones * 100.0
    rate_df = pd.DataFrame(rate, index=nats, columns=tag_cols)[tag_order]
    draw_heatmap(
        rate_df.to_numpy(), nats, tag_order,
        "국적별 오류유형 발생률 (100음소당 오류 수)",
        os.path.join(out_dir, f"heatmap_rate{sfx}.png"),
        cmap="YlOrRd", center_zero=False, fmt="{:.2f}",
        cbar_label="100음소당 오류 수")

    # 콘솔: 국적별 가장 특징적인(잔차 최대) 오류 top2
    print("\n[국적별 특징적 오류 (표준화 잔차 상위)]")
    for nat in nats:
        r = resid_df.loc[nat].sort_values(ascending=False)
        hi = ", ".join(f"{t}(+{v:.1f})" for t, v in r.head(2).items())
        lo_t, lo_v = r.index[-1], r.iloc[-1]
        print(f"  {nat:22s} 과다: {hi}   | 과소: {lo_t}({lo_v:.1f})")

    print("\n[출력]", out_dir)
    for fn in [f"chi2_result{sfx}.txt", f"standardized_residuals{sfx}.csv",
               f"heatmap_residuals{sfx}.png", f"heatmap_rate{sfx}.png"]:
        print("   -", fn)


if __name__ == "__main__":
    main()
