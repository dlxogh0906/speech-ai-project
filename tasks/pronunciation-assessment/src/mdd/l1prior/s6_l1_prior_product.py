# -*- coding: utf-8 -*-
"""
s6_l1_prior_product.py
~~~~~~~~~~~~~~~~~~~~~~~
L1 prior 제품화 — 잔차/율 통계를 **진단 엔진이 바로 쓰는 prior 가중치**로 변환하고,
기존 국적별 피드백 JSON(정성)을 비파괴로 확장한다.
(로드맵 Sprint 2 ③ "L1 prior 통합 — 국적별 오류 확률↑ 를 진단에 반영")

가중치 정의(진단 시 곱해지는 사전배수):
  p_g(t)   = 전역 P(오류유형 t | 음소)              = Σcount(t) / Σphones
  p_L1(t)  = (count_L1(t) + k·p_g(t)) / (phones_L1 + k)   # 가법 스무딩(k=SHRINK)
  lift(t|L1) = p_L1(t) / p_g(t)                     # 1보다 크면 그 L1이 t를 자주 냄
  prior_weight = clip(lift, WMIN, WMAX)             # 진단 확률에 곱하는 배수
근거지표:
  residual  = gold 표준화잔차(있으면)                # 통계적 유의(+약점/−강점)
  speech_lift = Speech_lab 실현형 기반 lift(보조·상보)

입력(기존 산출물, 무거운 스캔 불필요):
  _error_tag_analysis/combined/by_nationality_pivot.csv          (gold 카운트/음소)
  _error_tag_analysis/combined/standardized_residuals_core.csv   (gold 잔차)
  _error_tag_analysis/combined/nationality_pronunciation_feedback.json (정성 피드백)
  _mdd_analysis/<speech_tag>/l1_errortype_rate.csv               (speech 율, 보조)
  _mdd_analysis/<speech_tag>/l1_summary.csv

출력(_mdd_analysis/l1_prior_product/):
  l1_prior_weights.json     국적별 기계용 prior 가중치(진단 엔진 입력)
  l1_prior_weights_long.csv 롱포맷(시각화/스프레드시트)
  nationality_feedback_enriched.json  기존 정성피드백 + 기계 가중치(top_priors) 부착
  l1_prior_product_report.md

실행:
  python s6_l1_prior_product.py --min-files 500 --speech-tag 040_041_042_043_both
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import os
import json
import argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 하위 -> 데이터 루트는 부모
from mdd.l1prior.s0_taxonomy import CORE_TAGS  # noqa: E402

COMB = os.path.join(ROOT, "_error_tag_analysis", "combined")
SHRINK = 6000.0     # 스무딩 강도(음소 등가). 소표본을 lift=1로 당김
WMIN, WMAX = 0.20, 5.0


def signal_strength(n_files, max_abs_resid):
    if n_files < 500:
        return "weak"
    if max_abs_resid >= 20:
        return "strong"
    if max_abs_resid >= 8:
        return "moderate"
    return "weak"


def main(min_files, speech_tag):
    outdir = os.path.join(ROOT, "_mdd_analysis", "l1_prior_product")
    os.makedirs(outdir, exist_ok=True)

    piv = pd.read_csv(os.path.join(COMB, "by_nationality_pivot.csv"),
                      encoding="utf-8-sig").set_index("nationality")
    resid = pd.read_csv(os.path.join(COMB, "standardized_residuals_core.csv"),
                        encoding="utf-8-sig", index_col=0)

    tags = [t for t in CORE_TAGS if t in piv.columns]

    # ── 전역 사전확률 p_g(t) = Σcount / Σphones ──
    tot_phones = float(piv["total_phones"].sum())
    p_g = {t: float(piv[t].sum()) / tot_phones for t in tags}

    # ── speech 보조 lift (있으면) ──
    speech_lift = {}
    sp_rate_path = os.path.join(ROOT, "_mdd_analysis", speech_tag, "l1_errortype_rate.csv")
    if os.path.exists(sp_rate_path):
        sr = pd.read_csv(sp_rate_path, encoding="utf-8-sig").set_index("L1")
        # 전역 speech 율(열평균 가중은 생략, 단순 국가평균 baseline)
        for L1 in sr.index:
            speech_lift[L1] = {}
            for t in tags:
                if t in sr.columns:
                    g = sr[t].mean() or 1e-9
                    speech_lift[L1][t] = round(float(sr.loc[L1, t]) / g, 3)

    # ── 국적별 가중치 ──
    priors = {}
    long_rows = []
    for nat, row in piv.iterrows():
        n_files = int(row["n_files"])
        phones = float(row["total_phones"]) or 1.0
        if n_files < min_files:
            continue
        weights = {}
        max_abs_resid = 0.0
        for t in tags:
            c = float(row[t])
            p_l1 = (c + SHRINK * p_g[t]) / (phones + SHRINK)
            lift = p_l1 / p_g[t] if p_g[t] else 1.0
            pw = float(np.clip(lift, WMIN, WMAX))
            r = float(resid.loc[nat, t]) if (nat in resid.index and t in resid.columns) else None
            if r is not None:
                max_abs_resid = max(max_abs_resid, abs(r))
            weights[t] = {
                "count": int(c),
                "rate_per100phones": round(c / phones * 100, 4),
                "lift": round(lift, 3),
                "prior_weight": round(pw, 3),
                "residual": round(r, 2) if r is not None else None,
                "speech_lift": speech_lift.get(nat, {}).get(t),
            }
        # 랭크: 잔차 우선(없으면 lift)
        def _key(t):
            r = weights[t]["residual"]
            return r if r is not None else (weights[t]["lift"] - 1) * 100
        ranked = sorted(tags, key=_key, reverse=True)
        for rk, t in enumerate(ranked, 1):
            weights[t]["rank"] = rk
            long_rows.append([nat, t, rk, weights[t]["count"],
                              weights[t]["rate_per100phones"], weights[t]["lift"],
                              weights[t]["prior_weight"], weights[t]["residual"],
                              weights[t]["speech_lift"]])
        sig = signal_strength(n_files, max_abs_resid)
        top = [t for t in ranked if (weights[t]["residual"] or 0) > 0
               or weights[t]["lift"] > 1.15][:4]
        priors[nat] = {
            "l1": nat, "n_files": n_files, "total_phones": int(phones),
            "signal_strength": sig,
            "top_priors": [{"error_type": t, "prior_weight": weights[t]["prior_weight"],
                            "residual": weights[t]["residual"]} for t in top],
            "weights": weights,
        }

    meta = {
        "title": "L1 diagnostic prior weights (국적별 오류유형 사전배수)",
        "basis": "gold error_tags 국적별 카운트/음소 → 스무딩 lift. residual=표준화잔차. "
                 "speech_lift=Speech_lab 실현형 보조.",
        "shrink_k": SHRINK, "weight_clip": [WMIN, WMAX],
        "min_files": min_files, "n_countries": len(priors),
        "core_tags": tags,
        "how_to_use": "진단 시 P(오류유형 t | 음향) 에 prior_weight[L1][t] 를 곱해 "
                      "국적 사전지식을 반영(경험적 배수, 확률 아님).",
    }
    with open(os.path.join(outdir, "l1_prior_weights.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "priors": priors}, f, ensure_ascii=False, indent=2)

    pd.DataFrame(long_rows, columns=[
        "L1", "error_type", "rank", "count", "rate_per100phones", "lift",
        "prior_weight", "residual", "speech_lift"]
    ).to_csv(os.path.join(outdir, "l1_prior_weights_long.csv"),
             index=False, encoding="utf-8-sig")

    # ── 기존 정성 피드백 확장(비파괴) ──
    fb_path = os.path.join(COMB, "nationality_pronunciation_feedback.json")
    n_enriched = 0
    if os.path.exists(fb_path):
        fb = json.load(open(fb_path, encoding="utf-8"))
        feats = fb.get("feedback", {})
        for country, ent in feats.items():
            if country in priors:
                p = priors[country]
                ent["prior_weights"] = p["weights"]
                ent["top_priors_ko"] = [
                    f'{tp["error_type"]}(×{tp["prior_weight"]})' for tp in p["top_priors"]]
                ent["signal_strength_data"] = p["signal_strength"]
                n_enriched += 1
        fb.setdefault("meta", {})["enriched_with"] = \
            "s6_l1_prior_product: prior_weights(기계용 진단 배수) 부착"
        # 정성 피드백 없는(가중치만 존재) 국가 목록
        fb["meta"]["weights_only_countries"] = sorted(set(priors) - set(feats))
        with open(os.path.join(outdir, "nationality_feedback_enriched.json"),
                  "w", encoding="utf-8") as f:
            json.dump(fb, f, ensure_ascii=False, indent=2)

    # ── 리포트 ──
    L = ["# L1 prior 제품화 — 진단 가중치\n",
         f"- 대상 국적: **{len(priors)}개국**(n_files≥{min_files}) · core {len(tags)}유형",
         f"- 스무딩 k={SHRINK:.0f}, 가중치 clip [{WMIN},{WMAX}]",
         f"- 정성 피드백 확장: {n_enriched}개국(기존 JSON에 prior_weights 부착)\n",
         "## 대표 국적 top prior (prior_weight = 진단 사전배수)"]
    show = ["China", "Vietnam", "Japan", "Philippines", "Indonesia",
            "Spain", "Thailand", "United States", "Mongolia", "Russian Federation"]
    L.append("| 국적 | 신호 | top priors (배수) |")
    L.append("|---|---|---|")
    for nat in show:
        if nat in priors:
            p = priors[nat]
            tp = ", ".join(f'{x["error_type"]}×{x["prior_weight"]}' for x in p["top_priors"])
            L.append(f"| {nat} | {p['signal_strength']} | {tp} |")
    L.append("\n## 사용법")
    L.append("- `l1_prior_weights.json` → 진단 엔진이 `priors[L1].weights[type].prior_weight` 로 조회.")
    L.append("- 음향 점수 P(t|audio) × prior_weight → 국적 사전지식 반영(경험적 배수).")
    L.append("- `nationality_feedback_enriched.json` → 학습자 UI(정성 피드백 + 기계 가중치).")
    open(os.path.join(outdir, "l1_prior_product_report.md"), "w",
         encoding="utf-8").write("\n".join(L))

    print(f"[완료] {outdir}")
    print(f"  국적 {len(priors)}개 · 정성피드백 확장 {n_enriched}개")
    for nat in ["China", "Vietnam", "Philippines", "Japan", "Spain"]:
        if nat in priors:
            tp = ", ".join(f'{x["error_type"]}×{x["prior_weight"]}'
                           for x in priors[nat]["top_priors"])
            print(f"    {nat:16s} {priors[nat]['signal_strength']:8s} {tp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-files", type=int, default=500)
    ap.add_argument("--speech-tag", default="040_041_042_043_both")
    a = ap.parse_args()
    main(a.min_files, a.speech_tag)
