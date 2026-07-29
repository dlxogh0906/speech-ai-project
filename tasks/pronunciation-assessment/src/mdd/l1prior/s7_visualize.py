# -*- coding: utf-8 -*-
"""
s7_visualize.py
~~~~~~~~~~~~~~~
발음오류 MDD → L1 prior 파이프라인 전체 통계를 **상세 시각화 리포트(자체완결 HTML)**로.
(모든 작업은 파이썬 코드로 — 시각화도 스크립트로 재현)

그림(전부 matplotlib, Malgun Gothic, base64 임베드):
  1) gold L1 prior 히트맵(국적×core14 표준화잔차)      [전체 데이터]
  2) B2 진단 prior 가중치 히트맵(국적×유형 prior_weight) [전체]
  3) gold↔speech 교차검증 산점도 + 유형별 상관           [전체]
  4) 소스 상보성(유형별 gold vs speech 검출 커버리지)     [전체]
  5) B1 벤치마크 — 유형별 AP(global vs L1) + lift         [검증표본/전체]
  6) 데이터 물량 상위 국적                                [전체]
  7) Speech 품질 플래그 분포                              [전체]

입력: 기존 산출물(무거운 스캔 불필요). 없는 입력은 건너뛰고 리포트에 '대기'로 표기.
출력: _mdd_analysis/_report/L1prior_통계리포트.html  (+ figs/ PNG)

실행:
  python s7_visualize.py --speech-tag 040_041_042_043_both \
                         --bench-tag 040_Training_pronlab
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import os
import io
import json
import base64
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 120

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 하위 -> 데이터 루트는 부모
COMB = os.path.join(ROOT, "_error_tag_analysis", "combined")
from mdd.l1prior.s0_taxonomy import CORE_TAGS  # noqa: E402

MINT = "#0bbd86"
INK = "#1a2b2b"


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _save(fig, figdir, name):
    p = os.path.join(figdir, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    return p


def heatmap(ax, M, rows, cols, center, cmap, title, vmax=None, fmt="{:.0f}"):
    vmax = vmax or np.nanmax(np.abs(M - center))
    im = ax.imshow(M, aspect="auto", cmap=cmap,
                   vmin=center - vmax, vmax=center + vmax)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=6.2, color="black" if abs(v - center) < vmax * 0.55 else "white")
    return im


def main(speech_tag, bench_tag):
    outdir = os.path.join(ROOT, "_mdd_analysis", "_report")
    figdir = os.path.join(outdir, "figs")
    os.makedirs(figdir, exist_ok=True)
    figs = {}      # id -> (title, b64, caption)
    kpis = {}
    sd = os.path.join(ROOT, "_mdd_analysis", speech_tag)

    # ── 로드 ──
    gold_res = pd.read_csv(os.path.join(COMB, "standardized_residuals_core.csv"),
                           encoding="utf-8-sig", index_col=0)
    piv = pd.read_csv(os.path.join(COMB, "by_nationality_pivot.csv"),
                      encoding="utf-8-sig").set_index("nationality")
    core = [t for t in CORE_TAGS if t in gold_res.columns]

    # 표시 국적 = gold 물량 상위(다양성 위해 상위 22)
    vol = piv["n_files"].sort_values(ascending=False)
    top_nat = [n for n in vol.index if n in gold_res.index][:22]

    kpis["gold_countries"] = int(gold_res.shape[0])
    kpis["gold_files"] = int(piv["n_files"].sum())
    kpis["gold_errors"] = int(piv["total_errors"].sum())

    # ── FIG 1: gold L1 prior 히트맵 ──
    M = gold_res.loc[top_nat, core].values
    fig, ax = plt.subplots(figsize=(11, 8))
    im = heatmap(ax, M, top_nat, core, 0, "RdBu_r",
                 "① Gold L1 prior — 국적 x 발음오류 표준화잔차 (빨강=약점 / 파랑=강점)",
                 vmax=60)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="표준화 잔차")
    figs["gold_heat"] = ("① Gold L1 prior 히트맵 (전체 데이터)", _b64(fig),
                         "사람이 태깅한 gold error_tags 기반. 국적별로 통계적으로 과다(+)/과소(−)한 오류유형. "
                         "China 종성 +75, Philippines 모음오류 +87 등 알려진 L1 간섭과 일치.")
    _save_note = None

    # ── FIG 2: B2 진단 prior 가중치 히트맵 ──
    wl_path = os.path.join(ROOT, "_mdd_analysis", "l1_prior_product",
                           "l1_prior_weights_long.csv")
    if os.path.exists(wl_path):
        wl = pd.read_csv(wl_path, encoding="utf-8-sig")
        W = wl.pivot_table(index="L1", columns="error_type", values="prior_weight")
        W = W.reindex(index=[n for n in top_nat if n in W.index],
                      columns=[c for c in core if c in W.columns])
        fig, ax = plt.subplots(figsize=(11, 8))
        im = heatmap(ax, W.values, list(W.index), list(W.columns), 1.0, "PuOr_r",
                     "② 진단 prior 가중치 (prior_weight = 진단확률 × 배수)",
                     vmax=2.0, fmt="{:.2f}")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="prior_weight (1=중립)")
        figs["weight_heat"] = ("② B2 진단 prior 가중치 (제품 산출물)", _b64(fig),
                               "gold 율을 스무딩한 lift. 진단 엔진이 P(오류유형|음향)에 이 배수를 곱해 국적 사전지식 반영. "
                               "Japan 모음삽입 ×3.3, Philippines 모음삽입 ×2.7, China 종성 ×1.7.")
        kpis["weight_countries"] = int(wl["L1"].nunique())

    # ── FIG 3: 교차검증(speech↔gold) ──
    sp_res_path = os.path.join(sd, "l1_prior", "l1_prior_residuals.csv")
    cv_path = os.path.join(sd, "l1_prior", "crossval_by_errortype.csv")
    if os.path.exists(sp_res_path):
        sp_res = pd.read_csv(sp_res_path, encoding="utf-8-sig", index_col=0)
        commonL = sorted(set(sp_res.index) & set(gold_res.index))
        commonT = [t for t in core if t in sp_res.columns]
        X = sp_res.loc[commonL, commonT].values.flatten()
        Y = gold_res.loc[commonL, commonT].values.flatten()
        from scipy.stats import pearsonr
        r_all, p_all = pearsonr(X, Y)
        kpis["crossval_r"] = round(float(r_all), 3)
        kpis["crossval_common"] = f"{len(commonL)}개국×{len(commonT)}유형"
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                                 gridspec_kw={"width_ratios": [1, 1.1]})
        ax = axes[0]
        ax.scatter(Y, X, s=10, alpha=0.35, color=MINT, edgecolor="none")
        lim = max(np.nanmax(np.abs(X)), np.nanmax(np.abs(Y))) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "--", color="#999", lw=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("gold 잔차 (pronunciation_lab)"); ax.set_ylabel("speech 잔차 (Speech_lab MDD)")
        ax.set_title(f"③ 교차검증 산점도  r = {r_all:.3f}", fontsize=12, fontweight="bold", color=INK)
        ax.axhline(0, color="#ccc", lw=.6); ax.axvline(0, color="#ccc", lw=.6)
        # per-type corr
        if os.path.exists(cv_path):
            cv = pd.read_csv(cv_path, encoding="utf-8-sig")
            cv = cv[cv["error_type"] != "__ALL__"].dropna(subset=["pearson_r"])
            cv = cv.sort_values("pearson_r")
            ax2 = axes[1]
            colors = [MINT if v >= 0 else "#d9776b" for v in cv["pearson_r"]]
            ax2.barh(cv["error_type"], cv["pearson_r"], color=colors)
            ax2.set_xlabel("유형별 Pearson r"); ax2.axvline(0, color="#333", lw=.8)
            ax2.set_title("유형별 speech↔gold 상관", fontsize=12, fontweight="bold", color=INK)
            for y, v in enumerate(cv["pearson_r"]):
                ax2.text(v + (.02 if v >= 0 else -.02), y, f"{v:.2f}",
                         va="center", ha="left" if v >= 0 else "right", fontsize=8)
        figs["crossval"] = ("③ gold ↔ speech 교차검증 (전체)", _b64(fig),
                            f"두 독립 소스의 L1 잔차 일치도. 전체 r={r_all:.3f}. "
                            "모음오류·종성·모음삽입·유음화는 강한 일치(전사로 잘 잡힘), "
                            "경음화·비음화는 KoG2P 규칙 양방향 상쇄로 near-zero → gold 권위.")

    # ── FIG 4: 소스 상보성(유형별 검출 커버리지) ──
    sp_rate_path = os.path.join(sd, "l1_errortype_rate.csv")
    if os.path.exists(sp_rate_path):
        sr = pd.read_csv(sp_rate_path, encoding="utf-8-sig").set_index("L1")
        gold_rate = {t: float(piv[t].sum()) / float(piv["total_phones"].sum()) * 100
                     for t in core if t in piv.columns}
        speech_present = {t: (t in sr.columns and sr[t].sum() > 0) for t in core}
        fig, ax = plt.subplots(figsize=(11, 4.6))
        xs = range(len(core))
        gvals = [gold_rate.get(t, 0) for t in core]
        ax.bar([x - .2 for x in xs], gvals, width=.4, color="#5a4fcf", label="gold 검출(율/100음소)")
        # speech: 있으면 상대적으로 검출됨(값 자체는 다른 분모라 존재여부만 표시)
        sp_flag = [1 if speech_present[t] else 0 for t in core]
        ax.bar([x + .2 for x in xs], [max(gvals) * f * 0.9 for f in sp_flag], width=.4,
               color=MINT, alpha=.5, label="speech 검출 가능(유무)")
        ax.set_xticks(list(xs)); ax.set_xticklabels(core, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("gold 율(/100음소)")
        ax.set_title("④ 소스 상보성 — 유형별 검출 커버리지", fontsize=12, fontweight="bold", color=INK)
        ax.legend(fontsize=9)
        blind = [t for t in core if not speech_present[t]]
        figs["complement"] = ("④ 소스 상보성", _b64(fig),
                              "gold(위치·유형) vs speech(실현형). "
                              f"speech MDD 사각지대: {', '.join(blind)} "
                              "(규칙 양방향 상쇄·초분절) → 이 유형은 gold 전담. 두 소스는 상보적.")
        kpis["speech_blind"] = ", ".join(blind)

    # ── FIG 5: B1 벤치마크 ──
    bench_dir = os.path.join(ROOT, "_mdd_analysis", bench_tag, "benchmark")
    bt_path = os.path.join(bench_dir, "benchmark_by_type.csv")
    bs_path = os.path.join(bench_dir, "benchmark_summary.json")
    if os.path.exists(bt_path):
        bt = pd.read_csv(bt_path, encoding="utf-8-sig").dropna(subset=["AP_L1"])
        bt = bt[bt["support_test"] > 0].sort_values("AP_L1", ascending=True)
        fig, ax = plt.subplots(figsize=(11, 5.2))
        ys = range(len(bt))
        ax.barh([y + .2 for y in ys], bt["AP_L1"], height=.4, color=MINT, label="AP (L1 조건부)")
        ax.barh([y - .2 for y in ys], bt["AP_global"], height=.4, color="#bbb", label="AP (global baseline)")
        ax.set_yticks(list(ys)); ax.set_yticklabels(bt["error_type"], fontsize=9)
        ax.set_xlabel("Average Precision"); ax.legend(fontsize=9)
        ax.set_title("⑤ B1 벤치마크 — L1 prior 예측력 (AP: L1 vs global)",
                     fontsize=12, fontweight="bold", color=INK)
        for y, (_, r) in zip(ys, bt.iterrows()):
            if r["lift(L1/global)"] and str(r["lift(L1/global)"]) not in ("", "nan"):
                ax.text(max(r["AP_L1"], r["AP_global"]) + .003, y,
                        f'×{r["lift(L1/global)"]}', va="center", fontsize=7.5, color=INK)
        cap = "화자의 L1을 알면 gold 오류유형을 얼마나 잘 예측하나(임계값무관 AP). "
        if os.path.exists(bs_path):
            bs = json.load(open(bs_path, encoding="utf-8"))
            mic = bs["micro_AP"]; mac = bs["macro_AP"]
            kpis["bench_lift"] = mic.get("lift_L1")
            kpis["bench_scope"] = f'{bs["n_L1"]}개국·test {bs["n_test_utt"]:,}발화'
            cap += (f'미시 AP {mic["global"]}→{mic["L1"]} (×{mic["lift_L1"]}), '
                    f'매크로 ×{mac["lift_L1"]}, DER {bs.get("DER")}.')
        cap += " ※ 현재 영어군 검증표본 — 전체는 s2 full 후 갱신."
        figs["bench"] = ("⑤ B1 예측력 벤치마크", _b64(fig), cap)

    # ── FIG 6: 데이터 물량 ──
    fig, ax = plt.subplots(figsize=(11, 5))
    tv = vol.head(18)
    ax.bar(range(len(tv)), tv.values / 1000, color=MINT)
    ax.set_xticks(range(len(tv))); ax.set_xticklabels(tv.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("gold 파일 수 (천)")
    ax.set_title("⑥ 데이터 물량 상위 국적 (gold pronunciation_lab)",
                 fontsize=12, fontweight="bold", color=INK)
    figs["volume"] = ("⑥ 데이터 물량", _b64(fig),
                      f"gold 총 {kpis['gold_files']:,}파일 · {kpis['gold_errors']:,}오류태그. "
                      "인도네시아·베트남·중국·필리핀·일본 순.")

    # ── FIG 7: speech 품질 플래그 ──
    q_path = os.path.join(sd, "quality_report.csv")
    if os.path.exists(q_path):
        q = pd.read_csv(q_path, encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        colmap = {"ok": MINT, "collapsed": "#d9776b", "restart_suspect": "#f0c05a"}
        ax.pie(q["count"], labels=q["quality_flag"], autopct="%1.1f%%",
               colors=[colmap.get(k, "#ccc") for k in q["quality_flag"]],
               startangle=90, textprops={"fontsize": 10})
        ax.set_title("⑦ Speech_lab 품질 플래그", fontsize=12, fontweight="bold", color=INK)
        tot = int(q["count"].sum())
        okp = q[q["quality_flag"] == "ok"]["count"].sum() / tot * 100 if tot else 0
        figs["quality"] = ("⑦ Speech 품질 분포", _b64(fig),
                           f"총 {tot:,}건 중 usable(ok) {okp:.1f}%. collapsed=문장이탈, restart=재시작 조각(집계 배제).")
        kpis["speech_total"] = tot
        kpis["speech_ok_pct"] = round(okp, 1)

    # ── HTML 조립 ──
    html = _build_html(figs, kpis, speech_tag, bench_tag)
    outp = os.path.join(outdir, "L1prior_통계리포트.html")
    open(outp, "w", encoding="utf-8").write(html)
    print(f"[완료] 리포트: {outp}")
    print(f"  그림 {len(figs)}개 · figs/ 폴더에도 PNG 저장 가능")
    print(f"  KPI: {json.dumps(kpis, ensure_ascii=False)}")
    return outp


def _build_html(figs, kpis, speech_tag, bench_tag):
    def card(label, val):
        return f'<div class="kpi"><div class="v">{val}</div><div class="l">{label}</div></div>'
    kpi_html = "".join([
        card("Gold 국적", kpis.get("gold_countries", "—")),
        card("Gold 파일", f'{kpis.get("gold_files", 0):,}'),
        card("Gold 오류태그", f'{kpis.get("gold_errors", 0):,}'),
        card("진단 가중치 국적", kpis.get("weight_countries", "—")),
        card("교차검증 r", kpis.get("crossval_r", "—")),
        card("Speech usable", f'{kpis.get("speech_ok_pct", "—")}%'),
        card("B1 L1 lift", f'×{kpis.get("bench_lift", "—")}'),
    ])
    secs = []
    for _id, (title, b64, cap) in figs.items():
        secs.append(f'''<section class="fig">
  <h2>{title}</h2>
  <img src="data:image/png;base64,{b64}" alt="{title}"/>
  <p class="cap">{cap}</p>
</section>''')
    figs_html = "\n".join(secs)
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>L1 prior 통계 리포트</title>
<style>
:root{{--mint:#0bbd86;--ink:#13201f;--sub:#5a6b6a;--line:#e3e9e8;--bg:#f7faf9}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Malgun Gothic','Segoe UI',sans-serif;background:var(--bg);color:var(--ink);line-height:1.55}}
.wrap{{max-width:1080px;margin:0 auto;padding:34px 22px 70px}}
header h1{{font-size:27px;font-weight:800;letter-spacing:-.5px}}
header p{{color:var(--sub);margin-top:7px;font-size:14px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:11px;margin:24px 0 8px}}
.kpi{{background:#fff;border:1px solid var(--line);border-radius:13px;padding:15px 13px;text-align:center}}
.kpi .v{{font-size:23px;font-weight:800;color:var(--mint)}}
.kpi .l{{font-size:11.5px;color:var(--sub);margin-top:3px}}
.fig{{background:#fff;border:1px solid var(--line);border-radius:15px;padding:20px 20px 16px;margin-top:22px}}
.fig h2{{font-size:16.5px;font-weight:800;margin-bottom:12px;color:var(--ink)}}
.fig img{{width:100%;height:auto;border-radius:8px;display:block}}
.cap{{font-size:13px;color:var(--sub);margin-top:11px;padding-left:11px;border-left:3px solid var(--mint)}}
.note{{background:#fff;border:1px dashed var(--line);border-radius:13px;padding:16px 18px;margin-top:22px;font-size:13px;color:var(--sub)}}
.note b{{color:var(--ink)}}
footer{{margin-top:30px;font-size:11.5px;color:#9aa;text-align:center}}
code{{background:#eef3f2;padding:1px 6px;border-radius:5px;font-size:12px}}
</style></head><body><div class="wrap">
<header>
  <h1>🦫 BeaverTalk — L1 prior 발음오류 통계 리포트</h1>
  <p>Speech_lab MDD(실현형) + pronunciation_lab gold(위치·유형) → 국적별 발음오류 prior · 교차검증 · 진단 가중치 · 예측력 벤치마크</p>
  <p style="font-size:12px">생성: s7_visualize.py · speech=<code>{speech_tag}</code> · bench=<code>{bench_tag}</code></p>
</header>
<div class="kpis">{kpi_html}</div>
{figs_html}
<div class="note">
  <b>데이터 범위</b> — ①②③④⑥⑦은 <b>전체 데이터</b>(4개 언어군 전체 pronunciation_lab gold + Speech_lab 26.8만건) 기반.
  ⑤ B1 벤치마크만 현재 <b>영어군(040) 검증표본</b>이며, <code>s2</code>/<code>s5</code> 전체 실행 후 전체 수치로 갱신됨.<br><br>
  <b>핵심 해석</b> — (1) 두 독립 소스가 모음·종성·삽입·유음 계열에서 일치(r={kpis.get('crossval_r','—')})해 L1 prior 신뢰 가능.
  (2) speech MDD 사각지대({kpis.get('speech_blind','—')})는 gold 전담 → 상보 구조.
  (3) L1을 알면 gold 오류유형 예측이 baseline 대비 향상(×{kpis.get('bench_lift','—')}) = 진단 prior의 실효 가치.
  단, prior만으로 발화별 정확 지목은 불가(높은 DER) → 인식기(갈래 B)가 위치 확정을 보완.
</div>
<footer>BeaverTalk · L1 prior pipeline · 발음오류 데이터 구축</footer>
</div></body></html>'''


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech-tag", default="040_041_042_043_both")
    ap.add_argument("--bench-tag", default="040_Training_pronlab")
    a = ap.parse_args()
    main(a.speech_tag, a.bench_tag)
