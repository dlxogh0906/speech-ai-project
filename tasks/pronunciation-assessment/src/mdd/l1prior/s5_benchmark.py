# -*- coding: utf-8 -*-
"""
s5_benchmark.py
~~~~~~~~~~~~~~~
MDD/L1-prior 벤치마크 정량화 (작업지시서 6절 · 로드맵 지표 P·R·F1·FAR·FRR·DER).

■ 왜 이 형태인가 (데이터 제약의 정직한 반영)
  로드맵의 "MDD F1"은 <인식기 발음열 ↔ 표준 G2P ↔ 전문가 전사> 3자 비교로 정의된다.
  그런데 우리에겐 아직 인식기(갈래 B)가 없고, gold(pronunciation_lab)는
    · error_tag(유형) + 시간창(위치)만 주고 '실제 오발음 음소'는 기록하지 않으며
    · phones 티어는 표준음(실측 확인)이라 실현형이 아니다.
  또한 Speech_lab(LAR)과 gold(RW/RS/RP)는 서로 다른 발화라 phone 단위 매칭 불가.
  → 인식기 없이 계산 가능한 '진짜' 벤치마크 = **L1 prior 의 예측력 검증**:
     "화자의 L1을 알면, 그 발화에 어떤 오류유형이 나타날지를
      전역(무지) baseline 보다 잘 맞히는가?"  = 제품(진단 prior)의 실효 가치.

■ 설계
  단위 = (발화 u, 오류유형 t).  positive = gold 에서 u 가 t 를 가짐.
  예측기:
    GLOBAL : P(t)         (L1 무시, marginal)
    L1     : P(t | L1)
    L1+PROF: P(t | L1, proficiency)   (선택)
  train/test = utterance_id 해시로 결정적 분할(누수 없음).
  지표:
    AP(평균정밀도, 임계값 무관) — GLOBAL vs L1 비교 → **lift = AP_L1 / AP_global**
    임계값 θ*(train F1 최대)에서 P·R·F1·FAR·FRR
    DER(진단오류율) = 양측이 오류를 지목한 발화에서 유형집합 불일치(1 - 평균 Jaccard)

지표 정의(리포트에도 기재):
  P=TP/(TP+FP), R=TP/(TP+FN), F1=2PR/(P+R)
  FRR(오거부, 과지적)=FP/(TN+FP)=정상을 오류로 지목할 확률
  FAR(오수용, 미검출)=FN/(TP+FN)=실제 오류를 놓칠 확률(=1-Recall)
  DER=양측 오류지목 발화의 유형집합 진단 불일치율

입력:
  _mdd_analysis/<pron_tag>/l1_utterance_labels.csv   (s2 산출; 발화별 gold)
출력(_mdd_analysis/<pron_tag>/benchmark/):
  benchmark_by_type.csv     유형별 support/AP_global/AP_L1/lift/θ*/P/R/F1/FAR/FRR
  benchmark_by_L1.csv       L1별 AP(global vs L1) + lift
  benchmark_summary.md      요약 + 지표 정의
  benchmark_summary.json    기계용 요약

실행:
  python s5_benchmark.py --pron-tag 040_041_042_043_both_pronlab --min-utt 300
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import os
import csv
import json
import hashlib
import argparse
import collections
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 하위 -> 데이터 루트는 부모
from mdd.l1prior.s0_taxonomy import CORE_TAGS  # noqa: E402

TEST_FRAC = 0.30


def _is_test(uid):
    """utterance_id 해시로 결정적 test 분할(약 TEST_FRAC)."""
    h = int(hashlib.md5(str(uid).encode("utf-8")).hexdigest()[:8], 16)
    return (h % 1000) < int(TEST_FRAC * 1000)


def average_precision(scores, labels):
    """PR 곡선 아래 면적(평균정밀도). scores 큰 순 정렬 후 표준 정의."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    y = np.asarray(labels, dtype=float)[order]
    P = y.sum()
    if P == 0:
        return float("nan")
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / P
    # ΔRecall 가중 정밀도 합(계단식)
    drec = np.diff(np.concatenate([[0.0], rec]))
    return float(np.sum(prec * drec))


def prf_at_threshold(scores, labels, thr):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    pred = (s >= thr).astype(float)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    FRR = fp / (tn + fp) if (tn + fp) else 0.0   # 과지적(정상→오류)
    FAR = fn / (tp + fn) if (tp + fn) else 0.0   # 미검출(오류 놓침)=1-R
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, P=P, R=R, F1=F1, FRR=FRR, FAR=FAR)


def best_threshold(scores, labels):
    """train F1 최대 임계값(후보=고유 score)."""
    cand = sorted(set(np.round(np.asarray(scores, dtype=float), 6)))
    if not cand:
        return 0.5, 0.0
    best_t, best_f = cand[0], -1.0
    for t in cand:
        f = prf_at_threshold(scores, labels, t)["F1"]
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


def main(pron_tag, min_utt):
    d = os.path.join(ROOT, "_mdd_analysis", pron_tag)
    outdir = os.path.join(d, "benchmark")
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(os.path.join(d, "l1_utterance_labels.csv"),
                     encoding="utf-8-sig")
    df["types"] = df["err_types_core"].fillna("").apply(
        lambda s: set(t for t in str(s).split("|") if t))
    df["is_test"] = df["utterance_id"].apply(_is_test)

    # L1 필터: train·test 최소 발화수
    keep = []
    for L1, g in df.groupby("L1"):
        if (g["is_test"] == False).sum() >= min_utt and (g["is_test"]).sum() >= max(50, min_utt // 5):
            keep.append(L1)
    df = df[df["L1"].isin(keep)].copy()
    tr = df[~df["is_test"]]
    te = df[df["is_test"]]
    print(f"[*] L1 {len(keep)}개국 | train {len(tr):,} / test {len(te):,} 발화")

    types = [t for t in CORE_TAGS]

    # ── 사전확률 학습(train) ──
    n_tr = len(tr)
    global_p = {t: (tr["types"].apply(lambda s: t in s).sum() / n_tr) for t in types}
    l1_p = {}     # (L1,t) -> prob
    l1_prof_p = {}
    for L1, g in tr.groupby("L1"):
        ng = len(g)
        for t in types:
            l1_p[(L1, t)] = g["types"].apply(lambda s: t in s).sum() / ng
    for (L1, prof), g in tr.groupby(["L1", "proficiency"]):
        ng = len(g)
        for t in types:
            l1_prof_p[(L1, prof, t)] = g["types"].apply(lambda s: t in s).sum() / ng

    # ── test 평가: (발화,유형) 단위 ──
    # 각 예측기의 score/label 을 유형별로 모음
    per_type = {t: {"g_sc": [], "l_sc": [], "p_sc": [], "y": []} for t in types}
    for _, r in te.iterrows():
        L1, prof, gold = r["L1"], r["proficiency"], r["types"]
        for t in types:
            per_type[t]["y"].append(1 if t in gold else 0)
            per_type[t]["g_sc"].append(global_p[t])
            per_type[t]["l_sc"].append(l1_p.get((L1, t), global_p[t]))
            per_type[t]["p_sc"].append(
                l1_prof_p.get((L1, prof, t), l1_p.get((L1, t), global_p[t])))

    # θ* 는 train 에서 L1 예측기로 산출
    tr_type = {t: {"l_sc": [], "y": []} for t in types}
    for _, r in tr.iterrows():
        L1, gold = r["L1"], r["types"]
        for t in types:
            tr_type[t]["y"].append(1 if t in gold else 0)
            tr_type[t]["l_sc"].append(l1_p.get((L1, t), global_p[t]))

    rows = []
    micro = {"g": [], "l": [], "p": [], "y": []}
    for t in types:
        pt = per_type[t]
        support = int(np.sum(pt["y"]))
        ap_g = average_precision(pt["g_sc"], pt["y"])
        ap_l = average_precision(pt["l_sc"], pt["y"])
        ap_p = average_precision(pt["p_sc"], pt["y"])
        thr, _ = best_threshold(tr_type[t]["l_sc"], tr_type[t]["y"])
        m = prf_at_threshold(pt["l_sc"], pt["y"], thr)
        lift = (ap_l / ap_g) if (ap_g and ap_g == ap_g and ap_g > 0) else float("nan")
        rows.append([t, support,
                     round(ap_g, 4), round(ap_l, 4), round(ap_p, 4),
                     round(lift, 3) if lift == lift else "",
                     round(thr, 4), round(m["P"], 3), round(m["R"], 3),
                     round(m["F1"], 3), round(m["FAR"], 3), round(m["FRR"], 3)])
        micro["y"] += pt["y"]; micro["g"] += pt["g_sc"]
        micro["l"] += pt["l_sc"]; micro["p"] += pt["p_sc"]

    with open(os.path.join(outdir, "benchmark_by_type.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["error_type", "support_test", "AP_global", "AP_L1", "AP_L1prof",
                    "lift(L1/global)", "theta*", "P", "R", "F1", "FAR", "FRR"])
        w.writerows(rows)

    # 미시 평균 AP(모든 유형 pool)
    ap_g_micro = average_precision(micro["g"], micro["y"])
    ap_l_micro = average_precision(micro["l"], micro["y"])
    ap_p_micro = average_precision(micro["p"], micro["y"])
    # 매크로 = 유형별 AP 평균
    macro_g = np.nanmean([r[2] for r in rows])
    macro_l = np.nanmean([r[3] for r in rows])
    macro_p = np.nanmean([r[4] for r in rows])

    # ── DER: 양측 오류지목 발화의 유형집합 진단 불일치 ──
    #  예측 유형집합(L1 예측기, θ*) vs gold 유형집합. 둘 다 비어있지 않은 발화만.
    thr_by_t = {r[0]: r[6] for r in rows}
    jacc = []
    exact = 0; both_n = 0
    for _, r in te.iterrows():
        L1, gold = r["L1"], r["types"]
        pred = set(t for t in types if l1_p.get((L1, t), global_p[t]) >= thr_by_t[t])
        if gold and pred:
            both_n += 1
            inter = len(gold & pred); uni = len(gold | pred)
            j = inter / uni if uni else 0
            jacc.append(j)
            if gold == pred:
                exact += 1
    der = 1 - float(np.mean(jacc)) if jacc else float("nan")

    # ── L1별 AP ──
    l1_rows = []
    for L1, g in te.groupby("L1"):
        gs, ls, ys = [], [], []
        gi = g.reset_index(drop=True)
        for _, r in gi.iterrows():
            gold = r["types"]
            for t in types:
                ys.append(1 if t in gold else 0)
                gs.append(global_p[t])
                ls.append(l1_p.get((L1, t), global_p[t]))
        apg = average_precision(gs, ys); apl = average_precision(ls, ys)
        lift = (apl / apg) if (apg and apg > 0) else float("nan")
        l1_rows.append([L1, len(g), round(apg, 4), round(apl, 4),
                        round(lift, 3) if lift == lift else ""])
    l1_rows.sort(key=lambda x: (-(x[4] if isinstance(x[4], float) else -1)))
    with open(os.path.join(outdir, "benchmark_by_L1.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["L1", "test_utt", "AP_global", "AP_L1", "lift(L1/global)"])
        w.writerows(l1_rows)

    summary = {
        "pron_tag": pron_tag, "n_L1": len(keep),
        "n_train_utt": int(n_tr), "n_test_utt": int(len(te)),
        "micro_AP": {"global": round(ap_g_micro, 4), "L1": round(ap_l_micro, 4),
                     "L1prof": round(ap_p_micro, 4),
                     "lift_L1": round(ap_l_micro / ap_g_micro, 3) if ap_g_micro else None},
        "macro_AP": {"global": round(float(macro_g), 4), "L1": round(float(macro_l), 4),
                     "L1prof": round(float(macro_p), 4),
                     "lift_L1": round(float(macro_l / macro_g), 3) if macro_g else None},
        "DER": round(der, 4) if der == der else None,
        "DER_both_flagged_utt": both_n,
        "DER_exact_match_rate": round(exact / both_n, 4) if both_n else None,
    }
    with open(os.path.join(outdir, "benchmark_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── 리포트 ──
    L = []
    L.append("# MDD / L1-prior 벤치마크\n")
    L.append(f"- 대상: `{pron_tag}` · L1 {len(keep)}개국 · train {n_tr:,} / test {len(te):,} 발화")
    L.append(f"- 단위 = (발화 × 오류유형). positive = gold 에 해당 유형 존재.\n")
    L.append("## 핵심 결과 — L1 prior 의 예측 가치")
    L.append(f"- **미시 평균정밀도(AP): global {ap_g_micro:.3f} → L1 {ap_l_micro:.3f} "
             f"(lift ×{ap_l_micro/ap_g_micro:.2f})**" if ap_g_micro else "")
    L.append(f"- 매크로 AP: global {macro_g:.3f} → L1 {macro_l:.3f} → L1+prof {macro_p:.3f}")
    L.append(f"- DER(진단오류율, 양측 오류지목 {both_n:,}발화): {der:.3f} "
             f"(정확일치율 {exact/both_n:.3f})" if both_n else "")
    L.append("\n## 유형별 (θ*=train F1 최대)")
    L.append("| 유형 | support | AP_global | AP_L1 | lift | P | R | F1 | FAR | FRR |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in sorted(rows, key=lambda x: -(x[3] if x[3] == x[3] else 0)):
        L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[5]} | "
                 f"{r[7]} | {r[8]} | {r[9]} | {r[10]} | {r[11]} |")
    L.append("\n## 지표 정의")
    L.append("- **AP**: PR곡선 아래 면적(임계값 무관). global=L1 무시, L1=P(유형|국적).")
    L.append("- **lift = AP_L1 / AP_global** > 1 이면 L1 prior 가 예측에 기여.")
    L.append("- **P/R/F1**: 표준. **FAR**(오수용,미검출)=FN/(TP+FN)=1−R. "
             "**FRR**(오거부,과지적)=FP/(TN+FP).")
    L.append("- **DER**: 양측이 오류를 지목한 발화에서 유형집합 진단 불일치(1−평균 Jaccard).")
    L.append("\n## 한계(정직한 기재)")
    L.append("- 인식기(갈래 B) 부재로 **phone 단위 MDD F1(인식↔전사)** 은 아직 산출 불가.")
    L.append("- 본 벤치마크는 그 대체로서 **L1 prior 예측력**을 정량화 — 제품(진단 가중치) 가치의 직접 근거.")
    L.append("- gold=pronunciation_lab(위치·유형), 실현형은 Speech_lab 담당(상보). 두 축은 다른 발화.")
    open(os.path.join(outdir, "benchmark_summary.md"), "w",
         encoding="utf-8").write("\n".join(x for x in L if x is not None))

    print(f"[완료] {outdir}")
    print(f"  미시 AP: global {ap_g_micro:.3f} -> L1 {ap_l_micro:.3f} "
          f"(lift x{ap_l_micro/ap_g_micro:.2f})" if ap_g_micro else "")
    print(f"  매크로 AP: global {macro_g:.3f} -> L1 {macro_l:.3f} -> L1+prof {macro_p:.3f}")
    print(f"  DER {der:.3f} (both-flagged {both_n:,}, exact {exact/both_n:.3f})" if both_n else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pron-tag", default="040_041_042_043_both_pronlab")
    ap.add_argument("--min-utt", type=int, default=300,
                    help="L1별 최소 train 발화수(이하 제외)")
    a = ap.parse_args()
    main(a.pron_tag, a.min_utt)
