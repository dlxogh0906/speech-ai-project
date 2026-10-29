# -*- coding: utf-8 -*-
"""속도·쉼만 보는 유창성 점수 — 발음과 **독립된** 칸을 만들기 위한 규칙.

배경: AIHub 사람 점수 세 칸은 서로 상관 0.84(화자 단위 0.90)다. 발음 4.5+ 발화에서 유창성
3점 미만은 0.0%. 그 라벨을 예측하는 회귀기는 「또박또박 읽었지만 틀리게 읽은 녹음」의
유창성을 바닥으로 준다(특징에서 발음 항목을 다 빼도 2.88 vs 2.80 — 라벨 자체가 그렇다).
외부 API 는 같은 녹음에 92~98점을 준다. 그래서 유창성 칸만 이 규칙으로 따로 계산한다.

기준 구간은 `scripts/score/fit_fluency_rule.py` 가 만든 `fluency_rule.json`.
사람 라벨과의 상관은 발화 0.441 · 화자 0.678 로 회귀기(0.610/0.762)보다 낮다 — 대신
발음과 섞이지 않는다(발음 2.0 이하 발화 평균 82.9 · 발음 4.5+ 평균 97.0).
"""

REF_FLU = 4.5           # 이 점수 이상 발화를 "유창한 읽기"의 기준으로 삼는다
# (특징, 벌점 가중, 방향) — slow: 기준보다 느리면 / high: 기준보다 크면 벌점
RULES = [("artic_rate", 45, "slow"), ("pause_frac", 30, "high"),
         ("max_pause", 15, "high"), ("pause_rate", 10, "high")]


def band_of(rows, percentile):
    """기준 발화들의 특징 분포. percentile(values, q) 는 호출측이 넣는다(numpy 의존 회피)."""
    out = {}
    for k, _, _ in RULES:
        v = [float(r[k]) for r in rows if r.get(k) is not None]
        out[k] = {"p10": percentile(v, 10), "med": percentile(v, 50), "p90": percentile(v, 90)}
    return out


def fluency_rule(feats, band):
    """0~100. 느리거나 쉼이 길고 잦으면 깎는다. **빠른 것은 벌하지 않는다.**"""
    pen = 0.0
    for k, w, direction in RULES:
        v = feats.get(k)
        b = band.get(k)
        if v is None or not b:
            continue
        v = float(v)
        if direction == "slow" and v < b["p10"]:
            pen += w * min(1.0, (b["p10"] - v) / max(1e-6, b["p10"] * 0.6))
        elif direction == "high" and v > b["p90"]:
            pen += w * min(1.0, (v - b["p90"]) / max(1e-6, b["p90"]))
    return max(0.0, min(100.0, 100.0 - pen))
