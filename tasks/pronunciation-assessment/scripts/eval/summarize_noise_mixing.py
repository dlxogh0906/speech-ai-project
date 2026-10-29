# -*- coding: utf-8 -*-
"""Report the frozen observation-adding holdout, including paired uncertainty."""

import sys as _sys
import pathlib as _pl

_root = _pl.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    if _root.parent == _root:
        raise RuntimeError("Repository root not found")
    _root = _root.parent
_sys.path[:0] = [str(_root), *(str(p) for p in sorted(_root.glob("tasks/*/src")))]

import argparse
import json
from collections import defaultdict

import numpy as np

from compare_noise_mixing import PRIMARY, SAFETY
from summarize_noise_preprocess import paired_deltas


def pooled_primary(rows, method):
    groups = defaultdict(list)
    for row in rows:
        if row["method"] == method and row["condition"] in PRIMARY:
            groups[row["k"]].append(row)
    sums = []
    for k in sorted(groups):
        records = groups[k]
        assert {r["condition"] for r in records} == set(PRIMARY)
        sums.append({"k": k, **{key: sum(r[key] for r in records) for key in (
            "api_tp", "n_err", "api_fp", "n_ok", "preserved", "jer_num", "jer_den")}})
    return sums


def report(directory, bootstraps=3000):
    directory = _pl.Path(directory)
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    valid = json.loads((directory / "valid/summary.json").read_text(encoding="utf-8"))
    test = json.loads((directory / "test/summary.json").read_text(encoding="utf-8"))
    audit = json.loads((directory / "split_audit.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "test/utterances.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == test["n_records"], "Incomplete or stale test summary"
    chosen = selection["chosen"]
    method = chosen["method"]
    clean = sorted([r for r in rows if r["condition"] == "clean" and r["method"] == "baseline"], key=lambda r: r["k"])
    rng = np.random.RandomState(20261011)
    strata = [[i for i, r in enumerate(clean) if r["cohort"] == cohort] for cohort in ("normal", "error")]
    draws = np.concatenate([rng.choice(s, (bootstraps, len(s)), replace=True) for s in strata if s], axis=1)
    primary = paired_deltas(pooled_primary(rows, "baseline"), pooled_primary(rows, method), draws)
    test_deltas = test["paired_vs_baseline"]
    point_safety = all(test_deltas[c][method][name]["delta_pp"] >= -2.0
                       for c in SAFETY for name in ("api_recall", "preservation"))
    clean_safe = test_deltas["clean"][method]["api_frr"]["delta_pp"] <= 0.3
    confirmed_frr_gain = primary["api_frr"]["ci95_pp"][1] < 0
    noninferiority = all(primary[name]["ci95_pp"][0] >= -2.0 for name in ("api_recall", "preservation"))
    if not selection["validation_supports_use"] or not point_safety or not clean_safe:
        verdict = "not_supported"
        message = "현재 혼합 설정의 적용을 뒷받침하지 못했다."
    elif confirmed_frr_gain and noninferiority:
        verdict = "supported_on_synthetic_holdout"
        message = "합성 소음 시험에서 헛지적 감소와 주 조건의 오류 검출/보존 유지가 확인됐다. 실제 부스 검증은 별도로 필요하다."
    else:
        verdict = "inconclusive"
        message = "개선 방향은 보이지만 불확실성이 남아 적용 근거가 충분하지 않다."
    decision = {"verdict": verdict, "message": message, "selection": chosen,
                "valid_supports_use": selection["validation_supports_use"],
                "test_point_safety_pass": point_safety, "clean_frr_safety_pass": clean_safe,
                "primary_frr_gain_ci_excludes_zero": confirmed_frr_gain,
                "primary_recall_preservation_noninferiority_2pp": noninferiority,
                "primary_paired_vs_baseline": primary,
                "primary_conditions": PRIMARY, "bootstraps": bootstraps,
                "uncertainty_scope": "Paired utterance bootstrap, normal/error stratified; not speaker-clustered",
                "production_changed": False}
    (directory / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["원음과 RNNoise 처리음 혼합 — 검증/시험 분리 실험", "",
             "판단: " + message,
             "고정 후보: 원음 %.0f%% + RNNoise 처리음 %.0f%%" % (100 * chosen["alpha"], 100 * (1 - chosen["alpha"])),
             "검증 %d개, 이전 전처리 비교에 쓰지 않은 시험 %d개." % (valid["n_utts"], test["n_utts"]),
             "각 집합은 정상/오발음 발화를 같은 수로 선정. 시험 결과를 보고 비율을 다시 고르지 않았다.",
             "시험 오류 자리 %d개 · 정상 자리 %d개." % (test["metrics"]["clean"]["baseline"]["error_sites"], test["metrics"]["clean"]["baseline"]["correct_sites"]),
             "검증/시험 음원 경로 겹침 0 · 잡음 음원 경로 겹침 0 · 이전 시험 %d개 제외." % audit["prior_excluded_samples"],
             "화자 겹침은 확인하지 않았으며 화자 메타데이터를 읽지 않았다.", "",
             "검증 데이터에서 후보 선택 (주 조건 = 웅성거림 10/5 dB)",
             "| 원음 비율 | 평균 API 헛지적 | 원음 대비 변화 %p | 안전 조건 통과 |",
             "|---:|---:|---:|---|"]
    for candidate in selection["candidates"]:
        lines.append("| %.0f%% | %.2f%% | %+.2f | %s |" % (
            100 * candidate["alpha"], 100 * candidate["primary_frr"],
            candidate["primary_frr_delta_pp"], candidate["valid_safety_pass"]))
    lines += ["선택 규칙: " + selection["policy"],
              "안전 조건을 만족하는 혼합이 없으면 75% 원음을 보수적인 진단 후보로만 시험하며 적용 권장은 하지 않는다.", "",
              "최종 시험 — 전체 입력을 채점한 결과", "검출/헛지적은 현재 API의 오류표시 규칙을 그대로 적용했다.",
              "보존은 사람이 들은 실제 오발음을 그대로 받아쓴 비율. 오류 존재 검출률과 다르다.",
              "| 조건 | 방법 | API 검출 | API 헛지적 | 오발음 보존 | JER |",
              "|---|---|---:|---:|---:|---:|"]
    for condition, metrics in test["metrics"].items():
        for name in ("baseline", "rnnoise", method):
            values = metrics[name]
            label = "원음" if name == "baseline" else "RNNoise만" if name == "rnnoise" else "원음 %.0f%% 혼합" % (100 * chosen["alpha"])
            lines.append("| %s | %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% |" % (
                condition, label, *(100 * values[k] for k in ("api_recall", "api_frr", "preservation", "jer"))))
    lines += ["", "시험 주 조건의 원음 대비 변화 — 발화 단위 paired bootstrap 95% 구간"]
    for label, key in (("API 헛지적", "api_frr"), ("API 오류 검출", "api_recall"),
                       ("오발음 보존", "preservation"), ("자모 JER", "jer")):
        value = primary[key]
        lines.append("- %s: %+.2f%%p [95%% CI %+.2f, %+.2f]" % (label, value["delta_pp"], *value["ci95_pp"]))
    lines += ["", "판단 조건", "- validation_supports_use: %s" % selection["validation_supports_use"],
              "- test_point_safety: %s" % point_safety, "- clean_frr_safety: %s" % clean_safe,
              "- primary_frr_gain_confirmed: %s" % confirmed_frr_gain,
              "- primary_recall/preservation_noninferiority_2pp: %s" % noninferiority, "",
              "해석 범위", "- 잡음 제거기의 고정 20 ms 지연을 보정한 뒤 원음과 섞었다. 각각 음량을 독립적으로 정규화하지 않았다.",
              "- 게이트로 어려운 발화를 빼서 성능을 높이지 않았다. 전체 쌍을 채점했으며 거절률은 하위 report.txt에 별도로 있다.",
              "- 합성 웅성거림/저주파 소음, 기존 학습 문장에 대한 결과다. 실제 부스 녹음 및 처음 보는 문장 결과가 아니다.",
              "- 정상/오발음 균형 표본이므로 실제 서비스의 오류 정밀도/빈도를 추정하지 않는다.",
              "- bootstrap은 발화 단위이며 같은 화자/문장 반복의 상관을 반영하지 않는다.",
              "- CPU 제거음을 비율 간 재사용했다. 시간 통계는 요청 지연이나 NPU 서비스 성능 측정으로 해석하지 않는다.",
              "- 코드/라이브러리/보정표 hash와 후보 선택은 experiment.json 및 selection.json에 남겼다.",
              "- 운영 API 파일, 모델, 보정표, 서버 재시작은 변경하지 않았다."]
    (directory / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9]), flush=True)
    print("[saved] " + str(directory / "report.txt"), flush=True)
    return decision


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory")
    ap.add_argument("--bootstraps", type=int, default=3000)
    args = ap.parse_args()
    report(args.directory, args.bootstraps)


if __name__ == "__main__":
    main()
