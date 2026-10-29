# -*- coding: utf-8 -*-
"""Aggregate paired preprocessing results and utterance-bootstrap differences."""

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np


def ratio(rows, numerator, denominator):
    den = sum(r[denominator] for r in rows)
    return sum(r[numerator] for r in rows) / den if den else None


def aggregate(rows):
    normals = [r for r in rows if r["cohort"] == "normal"]
    errors = [r for r in rows if r["cohort"] == "error"]
    result = {"n": len(rows), "n_normal": len(normals), "n_error": len(errors),
              "error_sites": sum(r["n_err"] for r in rows),
              "correct_sites": sum(r["n_ok"] for r in rows),
              "raw_recall": ratio(rows, "raw_tp", "n_err"),
              "raw_frr": ratio(rows, "raw_fp", "n_ok"),
              "api_recall": ratio(rows, "api_tp", "n_err"),
              "api_frr": ratio(rows, "api_fp", "n_ok"),
              "api_normal_frr": ratio(normals, "api_fp", "n_ok"),
              "preservation": ratio(rows, "preserved", "n_err"),
              "reversion": ratio(rows, "reverted", "n_err"),
              "jer": ratio(rows, "jer_num", "jer_den"),
              "insertion_ratio": ratio(rows, "insertions", "n_canon"),
              "normal_utterance_false_flag": float(np.mean([r["api_flagged_utterance"] for r in normals])) if normals else None,
              "error_utterance_detected": float(np.mean([r["api_tp"] > 0 for r in errors])) if errors else None,
              "original_gate_reject": float(np.mean([not r["original_check_pass"] for r in rows])),
              "post_denoise_gate_reject": float(np.mean([not r["post_denoise_check_pass"] for r in rows])),
              "silero_no_speech": float(np.mean([r["silero_no_speech"] for r in rows]))}
    if rows[0]["method"] == "silero":
        result["silero_gate_reject"] = float(np.mean([not r["silero_check_pass"] for r in rows]))
    for key in ("prep_ms", "model_ms", "total_ms", "processed_sec", "cut_sec"):
        values = [r[key] for r in rows]
        result[key + "_p50"] = float(np.percentile(values, 50))
        result[key + "_p95"] = float(np.percentile(values, 95))
    accepted = [r for r in rows if (r["silero_check_pass"] if r["method"] == "silero"
                                  else r["post_denoise_check_pass"])]
    result["prospective_gate_accepted"] = {"n": len(accepted),
        "api_recall": ratio(accepted, "api_tp", "n_err"),
        "api_frr": ratio(accepted, "api_fp", "n_ok")}
    return result


def paired_deltas(baseline, candidate, bootstrap_indices):
    base = {r["k"]: r for r in baseline}
    alt = {r["k"]: r for r in candidate}
    keys = sorted(base)
    assert keys == sorted(alt), "Paired sample mismatch"
    metrics = (("api_recall", "api_tp", "n_err"),
               ("api_frr", "api_fp", "n_ok"),
               ("preservation", "preserved", "n_err"),
               ("jer", "jer_num", "jer_den"))
    result = {}
    for name, numerator, denominator in metrics:
        den = np.array([base[k][denominator] for k in keys], dtype=float)
        num = np.array([alt[k][numerator] - base[k][numerator] for k in keys], dtype=float)
        boot_den = den[bootstrap_indices].sum(axis=1)
        boot = num[bootstrap_indices].sum(axis=1) / np.maximum(boot_den, 1)
        result[name] = {"delta_pp": float(100 * num.sum() / max(1, den.sum())),
                       "ci95_pp": [float(x) for x in 100 * np.percentile(boot, (2.5, 97.5))]}
    return result


def summarize(directory, bootstraps=3000):
    directory = Path(directory)
    metadata = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "utterances.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = metadata["n_utts"] * len(metadata["conditions"]) * len(metadata["methods"])
    if len(rows) != expected:
        raise RuntimeError("Incomplete experiment: %d records, expected %d" % (len(rows), expected))
    groups = collections.defaultdict(list)
    truth = {}
    seen = set()
    for row in rows:
        identity = (row["k"], row["condition"], row["method"])
        assert identity not in seen, "Duplicate result"
        seen.add(identity)
        invariant = (row["cohort"], row["n_err"], row["n_ok"], row["jer_den"])
        assert truth.setdefault(row["k"], invariant) == invariant, "Truth changed across pairs"
        groups[(row["condition"], row["method"])].append(row)
    assert len(truth) == metadata["n_utts"]
    clean = sorted(groups[("clean", "baseline")], key=lambda r: r["k"])
    assert all(r["original_check_pass"] for r in clean), "Clean selection gate mismatch"
    rng = np.random.RandomState(20261009)
    strata = [[i for i, r in enumerate(clean) if r["cohort"] == cohort]
              for cohort in ("normal", "error")]
    draws = np.concatenate([rng.choice(s, (bootstraps, len(s)), replace=True)
                            for s in strata if s], axis=1)
    aggregates, deltas, flat = {}, {}, []
    for condition, *_ in metadata["conditions"]:
        aggregates[condition] = {}
        deltas[condition] = {}
        for method in metadata["methods"]:
            result = aggregate(groups[(condition, method)])
            aggregates[condition][method] = result
            flat.append({"condition": condition, "method": method,
                         **{k: v for k, v in result.items() if not isinstance(v, dict)}})
            if method != "baseline":
                deltas[condition][method] = paired_deltas(groups[(condition, "baseline")],
                                                        groups[(condition, method)], draws)
    summary = {"n_utts": metadata["n_utts"], "n_records": len(rows),
               "strata": dict(collections.Counter(r["cohort"] for r in clean)),
               "scope": metadata["scope"], "selection": metadata["selection"],
               "bootstrap": "%d paired resamples by original utterance, stratified by normal/error; not speaker-clustered" % bootstraps,
               "metrics": aggregates, "paired_vs_baseline": deltas,
               "limitations": [
                   "Balanced normal/error cohorts are not real traffic prevalence; precision is not estimated.",
                   "Synthetic six-talker babble and synthetic low-frequency rumble; no real booth/app recordings.",
                   "Same trained-sentence corpus; human actual-phone labels, not deliberately corrupted clean audio.",
                   "Production phone-flag criterion reproduced; no browser/HTTP/NPU end-to-end evaluation.",
                   "All paired inputs scored, including prospective rejects; gate acceptance is reported separately.",
                   "WebRTC condition simulates suppression only; actual browser settings, AGC and echo cancellation are not tested.",
                   "Bootstrap by utterance does not account for potentially repeated speakers/prompts."]}
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    lines = ["발음평가 전처리 비교 — " + str(metadata["n_utts"]) + "개 발화, " + str(len(rows)) + "개 조건 결과",
             "정상/오발음 균형 표본. 원본이 기존 low_speech 검사를 통과한 경우만 선정.",
             "검출 = 실제 오류 자리 중 API 오류표시 기준으로 지적한 비율. 헛지적 = 정상 자리 중 지적한 비율.",
             "실제 오발음을 들린 그대로 보존한 비율과 오류 존재 검출은 다르다.",
             "아래 표는 재녹음 후보를 포함한 전체 쌍을 채점한 결과이며, 거절률은 별도로 표시한다.", "",
             "| 조건 | 방법 | API 검출 | API 헛지적 | 오발음 보존 | JER | 정상발화 헛지적 | 전처리 p50/p95 ms | 전체 p50 ms |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def pct(value):
        return "—" if value is None else "%.1f%%" % (100 * value)
    for row in flat:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %.1f / %.1f | %.1f |" % (
            row["condition"], row["method"], pct(row["api_recall"]), pct(row["api_frr"]),
            pct(row["preservation"]), pct(row["jer"]), pct(row["normal_utterance_false_flag"]),
            row["prep_ms_p50"], row["prep_ms_p95"], row["total_ms_p50"]))
    lines += ["", "기존 방식 대비 변화: %p [발화 단위 paired bootstrap 95% 구간]",
              "| 조건 | 방법 | API 검출 변화 | API 헛지적 변화 | 오발음 보존 변화 | JER 변화 |",
              "|---|---|---:|---:|---:|---:|"]
    def delta(result):
        return "%+.1f [%+.1f, %+.1f]" % (result["delta_pp"], *result["ci95_pp"])
    for condition, methods in deltas.items():
        for method, values in methods.items():
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                condition, method, *(delta(values[key]) for key in ("api_recall", "api_frr", "preservation", "jer"))))
    lines += ["", "거절률 — 정확도 개선과 구별해야 한다.",
              "| 조건 | 방법 | 기존 입력 검사 거절 | 처리 후 에너지 검사 거절 | Silero 발화시간 검사 거절 | Silero 말소리 없음 |",
              "|---|---|---:|---:|---:|---:|"]
    for row in flat:
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            row["condition"], row["method"], pct(row["original_gate_reject"]),
            pct(row["post_denoise_gate_reject"]), pct(row.get("silero_gate_reject")),
            pct(row["silero_no_speech"])))
    lines += ["", "한계:"] + ["- " + item for item in summary["limitations"]]
    (directory / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6] + ["[saved] " + str(directory / "report.txt")]))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory")
    ap.add_argument("--bootstraps", type=int, default=3000)
    args = ap.parse_args()
    summarize(args.directory, args.bootstraps)


if __name__ == "__main__":
    main()
