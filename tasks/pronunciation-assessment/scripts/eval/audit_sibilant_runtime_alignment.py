"""Audit runtime edit-alignment ties using freshly captured complete hypotheses.

Uses calibration by default. It does not rewrite any service decision and does
not infer missing inserted tokens from cached aligned sites.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

import argparse
import collections
import json
import shutil
import time

import eval_sibilant_directional as service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--service-metadata", required=True)
    parser.add_argument("--split", choices=["cal", "test"], default="cal")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, out = Path(args.baseline_dir), Path(args.out)
    meta = json.loads(Path(args.service_metadata).read_text())["service_metadata"]
    for p, digest in dict(meta["source_sha256"], **meta["scoring_sha256"]).items():
        if service.sha(p) != digest:
            raise ValueError("baseline source/scorer changed")
    records = service.read_jsonl(source / (args.split + ".responses.jsonl"))
    labels = service.read_jsonl(source / (args.split + ".labels.jsonl"))
    by_uid = collections.defaultdict(list)
    for row in labels:
        by_uid[row["uid"]].append(row)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    protocol = {"scope": __doc__, "split": args.split, "deployment": False,
                "source_sha256": {str(p): service.sha(p) for p in [Path(__file__), Path(args.service_metadata),
                     source / (args.split + ".responses.jsonl"), source / (args.split + ".labels.jsonl")]},
                "hypothesis": "A single minimum-edit alignment can place a correct fricative at another reference position; test whether current false alerts have alternative equally optimal normal readings.",
                "limits": "Edit ties are not independent acoustic evidence. Any later uncertainty policy must measure lost real errors and new abstentions. No policy is selected by test here."}
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    paths = {Path(p).name: p for p in meta["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"], norms=paths["norms.json"],
                               calib=paths["calibration.json"], flu_rule=paths["fluency_rule.json"], utt_pct=paths["utt_percentile.json"])
    app = service.app_module(config, meta["base_ckpt"])
    rows, reproduced, started = [], 0, time.monotonic()
    with (out / "hypotheses.jsonl").open("w") as stream:
        for number, record in enumerate(records, 1):
            options, hyp = None, None
            if record["raw"] is not None:
                row = record["row"]
                wav = service.load_audio(row)[:320000]
                result, _, raw = app.analyze(wav, row["prompt"], "")
                if service.jsonable(service.response(app, row, raw)) != record["body"]:
                    raise AssertionError("cached API body not reproduced")
                if raw["canon"] != record["raw"]["canon"]:
                    raise AssertionError("runtime canonical changed")
                hyp = result["pred_jamo"]
                options = service.all_optimal_mapping(raw["canon"], hyp)
                if any(site["j"] not in options[site["i"]] for site in raw["sites"]):
                    raise AssertionError("service alignment absent from optimal-path set")
                reproduced += 1
                service.write_line(stream, {"uid": record["uid"], "canonical": raw["canon"], "hypothesis": hyp,
                    "optimal_actual_options": [sorted({None if j is None else hyp[j] for j in opts}, key=str) for opts in options]})
            for label in by_uid[record["uid"]]:
                row = dict(label)
                q = row["q_runtime"]
                tokens = {None if j is None else hyp[j] for j in options[q]} if options is not None and q is not None else set()
                runtime_expected = record["raw"]["canon"][q] if options is not None and q is not None else None
                row.update(runtime_actual_options=sorted(tokens, key=str), reading_ambiguous=len(tokens) > 1,
                           runtime_expected=runtime_expected, expected_is_option=runtime_expected in tokens)
                rows.append(row)
            if number % 100 == 0 or number == len(records):
                print(args.split, number, "reproduced", reproduced, round(time.monotonic() - started, 1), flush=True)
    with (out / "sites.jsonl").open("w") as stream:
        for row in rows:
            service.write_line(stream, row)
    report = {"split": args.split, "accepted_responses_reproduced": reproduced, "groups": {}}
    for expected in ("ㅅ", "ㅆ"):
        for kind in ("normal", "error"):
            group = [r for r in rows if r["expected"] == expected and (r["actual"] == expected) == (kind == "normal")]
            flagged = [r for r in group if (r.get("baseline_phone") or {}).get("errorType")]
            correct = [r for r in flagged if (r["baseline_phone"].get("alpha_jamo") == r["expected"]
                                             and r["baseline_phone"].get("actual_jamo") == r["actual"])] if kind == "error" else []
            report["groups"][expected + "/" + kind] = {
                "all_sites": len(group), "baseline_alerts": len(flagged), "correct_error_alerts": len(correct),
                "ambiguous_alerts": sum(r["reading_ambiguous"] for r in flagged),
                "ambiguous_alerts_with_normal_option": sum(r["reading_ambiguous"] and r["expected_is_option"] for r in flagged),
                "correct_error_alerts_with_normal_option": sum(r["reading_ambiguous"] and r["expected_is_option"] for r in correct)}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"report_sha256": service.sha(out / "report.json")}) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
