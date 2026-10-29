"""Compare the inactive adapter directly with the frozen operating-config app.

No preliminary TimePhraseBuilder or pair/auxiliary detector is enabled. Replay
every admitted time-phrase utterance; all other responses are cached unchanged
by construction. Existing TAG truth and all rejection denominators remain.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/train")]
import argparse
import collections
import json
import shutil
import eval_sibilant_directional as service
from eval_sibilant_typed_margin import app_from
from eval_sibilant_cv_agreement import detail
from eval_speech_rescue import observe
from train_sibilant_local_encoder import write
from mdd.canonical_time import TIME_PHRASE
from mdd.observed_reference_adapter import ObservedReferenceAdapter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    app = app_from(p)
    adapter = ObservedReferenceAdapter(app)
    report = {"scope": p["scope"], "deployment": False, "cohorts": {}}
    try:
        for name, c in p["cohorts"].items():
            rows, counts, audits = [], collections.Counter(), []
            with (out / (name + ".responses.jsonl")).open("x") as stream:
                for record in service.read_jsonl(c["source"]):
                    row, source_base = record["row"], record[c["base_key"]]
                    baseline = dict(record["original"])
                    candidate = dict(baseline)
                    if source_base["rejected"] is None and TIME_PHRASE.search(row["prompt"]):
                        result, trace = adapter.analyze_with_trace(service.load_audio(row)[:320000], row["prompt"], "")
                        if trace["forward_calls"] != 1:
                            raise AssertionError("extra acoustic forward")
                        before, after = trace["raw_before"], result[2]
                        baseline = {"raw": before, "body": service.jsonable(service.response(app, row, before)), "rejected": None}
                        candidate = {"raw": after, "body": service.jsonable(service.response(app, row, after)), "rejected": None}
                        if c["original_is_operating"] and baseline["body"] != record["original"]["body"]:
                            raise AssertionError("frozen operating response did not reproduce")
                        counts["time_utterances_reproduced"] += 1
                        counts[trace["reason"]] += 1
                        if trace["reference_changed"]:
                            counts["reference_changed"] += 1
                            audits.append({"uid": row["uid"], "accepted_positions": trace["accepted_positions"],
                                           "old_canonical": before["canon"], "new_canonical": after["canon"],
                                           "forward_calls": trace["forward_calls"], "acoustic_prediction_unchanged": True})
                    elif source_base["rejected"] is not None:
                        counts["input_rejected"] += 1
                    else:
                        counts["cached_unchanged_non_time"] += 1
                    if baseline["rejected"] != source_base["rejected"]:
                        raise AssertionError("input admission changed")
                    viewed = {"original": observe(baseline, row), "reference": observe(baseline, row),
                              "prior": observe(baseline, row), "candidate": observe(candidate, row)}
                    for site in row["sites"]:
                        value = dict(site, expected=site["canon"], uid=row["uid"], speaker=row["speaker"], item=row["item"], language=row.get("language"))
                        value.update({key: values[site["q"]] for key, values in viewed.items()})
                        rows.append(value)
                    service.write_line(stream, {"uid": row["uid"], "baseline": baseline["body"], "body": candidate["body"], "rejected": candidate["rejected"]})
            for suffix, values in (("sites", rows), ("reference_changes", audits)):
                with (out / (name + "." + suffix + ".jsonl")).open("x") as stream:
                    for value in values:
                        service.write_line(stream, value)
            result = detail(rows, counts)
            report["cohorts"][name] = result
            write(out / "report.json", report)
            print(name, json.dumps(result["versus_prior"], ensure_ascii=False), flush=True)
        write(out / "finished.json", {"sha256": service.sha(out / "report.json")})
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
