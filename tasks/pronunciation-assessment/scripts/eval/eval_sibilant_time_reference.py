"""Compare inactive time-reference candidate through the original API pipeline.

Runtime selection sees text and frozen CTC logits only. TAG labels are used for
counting after responses. Reused exploratory data; no recognition-gain claim.
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

import numpy as np
import eval_sibilant_directional as service
from eval_speech_rescue import observe, metrics
from mdd.canonical_time import TimePhraseBuilder, TIME_PHRASE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    p = json.loads(Path(args.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    source, out = Path(p["features"]), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, out / "protocol.json")
    for path in (Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/canonical_time.py"):
        shutil.copy2(path, out / path.name)
    meta = json.loads((source / "metadata.json").read_text())["service_metadata"]
    for path, digest in dict(meta["source_sha256"], **meta["scoring_sha256"]).items():
        if service.sha(path) != digest:
            raise ValueError("baseline source changed: " + path)
    paths = {Path(k).name: k for k in meta["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"],
                               norms=paths["norms.json"], calib=paths["calibration.json"],
                               flu_rule=paths["fluency_rule.json"], utt_pct=paths["utt_percentile.json"])
    app = service.app_module(config, meta["base_ckpt"])
    original_builder, forward = app.S["cb"], app.S["fwd"]
    candidate_builder, captured = TimePhraseBuilder(original_builder), {}

    def capture(dec, wavs):
        result = forward(dec, wavs)
        captured.update(wav=wavs[0].copy(), result=result)
        return result

    def replay(dec, wavs):
        if len(wavs) != 1 or not np.array_equal(wavs[0], captured["wav"]):
            raise AssertionError("candidate PCM changed")
        return captured["result"]

    report = {"scope": p["scope"], "splits": {}, "deployment": False}
    for split in p["splits"]:
        records = service.read_jsonl(source / (split + ".responses.jsonl"))
        counts, rows, changes = collections.Counter(), [], []
        with (out / (split + ".responses.jsonl")).open("w") as stream:
            for record in records:
                row = record["row"]
                target = bool(TIME_PHRASE.search(row["prompt"]))
                guard = not target and record["raw"] is not None and counts["guard_responses_reproduced"] < p["guard_count"]
                candidate = record
                if target:
                    counts["target_total"] += 1
                if record["raw"] is not None and (target or guard):
                    app.S["cb"], app.S["fwd"] = original_builder, capture
                    x = service.load_audio(row)[:320000]
                    _, _, raw_base = app.analyze(x, row["prompt"], "")
                    if service.jsonable(service.response(app, row, raw_base)) != record["body"]:
                        raise AssertionError("baseline body not reproduced")
                    app.S["cb"], app.S["fwd"] = candidate_builder, replay
                    tick = time.monotonic()
                    result, _, raw = app.analyze(x, row["prompt"], "")
                    elapsed_ms = 1000 * (time.monotonic() - tick)
                    candidate = dict(record, raw=raw, body=service.jsonable(service.response(app, row, raw)))
                    if guard:
                        if candidate["body"] != record["body"]:
                            raise AssertionError("guard response changed")
                        counts["guard_responses_reproduced"] += 1
                    else:
                        counts["target_responses_reproduced"] += 1
                        changed = [i for i, (a, b) in enumerate(zip(raw_base["canon"], raw["canon"])) if a != b]
                        if len(raw_base["canon"]) != len(raw["canon"]):
                            raise AssertionError("reference length changed")
                        if any(raw_base["canon"][i] != "ㅅ" or raw["canon"][i] != "ㅆ" for i in changed):
                            raise AssertionError("unexpected reference change")
                        counts["reference_changed_utterances"] += bool(changed)
                        old, new = service.phonemes(record["body"]), service.phonemes(candidate["body"])
                        other = [i for i, (a, b) in enumerate(zip(old, new)) if i not in changed and
                                 (a.get("errorType"), a.get("actual_jamo")) != (b.get("errorType"), b.get("actual_jamo"))]
                        counts["other_phone_flag_or_actual_changes"] += len(other)
                        changes.append({"uid": row["uid"], "prompt": row["prompt"], "reference_changes": changed,
                                        "other_phone_changes": other, "candidate_cached_forward_ms": elapsed_ms,
                                        "pred_jamo": result["pred_jamo"], "baseline_raw": raw_base, "candidate_raw": raw})
                base_sites, new_sites = observe(record, row), observe(candidate, row)
                for site in row["sites"]:
                    q = site["q"]
                    rows.append({"uid": row["uid"], "speaker": row["speaker"], "item": row["item"],
                                 "language": row.get("language"), "q": q, "expected": site["canon"], "actual": site["actual"],
                                 "following": row["canonical"][q + 1],
                                 "position": "initial" if q == 0 else "post_vowel" if row["canonical"][q - 1] in service.VOWELS else "post_consonant",
                                 "target_phrase": target, "baseline": base_sites[q], "candidate": new_sites[q]})
                service.write_line(stream, {"uid": row["uid"], "rejected": candidate["rejected"], "body": candidate["body"]})
        detail = {"counts": dict(counts), "overall": metrics(rows),
                  "target_phrase": metrics([r for r in rows if r["target_phrase"]])}
        for field in ("following", "position", "item", "language"):
            detail[field] = {str(v): metrics([r for r in rows if r[field] == v])
                             for v in sorted({r[field] for r in rows}, key=str)}
        for suffix, values in (("sites", rows), ("reference_changes", changes)):
            with (out / (split + "." + suffix + ".jsonl")).open("w") as stream:
                for value in values:
                    service.write_line(stream, value)
        report["splits"][split] = detail
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(split, json.dumps({"counts": detail["counts"], "overall": detail["overall"]}, ensure_ascii=False), flush=True)
    (out / "finished.json").write_text(json.dumps({"report_sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()
