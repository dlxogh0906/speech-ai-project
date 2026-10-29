"""Assemble the complete protected CAL inventory from verified runtime caches.

Runtime eligibility is based on the reference API, independent of TAG actuals.
The plan retains every utterance and requests fresh extraction for every missing
eligible clip. Assemble fails unless these gaps have been resolved or explicitly
reported unavailable by the unchanged runtime extractor.
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
import numpy as np
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from mdd.sibilant_transition import PAIR, VOWELS


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def lines(path, values):
    with path.open("x") as stream:
        for value in values:
            service.write_line(stream, value)


def same_input(first, other):
    data, index, feature, _ = first
    more, j, f, _ = other
    for key in ("canonical", "baseline_pred", "clip_sha256", "vad_pcm_sha256", "left", "right"):
        if feature[key] != f[key]:
            raise AssertionError("duplicate feature identity differs: " + key)
    length = int(data["lengths"][index])
    if length != int(more["lengths"][j]):
        raise AssertionError("duplicate observed length differs")
    for name in ("x", "relative"):
        np.testing.assert_array_equal(data[name][index], more[name][j])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("plan", "assemble"), required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, expected in p["sha256"].items():
        if service.sha(path) != expected:
            raise ValueError("source changed: " + path)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    inventory = service.read_jsonl(p["inventory"])
    wanted = {r["uid"] for r in inventory}
    if len(wanted) != len(inventory):
        raise ValueError("duplicate inventory UID")
    records, bank, counts = collections.defaultdict(list), collections.defaultdict(list), collections.Counter()
    unavailable = set()
    for name, prefix in p["banks"].items():
        data = dict(np.load(prefix + ".npz"))
        if set(data) != {"x", "relative", "lengths"}:
            raise ValueError("runtime inputs contain nonacoustic fields")
        features = service.read_jsonl(prefix + ".features.jsonl")
        if len(features) != len(data["x"]):
            raise ValueError("feature bank count differs")
        for i, f in enumerate(features):
            if f["uid"] in wanted:
                bank[(f["uid"], f["q_runtime"])].append((data, i, f, name))
        for record in service.read_jsonl(prefix + ".responses.jsonl"):
            if record["row"]["uid"] in wanted:
                records[record["row"]["uid"]].append(record)
                if name in p.get("fresh_banks", []):
                    unavailable.add(record["row"]["uid"])
    missing, output_records, labels, features, arrays = [], [], [], [], []
    for row in inventory:
        uid = row["uid"]
        candidates = records[uid]
        if not candidates:
            raise ValueError("uncached full utterance: " + uid)
        chosen = candidates[0]
        for record in candidates:
            for field in ("wav", "frames", "canonical", "actual", "prompt"):
                if record["row"][field] != row[field]:
                    raise AssertionError("inventory identity differs: " + uid + ":" + field)
            for key in ("original", "existing", "prior"):
                if record[key] != chosen[key]:
                    raise AssertionError("duplicate full API/raw response differs: " + uid + ":" + key)
        reference, by_q = chosen["existing"], {}
        raw = reference["raw"]
        missing_q = []
        if raw is None:
            counts["input_rejected"] += 1
        else:
            phones = service.phonemes(reference["body"])
            for q, token in enumerate(raw["canon"][:-1]):
                if (token not in PAIR or raw["canon"][q + 1] not in VOWELS
                        or raw["sites"][q].get("pred") not in PAIR or phones[q].get("errorType")):
                    continue
                counts["runtime_eligible_sites"] += 1
                entries = bank[(uid, q)]
                if not entries:
                    if uid in unavailable:
                        counts["fresh_extractor_unavailable_sites"] += 1
                    else:
                        missing_q.append(q)
                    continue
                data, index, feature, name = entries[0]
                if feature["canonical"] != token or feature["baseline_pred"] != raw["sites"][q].get("pred"):
                    raise AssertionError("reference eligibility identity differs")
                for other in entries[1:]:
                    same_input(entries[0], other)
                    counts["duplicate_inputs_exact"] += 1
                by_q[q] = len(features)
                length = int(data["lengths"][index])
                arrays.append((data["x"][index], data["relative"][index], length))
                features.append(dict(feature, cache_sources=[{"bank": entry[3], "index": entry[1]}
                                                            for entry in entries]))
                counts["features_from_" + name] += 1
        if missing_q:
            counts["missing_runtime_sites"] += len(missing_q)
            # The unchanged extractor gets the reference response, with pair
            # thresholds disabled. Its old-margin comparator is irrelevant here.
            missing.append({"row": row, "original": chosen["original"], "existing": reference,
                            "missing_runtime_q": missing_q})
        output_records.append(dict(chosen, row=row))
        viewed = observe(reference, row)
        mapping = service.all_optimal_mapping(row["canonical"], raw["canon"]) if raw else None
        for site in row["sites"]:
            options = mapping[site["q"]] if mapping is not None else set()
            rq = next(iter(options)) if len(options) == 1 and None not in options else None
            labels.append(dict(site, expected=site["canon"], uid=uid, speaker=row["speaker"], item=row["item"],
                               language=row.get("language"), q_runtime=rq, feature_index=by_q.get(rq),
                               existing=viewed[site["q"]]))
    lines(out / "missing_sources.jsonl", missing)
    report = {"scope": p["scope"], "stage": a.stage, "utterances": len(inventory), "all_sites": len(labels),
              "features": len(features), "counts": dict(counts), "missing_utterances": len(missing),
              "actual_or_TAG_time_model_inputs": False, "runtime_eligibility": "reference API, not prior margin"}
    if a.stage == "assemble":
        if missing:
            write(out / "incomplete.json", report)
            raise ValueError("unresolved runtime inputs; run unchanged extraction for missing_sources")
        data = {"x": np.asarray([v[0] for v in arrays], dtype=np.float32),
                "relative": np.asarray([v[1] for v in arrays], dtype=np.float32),
                "lengths": np.asarray([v[2] for v in arrays], dtype=np.int64)}
        np.savez(out / "cal.npz", **data)
        for suffix, values in (("responses", output_records), ("features", features), ("labels", labels)):
            lines(out / ("cal." + suffix + ".jsonl"), values)
        report["output_sha256"] = {suffix: service.sha(out / ("cal" + suffix))
                                   for suffix in (".npz", ".features.jsonl", ".labels.jsonl", ".responses.jsonl")}
    write(out / "report.json", report)
    write(out / "finished.json", {"sha256": service.sha(out / "report.json")})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
