"""Summarize complete GPU or NPU comparisons for matched encoder/head pairs."""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r:
        raise RuntimeError("Repository root not found")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import collections
import csv
import json

import numpy as np

from rescore_npu_heads import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-dir", type=_pl.Path, required=True)
    parser.add_argument("--records", type=_pl.Path, required=True)
    parser.add_argument("--out", type=_pl.Path, required=True)
    parser.add_argument("--scope", default="repeated validation")
    parser.add_argument("--backend", choices=("npu", "gpu"), default="npu")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--reference", action="append", default=["baseline", "replay", "lora_youtube"])
    args = parser.parse_args()
    expected = {row["job_id"]: row for row in map(json.loads, args.records.read_text().splitlines())}
    rows = {}
    for path in sorted(args.npu_dir.glob("shard_*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["job_id"] in rows or row["job_id"] not in expected:
                raise ValueError("Duplicate or unexpected NPU request")
            reference = expected[row["job_id"]]
            if any(row[key] != reference[key] for key in ("country", "speaker")):
                raise ValueError("NPU metadata differs")
            if "n" in reference["payload"] and row["n"] != reference["payload"]["n"]:
                raise ValueError("NPU split differs")
            rows[row["job_id"]] = row
    if set(rows) != set(expected):
        raise ValueError("Incomplete evaluation: {} of {}".format(len(rows), len(expected)))
    rows = [rows[key] for key in sorted(rows)]
    tags = list(rows[0]["variants"])
    if any(set(row["variants"]) != set(tags) for row in rows):
        raise ValueError("Incomplete encoder variants")
    stored_n, stored_match, stored_top1_match = 0, 0, 0
    for row in rows:
        reference = expected[row["job_id"]]["payload"].get("top5", row.get("recorded_baseline_top5"))
        if reference is not None:
            stored_n += 1
            found = row["variants"]["baseline"]["top5"]
            stored_match += reference == found
            stored_top1_match += (reference[0]["label"] if reference else None) == (found[0]["label"] if found else None)
    if args.backend == "npu" and stored_top1_match != stored_n:
        raise ValueError("Production NPU top1 reference changed")
    result, hits = {}, {}
    for tag in tags:
        result[tag], hits[tag] = metrics(rows, tag)
    foreign = np.array([row["country"] != "Korea" for row in rows])
    speakers = sorted({row["speaker"] for row in rows if row["country"] != "Korea"})
    indices = [[index for index, row in enumerate(rows) if row["speaker"] == speaker and foreign[index]]
               for speaker in speakers]
    sizes = np.array([len(ix) for ix in indices])
    draws = np.random.default_rng(42).integers(0, len(speakers), (args.bootstrap_reps, len(speakers)))
    comparisons = {}
    for tag in tags:
        for reference in args.reference:
            if tag == reference or reference not in tags:
                continue
            delta = hits[tag].astype(int) - hits[reference].astype(int)
            by_speaker = np.array([delta[ix].sum() for ix in indices])
            boot = by_speaker[draws].sum(1) / sizes[draws].sum(1) * 100
            comparisons[tag + "_vs_" + reference] = {
                "foreign_top1_delta_pp": float(delta[foreign].mean() * 100),
                "country_mean_delta_pp": (result[tag]["foreign_macro_top1"] - result[reference]["foreign_macro_top1"]) * 100,
                "speaker_bootstrap_95ci_pp": np.percentile(boot, [2.5, 97.5]).tolist(),
                "improved": [row["job_id"] for index, row in enumerate(rows) if foreign[index] and delta[index] == 1],
                "regressed": [row["job_id"] for index, row in enumerate(rows) if foreign[index] and delta[index] == -1]}
    summary = {"scope": args.scope, "backend": args.backend, "groups": len(rows), "speakers": len({row["speaker"] for row in rows}),
               "actual_parts": dict(collections.Counter(str(row["n"]) for row in rows)),
               "recorded_baseline_payload_parity": {"matching_complete_top5": stored_match,
                       "matching_top1": stored_top1_match, "recorded": stored_n},
               "metrics": result, "comparisons": comparisons,
               "note": "Small YouTube samples and opened datasets do not establish population accuracy."}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (args.out / "per_country.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "country", "requests", "speakers", "correct_top1", "top1", "top5", "errors"])
        writer.writeheader()
        for tag, score in result.items():
            for country, item in score["per_country"].items():
                writer.writerow({"variant": tag, "country": country, "requests": item["n"], "speakers": item["speakers"],
                                 "correct_top1": item["correct_top1"], "top1": item["top1"], "top5": item["top5"],
                                 "errors": json.dumps(item["errors"], ensure_ascii=False)})
    for tag, score in result.items():
        print(tag, {key: value for key, value in score.items() if key != "per_country"})


if __name__ == "__main__":
    main()
