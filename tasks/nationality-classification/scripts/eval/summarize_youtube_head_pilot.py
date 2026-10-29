"""Summarize the NPU head comparison and preserve per-country regressions."""
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
from shared.paths import NAT_TASK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-dir", type=_pl.Path, required=True)
    parser.add_argument("--expected-records", type=_pl.Path, required=True)
    parser.add_argument("--out", type=_pl.Path, required=True)
    parser.add_argument("--contrast", action="append", default=[],
                        help="CANDIDATE=CONTROL pair; defaults to matching youtube/replay_control variants")
    parser.add_argument("--limits-json", type=_pl.Path,
                        help="JSON list of experiment-specific limitations")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    expected = {}
    for line in args.expected_records.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        expected[row["job_id"]] = row
    found = {}
    for path in sorted(args.npu_dir.glob("shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["job_id"] in found:
                raise ValueError("Duplicate job ID: " + row["job_id"])
            found[row["job_id"]] = row
    if set(expected) != set(found):
        raise ValueError("Incomplete NPU results: {} of {}".format(len(found), len(expected)))
    rows = [found[key] for key in sorted(found)]
    variants = sorted(rows[0]["variants"])
    if any(sorted(row["variants"]) != variants for row in rows):
        raise ValueError("Variant set differs between jobs")
    countries = np.array([r["country"] for r in rows])
    foreign = countries != "Korea"
    mismatches = []
    for row in rows:
        if row["capture_rid"] != expected[row["job_id"]]["payload"]["rid"]:
            raise ValueError("Captured input differs")
        original = expected[row["job_id"]]["payload"]["top5"]
        if row["variants"]["baseline"]["top5"] != original:
            mismatches.append({"job_id": row["job_id"], "original": original,
                               "replayed": row["variants"]["baseline"]["top5"]})
    top1_mismatches = [r for r in mismatches if r["original"][0]["label"] != r["replayed"][0]["label"]]
    if top1_mismatches:
        raise ValueError("NPU baseline top-1 differs from recorded service: " + str(top1_mismatches))
    metrics, hits, predictions = {}, {}, {}
    for variant in variants:
        top5 = [r["variants"][variant]["top5"] for r in rows]
        prediction = np.array([items[0]["label"] for items in top5])
        hit = prediction == countries
        hit5 = np.array([r["country"] in [p["label"] for p in items] for r, items in zip(rows, top5)])
        predictions[variant], hits[variant] = prediction, hit
        per_country = {}
        for country in sorted(set(countries)):
            mask = countries == country
            per_country[country] = {"n": int(mask.sum()),
                                    "speakers": len({r["speaker"] for r in rows if r["country"] == country}),
                                    "correct_top1": int(hit[mask].sum()), "correct_top5": int(hit5[mask].sum()),
                                    "top1": float(hit[mask].mean()), "top5": float(hit5[mask].mean()),
                                    "errors": dict(collections.Counter(prediction[mask & ~hit].tolist()))}
        metrics[variant] = {"foreign_n": int(foreign.sum()),
                            "foreign_top1": float(hit[foreign].mean()), "foreign_top5": float(hit5[foreign].mean()),
                            "foreign_macro_top1": float(np.mean([r["top1"] for c, r in per_country.items() if c != "Korea"])),
                            "foreign_macro_top5": float(np.mean([r["top5"] for c, r in per_country.items() if c != "Korea"])),
                            "korean_top1": float(hit[~foreign].mean()), "per_country": per_country}
    contrasts = {}
    pairs = [item.split("=", 1) for item in args.contrast]
    if not pairs:
        pairs = [(v, "replay_control_" + v[len("youtube_"):]) for v in variants
                 if v.startswith("youtube_") and "replay_control_" + v[len("youtube_"):] in variants]
    for youtube, control in pairs:
        a, b = hits[youtube], hits[control]
        contrasts[youtube + "_vs_control"] = {
            "candidate": youtube, "control": control,
            "effect_scope": ("combined_changes_vs_production" if control == "baseline"
                             else "youtube_addition_vs_matched_retraining"),
            "foreign_top1_delta_pp": (metrics[youtube]["foreign_top1"] - metrics[control]["foreign_top1"]) * 100,
            "foreign_macro_top1_delta_pp": (metrics[youtube]["foreign_macro_top1"] - metrics[control]["foreign_macro_top1"]) * 100,
            "improved_groups": [r["job_id"] for i, r in enumerate(rows) if foreign[i] and a[i] and not b[i]],
            "regressed_groups": [r["job_id"] for i, r in enumerate(rows) if foreign[i] and b[i] and not a[i]]}
        speakers = sorted({r["speaker"] for i, r in enumerate(rows) if foreign[i]})
        sizes, differences = [], []
        for speaker in speakers:
            mask = foreign & np.array([r["speaker"] == speaker for r in rows])
            sizes.append(int(mask.sum()))
            differences.append(int(a[mask].sum()) - int(b[mask].sum()))
        rng = np.random.default_rng(args.seed)
        draws = rng.integers(0, len(speakers), size=(args.bootstrap_reps, len(speakers)))
        changes = np.asarray(differences)[draws].sum(1) / np.asarray(sizes)[draws].sum(1) * 100
        contrasts[youtube + "_vs_control"]["speaker_bootstrap_95ci_pp"] = np.percentile(changes, [2.5, 97.5]).tolist()
        contrasts[youtube + "_vs_control"]["bootstrap_reps"] = args.bootstrap_reps
    summary = {"scope": "validation; checkpoints selected using GPU reference validation",
               "npu_groups": len(rows), "validation_speakers": len({r["speaker"] for r in rows}),
               "parts_requested": 3, "actual_parts": dict(collections.Counter(str(r["n"]) for r in rows)),
               "baseline_top1_parity": {"matching": len(rows), "total": len(rows)},
               "baseline_payload_rounding_differences": mismatches,
               "metrics": metrics, "candidate_comparisons": contrasts,
               "youtube_incremental_effect": {k: v for k, v in contrasts.items() if v["control"] != "baseline"},
               "limits": (json.loads(args.limits_json.read_text()) if args.limits_json else
                          ["3 YouTube training speakers; 68 clips; one podcast channel",
                          "Frozen encoder and fixed Korean gate; head-only feasibility experiment",
                          "This is validation evidence, not independent test or application-user evidence"])}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    mapping = json.loads((NAT_TASK / "configs/country41_ko2en.json").read_text(encoding="utf-8"))
    flat = []
    for ko, country in sorted(mapping.items(), key=lambda x: x[1]):
        baseline = metrics["baseline"]["per_country"].get(country, {})
        row = {"country": country, "country_ko": ko, "n": baseline.get("n", 0),
               "speakers": baseline.get("speakers", 0)}
        for variant in variants:
            item = metrics[variant]["per_country"].get(country, {})
            row[variant + "_top1"] = item.get("top1", "")
            row[variant + "_top5"] = item.get("top5", "")
            row[variant + "_errors"] = json.dumps(item.get("errors", {}), ensure_ascii=False)
        flat.append(row)
    with (args.out / "per_country.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    for variant, item in metrics.items():
        print(variant, {key: value for key, value in item.items() if key != "per_country"})
    print("Candidate comparisons", contrasts)


if __name__ == "__main__":
    main()
