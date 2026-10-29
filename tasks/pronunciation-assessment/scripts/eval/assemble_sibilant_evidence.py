"""Join frozen acoustic observations into review records without producing scores."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

import argparse
from collections import Counter
import datetime
import hashlib
import importlib.util
import json

from mdd.sibilant_evidence import assemble_evidence


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_rows(rows):
    result = {r["id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate target IDs")
    return result


def review_groups(records):
    groups = {}
    for record in records:
        for flag in record["context_review_flags"]:
            groups.setdefault(flag, []).append(record["id"])
        for view, evidence in record["periodicity_views"].items():
            for flag in evidence["review_flags"]:
                groups.setdefault(view + ":" + flag, []).append(record["id"])
        if record["target_association"] not in ("not_independently_verified",):
            groups.setdefault("declared_target_association:" + record["target_association"], []).append(record["id"])
    return {k: sorted(set(v)) for k, v in sorted(groups.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--intensity", type=Path, required=True, help="current speech_v3 directory with actual support unions")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("refusing to overwrite a review bundle")
    manifest = args.packet / "items.private.jsonl"
    packet = unique_rows([json.loads(s) for s in manifest.read_text().splitlines()])
    observations_summary = json.loads((args.observations / "summary.json").read_text())
    intensity_path = args.intensity / "learner.json"
    intensity = unique_rows(json.loads(intensity_path.read_text()))
    intensity_provenance = json.loads((args.intensity / "provenance.json").read_text())
    # Validate the frozen inputs and code used to produce both source artifacts.
    for source in (observations_summary, intensity_provenance):
        for path, expected in source["provenance"].items():
            if digest(path) != expected:
                raise ValueError("frozen source provenance changed: " + path)
    observation_ids = {p.stem for p in (args.observations / "tracks").glob("*.json")}
    if set(packet) != set(intensity) or set(packet) != observation_ids:
        raise ValueError("source target sets differ; no silent inner join or exclusion")
    helper_path = Path(__file__).with_name("audit_sibilant_periodicity.py")
    spec = importlib.util.spec_from_file_location("frozen_tag_reader", str(helper_path))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    args.out.mkdir(parents=True)
    (args.out / "records").mkdir()
    started = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "scope": "offline_review_not_blinded_listener_material_or_scoring",
               "inputs": {str(p): digest(p) for p in (manifest, args.observations / "summary.json",
                           intensity_path, args.intensity / "provenance.json")}}
    (args.out / "run_started.json").write_text(json.dumps(started, indent=2) + "\n")
    file_hashes, records, output_hashes = {}, [], {}
    for key, item in packet.items():
        for field, checksum_field in (("source_wav", "source_sha256"), ("tag_file", "tag_sha256")):
            path = item[field]
            if path not in file_hashes:
                file_hashes[path] = digest(path)
            if file_hashes[path] != item[checksum_field]:
                raise ValueError("original audio/TAG changed: " + key)
        observation_path = args.observations / "tracks" / (key + ".json")
        observation = json.loads(observation_path.read_text())
        record = assemble_evidence(item, observation, intensity[key], helper.read_tag_phones(item))
        record["actual_file_integrity_verified_here"] = True
        record["source_artifacts"] = {"periodicity_observation": {"path": str(observation_path), "sha256": digest(observation_path)},
                                      "intensity_bundle": {"path": str(intensity_path), "sha256": started["inputs"][str(intensity_path)]}}
        target = args.out / "records" / (key + ".json")
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        output_hashes[key] = digest(target)
        records.append(record)
    groups = review_groups(records)
    review = {"scope": "review_requirements_not_error_predictions_or_independent_validation_selection",
              "all_ids": sorted(packet), "groups": groups,
              "new_human_labels": 0, "pronunciation_decisions_produced": 0}
    (args.out / "review_groups.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    summary = {"items": len(records), "all_source_ids_retained": True,
               "context_status_counts": dict(Counter(r["vowel_context"]["status"] for r in records)),
               "review_group_counts": {k: len(v) for k, v in groups.items()},
               "original_audio_and_TAG_hashes": file_hashes, "record_sha256": output_hashes,
               "input_provenance": started["inputs"], "new_human_labels": 0,
               "pronunciation_decisions_produced": 0, "service_callers_added": 0,
               "code_provenance": {str(p): digest(p) for p in (Path(__file__), helper_path,
                    ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_evidence.py")}}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.out / "source_snapshot").mkdir()
    for path in summary["code_provenance"]:
        (args.out / "source_snapshot" / Path(path).name).write_bytes(Path(path).read_bytes())
    print(json.dumps({k: summary[k] for k in ("items", "all_source_ids_retained", "context_status_counts", "review_group_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
