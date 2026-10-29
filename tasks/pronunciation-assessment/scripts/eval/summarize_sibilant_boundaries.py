"""Validate supplied expert TextGrids without filling unobserved boundaries.

Only source times are converted. Empty templates are not annotations; points
without reviewer notes are unattested observations, never human ground truth.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import wave

LANDMARKS = ("frication_start", "frication_to_diffuse_noise", "first_vowel_voicing",
             "first_regular_run_start", "first_regular_run_end")
STATUSES = {"observed", "absent", "uncertain", "unreviewed"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_time(value, duration):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= duration:
        raise ValueError("invalid annotation time")
    return float(value)


def read_points(path, duration):
    import parselmouth
    from parselmouth.praat import call
    grid = parselmouth.read(str(path))
    if grid.class_name != "TextGrid":
        raise ValueError("expected TextGrid")
    if abs(call(grid, "Get start time")) > 1e-7 or abs(call(grid, "Get end time") - duration) > 1e-7:
        raise ValueError("TextGrid time domain differs from exported audio")
    if call(grid, "Get number of tiers") != len(LANDMARKS):
        raise ValueError("unexpected annotation tiers")
    points = {}
    for tier in range(1, len(LANDMARKS) + 1):
        name = call(grid, "Get tier name", tier)
        if name not in LANDMARKS or name in points or call(grid, "Is interval tier", tier):
            raise ValueError("expected unique named point tiers")
        n = call(grid, "Get number of points", tier)
        if n > 1:
            raise ValueError("multiple candidate points: use uncertain range in notes")
        points[name] = finite_time(call(grid, "Get time of point", tier, 1), duration) if n else None
    for first, second in (("frication_start", "frication_to_diffuse_noise"),
                          ("first_vowel_voicing", "first_regular_run_start"),
                          ("first_regular_run_start", "first_regular_run_end")):
        if points[first] is not None and points[second] is not None and points[first] > points[second]:
            raise ValueError("inconsistent landmark order")
    begin, end = points["first_regular_run_start"], points["first_regular_run_end"]
    if begin is not None and end is not None and begin == end:
        raise ValueError("regular-cycle window has zero duration")
    return points


def observation(points, note, who, duration, offset):
    statuses = {k: "observed" if points[k] is not None else "unreviewed" for k in LANDMARKS}
    declared = note.get("landmark_status", {}) if note else {}
    if set(declared) - set(LANDMARKS):
        raise ValueError("unknown landmark status")
    statuses.update(declared)
    for k, status in statuses.items():
        if status not in STATUSES or (status == "observed") != (points[k] is not None):
            raise ValueError("point/status mismatch: " + k)
    ranges = note.get("uncertain_ranges_relative_seconds", {}) if note else {}
    if set(ranges) - set(LANDMARKS):
        raise ValueError("unknown uncertain landmark")
    for k, bounds in ranges.items():
        if statuses[k] != "uncertain" or not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("range requires an uncertain landmark with no point")
        left, right = [finite_time(t, duration) for t in bounds]
        if left > right:
            raise ValueError("reversed uncertain range")
    attested = bool(note and note.get("actor_type") == "human_expert" and note.get("annotator") == who)
    if note and not attested:
        raise ValueError("reviewer provenance mismatch")
    association = note.get("target_association", "unreviewed") if note else "unreviewed"
    if association not in ("confirmed", "uncertain", "outside_clip", "unreviewed"):
        raise ValueError("invalid target association")
    transition = note.get("transition_kind", "unreviewed") if note else "unreviewed"
    if transition not in ("aspiration", "silence", "mixed", "overlap", "none", "uncertain", "unreviewed"):
        raise ValueError("invalid transition kind")
    reviewed = any(s != "unreviewed" for s in statuses.values())
    complete = attested and all(s != "unreviewed" for s in statuses.values()) and association != "unreviewed" and transition != "unreviewed"
    return {"status": "complete_review" if complete else "partial_review" if reviewed and attested else "unattested_points" if reviewed else "empty_template",
            "actor_type": "human_self_declared_expert" if attested else "unattested",
            "target_association": association, "transition_kind": transition,
            "landmark_status": statuses, "points_relative_seconds": points,
            "points_original_seconds": {k: v + offset if v is not None else None for k, v in points.items()},
            "uncertain_ranges_original_seconds": {k: [v + offset for v in b] for k, b in ranges.items()},
            "note": note.get("note", "") if note else ""}


def summarize(packet, reviewers):
    mapping_path, protocol_path = packet / "source_offsets.private.json", packet / "annotation_protocol.json"
    mapping = json.loads(mapping_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    ids = {r["id"] for r in mapping}
    if len(ids) != len(mapping) or len(ids) != protocol["items"]:
        raise ValueError("annotation inventory differs")
    provenance = {str(p): digest(p) for p in (mapping_path, protocol_path)}
    notes = {}
    for who in reviewers:
        if Path(who).name != who or who in (".", ".."):
            raise ValueError("invalid reviewer name")
        path = packet / who / "notes.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
        if path.exists():
            provenance[str(path)] = digest(path)
        notes[who] = {r["id"]: r for r in rows}
        if len(notes[who]) != len(rows) or set(notes[who]) - ids:
            raise ValueError("duplicate or unknown item in reviewer notes")
    records = []
    for item in mapping:
        path = packet / "audio" / (item["id"] + ".wav")
        if digest(path) != item.get("audio_sha256"):
            raise ValueError("exported audio changed or lacks fingerprint")
        with wave.open(str(path), "rb") as f:
            duration = f.getnframes() / f.getframerate()
            if (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()) != (item["sample_rate"], 1, 2, item["end_sample"] - item["start_sample"]):
                raise ValueError("audio/offset metadata mismatch")
        offset = item["start_sample"] / item["sample_rate"]
        if abs(offset - item["source_offset_seconds"]) > 1e-9:
            raise ValueError("source offset differs from sample index")
        observations = {}
        for who in reviewers:
            grid = packet / who / (item["id"] + ".TextGrid")
            if not grid.exists():
                observations[who] = {"status": "missing_file"}
                continue
            provenance[str(grid)] = digest(grid)
            points = read_points(grid, duration)
            observations[who] = observation(points, notes[who].get(item["id"]), who, duration, offset)
        records.append({"id": item["id"], "source_sha256": item["source_sha256"], "reviewers": observations})
    paired = {}
    for a, b in itertools.combinations(reviewers, 2):
        differences = {k: [] for k in LANDMARKS}
        for row in records:
            left, right = row["reviewers"][a], row["reviewers"][b]
            if any(r.get("actor_type") != "human_self_declared_expert" or r.get("target_association") != "confirmed" for r in (left, right)):
                continue
            for k in LANDMARKS:
                x, y = left["points_original_seconds"][k], right["points_original_seconds"][k]
                if x is not None and y is not None:
                    differences[k].append({"id": row["id"], "signed_difference_ms": 1000 * (x - y)})
        paired[a + "_" + b] = differences
    return {"status": "expert_input_audit_not_adjudicated_ground_truth", "items": len(records),
            "source_packet_sha256": protocol["source_packet_sha256"],
            "counts_by_reviewer": {w: dict(Counter(r["reviewers"][w]["status"] for r in records)) for w in reviewers},
            "paired_observed_differences": paired, "records": records, "provenance": provenance,
            "limitations": ["Reviewers and independence are self-declared, not verified by software.",
                            "Complete review can contain absent or uncertain landmarks; it does not mean all features are measurable.",
                            "Uncertain ranges never supply substitute points. Frication and voicing may overlap.",
                            "Missing files, empty templates, unattested observations and actual review are counted separately."]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--reviewers", nargs="+", default=["expert_A", "expert_B"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists() or len(set(a.reviewers)) != len(a.reviewers):
        ap.error("output must be new and reviewers unique")
    result = summarize(a.packet, a.reviewers)
    result["provenance"][str(Path(__file__))] = digest(Path(__file__))
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"items": result["items"], "counts_by_reviewer": result["counts_by_reviewer"]}))


if __name__ == "__main__":
    main()
