"""Attach each expert's event observations to frozen acoustic candidate traces.

No reviewer averaging, uncertain-range midpoint, or TAG fallback is allowed.
This is an offline handoff, not an adjudication or a pronunciation classifier.
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
from collections import Counter
import copy
import datetime
import hashlib
import importlib.util
import json
import math
import wave

from mdd.sibilant_evidence import _periodicity_views
from mdd.sibilant_observations import interval_observations


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def helper(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unique(rows):
    result = {r["id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate target IDs")
    return result


def expert_views(item, observation, annotation, phones):
    """Preserve all reviewer states; calculate only attested, associated points.

    The CLI validates the actual TextGrid, notes, exported audio and fingerprints
    before this join. Self-declared expertise does not establish independence.
    """
    if not item["id"] == observation["id"] == annotation["id"]:
        raise ValueError("target identity mismatch")
    if not item["source_sha256"] == observation["source_sha256"] == annotation["source_sha256"]:
        raise ValueError("source fingerprint mismatch")
    if observation.get("schema_version") != 3 or observation["TAG_sha256"] != item["tag_sha256"]:
        raise ValueError("observation schema or TAG identity mismatch")
    sr, offset = observation["sample_rate"], observation["offset_s"]
    expected = [round(v * sr) for v in item["analysis_interval"]]
    if observation["analysis_interval_samples"] != expected or abs(offset - expected[0] / sr) > 1e-9:
        raise ValueError("analysis sample range or time offset mismatch")
    duration = (expected[1] - expected[0]) / sr
    if abs(observation["periodicity"]["clip_duration_s"] - duration) > 1e-9:
        raise ValueError("trace duration mismatch")
    if observation["pronunciation_decision"] is not None or observation["new_human_gold"] is not False:
        raise ValueError("unexpected decision or gold claim in candidate trace")
    result = {"id": item["id"], "source_sha256": item["source_sha256"],
              "TAG_sha256": item["tag_sha256"], "schema_version": 1,
              "vowel_context": copy.deepcopy(observation["vowel_context"]),
              "reviewers": {}, "reviewer_measurements_averaged": False,
              "independence_verified": False, "new_adjudicated_gold": False,
              "pronunciation_decision": None}
    for who, review in annotation["reviewers"].items():
        states = review.get("landmark_status", {})
        points = review.get("points_original_seconds", {})
        for landmark, state in states.items():
            point = points.get(landmark)
            if state not in ("observed", "absent", "uncertain", "unreviewed"):
                raise ValueError("unknown landmark state")
            if (state == "observed") != (point is not None):
                raise ValueError("point/state mismatch")
            if point is not None and (isinstance(point, bool) or not isinstance(point, (int, float)) or not math.isfinite(point)):
                raise ValueError("nonfinite landmark")
        specs = {"first_voicing_40ms": ("first_vowel_voicing",),
                 "regular_run": ("first_regular_run_start", "first_regular_run_end")}
        intervals, derivations = {}, {}
        for name, required in specs.items():
            provenance = "separate_reviewer:" + who + ":" + name + ";not_adjudicated_gold"
            view = {"interval_provenance": provenance}
            derivations[name] = {"required_landmarks": list(required),
                                 "landmark_states": {k: states.get(k, "unreviewed") for k in required},
                                 "uncertainty_midpoint_used": False, "TAG_fallback_used": False}
            if review.get("actor_type") != "human_self_declared_expert":
                view["status"] = "reviewer_input_not_attested"
            elif review.get("target_association") != "confirmed":
                view["status"] = "target_association_not_confirmed"
            elif any(states.get(k) != "observed" for k in required):
                view["status"] = "required_landmark_not_observed"
            else:
                span = [points[required[0]], points[required[-1]]]
                if name == "first_voicing_40ms":
                    span[1] = span[0] + .040
                if span[0] >= span[1]:
                    raise ValueError("reversed or empty observed interval")
                relative = [t - offset for t in span]
                if relative[0] < 0 or relative[1] > duration:
                    view["status"] = "interval_outside_analysis_clip"
                else:
                    view = interval_observations(observation["periodicity"], *relative, provenance)
                    view["status"] = "observed_candidates_only"
                view["original_interval_s"] = span
            intervals[name] = view
        local = dict(observation, intervals=intervals)
        result["reviewers"][who] = {"annotation": copy.deepcopy(review),
            "window_derivation": derivations, "periodicity_views": _periodicity_views(local, item, phones),
            "interpretation": "An observed event does not validate automatic F0, vowel identity, or the entire following 40ms."}
    return result


def export(packet, expert_packet, observations, reviewers, out):
    if out.exists() or len(set(reviewers)) != len(reviewers):
        raise ValueError("output must be new and reviewers unique")
    boundary_helper = helper("summarize_sibilant_boundaries")
    tag_helper = helper("audit_sibilant_periodicity")
    manifest = packet / "items.private.jsonl"
    items = unique([json.loads(s) for s in manifest.read_text().splitlines() if s.strip()])
    # Read current expert files, rather than trusting a possibly stale audit.
    audit = boundary_helper.summarize(expert_packet, reviewers)
    annotations = unique(audit["records"])
    if audit["source_packet_sha256"] != digest(manifest):
        raise ValueError("expert packet belongs to a different source packet")
    if set(items) != set(annotations) or set(items) != {p.stem for p in (observations / "tracks").glob("*.json")}:
        raise ValueError("target sets differ; no silent exclusion")
    source_summary = observations / "summary.json"
    for path, expected in json.loads(source_summary.read_text())["provenance"].items():
        if digest(path) != expected:
            raise ValueError("frozen trace source changed: " + path)
    out.mkdir(parents=True)
    (out / "records").mkdir()
    (out / "source_snapshot").mkdir()
    write = lambda p, r: p.write_text(json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write(out / "expert_input_audit.json", audit)
    source_hashes, record_hashes, trace_hashes = {}, {}, {}
    counts = {who: Counter() for who in reviewers}
    for key, item in items.items():
        for field, fingerprint in (("source_wav", "source_sha256"), ("tag_file", "tag_sha256")):
            path = item[field]
            if path not in source_hashes:
                source_hashes[path] = digest(path)
            if source_hashes[path] != item[fingerprint]:
                raise ValueError("original source changed: " + key)
        path = observations / "tracks" / (key + ".json")
        trace_hashes[key] = digest(path)
        observation = json.loads(path.read_text())
        with wave.open(item["source_wav"], "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (observation["sample_rate"], 1, 2, "NONE"):
                raise ValueError("source audio format mismatch")
            begin, end = observation["analysis_interval_samples"]
            wav.setpos(begin)
            if hashlib.sha256(wav.readframes(end - begin)).hexdigest() != observation["pcm_bytes_sha256"]:
                raise ValueError("cached trace PCM differs from original")
        record = expert_views(item, observation, annotations[key], tag_helper.read_tag_phones(item))
        for who, entry in record["reviewers"].items():
            for name, view in entry["periodicity_views"].items():
                counts[who][name + ":" + view["observation_status"]] += 1
        target = out / "records" / (key + ".json")
        write(target, record)
        record_hashes[key] = digest(target)
    sources = [Path(__file__), Path(boundary_helper.__file__), Path(tag_helper.__file__),
               ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_evidence.py",
               ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_observations.py"]
    for path in sources:
        (out / "source_snapshot" / path.name).write_bytes(path.read_bytes())
    summary = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "items": len(items), "all_source_ids_retained": True,
        "review_counts": audit["counts_by_reviewer"],
        "window_status_counts": {who: dict(counts[who]) for who in reviewers},
        "reviewer_measurements_averaged": False, "independence_verified": False,
        "new_adjudicated_gold": False, "pronunciation_decisions_produced": 0,
        "source_packet_sha256": digest(manifest), "trace_summary_sha256": digest(source_summary),
        "expert_audit_sha256": digest(out / "expert_input_audit.json"),
        "original_source_sha256": source_hashes, "trace_sha256": trace_hashes,
        "record_sha256": record_hashes, "code_provenance": {str(p): digest(p) for p in sources}}
    write(out / "summary.json", summary)
    return {k: summary[k] for k in ("items", "review_counts", "window_status_counts")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "expert-packet", "observations", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--reviewers", nargs="+", default=["expert_A", "expert_B"])
    args = ap.parse_args()
    print(json.dumps(export(args.packet, args.expert_packet, args.observations, args.reviewers, args.out)))


if __name__ == "__main__":
    main()
