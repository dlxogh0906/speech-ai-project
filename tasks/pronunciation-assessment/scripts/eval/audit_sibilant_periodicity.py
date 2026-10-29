"""Export paired noise/periodicity observations for a frozen learner packet.

Original TAG spans and optional provisional cycle intervals are retained as
separate views. Neither supplies new human labels or validated event boundaries.
This command does not train, score pronunciation, or change any service.
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
import hashlib
import json
import time
import wave

import numpy as np
from mdd.sibilant_observations import noise_observations, periodicity_observations, interval_observations, interval_relation


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vowel_context_observation(item):
    """Keep expected and transcribed contexts distinct; neither verifies the sound."""
    expected, transcribed = item.get("following_vowel"), item.get("actual_vowel")
    if not expected or not transcribed:
        status = "context_labels_incomplete"
        labels_agree = None
    else:
        labels_agree = expected == transcribed
        status = "labels_agree_not_acoustically_verified" if labels_agree else "context_label_disagreement"
    return {"expected_following_vowel": expected, "existing_TAG_following_vowel": transcribed,
            "existing_TAG_interval_label": item.get("tag_vowel_label"),
            "expected_preceding_phone": item.get("preceding_canonical"),
            "existing_TAG_preceding_phone": item.get("previous_tag_phone"),
            "TAG_intervals_isolated": item.get("tag_intervals_isolated"),
            "status": status, "labels_agree": labels_agree,
            "acoustic_vowel_identity_verified": False,
            "pronunciation_error_inferred_from_label_disagreement": False,
            "provenance": "frozen_packet_expected_LAB_and_existing_TAG;not_new_independent_labels"}


def read_tag_phones(item):
    """Use the original fingerprinted phone tier, not inferred neighboring text."""
    import parselmouth
    from parselmouth.praat import call
    path = Path(item["tag_file"])
    if digest(path) != item["tag_sha256"]:
        raise ValueError("original TAG fingerprint differs")
    grid = parselmouth.read(str(path))
    tiers = [i for i in range(1, int(call(grid, "Get number of tiers")) + 1)
             if call(grid, "Get tier name", i) == "phones"]
    if len(tiers) != 1:
        raise ValueError("expected one original TAG phones tier")
    tier = tiers[0]
    phones = [[float(call(grid, "Get start time of interval", tier, j)),
               float(call(grid, "Get end time of interval", tier, j)),
               str(call(grid, "Get label of interval", tier, j))]
              for j in range(1, int(call(grid, "Get number of intervals", tier)) + 1)]
    targets = [r for r in phones if r[:2] == item["tag_vowel"] and r[2] == item["tag_vowel_label"]]
    if len(targets) != 1:
        raise ValueError("frozen vowel cannot be uniquely located in original TAG")
    return phones


def inspect_item(item, review=None):
    path = Path(item["source_wav"])
    if digest(path) != item["source_sha256"]:
        raise ValueError("source PCM file hash differs from frozen packet: " + item["id"])
    if review is not None and review["source_sha256"] != item["source_sha256"]:
        raise ValueError("landmark source hash differs")
    tag_phones = read_tag_phones(item)
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("expected original mono PCM16 WAV")
        sr = wav.getframerate()
        begin, end = [round(t * sr) for t in item["analysis_interval"]]
        if not 0 <= begin < end <= wav.getnframes():
            raise ValueError("analysis interval outside source")
        wav.setpos(begin)
        raw = wav.readframes(end - begin)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.
    original = hashlib.sha256(x.tobytes()).hexdigest()
    offset = begin / sr
    noise, periodicity = noise_observations(x, sr), periodicity_observations(x, sr)
    requested = [("existing_TAG_vowel", item["tag_vowel"], "existing_TAG_interval_not_acoustic_gold")]
    tag_start, tag_end = item["tag_vowel"]
    requested.append(("TAG_vowel_first40ms", [tag_start, min(tag_end, tag_start + .040)],
                      "existing_TAG_onset_fixed_40ms_clipped_at_TAG_vowel_end"))
    if review is not None:
        a, b = review.get("first_regular_run_start"), review.get("first_regular_run_end")
        requested.append(("reviewed_cycle_group", [a, b], review["actor_type"] + ":" + review["review_status"]))
        onset = review.get("voicing_first")
        requested.append(("reviewed_first_voicing_first40ms", [onset, onset + .040 if onset is not None else None],
                          review["actor_type"] + ":fixed_40ms_from_provisional_voicing;single_phone_containment_unverified"))
    intervals = {}
    for name, endpoints, provenance in requested:
        if any(v is None for v in endpoints):
            intervals[name] = {"status": "interval_not_annotated", "interval_provenance": provenance}
            continue
        reference_relation = interval_relation(*endpoints, *item["tag_vowel"],
                                               "existing_TAG_vowel_not_acoustic_gold")
        overlaps = [{"original_interval_s": [s, e], "label": label,
                     "overlap_s": min(e, endpoints[1]) - max(s, endpoints[0]),
                     "is_target_TAG_interval": [s, e] == item["tag_vowel"]}
                    for s, e, label in tag_phones if min(e, endpoints[1]) > max(s, endpoints[0])]
        a, b = [float(v) - offset for v in endpoints]
        if not 0 <= a < b <= len(x) / sr:
            intervals[name] = {"status": "interval_outside_analysis_clip", "original_interval_s": endpoints,
                               "interval_provenance": provenance, "TAG_vowel_relation": reference_relation,
                               "overlapping_TAG_phone_intervals": overlaps}
            continue
        intervals[name] = interval_observations(periodicity, a, b, provenance)
        intervals[name]["status"] = "observed_candidates_only"
        intervals[name]["original_interval_s"] = endpoints
        intervals[name]["TAG_vowel_relation"] = reference_relation
        intervals[name]["overlapping_TAG_phone_intervals"] = overlaps
    if hashlib.sha256(x.tobytes()).hexdigest() != original:
        raise RuntimeError("measurement mutated source PCM")
    return {"schema_version": 3, "id": item["id"], "source_sha256": item["source_sha256"],
            "TAG_sha256": item["tag_sha256"], "TAG_phone_context_loaded": True, "sample_rate": sr,
            "vowel_context": vowel_context_observation(item),
            "offset_s": offset, "analysis_interval_samples": [begin, end],
            "time_reference": "all trace and support times are seconds relative to offset_s",
            "pcm_bytes_sha256": hashlib.sha256(raw).hexdigest(), "pcm_unchanged": True,
            "noise": noise, "periodicity": periodicity, "intervals": intervals,
            "target_association": review.get("target_association_status", "not_independently_verified") if review else "not_independently_verified",
            "pronunciation_decision": None, "new_human_gold": False}


def summarize(records):
    states = Counter()
    octave_items, branch_items = [], []
    by_view = {}
    for r in records:
        for name, view in r["intervals"].items():
            totals = by_view.setdefault(name, {"status_counts": Counter(), "setting_states": Counter(),
                                              "octave_review_ids": [], "raw_low_return_disagreement_ids": [],
                                              "path_choice_disagreement_ids": [], "path_choice_octave_review_ids": [],
                                              "extends_outside_TAG_vowel_ids": []})
            totals["status_counts"][view["status"]] += 1
            if "TAG_vowel_relation" in view and not view["TAG_vowel_relation"]["within_reference_interval"]:
                totals["extends_outside_TAG_vowel_ids"].append(r["id"])
            if view["status"] != "observed_candidates_only":
                continue
            for setting in view["settings"]:
                for branch, info in setting["branches"].items():
                    totals["setting_states"]["%g:%s:%s" % (setting["floor_hz"], branch, info["state"])] += 1
            if any(v["near_octave_review_flag"] for v in view["cross_floor_comparisons"]):
                totals["octave_review_ids"].append(r["id"])
            if any(v["branch_return_disagreements"] for v in view["settings"]):
                totals["raw_low_return_disagreement_ids"].append(r["id"])
            for field, key in [("path_choice_disagreement_frames", "path_choice_disagreement_ids"),
                               ("path_choice_octave_review_frames", "path_choice_octave_review_ids")]:
                if any(branch[field] for setting in view["settings"] for branch in setting["branches"].values()):
                    totals[key].append(r["id"])
        interval = r["intervals"]["existing_TAG_vowel"]
        if interval["status"] != "observed_candidates_only":
            states[interval["status"]] += 1
            continue
        for setting in interval["settings"]:
            for branch, info in setting["branches"].items():
                states["%g:%s:%s" % (setting["floor_hz"], branch, info["state"])] += 1
        if any(v["near_octave_review_flag"] for v in interval["cross_floor_comparisons"]):
            octave_items.append(r["id"])
        if any(v["branch_return_disagreements"] for v in interval["settings"]):
            branch_items.append(r["id"])
    return {"items": len(records), "TAG_interval_states": dict(states),
            "vowel_context_status_counts": dict(Counter(r["vowel_context"]["status"] for r in records)),
            "vowel_context_disagreement_ids": [r["id"] for r in records if r["vowel_context"]["labels_agree"] is False],
            "by_interval_view": by_view,
            "TAG_interval_octave_review_ids": octave_items,
            "TAG_interval_raw_low_return_disagreement_ids": branch_items,
            "scope": "measurement_diagnostics_not_phonation_or_pronunciation_accuracy",
            "new_human_labels": 0, "pronunciation_scores_produced": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--visual-landmarks", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="+")
    args = ap.parse_args()
    if args.out.exists():
        ap.error("refusing to overwrite an observation run")
    manifest = args.packet / "items.private.jsonl"
    items = [json.loads(s) for s in manifest.read_text().splitlines()]
    if len({r["id"] for r in items}) != len(items):
        ap.error("duplicate packet IDs")
    if args.only:
        if set(args.only) - {r["id"] for r in items}:
            ap.error("unknown requested packet ID")
        items = [r for r in items if r["id"] in args.only]
    visual = {}
    if args.visual_landmarks:
        rows = [json.loads(s) for s in args.visual_landmarks.read_text().splitlines()]
        visual = {r["id"]: r for r in rows}
        if len(visual) != len(rows):
            ap.error("duplicate landmark IDs")
        if set(r["id"] for r in items) - set(visual):
            ap.error("missing landmark records; missing intervals must be explicit")
    args.out.mkdir(parents=True)
    (args.out / "tracks").mkdir()
    source_files = [Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_observations.py"]
    provenance = {str(p): digest(p) for p in [manifest] + source_files + ([args.visual_landmarks] if args.visual_landmarks else [])}
    (args.out / "run_started.json").write_text(json.dumps({"provenance": provenance,
        "scope": "offline_observations_no_classifier_or_service_change"}, indent=2) + "\n")
    records = []
    started = time.monotonic()
    for k, item in enumerate(items, 1):
        record = inspect_item(item, visual.get(item["id"]))
        (args.out / "tracks" / (item["id"] + ".json")).write_text(json.dumps(record, allow_nan=False) + "\n")
        records.append({key: record[key] for key in ("id", "source_sha256", "sample_rate", "intervals", "target_association", "pcm_unchanged", "vowel_context")})
        if k % 10 == 0 or k == len(items):
            print("observed %d/%d %.1fs" % (k, len(items), time.monotonic() - started), flush=True)
    summary = summarize(records)
    summary.update(provenance=provenance, records=records, elapsed_seconds=time.monotonic() - started)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.out / "source_snapshot").mkdir()
    for p in source_files:
        (args.out / "source_snapshot" / p.name).write_bytes(p.read_bytes())
    print(json.dumps({k: v for k, v in summary.items() if k not in ("records", "provenance")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
