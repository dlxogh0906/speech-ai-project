"""Audit/train-only weak window occupancy, preserving existing acoustic inputs.

Reproduce corrected hard roles first. Compute weighted phonemic label support
over the two analysis windows, retaining uncertainty around old TAG boundaries.
These are weak targets, not measured acoustic source proportions or new labels.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/eval")]
import argparse
import collections
import hashlib
import json
import shutil
import numpy as np
from transformers import SeamlessM4TFeatureExtractor
import eval_sibilant_directional as service
from extract_sibilant_localization_diagnostic import collect
from audit_sibilant_anchor_coverage import retained_samples
from mdd.sibilant_transition import VOWELS
from mdd.canonical_fix import CONSONANTS
from mdd.sibilant_tag_support import single_phone_intervals, coarse_roles
from mdd.sibilant_frame_clock import stacked_window_centers
from mdd.sibilant_window_support import stacked_support_kernel, window_role_support


def summarize(rows):
    known = [r for r in rows if r["target_known_frames"]]
    return {"all_sites": len(rows), "target_supported_sites": len(known),
            "weak_target_strict_three_peak_possible": sum(r["strict_target"] for r in known),
            "weak_target_joint_CV_peak_possible": sum(r["joint_target"] for r in known),
            "vowel_duration_p10_p50_p90_ms": np.quantile([r["vowel_duration_ms"] for r in known], [.1, .5, .9]).tolist() if known else None,
            "shorter_than40ms": sum(r["vowel_duration_ms"] < 40 for r in known),
            "hard_CEL_frames": sum(r["hard_CEL_frames"] for r in rows),
            "hard_CEL_frames_mixed_gt20percent": sum(r["hard_CEL_mixed"] for r in rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    fe = SeamlessM4TFeatureExtractor.from_pretrained(p["checkpoint"])
    kernel = stacked_support_kernel(fe.window)
    np.save(out / "analysis_kernel.npy", kernel)
    prefix = p["train_prefix"]
    records = service.read_jsonl(prefix + ".responses.jsonl")
    features = service.read_jsonl(prefix + ".features.jsonl")
    labels = service.read_jsonl(prefix + ".labels.jsonl")
    data = dict(np.load(prefix + ".npz"))
    occupancy = np.zeros(data["phase"].shape + (4,), np.float32)
    annotations = collect(p["train_annotations"], {r["row"]["uid"] for r in records})
    with (out / "train_only_annotations.jsonl").open("x") as stream:
        for uid in sorted(annotations):
            service.write_line(stream, annotations[uid])
    by_uid = collections.defaultdict(list)
    for z in labels:
        by_uid[z["uid"]].append(z)
    counts, rows, seen = collections.Counter(), [], set()
    for record in records:
        row = record["row"]
        intervals = single_phone_intervals(annotations[row["uid"]]["tag_phones"], row["actual"])
        if intervals is None:
            counts["TAG_sequence_unresolved_utterances"] += 1
            continue
        needed = [z for z in by_uid[row["uid"]] if z.get("feature_index") is not None and not z["alignment_ambiguous"]
                  and z.get("a") is not None and z["actual"] in CONSONANTS]
        if not needed:
            continue
        x = service.load_audio(row)[:320000]
        indices = retained_samples(x)
        if hashlib.sha256(x[indices].tobytes()).hexdigest() != features[needed[0]["feature_index"]]["vad_pcm_sha256"]:
            raise AssertionError("VAD input differs")
        for z in needed:
            i, actual_index = z["feature_index"], z["a"]
            if i in seen:
                raise AssertionError("duplicate target index")
            seen.add(i)
            f, length = features[i], int(data["lengths"][i])
            centers = stacked_window_centers(f["left"], f["right"], length) - f["left"]
            times = indices[f["left"]:f["right"]] / 16000.
            sample_roles = coarse_roles(times, intervals, actual_index, VOWELS, p["vowel_prefix_seconds"], p["edge_ignore_seconds"])
            hard = sample_roles[centers.astype(int)]
            np.testing.assert_array_equal(hard, data["phase"][i, :length])
            raw = window_role_support(sample_roles, 0, len(times), length, kernel, 0.)
            valid = raw.sum(-1) >= p["minimum_known_mass"]
            support = raw.copy()
            support[~valid] = 0
            occupancy[i, :length] = support
            conditional = support / np.maximum(support.sum(-1, keepdims=True), 1e-20)
            maxima, peaks = conditional.max(0), conditional.argmax(0)
            strict = bool((maxima[1:] >= .5).all() and peaks[1] < peaks[2] < peaks[3])
            joint_v = conditional[:, 2:].sum(-1)
            joint = bool(maxima[1] >= .5 and joint_v.max() >= .5 and peaks[1] < joint_v.argmax())
            c_or_v = (hard > 0) & valid
            own = conditional[np.arange(length), hard.clip(min=0)]
            mixed = int((c_or_v & (own < .8)).sum())
            v = intervals[actual_index + 1] if actual_index + 1 < len(intervals) else None
            value = {"uid": row["uid"], "q": z["q"], "feature_index": i, "following": z.get("following"),
                     "actual_following": v[2] if v else None, "target_known_frames": int(valid.sum()),
                     "strict_target": strict, "joint_target": joint,
                     "vowel_duration_ms": (v[1] - v[0]) * 1000 if v else None,
                     "hard_CEL_frames": int(c_or_v.sum()), "hard_CEL_mixed": mixed,
                     "maximum_conditional_C_E_L": maxima[1:].tolist()}
            rows.append(value)
            counts["corrected_hard_sites_reproduced"] += 1
            counts["known_window_frames"] += int(valid.sum())
            counts["hard_CEL_frames_all"] += int((hard > 0).sum())
            counts["hard_CEL_frames_insufficient_known_mass"] += int(((hard > 0) & ~valid).sum())
            counts["hard_unknown_newly_supported_frames"] += int(((hard == -100) & valid).sum())
        counts["VAD_waveforms_exact"] += 1
        if counts["VAD_waveforms_exact"] % 1000 == 0:
            print(dict(counts), flush=True)
    np.savez(out / "train_support.npz", support=occupancy)
    with (out / "sites.jsonl").open("x") as stream:
        for value in rows:
            service.write_line(stream, value)
    total = summarize(rows)
    report = {"scope": p["scope"], "counts": dict(counts), "total_input_features": len(features), "summary": total,
              "by_following": {str(v): summarize([r for r in rows if r["following"] == v]) for v in sorted({r["following"] for r in rows}, key=str)},
              "source_inputs_untouched": True, "GT_model_inputs": False, "service_improvement_proven": False,
              "support_sha256": service.sha(out / "train_support.npz"), "train_annotations_sha256": service.sha(out / "train_only_annotations.jsonl")}
    report["overlap_hypothesis_supported"] = bool(total["hard_CEL_frames"] >= p["criterion"]["minimum_hard_frames"]
        and total["hard_CEL_frames_mixed_gt20percent"] >= p["criterion"]["minimum_mixed_fraction"] * total["hard_CEL_frames"])
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"sha256": service.sha(out / "report.json")}) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
