"""Train-only label clock audit, leaving all inference inputs unchanged.

Reproduce old coarse TAG role labels at the old uniformly stretched clip clock,
then recompute them at the actual analysis-window centers. Do not infer physical
frication/aspiration/glottal boundaries from these phonemic intervals.
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
import eval_sibilant_directional as service
from extract_sibilant_localization_diagnostic import collect
from audit_sibilant_anchor_coverage import retained_samples
from mdd.sibilant_transition import VOWELS
from mdd.canonical_fix import CONSONANTS
from mdd.sibilant_tag_support import single_phone_intervals, coarse_roles
from mdd.sibilant_frame_clock import stacked_window_centers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    prefix = p["train_prefix"]
    records = service.read_jsonl(prefix + ".responses.jsonl")
    features = service.read_jsonl(prefix + ".features.jsonl")
    labels = service.read_jsonl(prefix + ".labels.jsonl")
    data = dict(np.load(prefix + ".npz"))
    phases = data["phase"].copy()
    annotations = collect(p["train_annotations"], {r["row"]["uid"] for r in records})
    by_uid = collections.defaultdict(list)
    for label in labels:
        by_uid[label["uid"]].append(label)
    counts, subgroups, changed = collections.Counter(), collections.defaultdict(collections.Counter), []
    shifts, seen = [], set()
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
        pcm_digest = hashlib.sha256(x[indices].tobytes()).hexdigest()
        for z in needed:
            ix = z["feature_index"]
            if ix in seen:
                raise AssertionError("ambiguous duplicate train role index")
            seen.add(ix)
            f, length = features[ix], int(data["lengths"][ix])
            if pcm_digest != f["vad_pcm_sha256"]:
                raise AssertionError("original VAD waveform differs")
            old_centers = f["left"] + (np.arange(length) + .5) * (f["right"] - f["left"]) / length
            centers = stacked_window_centers(f["left"], f["right"], length)
            old_times = indices[np.minimum(old_centers.astype(int), len(indices) - 1)] / 16000.
            times = indices[np.minimum(centers.astype(int), len(indices) - 1)] / 16000.
            old = coarse_roles(old_times, intervals, z["a"], VOWELS, p["vowel_prefix_seconds"], p["edge_ignore_seconds"])
            np.testing.assert_array_equal(old, data["phase"][ix, :length])
            new = coarse_roles(times, intervals, z["a"], VOWELS, p["vowel_prefix_seconds"], p["edge_ignore_seconds"])
            phases[ix, :length] = new
            shifts.extend(((old_centers - centers) / 16.).tolist())
            delta = int((old != new).sum())
            counts["old_role_sites_reproduced_exactly"] += 1
            counts["changed_sites"] += int(delta > 0)
            counts["changed_frames"] += delta
            counts["newly_complete_C_E_L"] += int(all((new == k).any() for k in [1, 2, 3]) and not all((old == k).any() for k in [1, 2, 3]))
            counts["lost_complete_C_E_L"] += int(all((old == k).any() for k in [1, 2, 3]) and not all((new == k).any() for k in [1, 2, 3]))
            group = z.get("following", "unknown")
            subgroups[group]["sites"] += 1
            subgroups[group]["changed_sites"] += int(delta > 0)
            if delta:
                changed.append({"uid": row["uid"], "q": z["q"], "following": group, "changed_frames": delta})
        counts["VAD_waveforms_exact"] += 1
        if counts["VAD_waveforms_exact"] % 1000 == 0:
            print(dict(counts), flush=True)
    data["phase"] = phases
    np.savez(out / "train.npz", **data)
    for suffix in (".features.jsonl", ".labels.jsonl", ".responses.jsonl"):
        (out / ("train" + suffix)).symlink_to(Path(prefix + suffix).resolve())
    report = {"scope": p["scope"], "counts": dict(counts), "subgroups": dict(subgroups),
              "old_minus_analysis_center_ms_min_p50_max": np.quantile(shifts, [0, .5, 1]).tolist(),
              "nonphase_arrays_preserved": ["x", "relative", "lengths", "y"],
              "model_trained": False, "service_improvement_proven": False,
              "corrected_train_sha256": service.sha(out / "train.npz")}
    (out / "changed_sites.json").write_text(json.dumps(changed, ensure_ascii=False, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"sha256": service.sha(out / "report.json")}) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
