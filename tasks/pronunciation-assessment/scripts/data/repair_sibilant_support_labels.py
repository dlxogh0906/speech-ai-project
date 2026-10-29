"""Reuse frozen audio/features and repair train-only TAG index resolution."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/eval"), str(ROOT / "tasks/pronunciation-assessment/scripts/train")]
import argparse
import json
import collections
import numpy as np
import eval_sibilant_directional as service
from extract_sibilant_localization_diagnostic import collect
from audit_sibilant_anchor_coverage import retained_samples
from mdd.sibilant_transition import VOWELS
from mdd.canonical_fix import CONSONANTS
from mdd.sibilant_tag_support import single_phone_intervals, coarse_roles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed")
    source, out = Path(a.source), Path(a.out)
    if not (source / "finished.json").exists():
        raise ValueError("original preparation incomplete")
    out.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.name not in ("train.npz", "report.json", "finished.json", "protocol.json"):
            (out / path.name).symlink_to(path.resolve())
    records = service.read_jsonl(source / "train.responses.jsonl")
    features = service.read_jsonl(source / "train.features.jsonl")
    labels = service.read_jsonl(source / "train.labels.jsonl")
    data = dict(np.load(source / "train.npz"))
    phases = data["phase"].copy()
    annotations = collect(p["train_annotations"], {r["row"]["uid"] for r in records})
    by_uid = collections.defaultdict(list)
    for label in labels:
        by_uid[label["uid"]].append(label)
    counts = collections.Counter()
    for record in records:
        row = record["row"]
        intervals = single_phone_intervals(annotations[row["uid"]]["tag_phones"], row["actual"])
        if intervals is None:
            counts["still_sequence_unresolved_utterances"] += 1
            continue
        needed = [r for r in by_uid[row["uid"]] if r["feature_index"] is not None and not r["alignment_ambiguous"]
                  and r["a"] is not None and r["actual"] in CONSONANTS]
        if not needed:
            continue
        x = service.load_audio(row)[:320000]
        indices = retained_samples(x)
        for label in needed:
            ix = label["feature_index"]
            f = features[ix]
            length = int(data["lengths"][ix])
            centers = f["left"] + (np.arange(length) + .5) * (f["right"] - f["left"]) / length
            times = indices[np.minimum(centers.astype(int), len(indices) - 1)] / 16000.
            new = coarse_roles(times, intervals, label["a"], VOWELS, p["vowel_prefix_seconds"], p["edge_ignore_seconds"])
            old = phases[ix, :length]
            if (old != -100).any() and not np.array_equal(old, new):
                raise AssertionError("previously supported roles changed")
            counts["newly_supervised_sites"] += not (old != -100).any() and (new != -100).any()
            counts["still_unknown_sites"] += not (new != -100).any()
            phases[ix, :length] = new
    data["phase"] = phases
    np.savez(out / "train.npz", **data)
    original = np.load(source / "train.npz")
    for key in ("x", "relative", "lengths", "y"):
        np.testing.assert_array_equal(data[key], original[key])
    report = {"scope": __doc__, "counts": {k: int(v) for k, v in counts.items()}, "nonphase_arrays_exact": ["x", "relative", "lengths", "y"],
              "supervised_sites_before_after": [int((original["phase"] != -100).any(1).sum()), int((phases != -100).any(1).sum())],
              "original_features_sha256": service.sha(source / "train.npz"), "repaired_sha256": service.sha(out / "train.npz")}
    (out / "protocol.json").write_text(json.dumps(p, ensure_ascii=False, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"sha256": service.sha(out / "report.json")}) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
