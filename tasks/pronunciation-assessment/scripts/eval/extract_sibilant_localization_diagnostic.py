"""Prepare matched runtime/TAG-localized readouts without fitting on heldout data."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/train")]

import argparse
import collections
import gzip
import hashlib
import json
import shutil
import time

import numpy as np
import torch

import eval_sibilant_directional as service
from audit_sibilant_anchor_coverage import retained_samples
from train_sibilant_linear import frozen_load
from mdd.vad import trim
from mdd.sibilant_hidden import project_sequences
from mdd.sibilant_linear import temporal_summary
from mdd.sibilant_localization_diagnostic import localized_summary


def collect(path, wanted):
    selected = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            uid = row["stem"]
            if uid in wanted:
                value = {k: row[k] for k in ["stem", "spk", "item", "tag_phones"]}
                if uid in selected and selected[uid] != value:
                    raise ValueError("conflicting annotation identity: " + uid)
                selected[uid] = value
            if number % 250000 == 0:
                print("annotation rows", number, "selected", len(selected), flush=True)
    if set(selected) != wanted:
        raise ValueError("missing annotations: %d" % len(wanted - set(selected)))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    for p, digest in protocol["source_sha256"].items():
        if service.sha(p) != digest:
            raise ValueError("frozen diagnostic source changed: " + p)
    previous = json.loads(Path(protocol["linear_protocol"]).read_text())
    source = Path(previous["hidden_features"])
    cache_protocol = json.loads((source / "metadata.json").read_text())["protocol"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    if service.sha(previous["projection_artifact"]) != previous["projection_artifact_sha256"]:
        raise ValueError("original training-only PCA changed")
    projection = torch.load(previous["projection_artifact"], map_location="cpu")["projection"]
    wanted = {}
    for split in ("train", "cal", "test"):
        cache = Path(cache_protocol["cache_sources"][split]["path"])
        wanted[split] = {r["uid"] for r in service.read_jsonl(cache / (split + ".responses.jsonl"))}
    train_tags = collect(protocol["train_annotations"], wanted["train"])
    evaluation_tags = collect(protocol["evaluation_annotations"], wanted["cal"] | wanted["test"])
    for name, values in [("train", train_tags), ("evaluation", evaluation_tags)]:
        with (out / (name + ".annotations.jsonl")).open("w") as stream:
            for uid in sorted(values):
                service.write_line(stream, values[uid])
    report = {"scope": "TAG intervals supplied to diagnostic feature pooling only; not runtime performance", "splits": {}}
    for split in ("train", "cal", "test"):
        tick = time.monotonic()
        arrays, labels, features, responses = frozen_load(source, cache_protocol, previous, split)
        projected = project_sequences(arrays["h24"], arrays["mask"], projection)
        runtime = temporal_summary(projected, arrays["mask"], arrays["context"], True)
        phases = np.zeros((len(features), 3, 31), bool)
        by_uid = collections.defaultdict(list)
        for label in labels:
            by_uid[label["uid"]].append(label)
        tags = train_tags if split == "train" else evaluation_tags
        audit, reproduced = [], 0
        for record in responses:
            row = record["row"]
            meta = tags[row["uid"]]
            if meta["spk"] != row["speaker"] or meta["item"] != row["item"]:
                raise ValueError("annotation identity mismatch")
            tag = [t for t in meta["tag_phones"] if t[2] not in ("", ".", "-", "spn", "<unk>")]
            tag_matches = [t[2] for t in tag] == row["actual"]
            source_times = None
            if record["raw"] is not None:
                wav = service.load_audio(row)[:320000]
                indices = retained_samples(wav)
                clipped, _ = trim(wav, 16000)
                np.testing.assert_array_equal(wav[indices].astype(np.float32), clipped)
                digest = hashlib.sha256(clipped.tobytes()).hexdigest()
                if any(digest != features[i]["vad_pcm_sha256"] for i in record["feature_indices"].values()):
                    raise ValueError("VAD PCM changed")
                source_times = indices / 16000.
                reproduced += 1
            for label in by_uid[row["uid"]]:
                ix = label["feature_index"]
                entry = {k: label[k] for k in ["uid", "q", "expected", "actual", "state", "feature_index"]}
                entry["diagnostic_state"] = label["state"]
                if ix is not None:
                    if not tag_matches:
                        entry["diagnostic_state"] = "tag_sequence_mismatch"
                    else:
                        a = label["a"]
                        c, v = tag[a], tag[a + 1]
                        if not (c[0] < c[1] <= v[0] < v[1]):
                            entry["diagnostic_state"] = "invalid_tag_intervals"
                        else:
                            feature = features[ix]
                            frame = np.arange(feature["anchor_frame"] - 15, feature["anchor_frame"] + 16)
                            samples = np.floor((frame + .5) * feature["frame_seconds"] * 16000).astype(int)
                            valid = arrays["mask"][ix] & (samples >= 0) & (samples < len(source_times))
                            times = np.full(31, np.nan)
                            times[valid] = source_times[samples[valid]]
                            mid = (v[0] + v[1]) / 2
                            for j, (lo, hi) in enumerate([(c[0], c[1]), (v[0], mid), (mid, v[1])]):
                                phases[ix, j] = valid & (times >= lo) & (times < hi)
                            counts = phases[ix].sum(1).tolist()
                            entry.update(target_tag_seconds=c[:2], vowel_tag_seconds=v[:2], phase_frames=counts)
                            if not all(counts):
                                entry["diagnostic_state"] = "phase_without_observed_frame"
                audit.append(entry)
        localized, available = localized_summary(projected, arrays["mask"], arrays["context"], phases)
        used = [r for r in labels if r["feature_index"] is not None and available[r["feature_index"]]]
        np.savez_compressed(out / (split + ".npz"), runtime=runtime, localized=localized,
                            available=available, phase_masks=phases)
        with (out / (split + ".audit.jsonl")).open("w") as stream:
            for value in audit:
                service.write_line(stream, value)
        report["splits"][split] = {"runtime_features": len(features), "localized_available": int(available.sum()),
                                   "states": dict(collections.Counter(r["diagnostic_state"] for r in audit)),
                                   "supported_groups": dict(collections.Counter(r["expected"] + "→" + r["actual"] for r in used)),
                                   "reproduced_vad_inputs": reproduced, "seconds": time.monotonic() - tick,
                                   "matrix_sha256": service.sha(out / (split + ".npz"))}
        print(split, report["splits"][split], flush=True)
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps(report["splits"], ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
