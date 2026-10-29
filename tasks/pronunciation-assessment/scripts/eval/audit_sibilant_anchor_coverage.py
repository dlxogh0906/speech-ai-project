"""Diagnostic TAG interval overlap of frozen runtime anchors, never service input.

TAG intervals are annotation intervals, not independently reviewed acoustic
landmarks. Hidden-state receptive fields also extend beyond nominal frames.
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
import collections
import hashlib
import json
import shutil

import numpy as np

import eval_sibilant_directional as service
from mdd.vad import voiced_mask, trim, FRAME_SEC, KEEP_SEC, MAX_GAP


def retained_samples(wav):
    mask, frame = voiced_mask(wav, 16000)
    if not mask.any():
        return np.arange(len(wav))
    active = np.flatnonzero(mask)
    padding, gap = int(KEEP_SEC / FRAME_SEC), int(MAX_GAP / FRAME_SEC)
    lo, hi = max(0, active[0] - padding), min(len(mask) - 1, active[-1] + padding)
    keep, run = [], 0
    for i in range(lo, hi + 1):
        run = 0 if mask[i] else run + 1
        if run <= gap:
            keep.append(i)
    return np.concatenate([np.arange(i * frame, (i + 1) * frame) for i in keep])


def summarize(rows):
    valid = [r for r in rows if r["audit_state"] == "available"]
    return {"all_sites": len(rows), "audit_states": dict(collections.Counter(r["audit_state"] for r in rows)),
            "available": len(valid), "anchor_in_target_tag": sum(r["anchor_in_target_tag"] for r in valid),
            "anchor_in_following_vowel_tag": sum(r["anchor_in_following_vowel_tag"] for r in valid),
            "window_has_both_target_and_vowel": sum(r["target_frames"] > 0 and r["vowel_frames"] > 0 for r in valid),
            "other_frame_fraction_p50_p90": np.quantile([r["other_fraction"] for r in valid], [.5, .9]).tolist() if valid else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, out = Path(args.features), Path(args.out)
    responses = service.read_jsonl(source / (args.split + ".responses.jsonl"))
    features = service.read_jsonl(source / (args.split + ".features.jsonl"))
    labels = service.read_jsonl(source / (args.split + ".labels.jsonl"))
    wanted = {r["uid"] for r in responses}
    annotations = {}
    with Path(args.metadata).open() as stream:
        for line in stream:
            meta = json.loads(line)
            if meta["stem"] in wanted:
                if meta["stem"] in annotations:
                    raise ValueError("duplicate annotation UID")
                annotations[meta["stem"]] = meta
    if set(annotations) != wanted:
        raise ValueError("missing source annotations")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    by_uid = collections.defaultdict(list)
    for row in labels:
        by_uid[row["uid"]].append(row)
    rows, reproduced = [], 0
    for record in responses:
        row = record["row"]
        tag = [t for t in annotations[row["uid"]]["tag_phones"] if t[2] not in ("", ".", "-", "spn", "<unk>")]
        tag_matches = [t[2] for t in tag] == row["actual"]
        source_times = None
        if record["raw"] is not None:
            wav = service.load_audio(row)[:320000]
            indices = retained_samples(wav)
            clipped, _ = trim(wav, 16000)
            np.testing.assert_array_equal(wav[indices].astype(np.float32), clipped)
            digest = hashlib.sha256(clipped.tobytes()).hexdigest()
            for i in record["feature_indices"].values():
                if digest != features[i]["vad_pcm_sha256"]:
                    raise AssertionError("runtime VAD PCM changed")
            reproduced += 1
            source_times = indices / 16000.
        for label in by_uid[row["uid"]]:
            result = {k: label.get(k) for k in ("uid", "item", "speaker", "expected", "actual", "following", "position", "state")}
            result["prompt"] = row["prompt"]
            result["audit_state"] = label["state"]
            if label["feature_index"] is not None:
                if not tag_matches:
                    result["audit_state"] = "tag_sequence_mismatch"
                else:
                    a = label["a"]
                    target, vowel = tag[a], tag[a + 1]
                    if not (target[0] < target[1] <= vowel[1] and vowel[0] < vowel[1]):
                        result["audit_state"] = "invalid_tag_interval"
                    else:
                        feature = features[label["feature_index"]]
                        frames = np.arange(feature["anchor_frame"] - 15, feature["anchor_frame"] + 16)
                        nominal_samples = np.floor((frames + .5) * feature["frame_seconds"] * 16000).astype(int)
                        valid = (nominal_samples >= 0) & (nominal_samples < len(source_times))
                        times = source_times[nominal_samples[valid]]
                        target_mask = (times >= target[0]) & (times < target[1])
                        vowel_mask = (times >= vowel[0]) & (times < vowel[1])
                        anchor_sample = int((feature["anchor_frame"] + .5) * feature["frame_seconds"] * 16000)
                        anchor_time = float(source_times[anchor_sample])
                        result.update(audit_state="available", target_tag_seconds=target[:2], vowel_tag_seconds=vowel[:2],
                                      anchor_source_seconds=anchor_time,
                                      anchor_in_target_tag=bool(target[0] <= anchor_time < target[1]),
                                      anchor_in_following_vowel_tag=bool(vowel[0] <= anchor_time < vowel[1]),
                                      valid_frames=int(valid.sum()), target_frames=int(target_mask.sum()), vowel_frames=int(vowel_mask.sum()),
                                      other_fraction=float((~(target_mask | vowel_mask)).mean()))
            rows.append(result)
    with (out / "sites.jsonl").open("w") as stream:
        for row in rows:
            service.write_line(stream, row)
    report = {"scope": __doc__, "reproduced_vad_inputs": reproduced, "overall": summarize(rows),
              "source_sha256": {str(p): service.sha(p) for p in [Path(__file__), Path(args.metadata)] +
                                [source / (args.split + s) for s in (".features.jsonl", ".responses.jsonl", ".labels.jsonl")]}}
    for field in ("item", "following", "position"):
        report[field] = {str(k): summarize([r for r in rows if r[field] == k]) for k in sorted({r[field] for r in rows}, key=str)}
    report["error_only"] = summarize([r for r in rows if r["expected"] != r["actual"]])
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["reproduced_vad_inputs", "overall", "error_only"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
