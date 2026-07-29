#!/usr/bin/env python3
"""Build policy31 CSV manifests for feature extraction.

Uses:
  * source CSVs from $NATIONALITY_DATA/aihub_nationality_fine_811
  * policy labels from xlsr_nationality_inference_bundle/labels/label_map.json
  * speaker assignment from feature_resplit_policy31_seed42/speaker_assignment.json

Writes train.csv/valid.csv/test.csv plus labels.txt/label2id.json/id2label.json.
"""

from __future__ import annotations
import os

import argparse
import csv
import json
from pathlib import Path


CSV_FIELDS = ["audio_path", "json_path", "speaker_id", "label_name", "label_id", "duration_sec"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_manifest_dir", type=Path, default=Path(os.path.expandvars("$NATIONALITY_DATA/aihub_nationality_fine_811")))
    p.add_argument(
        "--label_map",
        type=Path,
        default=Path(os.path.expandvars("$NATIONALITY_OUTPUTS/xlsr_nationality_inference_bundle/labels/label_map.json")),
    )
    p.add_argument(
        "--speaker_assignment",
        type=Path,
        default=Path(os.path.expandvars("$NATIONALITY_OUTPUTS/feature_resplit_policy31_seed42/speaker_assignment.json")),
    )
    p.add_argument("--out_dir", type=Path, default=Path(os.path.expandvars("$NATIONALITY_OUTPUTS/policy31_manifest_seed42")))
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    policy = json.loads(args.label_map.read_text(encoding="utf-8"))
    label_to_id = policy["label_to_id"]
    id_to_label = policy["id_to_label"]
    aliases = policy["label_rule"]["aliases"]
    assignment = json.loads(args.speaker_assignment.read_text(encoding="utf-8"))

    labels = [id_to_label[str(i)] for i in range(len(id_to_label))]
    (args.out_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    (args.out_dir / "label2id.json").write_text(json.dumps(label_to_id, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "id2label.json").write_text(json.dumps(id_to_label, ensure_ascii=False, indent=2), encoding="utf-8")

    files = {}
    writers = {}
    counts = {split: 0 for split in ["train", "valid", "test"]}
    class_counts = {split: {} for split in ["train", "valid", "test"]}
    speakers = {split: set() for split in ["train", "valid", "test"]}
    try:
        for split in ["train", "valid", "test"]:
            f = (args.out_dir / f"{split}.csv").open("w", encoding="utf-8", newline="")
            files[split] = f
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            writers[split] = w

        seen_audio = set()
        for source_split in ["train", "valid", "test"]:
            src = args.source_manifest_dir / f"{source_split}.csv"
            with src.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    audio_path = row["audio_path"]
                    if audio_path in seen_audio:
                        continue
                    seen_audio.add(audio_path)
                    speaker_id = row["speaker_id"]
                    out_split = assignment.get(speaker_id)
                    if out_split is None:
                        raise KeyError(f"speaker missing from assignment: {speaker_id}")
                    final_label = aliases.get(row["label_name"], row["label_name"])
                    if final_label not in label_to_id:
                        raise KeyError(f"label missing from policy map: {row['label_name']} -> {final_label}")
                    out_row = {
                        "audio_path": audio_path,
                        "json_path": row.get("json_path", ""),
                        "speaker_id": speaker_id,
                        "label_name": final_label,
                        "label_id": label_to_id[final_label],
                        "duration_sec": row.get("duration_sec", ""),
                    }
                    writers[out_split].writerow(out_row)
                    counts[out_split] += 1
                    speakers[out_split].add(speaker_id)
                    class_counts[out_split][final_label] = class_counts[out_split].get(final_label, 0) + 1
    finally:
        for f in files.values():
            f.close()

    stats = {
        "source_manifest_dir": str(args.source_manifest_dir),
        "speaker_assignment": str(args.speaker_assignment),
        "label_map": str(args.label_map),
        "splits": {
            split: {
                "utts": counts[split],
                "speakers": len(speakers[split]),
                "classes_present": len(class_counts[split]),
            }
            for split in ["train", "valid", "test"]
        },
        "classes_absent_from_test": [label for label in labels if class_counts["test"].get(label, 0) == 0],
    }
    (args.out_dir / "manifest_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
