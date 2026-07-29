#!/usr/bin/env python3
"""Create a policy-aware 31-class speaker-disjoint split from cached features.

This differs from 61-class split -> policy remap. Here, labels are first remapped
through LABEL_POLICY/label_map.json, then speakers are split per final 31-class
label. That makes two-speaker merged labels such as 국가명_불명확 and
희소_인도유럽_기타 eligible for train/test splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--features_dir", type=Path, default=Path("$NATIONALITY_RUNS/features_fine_811"))
    p.add_argument(
        "--label_map",
        type=Path,
        default=Path("$NATIONALITY_OUTPUTS/xlsr_nationality_inference_bundle/labels/label_map.json"),
    )
    p.add_argument("--out_dir", type=Path, default=Path("$NATIONALITY_OUTPUTS/feature_resplit_policy31_seed42"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--valid_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--rounding", choices=["floor", "round", "ceil"], default="round")
    return p.parse_args()


def ratio_count(n, ratio, rounding):
    value = n * ratio
    if rounding == "floor":
        return int(math.floor(value))
    if rounding == "ceil":
        return int(math.ceil(value))
    return int(round(value))


def quotas(n, valid_ratio, test_ratio, rounding):
    if n == 1:
        return {"train": 1, "valid": 0, "test": 0}
    if n == 2:
        return {"train": 1, "valid": 0, "test": 1}
    nv = max(1, ratio_count(n, valid_ratio, rounding))
    nt = max(1, ratio_count(n, test_ratio, rounding))
    if nv + nt >= n:
        nv = 1
        nt = 1
    return {"train": n - nv - nt, "valid": nv, "test": nt}


def load_all(features_dir):
    xs, ys, speakers, source_split = [], [], [], []
    for split in ["train", "valid", "test"]:
        d = np.load(features_dir / f"{split}.npz", allow_pickle=True)
        xs.append(d["X"])
        ys.append(d["y"].astype(np.int64))
        speakers.append(d["speaker"].astype(str))
        source_split.extend([split] * len(d["y"]))
        print(f"loaded {split}: X={d['X'].shape}", flush=True)
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(speakers, axis=0),
        np.asarray(source_split, dtype=object),
    )


def remap_y(y61, source_id2label, policy):
    aliases = policy["label_rule"]["aliases"]
    label_to_id = policy["label_to_id"]
    y31 = np.empty_like(y61)
    for i, yi in enumerate(y61):
        src = source_id2label[str(int(yi))]
        final = aliases.get(src, src)
        y31[i] = label_to_id[final]
    return y31


def assign_by_policy_label(y, speaker, seed, valid_ratio, test_ratio, rounding):
    by_label_spk = defaultdict(lambda: defaultdict(int))
    for yi, spk in zip(y, speaker):
        by_label_spk[int(yi)][spk] += 1

    rng = random.Random(seed)
    assignment = {}
    label_assignment = {}
    for label in sorted(by_label_spk):
        speakers = sorted(by_label_spk[label])
        rng.shuffle(speakers)
        q = quotas(len(speakers), valid_ratio, test_ratio, rounding)
        if len(speakers) == 1:
            parts = {"train": speakers, "valid": [], "test": []}
        elif len(speakers) == 2:
            parts = {"train": [speakers[0]], "valid": [], "test": [speakers[1]]}
        else:
            ntr, nv = q["train"], q["valid"]
            parts = {
                "train": speakers[:ntr],
                "valid": speakers[ntr : ntr + nv],
                "test": speakers[ntr + nv :],
            }
        label_assignment[str(label)] = parts
        for split, split_speakers in parts.items():
            for spk in split_speakers:
                if spk in assignment and assignment[spk] != split:
                    raise RuntimeError(f"speaker {spk} assigned twice")
                assignment[spk] = split
    return assignment, label_assignment


def summarize(y, speaker, assignment, id_to_label):
    split_utts = Counter()
    split_speakers = defaultdict(set)
    split_labels = defaultdict(set)
    split_label_utts = defaultdict(Counter)
    split_label_speakers = defaultdict(lambda: defaultdict(set))
    for yi, spk in zip(y, speaker):
        split = assignment[spk]
        label = id_to_label[str(int(yi))]
        split_utts[split] += 1
        split_speakers[split].add(spk)
        split_labels[split].add(label)
        split_label_utts[split][label] += 1
        split_label_speakers[split][label].add(spk)

    overlaps = {}
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        overlaps[f"{a}_vs_{b}"] = len(split_speakers[a] & split_speakers[b])

    labels = [id_to_label[str(i)] for i in range(len(id_to_label))]
    rows = []
    for label in labels:
        row = {"label_id": labels.index(label), "label": label}
        for split in ["train", "valid", "test"]:
            row[split] = int(split_label_utts[split][label])
            row[f"{split}_speakers"] = len(split_label_speakers[split][label])
        rows.append(row)

    return {
        "splits": {
            split: {
                "utts": int(split_utts[split]),
                "speakers": len(split_speakers[split]),
                "classes_present": len(split_labels[split]),
            }
            for split in ["train", "valid", "test"]
        },
        "speaker_overlap": overlaps,
        "classes_absent_from_test": [r["label"] for r in rows if r["test"] == 0],
        "class_rows": rows,
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_id2label = json.loads((args.features_dir / "id2label.json").read_text(encoding="utf-8"))
    policy = json.loads(args.label_map.read_text(encoding="utf-8"))
    id_to_label = policy["id_to_label"]
    label_to_id = policy["label_to_id"]

    X, y61, speaker, source_split = load_all(args.features_dir)
    y31 = remap_y(y61, source_id2label, policy)
    assignment, label_assignment = assign_by_policy_label(
        y31, speaker, args.seed, args.valid_ratio, args.test_ratio, args.rounding
    )
    summary = summarize(y31, speaker, assignment, id_to_label)
    if any(summary["speaker_overlap"].values()):
        raise RuntimeError(summary["speaker_overlap"])

    print("summary:", json.dumps(summary["splits"], ensure_ascii=False), flush=True)
    print("speaker_overlap:", summary["speaker_overlap"], flush=True)
    print("classes_absent_from_test:", summary["classes_absent_from_test"], flush=True)

    for split in ["train", "valid", "test"]:
        idx = np.asarray([i for i, spk in enumerate(speaker) if assignment[spk] == split], dtype=np.int64)
        np.savez(
            args.out_dir / f"{split}.npz",
            X=X[idx],
            y=y31[idx],
            speaker=speaker[idx],
            source_y61=y61[idx],
            source_split=source_split[idx],
            source_index=idx,
        )
        print(f"wrote {split}: X={X[idx].shape}", flush=True)

    (args.out_dir / "id2label.json").write_text(json.dumps(id_to_label, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "label2id.json").write_text(json.dumps(label_to_id, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "speaker_assignment.json").write_text(json.dumps(assignment, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "label_speaker_assignment.json").write_text(
        json.dumps(label_assignment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stats = {k: v for k, v in summary.items() if k != "class_rows"}
    stats.update(
        {
            "features_dir": str(args.features_dir),
            "label_map": str(args.label_map),
            "out_dir": str(args.out_dir),
            "seed": args.seed,
            "ratios": {"train": 1 - args.valid_ratio - args.test_ratio, "valid": args.valid_ratio, "test": args.test_ratio},
            "rounding": args.rounding,
            "rules": [
                "remap labels to policy31 first",
                "speaker-disjoint",
                "1-speaker labels -> train only",
                "2-speaker labels -> train/test, no valid",
                ">=3-speaker labels -> seed shuffle, speaker-count ratio split",
            ],
        }
    )
    (args.out_dir / "split_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out_dir / "class_counts.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["label_id", "label", "train", "train_speakers", "valid", "valid_speakers", "test", "test_speakers"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary["class_rows"])


if __name__ == "__main__":
    main()
