#!/usr/bin/env python3
"""Resplit cached 61-class XLS-R features with speaker-disjoint rare-class rules.

Input features are the already-extracted cached features:
  $NATIONALITY_RUNS/features_fine_811/{train,valid,test}.npz

No audio is loaded and no XLS-R forward pass is run. The script concatenates the
existing feature cache, assigns speakers to new splits per label, and writes new
feature npz files under $NATIONALITY_OUTPUTS.

Rules:
  * split by speaker; no speaker appears in more than one split
  * labels with 1 speaker: train only
  * labels with 2 speakers: train/test, no valid
  * labels with >=3 speakers: seed shuffle, then speaker-count 0.8/0.1/0.1
    with at least one valid and one test speaker
"""

from __future__ import annotations

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_OUTPUTS, NAT_RUNS



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
    p.add_argument("--features_dir", type=Path, default=Path(f"{NAT_RUNS}/features_fine_811"))
    p.add_argument("--out_dir", type=Path, default=Path(f"{NAT_OUTPUTS}/feature_resplit_61_merged6_seed42"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--valid_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument(
        "--rounding",
        choices=["floor", "round", "ceil"],
        default="round",
        help="How to convert valid/test speaker counts for labels with >=3 speakers.",
    )
    p.add_argument(
        "--strategy",
        choices=["shuffle_slices", "target_greedy"],
        default="shuffle_slices",
        help=(
            "shuffle_slices follows the literal seed-shuffle then split-slice rule. "
            "target_greedy keeps the same per-label speaker quotas but chooses speakers "
            "to match target sample counts, then uses within-label swaps to improve exactness."
        ),
    )
    p.add_argument("--target_train", type=int, default=None)
    p.add_argument("--target_valid", type=int, default=None)
    p.add_argument("--target_test", type=int, default=None)
    return p.parse_args()


def read_features(features_dir: Path):
    arrays = []
    labels = []
    speakers = []
    source_split = []
    for split in ["train", "valid", "test"]:
        path = features_dir / f"{split}.npz"
        d = np.load(path, allow_pickle=True)
        x = d["X"]
        y = d["y"].astype(np.int64)
        spk = d["speaker"].astype(str)
        arrays.append(x)
        labels.append(y)
        speakers.append(spk)
        source_split.extend([split] * len(y))
        print(f"loaded {split}: X={x.shape} y={y.shape} speaker={spk.shape}", flush=True)
    return (
        np.concatenate(arrays, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(speakers, axis=0),
        np.asarray(source_split, dtype=object),
    )


def count_for_ratio(n: int, ratio: float, rounding: str):
    value = n * ratio
    if rounding == "floor":
        return int(math.floor(value))
    if rounding == "ceil":
        return int(math.ceil(value))
    return int(round(value))


def split_quotas(n, valid_ratio, test_ratio, rounding):
    if n == 1:
        return {"train": 1, "valid": 0, "test": 0}
    if n == 2:
        return {"train": 1, "valid": 0, "test": 1}
    n_valid = max(1, count_for_ratio(n, valid_ratio, rounding))
    n_test = max(1, count_for_ratio(n, test_ratio, rounding))
    if n_valid + n_test >= n:
        n_valid = 1
        n_test = 1
    return {"train": n - n_valid - n_test, "valid": n_valid, "test": n_test}


def assign_speakers(y, speaker, seed, valid_ratio, test_ratio, rounding,
                    strategy="shuffle_slices", target_counts=None):
    by_label = defaultdict(set)
    label_speaker_utts = defaultdict(lambda: defaultdict(int))
    for yi, spk in zip(y, speaker):
        yi = int(yi)
        by_label[yi].add(spk)
        label_speaker_utts[yi][spk] += 1

    rng = random.Random(seed)
    assignment = {}
    label_assignment = {}

    label_order = sorted(by_label)
    if strategy == "target_greedy":
        label_order = sorted(by_label, key=lambda lab: (len(by_label[lab]), lab))

    current_counts = Counter()
    speaker_size = {}
    speaker_label = {}

    for label in label_order:
        speakers = sorted(by_label[label])
        n = len(speakers)
        quotas = split_quotas(n, valid_ratio, test_ratio, rounding)

        if strategy == "shuffle_slices":
            rng.shuffle(speakers)
            if n == 1:
                parts = {"train": speakers, "valid": [], "test": []}
            elif n == 2:
                parts = {"train": [speakers[0]], "valid": [], "test": [speakers[1]]}
            else:
                n_train = quotas["train"]
                n_valid = quotas["valid"]
                parts = {
                    "train": speakers[:n_train],
                    "valid": speakers[n_train : n_train + n_valid],
                    "test": speakers[n_train + n_valid :],
                }
        else:
            if target_counts is None:
                raise ValueError("target_greedy requires target_counts")
            remaining = quotas.copy()
            parts = {"train": [], "valid": [], "test": []}
            for spk, count in sorted(label_speaker_utts[label].items(), key=lambda kv: kv[1], reverse=True):
                choices = [split for split, quota in remaining.items() if quota > 0]
                split = max(
                    choices,
                    key=lambda name: (target_counts[name] - current_counts[name]) / max(1, target_counts[name]),
                )
                parts[split].append(spk)
                remaining[split] -= 1
                current_counts[split] += count

        label_assignment[str(label)] = {k: list(v) for k, v in parts.items()}
        for split, split_speakers in parts.items():
            for spk in split_speakers:
                if spk in assignment and assignment[spk] != split:
                    raise RuntimeError(f"Speaker {spk} assigned to multiple splits")
                assignment[spk] = split
                speaker_label[spk] = label
                speaker_size[spk] = label_speaker_utts[label][spk]

    if strategy == "target_greedy":
        assignment = improve_by_within_label_swaps(
            assignment=assignment,
            speaker_label=speaker_label,
            speaker_size=speaker_size,
            target_counts=target_counts,
        )
        # rebuild label assignment after swaps
        label_assignment = defaultdict(lambda: {"train": [], "valid": [], "test": []})
        for spk, split in assignment.items():
            label_assignment[str(speaker_label[spk])][split].append(spk)
        label_assignment = {k: {sp: sorted(vv) for sp, vv in v.items()} for k, v in label_assignment.items()}
    return assignment, label_assignment


def improve_by_within_label_swaps(assignment, speaker_label, speaker_size, target_counts):
    """Swap speakers only within the same label, preserving per-label split quotas."""

    by_label = defaultdict(list)
    current = Counter()
    for spk, split in assignment.items():
        by_label[speaker_label[spk]].append(spk)
        current[split] += speaker_size[spk]

    def objective(counts):
        return sum(abs(counts[k] - target_counts[k]) for k in ["train", "valid", "test"])

    for _ in range(10000):
        base = objective(current)
        if base == 0:
            break
        best = None
        best_delta = 0
        for speakers in by_label.values():
            for i, a in enumerate(speakers):
                split_a = assignment[a]
                for b in speakers[i + 1 :]:
                    split_b = assignment[b]
                    if split_a == split_b:
                        continue
                    size_a = speaker_size[a]
                    size_b = speaker_size[b]
                    candidate = current.copy()
                    candidate[split_a] += size_b - size_a
                    candidate[split_b] += size_a - size_b
                    delta = base - objective(candidate)
                    if delta > best_delta:
                        best_delta = delta
                        best = (a, b, candidate)
        if best is None:
            break
        a, b, current = best
        assignment[a], assignment[b] = assignment[b], assignment[a]
    return assignment


def summarize(y, speaker, assignment, id2label):
    split_utts = Counter()
    split_speakers = defaultdict(set)
    split_labels = defaultdict(set)
    split_label_utts = defaultdict(Counter)
    split_label_speakers = defaultdict(lambda: defaultdict(set))

    for yi, spk in zip(y, speaker):
        split = assignment[spk]
        label = id2label[str(int(yi))]
        split_utts[split] += 1
        split_speakers[split].add(spk)
        split_labels[split].add(label)
        split_label_utts[split][label] += 1
        split_label_speakers[split][label].add(spk)

    overlaps = {}
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        overlaps[f"{a}_vs_{b}"] = len(split_speakers[a] & split_speakers[b])

    labels = [id2label[str(i)] for i in range(len(id2label))]
    class_rows = []
    for label in labels:
        row = {"label": label}
        for split in ["train", "valid", "test"]:
            row[f"{split}_utts"] = int(split_label_utts[split][label])
            row[f"{split}_speakers"] = len(split_label_speakers[split][label])
        class_rows.append(row)

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
        "classes_absent_from_test": [r["label"] for r in class_rows if r["test_utts"] == 0],
        "class_rows": class_rows,
    }


def write_split_npz(out_dir: Path, X, y, speaker, source_split, assignment):
    out_dir.mkdir(parents=True, exist_ok=True)
    split_indices = {}
    for split in ["train", "valid", "test"]:
        idx = np.asarray([i for i, spk in enumerate(speaker) if assignment[spk] == split], dtype=np.int64)
        split_indices[split] = idx
        out_path = out_dir / f"{split}.npz"
        np.savez(
            out_path,
            X=X[idx],
            y=y[idx],
            speaker=speaker[idx],
            source_split=source_split[idx],
            source_index=idx,
        )
        print(f"wrote {split}: X={X[idx].shape} -> {out_path}", flush=True)
    return split_indices


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    id2label = json.loads((args.features_dir / "id2label.json").read_text(encoding="utf-8"))
    label2id = json.loads((args.features_dir / "label2id.json").read_text(encoding="utf-8"))

    X, y, speaker, source_split = read_features(args.features_dir)
    target_counts = None
    if args.strategy == "target_greedy":
        if args.target_train is None or args.target_valid is None or args.target_test is None:
            raise ValueError("target_greedy requires --target_train/--target_valid/--target_test")
        target_counts = {"train": args.target_train, "valid": args.target_valid, "test": args.target_test}

    assignment, label_assignment = assign_speakers(
        y=y,
        speaker=speaker,
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        rounding=args.rounding,
        strategy=args.strategy,
        target_counts=target_counts,
    )

    summary = summarize(y, speaker, assignment, id2label)
    if any(summary["speaker_overlap"].values()):
        raise RuntimeError(f"speaker leakage: {summary['speaker_overlap']}")

    print("summary:", json.dumps(summary["splits"], ensure_ascii=False), flush=True)
    print("speaker_overlap:", summary["speaker_overlap"], flush=True)
    print("classes_absent_from_test:", summary["classes_absent_from_test"], flush=True)

    write_split_npz(args.out_dir, X, y, speaker, source_split, assignment)

    (args.out_dir / "id2label.json").write_text(json.dumps(id2label, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "label2id.json").write_text(json.dumps(label2id, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "speaker_assignment.json").write_text(
        json.dumps(assignment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "label_speaker_assignment.json").write_text(
        json.dumps(label_assignment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    run_config = {
        "features_dir": str(args.features_dir),
        "out_dir": str(args.out_dir),
        "seed": args.seed,
        "ratios": {"train": 1.0 - args.valid_ratio - args.test_ratio, "valid": args.valid_ratio, "test": args.test_ratio},
        "rounding": args.rounding,
        "strategy": args.strategy,
        "target_counts": target_counts,
        "rules": [
            "speaker-disjoint",
            "1-speaker labels -> train only",
            "2-speaker labels -> train/test, no valid",
            ">=3-speaker labels -> seed shuffle, speaker-count ratio split with at least one valid/test speaker",
        ],
    }
    full_summary = {k: v for k, v in summary.items() if k != "class_rows"}
    full_summary.update(run_config)
    (args.out_dir / "split_stats.json").write_text(
        json.dumps(full_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.out_dir / "class_counts.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "label",
            "train_utts",
            "train_speakers",
            "valid_utts",
            "valid_speakers",
            "test_utts",
            "test_speakers",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["class_rows"])


if __name__ == "__main__":
    main()
