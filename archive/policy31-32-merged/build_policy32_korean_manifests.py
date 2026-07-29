#!/usr/bin/env python3
"""Build native-Korean and combined policy32 CSV manifests."""

from __future__ import annotations
import os

import csv
import hashlib
import json
import random
import shutil
import sqlite3
from collections import Counter
from pathlib import Path


DB_PATH = Path(os.path.expandvars("$SPEECH_AI_DATA/taeho/korean_native_index/korean_native_index.sqlite"))
KOREAN_OUT = Path(os.path.expandvars("$SPEECH_AI_DATA/taeho/korean_native_manifest_seed42"))
FOREIGN_MANIFEST = Path(os.path.expandvars("$NATIONALITY_OUTPUTS/policy31_manifest_othergpu_local"))
POLICY32_OUT = Path(os.path.expandvars("$SPEECH_AI_DATA/taeho/policy32_othergpu_split_seed42"))
SEED = 42
MIN_SEC = 2.0
MAX_SEC = 16.0
MAX_PER_SPEAKER = 100
TARGETS = {"train": 160_000, "valid": 20_000, "test": 20_000}
FIELDS = [
    "audio_path",
    "json_path",
    "speaker_id",
    "label_name",
    "label_id",
    "duration_sec",
]


def stable_seed(text: str):
    digest = hashlib.sha256(f"{SEED}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def split_speakers(speakers):
    speakers = sorted(speakers)
    random.Random(SEED).shuffle(speakers)
    n = len(speakers)
    n_valid = round(n * 0.1)
    n_test = round(n * 0.1)
    n_train = n - n_valid - n_test
    return {
        "train": speakers[:n_train],
        "valid": speakers[n_train : n_train + n_valid],
        "test": speakers[n_train + n_valid :],
    }


def load_candidates(conn, speaker_id):
    rows = conn.execute(
        """
        SELECT
            resolved_audio_path,
            json_path,
            speaker_id,
            duration_sec,
            source_split,
            region,
            gender,
            age,
            environment
        FROM utterances
        WHERE
            eligible_for_sampling=1
            AND speaker_id=?
            AND duration_sec BETWEEN ? AND ?
        """,
        (speaker_id, MIN_SEC, MAX_SEC),
    ).fetchall()
    random.Random(stable_seed(speaker_id)).shuffle(rows)
    return rows[:MAX_PER_SPEAKER]


def sample_split(conn, speakers, target):
    candidates = {speaker: load_candidates(conn, speaker) for speaker in speakers}
    capacity = sum(len(rows) for rows in candidates.values())
    if capacity < target:
        raise RuntimeError(f"capacity {capacity:,} < target {target:,}")

    selected = []
    order = list(speakers)
    random.Random(stable_seed(f"round-robin:{target}")).shuffle(order)
    for level in range(MAX_PER_SPEAKER):
        for speaker in order:
            rows = candidates[speaker]
            if level < len(rows):
                selected.append(rows[level])
                if len(selected) == target:
                    return selected, capacity, candidates
    raise RuntimeError("round-robin exhausted unexpectedly")


def write_korean_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "audio_path": row[0],
                    "json_path": row[1],
                    "speaker_id": row[2],
                    "label_name": "한국어_모국어",
                    "label_id": 31,
                    "duration_sec": row[3],
                }
            )


def combine_csv(foreign_path, korean_path, output_path):
    counts = Counter()
    speakers = set()
    labels = set()
    with output_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for source, source_name in (
            (foreign_path, "foreign"),
            (korean_path, "korean"),
        ):
            with source.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    writer.writerow({field: row.get(field, "") for field in FIELDS})
                    counts[source_name] += 1
                    speakers.add(row["speaker_id"])
                    labels.add(int(row["label_id"]))
    return {
        "total": sum(counts.values()),
        "foreign": counts["foreign"],
        "korean": counts["korean"],
        "speakers": len(speakers),
        "classes_present": len(labels),
    }


def main():
    KOREAN_OUT.mkdir(parents=True, exist_ok=True)
    POLICY32_OUT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    eligible_speakers = [
        row[0]
        for row in conn.execute(
            """
            SELECT speaker_id
            FROM utterances
            WHERE
                eligible_for_sampling=1
                AND duration_sec BETWEEN ? AND ?
            GROUP BY speaker_id
            HAVING COUNT(*) > 0
            ORDER BY speaker_id
            """,
            (MIN_SEC, MAX_SEC),
        )
    ]
    assignments = split_speakers(eligible_speakers)

    selected_by_split = {}
    korean_stats = {}
    for split in ("train", "valid", "test"):
        selected, capacity, candidates = sample_split(
            conn, assignments[split], TARGETS[split]
        )
        selected_by_split[split] = selected
        per_speaker = Counter(row[2] for row in selected)
        write_korean_csv(KOREAN_OUT / f"{split}.csv", selected)
        korean_stats[split] = {
            "samples": len(selected),
            "speakers_assigned": len(assignments[split]),
            "speakers_selected": len(per_speaker),
            "candidate_capacity_at_cap100": capacity,
            "min_selected_per_speaker": min(per_speaker.values()),
            "max_selected_per_speaker": max(per_speaker.values()),
            "mean_selected_per_speaker": len(selected) / len(per_speaker),
            "source_split_counts": dict(Counter(row[4] for row in selected)),
            "region_counts": dict(Counter(row[5] for row in selected)),
            "gender_counts": dict(Counter(row[6] for row in selected)),
            "environment_counts": dict(Counter(row[8] for row in selected)),
        }

    speaker_sets = {split: set(assignments[split]) for split in assignments}
    speaker_overlap = {
        "train_vs_valid": len(speaker_sets["train"] & speaker_sets["valid"]),
        "train_vs_test": len(speaker_sets["train"] & speaker_sets["test"]),
        "valid_vs_test": len(speaker_sets["valid"] & speaker_sets["test"]),
    }
    korean_summary = {
        "database": str(DB_PATH),
        "seed": SEED,
        "duration_filter_sec": [MIN_SEC, MAX_SEC],
        "max_samples_per_speaker": MAX_PER_SPEAKER,
        "label": {"id": 31, "name": "한국어_모국어"},
        "eligible_speakers_after_duration_filter": len(eligible_speakers),
        "targets": TARGETS,
        "splits": korean_stats,
        "speaker_overlap": speaker_overlap,
    }
    (KOREAN_OUT / "split_stats.json").write_text(
        json.dumps(korean_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (KOREAN_OUT / "speaker_assignment.json").write_text(
        json.dumps(
            {
                speaker: split
                for split, speakers in assignments.items()
                for speaker in speakers
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    policy = json.loads((FOREIGN_MANIFEST / "label_map.json").read_text(encoding="utf-8"))
    policy["label_to_id"]["한국어_모국어"] = 31
    policy["id_to_label"]["31"] = "한국어_모국어"
    policy["num_labels"] = 32
    if isinstance(policy.get("label_counts"), dict):
        policy["label_counts"]["한국어_모국어"] = sum(TARGETS.values())
    policy["policy32_extension"] = {
        "added_label": "한국어_모국어",
        "label_id": 31,
        "korean_manifest": str(KOREAN_OUT),
        "foreign_manifest": str(FOREIGN_MANIFEST),
    }
    labels = [policy["id_to_label"][str(i)] for i in range(32)]
    (POLICY32_OUT / "label_map.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (POLICY32_OUT / "labels.txt").write_text(
        "\n".join(labels) + "\n", encoding="utf-8"
    )
    (POLICY32_OUT / "label2id.json").write_text(
        json.dumps(policy["label_to_id"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (POLICY32_OUT / "id2label.json").write_text(
        json.dumps(policy["id_to_label"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    combined_stats = {}
    for split in ("train", "valid", "test"):
        combined_stats[split] = combine_csv(
            FOREIGN_MANIFEST / f"{split}.csv",
            KOREAN_OUT / f"{split}.csv",
            POLICY32_OUT / f"{split}.csv",
        )

    combined_summary = {
        "foreign_manifest": str(FOREIGN_MANIFEST),
        "korean_manifest": str(KOREAN_OUT),
        "label_count": 32,
        "added_label": {"id": 31, "name": "한국어_모국어"},
        "splits": combined_stats,
        "speaker_overlap_korean": speaker_overlap,
        "rules": [
            "foreign othergpu split preserved unchanged",
            "native-Korean split is speaker-disjoint by normalized recorderId",
            "native-Korean duration filter is 2-16 seconds",
            "native-Korean maximum is 100 samples per speaker",
            "metadata-conflict recorder IDs are excluded",
        ],
    }
    (POLICY32_OUT / "manifest_stats.json").write_text(
        json.dumps(combined_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copy2(KOREAN_OUT / "speaker_assignment.json", POLICY32_OUT / "korean_speaker_assignment.json")
    conn.close()

    print(json.dumps(korean_summary, ensure_ascii=False, indent=2))
    print(json.dumps(combined_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
