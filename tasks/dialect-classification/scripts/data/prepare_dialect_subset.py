"""Prepare an approximate 10% dialect subset with recording-disjoint splits.

Archives are processed sequentially into 16 kHz FLAC to limit peak disk use.
Labels use principal_residence. Repeated local speaker IDs are scoped to a
recording; this does not establish independence of people across recordings.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    if _root.parent == _root:
        raise RuntimeError("Repository root not found")
    _root = _root.parent
sys.path.insert(0, str(_root))

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sqlite3
import time
import zipfile
from collections import Counter

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from download_aihub_dialect import download, write_json

LABELS = ["Gangwon", "Gyeongsang", "Jeolla", "Jeju", "Chungcheong"]
LABELS_KO = ["강원도", "경상도", "전라도", "제주도", "충청도"]
FIELDS = ["row_id", "audio_path", "start_sec", "duration_sec", "label_id", "label_name", "speaker_key", "record_key", "topic", "original_split"]


def residence_label(value):
    value = str(value or "")
    aliases = [(0, ["강원"]), (1, ["경상", "경남", "경북", "부산", "대구", "울산"]),
               (2, ["전라", "전남", "전북", "광주"]), (3, ["제주"]),
               (4, ["충청", "충남", "충북", "대전", "세종"])]
    for label, names in aliases:
        if any(name in value for name in names):
            return label
    return None


def zip_name(info):
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def build_labels(archives, database):
    connection = sqlite3.connect(str(database))
    connection.execute("CREATE TABLE IF NOT EXISTS records (dataset INTEGER, record_id TEXT, original_split TEXT, topic TEXT, speakers TEXT, utterances TEXT, PRIMARY KEY(dataset,record_id))")
    connection.execute("CREATE TABLE IF NOT EXISTS imported (filekey INTEGER PRIMARY KEY)")
    summaries = []
    for marker in sorted(Path(archives).glob("*/*/download.json")):
        state = json.loads(marker.read_text())
        archive = marker.parent / state["archive"]
        filekey = state["filekey"]
        if "학습" not in archive.name or not archive.is_file():
            continue
        if connection.execute("SELECT 1 FROM imported WHERE filekey=?", (filekey,)).fetchone():
            continue
        original_split = "unknown"
        # Catalog filenames carry split information; the numeric suffix does not.
        catalog_path = Path("/data/speech-ai/tmp/aihub-dialect-catalog") / ("%s-files.json" % state["datasetkey"])
        if catalog_path.exists():
            files = json.loads(catalog_path.read_text())["list"]
            path = next((f["fileStreCours"] for f in files if int(f["fileSn"]) == filekey), "")
            original_split = "Validation" if "Validation" in path else "Training"
        count = 0
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                if not info.filename.lower().endswith(".json"):
                    continue
                record = json.loads(bundle.read(info).decode("utf-8-sig"))
                if not isinstance(record, dict) or not isinstance(record.get("utterance"), list):
                    raise RuntimeError("Unrecognized label JSON in archive %s" % filekey)
                rid = str(record.get("id") or Path(zip_name(info)).stem)
                speakers = {}
                for speaker in record.get("speaker", []):
                    sid = str(speaker.get("id", ""))
                    speakers[sid] = {"label_id": residence_label(speaker.get("principal_residence")), "age": speaker.get("age"), "sex": speaker.get("sex")}
                utterances = []
                for number, utterance in enumerate(record["utterance"]):
                    start, end = float(utterance["start"]), float(utterance["end"])
                    if not (math.isfinite(start) and math.isfinite(end) and end > start >= 0):
                        continue
                    utterances.append([number, str(utterance.get("speaker_id", "")), start, end])
                topic = str((record.get("metadata") or {}).get("topic", ""))
                connection.execute("INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?)", (state["datasetkey"], rid, original_split, topic, json.dumps(speakers, ensure_ascii=False), json.dumps(utterances)))
                count += 1
        connection.execute("INSERT INTO imported VALUES (?)", (filekey,))
        connection.commit()
        summary = {"dataset": state["datasetkey"], "filekey": filekey, "json_records": count, "original_split": original_split}
        summaries.append(summary)
        print("[labels] " + json.dumps(summary, ensure_ascii=False), flush=True)
    write_json(database.with_name(database.name + ".summary.json"), {"new_imports": summaries, "records_per_dataset": list(connection.execute("SELECT dataset,COUNT(*) FROM records GROUP BY dataset")), "split_rule": "recording-disjoint; true person IDs are unavailable"})
    return connection


def split_for(record_key, seed):
    value = int(hashlib.sha256((str(seed) + ":" + record_key).encode()).hexdigest()[:8], 16) % 1000
    return "train" if value < 800 else ("valid" if value < 900 else "test")


def windows(utterances, speakers, audio_duration, min_sec=3, max_sec=12):
    result = []
    utterances = sorted(utterances, key=lambda u: (u[2], u[3]))
    active = []
    for index, (number, sid, start, end) in enumerate(utterances):
        active = [u for u in active if u[3] > start + 0.1]
        speaker = speakers.get(sid)
        # Exclude cross talk, using all annotated speakers, including unlabeled ones.
        overlap = any(other_sid != sid and other_start < end - 0.1
                      for _, other_sid, other_start, _ in active)
        cursor = index + 1
        while not overlap and cursor < len(utterances) and utterances[cursor][2] < end - 0.1:
            other = utterances[cursor]
            overlap = other[1] != sid and other[3] > start + 0.1
            cursor += 1
        active.append(utterances[index])
        if overlap or speaker is None or speaker["label_id"] is None:
            continue
        end = min(end, audio_duration)
        if end - start < min_sec:
            continue
        count = max(1, int(math.ceil((end - start) / max_sec)))
        length = (end - start) / count
        if length < min_sec:
            continue
        for segment in range(count):
            result.append(("%s.%s" % (number, segment), sid, start + segment * length, length, speaker["label_id"]))
    return result


def process_archive(dataset, file_info, archives, audio_root, connection, seed, reserve_gib):
    filekey = int(file_info["filekey"])
    folder = Path(archives) / str(dataset) / str(filekey)
    manifest = folder / "rows.csv"
    marker = folder / "processed.json"
    if marker.exists() and manifest.exists():
        print("[processed skip] dataset=%s file=%s" % (dataset, filekey), flush=True)
        return manifest
    archive = download(dataset, file_info, archives, reserve_gib)
    target = Path(audio_root) / str(dataset)
    target.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    started = time.time()
    tmp_manifest = manifest.with_suffix(".csv.tmp")
    with zipfile.ZipFile(archive) as bundle, tmp_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for info in bundle.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".wav"):
                continue
            rid = Path(zip_name(info)).stem
            stats["source_recordings"] += 1
            label = connection.execute("SELECT original_split,topic,speakers,utterances FROM records WHERE dataset=? AND record_id=?", (dataset, rid)).fetchone()
            if label is None:
                stats["missing_json_label"] += 1
                continue
            original_split, topic, speakers, utterances = label
            speakers, utterances = json.loads(speakers), json.loads(utterances)
            if not any(s["label_id"] is not None for s in speakers.values()):
                stats["no_supported_residence"] += 1
                continue
            if shutil.disk_usage(str(target)).free < int(reserve_gib * 2**30):
                raise RuntimeError("Disk reserve reached during conversion")
            if info.file_size > 2 * 2**30:
                raise RuntimeError("Unexpectedly large single recording: " + rid)
            # Reading the full zip member checks its CRC; originals are not unpacked to disk.
            raw = bundle.read(info)
            wave, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            del raw
            wave = wave.mean(axis=1)
            if rate != 16000:
                divisor = math.gcd(int(rate), 16000)
                wave = resample_poly(wave, 16000 // divisor, int(rate) // divisor).astype(np.float32)
            duration = len(wave) / 16000.0
            clips = windows(utterances, speakers, duration)
            if not clips:
                stats["no_usable_windows"] += 1
                continue
            destination = target / (rid + ".flac")
            temporary = destination.with_name(destination.stem + ".tmp.flac")
            sf.write(str(temporary), wave, 16000, format="FLAC", subtype="PCM_16")
            del wave
            saved = sf.info(str(temporary))
            if saved.samplerate != 16000 or saved.channels != 1 or abs(saved.duration - duration) > 1 / 16000:
                raise RuntimeError("Converted audio metadata mismatch: " + rid)
            os.replace(str(temporary), str(destination))
            key = "%s:%s" % (dataset, rid)
            for segment, sid, start, length, label_id in clips:
                writer.writerow({"row_id": key + ":" + sid + ":" + segment, "audio_path": str(destination), "start_sec": "%.6f" % start, "duration_sec": "%.6f" % length, "label_id": label_id, "label_name": LABELS[label_id], "speaker_key": key + ":" + sid, "record_key": key, "topic": topic, "original_split": original_split})
                stats["windows"] += 1
                stats["speech_seconds"] += length
            stats["converted_recordings"] += 1
            print("[convert] dataset=%s file=%s recordings=%s windows=%s" % (dataset, filekey, stats["converted_recordings"], stats["windows"]), flush=True)
    if not stats["converted_recordings"]:
        raise RuntimeError("No labeled audio matched archive %s; inspect filename/label alignment" % filekey)
    os.replace(str(tmp_manifest), str(manifest))
    write_json(marker, {"dataset": dataset, "filekey": filekey, "statistics": dict(stats), "archive": archive.name, "elapsed_seconds": round(time.time() - started, 1)})
    # Only the newly downloaded source archive is discarded after conversion and CRC checks.
    archive.unlink()
    return manifest


def assemble(manifests, destination, seed):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    handles = {split: (destination / (split + ".csv")).open("w", newline="") for split in ["train", "valid", "test"]}
    writers = {split: csv.DictWriter(handle, fieldnames=FIELDS) for split, handle in handles.items()}
    for writer in writers.values():
        writer.writeheader()
    seen = set()
    records = {split: set() for split in handles}
    counts = {split: Counter() for split in handles}
    durations = {split: Counter() for split in handles}
    try:
        for manifest in manifests:
            with Path(manifest).open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["row_id"] in seen:
                        continue
                    seen.add(row["row_id"])
                    split = split_for(row["record_key"], seed)
                    writers[split].writerow(row)
                    counts[split][row["label_name"]] += 1
                    durations[split][row["label_name"]] += float(row["duration_sec"])
                    records[split].add(row["record_key"])
    finally:
        for handle in handles.values():
            handle.close()
    if any(records[a] & records[b] for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]):
        raise RuntimeError("Recording leakage across splits")
    report = {"labels": LABELS, "labels_ko": LABELS_KO, "counts": {split: dict(count) for split, count in counts.items()}, "speech_hours": {split: {key: round(seconds / 3600, 3) for key, seconds in duration.items()} for split, duration in durations.items()}, "recordings": {split: len(value) for split, value in records.items()}, "seed": seed, "split_rule": "SHA256(recording_key), 80/10/10", "label_source": "speaker.principal_residence", "limitations": ["Speaker IDs are local to recordings; disjoint actual people cannot be established.", "Jeolla Training labels contain transcripts without timestamped JSON; use JSON-labeled Validation recordings and split those by recording.", "Subset archive sizes approximate 10%; downloaded bytes, recording count, and speech hours are different quantities."]}
    write_json(destination / "summary.json", report)
    print("[manifests] " + json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--root", default="/data/speech-ai/corpus/dialect_10pct_20261010")
    parser.add_argument("--out", default="/data/speech-ai/out/dialect/20261010")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reserve-gib", type=float, default=40)
    parser.add_argument("--labels-only", action="store_true")
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archives = root / "archives"
    connection = build_labels(archives, out / "labels.sqlite")
    if args.labels_only:
        return
    plan = json.loads(Path(args.plan).read_text())
    manifests = []
    for dataset in plan["datasets"]:
        for file_info in dataset["selected_files"]:
            if "라벨링" in file_info["path"]:
                continue
            manifest = process_archive(dataset["datasetkey"], file_info, archives, root / "audio", connection, args.seed, args.reserve_gib)
            manifests.append(manifest)
    assemble(manifests, out / "manifests", args.seed)


if __name__ == "__main__":
    main()
