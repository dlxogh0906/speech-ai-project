#!/usr/bin/env python3
"""Index the AIHub native-Korean conversation corpus by recorderId."""

from __future__ import annotations

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT



import argparse
import csv
import json
import os
import sqlite3
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


DEFAULT_ROOT = Path(f"{DATA_ROOT}/aihub/001.자유대화(일반남여)/01.데이터")
DEFAULT_OUT = Path(f"{DATA_ROOT}/taeho/korean_native_index")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=100000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def classify_path(path: str):
    parts = Path(path).parts
    split = next((x for x in parts if x in ("1.Training", "2.Validation")), "")
    batch = next(
        (
            x
            for x in parts
            if x.startswith(("1.라벨링데이터_", "2.원천데이터_"))
        ),
        "",
    )
    batch_id = batch.rsplit("_", 2)[-2] if batch else ""
    speaker_dir = Path(path).parent.name
    return split, batch_id, speaker_dir


def make_key(split: str, batch_id: str, speaker_dir: str, stem: str):
    return "\t".join((split, batch_id, speaker_dir, stem))


def iter_files(root: Path, suffix: str):
    suffix = suffix.lower()
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(suffix):
                yield os.path.join(dirpath, filename)


def parse_json_record(path: str):
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        utterance = data.get("발화정보") or {}
        conversation = data.get("대화정보") or {}
        recorder = data.get("녹음자정보") or {}

        split, batch_id, speaker_dir = classify_path(path)
        wav_name = str(utterance.get("fileNm") or (Path(path).stem + ".wav"))
        stem = Path(wav_name).stem
        recorder_raw = str(recorder.get("recorderId") or "").strip()
        speaker_id = f"KOR_{recorder_raw}" if recorder_raw else ""

        duration = utterance.get("recrdTime")
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = None
        age = recorder.get("age")
        try:
            age = int(age)
        except (TypeError, ValueError):
            age = None

        return (
            make_key(split, batch_id, speaker_dir, stem),
            split,
            batch_id,
            speaker_dir,
            stem,
            wav_name,
            path,
            recorder_raw,
            speaker_id,
            str(recorder.get("gender") or ""),
            age,
            duration,
            str(utterance.get("recrdQuality") or ""),
            str(utterance.get("recrdDt") or ""),
            str(conversation.get("recrdEnvrn") or ""),
            str(conversation.get("colctUnitCode") or ""),
            str(conversation.get("cityCode") or ""),
            str(conversation.get("recrdUnit") or ""),
            str(conversation.get("convrsThema") or ""),
            len(str(utterance.get("stt") or "")),
            "",
        )
    except Exception as exc:
        split, batch_id, speaker_dir = classify_path(path)
        stem = Path(path).stem
        return (
            make_key(split, batch_id, speaker_dir, stem),
            split,
            batch_id,
            speaker_dir,
            stem,
            stem + ".wav",
            path,
            "",
            "",
            "",
            None,
            None,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            0,
            f"{type(exc).__name__}: {exc}",
        )


def open_db(path: Path):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        """
        CREATE TABLE utterances (
            record_key TEXT PRIMARY KEY,
            source_split TEXT,
            batch_id TEXT,
            speaker_dir TEXT,
            utterance_id TEXT,
            wav_name TEXT,
            wav_path TEXT,
            json_path TEXT,
            recorder_id TEXT,
            speaker_id TEXT,
            gender TEXT,
            age INTEGER,
            duration_sec REAL,
            record_quality TEXT,
            record_datetime TEXT,
            environment TEXT,
            collection_unit TEXT,
            region TEXT,
            record_device TEXT,
            conversation_theme TEXT,
            transcript_chars INTEGER,
            json_error TEXT
        )
        """
    )
    return conn


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.out_dir / "korean_native_index.sqlite"
    if db_path.exists():
        if not args.overwrite:
            raise SystemExit(f"{db_path} exists; pass --overwrite")
        db_path.unlink()
    for suffix in ("-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()

    conn = open_db(db_path)
    started = time.time()

    print("[1/4] indexing WAV files", flush=True)
    wav_rows = []
    wav_count = 0
    wav_duplicate_keys = 0
    for path in iter_files(args.root, ".wav"):
        split, batch_id, speaker_dir = classify_path(path)
        stem = Path(path).stem
        wav_rows.append(
            (
                make_key(split, batch_id, speaker_dir, stem),
                split,
                batch_id,
                speaker_dir,
                stem,
                Path(path).name,
                path,
            )
        )
        if len(wav_rows) >= args.batch_size:
            before = conn.total_changes
            conn.executemany(
                """
                INSERT OR IGNORE INTO utterances
                    (record_key, source_split, batch_id, speaker_dir,
                     utterance_id, wav_name, wav_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                wav_rows,
            )
            wav_duplicate_keys += len(wav_rows) - (conn.total_changes - before)
            wav_count += len(wav_rows)
            wav_rows.clear()
            if wav_count % args.log_every < args.batch_size:
                print(f"  WAV {wav_count:,}", flush=True)
    if wav_rows:
        before = conn.total_changes
        conn.executemany(
            """
            INSERT OR IGNORE INTO utterances
                (record_key, source_split, batch_id, speaker_dir,
                 utterance_id, wav_name, wav_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            wav_rows,
        )
        wav_duplicate_keys += len(wav_rows) - (conn.total_changes - before)
        wav_count += len(wav_rows)
    conn.commit()
    print(f"  WAV done: {wav_count:,}, duplicate keys={wav_duplicate_keys:,}", flush=True)

    print("[2/4] parsing JSON and merging by record key", flush=True)
    update_sql = """
        INSERT INTO utterances (
            record_key, source_split, batch_id, speaker_dir, utterance_id,
            wav_name, json_path, recorder_id, speaker_id, gender, age,
            duration_sec, record_quality, record_datetime, environment,
            collection_unit, region, record_device, conversation_theme,
            transcript_chars, json_error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(record_key) DO UPDATE SET
            json_path=excluded.json_path,
            recorder_id=excluded.recorder_id,
            speaker_id=excluded.speaker_id,
            gender=excluded.gender,
            age=excluded.age,
            duration_sec=excluded.duration_sec,
            record_quality=excluded.record_quality,
            record_datetime=excluded.record_datetime,
            environment=excluded.environment,
            collection_unit=excluded.collection_unit,
            region=excluded.region,
            record_device=excluded.record_device,
            conversation_theme=excluded.conversation_theme,
            transcript_chars=excluded.transcript_chars,
            json_error=excluded.json_error
    """
    json_count = 0
    batch = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(
            parse_json_record,
            iter_files(args.root, ".json"),
            chunksize=200,
        ):
            batch.append(row)
            if len(batch) >= args.batch_size:
                conn.executemany(update_sql, batch)
                json_count += len(batch)
                batch.clear()
                if json_count % args.log_every < args.batch_size:
                    conn.commit()
                    elapsed = time.time() - started
                    print(
                        f"  JSON {json_count:,} ({json_count / max(elapsed, 1):.0f}/s)",
                        flush=True,
                    )
    if batch:
        conn.executemany(update_sql, batch)
        json_count += len(batch)
    conn.commit()
    print(f"  JSON done: {json_count:,}", flush=True)

    print("[3/4] creating indexes and summaries", flush=True)
    conn.executescript(
        """
        CREATE INDEX idx_utterances_speaker ON utterances(speaker_id);
        CREATE INDEX idx_utterances_split ON utterances(source_split);
        CREATE INDEX idx_utterances_match ON utterances(wav_path, json_path);
        """
    )
    conn.commit()

    speaker_csv = args.out_dir / "speaker_counts.csv"
    with speaker_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "speaker_id",
                "recorder_id",
                "utterances_total",
                "utterances_with_wav",
                "utterances_with_json",
                "training_utterances",
                "validation_utterances",
                "gender",
                "min_age",
                "max_age",
                "duration_hours",
            ]
        )
        writer.writerows(
            conn.execute(
                """
                SELECT
                    speaker_id,
                    recorder_id,
                    COUNT(*),
                    SUM(wav_path IS NOT NULL),
                    SUM(json_path IS NOT NULL),
                    SUM(source_split = '1.Training'),
                    SUM(source_split = '2.Validation'),
                    MIN(NULLIF(gender, '')),
                    MIN(age),
                    MAX(age),
                    SUM(COALESCE(duration_sec, 0)) / 3600.0
                FROM utterances
                WHERE speaker_id <> ''
                GROUP BY speaker_id, recorder_id
                ORDER BY COUNT(*) DESC
                """
            )
        )

    mismatch_csv = args.out_dir / "file_mismatches.csv"
    with mismatch_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "record_key",
                "wav_path",
                "json_path",
                "speaker_id",
                "json_error",
                "status",
            ]
        )
        for row in conn.execute(
            """
            SELECT
                record_key, wav_path, json_path, speaker_id, json_error,
                CASE
                    WHEN wav_path IS NULL THEN 'json_only'
                    WHEN json_path IS NULL THEN 'wav_only'
                    WHEN json_error <> '' THEN 'json_error'
                    ELSE 'matched'
                END
            FROM utterances
            WHERE wav_path IS NULL OR json_path IS NULL OR json_error <> ''
            ORDER BY 6, record_key
            """
        ):
            writer.writerow(row)

    scalar = lambda query: conn.execute(query).fetchone()[0]
    summary = {
        "dataset_root": str(args.root),
        "database": str(db_path),
        "wav_files_scanned": wav_count,
        "json_files_scanned": json_count,
        "wav_duplicate_record_keys": wav_duplicate_keys,
        "indexed_records": scalar("SELECT COUNT(*) FROM utterances"),
        "matched_wav_json": scalar(
            "SELECT COUNT(*) FROM utterances WHERE wav_path IS NOT NULL AND json_path IS NOT NULL"
        ),
        "wav_only": scalar(
            "SELECT COUNT(*) FROM utterances WHERE wav_path IS NOT NULL AND json_path IS NULL"
        ),
        "json_only": scalar(
            "SELECT COUNT(*) FROM utterances WHERE wav_path IS NULL AND json_path IS NOT NULL"
        ),
        "json_errors": scalar(
            "SELECT COUNT(*) FROM utterances WHERE json_error <> ''"
        ),
        "records_without_recorder_id": scalar(
            "SELECT COUNT(*) FROM utterances WHERE json_path IS NOT NULL AND speaker_id = ''"
        ),
        "unique_speakers": scalar(
            "SELECT COUNT(DISTINCT speaker_id) FROM utterances WHERE speaker_id <> ''"
        ),
        "training_unique_speakers": scalar(
            """
            SELECT COUNT(DISTINCT speaker_id) FROM utterances
            WHERE speaker_id <> '' AND source_split = '1.Training'
            """
        ),
        "validation_unique_speakers": scalar(
            """
            SELECT COUNT(DISTINCT speaker_id) FROM utterances
            WHERE speaker_id <> '' AND source_split = '2.Validation'
            """
        ),
        "training_validation_speaker_overlap": scalar(
            """
            SELECT COUNT(*) FROM (
                SELECT speaker_id
                FROM utterances
                WHERE speaker_id <> ''
                GROUP BY speaker_id
                HAVING
                    SUM(source_split = '1.Training') > 0
                    AND SUM(source_split = '2.Validation') > 0
            )
            """
        ),
        "duration_hours_json": scalar(
            "SELECT SUM(COALESCE(duration_sec, 0)) / 3600.0 FROM utterances"
        ),
        "source_split_records": dict(
            conn.execute(
                "SELECT source_split, COUNT(*) FROM utterances GROUP BY source_split"
            ).fetchall()
        ),
        "gender_records": dict(
            conn.execute(
                """
                SELECT gender, COUNT(*) FROM utterances
                WHERE gender <> '' GROUP BY gender
                """
            ).fetchall()
        ),
        "region_records": dict(
            conn.execute(
                """
                SELECT region, COUNT(*) FROM utterances
                WHERE region <> '' GROUP BY region
                ORDER BY COUNT(*) DESC
                """
            ).fetchall()
        ),
        "elapsed_sec": time.time() - started,
        "speaker_id_rule": "speaker_id = 'KOR_' + str(녹음자정보.recorderId).strip()",
    }
    (args.out_dir / "index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")
    conn.close()

    print("[4/4] done", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
