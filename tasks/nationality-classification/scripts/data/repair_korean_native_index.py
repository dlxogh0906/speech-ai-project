#!/usr/bin/env python3
"""Repair non-WAV audio entries and mark safe training eligibility."""

from __future__ import annotations

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import DATA_ROOT



import csv
import json
import os
import shutil
import sqlite3
import wave
from pathlib import Path


ROOT = Path(f"{DATA_ROOT}/aihub/001.자유대화(일반남여)/01.데이터")
INDEX_DIR = Path(f"{DATA_ROOT}/taeho/korean_native_index")
DB_PATH = INDEX_DIR / "korean_native_index.sqlite"
REPAIRED_DIR = INDEX_DIR / "repaired_audio"


def source_audio_index():
    result = {}
    for dirpath, _, filenames in os.walk(ROOT):
        for filename in filenames:
            suffix = Path(filename).suffix.lower()
            if suffix in {".pcm", ".wavp"}:
                result[Path(filename).stem] = Path(dirpath) / filename
    return result


def ensure_columns(conn):
    columns = {row[1] for row in conn.execute("PRAGMA table_info(utterances)")}
    if "resolved_audio_path" not in columns:
        conn.execute("ALTER TABLE utterances ADD COLUMN resolved_audio_path TEXT")
    if "eligible_for_sampling" not in columns:
        conn.execute(
            "ALTER TABLE utterances ADD COLUMN eligible_for_sampling INTEGER DEFAULT 0"
        )
    if "exclusion_reason" not in columns:
        conn.execute("ALTER TABLE utterances ADD COLUMN exclusion_reason TEXT DEFAULT ''")
    conn.commit()


def write_pcm_as_wav(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src:
        payload = src.read()
    with wave.open(str(target), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(payload)


def copy_wavp_as_wav(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    with wave.open(str(target), "rb") as wav:
        if (
            wav.getnchannels() != 1
            or wav.getsampwidth() != 2
            or wav.getframerate() != 16000
        ):
            raise ValueError(f"unexpected WAVP format: {source}")


def main():
    REPAIRED_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    ensure_columns(conn)

    conflict_speakers = {
        row[0]
        for row in conn.execute(
            """
            SELECT speaker_id
            FROM utterances
            WHERE speaker_id <> ''
            GROUP BY speaker_id
            HAVING
                COUNT(DISTINCT CASE WHEN gender <> '' THEN gender END) > 1
                OR COUNT(DISTINCT age) > 1
            """
        )
    }

    audio_index = source_audio_index()
    repairs = []
    for record_key, utterance_id, speaker_id, duration_sec in conn.execute(
        """
        SELECT record_key, utterance_id, speaker_id, duration_sec
        FROM utterances
        WHERE wav_path IS NULL
        ORDER BY record_key
        """
    ):
        source = audio_index.get(utterance_id)
        if source is None:
            repairs.append(
                {
                    "record_key": record_key,
                    "speaker_id": speaker_id,
                    "source_path": "",
                    "repaired_path": "",
                    "status": "source_not_found",
                    "expected_duration_sec": duration_sec,
                    "actual_duration_sec": "",
                }
            )
            continue
        target = REPAIRED_DIR / speaker_id / f"{utterance_id}.wav"
        if source.suffix.lower() == ".pcm":
            write_pcm_as_wav(source, target)
            source_type = "pcm_16k_s16le_mono"
        else:
            copy_wavp_as_wav(source, target)
            source_type = "wavp_riff_renamed"
        with wave.open(str(target), "rb") as wav:
            actual_duration = wav.getnframes() / float(wav.getframerate())
        conn.execute(
            "UPDATE utterances SET resolved_audio_path=? WHERE record_key=?",
            (str(target), record_key),
        )
        repairs.append(
            {
                "record_key": record_key,
                "speaker_id": speaker_id,
                "source_path": str(source),
                "repaired_path": str(target),
                "status": source_type,
                "expected_duration_sec": duration_sec,
                "actual_duration_sec": actual_duration,
            }
        )

    conn.execute(
        """
        UPDATE utterances
        SET
            resolved_audio_path = wav_path,
            eligible_for_sampling = 1,
            exclusion_reason = ''
        WHERE wav_path IS NOT NULL
        """
    )
    conn.execute(
        """
        UPDATE utterances
        SET
            eligible_for_sampling = 1,
            exclusion_reason = ''
        WHERE wav_path IS NULL AND resolved_audio_path IS NOT NULL
        """
    )
    placeholders = ",".join("?" for _ in conflict_speakers)
    conn.execute(
        f"""
        UPDATE utterances
        SET eligible_for_sampling=0, exclusion_reason='recorder_id_metadata_conflict'
        WHERE speaker_id IN ({placeholders})
        """,
        sorted(conflict_speakers),
    )
    conn.execute(
        """
        UPDATE utterances
        SET eligible_for_sampling=0, exclusion_reason='audio_unresolved'
        WHERE resolved_audio_path IS NULL
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_utterances_eligible
        ON utterances(eligible_for_sampling, speaker_id)
        """
    )
    conn.commit()

    repair_csv = INDEX_DIR / "audio_repairs.csv"
    with repair_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(repairs[0]))
        writer.writeheader()
        writer.writerows(repairs)

    scalar = lambda q: conn.execute(q).fetchone()[0]
    summary = {
        "repaired_audio_records": scalar(
            "SELECT COUNT(*) FROM utterances WHERE wav_path IS NULL AND resolved_audio_path IS NOT NULL"
        ),
        "unresolved_audio_records": scalar(
            "SELECT COUNT(*) FROM utterances WHERE resolved_audio_path IS NULL"
        ),
        "eligible_records": scalar(
            "SELECT COUNT(*) FROM utterances WHERE eligible_for_sampling=1"
        ),
        "eligible_speakers": scalar(
            """
            SELECT COUNT(DISTINCT speaker_id)
            FROM utterances
            WHERE eligible_for_sampling=1
            """
        ),
        "excluded_conflict_speakers": len(conflict_speakers),
        "excluded_conflict_records": scalar(
            """
            SELECT COUNT(*) FROM utterances
            WHERE exclusion_reason='recorder_id_metadata_conflict'
            """
        ),
        "exclusion_counts": dict(
            conn.execute(
                """
                SELECT exclusion_reason, COUNT(*)
                FROM utterances
                WHERE eligible_for_sampling=0
                GROUP BY exclusion_reason
                """
            ).fetchall()
        ),
        "repair_status_counts": {},
        "repaired_audio_dir": str(REPAIRED_DIR),
        "sampling_audio_column": "resolved_audio_path",
        "sampling_filter": "eligible_for_sampling = 1",
    }
    for row in repairs:
        status = row["status"]
        summary["repair_status_counts"][status] = (
            summary["repair_status_counts"].get(status, 0) + 1
        )
    (INDEX_DIR / "repair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
