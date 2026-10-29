"""Freeze and stage speaker-disjoint test requests before candidate selection."""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r:
        raise RuntimeError("Repository root not found")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import collections
import csv
import hashlib
import json
import time
import numpy as np
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=_pl.Path, required=True)
    parser.add_argument("--out", type=_pl.Path, required=True)
    parser.add_argument("--mode", choices=("three_sentences", "word"), default="three_sentences")
    args = parser.parse_args()
    rows = {name: read_rows(args.manifest_dir / (name + ".csv")) for name in ("train", "valid", "test")}
    speakers = {name: {r["speaker"] for r in values} for name, values in rows.items()}
    if speakers["test"] & (speakers["train"] | speakers["valid"]):
        raise ValueError("Test speakers overlap training or validation")
    grouped, missing = collections.defaultdict(list), []
    for index, row in enumerate(rows["test"]):
        eligible = ((row["domain"].startswith("web") and row["task"] != "RW") or row["label_en"] == "Korea"
                    if args.mode == "three_sentences" else row["task"] == "RW")
        if not eligible:
            continue
        if not _pl.Path(row["audio_path"]).is_file():
            missing.append(dict(index=index, **row)); continue
        grouped[(row["speaker"], row["label_en"])].append((index, row))
    jobs = []
    group_size = 3 if args.mode == "three_sentences" else 1
    for (speaker, country), sources in sorted(grouped.items()):
        for first in range(0, len(sources) - group_size + 1, group_size):
            group = sources[first:first + group_size]
            job_id = ("test_" if group_size == 3 else "word_test_") + "{:05d}".format(len(jobs))
            jobs.append({"job_id": job_id, "country": country, "speaker": speaker,
                         "source_indices": [i for i, _ in group], "source_paths": [r["audio_path"] for _, r in group],
                         "domain": group[0][1]["domain"], "task": "three_test_sentences" if group_size == 3 else "RW",
                         "audio_file": job_id + ".wav", "payload": {"rid": job_id, "parts_requested": 3}})
    args.out.mkdir(parents=True, exist_ok=True)
    audio = args.out / "audio"; audio.mkdir(exist_ok=True)
    records_path = args.out / "records.jsonl"
    lines = "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs)
    if records_path.exists() and records_path.read_text() != lines:
        raise ValueError("Frozen test request definitions changed")
    records_path.write_text(lines)
    meta = {"manifest_dir": str(args.manifest_dir), "manifest_sha256": {k: sha(args.manifest_dir / (k + ".csv")) for k in rows},
            "records_sha256": sha(records_path), "jobs": len(jobs), "speakers": len({r["speaker"] for r in jobs}),
            "foreign_jobs": sum(r["country"] != "Korea" for r in jobs),
            "foreign_speakers": len({r["speaker"] for r in jobs if r["country"] != "Korea"}),
            "per_country": dict(collections.Counter(r["country"] for r in jobs)), "missing_eligible_audio": missing,
            "speaker_overlap_with_train_or_validation": 0,
            "grouping": "All eligible test rows in manifest order; same-speaker nonoverlapping groups of {}; parts=3".format(group_size),
            "scope": ("Web sentence test and Korean test; p505 and RW excluded exactly as in three-sentence validation"
                      if group_size == 3 else "All test RW words; no outcome-based filtering"),
            "selection": "Freeze test requests before training; use validation only to select final candidate; open test scores once"}
    (args.out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    started = time.monotonic()
    for i, job in enumerate(jobs):
        output = audio / job["audio_file"]
        if output.exists():
            continue
        waves = []
        for path in job["source_paths"]:
            wave, sr = sf.read(path, dtype="float32")
            if sr != 16000 or wave.ndim != 1:
                raise ValueError("Expected mono16k test input: " + path)
            waves.append(wave)
        sf.write(output, np.concatenate(waves), 16000, subtype="PCM_16")
        if i % 200 == 0:
            print("[stage] {}/{} {:.1f}s".format(i, len(jobs), time.monotonic() - started), flush=True)
    print("[done] {} requests, {} test speakers".format(len(jobs), meta["speakers"]), flush=True)


if __name__ == "__main__":
    main()
