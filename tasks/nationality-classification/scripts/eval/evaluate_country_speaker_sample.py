"""Paired, speaker-balanced validation through the deployed nationality API.

Freezes a prediction-independent sample, restores archived request audio, then
evaluates one recording versus three concatenated recordings of each speaker.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

import numpy as np
import soundfile as sf

from eval_validation_npu_service import predict, session


COUNTRIES = ("China", "Uzbekistan")
MODES = ("single", "three")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def prepare(args):
    manifest = args.manifests / "valid.csv"
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    old_metadata = json.loads((args.archive / "metadata.json").read_text())
    assert manifest_sha == old_metadata["manifest_sha256"], "Archive manifest mismatch"
    splits = {split: read_csv(args.manifests / (split + ".csv")) for split in ("train", "valid", "test")}
    speakers = {split: {r["speaker"] for r in rows} for split, rows in splits.items()}
    assert not speakers["train"] & speakers["valid"], "Train/validation speaker overlap"
    assert not speakers["test"] & speakers["valid"], "Test/validation speaker overlap"
    available, cohort = {}, []
    for country in COUNTRIES:
        groups = defaultdict(list)
        for index, row in enumerate(splits["valid"]):
            if row["label_en"] == country and row["task"] != "RW" and 3 <= float(row["duration_sec"]) <= 20:
                groups[row["speaker"]].append({"index": index, **row})
        eligible = [speaker for speaker, rows in groups.items() if len(rows) >= 3]
        chosen = sorted(eligible, key=lambda s: digest("speaker:42:" + s))[:50]
        available[country] = {
            **{split + "_speakers": len({r["speaker"] for r in rows if r["label_en"] == country})
               for split, rows in splits.items()},
            "valid_recordings": sum(r["label_en"] == country for r in splits["valid"]),
            "eligible_speakers": len(eligible), "selected_speakers": len(chosen),
        }
        for number, speaker in enumerate(chosen, 1):
            rows = sorted(groups[speaker], key=lambda r: digest("row:42:" + r["audio_path"]))[:3]
            cohort.append({"id": country.lower() + "_{:02d}".format(number),
                           "country": country, "speaker": speaker, "rows": rows})
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(), "manifest": str(manifest),
        "manifest_sha256": manifest_sha, "seed": 42, "requested_speakers_per_country": 50,
        "available": available, "train_valid_speaker_overlap": 0, "valid_test_speaker_overlap": 0,
        "selection": "SHA256 ordering by speaker:42:<speaker>, then row:42:<audio_path>; no prediction filtering",
        "eligibility": "At least three validation recordings with task != RW and duration 3–20 seconds",
        "single": "First of three selected recordings; /predict_long with parts=1",
        "three": "Same speaker's three recordings concatenated without gaps; /predict_long with parts=3",
        "scoring": "One outcome per speaker and mode. Rejections/errors count as incorrect. No replacement after prediction.",
        "caveats": ["Three recordings are concatenated; this is not a naturally continuous recording.",
                    "The service splits by duration, not sentence boundaries; returned parts can differ from the requested count.",
                    "Uzbekistan validation contains only 10 distinct speakers; 50 unseen validation speakers are unavailable.",
                    "Speaker identities are those in the existing split manifest; no new cross-corpus identity audit was performed."],
    }
    # The sample is frozen before looking up old API records or their outcomes.
    dump(args.out / "protocol.json", protocol)
    dump(args.out / "selection.json", cohort)
    old_records = {}
    with (args.archive / "responses.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            old_records[record["job_id"]] = record
    for person in cohort:
        for row in person["rows"]:
            old = old_records["single_{:05d}".format(row["index"])]
            assert old["source_indices"] == [row["index"]]
            assert old["source_paths"] == [row["audio_path"]]
            assert old["country"] == person["country"] and old["speaker"] == person["speaker"]
            rid = old["payload"]["rid"]
            assert re.fullmatch(r"20261009-\d{6}-[0-9a-f]{6}", rid), rid
            row["capture_rid"] = rid
            row["archived_request_bytes"] = old["bytes"]
            row["restored_path"] = str(args.out / "audio" / (rid + ".wav"))
    dump(args.out / "selection.json", cohort)
    names = sorted({r["capture_rid"] + ".wav" for p in cohort for r in p["rows"]})
    archive = args.out / "restored_audio.tar"
    with archive.open("wb") as stream:
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "npu",
                        "tar", "-C", "/root/nat_serve/captures/20261009", "-cf", "-", "-T", "-"],
                       input=("\n".join(names) + "\n").encode(), stdout=stream, check=True, timeout=180)
    audio_dir = args.out / "audio"
    audio_dir.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert {m.name for m in members} == set(names)
        for member in members:
            assert member.isfile() and Path(member.name).name == member.name
            (audio_dir / member.name).write_bytes(tar.extractfile(member).read())
    integrity = []
    for person in cohort:
        for row in person["rows"]:
            path = Path(row["restored_path"])
            info = sf.info(str(path))
            assert path.stat().st_size == row["archived_request_bytes"], "Archived request size mismatch"
            assert abs(info.duration - float(row["duration_sec"])) < .15, (path, info.duration)
            integrity.append({"index": row["index"], "capture_rid": row["capture_rid"],
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "bytes": path.stat().st_size, "duration_sec": info.duration,
                              "samplerate": info.samplerate, "channels": info.channels})
    dump(args.out / "audio_integrity.json", integrity)
    return protocol, cohort


def summarize(records):
    summary = {}
    for country in COUNTRIES:
        summary[country] = {}
        for mode in MODES:
            chosen = [r for r in records if r["country"] == country and r["mode"] == mode]
            assert len(chosen) == len({r["speaker"] for r in chosen})
            predictions = [r for r in chosen if r["status"] == 200 and r["payload"].get("top5")]
            counts = Counter(r["payload"]["top5"][0]["label"] for r in predictions)
            rejected = len(chosen) - len(predictions)
            if rejected:
                counts["Rejected/error"] = rejected
            h1 = counts[country]
            h5 = sum(any(p["label"] == country for p in r["payload"]["top5"]) for r in predictions)
            summary[country][mode] = {
                "n_speakers": len(chosen), "correct_top1": h1, "correct_top5": h5,
                "accuracy_top1": h1 / len(chosen), "accuracy_top5": h5 / len(chosen),
                "distribution": dict(counts.most_common()), "rejections_or_errors": rejected,
                "actual_parts": dict(Counter(str(r["payload"].get("n")) for r in predictions)),
                "mean_recording_duration_sec": float(np.mean([r["source_duration_sec"] for r in chosen])),
                "low_confidence": sum(bool(r["payload"].get("low_confidence")) for r in predictions),
            }
            assert sum(counts.values()) == len(chosen)
    return summary


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / "responses.jsonl").exists(), "Use a fresh output directory to avoid mixing runs"
    protocol, cohort = prepare(args)
    print(json.dumps({"sample_frozen": protocol["available"], "restored_files": 3 * len(cohort)}), flush=True)
    health = session().get(args.url.rstrip("/") + "/health", timeout=20)
    health.raise_for_status()
    before = health.json()
    assert before["ok"] and before["backend"] == "rbln" and before["classes"] == 41
    dump(args.out / "health_before.json", before)
    jobs = []
    for person in cohort:
        for mode, count in (("single", 1), ("three", 3)):
            rows = person["rows"][:count]
            jobs.append({"job_id": person["id"] + "_" + mode, "person_id": person["id"],
                         "mode": mode, "country": person["country"], "speaker": person["speaker"],
                         "source_indices": [r["index"] for r in rows],
                         "source_paths": [r["restored_path"] for r in rows],
                         "source_duration_sec": sum(float(r["duration_sec"]) for r in rows)})
    records = []
    with (args.out / "responses.jsonl").open("w", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(predict, job, args.url, 1 if job["mode"] == "single" else 3) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if len(records) % 10 == 0:
                print("completed {}/{}; latest HTTP {}".format(len(records), len(jobs), record["status"]), flush=True)
    health = session().get(args.url.rstrip("/") + "/health", timeout=20)
    health.raise_for_status()
    after = health.json()
    dump(args.out / "health_after.json", after)
    for key in ("model", "backend", "classes", "silence_norm", "gate", "gate_tau", "gate_input", "korea_thr", "lowconf", "min_voiced_sec"):
        assert before.get(key) == after.get(key), "Service configuration changed: " + key
    assert len(records) == 2 * len(cohort)
    summary = summarize(records)
    report = {"completed_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
              "model_health": before, "metrics": summary}
    dump(args.out / "report.json", report)
    with (args.out / "per_speaker.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["person_id", "true_country", "mode", "prediction", "correct_top1", "correct_top5", "confidence", "status", "parts", "rid"])
        for r in sorted(records, key=lambda r: r["job_id"]):
            top5 = r["payload"].get("top5", []) if r["status"] == 200 else []
            label = top5[0]["label"] if top5 else "Rejected/error"
            writer.writerow([r["person_id"], r["country"], r["mode"], label, int(label == r["country"]),
                             int(any(p["label"] == r["country"] for p in top5)), top5[0]["prob"] if top5 else "",
                             r["status"], r["payload"].get("n", ""), r["payload"].get("rid", "")])
    with (args.out / "predicted_distribution.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_country", "mode", "predicted_country", "people", "percentage"])
        for country in COUNTRIES:
            for mode in MODES:
                m = summary[country][mode]
                for label, count in m["distribution"].items():
                    writer.writerow([country, mode, label, count, 100 * count / m["n_speakers"]])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, default=Path("/data/speech-ai/out/nationality/webmic_kd/lab131full_zeroth"))
    parser.add_argument("--archive", type=Path, default=Path("experiments/nationality_validation_20261009/npu_single"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", required=True, help="국적 API 주소 (예: http://<NPU-tailscale-IP>:8905)")
    run(parser.parse_args())
