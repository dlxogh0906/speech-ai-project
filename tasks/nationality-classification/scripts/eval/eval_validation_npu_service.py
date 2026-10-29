# -*- coding: utf-8 -*-
"""Evaluate validation audio through the running NPU /predict_long service."""

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import requests
import soundfile as sf

from summarize_validation_per_country import COUNTRY_KO


LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.trust_env = False
        LOCAL.session.headers["User-Agent"] = "nationality-validation-eval/20261009"
    return LOCAL.session


def read_csv(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def jobs_for(rows, mode):
    missing = [{"index": index, **row} for index, row in enumerate(rows) if not Path(row["audio_path"]).is_file()]
    existing = [(index, row) for index, row in enumerate(rows) if Path(row["audio_path"]).is_file()]
    if mode == "single":
        return [{"job_id": "single_{:05d}".format(index), "source_indices": [index],
                 "source_paths": [row["audio_path"]], "country": row["label_en"],
                 "speaker": row["speaker"], "domain": row["domain"], "task": row["task"],
                 "source_duration_sec": float(row["duration_sec"])} for index, row in existing], missing
    grouped = collections.defaultdict(list)
    for index, row in existing:
        if (row["domain"].startswith("web") and row["task"] != "RW") or row["label_en"] == "Korea":
            grouped[(row["speaker"], row["label_en"])].append((index, row))
    jobs = []
    for (speaker, country), sources in sorted(grouped.items()):
        for first in range(0, len(sources) - 2, 3):
            group = sources[first:first + 3]
            jobs.append({"job_id": "three_{:05d}".format(len(jobs)),
                         "source_indices": [index for index, _ in group],
                         "source_paths": [row["audio_path"] for _, row in group],
                         "country": country, "speaker": speaker, "domain": group[0][1]["domain"],
                         "task": "three_validation_sentences",
                         "source_duration_sec": sum(float(row["duration_sec"]) for _, row in group)})
    return jobs, missing


def concatenated_wav(paths):
    waves = []
    for path in paths:
        decoded = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
                                  "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        waves.append(np.frombuffer(decoded.stdout, np.float32))
    buffer = io.BytesIO()
    sf.write(buffer, np.concatenate(waves), 16000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def predict(job, url, parts):
    started = time.monotonic()
    record = dict(job)
    try:
        audio = (Path(job["source_paths"][0]).read_bytes() if len(job["source_paths"]) == 1
                 else concatenated_wav(job["source_paths"]))
        response = session().post(url.rstrip("/") + "/predict_long",
                                  files={"audio": ("nat_validation_" + job["job_id"] + ".wav", audio, "audio/wav")},
                                  data={"parts": str(parts)}, timeout=(30, 90))
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": response.text[:300], "code": "non_json_response"}
        record.update({"status": response.status_code, "payload": payload, "bytes": len(audio)})
    except (requests.RequestException, OSError, subprocess.CalledProcessError) as error:
        record.update({"status": 0, "payload": {"code": type(error).__name__, "error": str(error)[:300]}})
    record["elapsed_sec"] = round(time.monotonic() - started, 3)
    record["time_utc"] = datetime.now(timezone.utc).isoformat()
    return record


def summarize(records, manifest_rows, train_rows, metadata):
    train_speakers = collections.defaultdict(set)
    for row in train_rows:
        train_speakers[row["label_en"]].add(row["speaker"])

    def metrics(selected):
        if not selected:
            return None
        valid = [record for record in selected if record["status"] == 200 and record["payload"].get("top5")]
        h1 = sum(record["payload"]["top5"][0]["label"] == record["country"] for record in valid)
        h5 = sum(any(prediction["label"] == record["country"] for prediction in record["payload"]["top5"]) for record in valid)
        by_speaker = collections.defaultdict(list)
        for record in selected:
            top5 = record["payload"].get("top5", []) if record["status"] == 200 else []
            by_speaker[record["speaker"]].append((bool(top5 and top5[0]["label"] == record["country"]),
                                                 any(item["label"] == record["country"] for item in top5)))
        return {"n": len(selected), "speakers": len(by_speaker), "predictions": len(valid),
                "rejections_or_errors": len(selected) - len(valid), "correct_top1": h1, "correct_top5": h5,
                "top1": h1 / len(selected), "top5": h5 / len(selected),
                "top1_given_prediction": h1 / len(valid) if valid else None,
                "top5_given_prediction": h5 / len(valid) if valid else None,
                "speaker_weighted_top1": float(np.mean([np.mean([hit[0] for hit in hits]) for hits in by_speaker.values()])),
                "speaker_weighted_top5": float(np.mean([np.mean([hit[1] for hit in hits]) for hits in by_speaker.values()])),
                "actual_parts": dict(collections.Counter(str(record["payload"].get("n")) for record in valid)),
                "low_confidence": sum(bool(record["payload"].get("low_confidence")) for record in valid)}

    groups = {"all": records, "foreign": [record for record in records if record["country"] != "Korea"],
              "web_sentence": [record for record in records if record["domain"].startswith("web") and record["task"] != "RW"],
              "web_word": [record for record in records if record["domain"].startswith("web") and record["task"] == "RW"],
              "p505": [record for record in records if record["domain"] == "p505"],
              "korean": [record for record in records if record["country"] == "Korea"]}
    by_country = []
    for country in sorted(set(row["label_en"] for row in train_rows)):
        selected = [record for record in records if record["country"] == country]
        errors = collections.Counter(record["payload"]["top5"][0]["label"] for record in selected
                                     if record["status"] == 200 and record["payload"].get("top5")
                                     and record["payload"]["top5"][0]["label"] != country)
        country_groups = {key: metrics([record for record in values if record["country"] == country])
                          for key, values in groups.items() if key != "foreign"}
        by_country.append({"country": country, "country_ko": COUNTRY_KO.get(country, country),
                           "train_speakers": len(train_speakers[country]), **country_groups,
                           "top_errors": [{"country": other, "country_ko": COUNTRY_KO.get(other, other), "n": count}
                                          for other, count in errors.most_common(5)],
                           "classified_as_philippines": sum(record["status"] == 200 and bool(record["payload"].get("top5"))
                                                            and record["payload"]["top5"][0]["label"] == "Philippines"
                                                            for record in selected),
                           "rejection_codes": dict(collections.Counter(record["payload"].get("code", "unknown")
                                                                       for record in selected if record["status"] != 200))})
    by_country.sort(key=lambda item: -(item["all"]["top1"] if item["all"] else -1))
    report = {"metadata": metadata, "overview": {key: metrics(values) for key, values in groups.items()},
              "http_statuses": dict(collections.Counter(str(record["status"]) for record in records)),
              "rejection_codes": dict(collections.Counter(record["payload"].get("code", "unknown")
                                                           for record in records if record["status"] != 200)),
              "per_country": by_country}
    return report


def save_summary(output, report):
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (output / "per_country.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["country", "country_ko", "train_speakers", "validation_speakers", "n", "predictions", "rejections",
                  "top1", "top5", "speaker_weighted_top1", "speaker_weighted_top5", "top1_given_prediction",
                  "sentence_n", "sentence_top1", "sentence_top5", "word_n", "word_top1", "actual_parts", "top_errors"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in report["per_country"]:
            scores = item["all"] or {}
            sent, words = item["web_sentence"] or {}, item["web_word"] or {}
            writer.writerow({"country": item["country"], "country_ko": item["country_ko"], "train_speakers": item["train_speakers"],
                             "validation_speakers": scores.get("speakers", 0), "n": scores.get("n", 0),
                             "predictions": scores.get("predictions", 0), "rejections": scores.get("rejections_or_errors", 0),
                             "top1": scores.get("top1"), "top5": scores.get("top5"),
                             "speaker_weighted_top1": scores.get("speaker_weighted_top1"),
                             "speaker_weighted_top5": scores.get("speaker_weighted_top5"),
                             "top1_given_prediction": scores.get("top1_given_prediction"),
                             "sentence_n": sent.get("n", 0), "sentence_top1": sent.get("top1"), "sentence_top5": sent.get("top5"),
                             "word_n": words.get("n", 0), "word_top1": words.get("top1"),
                             "actual_parts": json.dumps(scores.get("actual_parts", {})),
                             "top_errors": json.dumps(item["top_errors"], ensure_ascii=False)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["single", "three_sentences"], default="single")
    parser.add_argument("--parts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true", help="With --resume, retry transport errors and HTTP 5xx only")
    args = parser.parse_args()
    if args.manifest.name != "valid.csv":
        raise ValueError("Use the training run's actual valid.csv manifest")
    rows = read_csv(args.manifest)
    train = read_csv(args.manifest.with_name("train.csv"))
    test = read_csv(args.manifest.with_name("test.csv"))
    valid_speakers = set(row["speaker"] for row in rows)
    overlap = {"train_valid": len(valid_speakers & set(row["speaker"] for row in train)),
               "valid_test": len(valid_speakers & set(row["speaker"] for row in test))}
    jobs, missing = jobs_for(rows, args.mode)
    args.out.mkdir(parents=True, exist_ok=True)
    records_file = args.out / "responses.jsonl"
    metadata_file = args.out / "metadata.json"
    fingerprint = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    health = session().get(args.url.rstrip("/") + "/health", timeout=15)
    health.raise_for_status()
    health_data = health.json()
    if health_data.get("backend") != "rbln" or not health_data.get("gate") or not health_data.get("silence_norm"):
        raise RuntimeError("Expected the current NPU RBLN service with its gate and silence normalization")
    records = []
    if records_file.exists():
        if not args.resume:
            raise RuntimeError("Output exists; use --resume to keep previous responses")
        metadata = json.loads(metadata_file.read_text())
        for key, value in {"manifest_sha256": fingerprint, "mode": args.mode, "url": args.url,
                           "parts_requested": args.parts}.items():
            if metadata[key] != value:
                raise ValueError("Resume configuration differs: " + key)
        attempts = [json.loads(line) for line in records_file.read_text().splitlines() if line]
        records = list({record["job_id"]: record for record in attempts}.values())
    else:
        metadata = {"started_at": datetime.now(timezone.utc).isoformat(), "manifest": str(args.manifest),
                    "manifest_sha256": fingerprint, "manifest_rows": len(rows), "manifest_speakers": len(valid_speakers),
                    "mode": args.mode, "url": args.url, "endpoint": "/predict_long", "parts_requested": args.parts,
                    "workers": args.workers, "jobs": len(jobs), "missing_audio": missing,
                    "speaker_overlap": overlap, "npu_health_before": health_data,
                    "method": "Each validation audio is sent intact to the actual NPU service." if args.mode == "single"
                    else "Three different validation utterances from one held-out speaker are concatenated into 16kHz PCM16 WAV, then sent once with parts=3. Groups do not reuse utterances.",
                    "scoring": "API rejections count as incorrect; unavailable files are reported separately. Both request-weighted and speaker-weighted accuracy are reported."}
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    completed = set(record["job_id"] for record in records
                    if not args.retry_errors or 0 < record["status"] < 500)
    jobs_by_id = {job["job_id"]: job for job in jobs}
    for record in records:
        if record["source_indices"] != jobs_by_id[record["job_id"]]["source_indices"]:
            raise ValueError("Resume grouping differs: " + record["job_id"])
    pending = [job for job in jobs if job["job_id"] not in completed]
    records_by_id = {record["job_id"]: record for record in records}
    metadata.setdefault("execution_runs", []).append({"started_at": datetime.now(timezone.utc).isoformat(),
                                                       "workers": args.workers, "remaining_jobs": len(pending)})
    metadata["last_workers"] = args.workers
    metadata["jobs"] = len(jobs)
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"mode": args.mode, "jobs": len(jobs), "remaining": len(pending), "missing_files": len(missing),
                      "speaker_overlap": overlap, "health": health_data}, ensure_ascii=False), flush=True)
    started = time.monotonic()
    last_update = started
    with records_file.open("a", encoding="utf-8", buffering=1) as stream:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(predict, job, args.url, args.parts) for job in pending]
            for future in as_completed(futures):
                record = future.result()
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_by_id[record["job_id"]] = record
                records = list(records_by_id.values())
                now = time.monotonic()
                if len(records) % 100 == 0 or now - last_update >= 15 or len(records) == len(jobs):
                    counts = collections.Counter(item["status"] for item in records)
                    parts = collections.Counter(item["payload"].get("n") for item in records if item["status"] == 200)
                    print("progress {}/{} elapsed {:.0f}s statuses {} actual_parts {}".format(
                        len(records), len(jobs), now - started, dict(counts), dict(parts)), flush=True)
                    last_update = now
    records.sort(key=lambda record: record["job_id"])
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    metadata["response_attempts"] = len(records_file.read_text().splitlines())
    metadata["npu_health_after"] = session().get(args.url.rstrip("/") + "/health", timeout=15).json()
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    report = summarize(records, rows, train, metadata)
    save_summary(args.out, report)
    print(json.dumps({"overview": report["overview"], "http_statuses": report["http_statuses"],
                      "rejection_codes": report["rejection_codes"]}, ensure_ascii=False, indent=2), flush=True)
    print("country\ttrain_spk\tvalid_spk\tn\ttop1%\ttop5%\trejections\terrors", flush=True)
    for item in report["per_country"]:
        scores = item["all"]
        if scores:
            print("{}\t{}\t{}\t{}\t{:.1f}\t{:.1f}\t{}\t{}".format(
                item["country_ko"], item["train_speakers"], scores["speakers"], scores["n"], 100 * scores["top1"],
                100 * scores["top5"], scores["rejections_or_errors"], item["top_errors"]), flush=True)
    print("Saved", args.out, flush=True)


if __name__ == "__main__":
    main()
