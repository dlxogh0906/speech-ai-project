"""Export explicitly reviewed speaker segments, preserving source provenance."""
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
import json
import re


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reviews", type=_pl.Path, required=True)
    ap.add_argument("--decisions", type=_pl.Path, required=True)
    ap.add_argument("--sources", type=_pl.Path, required=True)
    ap.add_argument("--reserved-videos", type=_pl.Path, required=True)
    ap.add_argument("--base-train", type=_pl.Path)
    ap.add_argument("--base-review", type=_pl.Path,
                    help="Optional reviewed clip CSV to retain source spans and transcripts from the first pilot")
    ap.add_argument("--out", type=_pl.Path, required=True)
    args = ap.parse_args()
    import soundfile as sf
    args.out.mkdir(parents=True, exist_ok=True)
    clips = args.out / "clips"
    clips.mkdir(exist_ok=True)
    decisions = json.loads(args.decisions.read_text())
    source_rows = json.loads(args.sources.read_text())
    sources = {s.get("source_id", s["video_id"]): s for s in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError("Multiple guests need distinct source_id values")
    reserved = {s["id"] for s in json.loads(args.reserved_videos.read_text())}
    rows = {"train": [], "holdout": []}
    selected_spans = collections.defaultdict(list)
    for source_id, decision in decisions.items():
        source = sources[source_id]
        vid = source["video_id"]
        split = source["split"]
        if split not in rows:
            raise ValueError("Invalid split " + split)
        if split == "train" and vid in reserved:
            raise ValueError("Prior evaluation video cannot enter training: " + vid)
        if "creative commons" not in source["license"].lower():
            raise ValueError("Missing reviewed source-use basis: " + vid)
        report = json.loads((args.reviews / (source.get("review_id", vid) + ".json")).read_text())
        if report["source"]["speaker_id"] != source.get("review_speaker_id", source["speaker_id"]):
            raise ValueError("Identity differs from reviewed source")
        wave, sr = sf.read(report["audio_path"], dtype="float32")
        if sr != 16000 or wave.ndim != 1:
            raise ValueError("Expected mono 16kHz")
        indices = decision.get("accepted_indices", [])
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate segment")
        segments = [("{:04d}".format(i), report["segments"][i]) for i in indices]
        for i, segment in enumerate(decision.get("accepted_ranges", [])):
            if not segment.get("transcript") or not segment.get("review_evidence"):
                raise ValueError("Explicit ranges need transcript and speaker review evidence")
            segments.append(("range{:04d}".format(i), segment))
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", source_id)
        for clip_suffix, segment in segments:
            a, b = segment["start"], segment["end"]
            if a < decision["min_start_sec"] or not 3 <= b - a <= 8.1 or b * sr > len(wave) + 1:
                raise ValueError("Unreviewed duration/intro for {}:{}".format(source_id, clip_suffix))
            if any(a < end and b > start for start, end in selected_spans[vid]):
                raise ValueError("Overlapping selected source spans for " + vid)
            selected_spans[vid].append((a, b))
            dest = clips / (stem + "_" + clip_suffix + ".wav")
            clip = wave[round(a * sr):round(b * sr)]
            sf.write(dest, clip, sr, subtype="PCM_16")
            rows[split].append({"clip_id": dest.stem, "audio_path": str(dest), "label_en": source["country"],
                                "speaker": source["speaker_id"], "domain": "youtube", "task": "RS",
                                "duration_sec": len(clip) / sr, "video_id": vid, "start_sec": a,
                                "end_sec": b, "transcript": segment["transcript"],
                                "review_status": "accepted_explicit_range_review" if "review_evidence" in segment
                                else "accepted_transcript_cluster_review"})
    fields = ["clip_id", "audio_path", "label_en", "speaker", "domain", "task", "duration_sec",
              "video_id", "start_sec", "end_sec", "transcript", "review_status"]
    new_counts = {k: len(v) for k, v in rows.items()}
    if args.base_train:
        base_review = {}
        if args.base_review:
            with args.base_review.open(encoding="utf-8", newline="") as f:
                base_review = {r["audio_path"]: r for r in csv.DictReader(f)}
        with args.base_train.open(encoding="utf-8", newline="") as f:
            base = []
            for x in csv.DictReader(f):
                metadata = dict(base_review.get(x["audio_path"], {}))
                metadata.update({k: v for k, v in x.items() if v or k not in metadata})
                stem = _pl.Path(x["audio_path"]).stem
                if not metadata.get("clip_id"):
                    metadata["clip_id"] = stem
                if not metadata.get("video_id"):
                    metadata["video_id"] = stem.rsplit("_", 1)[0]
                base.append({k: metadata.get(k, "") for k in fields})
        rows["train"] = base + rows["train"]
    if {x["speaker"] for x in rows["train"]} & {x["speaker"] for x in rows["holdout"]}:
        raise ValueError("Train/holdout speaker overlap")
    summary = {"new_clips": new_counts,
               "review_limit": "Transcript and speaker-cluster review; no exhaustive listening/diarization annotation",
               "sources": list(sources.values()), "decisions_path": str(args.decisions),
               "base_train_path": str(args.base_train) if args.base_train else None}
    for split, values in rows.items():
        with (args.out / ("youtube_" + split + ".csv")).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(values)
        summary[split] = {"clips": len(values), "duration_sec": sum(float(x["duration_sec"]) for x in values),
                          "speakers": len({x["speaker"] for x in values}),
                          "countries": dict(collections.Counter(x["label_en"] for x in values))}
    (args.out / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ("train", "holdout", "new_clips")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
