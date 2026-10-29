"""Inventory PCM limitations without interpreting them as pronunciation errors.

Exact constant runs are observations, not proof of a particular recording
failure. Original sample-rate metadata comes from the historical resampling
index; upsampling does not recover missing spectral information.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import argparse
import datetime
import hashlib
import json
import wave

import numpy as np


def constant_runs(x, sr, offset=0., minimum_seconds=.001):
    if not len(x):
        return []
    edges = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1, len(x)]
    return [{"start": offset + a / sr, "end": offset + b / sr,
             "samples": int(b - a), "value_int16": int(x[a])}
            for a, b in zip(edges[:-1], edges[1:])
            if b - a >= max(2, int(np.ceil(minimum_seconds * sr)))]


def inspect_interval(x, sr, begin, end):
    a, b = max(0, int(round(begin * sr))), min(len(x), int(round(end * sr)))
    y = x[a:b]
    if not len(y):
        raise ValueError("empty PCM interval")
    amplitude = np.abs(y.astype(np.int32))
    runs = constant_runs(y, sr, a / sr)
    return {"interval": [a / sr, b / sr], "samples": len(y),
            "rms_dbfs": float(10 * np.log10(np.mean((y.astype(float) / 32768) ** 2) + 1e-20)),
            "peak_abs_int16": int(amplitude.max()),
            "exact_zero_fraction": float(np.mean(y == 0)),
            "rail_samples": int(np.sum((y == -32768) | (y == 32767))),
            "constant_runs_at_least_1ms": runs,
            "max_constant_run_ms": max([1000 * r["samples"] / sr for r in runs], default=0.)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--source-audit", type=Path, required=True)
    ap.add_argument("--visual-landmarks", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("refusing to overwrite a PCM audit")
    manifest = a.packet / "items.private.jsonl"
    items = [json.loads(s) for s in manifest.read_text().splitlines()]
    original = {r["id"]: r for r in json.loads(a.source_audit.read_text())["records"]}
    visual = {r["id"]: r for r in map(json.loads, a.visual_landmarks.read_text().splitlines())} if a.visual_landmarks else {}
    records, cache = [], {}
    for item in items:
        path = item["source_wav"]
        if path not in cache:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            with wave.open(path) as w:
                if w.getnchannels() != 1 or w.getsampwidth() != 2:
                    raise ValueError("requires mono PCM16")
                sr = w.getframerate()
                x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            cache[path] = x, sr, digest
        x, sr, digest = cache[path]
        if digest != item["source_sha256"]:
            raise ValueError("source changed since packet preparation")
        intervals = {"analysis": item["analysis_interval"], "TAG_consonant": item["tag_consonant"], "TAG_vowel": item["tag_vowel"]}
        v = visual.get(item["id"])
        if v and v.get("voicing_first") is not None:
            if v["source_sha256"] != digest:
                raise ValueError("visual review source differs")
            intervals["assistant_first_voicing_40ms"] = [v["voicing_first"], v["voicing_first"] + .040]
        src_sr = int(original[item["id"]]["src_sr"]) if item["id"] in original else None
        records.append({"id": item["id"], "source_sha256": digest, "analysis_rate_hz": sr,
            "historical_original_rate_hz": src_sr,
            "nyquist_upper_bound_hz": min(sr, src_sr) / 2 if src_sr else sr / 2,
            "high_band_4_to_8khz_unavailable": src_sr <= 8000 if src_sr else None,
            "intervals": {name: inspect_interval(x, sr, *interval) for name, interval in intervals.items()}})
    result = {"status": "PCM_observations_not_phonetic_labels", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "items": len(records), "unique_recordings": len(cache),
        "items_with_constant_run_in_TAG_vowel": [r["id"] for r in records if r["intervals"]["TAG_vowel"]["constant_runs_at_least_1ms"]],
        "items_with_original_bandwidth_at_most_4khz": [r["id"] for r in records if r["high_band_4_to_8khz_unavailable"]],
        "items_with_rail_samples_in_analysis": [r["id"] for r in records if r["intervals"]["analysis"]["rail_samples"]],
        "limitations": ["Constant samples may arise from multiple acquisition or processing causes; this audit does not diagnose the cause.",
            "A 1ms inventory threshold is not a validated defect or exclusion threshold.",
            "Nyquist is an upper bound, not a measured usable channel bandwidth.",
            "TAG boundaries and assistant boundaries are unvalidated.",
            "All items are retained; unavailable measurements must not be mapped to normal or error."],
        "provenance": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [manifest, a.source_audit, Path(__file__)] + ([a.visual_landmarks] if a.visual_landmarks else [])},
        "records": records}
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k.startswith("items") or k == "unique_recordings"}))


if __name__ == "__main__":
    main()
