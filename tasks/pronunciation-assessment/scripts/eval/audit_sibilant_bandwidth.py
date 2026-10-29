"""Measure channel sensitivity of spectral cues using paired resampling.

An 8kHz down/up-sampling intervention changes the recording bandwidth, not the
speaker's articulation. No pronunciation labels or decision thresholds are fit.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

import argparse
import hashlib
import json
import wave

import numpy as np
import scipy
from scipy import signal
from mdd.sibilant_acoustics import frame_signal


def roundtrip_8k(x, filter_mode="default"):
    if filter_mode not in ("default", "strict"):
        raise ValueError("unknown resampling filter")
    window = ("kaiser", 5.) if filter_mode == "default" else signal.firwin(321, .5, window=("kaiser", 12.))
    return signal.resample_poly(signal.resample_poly(x, 1, 2, window=window),
                                2, 1, window=window)[:len(x)]


def tracks(x, sr=16000):
    times, frames = frame_signal(x, sr, .025, .005)
    frames = frames - frames.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), 1024)) ** 2
    freq = np.fft.rfftfreq(1024, 1. / sr)
    silent = np.mean(frames ** 2, axis=1) < 1e-10
    metrics = {}
    band = (freq >= 1000) & (freq < 7900)
    total = power[:, band].sum(axis=1)
    metrics["centroid_1_to_7p9khz_hz"] = np.sum(power[:, band] * freq[band], axis=1) / np.maximum(total, 1e-30)
    for cutoff in (2000, 3000, 4000):
        lo = power[:, (freq >= 100) & (freq < cutoff)].sum(axis=1)
        hi = power[:, (freq >= cutoff) & (freq < 7900)].sum(axis=1)
        metrics["balance_%dhz_db" % cutoff] = 10 * np.log10((hi + 1e-20) / (lo + 1e-20))
    # Disjoint bands avoid the 4kHz transition region. This is an observation,
    # not a reliable estimate of a channel cutoff from a single speech segment.
    hi = power[:, (freq >= 4200) & (freq < 6000)].sum(axis=1)
    lo = power[:, (freq >= 2500) & (freq < 3500)].sum(axis=1)
    metrics["upper_4p2_6_to_middle_2p5_3p5khz_db"] = 10 * np.log10((hi + 1e-20) / (lo + 1e-20))
    for key in metrics:
        metrics[key][silent] = np.nan
    return times, metrics


def describe(values):
    xs = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return {"available": len(xs), "median": float(np.median(xs)) if len(xs) else None,
            "minimum": float(xs.min()) if len(xs) else None, "maximum": float(xs.max()) if len(xs) else None}


def paired_window(times, original, changed, offset, begin, end):
    # The n=400 window center is between two samples; require the entire
    # physical 25ms window to lie inside the requested interval.
    selected = (times + offset - 199.5 / 16000 >= begin) & (times + offset + 200.5 / 16000 <= end)
    metrics = {}
    for key in original:
        valid = selected & np.isfinite(original[key]) & np.isfinite(changed[key])
        a, b = original[key][valid], changed[key][valid]
        metrics[key] = {"paired_frames": int(valid.sum()),
                        "original": describe(a), "roundtrip_8k": describe(b),
                        "paired_frame_difference": describe(b - a)}
    return {"interval_original_seconds": [begin, end], "candidate_frames": int(selected.sum()), "metrics": metrics}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--visual-landmarks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--filter", choices=("default", "strict"), default="default",
                    help="strict uses 321 taps / Kaiser beta=12 to check residual imaging")
    a = ap.parse_args()
    if a.out.exists():
        ap.error("refusing to overwrite a bandwidth audit")
    manifest = a.packet / "items.private.jsonl"
    items = [json.loads(s) for s in manifest.read_text().splitlines()]
    visual_rows = [json.loads(s) for s in a.visual_landmarks.read_text().splitlines()]
    visual = {r["id"]: r for r in visual_rows}
    if len(visual) != len(visual_rows):
        ap.error("duplicate visual IDs")
    rows, source_fingerprints = [], {}
    for item in items:
        path = Path(item["source_wav"])
        if str(path) not in source_fingerprints:
            source_fingerprints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if source_fingerprints[str(path)] != item["source_sha256"]:
            ap.error("source PCM file changed")
        with wave.open(str(path), "rb") as f:
            if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (16000, 1, 2):
                ap.error("expected 16k mono PCM16")
            # Extra context prevents resampling padding from dominating the
            # measured analysis interval. Original input is never written.
            start = max(0, int(round((item["analysis_interval"][0] - .1) * 16000)))
            stop = min(f.getnframes(), int(round((item["analysis_interval"][1] + .1) * 16000)))
            f.setpos(start)
            x = np.frombuffer(f.readframes(stop - start), np.int16).astype(float) / 32768
        before = hashlib.sha256(x.tobytes()).hexdigest()
        changed = roundtrip_8k(x, a.filter)
        times, original = tracks(x)
        changed_times, intervention = tracks(changed)
        if before != hashlib.sha256(x.tobytes()).hexdigest() or not np.array_equal(times, changed_times):
            raise RuntimeError("input mutation or time-axis mismatch")
        windows = {"analysis": paired_window(times, original, intervention, start / 16000, *item["analysis_interval"])}
        v = visual.get(item["id"])
        if v and v["source_sha256"] != item["source_sha256"]:
            ap.error("visual provenance differs")
        interval = None
        if v and v.get("frication_start") is not None and v.get("voicing_first") is not None:
            # This can include aspiration; it is deliberately not called a
            # pure frication interval or a human-labelled acoustic segment.
            interval = [v["frication_start"], v["voicing_first"]]
            if interval[1] > interval[0]:
                windows["assistant_noise_onset_to_voicing"] = paired_window(times, original, intervention, start / 16000, *interval)
        rows.append({"id": item["id"], "following_vowel": item["following_vowel"], "source_sha256": item["source_sha256"],
                     "provisional_noise_interval": interval, "windows": windows})
    summaries = {}
    for vowel in sorted({r["following_vowel"] for r in rows}):
        selected = [r for r in rows if r["following_vowel"] == vowel]
        summary = {"items": len(selected), "noise_windows": sum("assistant_noise_onset_to_voicing" in r["windows"] for r in selected), "paired_item_median_differences": {}}
        for key in rows[0]["windows"]["analysis"]["metrics"]:
            vals = [r["windows"]["assistant_noise_onset_to_voicing"]["metrics"][key]["paired_frame_difference"]["median"]
                    for r in selected if "assistant_noise_onset_to_voicing" in r["windows"]]
            summary["paired_item_median_differences"][key] = describe(vals)
        summaries[vowel] = summary
    sources = [manifest, a.visual_landmarks, Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_acoustics.py"]
    result = {"status": "paired_channel_intervention_not_pronunciation_validation", "items": len(rows),
              "method": {"analysis_rate_hz": 16000, "intermediate_rate_hz": 8000, "resampling": "scipy.signal.resample_poly twice",
                         "filter": a.filter, "filter_specification": "Kaiser beta=5; scipy default length" if a.filter == "default" else "firwin 321 taps; normalized cutoff .5; Kaiser beta=12",
                         "scipy_version": scipy.__version__, "spectrum_window_ms": 25, "spectrum_step_ms": 5, "fft_size": 1024,
                         "preemphasis": False, "original_waveform_immutability_checked": True},
              "by_vowel": summaries, "records": rows,
              "provenance": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "limitations": ["Synthetic bandwidth restriction is not a recording of a different pronunciation.",
                              "Noise-onset-to-voicing windows are provisional and may contain aspiration or preceding voiced overlap.",
                              "Observed upper-band energy cannot alone diagnose a recording filter or a pronunciation error.",
                              "Real speech values are available only below the original recording bandwidth; upsampling cannot recover lost information.",
                              "Finite resampling filters may introduce weak spectral images; newly observed high-frequency energy is not restored speech information.",
                              "Within-window frames are dependent; summary counts refer to items.",
                              "No classification thresholds or accuracy claims; missing intervals remain missing."]}
    a.out.mkdir(parents=True)
    (a.out / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (a.out / "source_snapshot").mkdir()
    for p in sources[2:]:
        (a.out / "source_snapshot" / p.name).write_bytes(p.read_bytes())
    print(json.dumps({"items": len(rows), "by_vowel": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
