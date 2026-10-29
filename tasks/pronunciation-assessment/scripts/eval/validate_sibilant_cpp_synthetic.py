"""Audit CPP dependence on available duration, pitch, filter, and added noise.

Praat 6.1.38 uses physical Gaussian support of 6/pitch_floor seconds,
shortening it to the entire signal when that signal is shorter. A successful
CPP call on a 20ms crop therefore does not imply comparable 100ms analysis.
All outputs here are engineering observations, never phoneme probabilities.
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
from collections import defaultdict
import hashlib
import itertools
import json

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import lfilter


def cpp(x, sr, pitch_floor=60.):
    if parselmouth.PRAAT_VERSION != "6.1.38":
        raise RuntimeError("this audit pins Praat 6.1.38; verify window semantics before changing")
    sound = parselmouth.Sound(x, sampling_frequency=sr)
    cepstrogram = call(sound, "To PowerCepstrogram", pitch_floor, .002, 5000., 50.)
    value = call(cepstrogram, "Get CPPS", False, 0., 0., 60., 500., .05, "Parabolic", .001, 0., "Straight", "Robust")
    matrix = call(cepstrogram, "To Matrix")
    duration = len(x) / sr
    return {"cpp_db": float(value) if np.isfinite(value) else None,
            "input_duration_ms": 1000 * duration,
            "requested_effective_window_ms": 3000 / pitch_floor,
            "actual_physical_support_ms": 1000 * min(duration, 6 / pitch_floor),
            "short_signal_window_override": duration < 6 / pitch_floor - 1e-9,
            "frame_count": int(matrix.nx),
            "first_frame_center_ms": 1000 * float(matrix.x1)}


def source(sr, f0, f1, f2, phase):
    t = np.arange(sr) / sr
    x = sum(np.sin(2 * np.pi * k * (f0 * t + phase)) / k for k in range(1, int(7900 / f0) + 1))
    for f, b in [(f1, 80.), (f2, 120.)]:
        radius = np.exp(-np.pi * b / sr)
        den = [1., -2 * radius * np.cos(2 * np.pi * f / sr), radius ** 2]
        x = lfilter([sum(den)], den, x)
    return .1 * x / np.sqrt(np.mean(x[int(.2 * sr):] ** 2))


def describe(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return {"total": len(values), "available": len(v), "median": float(np.median(v)) if v else None,
            "minimum": min(v) if v else None, "maximum": max(v) if v else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--praat-source-reference", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("refusing to overwrite a CPP audit")
    sr, rng = 16000, np.random.RandomState(20261024)
    grid = list(itertools.product((90., 120., 180., 240., 300.), (250., 500., 750.), (1200., 2400.), (0., .25)))
    records = []
    durations = (.020, .030, .040, .060, .100, .200)
    for i, (f0, f1, f2, phase) in enumerate(grid):
        x = source(sr, f0, f1, f2, phase)
        noise = rng.normal(size=len(x))
        noise /= np.sqrt(np.mean(noise ** 2))
        for snr in (None, 20., 10., 0.):
            noisy = x.copy() if snr is None else x + .1 * 10 ** (-snr / 20) * noise
            before = hashlib.sha256(noisy.tobytes()).hexdigest()
            for duration in durations:
                begin, end = int(round((.5 - duration / 2) * sr)), int(round((.5 + duration / 2) * sr))
                measure = cpp(noisy[begin:end], sr)
                records.append({"configuration": i, "F0": f0, "F1": f1, "F2": f2,
                    "phase_cycles": phase, "added_white_noise_SNR_db": snr,
                    "true_cycles_in_crop": duration * f0, **measure})
            if hashlib.sha256(noisy.tobytes()).hexdigest() != before:
                raise RuntimeError("CPP measurement modified PCM")
        if (i + 1) % 10 == 0:
            print("CPP configurations %d/%d" % (i + 1, len(grid)), flush=True)
    groups = defaultdict(list)
    paired = defaultdict(dict)
    for r in records:
        groups[(r["input_duration_ms"], r["added_white_noise_SNR_db"])].append(r)
        paired[(r["configuration"], r["added_white_noise_SNR_db"])][r["input_duration_ms"]] = r["cpp_db"]
    result = {"status": "synthetic_CPP_sensitivity_not_pronunciation_accuracy", "configurations": len(grid), "measurements": len(records),
        "versions": {"parselmouth": parselmouth.__version__, "praat": parselmouth.PRAAT_VERSION},
        "parameters": {"pitch_floor_for_window": 60., "time_step": .002, "maximum_frequency": 5000., "preemphasis_hz": 50.,
            "time_smoothing": 0., "quefrency_smoothing": 0., "peak_search_pitch_range": [60., 500.],
            "trend_range": [.001, 0.], "trend": "Straight", "fit": "Robust", "interpolation": "Parabolic"},
        "by_duration_and_added_noise": [{"duration_ms": key[0], "added_white_noise_SNR_db": key[1],
            "cpp_db": describe([r["cpp_db"] for r in rows])} for key, rows in groups.items()],
        "paired_difference_from_100ms": [{"duration_ms": d * 1000,
            "cpp_difference_db": describe([v[d * 1000] - v[100.] if v[d * 1000] is not None and v[100.] is not None else None for v in paired.values()])}
            for d in durations if d != .1],
        "limitations": ["Synthetic harmonic sources and known filters are not a realistic glottal model.",
            "Added white noise is recording-noise contamination, not a validated simulation of breathy voice.",
            "Duration, F0, filter, and noise all vary; pooled values are not reference thresholds.",
            "Short-signal Gaussian shortening changes both support and effective window; a finite value is not proof of reliability.",
            "CPP has no automatic interpretation as aspiration, breathiness, or a phoneme label."],
        "provenance": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), a.praat_source_reference / "PowerCepstrogram.cpp", a.praat_source_reference / "provenance.json"]},
        "records": records}
    a.out.mkdir(parents=True)
    (a.out / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (a.out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: result[k] for k in ("measurements", "by_duration_and_added_noise", "paired_difference_from_100ms")}, indent=2))


if __name__ == "__main__":
    main()
