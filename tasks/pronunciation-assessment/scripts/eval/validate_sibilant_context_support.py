"""Keep a 40ms signal identical while changing only its surrounding audio.

Oracle F0 isolates the effect of contextual formant estimation on corrected
harmonics. These synthetic interventions are not Korean pronunciation labels.
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
from collections import defaultdict
import hashlib
import importlib.util
import itertools
import json

import numpy as np
import parselmouth
from scipy.signal import lfilter
from mdd.sibilant_acoustics import harmonic_amplitudes


def source(sr, f0, f1, phase):
    t = np.arange(sr) / sr
    x = sum(np.sin(2 * np.pi * k * (f0 * t + phase)) / k for k in range(1, int(7900 / f0) + 1))
    for f, b in zip((f1, 1800., 2800., 3800., 4900.), (80., 120., 160., 200., 250.)):
        radius = np.exp(-np.pi * b / sr)
        denominator = [1., -2 * radius * np.cos(2 * np.pi * f / sr), radius ** 2]
        x = lfilter([sum(denominator)], denominator, x)
    return .1 * x / np.sqrt(np.mean(x[int(.2 * sr):] ** 2))


def replace_outside(x, other, sr, interval=(.48, .52), transition=.010):
    t = np.arange(len(x)) / sr
    distance = np.maximum(interval[0] - t, t - interval[1])
    weight = .5 - .5 * np.cos(np.pi * np.clip(distance / transition, 0, 1))
    y = x * (1 - weight) + other * weight
    # Preserve central samples exactly, including rounding at interval edges.
    begin, end = [int(round(v * sr)) for v in interval]
    y[begin:end] = x[begin:end]
    return y


def describe(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return {"total": len(values), "available": len(v), "median": float(np.median(v)) if v else None,
            "median_absolute": float(np.median(np.abs(v))) if v else None,
            "maximum_absolute": float(max(abs(x) for x in v)) if v else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opensauce-root", type=Path, required=True)
    ap.add_argument("--praat-source-reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("refusing to overwrite a context-support audit")
    correction_path = a.opensauce_root / "opensauce/harmonics.py"
    spec = importlib.util.spec_from_file_location("context_support_correction", correction_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sr, rng = 16000, np.random.RandomState(20261024)
    interval, centers = (.48, .52), (.490, .500, .510)
    grid = list(itertools.product((100., 200., 300.), (250., 500., 750.), (0., .25)))
    records = []
    for i, (f0, f1, phase) in enumerate(grid):
        x = source(sr, f0, f1, phase)
        other = source(sr, f0, 900., phase)
        noise = rng.normal(size=len(x))
        noise *= .1 / np.sqrt(np.mean(noise ** 2))
        variants = [("unchanged", x), ("outside_silence", replace_outside(x, np.zeros_like(x), sr)),
                    ("outside_alternate_vowel", replace_outside(x, other, sr)),
                    ("outside_noise_rms1", replace_outside(x, noise, sr)),
                    ("outside_noise_rms4", replace_outside(x, 4 * noise, sr))]
        target_hash = hashlib.sha256(x[7680:8320].tobytes()).hexdigest()
        for name, y in variants:
            if hashlib.sha256(y[7680:8320].tobytes()).hexdigest() != target_hash:
                raise RuntimeError("intervention changed the target 40ms")
            before = hashlib.sha256(y.tobytes()).hexdigest()
            fm = parselmouth.Sound(y, sampling_frequency=sr).to_formant_burg(time_step=.005,
                max_number_of_formants=5., maximum_formant=5500., window_length=.025, pre_emphasis_from=50.)
            for center in centers:
                ff = [fm.get_value_at_time(k, center) for k in (1, 2)]
                bb = [fm.get_bandwidth_at_time(k, center) for k in (1, 2)]
                h = harmonic_amplitudes(y, sr, center, f0, ff, bb, module.correction_iseli_i)
                contained = bool(h and h["window_start"] >= interval[0] - 1e-10 and h["window_end"] <= interval[1] + 1e-10)
                records.append({"configuration": i, "F0_oracle_hz": f0, "central_source_F1_hz": f1,
                    "phase": phase, "variant": name, "center": center, "target_PCM_float_sha256": target_hash,
                    "direct_harmonic_window_contained": contained,
                    "estimated_formants": [float(v) if np.isfinite(v) else None for v in ff],
                    "estimated_bandwidths": [float(v) if np.isfinite(v) else None for v in bb], "measurement": h})
            if hashlib.sha256(y.tobytes()).hexdigest() != before:
                raise RuntimeError("measurement changed synthetic input")
    baseline = {(r["configuration"], r["center"]): r for r in records if r["variant"] == "unchanged"}
    grouped = defaultdict(list)
    for row in records:
        if not row["direct_harmonic_window_contained"]:
            continue
        base = baseline[(row["configuration"], row["center"])]
        differences = {}
        for key in ("h1h2_raw", "h1h2_corrected", "h1a1_corrected"):
            x, y = base["measurement"][key], row["measurement"][key]
            differences[key] = y - x if x is not None and y is not None else None
        row["difference_from_unchanged_db"] = differences
        if abs(differences["h1h2_raw"]) > 1e-10:
            raise RuntimeError("oracle-F0 raw harmonics changed despite identical direct samples")
        grouped[row["variant"]].append(row)
    result = {"status": "synthetic_estimator_context_intervention_not_phonetic_validation",
        "configurations": len(grid), "candidate_measurements": len(records),
        "contained_direct_windows": sum(r["direct_harmonic_window_contained"] for r in records),
        "method": {"sample_rate": sr, "identical_interval": interval, "outside_crossfade_seconds": .010,
                   "oracle_F0": True, "Burg_formants": 5, "Burg_ceiling": 5500,
                   "Burg_effective_window_ms": 25, "Burg_physical_window_ms": 50,
                   "parselmouth": parselmouth.__version__, "praat": parselmouth.PRAAT_VERSION},
        "by_variant": {name: {key: describe([r["difference_from_unchanged_db"][key] for r in rows])
                        for key in ("h1h2_raw", "h1h2_corrected", "h1a1_corrected")} for name, rows in grouped.items()},
        "limitations": ["Five known stationary resonators and 1/k source harmonics are engineering signals, not natural Korean voice.",
            "Outside noise is contamination, not a validated breathy-voice simulation.",
            "LPC formant bias is not separated from contextual dependence; paired changes isolate input context while keeping target PCM identical.",
            "Oracle F0 removes pitch-estimation effects; production F0 has additional dependencies not quantified here.",
            "No correction threshold, phoneme classification, or native perceptual accuracy is established."],
        "records": records,
        "provenance": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), correction_path,
            ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_acoustics.py", a.praat_source_reference / "provenance.json"]}}
    a.out.mkdir(parents=True)
    (a.out / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (a.out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"contained_direct_windows": result["contained_direct_windows"], "by_variant": result["by_variant"]}, indent=2))


if __name__ == "__main__":
    main()
