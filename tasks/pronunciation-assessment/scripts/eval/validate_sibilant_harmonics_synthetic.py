"""Stress-test harmonic measurements against a known synthetic source.

The source has amplitude 1/k at harmonic k, then passes through known digital
resonators. These are engineering checks, not Korean speech or classification
validation. Oracle and deliberately perturbed parameters are reported apart.
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
import subprocess

import numpy as np
from scipy.signal import lfilter
from mdd.sibilant_acoustics import harmonic_amplitudes


def synthesize(sr, f0, formants, bandwidths, phase_cycles=0.):
    t = np.arange(sr) / sr
    harmonics = range(1, int((sr / 2 - 100) / f0) + 1)
    source = sum(np.sin(2 * np.pi * k * (f0 * t + phase_cycles)) / k for k in harmonics)
    filtered = source.copy()
    for f, b in zip(formants, bandwidths):
        radius = np.exp(-np.pi * b / sr)
        denominator = [1., -2 * radius * np.cos(2 * np.pi * f / sr), radius ** 2]
        filtered = lfilter([sum(denominator)], denominator, filtered)
    return .3 * filtered / np.max(np.abs(filtered))


def describe_errors(rows, metric):
    values = [r[metric] for r in rows if r.get(metric) is not None]
    a = np.abs(values)
    return {"configurations": len(rows), "available": len(values),
            "median_signed_error_db": float(np.median(values)) if len(values) else None,
            "median_absolute_error_db": float(np.median(a)) if len(a) else None,
            "p95_absolute_error_db": float(np.percentile(a, 95)) if len(a) else None,
            "maximum_absolute_error_db": float(a.max()) if len(a) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opensauce-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--praat", action="store_true", help="also inspect automatic formants on the same synthetic signals")
    a = ap.parse_args()
    if a.out.exists():
        ap.error("refusing to overwrite a synthetic audit")
    correction_path = a.opensauce_root / "opensauce/harmonics.py"
    spec = importlib.util.spec_from_file_location("synthetic_reference_correction", correction_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    correction = module.correction_iseli_i
    external = {"opensauce_commit": subprocess.check_output(["git", "-C", str(a.opensauce_root), "rev-parse", "HEAD"], text=True).strip(),
                "correction_sha256": hashlib.sha256(correction_path.read_bytes()).hexdigest()}
    if a.praat:
        import parselmouth
        external.update(parselmouth=parselmouth.__version__, praat=parselmouth.PRAAT_VERSION)
    sr = 16000
    grid = list(itertools.product((90., 120., 180., 240., 300.), (220., 300., 400., 550., 750.), (1200., 1800., 2400.), (60., 120.), (0., .25, .5)))
    records = []
    for i, (f0, f1, f2, b1, phase) in enumerate(grid):
        formants, bandwidths = [f1, f2], [b1, 120.]
        x = synthesize(sr, f0, formants, bandwidths, phase)
        before = hashlib.sha256(x.tobytes()).hexdigest()
        configs = [("oracle", f0, formants, bandwidths)]
        configs.extend(("F1_shift_%+dHz" % shift, f0, [f1 + shift, f2], bandwidths) for shift in (-100, -50, 50, 100))
        configs.extend(("F0_scale_%.2f" % scale, f0 * scale, formants, bandwidths) for scale in (.95, 1.05))
        configs.extend(("B1_scale_%.2f" % scale, f0, formants, [b1 * scale, 120.]) for scale in (.7, 1.3))
        if a.praat:
            sound = parselmouth.Sound(x, sampling_frequency=sr)
            for ceiling, n_formants in itertools.product((5000, 5500, 6000), (2., 5.)):
                fm = sound.to_formant_burg(time_step=.005, maximum_formant=ceiling, max_number_of_formants=n_formants)
                ff = [fm.get_value_at_time(k, .5) for k in (1, 2)]
                bb = [fm.get_bandwidth_at_time(k, .5) for k in (1, 2)]
                configs.append(("Praat_ceiling_%d_formants_%d_oracle_F0" % (ceiling, n_formants), f0, ff, bb))
        for name, estimate_f0, estimate_f, estimate_b in configs:
            h = harmonic_amplitudes(x, sr, .5, estimate_f0, estimate_f, estimate_b, correction)
            harmonic = h.get("a1_harmonic") if h else None
            expected_h1h2 = 20 * np.log10(2)
            # This comparison uses the known source at the harmonic selected
            # by the estimator. A wrong F0 can move the measurement off that
            # harmonic; the resulting deviation is deliberately retained.
            expected_h1a1 = 20 * np.log10(harmonic) if harmonic and harmonic > 1 else None
            records.append({"configuration": i, "variant": name, "true_F0": f0, "true_F1": f1, "true_F2": f2,
                "true_B1": b1, "phase_cycles": phase, "F1_to_F0": f1 / f0,
                "estimated_F0": estimate_f0,
                "estimated_formants": [float(v) if np.isfinite(v) else None for v in estimate_f],
                "estimated_bandwidths": [float(v) if np.isfinite(v) else None for v in estimate_b],
                "measurement": h, "source_H1_H2_db": float(expected_h1h2),
                "source_H1_A1_db": float(expected_h1a1) if expected_h1a1 is not None else None,
                "h1h2_corrected_error_db": float(h["h1h2_corrected"] - expected_h1h2) if h and h["h1h2_corrected"] is not None else None,
                "h1a1_corrected_error_db": float(h["h1a1_corrected"] - expected_h1a1) if h and h["h1a1_corrected"] is not None and expected_h1a1 is not None else None})
        if hashlib.sha256(x.tobytes()).hexdigest() != before:
            raise RuntimeError("measurement modified synthetic PCM")
        if (i + 1) % 50 == 0:
            print("synthetic configurations %d/%d" % (i + 1, len(grid)), flush=True)
    variants = defaultdict(list)
    for r in records:
        variants[r["variant"]].append(r)
    metrics = ("h1h2_corrected_error_db", "h1a1_corrected_error_db")
    result = {"status": "synthetic_engineering_validation_not_speech_accuracy", "sample_rate": sr,
        "source": "stationary noiseless sum of sine harmonics with amplitude 1/k through two known normalized digital resonators",
        "configurations": len(grid), "measurements": len(records), "external": external,
        "by_variant": {name: {metric: describe_errors(rows, metric) for metric in metrics} for name, rows in variants.items()},
        "oracle_by_F1_to_F0": {name: {metric: describe_errors([r for r in variants["oracle"] if select(r["F1_to_F0"])], metric) for metric in metrics}
            for name, select in [("below_1.5", lambda x: x < 1.5), ("1.5_to_3", lambda x: 1.5 <= x < 3), ("at_least_3", lambda x: x >= 3)]},
        "limitations": ["No Korean acoustic realism or classification performance follows from this experiment.",
            "The correction formula and synthetic resonators share their mathematical model; this validates implementation under that model.",
            "No noise, irregular voice, changing formants, nasal coupling, or channel filtering is modeled.",
            "The two-resonator source differs from natural speech; five-formant LPC has a deliberate model-order mismatch, and two-formant LPC is a separate matched-order diagnostic.",
            "Deliberate parameter perturbations are sensitivity scenarios, not observed estimation-error distributions.",
            "The source spectrum is fixed; changing F0 changes the number and spacing of harmonics."],
        "provenance": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_acoustics.py", correction_path]},
        "records": records}
    a.out.mkdir(parents=True)
    (a.out / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for p in [Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_acoustics.py"]:
        (a.out / p.name).write_bytes(p.read_bytes())
    print(json.dumps({"configurations": len(grid), "by_variant": result["by_variant"], "oracle_by_F1_to_F0": result["oracle_by_F1_to_F0"]}, indent=2))


if __name__ == "__main__":
    main()
