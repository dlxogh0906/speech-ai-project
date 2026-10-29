"""Offline noise/periodicity observations with explicit temporal support.

Returned pitch is a periodicity candidate, never a validated glottal source,
acoustic boundary, pronunciation label, or permission to compute voice quality.
The finite low-pass branch is for candidate comparison; spectral observations
always use the original PCM. No production caller or learned threshold is used.
"""
import math

import numpy as np
from scipy import signal


def _waveform(x, sr):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite, nonempty mono PCM is required")
    if isinstance(sr, bool) or not isinstance(sr, (int, float)) or not math.isfinite(sr) or sr < 8000:
        raise ValueError("sample rate must be finite and at least 8000Hz")
    return x


def finite_low_band(x, sr, cutoff=1200., filter_seconds=.005):
    """Symmetric FIR with a finite, reported support; no source mutation."""
    x = _waveform(x, sr)
    if not math.isfinite(cutoff) or not 0 < cutoff < sr / 2:
        raise ValueError("cutoff must be inside Nyquist")
    if not math.isfinite(filter_seconds) or not .001 <= filter_seconds <= .1:
        raise ValueError("filter duration must be between 1 and 100ms")
    taps = max(3, int(round(filter_seconds * sr)))
    taps += 1 - taps % 2
    kernel = signal.firwin(taps, cutoff, fs=sr, window="hann")
    filtered = signal.convolve(x, kernel, mode="same", method="direct")
    return filtered, {"method": "symmetric_hann_FIR", "taps": taps,
                      "cutoff_hz": cutoff, "half_support_s": (taps - 1) / (2 * sr),
                      "edge_handling": "zero extension; edge-dependent frames are not locally contained"}


def noise_observations(x, sr, window_seconds=.010, step_seconds=.002):
    """Raw short-time energy/spectra; no noise-class or boundary decisions."""
    x = _waveform(x, sr)
    if not all(math.isfinite(v) and v > 0 for v in (window_seconds, step_seconds)):
        raise ValueError("positive finite window and step required")
    n, step = max(2, round(window_seconds * sr)), max(1, round(step_seconds * sr))
    nfft = max(1024, 1 << (n - 1).bit_length())
    freq = np.fft.rfftfreq(nfft, 1. / sr)
    top = min(7900., sr / 2)
    rows = []
    for start in range(0, len(x) - n + 1, step):
        frame = x[start:start + n].copy()
        frame -= frame.mean()
        energy = float(np.mean(frame ** 2))
        power = np.abs(np.fft.rfft(frame * np.hanning(n), nfft)) ** 2
        bands = {}
        for name, lo, hi in (("low", 100., 1500.), ("mid", 1500., 3000.), ("high", 3000., top)):
            bands[name] = float(power[(freq >= lo) & (freq < hi)].sum())
        sel = (freq >= 1000.) & (freq < top)
        denominator = float(power[sel].sum())
        observed = energy >= 1e-10 and denominator > 0
        rows.append({"center_s": (start + (n - 1) / 2) / sr,
                     "support_s": [start / sr, (start + n) / sr],
                     "rms_db": 10 * math.log10(energy + 1e-20),
                     "band_power": bands,
                     "centroid_1000_to_top_hz": float(np.dot(freq[sel], power[sel]) / denominator) if observed else None,
                     "upper_lower_3000_db": 10 * math.log10((bands["high"] + 1e-20) / (bands["low"] + bands["mid"] + 1e-20)) if observed else None})
    return {"window_samples": n, "step_samples": step, "fft_samples": nfft,
            "spectral_top_hz": top, "waveform": "original_PCM",
            "rms_reference": "input PCM units, not calibrated sound pressure",
            "band_power_units": "unnormalized squared Hann-windowed DFT magnitude",
            "spectral_energy_floor_pcm_squared": 1e-10, "frames": rows}


def periodicity_observations(x, sr, floors=(50., 70., 100.), ceiling=500., step=.005):
    """Paired raw/low-band CC candidates; local support includes FIR support.

    The CC support rule follows the audited Praat 6.1.38 implementation.
    Global amplitude normalization and path selection still depend on the full
    clip. All local candidates are retained, including unselected ones. A second
    path with both transition penalties zero diagnoses path dependence; it is
    not a corrected pitch estimate. No frame is declared glottally voiced.
    """
    import parselmouth
    x = _waveform(x, sr)
    floors = tuple(float(f) for f in floors)
    if not floors or len(set(floors)) != len(floors) or not all(math.isfinite(f) and 20 <= f < ceiling for f in floors):
        raise ValueError("distinct finite pitch floors >=20Hz and below ceiling required")
    if not math.isfinite(ceiling) or not max(floors) < ceiling < sr / 2 or not math.isfinite(step) or step <= 0:
        raise ValueError("invalid ceiling or step")
    if parselmouth.PRAAT_VERSION != "6.1.38":
        raise ValueError("local support is audited for Praat 6.1.38 only; audit a different version before use")
    low, filter_info = finite_low_band(x, sr)
    settings = []
    for floor in floors:
        half = 1. / floor + 2. / sr
        paired_half = half + filter_info["half_support_s"]
        result = {"floor_hz": floor, "raw_cc_half_support_s": half,
                  "paired_half_support_s": paired_half, "frames": []}
        # Conservative context requirement, distinct from interval eligibility.
        if len(x) / sr < 3. / floor:
            result["status"] = "insufficient_clip_context"
            settings.append(result)
            continue
        sounds = [parselmouth.Sound(y, sampling_frequency=sr) for y in (x, low)]
        pitches = [sound.to_pitch_cc(time_step=step, pitch_floor=floor, pitch_ceiling=ceiling)
                   for sound in sounds]
        no_transition = [sound.to_pitch_cc(time_step=step, pitch_floor=floor, pitch_ceiling=ceiling,
                          octave_jump_cost=0., voiced_unvoiced_cost=0.) for sound in sounds]
        times = pitches[0].xs()
        if not all(np.array_equal(times, pitch.xs()) for pitch in pitches + no_transition):
            raise RuntimeError("raw/filtered/path-diagnostic frame grids differ")
        arrays = [pitch.selected_array for pitch in pitches]
        independent_arrays = [pitch.selected_array for pitch in no_transition]
        for i, t in enumerate(times):
            frame = {"center_s": float(t), "raw_cc_support_s": [float(t-half), float(t+half)],
                     "paired_support_s": [float(t-paired_half), float(t+paired_half)]}
            for name, selected, independent, pitch, diagnostic in zip(
                    ("raw", "low_band"), arrays, independent_arrays, pitches, no_transition):
                candidates = [(float(c.frequency), float(c.strength)) for c in pitch.get_frame(i + 1).candidates]
                alternatives = [(float(c.frequency), float(c.strength)) for c in diagnostic.get_frame(i + 1).candidates]
                # A rejected Praat candidate can have nonfinite strength even
                # when source PCM and the selected path are finite. Treat the
                # same nonfinite value as identical for the control comparison,
                # but retain its invalid state instead of exporting JSON NaN.
                signatures = lambda rows: sorted((repr(f), repr(s)) for f, s in rows)
                if signatures(candidates) != signatures(alternatives):
                    raise RuntimeError("path comparison changed local candidates")
                chosen = float(selected["frequency"][i]) or None
                independent_hz = float(independent["frequency"][i]) or None
                ratio = math.log2(independent_hz / chosen) if chosen and independent_hz else None
                frame[name] = {"candidate_hz": chosen, "strength": float(selected["strength"][i]),
                    "local_candidates": [{"frequency_hz": f if math.isfinite(f) and f else None,
                        "strength": strength if math.isfinite(strength) else None,
                        "nonfinite_fields": [field for field, value in (("frequency_hz", f), ("strength", strength)) if not math.isfinite(value)],
                        "kind": "periodicity_candidate" if f else "unvoiced_path_option",
                        "within_requested_frequency_range": floor <= f <= ceiling if f and math.isfinite(f) else None,
                        "selected_by_default_path": k == 0} for k, (f, strength) in enumerate(candidates)],
                    "path_comparison": {"no_transition_cost_candidate_hz": independent_hz,
                        "no_transition_cost_strength": float(independent["strength"][i]),
                        "local_candidates_identical": True, "selected_candidate_changed": chosen != independent_hz,
                        "log2_ratio": ratio, "near_octave_review_flag": abs(abs(ratio) - 1.) <= .12 if ratio is not None else False}}
            frame["both_returned"] = all(frame[k]["candidate_hz"] is not None for k in ("raw", "low_band"))
            frame["low_to_raw_log2_ratio"] = math.log2(frame["low_band"]["candidate_hz"] / frame["raw"]["candidate_hz"]) if frame["both_returned"] else None
            result["frames"].append(frame)
        result["status"] = "candidates_only"
        settings.append(result)
    return {"schema_version": 2, "parselmouth_version": parselmouth.__version__, "praat_version": parselmouth.PRAAT_VERSION,
            "method": "raw_cross_correlation_on_original_and_finite_low_band",
            "filter": filter_info, "time_step_s": step, "ceiling_hz": ceiling,
            "clip_duration_s": len(x) / sr,
            "settings": settings,
            "path_diagnostic": {"default_octave_jump_cost": .35, "default_voiced_unvoiced_cost": .14,
                "comparison_octave_jump_cost": 0., "comparison_voiced_unvoiced_cost": 0.,
                "unchanged_octave_cost": .01, "local_candidate_equality_checked": True,
                "interpretation": "selection sensitivity only; neither path validates a glottal F0"},
            "scope": "periodicity_candidates_not_glottal_source_or_pronunciation_labels",
            "full_clip_dependence": ["global_peak_normalization", "dynamic_candidate_path"],
            "glottal_source_validated": False}


def interval_relation(start, end, reference_start, reference_end, provenance):
    """Report time outside an existing label span without changing either span.

    Being inside a reference label does not establish true phone containment;
    references can have boundary errors, coupled phones, or the wrong target.
    """
    values = (start, end, reference_start, reference_end)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        raise ValueError("finite interval endpoints required")
    if not (0 <= start < end and 0 <= reference_start < reference_end):
        raise ValueError("valid nonempty intervals required")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("reference interval provenance required")
    overlap = max(0., min(end, reference_end) - max(start, reference_start))
    before = max(0., min(end, reference_start) - start)
    after = max(0., end - max(start, reference_end))
    return {"reference_interval_s": [reference_start, reference_end], "reference_provenance": provenance,
            "overlap_s": overlap, "before_reference_s": before, "after_reference_s": after,
            "within_reference_interval": before <= 1e-9 and after <= 1e-9,
            "acoustic_phone_containment_verified": False}


def interval_observations(periodicity, start, end, provenance):
    """Describe candidates contained in a supplied interval, without shifting it.

    The same paired support rule is applied to both branches. Missingness is
    an observation, never a normal/error decision. Floor comparisons use the
    nearest frame within half a hop and preserve the actual time difference.
    """
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (start, end)) or not 0 <= start < end:
        raise ValueError("valid positive interval required")
    if end > periodicity["clip_duration_s"]:
        raise ValueError("interval extends beyond analyzed PCM")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("interval provenance required")
    settings, selected = [], {}
    for setting in periodicity["settings"]:
        frames = [f for f in setting["frames"] if f["paired_support_s"][0] >= start and f["paired_support_s"][1] <= end]
        selected[setting["floor_hz"]] = frames
        row = {"floor_hz": setting["floor_hz"], "supported_frames": len(frames), "branches": {}}
        for name in ("raw", "low_band"):
            values = [f[name]["candidate_hz"] for f in frames if f[name]["candidate_hz"] is not None]
            row["branches"][name] = {"returned_frames": len(values),
                "state": "no_locally_supported_frame" if not frames else "candidate_returned" if values else "no_candidate_returned",
                "median_candidate_hz": float(np.median(values)) if values else None,
                "path_choice_disagreement_frames": sum(f[name].get("path_comparison", {}).get("selected_candidate_changed", False) for f in frames),
                "path_choice_octave_review_frames": sum(f[name].get("path_comparison", {}).get("near_octave_review_flag", False) for f in frames)}
        row["branch_return_disagreements"] = sum((f["raw"]["candidate_hz"] is None) != (f["low_band"]["candidate_hz"] is None) for f in frames)
        settings.append(row)
    comparisons = []
    floors = sorted(selected)
    for i, a in enumerate(floors):
        for b in floors[i+1:]:
            for left in selected[a]:
                if not selected[b]:
                    continue
                right = min(selected[b], key=lambda f: abs(f["center_s"] - left["center_s"]))
                delta = right["center_s"] - left["center_s"]
                if abs(delta) > periodicity["time_step_s"] / 2 + 1e-10:
                    continue
                for branch in ("raw", "low_band"):
                    f1, f2 = left[branch]["candidate_hz"], right[branch]["candidate_hz"]
                    if f1 is None or f2 is None:
                        continue
                    ratio = math.log2(f2 / f1)
                    comparisons.append({"floors_hz": [a, b], "branch": branch,
                        "center_s": left["center_s"], "other_center_offset_s": delta,
                        "candidate_hz": [f1, f2], "log2_ratio": ratio,
                        "near_octave_review_flag": abs(abs(ratio) - 1.) <= .12})
    return {"interval_s": [start, end], "interval_provenance": provenance,
            "settings": settings, "cross_floor_comparisons": comparisons,
            "octave_flag_rule": "abs(abs(log2_ratio)-1) <= 0.12; review heuristic, not a validated error detector",
            "glottal_source_validated": False, "pronunciation_decision": None}
