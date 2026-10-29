"""Time-resolved acoustic observations, separate from phoneme classification.

All time axes refer to the unmodified input PCM. A CTC posterior peak or a
pitch track is never exported as a validated acoustic boundary. Harmonic
measurements remain missing for unvoiced/unsupported windows.
"""
import math

import numpy as np
from scipy import signal


def frame_signal(x, sr, window_seconds=.025, step_seconds=.005):
    n = int(round(window_seconds * sr))
    step = max(1, int(round(step_seconds * sr)))
    if len(x) < n:
        return np.empty(0), np.empty((0, n))
    starts = np.arange(0, len(x) - n + 1, step)
    frames = np.asarray(x)[starts[:, None] + np.arange(n)]
    return (starts + (n - 1) / 2) / sr, frames


def frame_tracks(x, sr=16000):
    """25ms spectra and 40ms periodicity, sampled at 5ms; no boundary decision."""
    times, frames = frame_signal(x, sr)
    if not len(times):
        raise ValueError("at least 25ms of PCM is required")
    frames = frames - frames.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), 1024)) ** 2
    freq = np.fft.rfftfreq(1024, 1. / sr)
    energy = np.mean(frames ** 2, axis=1)
    bands = {}
    for name, lo, hi in (("low", 100, 1500), ("mid", 1500, 3000), ("high", 3000, min(7900, sr / 2))):
        bands[name] = spec[:, (freq >= lo) & (freq < hi)].sum(axis=1)
    sel = (freq >= 1000) & (freq < min(7900, sr / 2))
    p = spec[:, sel] + 1e-20
    centroid = np.sum(p * freq[sel], axis=1) / np.sum(p, axis=1)
    flatness = np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)
    centroid[energy < 1e-10] = np.nan
    flatness[energy < 1e-10] = np.nan
    pt, pf = frame_signal(x, sr, .040, .005)
    periodicity = np.full(len(times), np.nan)
    peak_pitch = np.full(len(times), np.nan)
    if len(pt):
        pf = pf - pf.mean(axis=1, keepdims=True)
        n = pf.shape[1]
        nfft = 1 << (2 * n - 1).bit_length()
        corr = np.fft.irfft(np.abs(np.fft.rfft(pf, nfft)) ** 2, nfft)[:, :n]
        cumul = np.c_[np.zeros(len(pf)), np.cumsum(pf ** 2, axis=1)]
        lags = np.arange(int(sr / 500), min(int(sr / 70) + 1, n))
        den = np.sqrt(np.maximum(cumul[:, n - lags] * (cumul[:, n, None] - cumul[:, lags]), 1e-30))
        ac = corr[:, lags] / den
        ii = ac.argmax(axis=1)
        peaks = ac[np.arange(len(pf)), ii]
        pitches = sr / lags[ii]
        silent = np.mean(pf ** 2, axis=1) < 1e-10
        peaks[silent], pitches[silent] = np.nan, np.nan
        periodicity = np.interp(times, pt, peaks, left=np.nan, right=np.nan)
        peak_pitch = np.interp(times, pt, pitches, left=np.nan, right=np.nan)
    return {"times": times, "rms_db": 10 * np.log10(energy + 1e-20),
            "low_db": 10 * np.log10(bands["low"] + 1e-20),
            "mid_db": 10 * np.log10(bands["mid"] + 1e-20),
            "high_db": 10 * np.log10(bands["high"] + 1e-20),
            "high_low_db": 10 * np.log10((bands["high"] + 1e-20) / (bands["low"] + 1e-20)),
            "centroid_hz": centroid, "flatness": flatness,
            # Maximum ACF lag can select a subharmonic. This is NOT an F0
            # estimator and must not be substituted for the Praat pitch track.
            "periodicity": periodicity, "acf_max_lag_frequency_hz_unvalidated": peak_pitch}


def partition_balance_tracks(x, sr=16000, cutoffs=(2000, 3000, 4000)):
    """Complete upper/lower spectral partitions for inspecting noise transitions.

    Ten-ms windows resolve short transitions better than the general 25-ms
    tracks. The cutoff is deliberately not a universal classifier: aspiration
    follows vocal-tract resonances, which change with the following vowel.
    Unlike high_low_db, each balance includes the entire 100Hz-to-Nyquist-minus
    edge band. Silence is missing, not a balanced aspiration observation.
    """
    times, frames = frame_signal(x, sr, .010, .002)
    if not len(times):
        raise ValueError("at least 10ms of PCM is required")
    frames = frames - frames.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), 1024)) ** 2
    frequencies = np.fft.rfftfreq(1024, 1. / sr)
    top = min(7900, sr / 2)
    silent = np.mean(frames ** 2, axis=1) < 1e-10
    result = {"times": times}
    for cutoff in cutoffs:
        if not 100 < cutoff < top:
            raise ValueError("cutoff must be inside the observed frequency band")
        lower = power[:, (frequencies >= 100) & (frequencies < cutoff)].sum(axis=1)
        upper = power[:, (frequencies >= cutoff) & (frequencies < top)].sum(axis=1)
        balance = 10 * np.log10((upper + 1e-20) / (lower + 1e-20))
        balance[silent] = np.nan
        result["upper_lower_db_%d" % cutoff] = balance
    return result


def spectrum(x, sr, window_seconds=.012, step_seconds=.002):
    n = int(round(window_seconds * sr))
    step = max(1, int(round(step_seconds * sr)))
    f, t, s = signal.spectrogram(x, sr, window="hann", nperseg=n,
                                 noverlap=n - step, nfft=1024, detrend="constant",
                                 scaling="density", mode="psd")
    return f, t, 10 * np.log10(s + 1e-20)


def harmonic_amplitudes(x, sr, center, f0, formants=None, bandwidths=None, correction=None):
    """Three-period Hann window; harmonic peaks constrained to their own bins.

    Formant-corrected outputs use the supplied, externally versioned
    OpenSauce correction function and F1/F2 with measured bandwidths.
    This is not a claim of exact VoiceSauce implementation equivalence.
    """
    if not np.isfinite(f0) or not 70 <= f0 <= 500:
        return None
    n = max(64, int(round(3 * sr / f0)))
    start = int(round(center * sr - n / 2))
    if start < 0 or start + n > len(x):
        return None
    # np.asarray may return a view when x already has float dtype. Center a
    # private copy: overlapping windows must never change the source signal.
    seg = np.array(x[start:start + n], dtype=float, copy=True)
    seg -= seg.mean()
    if np.mean(seg ** 2) < 1e-10:
        return None
    nfft = max(8192, 1 << n.bit_length())
    amplitude = np.abs(np.fft.rfft(seg * np.hanning(n), nfft))
    frequencies = np.fft.rfftfreq(nfft, 1. / sr)

    def peak(harmonic):
        wanted = harmonic * f0
        sel = np.where(np.abs(frequencies - wanted) <= .20 * f0)[0]
        if not len(sel):
            return None
        idx = sel[np.argmax(amplitude[sel])]
        return float(frequencies[idx]), float(20 * np.log10(amplitude[idx] + 1e-20))

    h1, h2 = peak(1), peak(2)
    if h1 is None or h2 is None:
        return None
    out = {"window_start": start / sr, "window_end": (start + n) / sr,
           "f0": float(f0), "h1_hz": h1[0], "h2_hz": h2[0],
           "h1h2_raw": h1[1] - h2[1], "h1a1_raw": None,
           "h1h2_corrected": None, "h1a1_corrected": None,
           "a1_harmonic": None, "formant_quality": "not_estimated"}
    if formants is None or bandwidths is None:
        return out
    ff, bb = np.asarray(formants, float), np.asarray(bandwidths, float)
    if len(ff) < 2 or len(bb) < 2 or not np.all(np.isfinite(ff[:2])) or not np.all(np.isfinite(bb[:2])):
        out["formant_quality"] = "missing"
        return out
    if not (150 <= ff[0] <= 1500 and ff[0] + 100 < ff[1] <= 4000 and np.all((bb[:2] >= 20) & (bb[:2] <= 800))):
        out["formant_quality"] = "implausible"
        return out
    harmonic = max(1, int(math.floor(ff[0] / f0 + .5)))
    a1 = peak(harmonic)
    if a1 is None:
        return out
    out.update(a1_harmonic=harmonic, h1a1_raw=h1[1] - a1[1],
               formant_quality="h1_equals_a1" if harmonic == 1 else "automatically_plausible_unverified")
    if correction is not None:
        def corrected(value):
            return value[1] - sum(float(correction(value[0], ff[k], bb[k], sr)) for k in range(2))
        out["h1h2_corrected"] = corrected(h1) - corrected(h2)
        # H1 and A1 can be the very same harmonic in high vowels. Keep this
        # contrast missing, rather than presenting its algebraic zero as evidence.
        if harmonic > 1:
            out["h1a1_corrected"] = corrected(h1) - corrected(a1)
    return out


def validate_landmarks(row, duration):
    """Validate reviewed observations, allowing frication/voicing overlap."""
    names = ("frication_start", "frication_end", "voicing_first", "voicing_stable")
    values = {k: row.get(k) for k in names}
    for k, v in values.items():
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= duration):
            raise ValueError("invalid landmark " + k)
    if values["frication_start"] is not None and values["frication_end"] is not None and values["frication_start"] > values["frication_end"]:
        raise ValueError("frication ends before it starts")
    if values["voicing_first"] is not None and values["voicing_stable"] is not None and values["voicing_first"] > values["voicing_stable"]:
        raise ValueError("stable voicing precedes first voicing")
    if row.get("actor_type") not in ("assistant_visual", "human_expert", "automatic_proxy", "synthetic_test"):
        raise ValueError("annotation provenance is required")
    if row.get("transition_kind") not in ("aspiration", "silence", "mixed", "overlap", "none", "uncertain"):
        raise ValueError("transition interpretation is required")
    return values
