"""Runtime-input-only observations for an offline sibilant transition pilot.

The local window contains context; its anchor is NOT a physical boundary.
ACF is periodicity evidence, not F0 or verified glottal voicing. No actual
transcription, expert boundary, or expected s/ss identity is a head input.
"""
import copy
import math

import numpy as np

VOWELS = tuple("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
PAIR = {"ㅅ": "ㅆ", "ㅆ": "ㅅ"}
ACOUSTIC_NAMES = ("relative_rms_db", "high_low_db", "centroid_khz", "flatness",
                  "acf_strength_unvalidated") + tuple("band_fraction_db_%d" % i for i in range(8))


def paired_anchor(lp, canonical_ids, q, pair_ids, blank):
    """Symmetric localization under both hypotheses, never actual-phone labels."""
    from mdd.gop_feats_v2 import viterbi_align
    if not (0 <= q < len(canonical_ids)) or canonical_ids[q] not in pair_ids:
        raise ValueError("invalid sibilant target")
    anchors, sequences, segments = [], [], []
    for token_id in pair_ids:
        ids = list(canonical_ids)
        ids[q] = token_id
        seg = viterbi_align(lp, ids, blank=blank)
        if seg is None or not seg[q]:
            return None
        anchors.append(float(np.mean(seg[q])))
        sequences.append(ids)
        segments.append(seg[q])
    return {"frame": int(math.floor(np.mean(anchors) + .5)),
            "hypothesis_frames": anchors, "sequences": sequences, "segments": segments}


def acoustic_tracks(wav, sr=16000):
    """Reuse existing tracks; append eight coarse spectral-envelope bands."""
    from mdd.sibilant_acoustics import frame_tracks, frame_signal
    x = np.asarray(wav, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or sr != 16000:
        raise ValueError("finite mono 16kHz input required")
    tracks = frame_tracks(x, sr)
    times, frames = frame_signal(x, sr)
    frames = frames - frames.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), 1024)) ** 2
    freq = np.fft.rfftfreq(1024, 1. / sr)
    edges = (100, 500, 1000, 1500, 2200, 3000, 4000, 5500, 7900)
    bands = np.stack([power[:, (freq >= lo) & (freq < hi)].sum(1)
                      for lo, hi in zip(edges[:-1], edges[1:])], axis=1)
    relative = 10 * np.log10((bands + 1e-20) / (bands.sum(1, keepdims=True) + 1e-20))
    relative[tracks["rms_db"] < -100] = np.nan
    # Absolute recording level is removed from the energy trajectory. Other
    # spectra are relative powers, not calibrated sound pressure or formants.
    rms = tracks["rms_db"] - np.percentile(tracks["rms_db"], 90)
    values = np.column_stack([rms, tracks["high_low_db"], tracks["centroid_hz"] / 1000,
                              tracks["flatness"], tracks["periodicity"], relative])
    values[tracks["rms_db"] < -100, 1:] = np.nan
    return times, values


def observation_window(wav, lp, canon, q, token_index, blank, tracks=None, radius=15):
    """Build 31 chronological frames around a symmetric CTC hypothesis anchor.

    Following-vowel/previous-position context comes from runtime canonical.
    Actual vowel identity is not supplied. Frame times are nominal model-grid
    times on VAD-processed audio; no expert event or support attribution exists.
    """
    if q < 0 or q + 1 >= len(canon) or canon[q] not in PAIR or canon[q + 1] not in VOWELS:
        raise ValueError("target must be a runtime s/ss onset")
    lp = np.asarray(lp)
    if lp.ndim != 2 or not len(lp) or not np.isfinite(lp).all():
        raise ValueError("finite frame log probabilities required")
    pair_ids = [token_index("ㅅ"), token_index("ㅆ")]
    anchor = paired_anchor(lp, [token_index(t) for t in canon], q, pair_ids, blank)
    if anchor is None:
        return None
    indices = np.arange(anchor["frame"] - radius, anchor["frame"] + radius + 1)
    valid = (indices >= 0) & (indices < len(lp))
    posterior = np.zeros((len(indices), lp.shape[1]), np.float32)
    posterior[valid] = np.clip(lp[indices[valid]], -20, 0) / 20
    dt = len(wav) / (16000. * len(lp))
    centers = (indices + .5) * dt
    times, values = tracks if tracks is not None else acoustic_tracks(wav)
    acoustic = np.full((len(indices), len(ACOUSTIC_NAMES)), np.nan, np.float32)
    # Nearest 5ms observation; no interpolation through missing measurements.
    ix = np.searchsorted(times, centers)
    right = np.minimum(ix, len(times) - 1)
    left = np.maximum(ix - 1, 0)
    near = np.where(abs(times[right] - centers) < abs(times[left] - centers), right, left)
    contained = valid & (centers >= .020) & (centers <= len(wav) / 16000. - .020)
    acoustic[contained] = values[near[contained]]
    context = np.zeros(len(VOWELS) + 3, np.float32)
    context[VOWELS.index(canon[q + 1])] = 1
    position = 0 if q == 0 else (1 if canon[q - 1] in VOWELS else 2)
    context[len(VOWELS) + position] = 1
    return {"posterior": posterior, "acoustic": acoustic, "mask": valid,
            "context": context, "anchor": anchor, "frame_seconds": dt,
            "position": ("initial", "post_vowel", "post_consonant")[position],
            "following": canon[q + 1]}


def fit_acoustic_normalization(arrays):
    """Training input only; never estimate statistics from calibration/test."""
    values = np.concatenate(arrays, axis=0).astype(np.float64)
    counts = np.isfinite(values).sum(0)
    means = np.nansum(values, axis=0) / np.maximum(counts, 1)
    var = np.nansum((values - means) ** 2, axis=0) / np.maximum(counts, 1)
    scales = np.maximum(np.sqrt(var), 1e-3)
    return {"mean": means.tolist(), "scale": scales.tolist(), "finite_counts": counts.tolist()}


def normalize_acoustics(values, normalization):
    mask = np.isfinite(values)
    z = (values - np.asarray(normalization["mean"])) / np.asarray(normalization["scale"])
    z = np.where(mask, np.clip(z, -8, 8), 0).astype(np.float32)
    return np.concatenate([z, mask.astype(np.float32)], axis=-1)


def annotate_probabilities(canon, base_sites, probabilities, thresholds):
    """Formatter adapter; preserve third-phone substitutions and deletions.

    probabilities maps runtime index to P(actual ss), not a calibrated error
    probability. thresholds are cal-only decision thresholds or None (off).
    """
    if len(canon) != len(base_sites):
        raise ValueError("canonical/site length mismatch")
    result = copy.deepcopy(base_sites)
    for q, site in enumerate(base_sites):
        if site.get("i") != q or site.get("canon") != canon[q] or "sz" in site:
            raise ValueError("invalid or already annotated baseline")
    for q, probability in probabilities.items():
        if not isinstance(q, int) or not 0 <= q < len(canon):
            raise ValueError("invalid probability index")
        p = float(probability)
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid actual-phone probability")
        token = canon[q]
        if token not in PAIR or q + 1 >= len(canon) or canon[q + 1] not in VOWELS:
            continue
        threshold = thresholds.get(token)
        if threshold is None or base_sites[q].get("pred") not in PAIR:
            continue
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("invalid threshold")
        error_score = p if token == "ㅅ" else 1 - p
        if error_score >= threshold:
            result[q]["sz"] = {"alt": PAIR[token], "p": error_score, "th": threshold}
    return result
