"""Offline candidate quality gate: separate outer silence from active span.

Energy activity is not a speech classifier. Keeping the duration requirement
does not establish rejection of every short noise burst. This module is not
registered by the application; evaluate noise acceptance before deployment.
"""
import numpy as np

from mdd.vad import FRAME_SEC, KEEP_SEC, voiced_mask


def assess_span(wav, sr, min_seconds, min_ratio=.30, keep=KEEP_SEC):
    x = np.asarray(wav)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("finite mono input required")
    if min_seconds <= 0 or not 0 <= min_ratio <= 1 or keep < 0:
        raise ValueError("invalid gate parameters")
    mask, frame = voiced_mask(x, sr)
    seconds = float(mask.sum()) * FRAME_SEC
    recording_ratio = float(mask.mean()) if len(mask) else 0.
    indices = np.flatnonzero(mask)
    lo, hi, span_ratio = 0, len(mask), 0.
    if len(indices):
        padding = int(keep / FRAME_SEC)
        lo, hi = max(0, int(indices[0]) - padding), min(len(mask), int(indices[-1]) + padding + 1)
        span_ratio = float(mask[lo:hi].mean())
    accepted = seconds >= min_seconds and span_ratio >= min_ratio
    return {"accepted": bool(accepted), "active_seconds": seconds,
            "recording_ratio": recording_ratio, "span_ratio": span_ratio,
            "span_samples": [lo * frame, hi * frame], "min_seconds": min_seconds,
            "min_ratio": min_ratio, "measurement": "energy activity, not independently verified speech"}
