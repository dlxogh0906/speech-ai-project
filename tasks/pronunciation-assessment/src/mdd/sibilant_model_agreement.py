"""Confirm actual-phone identity across two frozen heads, without score mixing."""
import numpy as np


def confirmed_availability(primary_probabilities, primary_available,
                           confirmation_probabilities, confirmation_available):
    a, b = np.asarray(primary_probabilities), np.asarray(confirmation_probabilities)
    av, bv = np.asarray(primary_available, dtype=bool), np.asarray(confirmation_available, dtype=bool)
    if a.ndim != 2 or a.shape[1] != 4 or a.shape != b.shape or av.shape != (len(a),) or bv.shape != av.shape:
        raise ValueError("identically ordered four-class observations required")
    if any(not np.isfinite(x).all() or (x < 0).any() or (x > 1).any() for x in (a, b)):
        raise ValueError("finite model probabilities required")
    actual, confirm = a.argmax(1), b.argmax(1)
    return av & bv & (actual < 2) & (actual == confirm)
