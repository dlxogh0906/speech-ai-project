"""Diagnostic pooling mass against known TAG roles; never an inference gate."""
import numpy as np


def support_measurements(probabilities, roles, observed):
    p = np.asarray(probabilities, dtype=np.float64)
    roles, observed = np.asarray(roles), np.asarray(observed, dtype=bool)
    if p.shape != (len(roles), 4) or observed.shape != roles.shape or not observed.any():
        raise ValueError("observed frame grid and four role probabilities required")
    if not np.isfinite(p[observed]).all():
        raise ValueError("nonfinite observed probabilities")
    counts = [int(np.sum(observed & (roles == k))) for k in (1, 2, 3)]
    peaks, masses, gated, squared = [], [], [], []
    for k in (1, 2, 3):
        weights = np.where(observed, p[:, k], 0.)
        if weights.sum() <= 0:
            raise ValueError("empty soft support")
        peak = int(np.argmax(weights))
        peaks.append(None if roles[peak] < 0 else bool(roles[peak] == k))
        own = roles == k
        unknown = observed & (roles < 0)
        masses.append({"own_known": float(weights[own].sum() / weights.sum()),
                       "unknown": float(weights[unknown].sum() / weights.sum())})
        for result, alternative in ((gated, np.maximum(weights - .5, 0.)), (squared, weights ** 2)):
            result.append(float(alternative[own].sum() / alternative.sum()) if alternative.sum() > 0 else None)
    known = observed & (roles >= 0)
    prediction = p.argmax(-1)
    confusion = np.zeros((4, 4), np.int64)
    np.add.at(confusion, (roles[known], prediction[known]), 1)
    return {"role_frames": counts, "complete_observed_roles": all(counts), "peak_hits": peaks,
            "soft_mass": masses, "confident_mass": gated, "squared_mass": squared,
            "confusion": confusion.tolist()}
