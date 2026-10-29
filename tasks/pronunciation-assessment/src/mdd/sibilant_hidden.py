"""Runtime-only hidden trajectories for a frozen sibilant pilot.

Windows are model-context observations, not independently verified phonetic
boundaries. Projection is fitted on training frames only. No serving default
enables this candidate.
"""
import math

import numpy as np

from mdd.sibilant_transition import PAIR


def hidden_window(hidden, anchor, radius=15):
    hidden = np.asarray(hidden)
    if hidden.ndim != 2 or not len(hidden) or not np.isfinite(hidden).all():
        raise ValueError("finite frame-by-channel hidden states required")
    if not isinstance(anchor, int) or not 0 <= anchor < len(hidden) or radius < 0:
        raise ValueError("invalid runtime anchor")
    frames = np.arange(anchor - radius, anchor + radius + 1)
    mask = (frames >= 0) & (frames < len(hidden))
    output = np.zeros((len(frames), hidden.shape[1]), dtype=hidden.dtype)
    output[mask] = hidden[frames[mask]]
    return output, mask


def fit_projection(training, mask, components=64, seed=20261024):
    from sklearn.decomposition import PCA
    values = np.asarray(training)[np.asarray(mask, bool)].astype(np.float32)
    if not np.isfinite(values).all() or min(values.shape) < components:
        raise ValueError("insufficient finite training observations")
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    pca.fit(values)
    return {"mean": pca.mean_, "components": pca.components_,
            "scale": np.maximum(np.sqrt(pca.explained_variance_), 1e-3),
            "training_frames": len(values), "explained_variance_ratio": pca.explained_variance_ratio_}


def project_sequences(values, mask, projection, pooled=False):
    values, mask = np.asarray(values), np.asarray(mask, bool)
    if values.shape[:2] != mask.shape or not mask.any(axis=1).all():
        raise ValueError("nonempty mask must match each observation window")
    output = np.zeros(values.shape[:2] + (len(projection["components"]),), np.float32)
    for start in range(0, len(values), 256):
        batch = np.asarray(values[start:start + 256], np.float32)
        x = ((batch - projection["mean"]) @ projection["components"].T) / projection["scale"]
        valid = mask[start:start + 256, :, None]
        if not np.isfinite(x[valid[..., 0]]).all():
            raise ValueError("nonfinite projected observations")
        # Fixed clipping is shared by all hidden arms; never estimated on test.
        x = np.clip(x, -8, 8)
        if pooled:
            mean = np.where(valid, x, 0).sum(1, keepdims=True) / valid.sum(1, keepdims=True)
            x = np.broadcast_to(mean, x.shape)
        output[start:start + 256] = np.where(valid, x, 0)
    return output


def zero_extra_thresholds(labels, features, scores_ss, minimum_normals=100):
    """Conservative cal-only tails in both directions, preserving old alerts.

    This constrains *additional* calibration false alerts, not the total FPR
    and not the test FPR. Scores are classifier outputs, not calibrated error
    probabilities. Rejected/unmapped normals never enlarge the denominator.
    """
    if len(scores_ss) != len(features):
        raise ValueError("feature/score count mismatch")
    if not all(math.isfinite(float(p)) and 0 <= p <= 1 for p in scores_ss):
        raise ValueError("finite binary classifier scores required")
    thresholds, audit = {}, {}
    for expected in PAIR:
        normals = [r for r in labels if r["expected"] == r["actual"] == expected
                   and r["state"] in ("available", "features_unavailable")]
        tail = []
        for r in normals:
            ix = r.get("feature_index")
            if ix is None or (r.get("baseline_phone") or {}).get("errorType"):
                continue
            f = features[ix]
            if f["uid"] != r["uid"] or f["q_runtime"] != r["q_runtime"]:
                raise ValueError("feature/label identity mismatch")
            if f["baseline_pred"] in PAIR:
                p = float(scores_ss[ix])
                tail.append(p if expected == "ㅅ" else 1 - p)
        threshold = None
        if len(normals) >= minimum_normals and tail:
            threshold = max(.5, float(np.nextafter(max(tail), np.inf)))
            if threshold > 1:
                threshold = None
        thresholds[expected] = threshold
        audit[expected] = {"normal_matched_accepted": len(normals), "eligible_unflagged": len(tail),
                           "existing_fp": sum(bool((r.get("baseline_phone") or {}).get("errorType")) for r in normals),
                           "threshold": threshold, "additional_calibration_fp":
                           sum(v >= threshold for v in tail) if threshold is not None else 0}
    return thresholds, audit
