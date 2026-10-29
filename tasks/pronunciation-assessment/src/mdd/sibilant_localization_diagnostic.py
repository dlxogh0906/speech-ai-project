"""TAG-localized pooling for diagnosis only; annotation masks are not service inputs."""
import numpy as np


def localized_summary(sequence, valid_frames, context, phase_masks):
    """Pool target C, early half V, late half V with the same 219-column layout.

    Retain the runtime coverage columns and context. Do not add human interval
    durations as predictors. Missing phases are unavailable, never filled with
    neighboring speech. These phases are TAG intervals, not verified acoustic
    events such as frication offset or first glottal pulse.
    """
    x = np.asarray(sequence, np.float64)
    valid = np.asarray(valid_frames, bool)
    phases = np.asarray(phase_masks, bool)
    context = np.asarray(context, np.float64)
    if (x.ndim != 3 or x.shape[1] != 31 or valid.shape != x.shape[:2]
            or phases.shape != (len(x), 3, 31) or len(context) != len(x)):
        raise ValueError("31-frame sequences and three annotation masks required")
    if (phases & ~valid[:, None, :]).any() or (phases.sum(1) > 1).any():
        raise ValueError("phases must be disjoint subsets of observed frames")
    if not np.isfinite(x[valid]).all() or not np.isfinite(context).all():
        raise ValueError("finite observed features required")
    x = np.where(valid[..., None], x, 0.)
    counts = phases.sum(2)
    available = (counts > 0).all(1)
    means = np.einsum("npt,ntd->npd", phases, x) / np.maximum(counts[..., None], 1)
    union = phases.any(1)
    overall = np.einsum("nt,ntd->nd", union, x) / np.maximum(union.sum(1)[:, None], 1)
    coverage = [valid[:, a:b].mean(1) for a, b in [(0, 10), (10, 21), (21, 31)]]
    summary = np.column_stack([overall, means[:, 0] - means[:, 1], means[:, 2] - means[:, 1]] + coverage + [context])
    return summary, available
