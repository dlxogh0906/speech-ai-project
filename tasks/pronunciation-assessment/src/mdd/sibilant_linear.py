"""Low-capacity readout of frozen hidden trajectories; no default service activation."""
import numpy as np

from mdd.sibilant_hidden import project_sequences


def temporal_summary(sequence, mask, context, temporal):
    x = np.asarray(sequence, np.float64)
    mask = np.asarray(mask, bool)
    context = np.asarray(context, np.float64)
    if x.ndim != 3 or x.shape[1] != 31 or x.shape[:2] != mask.shape or len(context) != len(x):
        raise ValueError("31-frame observations and corresponding masks/context required")
    if not mask.any(1).all() or not np.isfinite(x[mask]).all() or not np.isfinite(context).all():
        raise ValueError("finite observed frames and nonempty windows required")
    x = np.where(mask[..., None], x, 0)
    mean = x.sum(1) / mask.sum(1)[:, None]
    bounds = (0, 10, 21, 31)
    bins, coverage = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        count = mask[:, a:b].sum(1)
        bins.append(x[:, a:b].sum(1) / np.maximum(count[:, None], 1))
        coverage.append(count / (b - a))
    contrasts = [bins[0] - bins[1], bins[2] - bins[1]] if temporal else [np.zeros_like(mean)] * 2
    return np.column_stack([mean] + contrasts + coverage + [context])


def features(arrays, projection, temporal):
    sequence = project_sequences(arrays["h24"], arrays["mask"], projection)
    return temporal_summary(sequence, arrays["mask"], arrays["context"], temporal)


def predict_readout(values, artifact):
    x = (np.asarray(values, np.float64) - artifact["mean"]) / artifact["scale"]
    z = x @ artifact["coefficient"] + artifact["intercept"]
    if not np.isfinite(z).all():
        raise ValueError("finite readout logits required")
    return np.exp(-np.logaddexp(0, -z))
