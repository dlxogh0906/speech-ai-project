"""Version-matched mel normalization with an explicit reference scope.

Changing statistics is an experimental frontend choice, not a phonetic label.
Keep raw local frames, stacking grid and sample support unchanged.
"""
import numpy as np


def statistics(fbank):
    value = np.asarray(fbank)
    if value.ndim != 2 or len(value) < 2 or not np.isfinite(value).all():
        raise ValueError("finite mel frames with a sample variance required")
    return value.mean(0), value.var(0, ddof=1)


def stack_normalized(fbank, mean, variance, length, stride=2):
    value = np.asarray(fbank)
    mean, variance = np.asarray(mean), np.asarray(variance)
    if (value.ndim != 2 or mean.shape != value.shape[1:] or variance.shape != mean.shape
            or length < 1 or length * stride > len(value) or not np.isfinite(value).all()
            or not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance < 0).any()):
        raise ValueError("invalid mel statistics or observed stacking support")
    normalized = ((value - mean[None]) / np.sqrt(variance[None] + 1e-7)).astype(np.float32)
    return normalized[:length * stride].reshape(length, stride * value.shape[1])
