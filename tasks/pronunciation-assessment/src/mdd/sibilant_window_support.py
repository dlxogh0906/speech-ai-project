"""Weak phonemic support across the two overlapping fbank analysis windows.

The kernel averages squared window weights of the two stacked mel frames. Its
label fractions describe analysis support, not the acoustic power of each phone,
nor physical frication/aspiration/phonation boundaries.
"""
import numpy as np
import torch


def stacked_support_kernel(window):
    window = np.asarray(window, dtype=np.float64)
    if window.shape != (400,) or not np.isfinite(window).all() or not (window ** 2).sum() > 0:
        raise ValueError("pinned400-sample finite analysis window required")
    weight = window ** 2
    weight = weight / weight.sum()
    kernel = np.zeros(560, np.float64)
    kernel[:400] += .5 * weight
    kernel[160:] += .5 * weight
    return kernel


def window_role_support(sample_roles, left, right, length, kernel, minimum_known=.5):
    from mdd.sibilant_frame_clock import stacked_window_centers
    sample_roles = np.asarray(sample_roles)
    kernel = np.asarray(kernel)
    if kernel.shape != (560,) or not np.isfinite(kernel).all() or (kernel < 0).any() or not np.isclose(kernel.sum(), 1):
        raise ValueError("normalized stacked analysis support kernel required")
    stacked_window_centers(left, right, length)
    if right > len(sample_roles) or sample_roles.ndim != 1 or not 0 <= minimum_known <= 1:
        raise ValueError("observed sample roles and valid certainty floor required")
    indices = int(left) + 320 * np.arange(length)[:, None] + np.arange(560)[None]
    roles = sample_roles[indices]
    support = np.stack([((roles == k) * kernel).sum(-1) for k in range(4)], -1)
    support[support.sum(-1) < minimum_known] = 0
    return support.astype(np.float32)


def balanced_support_loss(logits, support):
    """Soft extension of the existing per-role-balanced CE; zeros are unknown."""
    if logits.shape != support.shape or logits.shape[-1] != 4:
        raise ValueError("matching four-role support targets required")
    if not torch.isfinite(support).all() or (support < 0).any() or (support.sum(-1) > 1.00001).any():
        raise ValueError("finite nonnegative weak support mass<=1 required")
    logp, terms = logits.log_softmax(-1), []
    for k in range(4):
        weight = support[..., k]
        total = weight.sum()
        if bool(total > 0):
            terms.append(-(weight * logp[..., k]).sum() / total)
    return torch.stack(terms).mean() if terms else logits.sum() * 0
