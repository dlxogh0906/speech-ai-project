"""TRAIN-only marginal class-mass correction for a fixed replay schedule.

This corrects class exposure, not within-class context/site selection bias.
All five actual-phone classes retain positive loss. No CAL/test priors enter.
"""
import numpy as np
import torch


def class_mass_weights(y, sampled, classes=5):
    y = np.asarray(y)
    sampled = np.asarray(sampled, dtype=np.int64)
    if y.ndim != 1 or sampled.ndim != 1 or not len(sampled):
        raise ValueError('nonempty one-dimensional TRAIN targets/draws required')
    if (sampled < 0).any() or (sampled >= len(y)).any() or (y[sampled] < 0).any():
        raise ValueError('invalid or unknown sampled TRAIN class')
    known = y[y >= 0]
    if not len(known) or (known >= classes).any():
        raise ValueError('invalid actual class')
    population = np.bincount(known, minlength=classes)
    draws = np.bincount(y[sampled], minlength=classes)
    if (population == 0).any() or (draws == 0).any():
        raise ValueError('each actual class must occur in TRAIN and replay')
    weights = (population / population.sum()) / (draws / draws.sum())
    return weights.astype(np.float32), population, draws


def actual_class_loss(logits, target, weights=None):
    if weights is None:
        return torch.nn.functional.cross_entropy(logits, target)
    weights = torch.as_tensor(weights, dtype=logits.dtype, device=logits.device)
    if (weights.shape != (logits.shape[-1],) or not bool(torch.isfinite(weights).all())
            or not bool((weights > 0).all()) or bool((target < 0).any())):
        raise ValueError('positive finite actual-class weights and known targets required')
    # Divide by draw count, not sum of batch weights: the fixed full replay's
    # mean class weight is one. Batch renormalization would change the objective.
    return (torch.nn.functional.cross_entropy(logits, target, reduction='none') * weights[target]).mean()
