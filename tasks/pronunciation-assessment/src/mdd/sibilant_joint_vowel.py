"""Undivided coarse vowel support, retaining uncertainty at actual TAG edges.

The midpoint of a vowel prefix is a readout convention, not an annotated
physical boundary. Preserve all four role logits and parameter layout, but
optionally supervise E+L jointly and derive temporal summaries from vowel mass.
"""
import numpy as np
import torch
from mdd.sibilant_cv_support import SupportReadout, reliable_support


def coarse_cv_roles(times, intervals, actual_index, vowels, prefix=.12, edge=.01):
    roles = np.full(len(times), -100, np.int64)
    if intervals is None or actual_index is None or actual_index + 1 >= len(intervals):
        return roles
    c, v = intervals[actual_index:actual_index + 2]
    if c is None or v is None or v[2] not in vowels or not (np.isfinite(c[:2]).all() and np.isfinite(v[:2]).all()
            and c[0] < c[1] <= v[0] < v[1]):
        return roles
    end = min(v[1], v[0] + prefix)
    roles[:] = 0
    roles[(times >= c[0]) & (times < c[1])] = 1
    roles[(times >= v[0]) & (times < end)] = 2
    for boundary in (c[0], c[1], v[0], end):
        roles[abs(times - boundary) <= edge] = -100
    return roles


def joint_vowel_loss(logits, targets):
    valid = targets != -100
    if not valid.any():
        return logits.sum() * 0
    if not ((targets[valid] >= 0) & (targets[valid] <= 2)).all():
        raise ValueError("background/C/undividedV targets required")
    lp = logits[valid].log_softmax(-1)
    merged = torch.stack([lp[:, 0], lp[:, 1], lp[:, 2:].logsumexp(-1)], -1)
    losses, weights = [], []
    for k, weight in ((0, 1.), (1, 1.), (2, 2.)):
        chosen = targets[valid] == k
        if chosen.any():
            losses.append(-merged[chosen, k].mean() * weight)
            weights.append(weight)
    return torch.stack(losses).sum() / sum(weights)


class JointVowelReadout(SupportReadout):
    def __init__(self, hidden_size, pooling):
        super().__init__(hidden_size, True)
        if pooling not in ("learned", "vowel_progress"):
            raise ValueError("learned E/L or vowel-progress pooling required")
        self.pooling = pooling

    def forward(self, hidden, mask, relative):
        if self.pooling == "learned":
            return super().forward(hidden, mask, relative)
        if mask.shape != hidden.shape[:2] or relative.shape != mask.shape or not mask.any(1).all():
            raise ValueError("observed mask and runtime positions required")
        phase = self.support(torch.cat([hidden.float(), relative[..., None]], -1))
        p = phase.softmax(-1) * mask[..., None]
        vowel = p[..., 2:].sum(-1)
        progress = (vowel.cumsum(-1) - .5 * vowel) / vowel.sum(-1, keepdim=True).clamp(min=1e-8)
        weights = torch.stack([p[..., 1], vowel * (1 - progress), vowel * progress], 1)
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)
        means = torch.bmm(weights, hidden.float())
        summary = torch.cat([means[:, 0], means[:, 1] - means[:, 0], means[:, 2] - means[:, 1]], -1)
        return self.actual(summary), phase


def strict_or_joint_support(logits, mask):
    """Observed confident C then combinedV; preserve existing strict support."""
    p = logits.softmax(-1).masked_fill(~mask[..., None], 0)
    c, v = p[..., 1], p[..., 2:].sum(-1)
    cm, cp = c.max(1)
    vm, vp = v.max(1)
    return reliable_support(logits, mask) | ((cm >= .5) & (vm >= .5) & (cp < vp))
