"""Reduce low-posterior temporal mixing in the coarse C/V readout.

Squaring is a differentiable pooling choice, not a phonetic cutoff or a claim
that inferred roles are independently validated acoustic boundaries.
"""
import torch
from mdd.sibilant_cv_support import SupportReadout


class SquaredSupportReadout(SupportReadout):
    def forward(self, hidden, mask, relative):
        original, phase = super().forward(hidden, mask, relative)
        if not self.learned:
            return original, phase
        weights = phase.softmax(-1)[:, :, 1:].transpose(1, 2).square()
        weights = weights * mask[:, None, :]
        if not (weights.sum(-1) > 0).all():
            raise ValueError("missing observed temporal support")
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)
        means = torch.bmm(weights, hidden.float())
        summary = torch.cat([means[:, 0], means[:, 1] - means[:, 0], means[:, 2] - means[:, 1]], -1)
        return self.actual(summary), phase
