"""Coarse weak TAG C/V roles, not physical frication/aspiration/glottal labels.

Inference uses local audio representation and runtime position, never TAG.
Actual classes are s, ss, affricate, other.
"""
import torch


class SupportReadout(torch.nn.Module):
    def __init__(self, hidden_size, learned):
        super().__init__()
        self.learned = learned
        self.support = torch.nn.Sequential(torch.nn.Linear(hidden_size + 1, 64), torch.nn.Tanh(), torch.nn.Linear(64, 4))
        self.actual = torch.nn.Sequential(torch.nn.Linear(3 * hidden_size, 64), torch.nn.Tanh(), torch.nn.Linear(64, 4))

    def forward(self, hidden, mask, relative):
        if mask.shape != hidden.shape[:2] or relative.shape != mask.shape or not mask.any(1).all():
            raise ValueError("observed frames and runtime relative positions required")
        phase = self.support(torch.cat([hidden.float(), relative[..., None]], -1))
        if self.learned:
            weights = phase.softmax(-1)[:, :, 1:].transpose(1, 2)
        else:
            position = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
            bins = (3 * position // mask.long().sum(1, keepdim=True)).clamp(max=2)
            weights = torch.stack([(bins == i).float() for i in range(3)], 1)
        weights = weights * mask[:, None, :]
        if not (weights.sum(-1) > 0).all():
            raise ValueError("missing temporal support")
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)
        means = torch.bmm(weights, hidden.float())
        summary = torch.cat([means[:, 0], means[:, 1] - means[:, 0], means[:, 2] - means[:, 1]], -1)
        return self.actual(summary), phase


def phase_loss(logits, target):
    """Unknown support and uncertain edges contribute no phase supervision."""
    valid = target != -100
    if not valid.any():
        return logits.sum() * 0
    losses = torch.nn.functional.cross_entropy(logits[valid], target[valid], reduction="none")
    groups = [losses[target[valid] == i].mean() for i in range(4) if (target[valid] == i).any()]
    return torch.stack(groups).mean()


def reliable_support(logits, mask):
    """Require observed >=.5 support for C/earlyV/lateV in temporal order.

    This checks predicted coarse-role confidence, not true acoustic boundaries.
    Unavailable support must abstain while keeping the site in evaluation.
    """
    p = logits.softmax(-1)[:, :, 1:].masked_fill(~mask[..., None], 0)
    maxima, peaks = p.max(1)
    return (maxima >= .5).all(1) & (peaks[:, 0] < peaks[:, 1]) & (peaks[:, 1] < peaks[:, 2])
