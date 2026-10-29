"""Test within-C representation change without claiming physical boundaries.

The existing coarse role weights and C/earlyV/lateV means are retained. An extra
zero-initialized projection receives either C mean (capacity control) or the
change between soft earlier/later C representations. No TAG is used at inference.
"""
import torch
from mdd.sibilant_cv_support import SupportReadout


def c_summary(hidden, mask, phase):
    weights = phase.softmax(-1)[..., 1:] * mask[..., None]
    weights = weights.transpose(1, 2)
    weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)
    means = torch.bmm(weights, hidden.float())
    c = weights[:, 0]
    progress = c.cumsum(-1) - .5 * c
    early, late = c * (1 - progress), c * progress
    early = early / early.sum(-1, keepdim=True).clamp(min=1e-8)
    late = late / late.sum(-1, keepdim=True).clamp(min=1e-8)
    drift = torch.bmm((late - early)[:, None], hidden.float())[:, 0]
    summary = torch.cat([means[:, 0], means[:, 1] - means[:, 0], means[:, 2] - means[:, 1]], -1)
    return summary, means[:, 0], drift


class CTrajectoryReadout(SupportReadout):
    def __init__(self, hidden_size, branch):
        super().__init__(hidden_size, True)
        if branch not in ("mean", "drift"):
            raise ValueError("mean capacity control or temporal drift required")
        self.branch = branch
        self.extra = torch.nn.Linear(hidden_size, 64, bias=False)
        torch.nn.init.zeros_(self.extra.weight)

    def load_control(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing != ["extra.weight"] or unexpected:
            raise ValueError("unexpected control parameter layout")
        torch.nn.init.zeros_(self.extra.weight)

    def forward(self, hidden, mask, relative):
        if mask.shape != hidden.shape[:2] or relative.shape != mask.shape or not mask.any(1).all():
            raise ValueError("observed frame mask and runtime position required")
        phase = self.support(torch.cat([hidden.float(), relative[..., None]], -1))
        summary, mean, drift = c_summary(hidden, mask, phase)
        value = self.actual[0](summary) + self.extra(drift if self.branch == "drift" else mean)
        return self.actual[2](self.actual[1](value)), phase
