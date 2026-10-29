"""Local observed fbank skip branch under the existing coarse predicted C role.

Retains all existing encoder summaries. The raw mean control and raw earlier/
later difference have equal parameter counts and zero initial contribution.
The feature extractor's normalized fbank is not raw pressure or a direct
measurement of aspiration/phonation boundaries.
"""
import torch
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_c_trajectory import c_summary


class RawTrajectoryReadout(SupportReadout):
    def __init__(self, hidden_size, branch, acoustic_size=160):
        super().__init__(hidden_size, True)
        if branch not in ("mean", "drift"):
            raise ValueError("mean capacity control or temporal drift required")
        self.branch = branch
        self.extra = torch.nn.Linear(acoustic_size, 64, bias=False)
        torch.nn.init.zeros_(self.extra.weight)

    def load_control(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing != ["extra.weight"] or unexpected:
            raise ValueError("unexpected control parameter layout")
        torch.nn.init.zeros_(self.extra.weight)

    def forward(self, hidden, mask, relative, acoustic):
        if hidden.shape[:2] != acoustic.shape[:2] or acoustic.shape[-1] != self.extra.in_features:
            raise ValueError("observed fbank and encoder time grids must agree")
        if mask.shape != hidden.shape[:2] or relative.shape != mask.shape or not mask.any(1).all():
            raise ValueError("observed mask and runtime relative position required")
        phase = self.support(torch.cat([hidden.float(), relative[..., None]], -1))
        summary, _, _ = c_summary(hidden, mask, phase)
        _, mean, drift = c_summary(acoustic, mask, phase)
        value = self.actual[0](summary) + self.extra(drift if self.branch == "drift" else mean)
        return self.actual[2](self.actual[1](value)), phase
