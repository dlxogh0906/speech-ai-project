"""Observed relative C/V log-fbank branch, with the original encoder unchanged.

Coarse learned roles are weights, not physical event boundaries. Global acoustic
standardization is fitted on observed TRAIN frames only, preserving per-clip
spectral contrasts. Relative spectra cancel a constant per-channel offset; this
does not establish microphone invariance or a universal phonetic rule.
"""
import torch
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_c_trajectory import c_summary


class RelativeSpectrumReadout(SupportReadout):
    def __init__(self, hidden_size, branch, acoustic_mean, acoustic_scale):
        super().__init__(hidden_size, True)
        if branch not in ("mean", "relative"):
            raise ValueError("mean capacity control or relative C/V spectrum required")
        mean, scale = torch.as_tensor(acoustic_mean).float(), torch.as_tensor(acoustic_scale).float()
        if mean.ndim != 1 or scale.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("finite training acoustic normalization required")
        self.branch = branch
        self.register_buffer("acoustic_mean", mean)
        self.register_buffer("acoustic_scale", scale)
        self.extra = torch.nn.Linear(len(mean), 64, bias=False)
        torch.nn.init.zeros_(self.extra.weight)

    def load_control(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if set(missing) != {"acoustic_mean", "acoustic_scale", "extra.weight"} or unexpected:
            raise ValueError("unexpected control parameter layout")
        torch.nn.init.zeros_(self.extra.weight)

    def forward(self, hidden, mask, relative, acoustic):
        if hidden.shape[:2] != acoustic.shape[:2] or acoustic.shape[-1] != self.extra.in_features:
            raise ValueError("observed acoustic and encoder time grids must agree")
        if mask.shape != hidden.shape[:2] or relative.shape != mask.shape or not mask.any(1).all():
            raise ValueError("observed mask and runtime relative position required")
        phase = self.support(torch.cat([hidden.float(), relative[..., None]], -1))
        summary, _, _ = c_summary(hidden, mask, phase)
        x = (acoustic.float() - self.acoustic_mean) / self.acoustic_scale
        weights = phase.softmax(-1)[..., 1:3].transpose(1, 2) * mask[:, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)
        means = torch.bmm(weights, x)
        evidence = means[:, 0] if self.branch == "mean" else means[:, 0] - means[:, 1]
        value = self.actual[0](summary) + self.extra(evidence)
        return self.actual[2](self.actual[1](value)), phase
