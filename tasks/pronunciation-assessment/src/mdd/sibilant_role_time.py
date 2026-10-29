"""Temporal moments of learned coarse phone roles, not acoustic landmarks.

20ms stacked-fbank grid masses are soft role occupancies. They are not verified
frication durations, vowel-resonance onsets, glottal onsets or F0 observations.
"""
import torch
from mdd.sibilant_cv_support import SupportReadout

SCHEMA = 'CEL_mass_centroid_spread_grid20ms_v1'
N_TEMPORAL = 9


def temporal_moments(logits, mask):
    if logits.ndim != 3 or logits.shape[-1] != 4 or mask.shape != logits.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('four role logits and observed-prefix mask required')
    if not bool(mask.any(1).all()) or not bool(mask[:, 0].all()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError('nonempty observed prefixes required')
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('finite learned role logits required')
    p = logits.float().softmax(-1)[..., 1:] * mask[..., None]
    mass = p.sum(1)
    weight = p / mass[:, None, :].clamp(min=1e-8)
    # Observed frame-center offsets from the center of the observed prefix.
    # This uses the pinned feature grid, never a CTC/physical phone boundary.
    length = mask.sum(1).float()
    time = (torch.arange(logits.shape[1], device=logits.device).float()[None, :]
            - (length[:, None] - 1.) / 2.) * (.02 / .31)
    mean = (weight * time[..., None]).sum(1)
    variance = (weight * (time[..., None] - mean[:, None, :]).square()).sum(1)
    spread = variance.clamp(min=1e-8).sqrt() * (mass > 1e-8)
    return torch.cat([mass * (.02 / .31), mean, spread], -1)


class RoleTimeReadout(SupportReadout):
    def __init__(self, hidden_size, active):
        super().__init__(hidden_size, True)
        self.active = bool(active)
        self.temporal = torch.nn.Linear(N_TEMPORAL, 64, bias=False)
        torch.nn.init.zeros_(self.temporal.weight)

    def load_base(self, state):
        result = self.load_state_dict(state, strict=False)
        if result.missing_keys != ['temporal.weight'] or result.unexpected_keys:
            raise ValueError('ordinary learned-support base state required')
        if bool(self.temporal.weight.detach().count_nonzero()):
            raise ValueError('temporal projection must begin at zero')

    def terms(self, hidden, mask, relative):
        if mask.shape != hidden.shape[:2] or relative.shape != mask.shape or not bool(mask.any(1).all()):
            raise ValueError('observed frame grid required')
        phase = self.support(torch.cat([hidden.float(), relative[..., None]], -1))
        weights = phase.softmax(-1)[:, :, 1:].transpose(1, 2) * mask[:, None, :]
        if not bool((weights.sum(-1) > 0).all()):
            raise ValueError('missing learned temporal support')
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)
        means = torch.bmm(weights, hidden.float())
        summary = torch.cat([means[:, 0], means[:, 1]-means[:, 0], means[:, 2]-means[:, 1]], -1)
        return self.actual[0](summary), temporal_moments(phase, mask), phase

    def from_terms(self, acoustic, temporal):
        if not self.active:
            temporal = torch.zeros_like(temporal)
        value = acoustic + self.temporal(temporal)
        return self.actual[2](self.actual[1](value))

    def forward(self, hidden, mask, relative):
        acoustic, temporal, phase = self.terms(hidden, mask, relative)
        return self.from_terms(acoustic, temporal), phase
