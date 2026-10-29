"""One ordered phonemic C/earlyV/lateV path instead of scattered role weights.

These are coarse learned phone regions, never physical frication/aspiration or
glottal boundaries. Conditioning on a path does not prove that speech exists;
the caller must retain the original raw-role confidence/availability rule.
"""
import torch
from mdd.sibilant_cv_support import SupportReadout


def ordered_phase_probabilities(logits, mask):
    if logits.ndim != 3 or logits.shape[-1] != 4 or mask.shape != logits.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("four role emissions and observed-prefix mask required")
    if not bool(torch.isfinite(logits).all()) or not bool((mask.sum(1) >= 3).all()):
        raise ValueError("finite observed C/V support required")
    if not bool(mask[:, 0].all()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError("observed frames must form a prefix")
    lp = logits.float().log_softmax(-1)
    # States: left background, C, earlyV, lateV, right background.
    emit = lp[..., [0, 1, 2, 3, 0]]
    b, t, _ = emit.shape
    neg = emit.new_full((b,), -10000.)
    alpha = torch.stack([emit[:, 0, 0], emit[:, 0, 1], neg, neg, neg], -1)
    forward = [alpha]
    for i in range(1, t):
        previous = torch.cat([neg[:, None], alpha[:, :-1]], -1)
        update = torch.logaddexp(alpha, previous) + emit[:, i]
        alpha = torch.where(mask[:, i, None], update, alpha)
        forward.append(alpha)
    log_z = torch.logsumexp(alpha[:, 3:], -1)
    zero = emit.new_zeros(b)
    beta = torch.stack([neg, neg, neg, zero, zero], -1)
    backward = [None] * t
    backward[-1] = beta
    for i in range(t - 2, -1, -1):
        stay = beta + emit[:, i + 1]
        advance = torch.cat([beta[:, 1:] + emit[:, i + 1, 1:], neg[:, None]], -1)
        update = torch.logaddexp(stay, advance)
        beta = torch.where(mask[:, i + 1, None], update, beta)
        backward[i] = beta
    posterior = (torch.stack(forward, 1) + torch.stack(backward, 1) - log_z[:, None, None]).exp()
    posterior = posterior * mask[..., None]
    return torch.stack([posterior[..., 0] + posterior[..., 4], posterior[..., 1],
                        posterior[..., 2], posterior[..., 3]], -1)


class OrderedSupportReadout(SupportReadout):
    """Identical parameters/summary shape; change only learned pooling weights."""
    def forward(self, hidden, mask, relative):
        if not self.learned:
            return super().forward(hidden, mask, relative)
        if mask.shape != hidden.shape[:2] or relative.shape != mask.shape:
            raise ValueError("runtime observation grid changed")
        phase = self.support(torch.cat([hidden.float(), relative[..., None]], -1))
        weights = ordered_phase_probabilities(phase, mask)[..., 1:].transpose(1, 2)
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)
        means = torch.bmm(weights, hidden.float())
        summary = torch.cat([means[:, 0], means[:, 1] - means[:, 0], means[:, 2] - means[:, 1]], -1)
        return self.actual(summary), phase
