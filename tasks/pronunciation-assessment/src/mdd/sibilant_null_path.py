"""Compare a target C/V path with a V-only path on coarse learned roles.

Unlike conditioning on an obligatory C path, both hypotheses are retained.
These are weak phonemic regions, not frication/glottal boundary measurements.
An optional observed-token frame binds the V state; it is not a V onset.
"""
import torch


def _partition(emission, mask, present, anchor):
    roles = [0, 1, 2, 0] if present else [0, 2, 0]
    vowel_state = 2 if present else 1
    values = emission[..., roles]
    b, t, n = values.shape
    neg = values.new_full((b,), -10000.)
    state = torch.arange(n, device=values.device)[None, :]
    first = values[:, 0].masked_fill((anchor == 0)[:, None] & (state != vowel_state), -10000.)
    alpha = torch.stack([first[:, i] if i < 2 else neg for i in range(n)], -1)
    for i in range(1, t):
        emit = values[:, i].masked_fill((anchor == i)[:, None] & (state != vowel_state), -10000.)
        advance = torch.cat([neg[:, None], alpha[:, :-1]], -1)
        update = torch.logaddexp(alpha, advance) + emit
        alpha = torch.where(mask[:, i, None], update, alpha)
    return torch.logsumexp(alpha[:, -2:], -1)


def null_path_evidence(logits, mask, anchor=None):
    """Log C/V versus V-only evidence, corrected for allowable path counts.

No anchor means a single V region anywhere in the observed prefix. With an
anchor, that region must contain the specified observed-token grid frame.
Uniform three-role emissions produce zero evidence at every input length.
"""
    if logits.ndim != 3 or logits.shape[-1] != 4 or mask.shape != logits.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('four coarse roles and observed-prefix mask required')
    if (not bool(torch.isfinite(logits).all()) or not bool(mask[:, 0].all())
            or bool((mask[:, 1:] & ~mask[:, :-1]).any()) or not bool((mask.sum(1) >= 2).all())):
        raise ValueError('finite nonempty observed prefixes of length>=2 required')
    if anchor is None:
        anchor = torch.full((len(mask),), -1, device=mask.device, dtype=torch.long)
    if (anchor.shape != mask.shape[:1] or anchor.dtype != torch.long
            or bool((anchor < -1).any()) or bool((anchor >= mask.sum(1)).any())
            or bool((anchor == 0).any())):
        raise ValueError('V anchor must follow at least one observed frame, or be -1')
    lp = logits.float().log_softmax(-1)
    emission = torch.stack([lp[..., 0], lp[..., 1], lp[..., 2:].logsumexp(-1)], -1)
    actual = _partition(emission, mask, True, anchor) - _partition(emission, mask, False, anchor)
    # Equal-length paths share the emission normalization. Zero emissions count
    # allowable paths under each topology, removing the C-path count advantage.
    zero = torch.zeros_like(emission)
    prior = _partition(zero, mask, True, anchor) - _partition(zero, mask, False, anchor)
    return actual - prior
