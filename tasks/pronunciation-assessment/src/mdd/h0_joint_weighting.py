"""Hard joint support and numerical feasibility repair for a fixed TRAIN QP."""
import numpy as np
from scipy import sparse


JOINT_BOUNDS = {'phone_source': (.8, 1.2), 'speaker_source': (.5, 2.)}


def repaired_certificate(Q, target, point, anchor, A, lower, upper, multipliers):
    """A feasible witness can use a valid lower bound tangent at another point.

    The convex tangent lower bound is global even if its anchor is infeasible.
    Recomputing it at the repaired point alone can unnecessarily weaken it.
    Neither the fixed primal witness nor the tolerance changes here.
    """
    from .h0_p6_feasibility import certificate
    current = certificate(Q,target,point,A,lower,upper,multipliers)
    old = certificate(Q,target,anchor,A,lower,upper,multipliers)
    current['witness_tangent_lower_bound'] = current['lower_bound']
    current['raw_anchor_lower_bound'] = old['lower_bound']
    current['lower_bound'] = max(current['lower_bound'],old['lower_bound'])
    current['upper_minus_lower'] = current['objective']-current['lower_bound']
    current['certified_to_tolerance'] = bool(current['numerically_feasible'] and
        abs(current['upper_minus_lower']) <= 2e-5*max(1.,abs(current['objective'])))
    current['lower_bound_scope'] = 'maximum of two valid global tangent bounds, same constraints and box'
    return current


def weighted_ctc_loss(logits, lengths, targets, weights):
    """The frozen proxy's actual loss: mean(w_i * NLL_i / target length)."""
    import torch
    sizes = torch.tensor([len(y) for y in targets], device=logits.device)
    if bool((sizes == 0).any()):
        raise ValueError('empty CTC target')
    flat = torch.tensor([t for y in targets for t in y], device=logits.device, dtype=torch.long)
    w = torch.as_tensor(weights, dtype=torch.float32, device=logits.device)
    if w.shape != sizes.shape or not torch.isfinite(w).all() or bool((w <= 0).any()):
        raise ValueError('invalid utterance weights')
    lp = logits.float().log_softmax(-1).transpose(0, 1)
    with torch.backends.cudnn.flags(enabled=False):
        loss = torch.nn.functional.ctc_loss(lp, flat, lengths, sizes, blank=0,
            reduction='none', zero_infinity=False)
    if not torch.isfinite(loss).all():
        raise ValueError('nonfinite CTC; no sample may be silently dropped')
    return (w * loss / sizes).mean()


def append_joint_constraints(A, lower, upper, matrices):
    blocks, lows, highs = [A], [lower], [upper]
    for name, (lo, hi) in JOINT_BOUNDS.items():
        block = matrices[name]
        blocks.append(block)
        lows.append(np.full(block.shape[0], lo))
        highs.append(np.full(block.shape[0], hi))
    return sparse.vstack(blocks, format='csr'), np.concatenate(lows), np.concatenate(highs)


def joint_metrics(matrices, weights, tolerance=2e-6):
    result = {}
    for name, (lo, hi) in JOINT_BOUNDS.items():
        ratios = matrices[name].dot(weights)
        outside = (ratios < lo-tolerance) | (ratios > hi+tolerance)
        result[name] = dict(groups=len(ratios), min=float(ratios.min()),
                            max=float(ratios.max()), outside=int(outside.sum()),
                            passed=not bool(outside.any()))
    return result


def repair_tiny_residual(weights, source_matrix, A, lower, upper):
    """Normalize disjoint source totals, then mix toward interior C0 if needed.

    This only removes floating-point feasibility residuals. Callers must require
    a certified near-feasible solution first and recheck its objective bound.
    """
    source = np.asarray(source_matrix.argmax(axis=0)).ravel()
    if np.any(np.asarray((source_matrix != 0).sum(axis=0)).ravel() != 1):
        raise ValueError('source partition must assign every variable exactly once')
    point = weights / source_matrix.dot(weights)[source]
    equal = lower == upper
    B = sparse.vstack([A[~equal], sparse.eye(len(weights))], format='csr')
    lo = np.r_[lower[~equal], np.full(len(weights), .25)]
    hi = np.r_[upper[~equal], np.full(len(weights), 4.)]
    base = B.dot(np.ones(len(weights)))
    if not (np.all(base > lo) and np.all(base < hi)):
        raise ValueError('C0 must be strictly inside all inequality bounds')
    value = B.dot(point); delta = value-base; alpha = 1.
    over, under = value > hi, value < lo
    if over.any():
        alpha = min(alpha, float(np.min((hi[over]-base[over])/delta[over])))
    if under.any():
        alpha = min(alpha, float(np.min((lo[under]-base[under])/delta[under])))
    if alpha < 1:
        alpha *= (1-1e-9)
    repaired = 1+alpha*(point-1)
    return repaired, dict(mix_fraction=alpha,
                         max_weight_change=float(np.max(abs(repaired-weights))))
