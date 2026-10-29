"""Exact CTC path diagnostics; alignment posterior is NOT a phone boundary.

These routines use a supplied target for offline diagnosis. In particular,
conditioning on a human TAG transcript is oracle analysis, never inference.
"""
import numpy as np


def trellis(log_probs, tokens, blank=0):
    """Forward sum, best path, and state occupancy given the whole target.

    Impossible targets retain -inf likelihood and have no posterior/path.
    State 2*q+1 represents target occurrence q (including repeated tokens).
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    ids = np.asarray(tokens, dtype=np.int64)
    if lp.ndim != 2 or np.isnan(lp).any() or np.isposinf(lp).any():
        raise ValueError('expected finite or -inf T by V log probabilities')
    if not 0 <= blank < lp.shape[1] or np.any(ids == blank):
        raise ValueError('blank must be in vocabulary and absent from target')
    if np.any(ids < 0) or np.any(ids >= lp.shape[1]):
        raise ValueError('target outside vocabulary')
    tmax, states = lp.shape[0], 2 * len(ids) + 1
    if tmax == 0:
        val = 0. if not len(ids) else -np.inf
        return dict(logsum=val, best=val, posterior=None, path=None)
    ext = np.full(states, blank, np.int64)
    ext[1::2] = ids
    skip = np.zeros(states, bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    alpha = np.full((tmax, states), -np.inf)
    best = np.full_like(alpha, -np.inf)
    back = np.zeros((tmax, states), np.int8)
    alpha[0, 0] = best[0, 0] = lp[0, blank]
    if len(ids):
        alpha[0, 1] = best[0, 1] = lp[0, ids[0]]
    for t in range(1, tmax):
        for prev, dest, is_max in ((alpha[t - 1], alpha[t], False),
                                   (best[t - 1], best[t], True)):
            candidates = np.full((3, states), -np.inf)
            candidates[0] = prev
            candidates[1, 1:] = prev[:-1]
            candidates[2, 2:] = np.where(skip[2:], prev[:-2], -np.inf)
            if is_max:
                back[t] = candidates.argmax(0)
                dest[:] = candidates.max(0) + lp[t, ext]
            else:
                dest[:] = np.logaddexp.reduce(candidates, axis=0) + lp[t, ext]
    ends = [states - 1] if not len(ids) else [states - 2, states - 1]
    z = float(np.logaddexp.reduce(alpha[-1, ends]))
    final = ends[int(best[-1, ends].argmax())]
    value = float(best[-1, final])
    if not np.isfinite(z):
        return dict(logsum=z, best=value, posterior=None, path=None)
    beta = np.full_like(alpha, -np.inf)
    beta[-1, ends] = 0.
    for t in range(tmax - 2, -1, -1):
        nxt = beta[t + 1] + lp[t + 1, ext]
        candidates = np.full((3, states), -np.inf)
        candidates[0] = nxt
        candidates[1, :-1] = nxt[1:]
        candidates[2, :-2] = np.where(skip[2:], nxt[2:], -np.inf)
        beta[t] = np.logaddexp.reduce(candidates, axis=0)
    path = np.empty(tmax, np.int64)
    for t in range(tmax - 1, -1, -1):
        path[t] = final
        final -= int(back[t, final])
    return dict(logsum=z, best=value, posterior=np.exp(alpha + beta - z), path=path)


def occurrence(result, q):
    """Summarize model-conditioned temporal occupancy, not physical boundaries."""
    if result['posterior'] is None:
        return None
    if not 0 <= q < (result['posterior'].shape[1] - 1) // 2:
        raise ValueError('invalid target occurrence')
    mass = result['posterior'][:, 2 * q + 1]
    frames = np.flatnonzero(result['path'] == 2 * q + 1)
    if mass.sum() <= 0 or not len(frames):
        return None
    p = mass / mass.sum()
    cdf = p.cumsum()
    lo, hi = [int(np.searchsorted(cdf, v)) for v in (.05, .95)]
    return dict(mean_frame=float(p @ np.arange(len(p))), frames=frames.tolist(),
                interval90=[lo, hi], expected_occupied_frames=float(mass.sum()),
                posterior_mass_on_best_span=float(p[frames].sum()))
