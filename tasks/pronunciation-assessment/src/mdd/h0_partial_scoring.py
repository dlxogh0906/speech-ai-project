"""Partial-reference CTC alignment and annotation-conditional stress metrics.

The wildcard is a language of zero or more phones, not a repaired label. CTC
composition retains repeated-label/blank semantics across wildcard boundaries.
"""
import collections
import numpy as np
from .h0_partial_reference import UNKNOWN, UNOBSERVED, DELETED


def mapping_indices(reference, target):
    """All optimal edit correspondences, preserving target occurrence identity."""
    n, m = len(reference), len(target); inf = n + m + 1
    f = [[inf] * (m + 1) for _ in range(n + 1)]; f[0][0] = 0
    b = [[inf] * (m + 1) for _ in range(n + 1)]; b[n][m] = 0
    for i in range(n + 1):
        rw = i < n and reference[i] == UNKNOWN
        for j in range(m + 1):
            tw = j < m and target[j] == UNKNOWN; cost = int(not (rw or tw))
            if i < n: f[i+1][j] = min(f[i+1][j], f[i][j] + cost)
            if j < m: f[i][j+1] = min(f[i][j+1], f[i][j] + cost)
            if i < n and j < m and not rw and not tw:
                f[i+1][j+1] = min(f[i+1][j+1], f[i][j] + int(reference[i] != target[j]))
    for i in range(n, -1, -1):
        rw = i < n and reference[i] == UNKNOWN
        for j in range(m, -1, -1):
            if i == n and j == m: continue
            tw = j < m and target[j] == UNKNOWN; cost = int(not (rw or tw)); best = inf
            if i < n: best = min(best, b[i+1][j] + cost)
            if j < m: best = min(best, b[i][j+1] + cost)
            if i < n and j < m and not rw and not tw:
                best = min(best, b[i+1][j+1] + int(reference[i] != target[j]))
            b[i][j] = best
    distance = f[n][m]; options = []
    for i, token in enumerate(reference):
        if token == UNKNOWN: options.append(None); continue
        values = set()
        for j in range(m + 1):
            tw = j < m and target[j] == UNKNOWN
            if f[i][j] + int(not tw) + b[i+1][j] == distance:
                values.add(UNOBSERVED if tw else None)
            if j < m and not tw and f[i][j] + int(token != target[j]) + b[i+1][j+1] == distance:
                values.add(j)
        options.append(frozenset(values))
    return options


def unique_index(values):
    if values and len(values) == 1:
        value = next(iter(values))
        if isinstance(value, int): return value
    return None


def value_options(indices, target):
    if indices is None: return None
    return frozenset(DELETED if j is None else (UNOBSERVED if j == UNOBSERVED else target[j]) for j in indices)


def gap_value(reference, target, indices, gap):
    """Internal gaps only; both bounding occurrences must be uniquely known."""
    if gap <= 0 or gap >= len(reference): return None
    if reference[gap-1] == UNKNOWN or reference[gap] == UNKNOWN: return None
    left, right = unique_index(indices[gap-1]), unique_index(indices[gap])
    if left is None or right is None or right <= left: return None
    content = target[left+1:right]
    return None if UNKNOWN in content else tuple(content)


def annotation_mask(record, indices):
    """Fixed before inference. Deletions require two real, known anchor phones."""
    sites = record['sites']; n = len(sites); result = []
    for i, s in enumerate(sites):
        if s['status'] not in ('normal', 'substitution', 'deletion'):
            result.append(s['status']); continue
        good = True
        for j in (i-1, i+1):
            if j < 0 or j >= n:
                good &= s['status'] != 'deletion'
            else:
                good &= (sites[j]['status'] in ('normal', 'substitution') and unique_index(indices[j]) is not None)
        result.append(None if good else 'annotation_boundary_unresolved')
    return result


def partial_ctc_viterbi(lp, reference_ids, phone_ids):
    """Max raw-frame CTC path for known IDs and None (zero-or-more phones).

    State (q, last_frame_symbol) composes a reference NFA with CTC collapse.
    Nonphone special symbols are forbidden, as in canonical CTC alignment.
    Returned owners identify known reference occurrences, not physical boundaries.
    """
    lp = np.asarray(lp, dtype=np.float64)
    T, V = lp.shape; N = len(reference_ids); Q = N + 1
    if T == 0: return None
    if Q * V >= 65535: raise ValueError('too many states for compact backpointers')
    wild = np.array([i for i, x in enumerate(reference_ids) if x is None], dtype=int)
    known = np.array([i for i, x in enumerate(reference_ids) if x is not None], dtype=int)
    labels = np.array([reference_ids[i] for i in known], dtype=int)
    if 0 in phone_ids: raise ValueError('blank cannot be a phone')
    separators = [0]
    lp = lp.copy()
    lp[:, sorted(set(range(V)) - set(phone_ids) - {0})] = -np.inf
    state_ids = np.arange(Q * V, dtype=np.uint16).reshape(Q, V)
    dp = np.full((Q, V), -np.inf); dp[0, 0] = 0.
    for q in wild: dp[q+1] = np.maximum(dp[q+1], dp[q])
    back = np.empty((T, Q, V), np.uint16)
    action = np.empty((T, Q, V), np.int16)
    rows = np.arange(Q)
    for t in range(T):
        best_id = dp.argmax(1); best = dp[rows, best_id]
        other = dp.copy(); other[rows, best_id] = -np.inf
        second_id = other.argmax(1); second = other[rows, second_id]
        cur = dp.copy(); bp = state_ids.copy(); owner = np.full((Q, V), -2, np.int16)
        for c in separators:
            cur[:, c] = best; bp[:, c] = state_ids[rows, best_id]; owner[:, c] = -1
        for q in wild:
            cur[q, :] = best[q]; bp[q, :] = state_ids[q, best_id[q]]
            owner[q, :] = -1; owner[q, best_id[q]] = -2
            owner[q, separators] = -1
        if len(known):
            prev_id = np.where(best_id[known] != labels, best_id[known], second_id[known])
            scores = np.where(best_id[known] != labels, best[known], second[known])
            improve = scores > cur[known+1, labels]
            k, c, v = known[improve], labels[improve], prev_id[improve]
            cur[k+1, c] = scores[improve]; bp[k+1, c] = state_ids[k, v]; owner[k+1, c] = k
        cur += lp[t][None, :]
        # Epsilon skips retain both the raw-frame last symbol and its owner action.
        for q in wild:
            improve = cur[q] > cur[q+1]
            cur[q+1, improve] = cur[q, improve]
            bp[q+1, improve] = bp[q, improve]; owner[q+1, improve] = owner[q, improve]
        back[t], action[t], dp = bp, owner, cur
    last = int(dp[N].argmax()); score = float(dp[N, last])
    if not np.isfinite(score): return None
    state = N * V + last; symbols = np.empty(T, int); actions = np.empty(T, int)
    for t in range(T-1, -1, -1):
        q, c = divmod(state, V); symbols[t] = c; actions[t] = action[t, q, c]
        state = int(back[t, q, c])
    owners = []; previous = -1
    for a in actions:
        if a != -2: previous = int(a)
        owners.append(previous)
    segments = [np.flatnonzero(np.asarray(owners) == q).tolist() if y is not None else []
                for q, y in enumerate(reference_ids)]
    return dict(score=score, segments=segments, symbols=symbols.tolist(), owners=owners)


def metric_record(sites, threshold, seam_band_samples):
    """Conditional observed rates plus bounds on the fixed annotation denominator."""
    out = collections.Counter()
    for s in sites:
        out['known_reference_sites'] += s['expected'] != UNKNOWN
        out['reference_entries'] += 1
        if s['annotation_exclusion']:
            out['excluded_' + s['annotation_exclusion']] += 1; continue
        error = s['actual'] != s['expected']; name = 'error' if error else 'normal'
        out[name] += 1; out['eligible_sites'] += 1
        if error: out['deletion' if s['actual'] == DELETED else 'substitution'] += 1
        pred = s['prediction_options']
        available = (s['score'] is not None and s['nearest_seam_samples'] > seam_band_samples and
                     len(pred) == 1 and (pred[0] != DELETED or s['prediction_deletion_anchored']))
        if not available:
            out['unobserved_' + name] += 1
            out['exact_upper'] += int(error); out['fp_upper'] += int(not error)
            continue
        out['observed_' + name] += 1
        warned = pred[0] != s['expected'] and (threshold is None or s['score'] >= threshold)
        fp = warned and not error; exact = warned and error and pred[0] == s['actual']
        out['fp'] += int(fp); out['exact'] += int(exact)
        out['fp_upper'] += int(fp); out['exact_upper'] += int(exact)
        out['wrong_actual'] += int(warned and error and not exact)
        if s['actual'] == DELETED:
            out['observed_deletion'] += 1; out['deletion_exact'] += int(exact)
    return dict(out)


def rates(c):
    def ratio(a, b): return c.get(a, 0) / c[b] if c.get(b, 0) else None
    return dict(annotation_coverage=ratio('eligible_sites', 'known_reference_sites'),
        normal_observation_coverage=ratio('observed_normal', 'normal'),
        error_observation_coverage=ratio('observed_error', 'error'),
        observed_normal_fpr=ratio('fp', 'observed_normal'),
        observed_exact_recall=ratio('exact', 'observed_error'),
        fixed_normal_fpr_lower=ratio('fp', 'normal'), fixed_normal_fpr_upper=ratio('fp_upper', 'normal'),
        fixed_exact_recall_lower=ratio('exact', 'error'), fixed_exact_recall_upper=ratio('exact_upper', 'error'))
