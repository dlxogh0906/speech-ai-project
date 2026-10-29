"""A single CTC localization path with a marginal s/ss target emission.

The union state may vary between s and ss over its frames. It localizes a
contrast family, not the realized phone or a physical frication boundary.
"""
import math
import numpy as np
from mdd.gop_feats_v2 import viterbi_align


def union_anchor(log_probs, canonical_ids, q, pair_ids, blank):
    lp = np.asarray(log_probs)
    if lp.ndim != 2 or not len(lp) or not np.isfinite(lp).all():
        raise ValueError('finite frame by vocabulary log probabilities required')
    if len(pair_ids) != 2 or len(set(pair_ids)) != 2 or blank in pair_ids:
        raise ValueError('two distinct nonblank contrast tokens required')
    if not 0 <= q < len(canonical_ids) or canonical_ids[q] not in pair_ids:
        raise ValueError('invalid contrast target')
    if any(not 0 <= i < lp.shape[1] for i in list(canonical_ids) + list(pair_ids)):
        raise ValueError('token outside vocabulary')
    # A merged identity would change CTC repetition constraints when an adjacent
    # canonical token is in the same family. Abstain instead of changing them.
    if (q > 0 and canonical_ids[q - 1] in pair_ids) or (q + 1 < len(canonical_ids) and canonical_ids[q + 1] in pair_ids):
        return None
    merged = np.logaddexp(lp[:, pair_ids[0]], lp[:, pair_ids[1]])
    extended = np.column_stack([lp, merged])
    target = list(canonical_ids)
    target[q] = lp.shape[1]
    segments = viterbi_align(extended, target, blank)
    if segments is None or not segments[q]:
        return None
    frames = segments[q]
    return {'frame': int(math.floor(float(np.mean(frames)) + .5)),
            'segment': frames, 'union_state_only_for_q': q}
