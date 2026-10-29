"""Reuse the frozen new-normal trainer with a bijective repetition-matched plan.

The original runner stays byte-for-byte unchanged for prior reproducibility.
Only its explicit TRAIN planning function is replaced in this process. Both
entrypoint hashes must be bound by the new protocol; no inference code changes.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/data'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import collections
import numpy as np
import train_sibilant_extra_normals as runner
from prepare_sibilant_extra_normal_pilot import key
from prepare_sibilant_rank_pilot import priority


def equal_plans(original, y, indexed, features, rows, eligibility, offset, seed):
    pools = [collections.defaultdict(list), collections.defaultdict(list)]
    for i,label in sorted(indexed.items()):
        if (y[i] not in (0,1) or label['label_kind'] != 'correct'
                or label['expected'] != features[i]['canonical']
                or any(z['label_kind'] != 'correct' for z in rows[label['uid']]['sites'])
                or (i < offset and eligibility[i]['target'] is None)):
            continue
        pools[int(i >= offset)][key(rows[label['uid']],label)].append(i)
    matched = {}
    for context in set(pools[0]) & set(pools[1]):
        count = min(len(pool[context]) for pool in pools)
        matched[context] = [sorted(pool[context],key=lambda i:priority(indexed[i]['uid']))[:count]
            for pool in pools]
    plans, changed, rng = [original.copy(), original.copy()], 0, np.random.RandomState(seed)
    for position in np.ndindex(original.shape):
        i = int(original[position])
        if eligibility[i]['target'] is None:
            for plan in plans:
                plan[position] = -1
            continue
        if y[i] < 0:
            raise ValueError('supported original draw has unknown target')
        if indexed[i]['label_kind'] != 'correct':
            continue
        context = key(rows[indexed[i]['uid']],indexed[i])
        pair = matched.get(context)
        if pair is None or not len(pair[0]):
            continue
        at = int(rng.random_sample()*len(pair[0]))
        for arm in (0,1):
            plans[arm][position] = pair[arm][at]
        changed += 1
    # A bijection of normal feature IDs proves identical repetition, not just
    # equal aggregate counts. Error slots and original unsupported slots stay put.
    forward, backward = {}, {}
    for position in np.ndindex(original.shape):
        old,new = [int(plan[position]) for plan in plans]
        if old < 0 or new < 0:
            if old != new:
                raise ValueError('support mismatch')
            continue
        if y[old] != y[new] or key(rows[indexed[old]['uid']],indexed[old]) != key(rows[indexed[new]['uid']],indexed[new]):
            raise ValueError('class/context mismatch')
        if indexed[old]['label_kind'] != 'correct':
            if old != new:
                raise ValueError('actual error identity changed')
            continue
        if old in forward and forward[old] != new or new in backward and backward[new] != old:
            raise ValueError('normal reuse pattern is not bijective')
        forward[old], backward[new] = new, old
    return plans, changed


if __name__ == '__main__':
    runner.matched_plans = equal_plans
    runner.main()
