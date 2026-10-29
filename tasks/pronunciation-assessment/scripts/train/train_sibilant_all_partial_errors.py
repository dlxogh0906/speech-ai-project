"""Test ALL675 recovered error inputs, after the context-matched140-input test.

The new question is data coverage, including contexts absent from the original
draws. Match actual class and exact unique/reuse exposure; contexts may differ
and must be reported. Original normal/third/deletion identities remain fixed.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import collections
import hashlib
import numpy as np
import train_sibilant_partial_errors as runner


def full_error_plans(original,y,labels,features,eligibility,offset,seed):
    pools = [collections.defaultdict(list),collections.defaultdict(list)]
    for i,label in sorted(labels.items()):
        if (y[i] not in (0,1) or label['label_kind'] != 'pair_error'
                or label['expected'] != features[i]['canonical']
                or (i<offset and eligibility[i]['target'] is None)):
            continue
        pools[int(i>=offset)][int(y[i])].append(i)
    def priority(i):
        return hashlib.sha256(('ss-rank-sample-v1:'+labels[i]['uid']).encode()).hexdigest(),features[i]['q_runtime'],i
    matched,slots = {},collections.defaultdict(list)
    plans = [original.copy(),original.copy()]
    for position in np.ndindex(original.shape):
        i = int(original[position])
        if eligibility[i]['target'] is None:
            for plan in plans:plan[position] = -1
            continue
        if y[i]<0:raise ValueError('original supported draw unknown')
        if labels[i]['label_kind']=='pair_error':slots[int(y[i])].append(position)
    rng = np.random.RandomState(seed)
    for cls in (0,1):
        count = len(pools[1][cls])
        if count == 0 or len(slots[cls])<count or len(pools[0][cls])<count:
            raise ValueError('cannot cover every new error with fixed class slots and equal unique control')
        matched[cls] = [sorted(pool[cls],key=priority)[:count] for pool in pools]
        schedule = []
        while len(schedule)<len(slots[cls]):
            order = np.arange(count);rng.shuffle(order);schedule.extend(order.tolist())
        for position,at in zip(slots[cls],schedule):
            for arm in (0,1):plans[arm][position] = matched[cls][arm][at]
    forward,backward = {},{}
    for position in np.ndindex(original.shape):
        i = int(original[position]);a,b = [int(plan[position]) for plan in plans]
        if eligibility[i]['target'] is None:
            if a != b or a != -1:raise ValueError('unsupported slot changed')
            continue
        if labels[i]['label_kind']!='pair_error':
            if a != i or b != i:raise ValueError('normal/third/deletion identity changed')
            continue
        if y[a] != y[b] or y[a] != y[i]:raise ValueError('actual class composition changed')
        if (a in forward and forward[a]!=b) or (b in backward and backward[b]!=a):
            raise ValueError('full error repetition is not bijective')
        forward[a],backward[b] = b,a
    new_used = sorted(set(int(i) for i in plans[1].ravel() if i>=offset))
    if set(new_used) != set(pools[1][0]+pools[1][1]):raise ValueError('new supported errors silently omitted')
    groups = {}
    for name,plan in zip(('old_errors','new_errors'),plans):
        groups[name] = {field:dict(collections.Counter(labels[int(i)].get(field) for i in plan.ravel()
            if i>=0 and labels[int(i)]['label_kind']=='pair_error')) for field in ('item','language','following','position')}
    return plans,{'changed_pair_draws':sum(map(len,slots.values())),'matched_contexts':0,
        'new_supported_pair_pool':len(new_used),'new_unique_used':len(new_used),
        'new_unique_direction_counts':dict(collections.Counter(labels[i]['expected']+'→'+labels[i]['actual'] for i in new_used)),
        'non_pair_draw_identity_unchanged':True,'pair_feature_reuse_bijective':True,
        'all_new_supported_errors_seen_at_least_once':True,'matching_fields':['actual_class','unique_features','exact_reuse_pattern'],
        'context_matching':False,'context_distribution_is_part_of_data_coverage_treatment':groups,
        'exposure_unit':'runtime acoustic ROI feature; recordings/speakers reported separately'}


if __name__ == '__main__':
    runner.error_plans = full_error_plans
    runner.main()
