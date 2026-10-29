"""Equal unique/repeated error observations while every non-error draw stays fixed."""
import collections
import hashlib
import numpy as np


def error_context(i,y,labels,features):
    label,f = labels[i],features[i]
    parts = label['uid'].split('-')
    sex = parts[1] if len(parts)>1 else 'unknown'
    return (label['item'],f['q_runtime'],f['canonical'],f['expected_V'],label['position'],
            label.get('language'),sex,int(y[i]))


def error_plans(original,y,labels,features,eligibility,offset,seed):
    pools = [collections.defaultdict(list),collections.defaultdict(list)]
    for i,label in sorted(labels.items()):
        f = features[i]
        if (y[i] not in (0,1) or label['label_kind'] != 'pair_error'
                or label['expected'] != f['canonical'] or label['following'] != f['expected_V']
                or (i<offset and eligibility[i]['target'] is None)):
            continue
        pools[int(i>=offset)][error_context(i,y,labels,features)].append(i)
    def priority(i):
        identity = 'ss-rank-sample-v1:'+labels[i]['uid']
        return hashlib.sha256(identity.encode()).hexdigest(),features[i]['q_runtime'],i
    matched = {}
    for context in set(pools[0]) & set(pools[1]):
        size = min(len(pool[context]) for pool in pools)
        matched[context] = [sorted(pool[context],key=priority)[:size] for pool in pools]
    plans = [original.copy(),original.copy()];rng = np.random.RandomState(seed);changed = 0
    for position in np.ndindex(original.shape):
        i = int(original[position])
        if eligibility[i]['target'] is None:
            for plan in plans:plan[position] = -1
            continue
        if y[i] < 0:
            raise ValueError('original supported draw has unknown class')
        if labels[i]['label_kind'] != 'pair_error':continue
        pair = matched.get(error_context(i,y,labels,features))
        if pair is None:continue
        at = int(rng.random_sample()*len(pair[0]))
        for arm in (0,1):plans[arm][position] = pair[arm][at]
        changed += 1
    forward,backward = {},{}
    for position in np.ndindex(original.shape):
        source = int(original[position]);a,b = [int(plan[position]) for plan in plans]
        if eligibility[source]['target'] is None:
            if a != -1 or b != -1:raise ValueError('original unsupported draw changed')
            continue
        if labels[source]['label_kind'] != 'pair_error':
            if a != source or b != source:raise ValueError('normal/third/deletion draw changed')
            continue
        if error_context(a,y,labels,features) != error_context(b,y,labels,features):
            raise ValueError('class/item/runtimeposition/vowel/phoneticposition/L1/sex mismatch')
        if (a in forward and forward[a] != b) or (b in backward and backward[b] != a):
            raise ValueError('error feature exposure/repetition is not bijective')
        forward[a],backward[b] = b,a
    new_used = sorted(set(int(i) for i in plans[1].ravel() if i>=offset))
    kinds = collections.Counter(labels[i]['expected']+'→'+labels[i]['actual'] for i in new_used)
    report = {'changed_pair_draws':changed,'matched_contexts':len(matched),
        'new_supported_pair_pool':sum(len(z) for z in pools[1].values()),
        'new_unique_used':len(new_used),'new_unique_direction_counts':dict(kinds),
        'non_pair_draw_identity_unchanged':True,'pair_feature_reuse_bijective':True,
        'matching_fields':['item','q_runtime','expected_C','following_V','position','language','sex','actual_class'],
        'exposure_unit':'runtime acoustic ROI feature; recording/speaker identity counts reported separately'}
    return plans,report
