"""Keep original context cadence and class quotas; cycle actual examples.

Only TRAIN example identities change. Affricate/other draws, step count and
the four context selections per step remain exactly the original control.
"""
import collections
import numpy as np
from mdd.sibilant_context_batches import make_plan as original_plan


def make_plan(y, labels, features, seed, steps, mode):
    if mode!='paired_context':
        raise ValueError('coverage comparison requires the original paired-context schedule')
    control, selected, report=original_plan(y,labels,features,seed,steps,mode)
    y=np.asarray(y);needed={tuple(k) for row in selected for k in row}
    byindex=collections.defaultdict(list)
    for r in labels:
        if r.get('feature_index') is not None:byindex[r['feature_index']].append(r)
    pools=collections.defaultdict(list)
    for i,rs in byindex.items():
        if len(rs)!=1 or y[i] not in (0,1):continue
        r=rs[0];key=(r['item'],r['q'],r['expected'],r['following'])
        if key not in needed or r['alignment_ambiguous'] or r['label_kind'] not in ('correct','pair_error') or features[i]['canonical']!=r['expected']:continue
        pools[(key,int(y[i]))].append(i)
    pools={k:np.array(sorted(v),np.int64) for k,v in pools.items()}
    rng=np.random.RandomState(seed^0x424147);bags={};positions=collections.Counter();plan=control.copy()
    for step,contexts in enumerate(selected):
        for j,context in enumerate(contexts):
            key=tuple(context)
            for cls in (0,1):
                k=(key,cls)
                if k not in bags or positions[k]==len(bags[k]):
                    bags[k]=rng.permutation(pools[k]);positions[k]=0
                plan[step,j+4*cls]=bags[k][positions[k]];positions[k]+=1
    if not np.array_equal(plan[:,8:],control[:,8:]) or not np.array_equal(y[plan],y[control]):
        raise AssertionError('class quotas or noncontrast draws changed')
    def coverage(a):
        seen=set(int(i) for i in a.ravel());kinds=collections.Counter()
        for i in seen:
            if len(byindex[i])==1:kinds[byindex[i][0]['label_kind']]+=1
        return {'unique_all':len(seen),'unique_by_class':{str(c):len({i for i in seen if y[i]==c}) for c in range(4)},'unique_by_label_kind':dict(kinds)}
    report.update(coverage_sampling='per-context/actual-class shuffled cycles; all context selections and affricate/other draws unchanged',
                  control_coverage=coverage(control),candidate_coverage=coverage(plan))
    return plan,selected,report
