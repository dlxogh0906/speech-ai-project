"""Factor error diversity against normals from the same recovered recordings."""
import collections
import hashlib
import numpy as np
from mdd.sibilant_conditional_retained import TOKENS,retained_plans

ARMS = ('old_errors_old_normals','new_errors_old_normals',
        'old_errors_new_normals','new_errors_new_normals')


def cohort_plans(original,y,labels,features,eligibility,offset,seed,steps=400,batch=16):
    error_plans,error_report = retained_plans(original,y,labels,features,eligibility,
                                            offset,seed,steps,batch)
    plans = {arm:{} for arm in ARMS};report = {}
    for branch,token in enumerate(TOKENS):
        prefix_size = error_report[token]['original_stream_draws']
        template = error_plans[token]['old'].ravel()
        retained = {int(i) for i in template[:prefix_size]
                    if labels[int(i)]['label_kind']=='correct'}
        pool = [[],[]]
        for i,lab in sorted(labels.items()):
            if (lab['label_kind']!='correct' or lab['expected']!=token
                    or features[i]['canonical']!=token or y[i]!=branch
                    or lab['following']!=features[i]['expected_V']):
                continue
            if i<offset and (eligibility[i]['target'] is None or i in retained):
                continue
            pool[int(i>=offset)].append(i)
        def priority(i):
            s = 'cohort-normal-v1:%s:%s:%s'%(seed,labels[i]['uid'],features[i]['q_runtime'])
            return hashlib.sha256(s.encode()).hexdigest(),i
        pool = [sorted(z,key=priority) for z in pool]
        count = len(pool[1])
        positions = [j for j in range(prefix_size,len(template))
                     if labels[int(template[j])]['label_kind']=='correct']
        if not count or len(pool[0])<count or len(positions)<count:
            raise ValueError('all recovered normals and matched old normals must fit')
        controls,extra = pool[0][:count],pool[1]
        assignment = np.resize(np.random.RandomState(seed+branch).permutation(count),len(positions))
        for arm in ARMS:
            error_data = 'new' if arm.startswith('new_errors') else 'old'
            normal_data = extra if arm.endswith('new_normals') else controls
            plan = error_plans[token][error_data].ravel().copy()
            for pos,at in zip(positions,assignment):plan[pos] = normal_data[int(at)]
            np.testing.assert_array_equal(plan[:prefix_size],template[:prefix_size])
            if not set(normal_data).issubset(set(plan.tolist())):
                raise ValueError('normal coverage incomplete')
            plans[arm][token] = plan.reshape(steps,batch)
        first = plans[ARMS[0]][token].ravel()
        first_classes = collections.Counter(y[first].tolist())
        first_unique = len(set(first.tolist()))
        for arm in ARMS[1:]:
            current = plans[arm][token].ravel()
            if collections.Counter(y[current].tolist())!=first_classes or len(set(current.tolist()))!=first_unique:
                raise ValueError('actual-class/exposure counts differ across factors')
            for a,b in zip(first,current):
                if labels[int(a)]['label_kind'] in ('third_phone','deletion') and a!=b:
                    raise ValueError('third/deletion context changed')
        mapping = dict(zip(controls,extra))
        for ea in ('old_errors','new_errors'):
            a,b = [plans[ea+'_'+na][token].ravel() for na in ('old_normals','new_normals')]
            for ia,ib in zip(a,b):
                if ia!=ib and mapping.get(int(ia))!=int(ib):
                    raise ValueError('normal feature repetition is not bijective')
        report[token] = dict(error_report[token],normal_features_added=count,
            original_unique_normals_retained=len(retained),unused_old_normal_pool=len(pool[0]),
            post_original_normal_slots=len(positions),unique_inputs_all_four=first_unique,
            normal_reuse_bijective=True,class_counts_all_four=dict(first_classes),
            third_deletion_ids_unchanged=True)
    return plans,report
