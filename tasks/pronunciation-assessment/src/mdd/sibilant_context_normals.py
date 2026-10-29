"""Equal-exposure normal controls for recovered actual-C error learning."""
import collections
import hashlib
import numpy as np
from mdd.sibilant_conditional_retained import TOKENS,retained_plans
from mdd.sibilant_source_shift import context

ARMS = ('legacy_diversity_control','new_errors_current_normals',
        'new_errors_random_normals','new_errors_context_normals')


def context_balanced_normals(pool,error_counts,size,labels,features,priority):
    groups = collections.defaultdict(list)
    for i in pool:
        key = context(labels[i],features[i])
        if key in error_counts:groups[key].append(i)
    groups = {key:sorted(values,key=priority) for key,values in groups.items()}
    keys = sorted(groups,key=repr)
    if not keys or len(keys)>size or sum(map(len,groups.values()))<size:
        raise ValueError('cannot cover available error contexts at fixed normal count')
    selected = {key:1 for key in keys}
    total = sum(error_counts.values())
    while sum(selected.values())<size:
        available = [key for key in keys if selected[key]<len(groups[key])]
        if not available:raise ValueError('context normal capacity exhausted')
        key = max(available,key=lambda key:size*error_counts[key]/float(total)-selected[key])
        selected[key]+=1
    result = [i for key in keys for i in groups[key][:selected[key]]]
    result.sort(key=priority)
    report = [{'context':list(key),'error_features':error_counts[key],
        'normal_available':len(groups[key]),'normal_selected':selected[key]} for key in keys]
    return result,report


def normal_context_plans(original,y,labels,features,eligibility,offset,normal_start,seed,
                         sizes=(297,256),steps=400,batch=16,minimum_coverage=.8):
    base,base_report = retained_plans(original,y,labels,features,eligibility,offset,seed,steps,batch)
    plans = {name:{} for name in ARMS};report = {'branches':{},'context_selection':{}}
    covered_errors,total_errors = 0,0
    for branch,token in enumerate(TOKENS):
        size = int(sizes[branch]);prefix_size = base_report[token]['original_stream_draws']
        first = base[token]['old'].ravel();retained = set(first[:prefix_size].tolist())
        pools = [[],[],[]];errors = collections.Counter()
        for i,lab in sorted(labels.items()):
            f = features[i]
            if lab['expected']!=token or f['canonical']!=token or lab['following']!=f['expected_V']:
                continue
            if offset<=i<normal_start and lab['label_kind']=='pair_error' and y[i]==1-branch:
                errors[context(lab,f)]+=1
            if lab['label_kind']!='correct' or y[i]!=branch:continue
            if i<offset:
                if eligibility[i]['target'] is None or i in retained:continue
                pool = 0
            else:pool = 1 if i<normal_start else 2
            pools[pool].append(i)
        def priority(i):
            s = 'context-normal-control-v1:%s:%s:%s'%(seed,labels[i]['uid'],features[i]['q_runtime'])
            return hashlib.sha256(s.encode()).hexdigest(),i
        pools = [sorted(z,key=priority) for z in pools]
        if any(len(z)<size for z in pools):raise ValueError('fixed unique-normal control count unavailable')
        matched,details = context_balanced_normals(pools[2],errors,size,labels,features,priority)
        selected = [pools[0][:size],pools[1][:size],pools[2][:size],matched]
        positions = [j for j in range(prefix_size,len(first)) if labels[int(first[j])]['label_kind']=='correct']
        if len(positions)<size:raise ValueError('fixed normal draw capacity insufficient')
        assignment = np.resize(np.random.RandomState(seed+branch).permutation(size),len(positions))
        for arm,chosen in zip(ARMS,selected):
            data_arm = 'old' if arm==ARMS[0] else 'new'
            plan = base[token][data_arm].ravel().copy()
            for j,k in zip(positions,assignment):plan[j] = chosen[int(k)]
            np.testing.assert_array_equal(plan[:prefix_size],first[:prefix_size])
            if not set(chosen).issubset(set(plan.tolist())):raise ValueError('not all chosen normals seen')
            plans[arm][token] = plan.reshape(steps,batch)
        reference = plans[ARMS[0]][token].ravel()
        reference_classes = collections.Counter(y[reference].tolist());unique = len(set(reference.tolist()))
        for arm in ARMS[1:]:
            current = plans[arm][token].ravel()
            if collections.Counter(y[current].tolist())!=reference_classes or len(set(current.tolist()))!=unique:
                raise ValueError('class/unique exposure differs between controls')
            for a,b in zip(reference,current):
                if labels[int(a)]['label_kind'] in ('third_phone','deletion') and a!=b:
                    raise ValueError('third/deletion input changed')
        # All three new-error arms have identical error feature identities.
        for arm in ARMS[2:]:
            a,b = [plans[name][token].ravel() for name in (ARMS[1],arm)]
            mask = np.asarray([labels[int(i)]['label_kind']=='pair_error' for i in a])
            np.testing.assert_array_equal(a[mask],b[mask])
        # The normal mapping across all4conditions is one-to-one, including reuse.
        mapping = [dict(zip(selected[0],z)) for z in selected]
        for pos in positions:
            a = int(reference[pos])
            if any(int(plans[arm][token].ravel()[pos])!=mp[a] for arm,mp in zip(ARMS,mapping)):
                raise ValueError('normal repetition is not bijective')
        available_contexts = {context(labels[i],features[i]) for i in pools[2]}
        available_errors = sum(n for key,n in errors.items() if key in available_contexts)
        total_errors+=sum(errors.values());covered_errors+=available_errors
        report['branches'][token] = dict(base_report[token],normal_unique_each=size,
            normal_pool_sizes=list(map(len,pools)),normal_slots=len(positions),
            source_error_features=sum(errors.values()),source_error_contexts=len(errors),
            source_errors_with_runtime_supported_normal_context=available_errors,
            normal_contexts_selected=len(details),unique_inputs_each_arm=unique,
            class_draw_counts_each_arm=dict(reference_classes),normal_reuse_bijective=True,
            original_first_cycle_unchanged=True,new_error_draws_identical_first3_new_arms=True)
        report['context_selection'][token] = details
    if total_errors!=675 and sizes==(297,256):raise ValueError('fixed675error exposure differs')
    report['source_error_features'] = total_errors
    report['source_errors_with_runtime_supported_normal_context'] = covered_errors
    report['normal_context_coverage'] = covered_errors/float(total_errors)
    if report['normal_context_coverage']<minimum_coverage:
        raise ValueError('runtime-supported normal-context coverage below frozen minimum')
    return plans,report
