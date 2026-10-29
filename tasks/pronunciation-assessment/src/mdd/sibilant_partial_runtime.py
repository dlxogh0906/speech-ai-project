"""Attach partial source labels only after automatic runtime observations exist."""
import collections
from mdd.sibilant_partial_annotation import partial_mapping


def runtime_labels(row,canonical,phones,by_q,rejected):
    links = collections.defaultdict(list)
    if canonical is not None:
        mapping,_ = partial_mapping(canonical,row['expected_partial'])
        for q in range(len(canonical)-1):
            if len(mapping[q]) == len(mapping[q+1]) == 1:
                a,v = next(iter(mapping[q])),next(iter(mapping[q+1]))
                if isinstance(a,int) and isinstance(v,int) and v == a+1:
                    links[a].append(q)
    labels = []
    for site in row['sites']:
        source_kind = site['label_kind']
        known = site['source_time_witness'] and source_kind in ('correct','pair_error')
        q = site['partial_q']
        candidates = [i for i in links[q] if canonical[i:i+2] == [site['expected'],site['following']]] if known else []
        rq = candidates[0] if len(candidates) == 1 else None
        state = ('input_rejected' if rejected else 'source_label_uncertain' if not known
                 else 'runtime_reference_unmapped_or_disagrees' if rq is None
                 else 'mapped')
        label = dict(site,uid=row['uid'],speaker=row['speaker'],item=row['item'],language=row.get('language'),
            q=q,q_runtime=rq,feature_index=by_q.get(rq) if known and not rejected else None,
            source_label_kind=source_kind,label_kind=source_kind if known else 'ambiguous',
            provisional_actual=site.get('actual'),actual=site.get('actual') if known else None,
            alignment_ambiguous=not known,source_runtime_mapping_state=state,
            existing={'state':state,'phone':phones[rq] if rq is not None and not rejected else None})
        labels.append(label)
    by_input = collections.defaultdict(list)
    for z in labels:
        if z['feature_index'] is not None:
            by_input[z['feature_index']].append(z)
    for group in by_input.values():
        if len(group) > 1:
            for z in group:
                z['feature_index'] = None
                z['source_runtime_mapping_state'] = 'multiple_source_labels_for_runtime_query'
    return labels
