"""Local label co-location in original TAG/LAB, not acoustic-boundary truth.

Unknown-span minimum-edit matching is a declared alignment model, not proof
of all possible unknown transcripts. Require an additional independent source
constraint: both literal phoneme tiers identify the SAME singleton C/V slots.
Times use the original TAG parser's1ms rounding; no timing enters inference.
"""
import math
from collections import Counter
from mdd.sibilant_partial_annotation import annotation_sequence

SCHEMA = 'partial_alloptimal_plus_unique_TAG_LAB_literal_same_1ms_CV_slots_v1'


def _bounds(interval):
    if interval is None or not all(math.isfinite(z) for z in interval[:2]):
        return None
    return round(interval[0],3),round(interval[1],3)


def witness_sites(row,allowed):
    expected = annotation_sequence(row['lab_phones'],allowed)
    actual = annotation_sequence(row['tag_phones'],allowed)
    counts = [Counter(_bounds(x) for x in seq['geometry'] if _bounds(x) is not None)
              for seq in (expected,actual)]
    results = []
    for site in row['sites']:
        result = dict(site,source_time_witness=False,usable_training_label=False)
        if site['label_kind'] not in ('correct','pair_error'):
            result['witness_state'] = 'prior_correspondence_unresolved'
            results.append(result);continue
        q,a = site['partial_q'],site['partial_actual_index']
        if (expected['tokens'][q:q+2] != [site['expected'],site['following']]
                or actual['tokens'][a:a+2] != [site['actual'],site['following']]):
            raise ValueError('provisional literal C/V labels changed')
        c,v = expected['geometry'][q:q+2]
        ca,va = actual['geometry'][a:a+2]
        if c is None or v is None:
            result['witness_state'] = 'expected_CV_not_literal_singletons'
            results.append(result);continue
        if (_bounds(c) is None or _bounds(v) is None or _bounds(c) != _bounds(ca)
                or _bounds(v) != _bounds(va)):
            result['witness_state'] = 'TAG_LAB_CV_slots_disagree'
            results.append(result);continue
        if any(counter[_bounds(x)] != 1 for counter in counts for x in (c,v)):
            result['witness_state'] = 'nonunique_source_time_slots'
            results.append(result);continue
        state = 'same_unique_literal_CV_slots'
        lo,hi = ca[0],va[1]
        for phones,own in [(row['lab_phones'],{expected['raw_indices'][q],expected['raw_indices'][q+1]}),
                           (row['tag_phones'],{actual['raw_indices'][a],actual['raw_indices'][a+1]})]:
            for j,(start,end,_) in enumerate(phones):
                try:
                    start,end = float(start),float(end)
                except (TypeError,ValueError):
                    state = 'unlocated_annotation_interval';break
                if (not math.isfinite(start) or not math.isfinite(end)
                        or not 0 <= start <= end <= row['frames']/16000.):
                    state = 'unlocated_annotation_interval';break
                if j not in own and end > lo and start < hi:
                    state = 'extra_annotation_inside_CV';break
            if state != 'same_unique_literal_CV_slots':break
        result['witness_state'] = state
        result['source_time_witness'] = state == 'same_unique_literal_CV_slots'
        # Runtime expected-position mapping, anchor availability and waveform
        # ROI support are deliberately still unchecked at this annotation stage.
        results.append(result)
    return results
