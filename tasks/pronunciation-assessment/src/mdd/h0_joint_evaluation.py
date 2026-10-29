"""Fixed-denominator partial-reference metrics for the preregistered fold pilot.

Annotation support is fixed before inference. Prediction ambiguity and temporal
masks reduce Coverage, never the identified normal/error denominators.
"""
import collections
import itertools
import numpy as np
from .h0_partial_reference import UNKNOWN, DELETED
from .h0_partial_scoring import (mapping_indices, value_options, unique_index,
                                 gap_value, partial_ctc_viterbi, rates)


def score_record(record, lp, vocab):
    from .gop_feats_v2 import viterbi_align
    inv = {v:k for k,v in vocab.items()}
    predicted = [inv[t] for t,_ in itertools.groupby(lp.argmax(1)) if t in inv]
    ref = record['reference']; idx = mapping_indices(ref, predicted)
    ids = [None if x == UNKNOWN else vocab[x] for x in ref]
    if None in ids:
        path = partial_ctc_viterbi(lp, ids, list(vocab.values()))
        segments = None if path is None else path['segments']
    else:
        segments = viterbi_align(lp, ids, blank=0)
    # Whole short utterances: no artificial chunk seams. Times only limit scoring.
    centers = 280 + 320*np.arange(len(lp))
    unknown = [o for t,o in zip(ref, record['reference_owners']) if t == UNKNOWN]
    unknown += [o for t,o in zip(record['actual'],record['actual_owners']) if t == UNKNOWN]
    masked = np.zeros(len(lp),bool)
    for o in unknown:
        masked |= ((centers >= round(o['start']*16000)-640) &
                   (centers <= round(o['end']*16000)+640))
    sites = []
    for q, truth in enumerate(record['sites']):
        frames = [] if segments is None else segments[q]
        score = None
        if ids[q] is not None and len(frames):
            block = lp[frames]; other = block.copy(); other[:,ids[q]] = -np.inf
            score = float((other.max(1)-block[:,ids[q]]).mean())
        options = value_options(idx[q],predicted)
        options = [] if options is None else sorted(options)
        anchored = (q>0 and q+1<len(idx) and unique_index(idx[q-1]) is not None
                    and unique_index(idx[q+1]) is not None)
        reason = record['annotation_mask'][q]
        if not reason and len(frames) and masked[frames].any(): reason = 'unknown_time_overlap'
        if not reason and score is None: reason = 'missing_GOP'
        if not reason and len(options) != 1: reason = 'prediction_alignment_ambiguous'
        if not reason and options == [DELETED] and not anchored: reason = 'prediction_deletion_unanchored'
        sites.append(dict(truth, score=score, prediction_options=options,
                          observation_failure=reason, observed=reason is None))
    gaps = []
    for g,truth in enumerate(record['truth_gaps']):
        guess = gap_value(ref,predicted,idx,g)
        # Keep the same conservative scoring support as the partial RP policy.
        available = (0<g<len(sites) and sites[g-1]['observed'] and sites[g]['observed'])
        gaps.append(dict(g=g,actual=truth,predicted=list(guess) if guess is not None and available else None))
    return dict(**{k:record[k] for k in ('uid','speaker','item','task','language','role')},
                predicted=predicted,sites=sites,gaps=gaps,alignment_available=segments is not None)


def site_counts(sites, threshold):
    c = collections.Counter()
    for s in sites:
        c['reference_entries'] += 1
        c['known_reference_sites'] += s['expected'] != UNKNOWN
        if s['status'] not in ('normal','substitution','deletion'):
            c['annotation_'+s['status']] += 1
            continue
        error = s['status'] != 'normal'; name = 'error' if error else 'normal'
        c[name] += 1; c['eligible_sites'] += 1
        if error: c[s['status']] += 1
        if not s['observed']:
            c['unobserved_'+name] += 1
            c['failure_'+str(s['observation_failure'])+'_'+name] += 1
            c['exact_upper'] += error; c['fp_upper'] += not error
            continue
        c['observed_'+name] += 1
        pred = s['prediction_options'][0]
        warned = pred != s['expected'] and (threshold is None or s['score'] >= threshold)
        fp = warned and not error; exact = warned and error and pred == s['actual']
        c['fp'] += fp; c['exact'] += exact; c['fp_upper'] += fp; c['exact_upper'] += exact
        c['wrong_actual'] += warned and error and not exact
        if s['status'] == 'deletion':
            c['observed_deletion'] += 1; c['deletion_exact'] += exact
    return dict(c)


def gap_counts(gaps):
    c = collections.Counter()
    for g in gaps:
        if g['actual'] is None:
            c['annotation_unknown'] += 1; continue
        error = bool(g['actual']); kind = 'error' if error else 'normal'; c[kind] += 1
        if g['predicted'] is None:
            c['unobserved_'+kind] += 1
            c['exact_upper'] += error; c['fp_upper'] += not error
            continue
        c['observed_'+kind] += 1
        fp = not error and bool(g['predicted'])
        exact = error and g['predicted'] == g['actual']
        c['fp'] += fp; c['exact'] += exact; c['fp_upper'] += fp; c['exact_upper'] += exact
    return dict(c)


def calibrate(records):
    sites = [s for r in records for s in r['sites'] if s['status']=='normal' and s['observed']]
    budget = len(sites)//100
    values = sorted([s['score'] for s in sites if s['prediction_options'][0]!=s['expected']],reverse=True)
    threshold = None if len(values)<=budget else float(np.nextafter(values[budget],np.inf))
    actual = sum(threshold is None or v>=threshold for v in values)
    assert actual<=budget
    return dict(threshold=threshold,observed_normal_denominator=len(sites),budget=budget,actual_fp=actual,
                rule='one percent of observed CAL normals; boundary ties excluded; no inner/outer fitting')


def aggregate(records, threshold, phones):
    from .h0_phone_audit import surface_position
    groups = collections.defaultdict(collections.Counter)
    for phone in phones: groups['phone:'+phone]  # Keep zero support, rates remain null.
    gaps = collections.defaultdict(collections.Counter)
    for r in records:
        common = ['all','item:'+r['item'],'task:'+r['task'],'L1:'+r['language']]
        ref = [s['expected'] for s in r['sites']]
        positions = [surface_position(ref,i) for i in range(len(ref))]
        for s,pos in zip(r['sites'],positions):
            counts = site_counts([s],threshold)
            keys = common+['phone:'+s['expected'],'position:'+pos,'truth:'+s['status']]
            for key in keys: groups[key].update(counts)
        for key in common: gaps[key].update(gap_counts(r['gaps']))
    pack = lambda cs:{k:dict(counts=dict(v),rates=rates(v)) for k,v in sorted(cs.items())}
    return dict(sites=pack(groups),gaps=pack(gaps),recordings=len(records))


def pilot_gate(c0, weighted):
    a,b = c0['sites']['all'],weighted['sites']['all']
    def value(x,key): return x['rates'][key]
    checks = dict(normal_FPR=value(b,'observed_normal_fpr') is not None and value(b,'observed_normal_fpr')<=.01,
        exact_recall_improves=value(b,'fixed_exact_recall_lower') is not None and value(a,'fixed_exact_recall_lower') is not None
                              and value(b,'fixed_exact_recall_lower')>value(a,'fixed_exact_recall_lower'))
    for kind in ('normal','error'):
        k=kind+'_observation_coverage'; checks[k]=value(a,k) is not None and value(b,k) is not None and value(b,k)>=value(a,k)
    protected=[]
    for key,old in c0['sites'].items():
        if (key.startswith('phone:') or key=='truth:deletion') and old['counts'].get('error',0)>=30:
            new=weighted['sites'][key]
            protected.append(dict(group=key,drop=value(old,'fixed_exact_recall_lower')-value(new,'fixed_exact_recall_lower')))
    ga,gb=c0['gaps']['all'],weighted['gaps']['all']
    if ga['counts'].get('error',0)>=30:
        protected.append(dict(group='insertion',drop=value(ga,'fixed_exact_recall_lower')-value(gb,'fixed_exact_recall_lower')))
    checks['protected_error_recall']=all(x['drop']<=.05 for x in protected)
    x,y=value(ga,'observed_normal_fpr'),value(gb,'observed_normal_fpr')
    checks['normal_gap_fpr']=x is not None and y is not None and y-x<=.002
    return dict(checks=checks,protected_groups=protected,continue_folds=all(checks.values()),
                scope='fixed inner pilot compute decision, not population impossibility or final success')
