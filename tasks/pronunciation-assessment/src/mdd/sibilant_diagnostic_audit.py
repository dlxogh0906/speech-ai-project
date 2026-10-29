"""Offline diagnostic accounting; observation states are not physical causes."""
import collections
import numpy as np


def flagged(view):
    return bool((view.get('phone') or {}).get('errorType'))


def exact(label, view):
    p = view.get('phone') or {}
    return (label['label_kind'] not in ('correct', 'ambiguous') and flagged(view)
            and p.get('alpha_jamo') == label['expected']
            and p.get('actual_jamo') == label.get('actual'))


def error_stage(label, view, raw_site=None):
    """One exclusive observable stage; never infer a bad TAG from disagreement."""
    if view['state'] == 'input_rejected':
        return 'input_rejected'
    if view.get('phone') is None:
        return 'reference_runtime_mapping_unresolved'
    if view['state'] == 'runtime_canonical_disagrees':
        return 'reference_runtime_canonical_disagrees'
    if exact(label, view):
        return 'exact_error_warning'
    phone = view['phone']
    if phone.get('actual_jamo') == label.get('actual'):
        return 'TAG_actual_in_final_phone_but_warning_suppressed'
    if flagged(view):
        return 'warning_with_wrong_actual'
    pred = (raw_site or {}).get('pred', phone.get('actual_jamo'))
    if pred is None:
        return 'decoder_C_unlinked_or_deleted'
    if pred == label['expected']:
        return 'decoder_expected_phone_instead_of_TAG_actual'
    return 'decoder_other_phone_instead_of_TAG_actual'


def false_warning_stage(label, view, raw_site=None):
    if not flagged(view):
        return None
    if view['state'] == 'runtime_canonical_disagrees':
        return 'reference_disagreement_requires_normative_review'
    phone = view['phone']
    if phone.get('actual_jamo') is None:
        return 'deletion_warning_requires_audio_and_alignment_review'
    if (raw_site or {}).get('pred') == label['expected']:
        return 'postdecoder_override_requires_policy_review'
    return 'substitution_warning_acoustic_alignment_or_TAG_unresolved'


def grouped_bootstrap(rows, metrics, repetitions=2000, seed=20261026):
    """Paired speaker cluster resampling, retaining zero-event speakers.

    metrics maps names to (numerator field, denominator field). Numerators can
    be signed for paired differences. The policy must already be frozen; these
    conditional intervals do not undo adaptive selection or repeated CAL use.
    """
    speakers = sorted({z['speaker'] for z in rows})
    lookup = {k: i for i, k in enumerate(speakers)}
    fields = sorted({k for pair in metrics.values() for k in pair})
    data = {k: np.zeros(len(speakers), np.float64) for k in fields}
    for row in rows:
        for k in fields:
            data[k][lookup[row['speaker']]] += row.get(k, 0)
    rng = np.random.RandomState(seed)
    sample = rng.randint(0, len(speakers), size=(repetitions, len(speakers)))
    result = {}
    for name, (num, den) in metrics.items():
        n, d = data[num].sum(), data[den].sum()
        dd = data[den][sample].sum(1)
        nn = data[num][sample].sum(1)
        valid = dd > 0
        values = nn[valid] / dd[valid]
        result[name] = {'numerator': float(n), 'denominator': float(d),
            'estimate': float(n / d) if d else None,
            'percentile95': np.percentile(values, [2.5, 97.5]).tolist() if len(values) else None,
            'invalid_zero_denominator_replicates': int((~valid).sum()),
            'speakers': len(speakers), 'replicates': repetitions}
    return result


def summary(rows):
    out = collections.Counter()
    for z in rows:
        out['sites'] += 1
        kind = z['label_kind']
        out[kind] += 1
        if kind == 'correct':
            out['normal_rejected'] += z['view']['state'] == 'input_rejected'
            out['normal_phone_available'] += z['view']['phone'] is not None
            out['normal_alerts'] += flagged(z['view'])
        elif kind != 'ambiguous':
            out['errors'] += 1
            out['errors_rejected'] += z['view']['state'] == 'input_rejected'
            out['exact'] += exact(z, z['view'])
            out['wrong_actual_warning'] += flagged(z['view']) and not exact(z, z['view'])
    return dict(out)
