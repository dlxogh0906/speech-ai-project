"""Describe frozen CTC evidence; descriptive ROC points are not chosen policies."""
import argparse
import collections
import json
import math
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tasks/pronunciation-assessment/src'))
from mdd.sibilant_diagnostic_audit import grouped_bootstrap


def read(path):
    return [json.loads(x) for x in Path(path).open() if x.strip()]


def quantiles(xs):
    xs = [float(v) for v in xs if v is not None and np.isfinite(v)]
    return {'n': len(xs), 'min_p25_median_p75_max': np.percentile(xs, [0, 25, 50, 75, 100]).tolist() if xs else None}


def auc(pos, neg):
    neg = np.sort(neg)
    if not len(pos) or not len(neg):
        return None
    lo, hi = np.searchsorted(neg, pos, side='left'), np.searchsorted(neg, pos, side='right')
    return float(np.mean((lo + hi) * .5 / len(neg)))


def positive_counts(rows, key):
    vals = [x[key] for x in rows if x.get(key) is not None]
    return {'available': len(vals), 'positive': sum(x > 0 for x in vals), 'distribution': quantiles(vals)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    run = Path(args.run)
    done = json.loads((run / 'finished.json').read_text())
    if done['smoke_limit']:
        raise ValueError('full run required')
    p = json.loads((run / 'protocol.json').read_text())
    sites = read(run / 'sites.jsonl'); ledger = read(p['ledger'])
    original = {(z['uid'], z['q']): z for z in ledger}
    if len(sites) != len(ledger) or len({(z['uid'], z['q']) for z in sites}) != len(ledger):
        raise ValueError('ledger not completely covered')
    for z in sites:
        z['phone_available'] = original[(z['uid'], z['q'])]['view']['phone'] is not None
    targets = [z for z in sites if z.get('target126')]
    normals = [z for z in sites if z['label_kind'] == 'correct']
    errors = [z for z in sites if z['label_kind'] not in ('correct', 'ambiguous')]
    controls = [z for z in sites if z.get('matched_normal_control')]
    site_index = {(z['uid'], z['q']): z for z in sites}
    matched = {tuple(z['target']): [tuple(k) for k in z['controls']]
               for z in json.loads((run / 'matched_controls.json').read_text())}
    if len(targets) != 126:
        raise ValueError('target126 incomplete')
    report = {'protocol': p, 'complete': done, 'sites': len(sites),
        'status': dict(collections.Counter(z['status'] for z in sites)),
        'target126': {'sites': len(targets), 'utterances': len({z['uid'] for z in targets}),
            'kinds': dict(collections.Counter(z['label_kind'] for z in targets)),
            'whole_TAG_minus_LAB': positive_counts(targets, 'whole_TAG_minus_LAB'),
            'single_site_actual_minus_expected': positive_counts(targets, 'oracle_single_site_delta'),
            'TAG_background_actual_minus_expected': positive_counts(targets, 'oracle_TAG_background_delta')},
        'matched_control_sites': len(controls)}
    td = report['target126']
    td['sum_recovers_negative_best_path'] = sum(z['alignment_diagnostic']['best_path_delta'] <= 0 < z['alignment_diagnostic']['sum_delta'] for z in targets)
    td['alignment_shift_abs_ms'] = quantiles([abs(z['alignment_diagnostic']['alignment_mean_shift_ms']) for z in targets if 'alignment_mean_shift_ms' in z['alignment_diagnostic']])
    td['raw_actual_wins_on_expected_span'] = sum(z['alignment_diagnostic'].get('expected_best_span', {}).get('actual_wins_any_frame', False) for z in targets)
    td['raw_actual_wins_on_union_span'] = sum(z['alignment_diagnostic'].get('union_best_spans', {}).get('actual_wins_any_frame', False) for z in targets)
    td['raw_actual_wins_in_C_next_token_window'] = sum(z['alignment_diagnostic'].get('consonant_next_token_window', {}).get('actual_wins_any_frame', False) for z in targets)
    report['directions'] = {}
    for expected, actual in [('ㅅ', 'ㅆ'), ('ㅆ', 'ㅅ')]:
        all_errors = [z for z in errors if z['expected'] == expected and z['actual'] == actual]
        pp = [z for z in all_errors if z.get('pair_delta') is not None]
        mm = [z for z in targets if z['expected'] == expected and z['actual'] == actual]
        nn = [z for z in normals if z['expected'] == expected and z.get('pair_delta') is not None]
        control_keys = {k for z in mm for k in matched[(z['uid'], z['q'])]}
        cc = [site_index[k] for k in sorted(control_keys)]
        concordance = [float(z['pair_delta'] > site_index[k]['pair_delta']) +
                       .5 * float(z['pair_delta'] == site_index[k]['pair_delta'])
                       for z in mm for k in matched[(z['uid'], z['q'])]]
        shifts = [abs(z['alignment_diagnostic']['alignment_mean_shift_ms']) for z in mm]
        d = {'errors_all': len(all_errors), 'errors_scored': len(pp), 'missed_targets': len(mm),
             'normals_scored': len(nn), 'matched_normals': len(cc),
             'pair_margin_errors': quantiles([z['pair_delta'] for z in pp]),
             'pair_margin_misses': quantiles([z['pair_delta'] for z in mm]),
             'pair_margin_normals': quantiles([z['pair_delta'] for z in nn]),
             'AUC_all_accepted_pairs_vs_normals': auc([z['pair_delta'] for z in pp], [z['pair_delta'] for z in nn]),
             'AUC_missed_pairs_vs_matched_normals_DIAGNOSTIC': auc([z['pair_delta'] for z in mm], [z['pair_delta'] for z in cc]),
             'within_context_pairwise_concordance': {'links': len(concordance), 'fraction': float(np.mean(concordance))},
             'miss_alignment_shift_abs_ms': quantiles(shifts),
             'miss_shift_at_least_40ms': sum(x >= 40 for x in shifts)}
        d['raw_peak_on_expected_span'] = {name: quantiles([z['alignment_diagnostic']['expected_best_span']['actual_peak_p'] for z in zs])
                                        for name, zs in [('misses', mm), ('matched_normal_alternative', cc)]}
        # This is a descriptive, optimistic in-sample operating envelope.
        # No threshold is exported as an acceptance policy, and it cannot be
        # added to baseline warnings already exceeding the normal FP budget.
        envelope = []
        neg = np.asarray([z['pair_delta'] for z in nn])
        pos = np.asarray([z['pair_delta'] for z in pp])
        miss = np.asarray([z['pair_delta'] for z in mm])
        choices = np.unique(np.r_[neg, pos, np.inf])
        for rate in (.0, .005, .01, .02, .05):
            budget = math.floor(rate * len(neg))
            valid = [v for v in choices if np.sum(neg > v) <= budget]
            cutoff = min(valid)
            envelope.append({'normal_FP_budget_rate': rate, 'FP': int(np.sum(neg > cutoff)),
                'TP_accepted_pairs': int(np.sum(pos > cutoff)), 'recovered_target_misses': int(np.sum(miss > cutoff)),
                'recall_accepted_pairs': float(np.mean(pos > cutoff)) if len(pos) else None})
        d['in_sample_ROC_envelope_NOT_VALIDATED_POLICY'] = envelope
        report['directions'][expected + '→' + actual] = d
    nn = [z['pair_delta'] for z in normals if z.get('pair_delta') is not None]
    pp = [z['pair_delta'] for z in errors if z['label_kind'] == 'pair_error' and z.get('pair_delta') is not None]
    mm = [z['pair_delta'] for z in targets if z['label_kind'] == 'pair_error']
    neg, pos, miss = np.asarray(nn), np.asarray(pp), np.asarray(mm)
    choices = np.unique(np.r_[neg, pos, np.inf])
    cutoff = min(v for v in choices if np.sum(neg > v) <= math.floor(.01 * len(neg)))
    report['shared_pair_margin_one_percent_in_sample_envelope'] = {
        'scope': 'One common margin cutoff across both directions; descriptive only; replaces pair decisions, not an add-on to existing warnings.',
        'normal_scored': len(nn), 'FP': int(np.sum(neg > cutoff)), 'accepted_pair_errors': len(pp),
        'TP': int(np.sum(pos > cutoff)), 'target_pair_misses': len(mm),
        'recovered_target_misses': int(np.sum(miss > cutoff))}
    report['fixed_zero_margin_diagnostic'] = {}
    for arm in ('pair', 'six_candidate'):
        br = []; changes = []
        for z in sites:
            norm = z['label_kind'] == 'correct'; err = z['label_kind'] not in ('correct', 'ambiguous')
            add = False; predicted = None
            if z['status'] == 'scored' and not z['baseline_warning']:
                if arm == 'pair':
                    add = z['pair_delta'] is not None and z['pair_delta'] > 0
                    predicted = {'ㅅ': 'ㅆ', 'ㅆ': 'ㅅ'}[z['expected']]
                else:
                    predicted = z['runtime_winner']
                    add = predicted != z['expected'] and z['runtime_candidate_deltas'][predicted] > 0
                    if predicted == '<delete>':
                        predicted = None
            new_exact = bool(err and add and predicted == z['actual'])
            br.append(dict(speaker=z['speaker'], errors=int(err), normal_known=int(norm and z['phone_available']),
                exact=int(z['baseline_exact']), FP=int(norm and z['baseline_warning']),
                delta_exact=int(new_exact), delta_FP=int(norm and add),
                candidate_exact=int(z['baseline_exact'] or new_exact), candidate_FP=int(norm and (z['baseline_warning'] or add))))
            if add:
                changes.append({k: z.get(k) for k in ('uid', 'q', 'expected', 'actual', 'label_kind', 'target126')})
                changes[-1].update(predicted=predicted, exact=new_exact)
        metrics = {'baseline_exact_recall': ('exact', 'errors'), 'baseline_false_alarm': ('FP', 'normal_known'),
                   'candidate_exact_recall': ('candidate_exact', 'errors'), 'candidate_false_alarm': ('candidate_FP', 'normal_known'),
                   'recall_difference': ('delta_exact', 'errors'), 'false_alarm_difference': ('delta_FP', 'normal_known')}
        report['fixed_zero_margin_diagnostic'][arm] = {
            'scope': 'Decision-only add-on, original warnings preserved; no API integration or calibrated threshold.',
            'speaker_cluster_CI': grouped_bootstrap(br, metrics),
            'prompt_cluster_CI': grouped_bootstrap([dict(x, speaker=z['item']) for x, z in zip(br, sites)], metrics),
            'changes': changes}
    report['baseline_coverage'] = {'errors_all': len(errors), 'errors_accepted': sum(z['status'] != 'input_rejected' for z in errors),
        'exact': sum(z['baseline_exact'] for z in errors), 'normal_all': len(normals),
        'normal_rejected': sum(z['status'] == 'input_rejected' for z in normals),
        'normal_known': sum(z['phone_available'] for z in normals), 'normal_false_alerts': sum(z['baseline_warning'] for z in normals)}
    report['limits'] = p['limitations'] + [
        'ROC envelopes reuse CAL labels and do not establish a usable threshold, confidence or deployment gain.',
        'Normal controls are matched for site context, not independently adjudicated acoustics.',
        'Speaker and prompt bootstrap intervals are separate marginal cluster analyses, not joint two-way cluster uncertainty.',
        'No new backbone comparison, training, listening adjudication or arbitrary-text generalization validation.']
    (run / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('sites', 'status', 'target126', 'baseline_coverage', 'directions')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
