"""Full CAL feasibility screen; no test fitting or production model changes.

Jointly budget false warnings after conservative warning deferral and opposite
S/SS calls. This is a decision-level screen, not proof of rendered API benefit.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import copy
import json
import math
import numpy as np
import torch
import eval_sibilant_directional as service
from train_sibilant_vowel_presence import model, scores
from mdd.sibilant_presence_policy import PAIR, decision


def flagged(view):
    return bool((view['phone'] or {}).get('errorType'))


def exact(label, view):
    phone = view['phone'] or {}
    return (label['label_kind'] not in ('correct', 'ambiguous') and flagged(view)
            and phone.get('alpha_jamo') == label['expected']
            and phone.get('actual_jamo') == label['actual'])


def changed(view, action):
    result = copy.deepcopy(view)
    if action is None:
        return result
    if result['phone'] is None:
        raise ValueError('policy attempted to change an unavailable API phone')
    phone = result['phone']
    if action['kind'] == 'defer':
        phone.pop('errorType', None)
        phone['uncertain'] = True
    else:
        phone['actual_jamo'] = action['actual']
        phone['errorType'] = 'screen_pair'
        phone.pop('uncertain', None)
    return result


def counts(labels, views):
    out = collections.Counter()
    for label, view in zip(labels, views):
        kind = label['label_kind']
        if kind == 'correct':
            out['normal_all'] += 1
            out['normal_known'] += view['phone'] is not None
            out['normal_alerts'] += flagged(view)
        elif kind == 'ambiguous':
            out['ambiguous_all'] += 1
            out['ambiguous_alerts'] += flagged(view)
        else:
            out['errors_all'] += 1
            out['exact'] += exact(label, view)
            out['wrong_actual'] += flagged(view) and not exact(label, view)
    return {k: int(out[k]) for k in ('normal_all', 'normal_known', 'normal_alerts',
            'ambiguous_all', 'ambiguous_alerts', 'errors_all', 'exact', 'wrong_actual')}


def row_changes(labels, before, after):
    result = collections.Counter()
    for label, a, b in zip(labels, before, after):
        result['additional_exact'] += exact(label, b) and not exact(label, a)
        result['lost_exact'] += exact(label, a) and not exact(label, b)
        if label['label_kind'] == 'correct':
            result['additional_normal_alerts'] += flagged(b) and not flagged(a)
            result['removed_normal_alerts'] += flagged(a) and not flagged(b)
    return dict(result)


def calibrate(features, labels, ps, av):
    by_feature = collections.defaultdict(list)
    for k, label in enumerate(labels):
        if label['feature_index'] is not None:
            by_feature[label['feature_index']].append(k)
    before = [z['existing'] for z in labels]
    baseline = counts(labels, before)
    policy = {'defer': {}, 'pair': dict.fromkeys(PAIR)}
    # A runtime feature with absent/ambiguous GT mapping is a safety negative,
    # never a positive. No such feature may change under the CAL policy.
    for token in PAIR:
        ci = 0 if token == 'ㅅ' else 1
        eligible = [i for i, f in enumerate(features) if f['canonical'] == token
                    and f['baseline_warning'] and av[i] and ps[i].argmax() == ci]
        negatives = []
        for i in eligible:
            ks = by_feature[i]
            safe = (len(ks) == 1 and labels[ks[0]]['label_kind'] == 'correct'
                    and labels[ks[0]]['expected'] == labels[ks[0]]['actual'] == token
                    and before[ks[0]]['phone'] is not None)
            if not safe:
                negatives.append(float(ps[i, ci]))
        threshold = max(.5, float(np.nextafter(max(negatives), np.inf))) if negatives else .5
        policy['defer'][token] = threshold if threshold <= 1. else None
    actions = [decision(f, ps[i], av[i], policy) for i, f in enumerate(features)]
    after = [changed(v, actions[z['feature_index']]) if z['feature_index'] is not None else copy.deepcopy(v)
             for z, v in zip(labels, before)]
    deferred = counts(labels, after)
    if deferred['exact'] != baseline['exact'] or deferred['wrong_actual'] != baseline['wrong_actual']:
        raise AssertionError('CAL known error was deferred')
    curves = []
    for token in PAIR:
        alt = 1 if token == 'ㅅ' else 0
        eligible = [i for i, f in enumerate(features) if f['canonical'] == token and av[i]
                    and ps[i].argmax() == alt and float(ps[i, alt]) >= .5
                    and not (f['baseline_warning'] and f['baseline_actual'] == PAIR[token])]
        groups = collections.defaultdict(list)
        for i in eligible:
            groups[float(ps[i, alt])].append(i)
        curve = [{'threshold': None, 'normal_alerts': 0, 'exact': 0, 'wrong_actual': 0,
                  'ambiguous_alerts': 0, 'unknown_changes': 0, 'changes': 0}]
        delta = dict(curve[0])
        for score in sorted(groups, reverse=True):
            for i in groups[score]:
                delta['changes'] += 1
                if len(by_feature[i]) != 1:
                    delta['unknown_changes'] += 1
                for k in by_feature[i]:
                    label = labels[k]
                    if after[k]['phone'] is None:
                        delta['unknown_changes'] += 1
                        continue
                    b = changed(after[k], {'kind': 'pair', 'actual': PAIR[token]})
                    a = counts([label], [after[k]])
                    c = counts([label], [b])
                    for key in ('normal_alerts', 'exact', 'wrong_actual', 'ambiguous_alerts'):
                        delta[key] += c[key] - a[key]
            delta['threshold'] = score
            curve.append(dict(delta))
        curves.append(curve)
    alternatives = {}
    for name, budget in [('one_percent', math.floor(baseline['normal_known'] * .01)),
                         ('non_worsening_intermediate', baseline['normal_alerts'])]:
        best = None
        for a in curves[0]:
            for b in curves[1]:
                fp = deferred['normal_alerts'] + a['normal_alerts'] + b['normal_alerts']
                tp = deferred['exact'] + a['exact'] + b['exact']
                wrong = deferred['wrong_actual'] + a['wrong_actual'] + b['wrong_actual']
                ambig = deferred['ambiguous_alerts'] + a['ambiguous_alerts'] + b['ambiguous_alerts']
                if (fp > budget or tp <= baseline['exact'] or wrong > baseline['wrong_actual']
                        or ambig > baseline['ambiguous_alerts']
                        or a['unknown_changes'] + b['unknown_changes']):
                    continue
                rank = (tp, -fp, -wrong, -a['changes'] - b['changes'])
                if best is None or rank > best[0]:
                    best = (rank, a, b)
        if best is None:
            alternatives[name] = {'feasible': False, 'normal_alert_budget': budget}
            continue
        selected = copy.deepcopy(policy)
        selected['pair'] = {t: c['threshold'] for t, c in zip(PAIR, best[1:])}
        acts = [decision(f, ps[i], av[i], selected) for i, f in enumerate(features)]
        views = [changed(v, acts[z['feature_index']]) if z['feature_index'] is not None else copy.deepcopy(v)
                 for z, v in zip(labels, before)]
        alternatives[name] = {'feasible': True, 'normal_alert_budget': budget, 'policy': selected,
                              'counts': counts(labels, views), 'changes': row_changes(labels, before, views)}
    return {'baseline': baseline, 'defer_policy': policy, 'after_defer': deferred,
            'defer_changes': row_changes(labels, before, after),
            'feature_mapping': dict(collections.Counter(len(by_feature[i]) for i in range(len(features)))),
            'pair_curve_sizes': [len(c) for c in curves], 'alternatives': alternatives}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(a.protocol).read_text())
    cache = Path(p['cal_cache'])
    data = dict(np.load(cache / 'data.npz'))
    features = service.read_jsonl(cache / 'data.features.jsonl')
    labels = service.read_jsonl(cache / 'data.labels.jsonl')
    torch.set_num_threads(2)
    report = {'scope': 'CAL decision screen, no rendered API or independent evaluation claim',
              'batch': p['batch'], 'all_sites': len(labels), 'all_inputs': len(features),
              'deployment': False, 'arms': {}}
    for arm, path in p['artifacts'].items():
        encoder, head = model(p, torch.load(path, map_location='cpu'))
        ps, av, sec = scores(encoder, head, data, batch=p['batch'])
        np.savez(out / (arm + '.scores.npz'), probabilities=ps, available=av)
        result = calibrate(features, labels, ps, av)
        result.update(seconds=sec, available=int(av.sum()))
        report['arms'][arm] = result
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(arm, json.dumps(result, ensure_ascii=False), flush=True)
        del encoder, head
        torch.cuda.empty_cache()
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')


if __name__ == '__main__':
    main()
