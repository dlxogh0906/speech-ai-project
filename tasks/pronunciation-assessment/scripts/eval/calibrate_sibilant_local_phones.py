"""CAL: pair-only versus full observed-consonant output from the same decoder.

Two expected-consonant thresholds in both arms, not one fitted threshold per
observed phone. Preserve exact known errors and ambiguous/unknown observations.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src')]
import argparse
import collections
import copy
import json
import math
import numpy as np
import eval_sibilant_directional as service
from eval_sibilant_presence_screen import counts, changed, exact, row_changes
from mdd.sibilant_local_phone_policy import local_phone_decision
from mdd.sibilant_transition import PAIR


def fit(features, labels, queries, mode):
    by_feature = collections.defaultdict(list)
    for k, z in enumerate(labels):
        if z['feature_index'] is not None:
            by_feature[z['feature_index']].append(k)
    before = [z['existing'] for z in labels]
    baseline = counts(labels, before)
    policy = {'mode': mode, 'defer': {}, 'actual': dict.fromkeys(PAIR)}
    for token in PAIR:
        negatives = []
        for i, (f, q) in enumerate(zip(features, queries)):
            if not (f['canonical'] == token and f['baseline_warning'] and q['state'] == 'local_C_V' and q['actual'] == token):
                continue
            ks = by_feature[i]
            safe = (len(ks) == 1 and labels[ks[0]]['label_kind'] == 'correct'
                    and labels[ks[0]]['expected'] == labels[ks[0]]['actual'] == token
                    and before[ks[0]]['phone'] is not None)
            if not safe:
                negatives.append(q['confidence'])
        threshold = max(.5, float(np.nextafter(max(negatives), np.inf))) if negatives else .5
        policy['defer'][token] = threshold if threshold <= 1 else None
    acts = [local_phone_decision(f, q, policy) for f, q in zip(features, queries)]
    after = [changed(v, acts[z['feature_index']]) if z['feature_index'] is not None else copy.deepcopy(v)
             for z, v in zip(labels, before)]
    deferred = counts(labels, after)
    if deferred['exact'] != baseline['exact'] or deferred['wrong_actual'] != baseline['wrong_actual']:
        raise AssertionError('CAL known-error warning was deferred')
    curves = []
    for token in PAIR:
        groups = collections.defaultdict(list)
        for i, (f, q) in enumerate(zip(features, queries)):
            if (f['canonical'] != token or q['state'] != 'local_C_V' or q['actual'] == token
                    or q['confidence'] < .5 or mode == 'pair' and q['actual'] != PAIR[token]
                    or f['baseline_warning'] and f['baseline_actual'] == q['actual']):
                continue
            groups[q['confidence']].append(i)
        curve = [{'threshold': None, 'normal_alerts': 0, 'exact': 0, 'wrong_actual': 0,
                  'unknown_changes': 0, 'lost_exact': 0, 'changes': 0}]
        delta = dict(curve[0])
        for confidence in sorted(groups, reverse=True):
            for i in groups[confidence]:
                delta['changes'] += 1
                if len(by_feature[i]) != 1:
                    delta['unknown_changes'] += 1
                for k in by_feature[i]:
                    z = labels[k]
                    if z['label_kind'] == 'ambiguous' or after[k]['phone'] is None:
                        delta['unknown_changes'] += 1
                        continue
                    newer = changed(after[k], {'kind': 'actual', 'actual': queries[i]['actual']})
                    delta['lost_exact'] += exact(z, after[k]) and not exact(z, newer)
                    old, new = counts([z], [after[k]]), counts([z], [newer])
                    for key in ('normal_alerts', 'exact', 'wrong_actual'):
                        delta[key] += new[key] - old[key]
            delta['threshold'] = confidence
            curve.append(dict(delta))
        curves.append(curve)
    alternatives = {}
    for name, budget in [('one_percent', math.floor(baseline['normal_known'] * .01)),
                         ('non_worsening_intermediate', baseline['normal_alerts'])]:
        best = None
        for a in curves[0]:
            if a['unknown_changes'] or a['lost_exact']:
                continue
            for b in curves[1]:
                fp = deferred['normal_alerts'] + a['normal_alerts'] + b['normal_alerts']
                tp = deferred['exact'] + a['exact'] + b['exact']
                wrong = deferred['wrong_actual'] + a['wrong_actual'] + b['wrong_actual']
                if (b['unknown_changes'] or b['lost_exact'] or fp > budget
                        or tp <= baseline['exact'] or wrong > baseline['wrong_actual']):
                    continue
                rank = (tp, -fp, -wrong, -a['changes'] - b['changes'])
                if best is None or rank > best[0]:
                    best = (rank, a, b)
        if best is None:
            alternatives[name] = {'feasible': False, 'normal_alert_budget': budget}
            continue
        selected = copy.deepcopy(policy)
        selected['actual'] = {token: c['threshold'] for token, c in zip(PAIR, best[1:])}
        acts = [local_phone_decision(f, q, selected) for f, q in zip(features, queries)]
        views = [changed(v, acts[z['feature_index']]) if z['feature_index'] is not None else copy.deepcopy(v)
                 for z, v in zip(labels, before)]
        alternatives[name] = {'feasible': True, 'normal_alert_budget': budget, 'policy': selected,
                              'counts': counts(labels, views), 'changes': row_changes(labels, before, views)}
    return {'baseline': baseline, 'defer_policy': policy, 'after_defer': deferred,
            'pair_or_full_curve_sizes': [len(c) for c in curves], 'alternatives': alternatives}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    cache = Path(p['cache'])
    if json.loads((cache / 'report.json').read_text())['partition'] != 'cal':
        raise ValueError('CAL-only fitting')
    features = service.read_jsonl(cache / 'data.features.jsonl')
    labels = service.read_jsonl(cache / 'data.labels.jsonl')
    if len(labels) != p['expected_sites']:
        raise ValueError('full CAL denominator changed')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(a.protocol).read_text())
    report = {'scope': p['scope'], 'deployment': False, 'all_sites': len(labels), 'arms': {}}
    for arm, path in p['observations'].items():
        queries = service.read_jsonl(path)
        if len(queries) != len(features) or any((q['uid'], q['q_runtime']) != (f['uid'], f['q_runtime']) for f, q in zip(features, queries)):
            raise ValueError('runtime observation identity differs')
        for mode in ('pair', 'full'):
            result = fit(features, labels, queries, mode)
            report['arms'][arm + '_' + mode] = result
            print(arm, mode, json.dumps(result, ensure_ascii=False), flush=True)
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')


if __name__ == '__main__':
    main()
