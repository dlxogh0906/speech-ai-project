"""Describe frozen M3 reranking by prompt and canonical vowel; no refitting."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('저장소 루트(pyproject.toml)를 찾지 못했습니다')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src')]
from shared.paths import MDD_OUTPUTS  # 경로는 .env 에서 (CLAUDE.md 경로 규칙)
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


RUN = MDD_OUTPUTS/'experiments/m3_candidate_rerank_2026-11-01_v1'
CONDITIONS = ('A_M3_peak', 'B_probability', 'C_segment')
FAMILIES = {'sibilants': ['ㅅ', 'ㅆ'], 'affricates': ['ㅈ', 'ㅉ', 'ㅊ']}
VOWELS = set('ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ')
KNOWN = {'normal', 'substitution', 'deletion'}


def read(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contexts(role):
    result = {}
    for r in read(RUN / (role + '_inputs.jsonl.gz')):
        ref = r['reference']
        for q, expected in enumerate(ref):
            if not any(expected in phones for phones in FAMILIES.values()):
                continue
            following = ref[q + 1] if q + 1 < len(ref) else '<END>'
            vowel = following if following in VOWELS else '<NO_IMMEDIATE_VOWEL>'
            result[r['uid'], q] = dict(expected=expected, vowel=vowel, following=following)
    return result


def groups(r, context):
    return [('all', 'all'), ('family', r['family']), ('item', r['item']),
            ('item_family', r['item'] + '/' + r['family']),
            ('phone_vowel', r['expected'] + '/' + context['vowel']),
            ('item_phone_vowel', r['item'] + '/' + r['expected'] + '/' + context['vowel'])]


def count(r):
    c = collections.Counter(total=1)
    if r['truth_status'] not in KNOWN:
        c['unknown'] = 1
        return c
    error = r['actual'] != r['expected']
    kind = 'error' if error else 'normal'
    c[kind] = 1
    if r['observed']:
        c['observed_' + kind] = 1
        c['fp'] = int(not error and r['warning'])
        c['exact'] = int(error and r['warning'] and r['prediction'] == r['actual'])
        c['wrong_actual'] = int(error and r['warning'] and r['prediction'] != r['actual'])
    return c


def metric_row(key, c):
    role, condition, dimension, group = key
    row = dict(role=role, condition=condition, dimension=dimension, group=group)
    for k in ('total', 'unknown', 'normal', 'error', 'observed_normal', 'observed_error',
              'fp', 'exact', 'wrong_actual'):
        row[k] = c[k]
    for name, a, b in [('FPR', 'fp', 'observed_normal'), ('exact_recall', 'exact', 'error'),
                       ('normal_coverage', 'observed_normal', 'normal'),
                       ('error_coverage', 'observed_error', 'error')]:
        row[name] = c[a] / c[b] if c[b] else None
    return row


def main():
    out = RUN / 'context_diagnostic_v1'
    out.mkdir(exist_ok=False)
    input_names = ['thresholds.json', 'frozen_head.json']
    for role in ('train', 'cal', 'seen', 'rp'):
        input_names += [role + '_inputs.jsonl.gz', role + '_features.jsonl.gz']
    for role in ('seen', 'rp'):
        input_names += [role + '_predictions.jsonl.gz', role + '_report.json']
    before = {name: sha(RUN / name) for name in input_names}
    protocol = dict(
        purpose='Post-hoc localization of frozen reranking successes and failures',
        training_updates=0, acoustic_inference_calls=0, threshold_refits=0,
        context='Immediately following canonical LAB vowel, or explicit missing category; not actual-vowel annotation',
        pair_score='Original emitted-peak logP(alternative)-logP(expected), using saved base log gaps',
        positive='Observed identified TAG equals the alternative; negative TAG equals expected',
        limitations=['Same selected peak and existing observation mask',
                     'RP has three scripts; context, speaker and input length are not jointly controlled',
                     'Sparse cells retained; pooled AUC is not recall at 1% FPR',
                     'Shared-speaker seen/RP subset controls speaker identity only, not training-speaker exposure'],
        source_sha256=sha(Path(__file__)), inputs_sha256=before)
    write_json(out / 'protocol.json', protocol)
    metric_counts = collections.defaultdict(collections.Counter)
    transition_counts = collections.defaultdict(collections.Counter)
    prediction_maps = {}
    features = {}
    ctx = {}
    support = collections.Counter()
    pair_data = collections.defaultdict(list)
    for role in ('train', 'cal', 'seen', 'rp'):
        ctx[role] = contexts(role)
        features[role] = {(r['uid'], r['q']): r for r in read(RUN / (role + '_features.jsonl.gz'))}
        for k, r in features[role].items():
            cc = ctx[role][k]
            assert cc['expected'] == r['expected']
            if not r['observed'] or r['truth_status'] not in KNOWN:
                continue
            support[role, r['family'], r['expected'], str(r['actual']), cc['vowel']] += 1
            phones = FAMILIES[r['family']]
            n = len(phones)
            assert len(r['base']) == 4 * n + 2
            for alt in phones:
                if alt == r['expected'] or r['actual'] not in (alt, r['expected']):
                    continue
                margin = r['base'][n + phones.index(alt)] - r['base'][n + phones.index(r['expected'])]
                for dim, group in [('all', 'all'), ('item', r['item']), ('vowel', cc['vowel']),
                                   ('item_vowel', r['item'] + '/' + cc['vowel'])]:
                    pair_data[role, r['expected'], alt, dim, group].append(
                        (int(r['actual'] == alt), margin, r['speaker'], r['item']))
        if role not in ('seen', 'rp'):
            continue
        pm = collections.defaultdict(dict)
        for r in read(RUN / (role + '_predictions.jsonl.gz')):
            k = (r['uid'], r['q'])
            assert r['condition'] not in pm[k]
            pm[k][r['condition']] = r
            for dim, group in groups(r, ctx[role][k]):
                metric_counts[role, r['condition'], dim, group].update(count(r))
        prediction_maps[role] = pm
        for k, rr in pm.items():
            assert set(rr) == set(CONDITIONS)
            a = rr[CONDITIONS[0]]
            for cond in CONDITIONS[1:]:
                b = rr[cond]
                for name in ('observed', 'truth_status', 'actual', 'expected'):
                    assert a[name] == b[name]
                if a['truth_status'] not in KNOWN or not a['observed']:
                    continue
                error = a['actual'] != a['expected']
                ae = error and a['warning'] and a['prediction'] == a['actual']
                be = error and b['warning'] and b['prediction'] == b['actual']
                t = dict(new_exact=int(be and not ae), lost_exact=int(ae and not be),
                         new_fp=int(not error and b['warning'] and not a['warning']),
                         removed_fp=int(not error and a['warning'] and not b['warning']))
                for dim, group in groups(a, ctx[role][k]):
                    transition_counts[role, cond, dim, group].update(t)
        # Compare full known denominators and frozen-policy counts to original report.
        for r in json.loads((RUN / (role + '_report.json')).read_text()):
            if r['group'] != 'all':
                continue
            c = metric_counts[role, r['condition'], 'all', 'all']
            for name in ('normal', 'error', 'observed_normal', 'observed_error', 'fp', 'exact'):
                assert c[name] == r.get(name, 0), (role, r['condition'], name)
    speakers = {role: {r['speaker'] for r in features[role].values()} for role in ('seen', 'rp')}
    shared = speakers['seen'] & speakers['rp']
    for role, pm in prediction_maps.items():
        for rr in pm.values():
            for r in rr.values():
                if r['speaker'] in shared:
                    metric_counts[role, r['condition'], 'shared_test_speakers', 'all'].update(count(r))
    metrics = [metric_row(k, c) for k, c in sorted(metric_counts.items())]
    transitions = []
    for k, c in sorted(transition_counts.items()):
        role, cond, dim, group = k
        transitions.append(dict(role=role, condition=cond, dimension=dim, group=group,
                                new_exact=c['new_exact'], lost_exact=c['lost_exact'],
                                new_fp=c['new_fp'], removed_fp=c['removed_fp']))
        base = metric_counts[role, CONDITIONS[0], dim, group]
        candidate = metric_counts[role, cond, dim, group]
        assert c['new_exact'] - c['lost_exact'] == candidate['exact'] - base['exact']
        assert c['new_fp'] - c['removed_fp'] == candidate['fp'] - base['fp']
    distributions = []
    for k, values in sorted(pair_data.items()):
        role, expected, alt, dim, group = k
        labels = np.array([x[0] for x in values])
        scores = np.array([x[1] for x in values])
        row = dict(role=role, expected=expected, alternative=alt, dimension=dim, group=group,
                   speakers=len({x[2] for x in values}), items=len({x[3] for x in values}))
        for flag, name in ((0, 'normal'), (1, 'error')):
            subset = scores[labels == flag]
            row[name + '_n'] = len(subset)
            for p in (10, 50, 90, 99):
                row[name + '_p' + str(p)] = float(np.percentile(subset, p)) if len(subset) else None
        row['auc'] = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
        distributions.append(row)
    write_csv(out / 'frozen_policy_contexts.csv', metrics)
    write_csv(out / 'paired_transitions.csv', transitions)
    write_csv(out / 'peak_pair_distributions.csv', distributions)
    write_csv(out / 'observed_label_support.csv', [dict(role=k[0], family=k[1], expected=k[2],
        actual=k[3], vowel=k[4], n=v) for k, v in sorted(support.items())])
    assert before == {name: sha(RUN / name) for name in input_names}
    summary = dict(protocol=protocol, shared_test_speakers=len(shared),
                   overall=[r for r in metrics if r['dimension'] in ('all', 'shared_test_speakers')],
                   item_family=[r for r in metrics if r['role'] == 'rp' and r['dimension'] == 'item_family'],
                   rp_context_transitions=[r for r in transitions if r['role'] == 'rp' and r['dimension'] == 'phone_vowel'],
                   source_unchanged=True, original_report_counts_match=True,
                   transition_count_identities_match=True, output_rows=dict(
                       metrics=len(metrics), transitions=len(transitions), distributions=len(distributions)))
    write_json(out / 'report.json', summary)
    print(json.dumps({k: summary[k] for k in ('shared_test_speakers', 'source_unchanged',
        'original_report_counts_match', 'transition_count_identities_match', 'output_rows')}))


if __name__ == '__main__':
    main()
