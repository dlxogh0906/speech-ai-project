"""Build TRAIN-only matched/shuffled pair links with identical batch exposure."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import json
import numpy as np
import eval_sibilant_directional as service
from mdd.sibilant_context_pairs import make_plan, context, SCHEMA


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for name, digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen pairing input changed: '+name)
    cache = Path(p['train_cache'])
    if json.loads((cache/'report.json').read_text())['partition'] != 'train':
        raise ValueError('only TRAIN labels may select comparison links')
    features = service.read_jsonl(p['features'])
    labels = service.read_jsonl(cache/'data.labels.jsonl')
    by = {z['feature_index']: z for z in labels if z['feature_index'] is not None}
    y = np.load(p['targets'])['y']
    original = np.load(p['batch_plan'])
    eligibility = service.read_jsonl(p['ctc_targets'])
    eligible = np.asarray([z['target'] is not None for z in eligibility])
    if len(features) != len(y) or len(eligibility) != len(y) or original.shape != (300, 16):
        raise ValueError('original TRAIN coverage/plan changed')
    for i, feature in enumerate(features):
        if feature['feature_index'] != i or any(feature[k] != by[i][k] for k in ('uid', 'q_runtime')):
            raise ValueError('comparison feature/label identity differs')
    plan, matched, shuffled, records = make_plan(original, y, eligible, features, by, p['seed'])
    audit, counts = [], collections.Counter()
    rare_before, rare_after = [], []
    for step, row in enumerate(plan):
        valid = row >= 0
        if not np.array_equal(valid, eligible[original[step]]):
            raise ValueError('skip pattern changed')
        if (y[row[valid]] < 0).any():
            raise ValueError('unknown actual label sampled')
        for previous, current in zip(original[step], row):
            if eligible[previous] and y[previous] >= 2:
                rare_before.append(int(previous))
                rare_after.append(int(current))
        for a, b in zip(matched[step], shuffled[step]):
            if a[0] < 0:
                if not (b == -1).all():
                    raise ValueError('pair padding differs')
                continue
            for link, aligned in ((a, True), (b, False)):
                ss, s = [int(row[int(k)]) for k in link]
                if y[ss] != 1 or y[s] != 0 or by[ss]['speaker'] == by[s]['speaker']:
                    raise ValueError('invalid actual class/speaker comparison')
                same = context(features[ss], by[ss]) == context(features[s], by[s])
                if same != aligned:
                    raise ValueError('matched/shuffled cell assignment differs')
            counts['pairs'] += 1
        n = int((matched[step, :, 0] >= 0).sum())
        for column in (0, 1):
            if sorted(matched[step, :n, column]) != sorted(shuffled[step, :n, column]):
                raise ValueError('comparison endpoint exposure differs')
        audit.append({'step': step, 'actual_counts': dict(collections.Counter(int(z) for z in y[row[valid]])),
                      'pairs': n})
    if rare_before != rare_after:
        raise ValueError('third/deletion site exposure changed')
    counts.update({'all_retained_GT': len(labels), 'inputs': len(features),
        'original_unknown': int((y < 0).sum()), 'no_input': sum(z['feature_index'] is None for z in labels),
        'draws': int((plan >= 0).sum()), 'skipped_draws': int((plan < 0).sum()),
        'unique_sampled': len(set(plan[plan >= 0].tolist())), 'rare_actual_draws_unchanged': len(rare_before)})
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    np.savez(out/'plan.npz', batch_plan=plan, matched=matched, shuffled=shuffled)
    (out/'pairs.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    (out/'batch_audit.json').write_text(json.dumps(audit, indent=2))
    exposure = []
    for name, ix in [('all', plan[plan >= 0]), ('error_endpoints', [z['error'] for r in records for z in r['pairs']]),
                     ('normal_endpoints', [z['normal'] for r in records for z in r['pairs']])]:
        counter = collections.Counter(int(i) for i in ix)
        exposure.append({'group': name, 'draws': sum(counter.values()), 'unique': len(counter),
            'max_repeat': max(counter.values()), 'actual_classes': dict(collections.Counter(int(y[i]) for i in ix)),
            'label_kinds': dict(collections.Counter(by[int(i)]['label_kind'] for i in ix)),
            'runtime_vowels': dict(collections.Counter(features[int(i)].get('observed_V') for i in ix))})
    report = {'scope': p['scope'], 'partition': 'train', 'schema': SCHEMA, 'counts': dict(counts),
        'exposure': exposure, 'identical_exposure_all_arms': True,
        'original_per_batch_actual_marginals_preserved': True, 'rare_draws_exactly_preserved': True,
        'shuffled_fixed_context_pairs': 0, 'TAG_or_pair_input_at_runtime': False,
        'plan_sha256': service.sha(out/'plan.npz'), 'deployment': False}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out/'finished.json').write_text(json.dumps({'report_sha256': service.sha(out/'report.json')}))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
