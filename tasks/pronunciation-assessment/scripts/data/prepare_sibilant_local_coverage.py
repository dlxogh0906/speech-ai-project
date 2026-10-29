"""Change local-CTC observation diversity while preserving each control draw type.

The merged TRAIN cache must retain the original cache as an exact prefix.
Each supported draw keeps its actual-phone class and normal/error label kind;
only the observed example changes. Unsupported control draws stay unsupported.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import json
import numpy as np
import eval_sibilant_directional as service


def unique_labels(rows):
    result = {}
    for row in rows:
        i = row['feature_index']
        if i is None:
            continue
        if i in result:
            raise ValueError('one GT site per supported training query required')
        result[i] = row
    return result


def matched_plan(original, old_y, old_targets, old_labels, y, targets, labels, seed):
    if original.ndim != 2 or np.any(original < 0) or np.any(original >= len(old_y)):
        raise ValueError('control plan index outside original TRAIN cache')
    if len(y) != len(targets) or len(old_y) != len(old_targets):
        raise ValueError('target count differs from input count')
    pools = collections.defaultdict(list)
    for i, target in enumerate(targets):
        if y[i] >= 0 and target['target'] is not None:
            if i not in labels or labels[i]['label_kind'] == 'ambiguous':
                raise ValueError('supported class has no unique nonambiguous GT site')
            pools[(int(y[i]), labels[i]['label_kind'])].append(i)
    rng = np.random.RandomState(seed)
    result = original.copy()
    for index in np.ndindex(original.shape):
        i = int(original[index])
        if old_y[i] < 0:
            raise ValueError('control sampled an unknown actual class')
        if old_targets[i]['target'] is None:
            continue
        key = (int(old_y[i]), old_labels[i]['label_kind'])
        if not pools[key]:
            raise ValueError('no expanded example of control draw type')
        result[index] = rng.choice(pools[key])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    old, cache = Path(p['control_cache']), Path(p['cache'])
    for folder in (old, cache):
        if json.loads((folder / 'report.json').read_text())['partition'] != 'train':
            raise ValueError('TRAIN-only observation expansion')
    old_y = np.load(old / 'targets.npz')['y']
    y = np.load(cache / 'targets.npz')['y']
    if not np.array_equal(old_y, y[:len(old_y)]):
        raise ValueError('original class targets changed')
    old_data, data = np.load(old / 'data.npz'), np.load(cache / 'data.npz')
    for key in old_data.files:
        if not np.array_equal(old_data[key], data[key][:len(old_y)]):
            raise ValueError('original acoustic inputs changed: ' + key)
    old_features = service.read_jsonl(old / 'data.features.jsonl')
    features = service.read_jsonl(cache / 'data.features.jsonl')
    if old_features != features[:len(old_features)]:
        raise ValueError('original feature identities changed')
    old_targets, targets = [service.read_jsonl(p[k]) for k in ('control_targets', 'targets')]
    if old_targets != targets[:len(old_targets)]:
        raise ValueError('original CTC supervision regions or actual sequences changed')
    old_labels = unique_labels(service.read_jsonl(old / 'data.labels.jsonl'))
    labels = unique_labels(service.read_jsonl(cache / 'data.labels.jsonl'))
    original = np.load(p['control_batch_plan'])
    plan = matched_plan(original, old_y, old_targets, old_labels, y, targets, labels, p['seed'])
    groups = collections.defaultdict(collections.Counter)
    for arm, draws, ts, labs, ys in [('control', original, old_targets, old_labels, old_y),
                                     ('expanded', plan, targets, labels, y)]:
        for raw in draws.ravel():
            i = int(raw)
            if ts[i]['target'] is None:
                groups[arm]['unavailable_draws'] += 1
                continue
            label = labs[i]
            groups[arm]['supported_draws'] += 1
            groups[arm]['actual_class_kind/' + str(int(ys[i])) + '/' + label['label_kind']] += 1
            groups[arm]['vowel/' + label['following']] += 1
            groups[arm]['position/' + label['position']] += 1
        groups[arm]['unique_supported_inputs'] = len({int(i) for i in draws.ravel() if ts[int(i)]['target'] is not None})
    for key, value in groups['control'].items():
        if key.startswith('actual_class_kind/') or key in ('supported_draws', 'unavailable_draws'):
            if groups['expanded'][key] != value:
                raise AssertionError('draw class/kind/support composition changed')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(a.protocol).read_text())
    np.save(out / 'batch_plan.npy', plan)
    report = {'scope': p['scope'], 'partition': 'train', 'deployment': False,
              'seed': p['seed'], 'all_inputs': len(y), 'control_inputs': len(old_y),
              'shape': list(plan.shape), 'draws_from_new_inputs': int((plan >= len(old_y)).sum()),
              'original_inputs_targets_exact_prefix': True,
              'groups': {k: dict(v) for k, v in groups.items()},
              'batch_plan_sha256': service.sha(out / 'batch_plan.npy')}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
