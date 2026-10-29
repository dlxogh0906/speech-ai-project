"""Compare original observed frame sequences without fitting a classifier."""
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
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from mdd.sibilant_frozen_probe import load_probe, project_observed
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_projection_residual import speaker_half
from mdd.sibilant_trajectory_abx import (METHODS, unit_rows, permutation,
    angular_matrix, dtw, class_balanced_abx, summarize)


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for name, digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen trajectory input changed: '+name)
    cache = Path(p['train_cache'])
    if json.loads((cache/'report.json').read_text())['partition'] != 'train' or p['bootstraps'] != 2000:
        raise ValueError('predeclared TRAIN-only observation required')
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_bytes(Path(args.protocol).read_bytes())
    features = service.read_jsonl(p['features'])
    labels = service.read_jsonl(cache/'data.labels.jsonl')
    by = {z['feature_index']: z for z in labels if z['feature_index'] is not None}
    y = np.load(p['targets'])['y']
    cells = json.loads(Path(p['cells']).read_text())
    original = torch.load(p['control_artifact'], map_location='cpu')
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(2)
    base = load_probe(original, p['binding'], device).requires_grad_(False).eval()
    before = tensor_digest(base.state_dict())
    selected = sorted({i for cell in cells for i in cell['indices']})
    data = dict(np.load(p['data']))
    sequences, shuffled, means, failures = {}, {}, {}, []
    tick = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(selected), 64):
            ix = selected[start:start+64]
            x = torch.as_tensor(data['x'][ix], device=device)
            lengths = data['lengths'][ix]
            mask = torch.arange(x.shape[1], device=device)[None] < torch.as_tensor(lengths, device=device)[:, None]
            projected = project_observed(x, mask, {k: getattr(base, k) for k in ('mean', 'scale', 'components')}).cpu().numpy()
            for j, i in enumerate(ix):
                identity = features[i]['uid']+'/'+str(features[i]['q_runtime'])
                try:
                    observed = projected[j, :int(lengths[j])]
                    sequence = unit_rows(observed)
                    mean = unit_rows(observed.astype(np.float64).mean(0, keepdims=True))[0]
                    order = permutation(identity, len(sequence), p['seed'])
                    sequences[i], shuffled[i], means[i] = sequence, sequence[order], mean
                except ValueError as exc:
                    failures.append({'feature_index': i, 'identity': identity, 'reason': str(exc)})
    if failures:
        write(out/'measurement_failures.json', failures)
        raise ValueError('observations failed; no failed rows silently removed')
    records, scores, all_distances, pairs = [], [], [], []
    for number, cell in enumerate(cells):
        left, right = cell['halves']
        assert set(left+right) == set(cell['indices'])
        assert len({by[i]['speaker'] for i in left+right}) == len(left+right)
        for half, indices in enumerate([left, right]):
            assert all(speaker_half(by[i]['speaker'], p['seed']) == half for i in indices)
        distances = np.empty((len(left), len(right), 3), np.float64)
        for a, i in enumerate(left):
            for b, j in enumerate(right):
                distances[a, b] = (dtw(angular_matrix(sequences[i], sequences[j])),
                    dtw(angular_matrix(shuffled[i], shuffled[j])),
                    float(angular_matrix(means[i][None], means[j][None])[0, 0]))
                pairs.append((number, i, j))
                all_distances.append(distances[a, b].copy())
        result, half_scores = [], {}
        for k, method in enumerate(METHODS):
            value, halves = class_balanced_abx(distances[:, :, k], y[left], y[right])
            result.append(value)
            half_scores[method] = halves
        records.append({'cell': cell['cell'], 'left': left, 'right': right,
                        'accuracy': dict(zip(METHODS, result)), 'half_class_accuracy': half_scores})
        scores.append(result)
        if number % 20 == 0:
            print(json.dumps({'cells_done': number+1, 'pairs_done': len(pairs),
                              'seconds': time.monotonic()-tick}), flush=True)
    result = summarize(scores, [cell['cell'][3] for cell in cells], p['seed'], p['bootstraps'])
    if tensor_digest(base.state_dict()) != before:
        raise ValueError('immutable original model changed')
    write(out/'cells.json', records)
    np.savez(out/'distances.npz', pairs=np.asarray(pairs, np.int64), distances=np.asarray(all_distances))
    report = {'scope': p['scope'], 'source_flow': json.loads(Path(p['prior_audit_report']).read_text())['flow'],
        'diagnostic_rows': len(selected), 'pair_distances': len(pairs), 'measurement_failures': 0,
        'result': result, 'no_model_or_threshold_fitting': True,
        'native_model_PCA_unchanged': True, 'seconds': time.monotonic()-tick,
        'deployment': False, 'goal_complete': False}
    write(out/'report.json', report)
    write(out/'finished.json', {'report_sha256': service.sha(out/'report.json')})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
