"""Freeze signal/permuted residual axes and original-PCM-native TRAIN features."""
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
import numpy as np
import torch
import eval_sibilant_directional as service
from mdd.sibilant_discarded_axes import fit_axes, axis_observations, vowel_groups, SCHEMA, MODES
from mdd.sibilant_projection_residual import signed_weights
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_vowel_presence import observed_vowel_support


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for name, digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen discarded-axis source changed: '+name)
    cache, audit = Path(p['train_cache']), Path(p['audit_result'])
    if (json.loads((cache/'report.json').read_text())['partition'] != 'train'
            or not json.loads((audit/'report.json').read_text())['result']['proceed_residual_candidate']):
        raise ValueError('positive predeclared TRAIN representation evidence required')
    cells = json.loads((audit/'cells.json').read_text())
    selected = dict(np.load(audit/'selected_observations.npz'))
    by_index = {int(i): j for j, i in enumerate(selected['indices'])}
    y = np.load(p['targets'])['y']
    rng = np.random.RandomState(p['permuted_axis_seed'])
    real, shuffled, changed = [], [], 0
    for cell in cells:
        a, b = [], []
        for ix in cell['halves']:
            observations = selected['full'][[by_index[i] for i in ix]].astype(np.float64)
            original = y[ix]
            permuted = rng.permutation(original)
            changed += int((original != permuted).sum())
            a.append(np.einsum('n,nrd->rd', signed_weights(original), observations))
            b.append(np.einsum('n,nrd->rd', signed_weights(permuted), observations))
        real.append(a)
        shuffled.append(b)
    real, shuffled = np.asarray(real), np.asarray(shuffled)
    np.testing.assert_allclose(real, np.load(audit/'cross_half_deltas.npy'), atol=1e-10, rtol=1e-10)
    if changed == 0:
        raise ValueError('conditional label permutation did not change any row')
    original = torch.load(p['control_artifact'], map_location='cpu')
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(2)
    axis_bank, axis_report = {}, {}
    for mode, deltas in [('signal', real), ('permuted', shuffled)]:
        axis_bank[mode], axis_report[mode] = fit_axes(deltas, original['projection']['components'], device)
    base = load_probe(original, p['binding'], device).requires_grad_(False).eval()
    before = tensor_digest(base.state_dict())
    features = service.read_jsonl(p['features'])
    data = dict(np.load(p['data']))
    values = {mode: [] for mode in MODES}
    logits_all, probabilities, available = [], [], []
    with torch.no_grad():
        for start in range(0, len(features), 64):
            x = torch.as_tensor(data['x'][start:start+64], device=device)
            n = torch.as_tensor(data['lengths'][start:start+64], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < n[:, None]
            relative = torch.as_tensor(data['relative_vowel'][start:start+64], device=device)
            logits, phase = base(x, mask, relative)
            for mode in MODES:
                raw = axis_observations(base, x, mask, phase, axis_bank[mode].to(device))
                values[mode].extend(raw.cpu().numpy())
            logits_all.extend(logits.cpu().numpy())
            probabilities.extend(logits.softmax(-1).cpu().numpy())
            available.extend(observed_vowel_support(phase, mask).cpu().numpy())
    old = np.load(p['base_scores'])
    np.testing.assert_array_equal(probabilities, old['probabilities'])
    np.testing.assert_array_equal(available, old['available'])
    if tensor_digest(base.state_dict()) != before:
        raise ValueError('original native parent changed')
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    normalization = {}
    for mode in MODES:
        x = np.asarray(values[mode], np.float32)
        mean = x.astype(np.float64).mean(0)
        scale = np.maximum(x.astype(np.float64).std(0), 1e-3)
        normalization[mode] = {'schema': SCHEMA, 'partition': 'train',
            'mean': mean.astype(np.float32).tolist(), 'scale': scale.astype(np.float32).tolist()}
        values[mode] = x
    np.savez(out/'observations.npz', **values, base_logits=np.asarray(logits_all, np.float32),
             available=np.asarray(available), vowel_group=vowel_groups(features))
    bank = {'axes': axis_bank, 'normalization': normalization, 'base_head_sha256': tensor_digest(original['head']),
            'schema': SCHEMA, 'axis_report': axis_report, 'permuted_rows_changed': changed}
    torch.save(bank, out/'bank.pt')
    report = {'scope': p['scope'], 'schema': SCHEMA, 'partition': 'train', 'inputs': len(features),
        'source_cells': len(cells), 'conditional_permutation_changed_rows': changed, 'axis_report': axis_report,
        'all13741_parent_probabilities_and_availability_bitwise_equal': True,
        'parent_unchanged': True, 'bank_sha256': service.sha(out/'bank.pt'),
        'observations_sha256': service.sha(out/'observations.npz'), 'deployment': False}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out/'finished.json').write_text(json.dumps({'report_sha256': service.sha(out/'report.json')}))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
