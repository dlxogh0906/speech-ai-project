"""Inspect discarded native representation using TRAIN split-speaker contrasts."""
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
import shutil
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from mdd.sibilant_projection_residual import SCHEMA, ROLES, select_cells, full_observations, replication
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
            raise ValueError('frozen residual audit source changed: '+name)
    cache = Path(p['train_cache'])
    if json.loads((cache/'report.json').read_text())['partition'] != 'train' or p['permutations'] != 1000:
        raise ValueError('predeclared TRAIN-only diagnostic required')
    features = service.read_jsonl(p['features'])
    labels = service.read_jsonl(cache/'data.labels.jsonl')
    by = {z['feature_index']: z for z in labels if z['feature_index'] is not None}
    y = np.load(p['targets'])['y']
    eligibility = np.asarray([z['target'] is not None for z in service.read_jsonl(p['ctc_targets'])])
    if len(features) != len(y) or len(y) != len(eligibility):
        raise ValueError('original source coverage differs')
    for i, f in enumerate(features):
        if f['feature_index'] != i or any(f[k] != by[i][k] for k in ('uid', 'q_runtime')):
            raise ValueError('source label/feature identity differs')
    cells, ledger, flow = select_cells(features, by, y, eligibility, p['seed'])
    flow.update(all_retained_GT=len(labels), no_input=sum(z['feature_index'] is None for z in labels))
    data = dict(np.load(p['data']))
    original = dict(np.load(p['original_summary']))
    artifact = torch.load(p['control_artifact'], map_location='cpu')
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(2)
    base = load_probe(artifact, p['binding'], device).requires_grad_(False).eval()
    before = tensor_digest(base.state_dict())
    width = len(base.mean)
    full = np.empty((len(y), 3, width), np.float32)
    maximum = 0.
    tick = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(y), 64):
            x = torch.as_tensor(data['x'][start:start+64], device=device)
            n = torch.as_tensor(data['lengths'][start:start+64], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < n[:, None]
            relative = torch.as_tensor(data['relative_vowel'][start:start+64], device=device)
            observations, phase = full_observations(base, x, mask, relative)
            _, expected_phase = base(x, mask, relative)
            if not torch.equal(phase, expected_phase):
                raise ValueError('automatic role logits changed')
            av = observed_vowel_support(phase, mask).cpu().numpy()
            np.testing.assert_array_equal(av, original['available'][start:start+64])
            kept = (observations@base.components).reshape(len(x), -1).cpu().numpy()
            expected = original['summary'][start:start+64]
            np.testing.assert_allclose(kept, expected, atol=p['projection_atol'], rtol=p['projection_rtol'])
            maximum = max(maximum, float(abs(kept-expected).max()))
            full[start:start+len(x)] = observations.cpu().numpy()
    if tensor_digest(base.state_dict()) != before:
        raise ValueError('immutable native model changed')
    selected = np.asarray(sorted({i for cell in cells for i in cell['indices']}), np.int64)
    if shutil.disk_usage(out.parent).free < full[selected].nbytes+100000000:
        raise ValueError('insufficient disk without deleting prior artifacts')
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    (out/'cells.json').write_text(json.dumps(cells, ensure_ascii=False, indent=2))
    (out/'all_cells_ledger.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2))
    np.savez(out/'selected_observations.npz', indices=selected, full=full[selected])
    print(json.dumps({'stage': 'observations_ready', 'flow': flow,
        'original_projected_summary_max_difference': maximum, 'seconds': time.monotonic()-tick}), flush=True)
    result, null, deltas = replication(cells, full, base.components.detach().cpu().numpy(), y,
                                      p['permutations'], p['seed'])
    np.savez(out/'conditional_null.npz', **null)
    np.save(out/'cross_half_deltas.npy', deltas)
    report = {'scope': p['scope'], 'schema': SCHEMA, 'partition': 'train', 'flow': flow,
        'roles': list(ROLES), 'original_projected_summary_max_difference': maximum,
        'all13741_automatic_phase_logits_and_availability_unchanged': True,
        'native_head_PCA_scaler_unchanged': True, 'TAG_boundary_or_actual_at_runtime': False,
        'original_model_and_PCA_TRAIN_exposure_not_undone_by_speaker_split': True,
        'result': result, 'permutations': p['permutations'], 'seed': p['seed'],
        'seconds': time.monotonic()-tick, 'deployment': False, 'goal_complete': False}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out/'finished.json').write_text(json.dumps({'report_sha256': service.sha(out/'report.json')}))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
