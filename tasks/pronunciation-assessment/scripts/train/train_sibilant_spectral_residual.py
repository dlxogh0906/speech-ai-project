"""Matched fine low-band spectra versus within-band averages on a fixed native head."""
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
import hashlib
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from train_sibilant_frozen_probe import write, scores as base_scores
from train_sibilant_hubert_adaptation import novel_proxy, retained_subgroups
from mdd.sibilant_pitchfree_spectrum import SCHEMA, normalized_input, normalization_digest
from mdd.sibilant_spectral_residual import SpectralResidual, spectral_artifact, load_spectral
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_vowel_presence import observed_vowel_support


def raw_arrays(cache, normalization, mode):
    return {'x': np.load(cache/'spectrum.npy', mmap_mode='r'),
            'energy': np.load(cache/'energy.npy', mmap_mode='r'),
            'relative_vowel': np.load(cache/'relative.npy', mmap_mode='r'),
            'lengths': np.load(cache/'lengths.npy'), 'normalization': normalization, 'mode': mode}


def batch_inputs(base, raw, ix, device):
    x = torch.as_tensor(base['x'][ix], device=device)
    n = torch.as_tensor(base['lengths'][ix], device=device)
    mask = torch.arange(x.shape[1], device=device)[None] < n[:, None]
    relative = torch.as_tensor(base['relative_vowel'][ix], device=device)
    z, valid = normalized_input(raw['x'][ix], raw['energy'][ix], raw['relative_vowel'][ix],
                                raw['normalization'], raw['mode'])
    physical = np.arange(z.shape[1])[None] < raw['lengths'][ix, None]
    return x, mask, relative, torch.as_tensor(z, device=device), torch.as_tensor(valid & physical, device=device)


def scores(model, base, raw, device, batch, maximum=None):
    size = len(base['x']) if maximum is None else maximum
    ps, av = [], []
    with torch.no_grad():
        for start in range(0, size, batch):
            args = batch_inputs(base, raw, list(range(start, min(size, start+batch))), device)
            logits, phase = model(*args)
            ps.extend(logits.softmax(-1).cpu().numpy())
            av.extend(observed_vowel_support(phase, args[1]).cpu().numpy())
    return np.asarray(ps), np.asarray(av)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen acoustic trial source changed: '+path)
    cache, acoustic = Path(p['train_cache']), Path(p['raw_cache'])
    if (json.loads((cache/'report.json').read_text())['partition'] != 'train'
            or json.loads((acoustic/'report.json').read_text())['partition'] != 'train'):
        raise ValueError('only TRAIN may fit residual')
    features = service.read_jsonl(p['features'])
    raw_features = service.read_jsonl(acoustic/'features.jsonl')
    if len(features) != len(raw_features):
        raise ValueError('raw/native site coverage differs')
    for a, b in zip(features, raw_features):
        if any(a[k] != b[k] for k in ('feature_index', 'uid', 'q_runtime', 'left', 'right', 'clip_sha256', 'observed_reference_seconds')):
            raise ValueError('raw/native source identity differs')
    base = dict(np.load(p['data']))
    normalization = json.loads((acoustic/'normalization.json').read_text())
    for name, digest in json.loads((acoustic/'report.json').read_text())['data_sha256'].items():
        if service.sha(acoustic/name) != digest:
            raise ValueError('spectral cache file changed: '+name)
    if normalization['schema'] != SCHEMA or normalization['partition'] != 'train':
        raise ValueError('TRAIN native acoustic normalization required')
    labels = service.read_jsonl(cache/'data.labels.jsonl')
    indexed = {z['feature_index']: z for z in labels if z['feature_index'] is not None}
    y = np.load(p['targets'])['y']
    eligibility = service.read_jsonl(p['ctc_targets'])
    plan = np.load(p['batch_plan'])
    if plan.shape != (300, 16):
        raise ValueError('fixed replay plan differs')
    batches = [[int(i) for i in draw if eligibility[int(i)]['target'] is not None] for draw in plan]
    used = [i for draw in batches for i in draw]
    if (y[used] < 0).any():
        raise ValueError('unknown target sampled')
    artifact = torch.load(p['control_artifact'], map_location='cpu')
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    report = {'scope': p['scope'], 'arms': {}, 'deployment': False}
    initial = None
    for mode in ('fine', 'coarse_bands'):
        data = raw_arrays(acoustic, normalization, mode)
        native = load_probe(artifact, p['binding'], device)
        before = tensor_digest(native.state_dict())
        torch.manual_seed(p['seed'])
        model = SpectralResidual(native, mode).to(device).eval()
        current = tensor_digest({k: v for k, v in model.state_dict().items() if not k.startswith('base.')})
        if initial is not None and initial != current:
            raise ValueError('matched residual initialization differs')
        initial = current
        ps0, av0 = scores(model, base, data, device, 64, maximum=64)
        expected, expected_av = base_scores(native, {k: v[:64] for k, v in base.items()}, device, 64)
        np.testing.assert_array_equal(ps0, expected)
        np.testing.assert_array_equal(av0, expected_av)
        trainable = [v for v in model.parameters() if v.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=.001, weight_decay=.01)
        losses, first_grad = [], None
        tick = time.monotonic()
        for step, ix in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(*batch_inputs(base, data, ix, device))
            loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(y[ix], device=device))
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite raw residual loss')
            loss.backward()
            gradient = model.layers[0].weight.grad
            if first_grad is None and gradient is not None and bool(gradient.abs().sum() > 0):
                first_grad = step
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        if first_grad is None or tensor_digest(native.state_dict()) != before:
            raise ValueError('branch did not learn or immutable native probe changed')
        ps, av = scores(model, base, data, device, 64)
        saved = spectral_artifact(model, artifact, normalization)
        torch.save(saved, out/(mode+'.pt'))
        np.savez(out/(mode+'.train_scores.npz'), probabilities=ps, available=av)
        restored = load_spectral(torch.load(out/(mode+'.pt'), map_location='cpu'), p['binding'], mode, device)
        check, valid = scores(restored, base, data, device, 64, maximum=64)
        np.testing.assert_array_equal(check, ps[:64])
        np.testing.assert_array_equal(valid, av[:64])
        proxy = novel_proxy(ps, av, y, features, indexed)
        subgroups = retained_subgroups(ps, av, y, features, labels, proxy)
        write(out/(mode+'.subgroups.json'), subgroups)
        report['arms'][mode] = {'TRAIN_novel_proxy': proxy, 'all_retained_counts': subgroups['counts'],
            'known': int((y >= 0).sum()), 'unknown': int((y < 0).sum()), 'available': int(av.sum()),
            'skipped_draws': int(plan.size-len(used)), 'unique_sampled': len(set(used)),
            'initial_branch_sha256': initial, 'first_feature_gradient_step': first_grad,
            'base_unchanged': True, 'initial_first64_base_exact': True, 'reload_first64_exact': True,
            'trainable_parameters': sum(v.numel() for v in trainable), 'fit_and_scoring_seconds': time.monotonic()-tick,
            'loss_first_last50': [float(np.mean(losses[:50])), float(np.mean(losses[-50:]))],
            'normalization_sha256': normalization_digest(normalization),
            'measurement_failure_sites': int((~np.isfinite(data['energy']).any(1)).sum()),
            'artifact_sha256': service.sha(out/(mode+'.pt'))}
        write(out/'report.json', report)
        print(mode, json.dumps(report['arms'][mode], ensure_ascii=False), flush=True)
        del data, model, restored, optimizer
    fine, coarse = [report['arms'][m]['TRAIN_novel_proxy'] for m in ('fine', 'coarse_bands')]
    report['proceed_fine_CAL'] = (fine['novel_TP'] > max(p['base_novel_TP'], coarse['novel_TP'])
        and fine['S_to_SS']['novel_TP'] > 0 and fine['SS_to_S']['novel_TP'] > 0)
    write(out/'report.json', report)
    write(out/'finished.json', {'report_sha256': service.sha(out/'report.json')})


if __name__ == '__main__':
    main()
