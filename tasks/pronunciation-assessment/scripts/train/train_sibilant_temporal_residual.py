"""Fixed TRAIN comparison of ordered and permuted temporal residual branches."""
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
from train_sibilant_frozen_probe import stream, write, pair_proxy
from mdd.sibilant_frozen_probe import FrozenProbe, validate_binding, projection_digest
from mdd.sibilant_temporal_residual import (TemporalResidualProbe, make_permutations,
    temporal_artifact, load_temporal_probe)
from mdd.sibilant_joint_vowel import joint_vowel_loss
from mdd.sibilant_vowel_presence import CLASSES, observed_vowel_support


def fingerprint(values):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in values)).hexdigest()


def scores(model, data, permutations, device, batch):
    probabilities, available = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data['x']), batch):
            x = torch.as_tensor(data['x'][start:start + batch], device=device)
            n = torch.as_tensor(data['lengths'][start:start + batch], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < n[:, None]
            relative = torch.as_tensor(data['relative_vowel'][start:start + batch], device=device)
            permutation = (torch.as_tensor(permutations[start:start + batch], device=device)
                           if permutations is not None else None)
            logits, role = model(x, mask, relative, permutation)
            probabilities.extend(logits.softmax(-1).cpu().numpy())
            available.extend(observed_vowel_support(role, mask).cpu().numpy())
    return np.asarray(probabilities), np.asarray(available)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    cache = Path(p['train_cache'])
    if json.loads((cache / 'report.json').read_text())['partition'] != 'train':
        raise ValueError('TRAIN-only residual fit required')
    torch.set_num_threads(2)
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    data = dict(np.load(p['data']))
    features = service.read_jsonl(p['features'])
    reference = service.read_jsonl(cache / 'data.features.jsonl')
    target = np.load(p['targets'])
    y, phase = target['y'], target['phase']
    np.testing.assert_array_equal(y, np.load(cache / 'targets.npz')['y'])
    if len(features) != len(reference) or len(data['x']) != len(y) or phase.shape != data['x'].shape[:2]:
        raise ValueError('source/target coverage differs')
    for i, (a, b) in enumerate(zip(features, reference)):
        if (any(a[k] != b[k] for k in ('uid', 'q_runtime', 'left', 'right', 'clip_sha256'))
                or a['feature_index'] != i or a['input_frames'] != data['lengths'][i]):
            raise ValueError('feature order or grid differs')
    labels = {z['feature_index']: z for z in stream(cache / 'data.labels.jsonl') if z['feature_index'] is not None}
    eligibility = service.read_jsonl(p['ctc_targets'])
    plan = np.load(p['batch_plan'])
    if plan.shape != (p['steps'], 16):
        raise ValueError('fixed sampling plan differs')
    control = torch.load(p['control_artifact'], map_location='cpu')
    validate_binding(control, p['binding'])
    if control['binding']['classes'] != list(CLASSES):
        raise ValueError('actual class order differs')
    projection = control['projection']
    permutations = make_permutations(features, data['lengths'], data['x'].shape[1], p['permutation_seed'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(args.protocol).read_text())
    np.save(out / 'permutations.npy', permutations)
    report = {'scope': p['scope'], 'partition': 'train', 'deployment': False, 'arms': {}}
    branch_initial = None
    for mode in ('ordered', 'shuffled'):
        torch.manual_seed(p['seed'])
        model = TemporalResidualProbe(projection, mode).to(device).eval()
        initial = fingerprint(model.head.parameters())
        if initial != p['initial_head_sha256']:
            raise ValueError('base head initialization differs')
        current = fingerprint(model.branch.parameters())
        if branch_initial is not None and current != branch_initial:
            raise ValueError('branch initialization differs between arms')
        branch_initial = current
        torch.manual_seed(p['seed'])
        base = FrozenProbe(projection).to(device).eval()
        with torch.no_grad():
            x = torch.as_tensor(data['x'][:64], device=device)
            n = torch.as_tensor(data['lengths'][:64], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < n[:, None]
            relative = torch.as_tensor(data['relative_vowel'][:64], device=device)
            perm = torch.as_tensor(permutations[:64], device=device) if mode == 'shuffled' else None
            a, ar = model(x, mask, relative, perm)
            b, br = base(x, mask, relative)
            if not torch.equal(a, b) or not torch.equal(ar, br):
                raise ValueError('zero residual changed initial base outputs')
        del base
        optimizer = torch.optim.AdamW(model.parameters(), lr=p['head_lr'], weight_decay=.01)
        losses, phases, used, skipped = [], [], set(), 0
        first_feature_gradient = None
        tick = time.monotonic()
        for step, requested in enumerate(plan, 1):
            ix = [int(i) for i in requested if eligibility[int(i)]['target'] is not None]
            skipped += len(requested) - len(ix)
            used.update(ix)
            if not ix or (y[ix] < 0).any():
                raise ValueError('empty/unknown class batch')
            x = torch.as_tensor(data['x'][ix], device=device)
            n = torch.as_tensor(data['lengths'][ix], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < n[:, None]
            relative = torch.as_tensor(data['relative_vowel'][ix], device=device)
            perm = torch.as_tensor(permutations[ix], device=device) if mode == 'shuffled' else None
            optimizer.zero_grad(set_to_none=True)
            logits, role = model(x, mask, relative, perm)
            cls = torch.nn.functional.cross_entropy(logits, torch.as_tensor(y[ix], device=device))
            aux = joint_vowel_loss(role, torch.as_tensor(phase[ix], device=device))
            loss = cls + p['phase_weight'] * aux
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite temporal residual loss')
            loss.backward()
            grad = model.branch.layers[0].weight.grad
            if first_feature_gradient is None and grad is not None and bool(grad.abs().sum() > 0):
                first_feature_gradient = step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(cls.detach()))
            phases.append(float(aux.detach()))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.monotonic() - tick
        if first_feature_gradient is None:
            raise ValueError('temporal feature layers did not receive gradients')
        artifact = temporal_artifact(model, p['binding'])
        if projection_digest(artifact['projection']) != projection_digest(projection):
            raise ValueError('fixed projection changed')
        perms = permutations if mode == 'shuffled' else None
        ps, av = scores(model, data, perms, device, p['batch'])
        torch.save(artifact, out / (mode + '.pt'))
        np.savez(out / (mode + '.train_scores.npz'), probabilities=ps, available=av)
        restored = load_temporal_probe(torch.load(out / (mode + '.pt'), map_location='cpu'),
                                       dict(p['binding'], temporal_mode=mode), device)
        check, valid = scores(restored, {k: v[:64] for k, v in data.items()},
                              perms[:64] if perms is not None else None, device, p['batch'])
        np.testing.assert_array_equal(check, ps[:64])
        np.testing.assert_array_equal(valid, av[:64])
        report['arms'][mode] = {
            'known': int((y >= 0).sum()), 'unknown': int((y < 0).sum()),
            'correct_coarse_class': int((ps.argmax(-1)[y >= 0] == y[y >= 0]).sum()),
            'available': int(av.sum()), 'skipped_draws': skipped, 'unique_sampled': len(used),
            'initial_head_sha256': initial, 'initial_branch_sha256': current,
            'initial_base_outputs_first64_exact': True, 'reload_first64_exact': True,
            'base_head_parameters': sum(v.numel() for v in model.head.parameters()),
            'branch_parameters': sum(v.numel() for v in model.branch.parameters()),
            'first_nonzero_feature_gradient_step': first_feature_gradient,
            'training_seconds': seconds,
            'class_loss_first_last50': [float(np.mean(losses[:50])), float(np.mean(losses[-50:]))],
            'phase_loss_first_last50': [float(np.mean(phases[:50])), float(np.mean(phases[-50:]))],
            'TRAIN_proxy': pair_proxy(ps, av, y, features, labels),
            'artifact_sha256': service.sha(out / (mode + '.pt'))}
        write(out / 'report.json', report)
        print(mode, json.dumps(report['arms'][mode], ensure_ascii=False), flush=True)
        del model, restored, optimizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    ordered, shuffled = [report['arms'][mode]['TRAIN_proxy'] for mode in ('ordered', 'shuffled')]
    report['base_pair_control'] = p['base_pair_control']
    report['proceed_ordered_CAL'] = (ordered['pair_exact'] > p['base_pair_control']
        and ordered['S_to_SS']['pair_exact'] > 0 and ordered['SS_to_S']['pair_exact'] > 0)
    report['ordered_better_than_shuffled_TRAIN'] = ordered['pair_exact'] > shuffled['pair_exact']
    write(out / 'report.json', report)
    write(out / 'finished.json', {'sha256': service.sha(out / 'report.json')})


if __name__ == '__main__':
    main()
