"""Frozen automatic C/V teacher: aligned versus identity-shuffled role assignment."""
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
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from train_sibilant_hires_residual import input_arrays, batch_inputs
from train_sibilant_hubert_adaptation import novel_proxy, retained_subgroups
from train_sibilant_frozen_probe import write, scores as base_scores
from mdd.sibilant_frozen_raw_roles import FrozenRawRoles, role_artifact, load_roles
from mdd.sibilant_raw_cv import load_cv
from mdd.sibilant_temporal_residual import make_permutations
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_vowel_presence import observed_vowel_support


def scores(model, base, raw, permutations, device, batch, phase_targets=None, maximum=None):
    size = len(base['x']) if maximum is None else maximum
    ps, av, locality = [], [], []
    with torch.no_grad():
        for start in range(0, size, batch):
            ix = list(range(start, min(size, start+batch)))
            args = batch_inputs(base, raw, ix, device)
            perm = torch.as_tensor(permutations[ix], device=device) if permutations is not None else None
            logits, base_phase, role = model(*args, permutation=perm)
            ps.extend(logits.softmax(-1).cpu().numpy())
            av.extend(observed_vowel_support(base_phase, args[1]).cpu().numpy())
            if phase_targets is not None:
                target = torch.as_tensor(phase_targets[ix], device=device)
                w = role.softmax(-1)[..., 1:].transpose(1, 2)*args[4][:, None]
                w = w/w.sum(-1, keepdim=True).clamp(min=1e-8)
                locality.extend(torch.stack([(w[:, 0]*(target == 1)).sum(-1),
                    (w[:, 1]*(target == 2)).sum(-1),
                    (w[:, 0]*(target < 0)).sum(-1), (w[:, 1]*(target < 0)).sum(-1)], -1).cpu().numpy())
    return np.asarray(ps), np.asarray(av), np.asarray(locality)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen role experiment source changed: '+path)
    cache = Path(p['train_cache'])
    if json.loads((cache/'report.json').read_text())['partition'] != 'train':
        raise ValueError('frozen-role student fit requires TRAIN')
    base, raw = dict(np.load(p['data'])), dict(np.load(Path(p['raw_cache'])/'data.npz'))
    teacher_artifact = torch.load(p['teacher_artifact'], map_location='cpu')
    normalization = teacher_artifact['normalization']
    data = input_arrays(raw, normalization, 'fine')
    features = service.read_jsonl(p['features'])
    source_raw = service.read_jsonl(Path(p['raw_cache'])/'features.jsonl')
    if len(features) != len(source_raw):
        raise ValueError('raw/native source coverage differs')
    for a, b in zip(features, source_raw):
        if any(a[k] != b[k] for k in ('feature_index', 'uid', 'q_runtime', 'left', 'right', 'clip_sha256')):
            raise ValueError('raw/native site identity differs')
    target = np.load(p['role_targets'])
    y, phase = target['y'], target['phase']
    np.testing.assert_array_equal(y, np.load(p['targets'])['y'])
    if phase.shape != raw['x'].shape[:2]:
        raise ValueError('raw role diagnostic grid differs')
    paired = (y >= 0) & (phase == 1).any(-1) & (phase == 2).any(-1)
    labels = service.read_jsonl(cache/'data.labels.jsonl')
    indexed = {z['feature_index']: z for z in labels if z['feature_index'] is not None}
    eligibility = service.read_jsonl(p['ctc_targets'])
    plan = np.load(p['batch_plan'])
    if plan.shape != (300, 16):
        raise ValueError('original300x16 replay plan required')
    batches = [[int(i) for i in draw if eligibility[int(i)]['target'] is not None] for draw in plan]
    used = [i for draw in batches for i in draw]
    if (y[used] < 0).any():
        raise ValueError('unknown sampled actual class')
    permutations = make_permutations(features, raw['lengths'], raw['x'].shape[1], p['seed'])
    artifact = torch.load(p['control_artifact'], map_location='cpu')
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    np.save(out/'permutations.npy', permutations)
    report = {'scope': p['scope'], 'paired_locality_sites': int(paired.sum()), 'arms': {}, 'deployment': False}
    initial = None
    for mode in ('aligned', 'permuted'):
        teacher = load_cv(teacher_artifact, p['binding'], 'guided', device)
        native = load_probe(artifact, p['binding'], device)
        before, teacher_before = tensor_digest(native.state_dict()), tensor_digest(teacher.state_dict())
        torch.manual_seed(p['seed'])
        model = FrozenRawRoles(native, teacher, mode).to(device).eval()
        current = tensor_digest({k: v for k, v in model.state_dict().items()
                                 if not k.startswith(('base.', 'teacher.'))})
        if initial is not None and initial != current:
            raise ValueError('matched student initialization differs')
        initial = current
        perm = permutations if mode == 'permuted' else None
        ps0, av0, _ = scores(model, base, data, perm, device, 64, maximum=64)
        expected, ea = base_scores(native, {k: v[:64] for k, v in base.items()}, device, 64)
        np.testing.assert_array_equal(ps0, expected)
        np.testing.assert_array_equal(av0, ea)
        parameters = [v for v in model.parameters() if v.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.01)
        losses, first_grad = [], None
        tick = time.monotonic()
        for step, ix in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            per = torch.as_tensor(perm[ix], device=device) if perm is not None else None
            logits, _, _ = model(*batch_inputs(base, data, ix, device), permutation=per)
            loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(y[ix], device=device))
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite student actual loss')
            loss.backward()
            if any(v.grad is not None for v in model.teacher.parameters()):
                raise ValueError('teacher received gradients')
            if first_grad is None and bool(model.layers[0].weight.grad.abs().sum() > 0):
                first_grad = step
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        if (first_grad is None or tensor_digest(native.state_dict()) != before
                or tensor_digest(teacher.state_dict()) != teacher_before):
            raise ValueError('student failed to learn or frozen state changed')
        ps, av, masses = scores(model, base, data, perm, device, 64, phase_targets=phase)
        saved = role_artifact(model, artifact, teacher_artifact, normalization)
        torch.save(saved, out/(mode+'.pt'))
        np.savez(out/(mode+'.train_scores.npz'), probabilities=ps, available=av)
        np.savez(out/(mode+'.locality.npz'), masses=masses, paired=paired)
        restored = load_roles(torch.load(out/(mode+'.pt'), map_location='cpu'), p['binding'], mode, device)
        check, valid, _ = scores(restored, base, data, perm, device, 64, maximum=64)
        np.testing.assert_array_equal(check, ps[:64])
        np.testing.assert_array_equal(valid, av[:64])
        proxy = novel_proxy(ps, av, y, features, indexed)
        subgroups = retained_subgroups(ps, av, y, features, labels, proxy)
        write(out/(mode+'.subgroups.json'), subgroups)
        report['arms'][mode] = {'TRAIN_novel_proxy': proxy, 'all_retained_counts': subgroups['counts'],
            'known': int((y >= 0).sum()), 'unknown': int((y < 0).sum()), 'available': int(av.sum()),
            'skipped_draws': int(plan.size-len(used)), 'unique_sampled': len(set(used)),
            'initial_branch_sha256': current, 'first_feature_gradient_step': first_grad,
            'base_unchanged': True, 'teacher_unchanged': True, 'teacher_gradients': False,
            'initial_first64_base_exact': True, 'reload_first64_exact': True,
            'trainable_parameters': sum(v.numel() for v in parameters),
            'fit_and_scoring_seconds': time.monotonic()-tick,
            'actual_loss_first_last50': [float(np.mean(losses[:50])), float(np.mean(losses[-50:]))],
            'median_C_V_own_mass': np.median(masses[paired, :2], axis=0).tolist(),
            'median_C_V_unknown_edge_mass': np.median(masses[paired, 2:], axis=0).tolist(),
            'artifact_sha256': service.sha(out/(mode+'.pt'))}
        write(out/'report.json', report)
        print(mode, json.dumps(report['arms'][mode], ensure_ascii=False), flush=True)
        del model, restored, optimizer, teacher
    a, b = [report['arms'][m]['TRAIN_novel_proxy'] for m in ('aligned', 'permuted')]
    gain = np.asarray(report['arms']['aligned']['median_C_V_own_mass'])-report['arms']['permuted']['median_C_V_own_mass']
    report['median_own_mass_gain_C_V'] = gain.tolist()
    report['proceed_aligned_CAL'] = (a['novel_TP'] > max(573, b['novel_TP'])
        and a['S_to_SS']['novel_TP'] > 0 and a['SS_to_S']['novel_TP'] > 0 and bool((gain >= .05).all()))
    write(out/'report.json', report)
    write(out/'finished.json', {'report_sha256': service.sha(out/'report.json')})


if __name__ == '__main__':
    main()
