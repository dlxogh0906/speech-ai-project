"""Matched undivided raw C/V pooling, actual CE versus coarse-role auxiliary loss."""
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
from train_sibilant_frozen_probe import write, scores as base_scores
from train_sibilant_hubert_adaptation import novel_proxy, retained_subgroups
from mdd.sibilant_raw_cv import (RawCVResidual, coarse_role_loss, cv_artifact, load_cv, ROLE_SCHEMA)
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_vowel_presence import observed_vowel_support


def scores(model, base, raw, device, batch, phase_targets=None, maximum=None):
    size = len(base['x']) if maximum is None else maximum
    ps, av, locality = [], [], []
    confusion = np.zeros((3, 3), np.int64)
    with torch.no_grad():
        for start in range(0, size, batch):
            ix = list(range(start, min(size, start+batch)))
            args = batch_inputs(base, raw, ix, device)
            logits, base_phase, role = model(*args)
            ps.extend(logits.softmax(-1).cpu().numpy())
            av.extend(observed_vowel_support(base_phase, args[1]).cpu().numpy())
            if phase_targets is not None:
                target = torch.as_tensor(phase_targets[ix], device=device)
                weights = role.softmax(-1)[..., 1:].transpose(1, 2)*args[4][:, None]
                weights = weights/weights.sum(-1, keepdim=True).clamp(min=1e-8)
                masses = torch.stack([(weights[:, 0]*(target == 1)).sum(-1),
                    (weights[:, 1]*(target == 2)).sum(-1),
                    (weights[:, 0]*(target < 0)).sum(-1),
                    (weights[:, 1]*(target < 0)).sum(-1)], -1)
                locality.extend(masses.cpu().numpy())
                known = target >= 0
                truth, pred = target[known].cpu().numpy(), role.argmax(-1)[known].cpu().numpy()
                np.add.at(confusion, (truth, pred), 1)
    return np.asarray(ps), np.asarray(av), np.asarray(locality), confusion


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen raw CV source changed: '+path)
    cache, acoustic = Path(p['train_cache']), Path(p['raw_cache'])
    if any(json.loads((f/'report.json').read_text())['partition'] != 'train' for f in (cache, acoustic)):
        raise ValueError('only TRAIN may fit role-guided residual')
    base, raw = dict(np.load(p['data'])), dict(np.load(acoustic/'data.npz'))
    normalization = json.loads((acoustic/'normalization.json').read_text())
    data = input_arrays(raw, normalization, 'fine')
    features = service.read_jsonl(p['features'])
    raw_features = service.read_jsonl(acoustic/'features.jsonl')
    if len(features) != len(raw_features):
        raise ValueError('raw/base coverage differs')
    for a, b in zip(features, raw_features):
        if any(a[k] != b[k] for k in ('feature_index', 'uid', 'q_runtime', 'left', 'right', 'clip_sha256')):
            raise ValueError('raw/base site identity differs')
    targets = np.load(p['role_targets'])
    y, phase = targets['y'], targets['phase']
    np.testing.assert_array_equal(y, np.load(p['targets'])['y'])
    if phase.shape != raw['x'].shape[:2]:
        raise ValueError('raw role/input frame grid differs')
    role_report = json.loads(Path(p['role_report']).read_text())
    if role_report['role_schema'] != ROLE_SCHEMA or role_report['partition'] != 'train':
        raise ValueError('TRAIN coarse role schema differs')
    labels = service.read_jsonl(cache/'data.labels.jsonl')
    indexed = {z['feature_index']: z for z in labels if z['feature_index'] is not None}
    eligibility = service.read_jsonl(p['ctc_targets'])
    plan = np.load(p['batch_plan'])
    if plan.shape != (300, 16):
        raise ValueError('fixed replay plan differs')
    batches = [[int(i) for i in draw if eligibility[int(i)]['target'] is not None] for draw in plan]
    used = [i for draw in batches for i in draw]
    if (y[used] < 0).any():
        raise ValueError('unknown actual class sampled')
    artifact = torch.load(p['control_artifact'], map_location='cpu')
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    paired = (y >= 0) & (phase == 1).any(-1) & (phase == 2).any(-1)
    report = {'scope': p['scope'], 'paired_locality_sites': int(paired.sum()), 'arms': {}, 'deployment': False}
    initial = None
    for mode, weight in [('unguided', 0.), ('guided', .5)]:
        native = load_probe(artifact, p['binding'], device)
        before = tensor_digest(native.state_dict())
        torch.manual_seed(p['seed'])
        model = RawCVResidual(native, mode).to(device).eval()
        current = tensor_digest({k: v for k, v in model.state_dict().items() if not k.startswith('base.')})
        if initial is not None and initial != current:
            raise ValueError('matched raw-CV branch initialization differs')
        initial = current
        ps0, av0, _, _ = scores(model, base, data, device, 64, maximum=64)
        expected, ea = base_scores(native, {k: v[:64] for k, v in base.items()}, device, 64)
        np.testing.assert_array_equal(ps0, expected)
        np.testing.assert_array_equal(av0, ea)
        trainable = [v for v in model.parameters() if v.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=.001, weight_decay=.01)
        losses, aux_losses, first_feature, first_role = [], [], None, None
        tick = time.monotonic()
        for step, ix in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            logits, _, role = model(*batch_inputs(base, data, ix, device))
            cls = torch.nn.functional.cross_entropy(logits, torch.as_tensor(y[ix], device=device))
            aux = coarse_role_loss(role, torch.as_tensor(phase[ix], device=device))
            loss = cls+weight*aux
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite raw CV loss')
            loss.backward()
            if first_feature is None and bool(model.layers[0].weight.grad.abs().sum() > 0):
                first_feature = step
            if first_role is None and bool(model.role.weight.grad.abs().sum() > 0):
                first_role = step
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            losses.append(float(cls.detach()))
            aux_losses.append(float(aux.detach()))
        if first_feature is None or first_role is None or tensor_digest(native.state_dict()) != before:
            raise ValueError('raw role training missing or immutable base changed')
        ps, av, masses, confusion = scores(model, base, data, device, 64, phase_targets=phase)
        saved = cv_artifact(model, artifact, normalization)
        torch.save(saved, out/(mode+'.pt'))
        np.savez(out/(mode+'.train_scores.npz'), probabilities=ps, available=av)
        np.savez(out/(mode+'.locality.npz'), masses=masses, paired=paired)
        restored = load_cv(torch.load(out/(mode+'.pt'), map_location='cpu'), p['binding'], mode, device)
        check, valid, _, _ = scores(restored, base, data, device, 64, maximum=64)
        np.testing.assert_array_equal(check, ps[:64])
        np.testing.assert_array_equal(valid, av[:64])
        proxy = novel_proxy(ps, av, y, features, indexed)
        subgroups = retained_subgroups(ps, av, y, features, labels, proxy)
        write(out/(mode+'.subgroups.json'), subgroups)
        report['arms'][mode] = {'TRAIN_novel_proxy': proxy, 'all_retained_counts': subgroups['counts'],
            'known': int((y >= 0).sum()), 'unknown': int((y < 0).sum()), 'available': int(av.sum()),
            'skipped_draws': int(plan.size-len(used)), 'unique_sampled': len(set(used)),
            'initial_branch_sha256': current, 'first_feature_gradient_step': first_feature,
            'first_role_gradient_step': first_role, 'base_unchanged': True,
            'initial_first64_base_exact': True, 'reload_first64_exact': True,
            'trainable_parameters': sum(v.numel() for v in trainable),
            'fit_and_scoring_seconds': time.monotonic()-tick,
            'actual_loss_first_last50': [float(np.mean(losses[:50])), float(np.mean(losses[-50:]))],
            'role_loss_first_last50': [float(np.mean(aux_losses[:50])), float(np.mean(aux_losses[-50:]))],
            'role_confusion_actual_rows_predicted_cols': confusion.tolist(),
            'median_C_V_own_mass': np.median(masses[paired, :2], axis=0).tolist(),
            'median_C_V_unknown_edge_mass': np.median(masses[paired, 2:], axis=0).tolist(),
            'artifact_sha256': service.sha(out/(mode+'.pt'))}
        write(out/'report.json', report)
        print(mode, json.dumps(report['arms'][mode], ensure_ascii=False), flush=True)
        del model, restored, optimizer
    a, b = [report['arms'][m]['TRAIN_novel_proxy'] for m in ('unguided', 'guided')]
    gain = np.asarray(report['arms']['guided']['median_C_V_own_mass'])-report['arms']['unguided']['median_C_V_own_mass']
    report['median_own_mass_gain_C_V'] = gain.tolist()
    report['proceed_guided_CAL'] = (b['novel_TP'] > max(566, a['novel_TP'])
        and b['S_to_SS']['novel_TP'] > 0 and b['SS_to_S']['novel_TP'] > 0 and bool((gain >= .05).all()))
    write(out/'report.json', report)
    write(out/'finished.json', {'report_sha256': service.sha(out/'report.json')})


if __name__ == '__main__':
    main()
