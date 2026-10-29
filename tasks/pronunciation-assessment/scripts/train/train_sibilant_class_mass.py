"""Single-factor class-mass correction, fixed native features/head/300 replay."""
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
from train_sibilant_hubert_adaptation import novel_proxy, retained_subgroups
from train_sibilant_frozen_probe import write, scores
from mdd.sibilant_frozen_probe import (FrozenProbe, load_probe, validate_binding,
    projection_digest, READOUT)
from mdd.sibilant_prior_risk import class_mass_weights, actual_class_loss
from mdd.sibilant_joint_vowel import joint_vowel_loss
from mdd.sibilant_vowel_presence import CLASSES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen input changed: ' + path)
    cache = Path(p['train_cache'])
    if json.loads((cache / 'report.json').read_text())['partition'] != 'train':
        raise ValueError('only TRAIN may determine class mass or model')
    data = dict(np.load(p['data']))
    features = service.read_jsonl(p['features'])
    reference = service.read_jsonl(cache / 'data.features.jsonl')
    targets = np.load(p['targets'])
    y, phase = targets['y'], targets['phase']
    np.testing.assert_array_equal(y, np.load(cache / 'targets.npz')['y'])
    if len(features) != len(reference) or len(y) != len(data['x']) or phase.shape != data['x'].shape[:2]:
        raise ValueError('TRAIN coverage differs')
    for i, (a, b) in enumerate(zip(features, reference)):
        if a['feature_index'] != i or any(a[k] != b[k] for k in ('uid', 'q_runtime', 'left', 'right', 'clip_sha256')):
            raise ValueError('runtime input identity differs')
    labels = service.read_jsonl(cache / 'data.labels.jsonl')
    indexed = {z['feature_index']: z for z in labels if z['feature_index'] is not None}
    eligibility = service.read_jsonl(p['ctc_targets'])
    plan = np.load(p['batch_plan'])
    if plan.shape != (300, 16):
        raise ValueError('original 300x16 plan required')
    batches = [[int(i) for i in batch if eligibility[int(i)]['target'] is not None] for batch in plan]
    used = [i for batch in batches for i in batch]
    weights, population, draws = class_mass_weights(y, used)
    np.testing.assert_array_equal(weights, np.asarray(p['class_weights'], dtype=np.float32))
    control = torch.load(p['control_artifact'], map_location='cpu')
    validate_binding(control, p['binding'])
    if control['binding']['classes'] != list(CLASSES):
        raise ValueError('class order differs')
    projection = control['projection']
    torch.set_num_threads(2)
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(args.protocol).read_text())
    report = {'scope': p['scope'], 'class_weights': weights.tolist(), 'population': population.tolist(),
              'draws': draws.tolist(), 'arms': {}, 'deployment': False}
    for mode in ('original', 'class_mass'):
        torch.manual_seed(p['seed'])
        model = FrozenProbe(projection).to(device).eval()
        initial = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.head.parameters())).hexdigest()
        if initial != p['initial_head_sha256']:
            raise ValueError('head initialization differs')
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=p['head_lr'], weight_decay=.01)
        losses, phases = [], []
        tick = time.monotonic()
        for ix in batches:
            if not ix:
                raise ValueError('empty replay batch')
            x = torch.as_tensor(data['x'][ix], device=device)
            n = torch.as_tensor(data['lengths'][ix], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < n[:, None]
            relative = torch.as_tensor(data['relative_vowel'][ix], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, role = model(x, mask, relative)
            cls = actual_class_loss(logits, torch.as_tensor(y[ix], device=device),
                                    weights if mode == 'class_mass' else None)
            aux = joint_vowel_loss(role, torch.as_tensor(phase[ix], device=device))
            loss = cls + .5 * aux
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite class-mass loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.)
            optimizer.step()
            losses.append(float(cls.detach()))
            phases.append(float(aux.detach()))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.monotonic() - tick
        ps, av = scores(model, data, device, 64)
        if mode == 'original':
            old = np.load(p['reference_scores'])
            np.testing.assert_array_equal(ps, old['probabilities'])
            np.testing.assert_array_equal(av, old['available'])
            for key, value in control['head'].items():
                if not torch.equal(value, model.head.state_dict()[key].cpu()):
                    raise ValueError('original control head did not reproduce')
        bound = dict(p['binding'], classes=list(CLASSES), projection_sha256=projection_digest(projection),
                     training_objective='fixed_replay_' + mode + '_actual5_CE_jointV_half_v1')
        artifact = {'readout': READOUT, 'binding': bound, 'projection': projection,
                    'head': {k: v.detach().cpu() for k, v in model.head.state_dict().items()}}
        torch.save(artifact, out / (mode + '.pt'))
        np.savez(out / (mode + '.train_scores.npz'), probabilities=ps, available=av)
        restored = load_probe(torch.load(out / (mode + '.pt'), map_location='cpu'), bound, device)
        check, valid = scores(restored, {k: v[:64] for k, v in data.items()}, device, 64)
        np.testing.assert_array_equal(check, ps[:64])
        np.testing.assert_array_equal(valid, av[:64])
        proxy = novel_proxy(ps, av, y, features, indexed)
        subgroups = retained_subgroups(ps, av, y, features, labels, proxy)
        write(out / (mode + '.subgroups.json'), subgroups)
        report['arms'][mode] = {'TRAIN_novel_proxy': proxy, 'all_retained_counts': subgroups['counts'],
            'known': int((y >= 0).sum()), 'unknown': int((y < 0).sum()), 'available': int(av.sum()),
            'correct_coarse_class': int((ps.argmax(-1)[y >= 0] == y[y >= 0]).sum()),
            'skipped_draws': int(plan.size - len(used)), 'unique_sampled': len(set(used)),
            'initial_head_sha256': initial, 'fixed_projection_sha256': projection_digest(projection),
            'training_seconds': seconds, 'reload_first64_exact': True,
            'class_loss_first_last50': [float(np.mean(losses[:50])), float(np.mean(losses[-50:]))],
            'phase_loss_first_last50': [float(np.mean(phases[:50])), float(np.mean(phases[-50:]))],
            'artifact_sha256': service.sha(out / (mode + '.pt'))}
        write(out / 'report.json', report)
        print(mode, json.dumps(report['arms'][mode], ensure_ascii=False), flush=True)
        del model, restored, optimizer
    a, b = [report['arms'][mode]['TRAIN_novel_proxy'] for mode in ('original', 'class_mass')]
    report['proceed_class_mass_CAL'] = (b['novel_TP'] > a['novel_TP']
        and b['S_to_SS']['novel_TP'] > 0 and b['SS_to_S']['novel_TP'] > 0)
    write(out / 'report.json', report)
    write(out / 'finished.json', {'report_sha256': service.sha(out / 'report.json')})


if __name__ == '__main__':
    main()
