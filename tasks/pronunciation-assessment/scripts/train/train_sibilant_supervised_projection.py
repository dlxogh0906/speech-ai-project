"""One-factor TRAIN comparison: update the existing 256-dimensional matrix."""
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
from train_sibilant_frozen_probe import stream, write, scores, pair_proxy
from mdd.sibilant_frozen_probe import load_probe, validate_binding, projection_digest
from mdd.sibilant_supervised_projection import SupervisedProjectionProbe, supervised_artifact
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
            raise ValueError('frozen source changed: ' + path)
    cache = Path(p['train_cache'])
    if json.loads((cache / 'report.json').read_text())['partition'] != 'train':
        raise ValueError('TRAIN-only supervised projection required')
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
        keys = ('uid', 'q_runtime', 'left', 'right', 'clip_sha256')
        if any(a[key] != b[key] for key in keys) or a['feature_index'] != i or a['input_frames'] != data['lengths'][i]:
            raise ValueError('source/window order or frame grid differs')
    labels = {z['feature_index']: z for z in stream(cache / 'data.labels.jsonl') if z['feature_index'] is not None}
    eligibility = service.read_jsonl(p['ctc_targets'])
    plan = np.load(p['batch_plan'])
    if plan.shape != (p['steps'], 16):
        raise ValueError('fixed batch plan differs')
    control = torch.load(p['control_artifact'], map_location='cpu')
    validate_binding(control, p['binding'])
    if control['binding']['classes'] != list(CLASSES):
        raise ValueError('actual class order differs')
    projection = control['projection']
    if projection['components'].shape[1] != p['dimensions']:
        raise ValueError('fixed projection dimension differs')
    torch.manual_seed(p['seed'])
    model = SupervisedProjectionProbe(projection).to(device).eval()
    initial_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
                                         for v in model.head.parameters())).hexdigest()
    if initial_hash != p['initial_head_sha256']:
        raise ValueError('head initialization differs from frozen control')
    optimizer = torch.optim.AdamW([
        {'params': model.head.parameters(), 'lr': p['head_lr']},
        {'params': [model.components], 'lr': p['projection_lr']}], weight_decay=.01)
    parameters = list(model.parameters())
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(args.protocol).read_text())
    losses, phases, used, skipped, gradient = [], [], set(), 0, None
    tick = time.monotonic()
    for requested in plan:
        ix = [int(i) for i in requested if eligibility[int(i)]['target'] is not None]
        skipped += len(requested) - len(ix)
        used.update(ix)
        if not ix or (y[ix] < 0).any():
            raise ValueError('unknown/empty batch sampled')
        x = torch.as_tensor(data['x'][ix], device=device)
        n = torch.as_tensor(data['lengths'][ix], device=device)
        mask = torch.arange(x.shape[1], device=device)[None] < n[:, None]
        relative = torch.as_tensor(data['relative_vowel'][ix], device=device)
        optimizer.zero_grad(set_to_none=True)
        logits, role = model(x, mask, relative)
        cls = torch.nn.functional.cross_entropy(logits, torch.as_tensor(y[ix], device=device))
        aux = joint_vowel_loss(role, torch.as_tensor(phase[ix], device=device))
        loss = cls + p['phase_weight'] * aux
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite supervised projection loss')
        loss.backward()
        if gradient is None:
            gradient = float(model.components.grad.norm())
            if not gradient > 0 or model.mean.grad is not None or model.scale.grad is not None:
                raise ValueError('projection/scaler gradient invariant failed')
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        losses.append(float(cls.detach()))
        phases.append(float(aux.detach()))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.monotonic() - tick
    artifact = supervised_artifact(model, p['binding'])
    learned = artifact['projection']
    for key in ('mean', 'scale'):
        if not torch.equal(learned[key], projection[key]):
            raise ValueError('fixed TRAIN scaler changed')
    if projection_digest(learned) == projection_digest(projection):
        raise ValueError('projection was not updated')
    ps, av = scores(model, data, device, p['batch'])
    torch.save(artifact, out / 'probe.pt')
    np.savez(out / 'train_scores.npz', probabilities=ps, available=av)
    reloaded = load_probe(torch.load(out / 'probe.pt', map_location='cpu'), p['binding'], device)
    check, valid = scores(reloaded, {key: value[:64] for key, value in data.items()}, device, p['batch'])
    np.testing.assert_array_equal(check, ps[:64])
    np.testing.assert_array_equal(valid, av[:64])
    proxy = pair_proxy(ps, av, y, features, labels)
    initial, final = projection['components'], learned['components']
    delta = final - initial
    novel = final - initial @ (initial.T @ final)
    report = {'scope': p['scope'], 'deployment': False, 'partition': 'train',
              'known': int((y >= 0).sum()), 'unknown': int((y < 0).sum()),
              'correct_coarse_class': int((ps.argmax(-1)[y >= 0] == y[y >= 0]).sum()),
              'available': int(av.sum()), 'skipped_draws': skipped, 'unique_sampled': len(used),
              'initial_head_sha256': initial_hash, 'first_projection_gradient_norm': gradient,
              'projection_delta_norm': float(delta.norm()), 'novel_subspace_matrix_norm': float(novel.norm()),
              'initial_pca_explained_variance_ratio': projection['explained_variance_ratio'],
              'learned_matrix_variance_ratio': None, 'scaler_unchanged': True,
              'training_seconds': seconds, 'class_loss_first_last50': [float(np.mean(losses[:50])), float(np.mean(losses[-50:]))],
              'phase_loss_first_last50': [float(np.mean(phases[:50])), float(np.mean(phases[-50:]))],
              'TRAIN_proxy': proxy, 'frozen_HuBERT_pair_control': p['frozen_HuBERT_pair_control'],
              'frozen_BERT_pair_control': p['frozen_BERT_pair_control'],
              'proceed_CAL': proxy['pair_exact'] > p['frozen_BERT_pair_control']
                  and proxy['S_to_SS']['pair_exact'] > 0 and proxy['SS_to_S']['pair_exact'] > 0,
              'artifact_sha256': service.sha(out / 'probe.pt'), 'reload_first64_exact': True}
    write(out / 'report.json', report)
    write(out / 'finished.json', {'sha256': service.sha(out / 'report.json')})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
