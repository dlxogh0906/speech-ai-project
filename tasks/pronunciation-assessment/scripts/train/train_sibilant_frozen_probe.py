"""Matched frozen-encoder feature probe; TRAIN support and fixed-capacity fit."""
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
import hashlib
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from mdd.sibilant_frozen_probe import sampled_frames, fit_projection, projection_digest, FrozenProbe, load_probe, READOUT
from mdd.sibilant_joint_vowel import joint_vowel_loss
from mdd.sibilant_vowel_presence import training_target, observed_vowel_support, CLASSES
from mdd.sibilant_vowel_reference import annotated_intended_vowel_interval
from mdd.sibilant_tag_support import single_phone_intervals


def stream(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def support(p, out):
    cache = Path(p['train_cache'])
    original = np.load(cache / 'targets.npz')
    features = service.read_jsonl(p['representations']['wave']['features'])
    wanted = {z['uid'] for z in features}
    annotations = {z['stem']: z for z in stream(p['annotations']) if z['stem'] in wanted}
    rows = {z['row']['uid']: z['row'] for z in stream(cache / 'data.responses.jsonl') if z['row']['uid'] in wanted}
    labels = {z['feature_index']: z for z in stream(cache / 'data.labels.jsonl') if z['feature_index'] is not None}
    wave = np.load(p['representations']['wave']['data'])
    lengths = wave['lengths']
    y = original['y'].copy()
    phase = np.full((len(y), wave['x'].shape[1]), -100, np.int64)
    geometry, counts = {}, collections.Counter()
    for i, f in enumerate(features):
        if f['feature_index'] != i or f['input_frames'] != lengths[i]:
            raise ValueError('wave feature identity/grid differs')
        if y[i] < 0:
            counts['original_unknown_preserved'] += 1
            continue
        uid, lab = f['uid'], labels[i]
        if uid not in geometry:
            row = rows[uid]
            geometry[uid] = (single_phone_intervals(annotations[uid]['tag_phones'], row['actual']),
                             service.all_optimal_mapping(row['canonical'], row['actual']))
        intervals, mapping = geometry[uid]
        opts = mapping[lab['q'] + 1]
        v = annotated_intended_vowel_interval(intervals, opts)
        c = intervals[lab['a']] if intervals is not None and lab['a'] is not None else None
        centers = (f['left'] + (p['wave_receptive_samples'] - 1) / 2.
                   + np.arange(int(lengths[i])) * p['wave_stride_samples']) / 16000.
        actual, target, state = training_target(lab, c, v, f['left'], f['right'], centers,
                                              lab['a'] is not None and opts == {lab['a'] + 1})
        if actual != y[i]:
            raise ValueError('retimed support changed an original known actual class')
        phase[i, :len(target)] = target
        counts[state] += 1
    np.savez(out / 'wave_targets.npz', y=y, phase=phase)
    report = {'scope': p['scope'], 'partition': 'train', 'counts': dict(counts),
              'classes_exactly_unchanged': bool(np.array_equal(y, original['y'])),
              'coarse_roles': {str(k): int((phase == k).sum()) for k in [-100, 0, 1, 2]},
              'no_interpolated_TAG_boundaries': True, 'runtime_TAG_input': False,
              'targets_sha256': service.sha(out / 'wave_targets.npz')}
    write(out / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def scores(model, data, device, batch):
    ps, available = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data['x']), batch):
            x = torch.as_tensor(data['x'][start:start + batch], device=device)
            n = torch.as_tensor(data['lengths'][start:start + batch], device=device)
            mask = torch.arange(x.shape[1], device=device)[None, :] < n[:, None]
            relative = torch.as_tensor(data['relative_vowel'][start:start + batch], device=device)
            logits, phase = model(x, mask, relative)
            ps.extend(logits.softmax(-1).cpu().numpy())
            available.extend(observed_vowel_support(phase, mask).cpu().numpy())
    return np.asarray(ps), np.asarray(available)


def pair_proxy(ps, available, y, features, labels):
    known = [i for i in range(len(y)) if y[i] >= 0]
    normal = sum(labels[i]['label_kind'] == 'correct' for i in known)
    budget = int(.01 * normal)
    curves = []
    for expected, actual, ci in [('ㅅ', 'ㅆ', 1), ('ㅆ', 'ㅅ', 0)]:
        groups = collections.defaultdict(collections.Counter)
        for i in known:
            if (features[i]['canonical'] != expected or not available[i]
                    or ps[i].argmax() != ci or ps[i, ci] < .5):
                continue
            lab, c = labels[i], groups[float(ps[i, ci])]
            if lab['label_kind'] == 'correct':
                c['false_normal'] += 1
            elif lab['label_kind'] == 'pair_error' and lab['actual'] == actual and lab['expected'] == expected:
                c['pair_exact'] += 1
            else:
                c['wrong_actual'] += 1
        counts = collections.Counter()
        curve = [{'threshold': None, 'pair_exact': 0, 'false_normal': 0, 'wrong_actual': 0}]
        for threshold in sorted(groups, reverse=True):
            counts.update(groups[threshold])
            curve.append(dict(threshold=threshold, **{k: counts[k] for k in ['pair_exact', 'false_normal', 'wrong_actual']}))
        curves.append([x for x in curve if x['false_normal'] <= budget])
    feasible = [(a, b) for a in curves[0] for b in curves[1] if a['false_normal'] + b['false_normal'] <= budget]
    a, b = max(feasible, key=lambda z: (z[0]['pair_exact'] + z[1]['pair_exact'],
                                       -z[0]['false_normal'] - z[1]['false_normal'],
                                       -z[0]['wrong_actual'] - z[1]['wrong_actual']))
    return {'supported_normal': normal, 'normal_budget': budget,
            'pair_exact': a['pair_exact'] + b['pair_exact'], 'normal_false_calls': a['false_normal'] + b['false_normal'],
            'third_or_deletion_wrong_calls': a['wrong_actual'] + b['wrong_actual'],
            'S_to_SS': a, 'SS_to_S': b,
            'scope': 'TRAIN diagnostic thresholds only; not calibrated service/API performance. Unknown inputs retained separately.'}


def fit(p, out):
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(2)
    cache = Path(p['train_cache'])
    reference_features = service.read_jsonl(cache / 'data.features.jsonl')
    labels = {z['feature_index']: z for z in stream(cache / 'data.labels.jsonl') if z['feature_index'] is not None}
    original = np.load(cache / 'targets.npz')
    y = original['y']
    targets = service.read_jsonl(p['ctc_targets'])
    plan = np.load(p['batch_plan'])
    if plan.shape != (p['steps'], 16):
        raise ValueError('fixed sampling plan differs')
    initial_hash, report = None, {'scope': p['scope'], 'deployment': False, 'arms': {}}
    for name, spec in p['representations'].items():
        data = dict(np.load(spec['data']))
        features = service.read_jsonl(spec['features'])
        for a, b in zip(reference_features, features):
            if (a['uid'], a['q_runtime']) != (b['uid'], b['q_runtime']):
                raise ValueError('representation site order differs')
        if len(data['x']) != len(y) or len(features) != len(y):
            raise ValueError('representation coverage differs')
        target = original if name == 'bert' else np.load(Path(p['support_dir']) / 'wave_targets.npz')
        if not np.array_equal(target['y'], y):
            raise ValueError('representation class targets differ')
        selected = sampled_frames(features, data['lengths'], p['projection_frames'])
        sample = data['x'][selected[:, 0], selected[:, 1]].astype(np.float32)
        tick = time.monotonic()
        projection = fit_projection(sample, p['dimensions'], device)
        projection_seconds = time.monotonic() - tick
        np.save(out / (name + '.pca_sample_frames.npy'), selected)
        torch.manual_seed(p['seed'])
        model = FrozenProbe(projection).to(device).eval()
        fingerprint = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.head.parameters())).hexdigest()
        if initial_hash is None:
            initial_hash = fingerprint
        elif fingerprint != initial_hash:
            raise ValueError('matched head initial parameters differ')
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=p['head_lr'], weight_decay=.01)
        losses, phases, used, skipped = [], [], set(), 0
        tick = time.monotonic()
        for step, requested in enumerate(plan, 1):
            ix = [int(i) for i in requested if targets[int(i)]['target'] is not None]
            skipped += len(requested) - len(ix)
            used.update(ix)
            if not ix or (y[ix] < 0).any():
                raise ValueError('unknown class sampled')
            x = torch.as_tensor(data['x'][ix], device=device)
            n = torch.as_tensor(data['lengths'][ix], device=device)
            mask = torch.arange(x.shape[1], device=device)[None, :] < n[:, None]
            relative = torch.as_tensor(data['relative_vowel'][ix], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, phase = model(x, mask, relative)
            cls = torch.nn.functional.cross_entropy(logits, torch.as_tensor(y[ix], device=device))
            aux = joint_vowel_loss(phase, torch.as_tensor(target['phase'][ix], device=device))
            loss = cls + p['phase_weight'] * aux
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite frozen probe loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.)
            optimizer.step()
            losses.append(float(cls.detach()))
            phases.append(float(aux.detach()))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.monotonic() - tick
        ps, available = scores(model, data, device, p['batch'])
        np.savez(out / (name + '.train_scores.npz'), probabilities=ps, available=available)
        binding = dict(spec['binding'], projection_sha256=projection_digest(projection), classes=list(CLASSES))
        artifact = {'readout': READOUT, 'binding': binding, 'projection': projection,
                    'head': {k: v.detach().cpu() for k, v in model.head.state_dict().items()}}
        torch.save(artifact, out / (name + '.pt'))
        reloaded = load_probe(artifact, spec['binding'], device)
        check, av = scores(reloaded, {k: v[:64] for k, v in data.items()}, device, p['batch'])
        np.testing.assert_array_equal(check, ps[:64])
        np.testing.assert_array_equal(av, available[:64])
        known = y >= 0
        report['arms'][name] = {'known': int(known.sum()), 'unknown': int((~known).sum()),
            'correct_coarse_class': int((ps.argmax(-1)[known] == y[known]).sum()),
            'available': int(available.sum()), 'skipped_draws': skipped, 'unique_sampled': len(used),
            'initial_head_sha256': fingerprint, 'projection_seconds': projection_seconds,
            'projection_explained_variance': projection['explained_variance_ratio'], 'training_seconds': seconds,
            'class_loss_first_last50': [float(np.mean(losses[:50])), float(np.mean(losses[-50:]))],
            'phase_loss_first_last50': [float(np.mean(phases[:50])), float(np.mean(phases[-50:]))],
            'TRAIN_proxy': pair_proxy(ps, available, y, features, labels),
            'artifact_sha256': service.sha(out / (name + '.pt')), 'reload_first64_exact': True}
        write(out / 'report.json', report)
        print(name, json.dumps(report['arms'][name], ensure_ascii=False), flush=True)
        del model, reloaded, optimizer, data
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    a, b = [report['arms'][name]['TRAIN_proxy'] for name in ('bert', 'wave')]
    report['proceed_wave_CAL'] = b['pair_exact'] > a['pair_exact'] and b['S_to_SS']['pair_exact'] > 0 and b['SS_to_S']['pair_exact'] > 0
    write(out / 'report.json', report)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage', choices=['support', 'fit'], required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen input changed: ' + path)
    if json.loads((Path(p['train_cache']) / 'report.json').read_text())['partition'] != 'train':
        raise ValueError('TRAIN-only projection/support/fit required')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(a.protocol).read_text())
    (support if a.stage == 'support' else fit)(p, out)
    write(out / 'finished.json', {'sha256': service.sha(out / 'report.json')})


if __name__ == '__main__':
    main()
