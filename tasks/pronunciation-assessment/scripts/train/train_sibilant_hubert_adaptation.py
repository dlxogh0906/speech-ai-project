"""Matched original-waveform frozen/tail2 Korean HuBERT actual-phone training."""
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
import functools
import hashlib
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_sibilant_presence_screen import exact
from train_sibilant_frozen_probe import write, pair_proxy
from mdd.sibilant_frozen_probe import FrozenProbe, validate_binding, projection_digest
from mdd.sibilant_hubert_adaptation import (load_native, encode_clip, coupled_artifact,
    load_coupled, tensor_digest)
from mdd.sibilant_joint_vowel import joint_vowel_loss
from mdd.sibilant_vowel_presence import CLASSES, observed_vowel_support


def novel_proxy(ps, available, y, features, labels):
    keys = ('normal_FP', 'novel_TP', 'duplicate_TP', 'wrong_actual')
    curves = []
    for token, ci in [('ㅅ', 1), ('ㅆ', 0)]:
        groups = collections.defaultdict(collections.Counter)
        for i, cls in enumerate(y):
            if (cls < 0 or features[i]['canonical'] != token or not available[i]
                    or ps[i].argmax() != ci or ps[i, ci] < .5):
                continue
            lab = labels[i]
            g = groups[float(ps[i, ci])]
            if lab['label_kind'] == 'correct':
                g['normal_FP'] += 1
            elif lab['label_kind'] == 'pair_error' and lab['expected'] == token:
                g['duplicate_TP' if exact(lab, lab['existing']) else 'novel_TP'] += 1
            else:
                g['wrong_actual'] += 1
        counts = collections.Counter()
        curve = [dict(threshold=None, **{k: 0 for k in keys})]
        for threshold in sorted(groups, reverse=True):
            counts.update(groups[threshold])
            if counts['normal_FP'] <= 87:
                curve.append(dict(threshold=threshold, **{k: counts[k] for k in keys}))
        curves.append(curve)
    def rank(pair):
        a, b = pair
        return (a['novel_TP'] + b['novel_TP'], -a['normal_FP'] - b['normal_FP'],
                -a['wrong_actual'] - b['wrong_actual'], a['duplicate_TP'] + b['duplicate_TP'])
    a, b = max(((a, b) for a in curves[0] for b in curves[1]
                if a['normal_FP'] + b['normal_FP'] <= 87), key=rank)
    return dict({k: a[k] + b[k] for k in keys}, S_to_SS=a, SS_to_S=b,
                scope='TRAIN diagnostic selection only; 87 supported-normal budget, not service 1%')


def retained_subgroups(ps, available, y, features, labels, proxy):
    counts = collections.Counter()
    grouped = {k: collections.defaultdict(collections.Counter)
               for k in ('following', 'position', 'item', 'language')}
    for lab in labels:
        i = lab['feature_index']
        counter = collections.Counter(all=1)
        if lab['label_kind'] == 'correct':
            counter['normal_all'] += 1
        if i is None:
            counter['no_input'] += 1
        else:
            counter['TRAIN_target_unknown' if y[i] < 0 else 'supported'] += 1
            f = features[i]
            direction = 'S_to_SS' if f['canonical'] == 'ㅅ' else 'SS_to_S'
            ci = 1 if f['canonical'] == 'ㅅ' else 0
            threshold = proxy[direction]['threshold']
            call = (threshold is not None and available[i] and ps[i].argmax() == ci
                    and ps[i, ci] >= threshold)
            if call:
                counter['calls'] += 1
                if y[i] < 0:
                    counter['TRAIN_target_unknown_calls'] += 1
                if lab['label_kind'] == 'correct':
                    counter['normal_false_calls'] += 1
                elif lab['label_kind'] == 'ambiguous':
                    counter['ambiguous_calls'] += 1
                elif lab['expected'] == f['canonical'] and lab['actual'] == ('ㅆ' if ci == 1 else 'ㅅ'):
                    counter['exact_pair_calls'] += 1
                    counter['duplicate_calls' if exact(lab, lab['existing']) else 'novel_calls'] += 1
                else:
                    counter['wrong_actual_calls'] += 1
        counts.update(counter)
        for key in grouped:
            grouped[key][str(lab.get(key))].update(counter)
    return {'counts': dict(counts), 'groups': {k: {v: dict(c) for v, c in values.items()}
            for k, values in grouped.items()},
            'scope': 'All retained TRAIN sites; standalone pair proposals, not final API. Unknown/no-input retained.'}


class RawDataset:
    def __init__(self, cache, features, relative, lengths):
        self.features, self.relative, self.lengths = features, relative, lengths
        wanted = {f['uid'] for f in features}
        self.rows = {}
        for line in (Path(cache) / 'data.responses.jsonl').open():
            z = json.loads(line)
            if z['row']['uid'] in wanted:
                row = z['row']
                self.rows[row['uid']] = {k: row[k] for k in ('uid', 'wav', 'frames') if k in row}
        if set(self.rows) != wanted:
            raise ValueError('missing original source rows')
        self.source = functools.lru_cache(maxsize=256)(self._source)

    def _source(self, uid):
        return service.load_audio(self.rows[uid])[:320000]

    def batch(self, encoder, processor, ix, device):
        width = self.relative.shape[1]
        hidden = []
        for i in ix:
            f = self.features[i]
            value = encode_clip(encoder, processor, self.source(f['uid']), f)
            if len(value) != self.lengths[i] or len(value) > width:
                raise ValueError('runtime grid length changed')
            hidden.append(torch.nn.functional.pad(value, (0, 0, 0, width - len(value))))
        x = torch.stack(hidden)
        n = torch.as_tensor(self.lengths[ix], device=device)
        mask = torch.arange(width, device=device)[None] < n[:, None]
        relative = torch.as_tensor(self.relative[ix], device=device)
        return x, mask, relative


def score_raw(encoder, processor, head, data, device, batch, count=None, progress=None):
    ps, av = [], []
    encoder.eval()
    head.eval()
    size = len(data.features) if count is None else count
    with torch.no_grad():
        for start in range(0, size, batch):
            ix = list(range(start, min(size, start + batch)))
            x, mask, relative = data.batch(encoder, processor, ix, device)
            logits, role = head(x, mask, relative)
            ps.extend(logits.softmax(-1).cpu().numpy())
            av.extend(observed_vowel_support(role, mask).cpu().numpy())
            if progress and (start % 512 == 0 or start + batch >= size):
                print(json.dumps({'scoring': progress, 'done': min(size, start + batch), 'total': size}), flush=True)
    return np.asarray(ps), np.asarray(av)


def fit(p, out, protocol_path):
    cache = Path(p['train_cache'])
    if json.loads((cache / 'report.json').read_text())['partition'] != 'train':
        raise ValueError('only TRAIN may fit native encoder')
    features = service.read_jsonl(p['features'])
    reference = service.read_jsonl(cache / 'data.features.jsonl')
    labels = service.read_jsonl(cache / 'data.labels.jsonl')
    indexed_labels = {z['feature_index']: z for z in labels if z['feature_index'] is not None}
    targets = np.load(p['targets'])
    y, phase = targets['y'], targets['phase']
    np.testing.assert_array_equal(y, np.load(cache / 'targets.npz')['y'])
    stored = np.load(p['data'])
    lengths, relative = stored['lengths'], stored['relative_vowel']
    if len(features) != len(reference) or len(y) != len(features) or phase.shape != relative.shape:
        raise ValueError('original TRAIN coverage differs')
    for i, (a, b) in enumerate(zip(features, reference)):
        if (any(a[k] != b[k] for k in ('uid', 'q_runtime', 'left', 'right', 'clip_sha256'))
                or a['feature_index'] != i or a['input_frames'] != lengths[i]):
            raise ValueError('runtime source order/grid differs')
    eligibility = service.read_jsonl(p['ctc_targets'])
    plan = np.load(p['batch_plan'])
    if plan.shape != (300, 16) or p['steps'] != 300:
        raise ValueError('fixed 300x16 replay plan required')
    control = torch.load(p['control_artifact'], map_location='cpu')
    validate_binding(control, p['binding'])
    if control['binding']['classes'] != list(CLASSES):
        raise ValueError('actual classes differ')
    data = RawDataset(cache, features, relative, lengths)
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    projection = control['projection']
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(protocol_path).read_text())
    report = {'scope': p['scope'], 'partition': 'train', 'deployment': False, 'arms': {}}
    for mode in ('frozen', 'tail2'):
        encoder, processor, binding = load_native(p['model_manifest'], device)
        if mode == 'tail2':
            encoder.encoder.layers[-2:].requires_grad_(True)
        torch.manual_seed(p['seed'])
        head = FrozenProbe(projection).to(device).eval()
        initial = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
                                         for v in head.head.parameters())).hexdigest()
        if initial != p['initial_head_sha256']:
            raise ValueError('matched fresh head initialization differs')
        tail_before = tensor_digest(encoder.encoder.layers[-2:].state_dict())
        prefix_before = tensor_digest({k: v for k, v in encoder.state_dict().items()
            if not k.startswith(('encoder.layers.10.', 'encoder.layers.11.'))})
        groups = [{'params': list(head.head.parameters()), 'lr': p['head_lr']}]
        if mode == 'tail2':
            groups.append({'params': list(encoder.encoder.layers[-2:].parameters()), 'lr': p['encoder_lr']})
        trainable = [v for group in groups for v in group['params']]
        optimizer = torch.optim.AdamW(groups, weight_decay=.01)
        losses, phase_losses, used, skipped = [], [], set(), 0
        first_grad = None
        tick = time.monotonic()
        for step, requested in enumerate(plan, 1):
            ix = [int(i) for i in requested if eligibility[int(i)]['target'] is not None]
            used.update(ix)
            skipped += len(requested) - len(ix)
            if not ix or (y[ix] < 0).any():
                raise ValueError('empty/unknown sampled class')
            optimizer.zero_grad(set_to_none=True)
            x, mask, rel = data.batch(encoder, processor, ix, device)
            logits, role = head(x, mask, rel)
            cls = torch.nn.functional.cross_entropy(logits, torch.as_tensor(y[ix], device=device))
            aux = joint_vowel_loss(role, torch.as_tensor(phase[ix], device=device))
            loss = cls + p['phase_weight'] * aux
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite native adaptation loss')
            loss.backward()
            if step == 1:
                forbidden = [name for name, value in encoder.named_parameters() if value.grad is not None
                             and not (mode == 'tail2' and name.startswith(('encoder.layers.10.', 'encoder.layers.11.')))]
                norms = [float(sum(v.grad.square().sum() for v in layer.parameters() if v.grad is not None).sqrt())
                         for layer in encoder.encoder.layers[-2:]] if mode == 'tail2' else []
                if forbidden or (mode == 'tail2' and not all(v > 0 and np.isfinite(v) for v in norms)):
                    raise ValueError('native gradient isolation failed')
                first_grad = {'tail_norms': norms, 'forbidden_gradients': forbidden}
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            losses.append(float(cls.detach()))
            phase_losses.append(float(aux.detach()))
            if step % 20 == 0:
                print(json.dumps({'arm': mode, 'step': step, 'class_loss': losses[-1],
                                  'phase_loss': phase_losses[-1], 'seconds': time.monotonic() - tick}), flush=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.monotonic() - tick
        if prefix_before != tensor_digest({k: v for k, v in encoder.state_dict().items()
                if not k.startswith(('encoder.layers.10.', 'encoder.layers.11.'))}):
            raise ValueError('frozen encoder prefix changed')
        artifact = coupled_artifact(encoder, head, binding, mode)
        if projection_digest(artifact['projection']) != projection_digest(projection):
            raise ValueError('fixed projection changed')
        tail_changed = artifact['binding']['tail_sha256'] != tail_before
        if tail_changed != (mode == 'tail2'):
            raise ValueError('unexpected tail update state')
        torch.save(artifact, out / (mode + '.pt'))
        del optimizer
        encoder.requires_grad_(False)
        ps, av = score_raw(encoder, processor, head, data, device, p['batch'], progress=mode)
        np.savez(out / (mode + '.train_scores.npz'), probabilities=ps, available=av)
        # Reload the serialized tail AND head; re-encode original PCM, no frozen
        # feature cache masquerading as an adapted representation.
        restored = load_coupled(torch.load(out / (mode + '.pt'), map_location='cpu'), binding, encoder, device)
        check, valid = score_raw(encoder, processor, restored, data, device, p['batch'], count=64)
        np.testing.assert_array_equal(check, ps[:64])
        np.testing.assert_array_equal(valid, av[:64])
        novel = novel_proxy(ps, av, y, features, indexed_labels)
        subgroups = retained_subgroups(ps, av, y, features, labels, novel)
        write(out / (mode + '.subgroups.json'), subgroups)
        report['arms'][mode] = {'known': int((y >= 0).sum()), 'unknown': int((y < 0).sum()),
            'available': int(av.sum()), 'correct_coarse_class': int((ps.argmax(-1)[y >= 0] == y[y >= 0]).sum()),
            'skipped_draws': skipped, 'unique_sampled': len(used), 'initial_head_sha256': initial,
            'prefix_unchanged': True, 'tail_changed': tail_changed, 'first_gradient': first_grad,
            'training_seconds': seconds, 'reload_raw_first64_exact': True,
            'class_loss_first_last50': [float(np.mean(losses[:50])), float(np.mean(losses[-50:]))],
            'phase_loss_first_last50': [float(np.mean(phase_losses[:50])), float(np.mean(phase_losses[-50:]))],
            'TRAIN_novel_proxy': novel, 'TRAIN_total_proxy': pair_proxy(ps, av, y, features, indexed_labels),
            'all_retained_counts': subgroups['counts'], 'artifact_sha256': service.sha(out / (mode + '.pt'))}
        write(out / 'report.json', report)
        print(mode, json.dumps(report['arms'][mode], ensure_ascii=False), flush=True)
        del encoder, head, restored, artifact
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    a, b = [report['arms'][m]['TRAIN_novel_proxy'] for m in ('frozen', 'tail2')]
    report['proceed_tail2_CAL'] = (b['novel_TP'] > a['novel_TP']
        and b['S_to_SS']['novel_TP'] > 0 and b['SS_to_S']['novel_TP'] > 0)
    write(out / 'report.json', report)
    write(out / 'finished.json', {'report_sha256': service.sha(out / 'report.json')})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p = json.loads(Path(args.protocol).read_text())
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    torch.set_num_threads(2)
    fit(p, Path(args.out), args.protocol)


if __name__ == '__main__':
    main()
