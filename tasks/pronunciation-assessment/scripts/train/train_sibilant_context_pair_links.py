"""Actual-only versus matched/shuffled context-pair TRAIN learning controls."""
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
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from train_sibilant_frozen_probe import scores, write
from train_sibilant_hubert_adaptation import novel_proxy, retained_subgroups
from mdd.sibilant_context_pairs import contrast_loss, SCHEMA
from mdd.sibilant_context_pair_head import ContextPairHead, pair_artifact, load_pair_head, MODES
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_hubert_adaptation import tensor_digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for name, digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen context-pair trial changed: '+name)
    cache, pair_cache = Path(p['train_cache']), Path(p['pair_cache'])
    report = json.loads((pair_cache/'report.json').read_text())
    if (json.loads((cache/'report.json').read_text())['partition'] != 'train'
            or report['partition'] != 'train' or report['schema'] != SCHEMA
            or service.sha(pair_cache/'plan.npz') != report['plan_sha256']):
        raise ValueError('verified TRAIN-only comparison plan required')
    features = service.read_jsonl(p['features'])
    labels = service.read_jsonl(cache/'data.labels.jsonl')
    by = {z['feature_index']: z for z in labels if z['feature_index'] is not None}
    data, y = dict(np.load(p['data'])), np.load(p['targets'])['y']
    plans = dict(np.load(pair_cache/'plan.npz'))
    plan = plans['batch_plan']
    if plan.shape != (300, 16):
        raise ValueError('fixed300x16 plan required')
    artifact = torch.load(p['control_artifact'], map_location='cpu')
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    baseline = load_probe(artifact, p['binding'], device)
    base_ps, base_av = scores(baseline, data, device, 64)
    base_proxy = novel_proxy(base_ps, base_av, y, features, by)
    if base_proxy['novel_TP'] != p['base_novel_TP']:
        raise ValueError('original native diagnostic did not reproduce')
    np.savez(out/'base.train_scores.npz', probabilities=base_ps, available=base_av)
    output = {'scope': p['scope'], 'baseline': base_proxy, 'arms': {}, 'deployment': False}
    initialization = None
    for mode in MODES:
        torch.manual_seed(p['seed'])
        native = load_probe(artifact, p['binding'], device)
        model = ContextPairHead(native, mode).to(device).eval()
        before = tensor_digest(native.state_dict())
        support = tensor_digest(model.head.support.state_dict())
        current = tensor_digest(model.head.state_dict())
        if initialization is not None and current != initialization:
            raise ValueError('actual-head initialization differs')
        initialization = current
        initial_ps, initial_av = scores(model, {k: v[:64] for k, v in data.items()}, device, 64)
        np.testing.assert_array_equal(initial_ps, base_ps[:64])
        np.testing.assert_array_equal(initial_av, base_av[:64])
        parameters = [v for v in model.parameters() if v.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=p['head_lr'], weight_decay=.01)
        cls_losses, pair_losses = [], []
        tick = time.monotonic()
        for step, row in enumerate(plan):
            valid = row >= 0
            ix = row[valid]
            if (y[ix] < 0).any():
                raise ValueError('unknown class sampled')
            x = torch.as_tensor(data['x'][ix], device=device)
            n = torch.as_tensor(data['lengths'][ix], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < n[:, None]
            relative = torch.as_tensor(data['relative_vowel'][ix], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x, mask, relative)
            cls = torch.nn.functional.cross_entropy(logits, torch.as_tensor(y[ix], device=device))
            pair = logits.sum()*0
            if mode != 'ce_only':
                link = plans[mode][step]
                link = link[link[:, 0] >= 0]
                remap = np.full(len(row), -1, np.int64)
                remap[valid] = np.arange(len(ix))
                link = remap[link]
                if (link < 0).any() or not ((y[ix[link[:, 0]]] == 1) & (y[ix[link[:, 1]]] == 0)).all():
                    raise ValueError('pair links include invalid padding or class')
                pair = contrast_loss(logits, link, p['pair_margin'])
            loss = cls+p['pair_weight']*pair
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite comparison loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            cls_losses.append(float(cls.detach()))
            pair_losses.append(float(pair.detach()))
        if (tensor_digest(native.state_dict()) != before
                or tensor_digest(model.head.support.state_dict()) != support
                or any(v.grad is not None for v in model.head.support.parameters())):
            raise ValueError('immutable original or support head changed')
        ps, av = scores(model, data, device, 64)
        np.testing.assert_array_equal(av, base_av)
        saved = pair_artifact(model, artifact)
        torch.save(saved, out/(mode+'.pt'))
        np.savez(out/(mode+'.train_scores.npz'), probabilities=ps, available=av)
        restored = load_pair_head(torch.load(out/(mode+'.pt'), map_location='cpu'), p['binding'], mode, device)
        check, check_av = scores(restored, {k: v[:64] for k, v in data.items()}, device, 64)
        np.testing.assert_array_equal(check, ps[:64])
        np.testing.assert_array_equal(check_av, av[:64])
        proxy = novel_proxy(ps, av, y, features, by)
        subgroups = retained_subgroups(ps, av, y, features, labels, proxy)
        write(out/(mode+'.subgroups.json'), subgroups)
        output['arms'][mode] = {'TRAIN_novel_proxy': proxy, 'all_retained_counts': subgroups['counts'],
            'known': int((y >= 0).sum()), 'unknown': int((y < 0).sum()), 'available': int(av.sum()),
            'initial_head_sha256': initialization, 'base_unchanged': True, 'support_unchanged': True,
            'initial_first64_base_exact': True, 'reload_first64_exact': True,
            'identical_batch_exposure': True, 'trainable_parameters': sum(v.numel() for v in parameters),
            'fit_and_scoring_seconds': time.monotonic()-tick,
            'class_loss_first_last50': [float(np.mean(cls_losses[:50])), float(np.mean(cls_losses[-50:]))],
            'pair_loss_first_last50': [float(np.mean(pair_losses[:50])), float(np.mean(pair_losses[-50:]))],
            'artifact_sha256': service.sha(out/(mode+'.pt'))}
        write(out/'report.json', output)
        print(mode, json.dumps(output['arms'][mode], ensure_ascii=False), flush=True)
    matched = output['arms']['matched']['TRAIN_novel_proxy']
    controls = [output['arms'][m]['TRAIN_novel_proxy']['novel_TP'] for m in ('ce_only', 'shuffled')]
    output['proceed_matched_CAL'] = (matched['novel_TP'] > max([p['base_novel_TP']]+controls)
        and matched['S_to_SS']['novel_TP'] > 0 and matched['SS_to_S']['novel_TP'] > 0)
    write(out/'report.json', output)
    write(out/'finished.json', {'report_sha256': service.sha(out/'report.json')})


if __name__ == '__main__':
    main()
