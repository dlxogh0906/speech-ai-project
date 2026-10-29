"""New discarded observations versus conditional-null axes on a fixed parent."""
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
from train_sibilant_frozen_probe import write
from train_sibilant_hubert_adaptation import novel_proxy, retained_subgroups
from mdd.sibilant_discarded_axes import (DiscardedAxisResidual, MODES, SCHEMA,
    axis_artifact, load_axis_residual)
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_hubert_adaptation import tensor_digest


def scores(model, data, mode, device, maximum=None):
    ps = []
    size = len(data['base_logits']) if maximum is None else maximum
    with torch.no_grad():
        for start in range(0, size, 64):
            end = min(start+64, size)
            logits = torch.as_tensor(data['base_logits'][start:end], device=device)
            raw = torch.as_tensor(data[mode][start:end], device=device)
            vowel = torch.as_tensor(data['vowel_group'][start:end], device=device)
            ps.extend(model.from_observations(logits, raw, vowel).softmax(-1).cpu().numpy())
    return np.asarray(ps), data['available'][:size].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for name, digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen axis trial input changed: '+name)
    cache, prepared = Path(p['train_cache']), Path(p['axis_cache'])
    report = json.loads((prepared/'report.json').read_text())
    if (json.loads((cache/'report.json').read_text())['partition'] != 'train'
            or report['partition'] != 'train' or report['schema'] != SCHEMA
            or service.sha(prepared/'bank.pt') != report['bank_sha256']
            or service.sha(prepared/'observations.npz') != report['observations_sha256']):
        raise ValueError('immutable TRAIN axis observations required')
    data = dict(np.load(prepared/'observations.npz'))
    bank = torch.load(prepared/'bank.pt', map_location='cpu')
    artifact = torch.load(p['control_artifact'], map_location='cpu')
    if tensor_digest(artifact['head']) != bank['base_head_sha256']:
        raise ValueError('axis bank fitted with a different parent')
    features = service.read_jsonl(p['features'])
    labels = service.read_jsonl(cache/'data.labels.jsonl')
    by = {z['feature_index']: z for z in labels if z['feature_index'] is not None}
    y = np.load(p['targets'])['y']
    original = np.load(p['base_scores'])
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(2)
    parent = novel_proxy(original['probabilities'], original['available'], y, features, by)
    if parent['novel_TP'] != p['base_novel_TP']:
        raise ValueError('original native diagnostic changed')
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    output = {'scope': p['scope'], 'baseline': parent, 'arms': {}, 'deployment': False}
    logits = torch.as_tensor(data['base_logits'], device=device)
    vowel = torch.as_tensor(data['vowel_group'], device=device)
    known = torch.as_tensor(y >= 0, device=device)
    target = torch.as_tensor(y[y >= 0], device=device)
    initial = None
    for mode in MODES:
        base = load_probe(artifact, p['binding'], device)
        before = tensor_digest(base.state_dict())
        norm = bank['normalization'][mode]
        model = DiscardedAxisResidual(base, bank['axes'][mode], norm, mode).to(device).eval()
        fingerprint = tensor_digest({'weight': model.weight, 'bias': model.bias})
        if initial is not None and initial != fingerprint:
            raise ValueError('matched trainable parameter initialization differs')
        initial = fingerprint
        ps0, av0 = scores(model, data, mode, device)
        np.testing.assert_array_equal(ps0, original['probabilities'])
        np.testing.assert_array_equal(av0, original['available'])
        trainable = [z for z in model.parameters() if z.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=p['head_lr'], weight_decay=.01)
        raw = torch.as_tensor(data[mode], device=device)
        losses = []
        tick = time.monotonic()
        for _ in range(p['steps']):
            optimizer.zero_grad(set_to_none=True)
            prediction = model.from_observations(logits, raw, vowel)
            loss = torch.nn.functional.cross_entropy(prediction[known], target)
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite discarded-axis loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        if tensor_digest(base.state_dict()) != before or not bool(model.weight.grad.abs().sum() > 0):
            raise ValueError('parent changed or new observations did not train')
        ps, av = scores(model, data, mode, device)
        np.testing.assert_array_equal(av, original['available'])
        np.testing.assert_allclose(ps[:, 2:], original['probabilities'][:, 2:], atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(ps[:, :2].sum(-1), original['probabilities'][:, :2].sum(-1), atol=2e-6, rtol=2e-5)
        saved = axis_artifact(model, artifact, norm)
        torch.save(saved, out/(mode+'.pt'))
        np.savez(out/(mode+'.train_scores.npz'), probabilities=ps, available=av)
        restored = load_axis_residual(torch.load(out/(mode+'.pt'), map_location='cpu'), p['binding'], mode, device)
        check, av_check = scores(restored, data, mode, device, maximum=64)
        np.testing.assert_array_equal(check, ps[:64])
        np.testing.assert_array_equal(av_check, av[:64])
        proxy = novel_proxy(ps, av, y, features, by)
        subgroups = retained_subgroups(ps, av, y, features, labels, proxy)
        write(out/(mode+'.subgroups.json'), subgroups)
        output['arms'][mode] = {'TRAIN_novel_proxy': proxy, 'all_retained_counts': subgroups['counts'],
            'known': int((y >= 0).sum()), 'unknown': int((y < 0).sum()),
            'initial_trainable_sha256': initial, 'trainable_parameters': sum(v.numel() for v in trainable),
            'parent_unchanged': True, 'initial_all13741_exact_parent': True, 'reload_first64_exact': True,
            'original_availability_unchanged': True,
            'other3_probability_max_difference': float(abs(ps[:, 2:]-original['probabilities'][:, 2:]).max()),
            'paired_total_probability_max_difference': float(abs(ps[:, :2].sum(-1)-original['probabilities'][:, :2].sum(-1)).max()),
            'fullpopulation_input_exposure': len(y)*p['steps'],
            'known_CE_exposure': int((y >= 0).sum())*p['steps'],
            'fit_and_scoring_seconds': time.monotonic()-tick,
            'class_loss_first_last50': [float(np.mean(losses[:50])), float(np.mean(losses[-50:]))],
            'artifact_sha256': service.sha(out/(mode+'.pt'))}
        write(out/'report.json', output)
        print(mode, json.dumps(output['arms'][mode], ensure_ascii=False), flush=True)
    signal, control = [output['arms'][m]['TRAIN_novel_proxy'] for m in MODES]
    output['proceed_signal_CAL'] = (signal['novel_TP'] > max(p['base_novel_TP'], control['novel_TP'])
        and signal['S_to_SS']['novel_TP'] > 0 and signal['SS_to_S']['novel_TP'] > 0)
    write(out/'report.json', output)
    write(out/'finished.json', {'report_sha256': service.sha(out/'report.json')})


if __name__ == '__main__':
    main()
