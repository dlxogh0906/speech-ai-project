"""Fixed whole-TRAIN CE/full-rank/precision-tail learning on immutable observations."""
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
from eval_sibilant_presence_screen import exact
from train_sibilant_frozen_probe import write
from train_sibilant_hubert_adaptation import novel_proxy, retained_subgroups
from mdd.sibilant_precision_tail import (PrecisionTailHead, MODES, SUMMARY_SCHEMA,
    rank_risk, tail_artifact, load_tail_head)
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_hubert_adaptation import tensor_digest


def load_inputs(p):
    cache, summary = Path(p['train_cache']), Path(p['summary_cache'])
    report = json.loads((summary/'report.json').read_text())
    artifact = torch.load(p['control_artifact'], map_location='cpu')
    if (json.loads((cache/'report.json').read_text())['partition'] != 'train'
            or report['partition'] != 'train' or report['schema'] != SUMMARY_SCHEMA
            or report['summary_sha256'] != service.sha(summary/'summary.npz')
            or report['base_head_sha256'] != tensor_digest(artifact['head'])
            or not report['all13741_original_probabilities_bitwise_equal']):
        raise ValueError('verified immutable TRAIN observations required')
    data = dict(np.load(summary/'summary.npz'))
    features = service.read_jsonl(p['features'])
    labels = service.read_jsonl(cache/'data.labels.jsonl')
    by = {z['feature_index']: z for z in labels if z['feature_index'] is not None}
    y = np.load(p['targets'])['y']
    if len(y) != len(features) or len(y) != len(data['summary']) or not np.isfinite(data['summary']).all():
        raise ValueError('summary/target population differs')
    groups = [{'positive': [], 'negative': []} for _ in (0, 1)]
    for i, feature in enumerate(features):
        lab = by[i]
        if feature['feature_index'] != i or any(feature[k] != lab[k] for k in ('uid', 'q_runtime')):
            raise ValueError('summary feature/label identity differs')
        if y[i] < 0 or lab['expected'] != feature['canonical']:
            continue
        opposite = 1 if feature['canonical'] == 'ㅅ' else 0
        if lab['label_kind'] == 'correct':
            if int(y[i]) != 1-opposite:
                raise ValueError('normal actual label differs from expected')
            groups[opposite]['negative'].append(i)
        elif lab['label_kind'] == 'pair_error' and not exact(lab, lab['existing']):
            if int(y[i]) != opposite:
                raise ValueError('real pair-error label differs from actual')
            groups[opposite]['positive'].append(i)
    counts = {'inputs': len(y), 'known': int((y >= 0).sum()), 'unknown': int((y < 0).sum()),
        'all_retained': len(labels), 'no_input': sum(z['feature_index'] is None for z in labels),
        'rank_positives': sum(len(z['positive']) for z in groups),
        'rank_normals': sum(len(z['negative']) for z in groups),
        'actual_class_counts': [int((y == k).sum()) for k in range(5)]}
    if any(counts[k] != p['expected_counts'][k] for k in p['expected_counts']):
        raise ValueError('predeclared TRAIN supervision counts differ: '+str(counts))
    return data, features, labels, by, y, groups, counts, artifact


def summary_scores(model, data, device, batch=64):
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data['summary']), batch):
            x = torch.as_tensor(data['summary'][start:start+batch], device=device)
            t = torch.as_tensor(data['moments'][start:start+batch], device=device)
            probabilities.extend(model.from_summary(x, t).softmax(-1).cpu().numpy())
    return np.asarray(probabilities), data['available'].copy()


def vector(gradients):
    return torch.cat([z.detach().reshape(-1) for z in gradients])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage', choices=['preflight', 'fit'], required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for name, digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen precision-tail trial changed: '+name)
    data, features, labels, by, y, groups, counts, artifact = load_inputs(p)
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(2)
    x = torch.as_tensor(data['summary'], device=device)
    moments = torch.as_tensor(data['moments'], device=device)
    known = torch.as_tensor(y >= 0, device=device)
    target = torch.as_tensor(y[y >= 0], device=device)
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    torch.manual_seed(p['seed'])
    native = load_probe(artifact, p['binding'], device)
    model = PrecisionTailHead(native, 'ce_full').to(device).eval()
    initial_ps, initial_av = summary_scores(model, data, device)
    original = np.load(Path(p['summary_cache'])/'base.scores.npz')
    np.testing.assert_array_equal(initial_ps, original['probabilities'])
    np.testing.assert_array_equal(initial_av, original['available'])
    proxy = novel_proxy(initial_ps, initial_av, y, features, by)
    if proxy['novel_TP'] != p['base_novel_TP']:
        raise ValueError('original native TRAIN selection not reproduced')
    if args.stage == 'preflight':
        before = tensor_digest(model.state_dict())
        parameters = [z for z in model.parameters() if z.requires_grad]
        gradients = {}
        for mode in MODES:
            logits = model.from_summary(x, moments)
            cls = torch.nn.functional.cross_entropy(logits[known], target)
            cg = torch.autograd.grad(cls, logits, retain_graph=True)[0]
            if not bool((cg[~known] == 0).all()):
                raise ValueError('unknown rows received CE supervision')
            class_grad = vector(torch.autograd.grad(cls, parameters, retain_graph=True))
            entry = {'class_loss': float(cls.detach()), 'class_gradient_norm': float(class_grad.norm()),
                     'unknown_CE_gradient_zero': True, 'selected_normals': []}
            if mode != 'ce_full':
                risk, selected = rank_risk(logits, groups, mode, p['tail_fraction'], p['rank_margin'])
                weighted = p['rank_weight']*risk
                lg = torch.autograd.grad(weighted, logits, retain_graph=True)[0]
                unsupported = np.ones(len(y), bool)
                for z in groups:
                    unsupported[z['positive']] = False
                    unsupported[z['negative']] = False
                if not bool((lg[torch.as_tensor(unsupported, device=device)] == 0).all()):
                    raise ValueError('rank supervision leaked outside confirmed sets')
                rg = vector(torch.autograd.grad(weighted, parameters))
                if not bool(torch.isfinite(rg).all()) or not bool(rg.norm() > 0):
                    raise ValueError('ineffective or nonfinite ranking gradient')
                entry.update(rank_loss=float(risk.detach()), rank_gradient_norm=float(rg.norm()),
                    rank_to_CE_gradient_norm_ratio=float(rg.norm()/class_grad.norm()),
                    gradient_cosine=float(torch.dot(rg, class_grad)/(rg.norm()*class_grad.norm())),
                    unsupported_rank_gradient_zero=True, selected_normals=[z.cpu().tolist() for z in selected])
            gradients[mode] = entry
        with torch.no_grad():
            full = model.from_summary(x, moments).softmax(-1).cpu().numpy()
        np.testing.assert_allclose(full, initial_ps, atol=2e-6, rtol=2e-5)
        if tensor_digest(model.state_dict()) != before:
            raise ValueError('preflight altered model weights')
        write(out/'supervision.json', {'groups': groups, 'counts': counts})
        report = {'scope': p['scope'], 'counts': counts, 'gradient': gradients, 'optimizer_steps': 0,
            'original_all13741_batch64_bitwise_equal': True,
            'fullbatch_max_probability_difference': float(abs(full-initial_ps).max()),
            'weights_unchanged': True, 'passed': True, 'deployment': False}
        write(out/'report.json', report)
        write(out/'finished.json', {'report_sha256': service.sha(out/'report.json')})
        print(json.dumps({k: report[k] for k in ('counts', 'passed', 'fullbatch_max_probability_difference')}, ensure_ascii=False), flush=True)
        return
    preflight = Path(p['preflight'])
    supervision = json.loads((preflight/'supervision.json').read_text())
    if (not json.loads((preflight/'report.json').read_text())['passed']
            or supervision['groups'] != groups or supervision['counts'] != counts):
        raise ValueError('preflight or fixed TRAIN sets changed')
    output = {'scope': p['scope'], 'counts': counts, 'baseline': proxy, 'arms': {}, 'deployment': False}
    initialization = None
    for mode in MODES:
        torch.manual_seed(p['seed'])
        base = load_probe(artifact, p['binding'], device)
        model = PrecisionTailHead(base, mode).to(device).eval()
        original_state = tensor_digest(base.state_dict())
        support = tensor_digest(model.head.support.state_dict())
        current = tensor_digest(model.head.state_dict())
        if initialization is not None and initialization != current:
            raise ValueError('initial actual heads differ')
        initialization = current
        parameters = [z for z in model.parameters() if z.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=p['head_lr'], weight_decay=.01)
        cls_losses, rank_losses = [], []
        tick = time.monotonic()
        for step in range(1, p['steps']+1):
            optimizer.zero_grad(set_to_none=True)
            logits = model.from_summary(x, moments)
            cls = torch.nn.functional.cross_entropy(logits[known], target)
            rank = logits.sum()*0
            if mode != 'ce_full':
                rank, _ = rank_risk(logits, groups, mode, p['tail_fraction'], p['rank_margin'])
            loss = cls+p['rank_weight']*rank
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite full population objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            cls_losses.append(float(cls.detach()))
            rank_losses.append(float(rank.detach()))
            if step % 100 == 0:
                print(json.dumps({'mode': mode, 'step': step, 'CE': cls_losses[-1], 'rank': rank_losses[-1]}), flush=True)
        if (tensor_digest(base.state_dict()) != original_state
                or tensor_digest(model.head.support.state_dict()) != support):
            raise ValueError('immutable native model or role head changed')
        ps, av = summary_scores(model, data, device)
        np.testing.assert_array_equal(av, initial_av)
        saved = tail_artifact(model, artifact)
        torch.save(saved, out/(mode+'.pt'))
        np.savez(out/(mode+'.train_scores.npz'), probabilities=ps, available=av)
        restored = load_tail_head(torch.load(out/(mode+'.pt'), map_location='cpu'), p['binding'], mode, device)
        small = {k: v[:64] for k, v in data.items()}
        check, check_av = summary_scores(restored, small, device)
        np.testing.assert_array_equal(check, ps[:64])
        np.testing.assert_array_equal(check_av, av[:64])
        result = novel_proxy(ps, av, y, features, by)
        subgroups = retained_subgroups(ps, av, y, features, labels, result)
        write(out/(mode+'.subgroups.json'), subgroups)
        output['arms'][mode] = {'TRAIN_novel_proxy': result, 'all_retained_counts': subgroups['counts'],
            'initial_head_sha256': initialization, 'trainable_parameters': sum(v.numel() for v in parameters),
            'base_unchanged': True, 'support_unchanged': True, 'availability_unchanged': True,
            'fullpopulation_input_exposure': len(y)*p['steps'], 'known_CE_exposure': int(known.sum())*p['steps'],
            'same_fullpopulation_inputs_all_arms': True, 'reload_first64_exact': True,
            'class_loss_first_last50': [float(np.mean(cls_losses[:50])), float(np.mean(cls_losses[-50:]))],
            'rank_loss_first_last50': [float(np.mean(rank_losses[:50])), float(np.mean(rank_losses[-50:]))],
            'fit_and_scoring_seconds': time.monotonic()-tick, 'artifact_sha256': service.sha(out/(mode+'.pt'))}
        write(out/'report.json', output)
        print(mode, json.dumps(output['arms'][mode], ensure_ascii=False), flush=True)
    tail = output['arms']['tailrank']['TRAIN_novel_proxy']
    output['proceed_tailrank_CAL'] = (tail['novel_TP'] > max(p['base_novel_TP'], *[
        output['arms'][m]['TRAIN_novel_proxy']['novel_TP'] for m in ('ce_full', 'fullrank')])
        and tail['S_to_SS']['novel_TP'] > 0 and tail['SS_to_S']['novel_TP'] > 0)
    write(out/'report.json', output)
    write(out/'finished.json', {'report_sha256': service.sha(out/'report.json')})


if __name__ == '__main__':
    main()
