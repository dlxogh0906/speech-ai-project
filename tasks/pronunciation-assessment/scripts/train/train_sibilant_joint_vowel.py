"""Matched2x2 CAL screen: split/undivided vowel targets and learned/progress pool.

All parameters, first300paired batches, acoustic/class inputs and update layers
are fixed. Existing corrected-clock split/learned control is reused exactly.
Strict versus strict-or-CV availability is a separate matched inference factor.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/eval")]
import argparse
import hashlib
import json
import shutil
import time
import numpy as np
import torch
import train_sibilant_cv_support as trainer
from train_sibilant_c_trajectory import aucs
from train_sibilant_local_encoder import load_encoder, hidden, write
from mdd.sibilant_context_batches import make_plan
from mdd.sibilant_cv_support import phase_loss, reliable_support
from mdd.sibilant_joint_vowel import JointVowelReadout, joint_vowel_loss, strict_or_joint_support


def score(encoder, head, data):
    if set(data) != {'x', 'lengths', 'relative'}:
        raise ValueError('inference accepts only observed inputs and runtime positions')
    ps, strict, joint = [], [], []
    device, started = next(encoder.parameters()).device, time.monotonic()
    with torch.no_grad():
        for start in range(0, len(data['x']), 64):
            sl = slice(start, start + 64)
            x = torch.as_tensor(data['x'][sl], device=device)
            lengths = torch.as_tensor(data['lengths'][sl], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
            relative = torch.as_tensor(data['relative'][sl], device=device)
            logits, phase = head(hidden(encoder, x, mask), mask, relative)
            ps.extend(logits.softmax(-1).cpu().numpy())
            strict.extend(reliable_support(phase, mask).cpu().tolist())
            joint.extend(strict_or_joint_support(phase, mask).cpu().tolist())
    torch.cuda.synchronize()
    return np.asarray(ps), np.asarray(strict), np.asarray(joint), time.monotonic() - started


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p['sha256'].items():
        if trainer.service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / 'protocol.json')
    shutil.copy2(__file__, out / Path(__file__).name)
    torch.set_num_threads(2)
    train = dict(np.load(p['train_prefix'] + '.npz'))
    cal = dict(np.load(p['cal_prefix'] + '.npz'))
    merged_phase = np.load(p['joint_phase'])['phase']
    if merged_phase.shape != train['phase'].shape or set(cal) != {'x', 'lengths', 'relative'}:
        raise ValueError('targets or inference schema differ')
    train_labels = trainer.service.read_jsonl(p['train_prefix'] + '.labels.jsonl')
    train_features = trainer.service.read_jsonl(p['train_prefix'] + '.features.jsonl')
    plan, _, _ = make_plan(train['y'], train_labels, train_features, p['seed'], p['steps'], 'paired_context')
    np.testing.assert_array_equal(plan, np.load(p['control_batch_plan']))
    np.save(out / 'batch_plan.npy', plan)
    control_fit = json.loads(Path(p['control_fit']).read_text())['arms']['learned_support']
    initial_guards, report = {}, {'scope': p['scope'], 'test_scored': False, 'inference_batch': 64,
                                 'service_improvement_proven': False, 'arms': {}, 'cells': {}, 'criterion': p['criterion']}
    labels = trainer.service.read_jsonl(p['cal_prefix'] + '.labels.jsonl')
    features = trainer.service.read_jsonl(p['cal_prefix'] + '.features.jsonl')
    definitions = [('split_learned', 'split', 'learned'), ('merged_learned', 'merged', 'learned'),
                   ('split_progress', 'split', 'vowel_progress'), ('merged_progress', 'merged', 'vowel_progress')]
    for arm, supervision, pooling in definitions:
        encoder = load_encoder(p, arm != 'split_learned')
        torch.manual_seed(p['seed'])
        head = JointVowelReadout(encoder.config.hidden_size, pooling).to(next(encoder.parameters()).device)
        initial_sha = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in head.parameters())).hexdigest()
        if initial_sha != control_fit['initial_head_sha256']:
            raise AssertionError('initial parameters differ')
        device, started, losses = next(encoder.parameters()).device, time.monotonic(), []
        initial_indices = plan[0]
        with torch.no_grad():
            x = torch.as_tensor(train['x'][initial_indices], device=device)
            lengths = torch.as_tensor(train['lengths'][initial_indices], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
            actual, phase = head(hidden(encoder, x, mask), mask, torch.as_tensor(train['relative'][initial_indices], device=device))
            guard = [actual.cpu().numpy(), phase.cpu().numpy()]
        if initial_guards:
            np.testing.assert_array_equal(guard[1], next(iter(initial_guards.values()))[1])
        if pooling in initial_guards:
            np.testing.assert_array_equal(guard[0], initial_guards[pooling][0])
        else:
            initial_guards[pooling] = guard
        if arm == 'split_learned':
            reused = torch.load(p['control_artifact'], map_location='cpu')
            encoder.encoder.layers[-2:].load_state_dict(reused['last2'])
            head.load_state_dict(reused['head'])
            train_seconds = 0.
        else:
            targets = train['phase'] if supervision == 'split' else merged_phase
            optimizer = torch.optim.AdamW([{'params': head.parameters(), 'lr': .001},
                                          {'params': [v for v in encoder.parameters() if v.requires_grad], 'lr': .00001}], weight_decay=.01)
            torch.cuda.reset_peak_memory_stats()
            for step, indices in enumerate(plan, 1):
                optimizer.zero_grad(set_to_none=True)
                x = torch.as_tensor(train['x'][indices], device=device)
                lengths = torch.as_tensor(train['lengths'][indices], device=device)
                mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
                logits, phase = head(hidden(encoder, x, mask), mask, torch.as_tensor(train['relative'][indices], device=device))
                loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(train['y'][indices], device=device))
                target = torch.as_tensor(targets[indices], device=device)
                auxiliary = phase_loss(phase, target) if supervision == 'split' else joint_vowel_loss(phase, target)
                loss = loss + p['phase_weight'] * auxiliary
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                loss.backward()
                if step == 1 and any(v.grad is not None for layer in encoder.encoder.layers[:-2] for v in layer.parameters()):
                    raise AssertionError('earlier encoder gradient')
                torch.nn.utils.clip_grad_norm_([v for group in optimizer.param_groups for v in group['params']], 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
                if step % 100 == 0:
                    print(arm, step, float(np.mean(losses[-50:])), flush=True)
            train_seconds = time.monotonic() - started
            del optimizer
        ps, strict, joint, seconds = score(encoder, head, cal)
        if arm == 'split_learned':
            previous = np.load(p['control_scores'])
            np.testing.assert_array_equal(ps, previous['probabilities'])
            np.testing.assert_array_equal(strict, previous['available'])
        np.savez(out / (arm + '.cal_scores.npz'), probabilities=ps, available_strict=strict, available_joint=joint)
        torch.save({'head': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    'last2': {k: v.detach().cpu() for k, v in encoder.encoder.layers[-2:].state_dict().items()},
                    'readout': 'joint_vowel', 'pooling': pooling, 'role_supervision': supervision, 'learned': True}, out / (arm + '.pt'))
        report['arms'][arm] = {'supervision': supervision, 'pooling': pooling, 'initial_parameter_sha256': initial_sha,
                               'same_initial_roles': True, 'same_initial_logits_within_pooling': True,
                               'same_initial_logits_across_poolings': 'not required; pooling is an explicit experimental factor',
                               'training_seconds': train_seconds, 'score_seconds': seconds,
                               'unconditional_feature_AUC': aucs(labels, ps, np.ones(len(ps), dtype=bool)),
                               'artifact_sha256': trainer.service.sha(out / (arm + '.pt'))}
        for policy, available in [('strict', strict), ('joint', joint)]:
            thresholds, audit = trainer.calibrate(labels, features, ps, available)
            for key in thresholds:
                if key.endswith('/affricate'):
                    thresholds[key] = None
            cell = arm + '/' + policy
            gain = sum(v['cal_additional_exact'] for k, v in audit.items() if k.endswith('/pair'))
            report['cells'][cell] = {'thresholds': thresholds, 'calibration': audit, 'additional_exact': gain,
                                     'available': int(available.sum()), 'all_input_features': len(ps),
                                     'all_CAL_sites': len(labels), 'conditional_AUC': aucs(labels, ps, available)}
        write(out / 'report.json', report)
        print(arm, {k: v['additional_exact'] for k, v in report['cells'].items() if k.startswith(arm + '/')}, flush=True)
        del head, encoder
        torch.cuda.empty_cache()
    choices = [(v['additional_exact'], -int(k.endswith('/joint')), -list(report['cells']).index(k), k)
               for k, v in report['cells'].items() if v['additional_exact'] >= p['criterion']['minimum_exact_CAL']]
    report['selected_for_single_request_check'] = max(choices)[-1] if choices else None
    write(out / 'report.json', report)
    write(out / 'finished.json', {'sha256': trainer.service.sha(out / 'report.json')})
    print('selected_for_single_request_check', report['selected_for_single_request_check'], flush=True)


if __name__ == '__main__':
    main()
