"""Same1500 batches/model, removing only proven unsupported class supervision.

Reuse corrected-clock learned-pooling control. Actual labels and all acoustic,
auxiliary and evaluation inputs remain unchanged. GT validity is TRAIN-only.
"""
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
import shutil
import time
import numpy as np
import torch
import train_sibilant_cv_support as trainer
from train_sibilant_c_trajectory import aucs
from train_sibilant_local_encoder import load_encoder, hidden, write
from mdd.sibilant_cv_support import SupportReadout, phase_loss
from mdd.sibilant_training_mask import masked_actual_loss
from eval_speech_rescue import observe


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
    if set(cal) != {'x', 'lengths', 'relative'}:
        raise ValueError('CAL inference contains labels')
    valid = np.load(p['class_mask'])['valid']
    if valid.dtype != bool or valid.shape != train['y'].shape:
        raise ValueError('train-only validity mask shape differs')
    plan = np.load(p['control_batch_plan'])
    if len(plan) != p['steps']:
        raise ValueError('fixed step count differs')
    np.save(out / 'batch_plan.npy', plan)
    control_fit = json.loads(Path(p['control_fit']).read_text())['arms']['learned_support']
    encoder = load_encoder(p, True)
    torch.manual_seed(p['seed'])
    head = SupportReadout(encoder.config.hidden_size, True).to(next(encoder.parameters()).device)
    initial = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in head.parameters())).hexdigest()
    if initial != control_fit['initial_head_sha256']:
        raise AssertionError('initial head differs from unchanged control')
    optimizer = torch.optim.AdamW([{'params': head.parameters(), 'lr': .001},
                                  {'params': [v for v in encoder.parameters() if v.requires_grad], 'lr': .00001}], weight_decay=.01)
    losses, started, device = [], time.monotonic(), next(encoder.parameters()).device
    torch.cuda.reset_peak_memory_stats()
    for step, indices in enumerate(plan, 1):
        optimizer.zero_grad(set_to_none=True)
        x = torch.as_tensor(train['x'][indices], device=device)
        lengths = torch.as_tensor(train['lengths'][indices], device=device)
        mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
        logits, phase = head(hidden(encoder, x, mask), mask, torch.as_tensor(train['relative'][indices], device=device))
        actual = masked_actual_loss(logits, torch.as_tensor(train['y'][indices], device=device),
                                    torch.as_tensor(valid[indices], device=device))
        auxiliary = phase_loss(phase, torch.as_tensor(train['phase'][indices], device=device))
        loss = actual + p['phase_weight'] * auxiliary
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        loss.backward()
        if step == 1 and any(v.grad is not None for layer in encoder.encoder.layers[:-2] for v in layer.parameters()):
            raise AssertionError('earlier encoder gradient')
        torch.nn.utils.clip_grad_norm_([v for group in optimizer.param_groups for v in group['params']], 1.)
        optimizer.step()
        losses.append(float(actual.detach()))
        if step % 150 == 0:
            print('validity_mask', step, 'class_loss', float(np.mean(losses[-50:])), flush=True)
    train_seconds = time.monotonic() - started
    ps, av, seconds = trainer.score(encoder, head, cal)
    np.savez(out / 'masked.cal_scores.npz', probabilities=ps, available=av)
    torch.save({'head': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                'last2': {k: v.detach().cpu() for k, v in encoder.encoder.layers[-2:].state_dict().items()},
                'readout': 'existing', 'learned': True, 'training_class_mask_sha256': trainer.service.sha(p['class_mask'])}, out / 'masked.pt')
    del optimizer, encoder, head
    torch.cuda.empty_cache()
    labels = trainer.service.read_jsonl(p['cal_prefix'] + '.labels.jsonl')
    features = trainer.service.read_jsonl(p['cal_prefix'] + '.features.jsonl')
    records = trainer.service.read_jsonl(p['cal_prefix'] + '.responses.jsonl')
    prior = {z['row']['uid']: observe(z['prior'], z['row']) for z in records}
    labels = [dict(z, existing=prior[z['uid']][z['q']]) for z in labels]
    # The shared calibrator's `existing` observation must describe the actual
    # union baseline. Preserve reference/original views in full responses.
    for suffix in ('.npz', '.features.jsonl', '.responses.jsonl'):
        (out / ('cal_prior' + suffix)).symlink_to(Path(p['cal_prefix'] + suffix).resolve())
    with (out / 'cal_prior.labels.jsonl').open('x') as stream:
        for z in labels:
            trainer.service.write_line(stream, z)
    report = {'scope': p['scope'], 'inference_batch': 64, 'test_scored': False,
              'initial_head_sha256': initial, 'same_batch_plan': True, 'same_acoustic_and_GT_labels': True,
              'classification_draws_disabled': int((~valid[plan]).sum()), 'all_draws': int(plan.size),
              'training_seconds': train_seconds, 'score_seconds': seconds, 'arms': {}}
    report['calibration_base'] = 'prior union baseline; label actuals/positions and inference arrays unchanged'
    control = np.load(p['control_scores'])
    for arm, values, available in [('control', control['probabilities'], control['available_strict']), ('masked', ps, av)]:
        thresholds, audit = trainer.calibrate(labels, features, values, available)
        for key in thresholds:
            if key.endswith('/affricate'):
                thresholds[key] = None
        report['arms'][arm] = {'thresholds': thresholds, 'calibration': audit,
                               'additional_exact': sum(v['cal_additional_exact'] for k, v in audit.items() if k.endswith('/pair')),
                               'available': int(available.sum()), 'all_input_features': len(values), 'all_CAL_sites': len(labels),
                               'unconditional_AUC': aucs(labels, values, np.ones(len(values), dtype=bool))}
    report['selected_for_single_request_preflight'] = report['arms']['masked']['additional_exact'] >= 1
    write(out / 'report.json', report)
    write(out / 'finished.json', {'sha256': trainer.service.sha(out / 'report.json')})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
