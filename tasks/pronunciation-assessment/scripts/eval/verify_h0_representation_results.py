"""Independently recount frozen P1 budgets and restore all saved readouts."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/train')]
import argparse
import csv
import hashlib
import json
import math
import numpy as np
import torch
from train_h0_backbone_control import file_sha
from diagnose_h0_representations import write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    root = Path(ap.parse_args().root)
    out = root / 'probes'
    p = json.loads((root / 'protocol.json').read_text())
    fit = json.loads((out / 'fit_protocol.json').read_text())
    classes = p['probe']['classes']; ids = {c: i for i, c in enumerate(classes)}
    rows = [json.loads(line) for line in (root / 'sites.jsonl').open()]
    e = np.array([ids[s['expected']] for s in rows])
    t = np.array([ids[s['actual'] or '<DEL>'] for s in rows])
    normal = e == t
    cal = np.array([s['split'] == 'valid' and s['role'] == 'calibration' for s in rows])
    selections = {sp: np.array([s['split'] == sp and s['role'] == 'selection' for s in rows])
                  for sp in ('valid', 'valid_unseen')}
    summary = list(csv.DictReader((out / 'summary.tsv').open(), delimiter='\t'))
    train = np.load(out / 'train_ids.npy'); draws = np.load(out / 'batch_draws.npy')
    assert hashlib.sha256(draws.tobytes()).hexdigest() == fit['draw_sha256']
    assert len(train) == fit['shared_training_sites']
    assert all(rows[i]['role'] == 'train' for i in train)
    assert np.array_equal(np.load(out / 'labels.npy'), t)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    results = []; pcm = set()
    for kind in ('w2vbert', 'xlsr300m'):
        for step in (2000, 4000):
            condition = '%s_%d' % (kind, step)
            receipt = json.loads((root / condition / 'finished.json').read_text())
            assert receipt['protocol_sha256'] == file_sha(root / 'protocol.json')
            assert receipt['validation_projection_matches'] == 14009
            assert receipt['encoder_updates'] == 0 and not receipt['smoke']
            pcm.add(receipt['audio_order_and_pcm_sha256'])
            available = np.load(root / condition / 'available.npy')
            for layer in (6, 12, 24):
                for wi, window in enumerate(('auto', 'multi', 'tag')):
                    key = '%s_L%d_%s' % (condition, layer, window)
                    dest = out / key
                    done = json.loads((dest / 'finished.json').read_text())
                    assert done['updates'] == p['probe']['updates']
                    assert done['encoder_updates'] == 0
                    state = torch.load(str(dest / 'probe.pt'), map_location='cpu')
                    assert state['fit_protocol_sha256'] == file_sha(out / 'fit_protocol.json')
                    logits = np.load(dest / 'logits.npy', mmap_mode='r')
                    pred = logits.argmax(1)
                    score = (logits.max(1) - logits[np.arange(len(rows)), e]).astype(np.float64)
                    calibration = json.loads((dest / 'calibration.json').read_text())
                    tau = -np.inf if calibration['negative_infinity'] else calibration['threshold']
                    alarm = (pred != e) & available[:, wi] & (score >= tau)
                    cal_fp = int((alarm & normal & cal).sum())
                    assert cal_fp == calibration['normal_fp']
                    assert cal_fp <= math.floor(.01 * int((normal & cal).sum()))
                    for sp, mask in selections.items():
                        observed = [r for r in summary if r['condition'] == condition and
                                    int(r['layer']) == layer and r['window'] == window and
                                    r['split'] == sp and r['group'] == 'all' and r['policy'] == 'calibrated']
                        assert len(observed) == 1
                        report = observed[0]
                        assert int(report['normal_fp']) == int((mask & normal & alarm).sum())
                        assert int(report['exact']) == int((mask & ~normal & alarm & (pred == t)).sum())
                        assert int(report['wrong_actual']) == int((mask & ~normal & alarm & (pred != t)).sum())
                    mm = np.load(root / condition / ('L%d_%s.npy' % (layer, window)), mmap_mode='r')
                    probe = torch.nn.Linear(1024, len(classes)).cuda()
                    probe.load_state_dict({k: state[k].cuda() for k in ('weight', 'bias')})
                    probe.eval()
                    # Same 4096-row inference shape as the saved execution, away from TRAIN's first rows.
                    lo = 65536; hi = lo + 4096
                    x = torch.from_numpy((mm[lo:hi].astype(np.float32) - state['mean']) / state['std']).cuda()
                    with torch.inference_mode():
                        restored = probe(x).cpu().numpy()
                    delta = float(np.max(abs(restored - logits[lo:hi])))
                    assert delta == 0., (key, delta)
                    results.append(dict(probe=key, cal_normal=int((normal & cal).sum()),
                                        cal_fp=cal_fp, restored_rows=hi-lo, maximum_logit_difference=delta))
    assert len(pcm) == 1 and len(results) == 36
    write_json(root / 'results_verification.json', dict(status='passed', probes=results,
        shared_audio_order_and_pcm_sha256=next(iter(pcm)),
        calibration_recount=True, selection_recount=True, all_saved_probes_restored=True,
        source_sha256=file_sha(__file__), protocol_sha256=file_sha(root/'protocol.json')))
    print('Verified 36 probe restores, CAL budgets, and all-site seen/unseen counts.')


if __name__ == '__main__':
    main()
