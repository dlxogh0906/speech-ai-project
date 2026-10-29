"""TRAIN diagnostic: C-role amount and optional-C/V evidence, no refitting.

Run inference before consulting TRAIN targets. Existing TAG support is used
only for subsequent summaries, never to position runtime paths or windows.
"""
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
import json
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import eval_sibilant_directional as service
from train_sibilant_vowel_presence import model
from train_sibilant_local_encoder import hidden
from mdd.sibilant_null_path import null_path_evidence


def quantile(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return {'n': len(x), 'p0_p25_p50_p75_p100': np.quantile(x, [0, .25, .5, .75, 1]).tolist() if len(x) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(a.protocol).read_text())
    cache = Path(p['train_cache'])
    data = dict(np.load(cache / 'data.npz'))
    if set(data) != {'x', 'relative_vowel', 'lengths'}:
        raise ValueError('typed V-reference input required')
    features = service.read_jsonl(cache / 'data.features.jsonl')
    old_uids = {z['row']['uid'] for z in service.read_jsonl(p['original_sources'])}
    torch.set_num_threads(2)
    report = {'scope': p['scope'], 'deployment': False, 'inputs': len(features), 'arms': {}}
    for name, path in p['artifacts'].items():
        encoder, head = model(p, torch.load(path, map_location='cpu'))
        encoder.eval()
        head.eval()
        saved = {k: [] for k in ['probabilities', 'phase', 'C_mean', 'C_max', 'V_max',
                                'V_at_anchor', 'anchor', 'CV_vs_V', 'bound_CV_vs_V', 'anchor_available']}
        tick = time.monotonic()
        with torch.no_grad():
            for start in range(0, len(features), p['batch']):
                x = torch.as_tensor(data['x'][start:start + p['batch']], device='cuda')
                length = torch.as_tensor(data['lengths'][start:start + p['batch']], device='cuda')
                mask = torch.arange(x.shape[1], device='cuda')[None, :] < length[:, None]
                rel = torch.as_tensor(data['relative_vowel'][start:start + p['batch']], device='cuda')
                logits, phase = head(hidden(encoder, x, mask), mask, rel)
                role = phase.softmax(-1) * mask[..., None]
                anchor = rel.abs().masked_fill(~mask, float('inf')).argmin(-1)
                valid = anchor > 0
                safe_anchor = torch.where(valid, anchor, torch.full_like(anchor, -1))
                anchored = null_path_evidence(phase, mask, safe_anchor)
                anchored = anchored.masked_fill(~valid, float('nan'))
                rows = torch.arange(len(x), device=x.device)
                values = {'probabilities': logits.softmax(-1), 'phase': role,
                          'C_mean': role[..., 1].sum(-1) / length,
                          'C_max': role[..., 1].max(-1).values,
                          'V_max': role[..., 2:].sum(-1).max(-1).values,
                          'V_at_anchor': role[rows, anchor, 2:].sum(-1),
                          'anchor': anchor, 'CV_vs_V': null_path_evidence(phase, mask),
                          'bound_CV_vs_V': anchored, 'anchor_available': valid}
                for k, value in values.items():
                    saved[k].extend(value.cpu().numpy())
        result = {k: np.asarray(v) for k, v in saved.items()}
        np.savez(out / (name + '.npz'), **result)
        # All acoustic/path inference above is finished before TAG is consulted.
        target = dict(np.load(cache / 'targets.npz'))
        y = target['y']
        normal = np.isin(y, [0, 1])
        present = (y >= 0) & (y < 4)
        absent = y == 4
        winner = result['probabilities'].argmax(-1)
        subset = np.array([f['uid'] not in old_uids for f in features])
        groups = {'all_supported_C': present, 'supported_S_SS': normal, 'TAG_deleted': absent,
                  'unknown': y < 0, 'new_to_300update_S_SS': normal & subset,
                  'new_to_300update_TAG_deleted': absent & subset}
        summaries = {}
        for group, take in groups.items():
            summaries[group] = {'inputs': int(take.sum()),
                'S_SS_winners': int((take & np.isin(winner, [0, 1])).sum()),
                'S_SS_winners_Cmax_below_half': int((take & np.isin(winner, [0, 1]) & (result['C_max'] < .5)).sum()),
                'metrics': {k: quantile(result[k][take]) for k in ['C_mean', 'C_max', 'V_at_anchor', 'CV_vs_V', 'bound_CV_vs_V']}}
        auc = {}
        eligible = present | absent
        for key in ['C_mean', 'C_max', 'CV_vs_V', 'bound_CV_vs_V']:
            take = eligible & np.isfinite(result[key])
            auc[key] = float(roc_auc_score(present[take], result[key][take])) if len(np.unique(present[take])) == 2 else None
        true_c = target['phase'] == 1
        c_known = true_c.sum(-1) > 0
        target_mass = (result['phase'][..., 1] * true_c).sum(-1) / np.maximum(true_c.sum(-1), 1)
        role_diag = {'known_C_frame_inputs': int(c_known.sum()),
                     'C_probability_at_TAG_C': quantile(target_mass[c_known]),
                     'TAG_deleted_known_non_C_frames': int(((target['phase'] >= 0) & absent[:, None]).sum()),
                     'max_C_on_TAG_deleted': quantile(result['C_max'][absent])}
        report['arms'][name] = {'seconds': time.monotonic() - tick, 'groups': summaries,
                               'present_vs_TAG_deleted_auc_TRAIN': auc, 'role_diagnostic': role_diag,
                               'artifact_sha256': service.sha(path), 'output_sha256': service.sha(out / (name + '.npz'))}
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(name, json.dumps({'auc': auc, 'TAG_deleted': summaries['TAG_deleted'],
                               'S_SS': summaries['supported_S_SS']}), flush=True)
        del encoder, head
        torch.cuda.empty_cache()
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')


if __name__ == '__main__':
    main()
