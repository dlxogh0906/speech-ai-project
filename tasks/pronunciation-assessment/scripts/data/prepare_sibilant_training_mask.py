"""Confirm train target absence beyond the retained TAG edge uncertainty.

Only classification supervision of pair-class clips can be disabled. Retain all
audio, GT labels, auxiliary targets, batches and evaluation rows. No inference
window or model decision uses expert intervals.
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
import collections
import hashlib
import json
import shutil
import numpy as np
import eval_sibilant_directional as service
from audit_sibilant_anchor_coverage import retained_samples


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
    shutil.copy2(a.protocol, out / 'protocol.json')
    shutil.copy2(__file__, out / Path(__file__).name)
    prefix = p['train_prefix']
    data = dict(np.load(prefix + '.npz'))
    fs = service.read_jsonl(prefix + '.features.jsonl')
    records = {z['row']['uid']: z['row'] for z in service.read_jsonl(prefix + '.responses.jsonl')}
    sites = service.read_jsonl(p['support_sites'])
    candidates = [z for z in sites if z['class_y'] in (0, 1) and z['state'] in ('target_outside_clip', 'target_removed_by_VAD', 'target_after_audio_limit')]
    by_uid = collections.defaultdict(list)
    for z in candidates:
        by_uid[z['uid']].append(z)
    valid, confirmed, uncertain = np.ones(len(fs), dtype=bool), [], []
    counts = collections.Counter()
    edge = int(round(p['edge_ignore_seconds'] * 16000))
    for uid, rows in by_uid.items():
        x = service.load_audio(records[uid])[:320000]
        indices = retained_samples(x)
        digest = hashlib.sha256(x[indices].tobytes()).hexdigest()
        for z in rows:
            i = z['feature_index']
            f = fs[i]
            if digest != f['vad_pcm_sha256'] or data['y'][i] != z['class_y'] or z['C_TAG_samples'] <= 0:
                raise AssertionError('target identity or source interval differs')
            lo = int(np.ceil(z['C_TAG_onset_seconds'] * 16000))
            hi = lo + int(z['C_TAG_samples'])
            clip = indices[f['left']:f['right']]
            plausible = int(((clip >= lo - edge) & (clip < hi + edge)).sum())
            value = dict(z, plausible_C_samples_with_edge_guard=plausible)
            if plausible:
                uncertain.append(value)
            else:
                valid[i] = False
                confirmed.append(value)
                counts['disabled_class_' + str(int(data['y'][i]))] += 1
                counts[z['state']] += 1
    plan = np.load(p['batch_plan'])
    counts['masked_draws_in_fixed_plan'] = int((~valid[plan]).sum())
    counts['unique_masked_draws'] = int(len(set(plan[~valid[plan]].tolist())))
    counts['fixed_plan_all_draws'] = int(plan.size)
    np.savez(out / 'train_class_mask.npz', valid=valid)
    report = {'scope': p['scope'], 'all_features': len(fs), 'zero_overlap_candidates': len(candidates),
              'confirmed_after_boundary_guard': len(confirmed), 'kept_due_boundary_uncertainty': len(uncertain),
              'counts': dict(counts), 'confirmed': confirmed, 'uncertain': uncertain,
              'audio_or_TAG_labels_changed': False, 'evaluation_rows_changed': False,
              'mask_sha256': service.sha(out / 'train_class_mask.npz')}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('confirmed', 'uncertain')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
