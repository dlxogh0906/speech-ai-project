"""TRAIN-only coarse C/V labels at raw observation centers, with guarded edges."""
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
import json
import time
import numpy as np
import eval_sibilant_directional as service
from mdd.sibilant_tag_support import single_phone_intervals
from mdd.sibilant_vowel_reference import annotated_intended_vowel_interval
from mdd.sibilant_vowel_presence import training_target
from mdd.sibilant_raw_cv import ROLE_SCHEMA


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen target source changed: '+path)
    cache, raw = Path(p['train_cache']), Path(p['raw_cache'])
    if any(json.loads((folder/'report.json').read_text())['partition'] != 'train' for folder in (cache, raw)):
        raise ValueError('TAG target preparation requires TRAIN partition')
    features = service.read_jsonl(raw/'features.jsonl')
    labels = service.read_jsonl(cache/'data.labels.jsonl')
    indexed = {z['feature_index']: z for z in labels if z['feature_index'] is not None}
    if len(indexed) != len(features) or len(indexed) != sum(z['feature_index'] is not None for z in labels):
        raise ValueError('one source TAG label per runtime input required')
    rows = {z['row']['uid']: z['row'] for z in service.read_jsonl(cache/'data.responses.jsonl')}
    annotations = {z['stem']: z for z in service.read_jsonl(p['annotations'])}
    original_y = np.load(p['original_targets'])['y']
    data = np.load(raw/'data.npz')
    lengths = data['lengths']
    width = data['relative_vowel'].shape[1]
    phase = np.full((len(features), width), -100, np.int64)
    y = np.full(len(features), -100, np.int64)
    geometry, states, cases = {}, collections.Counter(), []
    tick = time.monotonic()
    for i, f in enumerate(features):
        lab, uid = indexed[i], f['uid']
        if uid != lab['uid'] or f['feature_index'] != i:
            raise ValueError('runtime/TAG site order differs')
        if uid not in geometry:
            row = rows[uid]
            geometry[uid] = (single_phone_intervals(annotations[uid]['tag_phones'], row['actual']),
                             service.all_optimal_mapping(row['canonical'], row['actual']))
        intervals, mapping = geometry[uid]
        opts = mapping[lab['q']+1] if lab['q']+1 < len(mapping) else set()
        v = annotated_intended_vowel_interval(intervals, opts)
        c = intervals[lab['a']] if intervals is not None and lab['a'] is not None else None
        n = int(lengths[i])
        if f['acoustic_frame_first_source_sample'] != int(data['first_frame'][i])*32+79.5:
            raise ValueError('raw 10ms receptive-field center differs')
        centers = (f['acoustic_frame_first_source_sample']+np.arange(n)*32)/16000
        actual, target, state = training_target(lab, c, v, f['left'], f['right'], centers,
            lab['a'] is not None and opts == {lab['a']+1}, edge=.025)
        y[i], phase[i, :n] = actual, target
        states[state] += 1
        cases.append({'feature_index': i, 'uid': uid, 'q': lab['q'], 'state': state,
            'label_kind': lab['label_kind'], 'actual_class': int(actual),
            'known_C_frames': int((target == 1).sum()), 'known_V_frames': int((target == 2).sum()),
            'unknown_frames': int((target < 0).sum()), 'input_frames': n})
        if (i+1) % 1000 == 0:
            print(json.dumps({'targets': i+1, 'total': len(features), 'seconds': time.monotonic()-tick}), flush=True)
    np.testing.assert_array_equal(y, original_y)
    paired = (phase == 1).any(-1) & (phase == 2).any(-1)
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    np.savez(out/'targets.npz', y=y, phase=phase)
    with (out/'cases.jsonl').open('w') as f:
        for row in cases:
            service.write_line(f, row)
    report = {'scope': p['scope'], 'partition': 'train', 'role_schema': ROLE_SCHEMA,
        'all_retained_GT_sites': len(labels), 'all_inputs': len(features),
        'no_input_sites': sum(z['feature_index'] is None for z in labels),
        'class_y_exact_original': True, 'known_classes': int((y >= 0).sum()), 'unknown_classes': int((y < 0).sum()),
        'states': dict(states), 'phase_counts': {str(k): int((phase == k).sum()) for k in (-100, 0, 1, 2)},
        'paired_CV_sites_for_locality': int(paired.sum()),
        'known_class_without_C_target': int(((y >= 0) & ~(phase == 1).any(-1)).sum()),
        'known_class_without_V_target': int(((y >= 0) & ~(phase == 2).any(-1)).sum()),
        'edge_guard_seconds': .025, 'seconds': time.monotonic()-tick, 'deployment': False}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out/'finished.json').write_text(json.dumps({'report_sha256': service.sha(out/'report.json')}))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
