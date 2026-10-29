"""Post-extraction TAG diagnostic; never supplies boundaries to inference."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src')]
import argparse
import collections
import hashlib
import json
import numpy as np
import eval_sibilant_directional as service
from extract_sibilant_localization_diagnostic import collect
from audit_sibilant_anchor_coverage import retained_samples
from mdd.sibilant_tag_support import single_phone_intervals
from mdd.sibilant_transition import VOWELS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', required=True)
    ap.add_argument('--annotations', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    root, out = Path(a.features), Path(a.out)
    records = service.read_jsonl(root / 'cal.responses.jsonl')
    features = service.read_jsonl(root / 'cal.features.jsonl')
    labels = service.read_jsonl(root / 'cal.labels.jsonl')
    wanted = {r['row']['uid'] for r in records}
    annotations = collect(a.annotations, wanted)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'selected_annotations.jsonl').open('x') as stream:
        for uid in sorted(annotations):
            service.write_line(stream, annotations[uid])
    by_feature, by_uid = collections.defaultdict(list), collections.defaultdict(list)
    for label in labels:
        if label['feature_index'] is not None:
            by_feature[label['feature_index']].append(label)
    for i, f in enumerate(features):
        by_uid[f['uid']].append(i)
    rows, counts = [], collections.Counter()
    for record in records:
        row = record['row']
        wanted_indices = by_uid[row['uid']]
        if not wanted_indices:
            continue
        x = service.load_audio(row)[:320000]
        indices = retained_samples(x)
        if hashlib.sha256(x[indices].tobytes()).hexdigest() != features[wanted_indices[0]]['vad_pcm_sha256']:
            raise AssertionError('original VAD PCM differs')
        intervals = single_phone_intervals(annotations[row['uid']]['tag_phones'], row['actual'])
        mapping = service.all_optimal_mapping(row['canonical'], row['actual'])
        for i in wanted_indices:
            f, ls = features[i], by_feature[i]
            value = {'uid': row['uid'], 'q_runtime': f['q_runtime'], 'feature_index': i,
                     'state': 'label_or_alignment_unknown'}
            if len(ls) == 1 and ls[0].get('a') is not None and not ls[0].get('alignment_ambiguous'):
                label = ls[0]
                ai = label['a']
                if intervals is None or intervals[ai] is None:
                    value['state'] = 'TAG_bounds_unresolved'
                else:
                    start, end, token = intervals[ai]
                    target_lo, target_hi = int(np.ceil(start * 16000)), int(np.ceil(end * 16000))
                    clip = indices[f['left']:f['right']]
                    overlap = int(((clip >= target_lo) & (clip < target_hi)).sum())
                    retained = int(((indices >= target_lo) & (indices < target_hi)).sum())
                    ac = min(len(indices) - 1, int((f['anchor']['frame'] + .5) * f['frame_sec'] * 16000))
                    anchor_sample = int(indices[ac])
                    value.update(state='target_in_clip' if overlap else 'target_removed_by_VAD' if not retained else 'target_outside_clip',
                                 actual=token, expected=label['expected'], following=label.get('following'),
                                 C_clip_samples=overlap, C_retained_samples=retained, C_TAG_samples=target_hi-target_lo,
                                 clip_source_start_seconds=int(clip[0])/16000.,
                                 clip_source_end_seconds=int(clip[-1]+1)/16000.,
                                 anchor_in_TAG_C=target_lo <= anchor_sample < target_hi,
                                 anchor_source_seconds=anchor_sample / 16000., TAG_C_seconds=[start, end])
                    value['next_actual_vowel_slot'] = 'unavailable'
                    if ai + 1 < len(intervals) and intervals[ai + 1] is not None and intervals[ai + 1][2] in VOWELS:
                        vstart, vend, vt = intervals[ai + 1]
                        opts = mapping[label['q'] + 1] if label['q'] + 1 < len(mapping) else set()
                        value['next_actual_vowel_slot'] = 'unique_intended' if opts == {ai+1} else 'different_or_ambiguous'
                        value.update(V_clip_samples=int(((clip >= np.ceil(vstart*16000)) & (clip < np.ceil(vend*16000))).sum()),
                                     anchor_in_TAG_V=vstart <= anchor_sample/16000. < vend)
            rows.append(value)
            counts[value['state']] += 1
    with (out / 'sites.jsonl').open('x') as stream:
        for value in rows:
            service.write_line(stream, value)
    report = {'features_all': len(features), 'audited': len(rows), 'counts': dict(counts),
              'TAG_is_existing_annotation_not_new_physical_truth': True, 'runtime_inputs_modified': False,
              'annotation_source_sha256': service.sha(a.annotations),
              'selected_annotation_sha256': service.sha(out / 'selected_annotations.jsonl'),
              'anchor_in_TAG_C': sum(r.get('anchor_in_TAG_C', False) for r in rows),
              'anchor_in_TAG_V': sum(r.get('anchor_in_TAG_V', False) for r in rows),
              'both_CV_present': sum(r.get('C_clip_samples', 0)>0 and r.get('V_clip_samples', 0)>0 for r in rows)}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')})+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
