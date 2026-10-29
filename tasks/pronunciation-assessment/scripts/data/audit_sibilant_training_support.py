"""Check TRAIN acoustic support against existing TAG, never inference inputs.

An existing class label is not proof the automatic clip contains that target.
Also distinguish the next actual vowel from an unambiguously aligned intended
vowel slot, without deleting or relabeling any record in this diagnostic.
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
import collections
import hashlib
import json
import shutil
import numpy as np
import eval_sibilant_directional as service
from audit_sibilant_anchor_coverage import retained_samples
from mdd.sibilant_tag_support import single_phone_intervals
from mdd.sibilant_transition import VOWELS


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
    features = service.read_jsonl(prefix + '.features.jsonl')
    labels = service.read_jsonl(prefix + '.labels.jsonl')
    records = service.read_jsonl(prefix + '.responses.jsonl')
    annotations = {r['stem']: r for r in service.read_jsonl(p['selected_annotations'])}
    by_feature, by_uid = collections.defaultdict(list), collections.defaultdict(list)
    for z in labels:
        if z.get('feature_index') is not None:
            by_feature[z['feature_index']].append(z)
    for i, f in enumerate(features):
        by_uid[f['uid']].append(i)
    counts, rows, cases = collections.Counter(), [], []
    for record in records:
        row = record['row']
        wanted = by_uid[row['uid']]
        if not wanted:
            continue
        x = service.load_audio(row)[:320000]
        indices = retained_samples(x)
        if hashlib.sha256(x[indices].tobytes()).hexdigest() != features[wanted[0]]['vad_pcm_sha256']:
            raise AssertionError('original VAD PCM differs')
        intervals = single_phone_intervals(annotations[row['uid']]['tag_phones'], row['actual'])
        mapping = service.all_optimal_mapping(row['canonical'], row['actual'])
        for i in wanted:
            f = features[i]
            value = {'uid': row['uid'], 'feature_index': i, 'q_runtime': f['q_runtime'], 'class_y': int(data['y'][i]),
                     'state': 'uncertain', 'vowel_slot_state': 'unknown'}
            ls = by_feature[i]
            if len(ls) != 1 or ls[0].get('a') is None or ls[0]['alignment_ambiguous']:
                value['state'] = 'class_or_alignment_unknown'
            elif intervals is None or intervals[ls[0]['a']] is None:
                value['state'] = 'TAG_bounds_unresolved'
            else:
                z, actual_index = ls[0], ls[0]['a']
                start, end, token = intervals[actual_index]
                clip_indices = indices[f['left']:f['right']]
                lo, hi = int(np.ceil(start * 16000)), int(np.ceil(end * 16000))
                count = int(((clip_indices >= lo) & (clip_indices < hi)).sum())
                retained = int(((indices >= lo) & (indices < hi)).sum())
                value.update(q=z['q'], expected=z['expected'], actual=z['actual'], following=z.get('following'),
                             C_clip_samples=count, C_retained_samples=retained, C_TAG_samples=hi - lo,
                             C_TAG_onset_seconds=start, clip_source_start_seconds=float(clip_indices[0] / 16000.),
                             clip_source_end_seconds=float((clip_indices[-1] + 1) / 16000.))
                value['state'] = ('target_in_clip' if count else 'target_after_audio_limit' if lo >= len(x)
                                  else 'target_removed_by_VAD' if retained == 0 else 'target_outside_clip')
                if actual_index + 1 < len(intervals) and intervals[actual_index + 1] is not None and intervals[actual_index + 1][2] in VOWELS:
                    options = mapping[z['q'] + 1] if z['q'] + 1 < len(mapping) else set()
                    if options == {actual_index + 1}:
                        slot = 'unique_intended_vowel_slot'
                    elif options == {None}:
                        slot = 'intended_vowel_deleted'
                    elif len(options) == 1:
                        slot = 'next_actual_vowel_is_another_slot'
                    else:
                        slot = 'intended_vowel_slot_ambiguous'
                    value['vowel_slot_state'] = slot
                    value['vowel_slot_options'] = sorted(options, key=lambda v: -1 if v is None else v)
            rows.append(value)
            counts[value['state']] += 1
            if value['class_y'] in (0, 1):
                counts['pair_class_all'] += 1
                counts['pair_class/' + value['state']] += 1
                counts['pair_vowel/' + value['vowel_slot_state']] += 1
                if value['state'] not in ('target_in_clip', 'class_or_alignment_unknown', 'TAG_bounds_unresolved'):
                    cases.append(value)
        counts['VAD_waveforms_exact'] += 1
        if counts['VAD_waveforms_exact'] % 1500 == 0:
            print(dict(counts), flush=True)
    if len(rows) != len(features):
        raise AssertionError('feature coverage differs')
    with (out / 'sites.jsonl').open('x') as stream:
        for value in rows:
            service.write_line(stream, value)
    report = {'scope': p['scope'], 'all_input_features': len(features), 'counts': dict(counts),
              'pair_target_absent_examples': cases, 'data_or_labels_modified': False, 'service_improvement_proven': False}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'pair_target_absent_examples'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
