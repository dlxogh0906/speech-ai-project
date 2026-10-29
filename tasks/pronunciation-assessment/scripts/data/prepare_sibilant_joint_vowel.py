"""Train-only audit of the artificial vowel-midpoint exclusion zone.

Reuse selected original TAG annotations and the verified analysis clock. Remove
only the midpoint convention; keep uncertainty at every actual C/V prefix edge.
Acoustic inputs, actual-class labels and every evaluation record are unchanged.
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
from mdd.sibilant_tag_support import single_phone_intervals, coarse_roles
from mdd.sibilant_frame_clock import stacked_window_centers
from mdd.sibilant_joint_vowel import coarse_cv_roles
from mdd.sibilant_transition import VOWELS
from mdd.canonical_fix import CONSONANTS


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
    labels = service.read_jsonl(prefix + '.labels.jsonl')
    features = service.read_jsonl(prefix + '.features.jsonl')
    records = service.read_jsonl(prefix + '.responses.jsonl')
    annotations = {r['stem']: r for r in service.read_jsonl(p['selected_annotations'])}
    by_uid = collections.defaultdict(list)
    for z in labels:
        by_uid[z['uid']].append(z)
    cv_phase = data['phase'].copy()
    cv_phase[cv_phase == 3] = 2
    counts, groups, seen = collections.Counter(), collections.defaultdict(collections.Counter), set()
    for record in records:
        row = record['row']
        intervals = single_phone_intervals(annotations[row['uid']]['tag_phones'], row['actual'])
        if intervals is None:
            counts['unresolved_TAG_utterances'] += 1
            continue
        needed = [z for z in by_uid[row['uid']] if z.get('feature_index') is not None and not z['alignment_ambiguous']
                  and z.get('a') is not None and z['actual'] in CONSONANTS]
        if not needed:
            continue
        x = service.load_audio(row)[:320000]
        indices = retained_samples(x)
        digest = hashlib.sha256(x[indices].tobytes()).hexdigest()
        for z in needed:
            i = z['feature_index']
            if i in seen:
                raise AssertionError('duplicate target index')
            seen.add(i)
            f, length = features[i], int(data['lengths'][i])
            if digest != f['vad_pcm_sha256']:
                raise AssertionError('original VAD PCM differs')
            centers = stacked_window_centers(f['left'], f['right'], length)
            times = indices[centers.astype(int)] / 16000.
            old = coarse_roles(times, intervals, z['a'], VOWELS, p['vowel_prefix_seconds'], p['edge_ignore_seconds'])
            np.testing.assert_array_equal(old, data['phase'][i, :length])
            new = coarse_cv_roles(times, intervals, z['a'], VOWELS, p['vowel_prefix_seconds'], p['edge_ignore_seconds'])
            collapsed = old.copy()
            collapsed[collapsed == 3] = 2
            np.testing.assert_array_equal(new[old != -100], collapsed[old != -100])
            cv_phase[i, :length] = new
            change = {'sites': 1, 'old_vowel_frames': int(((old == 2) | (old == 3)).sum()),
                      'recovered_vowel_frames': int(((old == -100) & (new == 2)).sum()),
                      'old_both_CV_known': int((old == 1).any() and ((old == 2) | (old == 3)).any()),
                      'new_both_CV_known': int((new == 1).any() and (new == 2).any()),
                      'newly_any_vowel_known': int(not ((old == 2) | (old == 3)).any() and (new == 2).any())}
            counts.update(change)
            groups[str(z.get('following'))].update(change)
        counts['VAD_waveforms_exact'] += 1
        if counts['VAD_waveforms_exact'] % 1500 == 0:
            print(dict(counts), flush=True)
    np.savez(out / 'train_cv_phase.npz', phase=cv_phase)
    report = {'scope': p['scope'], 'counts': dict(counts), 'by_following': dict(groups),
              'preserved_old_known_family_labels': True, 'all_input_features': len(features),
              'acoustic_or_class_inputs_changed': False, 'service_improvement_proven': False,
              'phase_sha256': service.sha(out / 'train_cv_phase.npz')}
    report['midpoint_hypothesis_supported'] = bool(counts['recovered_vowel_frames'] >= p['criterion']['minimum_recovered_fraction'] * counts['old_vowel_frames']
                                                 and counts['newly_any_vowel_known'] >= p['criterion']['minimum_newly_supported_sites'])
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
