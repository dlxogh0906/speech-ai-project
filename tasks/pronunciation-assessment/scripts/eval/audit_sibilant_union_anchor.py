"""Paired TRAIN diagnostic of mean-of-paths versus one union-emission path.

Selected TRAIN sources intentionally enrich previously missing target support;
their frequency is not a service failure rate. TAG is read only after automatic
anchors are computed, never used to choose or repair an individual anchor.
"""
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
import time
import numpy as np
import eval_sibilant_directional as service
from eval_sibilant_typed_margin import app_from, render
from audit_sibilant_anchor_coverage import retained_samples
from mdd.canonical_time import TimePhraseBuilder
from mdd.sibilant_margin import annotate_margin
from mdd.sibilant_transition import paired_anchor
from mdd.sibilant_union_anchor import union_anchor
from mdd.sibilant_tag_support import single_phone_intervals


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
    records = service.read_jsonl(p['sources'])
    features = service.read_jsonl(p['train_prefix'] + '.features.jsonl')
    labels = service.read_jsonl(p['train_prefix'] + '.labels.jsonl')
    annotations = {v['stem']: v for v in service.read_jsonl(p['annotations'])}
    by_uid, by_feature = collections.defaultdict(list), collections.defaultdict(list)
    wanted = {v['row']['uid'] for v in records}
    for i, f in enumerate(features):
        if f['uid'] in wanted:
            by_uid[f['uid']].append((i, f))
    for label in labels:
        if label.get('feature_index') is not None:
            by_feature[label['feature_index']].append(label)
    app, captured = app_from(p), {}
    app.S['cb'] = TimePhraseBuilder(app.S['cb'])
    forward, dec = app.S['fwd'], app.S['dec']

    def capture(decoder, wavs):
        lp, lengths = forward(decoder, wavs)
        captured.update(wav=wavs[0].copy(), lp=lp[0, :lengths[0]].numpy().copy())
        return lp, lengths

    app.S['fwd'] = capture
    counts, transitions, rows, tick = collections.Counter(), collections.Counter(), [], time.monotonic()
    with (out / 'sites.jsonl').open('x') as stream:
        for n, record in enumerate(records, 1):
            row, prior = record['row'], record['prior']
            x = service.load_audio(row)[:320000]
            _, _, current = app.analyze(x, row['prompt'], '')
            wav, lp = captured['wav'], captured['lp']
            sites, _ = annotate_margin(lp, current['canon'], current['sites'], dec.d.index, dec.blank, p['pair_thresholds'])
            if current['canon'] != prior['raw']['canon'] or render(app, row, current, sites) != prior['body']:
                raise AssertionError('original API response did not reproduce: ' + row['uid'])
            indices = retained_samples(x)
            np.testing.assert_array_equal(x[indices], wav)
            digest = hashlib.sha256(wav.tobytes()).hexdigest()
            dt = len(wav) / (16000. * len(lp))
            ids, pair_ids = [dec.d.index(t) for t in current['canon']], [dec.d.index('ㅅ'), dec.d.index('ㅆ')]
            for i, f in by_uid[row['uid']]:
                if digest != f['vad_pcm_sha256']:
                    raise AssertionError('cached VAD PCM differs')
                q = f['q_runtime']
                old = paired_anchor(lp, ids, q, pair_ids, dec.blank)
                new = union_anchor(lp, ids, q, pair_ids, dec.blank)
                old_bounds = [max(0, int((old['frame']-15)*dt*16000)), min(len(wav), int((old['frame']+16)*dt*16000))]
                if old_bounds != [f['left'], f['right']] or hashlib.sha256(wav[f['left']:f['right']].tobytes()).hexdigest() != f['clip_sha256']:
                    raise AssertionError('old automatic anchor clip did not reproduce')
                value = {'uid': row['uid'], 'feature_index': i, 'q_runtime': q, 'old': old, 'new': new,
                         'following_runtime': current['canon'][q+1], 'state': 'unknown_TAG_support'}
                value['anchor_shift_frames'] = None if new is None else new['frame']-old['frame']
                counts['sites_all'] += 1
                counts['anchor_changed'] += int(new is not None and new['frame'] != old['frame'])
                # Only now attach existing labels and intervals to the diagnostic.
                ls = by_feature[i]
                intervals = single_phone_intervals(annotations[row['uid']]['tag_phones'], row['actual'])
                if len(ls) == 1 and ls[0].get('a') is not None and not ls[0]['alignment_ambiguous'] and intervals is not None and intervals[ls[0]['a']] is not None:
                    label = ls[0]
                    start, end, token = intervals[label['a']]
                    lo, hi = int(np.ceil(start*16000)), int(np.ceil(end*16000))
                    retained = int(((indices >= lo) & (indices < hi)).sum())
                    coverage = {}
                    for method, anchor in [('old', old), ('new', new)]:
                        if anchor is None:
                            coverage[method] = {'state': 'automatic_unavailable', 'samples': 0}
                            continue
                        left = max(0, int((anchor['frame']-15)*dt*16000))
                        right = min(len(wav), int((anchor['frame']+16)*dt*16000))
                        clip = indices[left:right]
                        c = int(((clip >= lo) & (clip < hi)).sum())
                        coverage[method] = {'state': 'present' if c else 'removed_by_VAD' if not retained else 'outside', 'samples': c}
                    value.update(state='TAG_support_known', expected=label['expected'], actual=token,
                                 label_kind=label['label_kind'], coverage=coverage,
                                 paired_hypothesis_spread=abs(old['hypothesis_frames'][0]-old['hypothesis_frames'][1]))
                    transitions[(coverage['old']['state'], coverage['new']['state'])] += 1
                rows.append(value)
                service.write_line(stream, value)
            counts['API_bodies_and_VAD_reproduced'] += 1
            if n % 100 == 0:
                print(n, dict(counts), 'seconds', round(time.monotonic()-tick, 1), flush=True)
    known = [v for v in rows if v['state']=='TAG_support_known']
    missing = [v for v in known if v['coverage']['old']['state']=='outside']
    present = [v for v in known if v['coverage']['old']['state']=='present']
    repaired = sum(v['coverage']['new']['state']=='present' for v in missing)
    harmed = sum(v['coverage']['new']['state']!='present' for v in present)
    # Gate declared before any diagnostic output was observed.
    passed = bool(missing and repaired/len(missing)>=.1 and present and harmed/len(present)<=.01 and repaired>harmed)
    report = {'scope': p['scope'], 'counts': dict(counts), 'transitions': {'/'.join(k):v for k,v in transitions.items()},
              'old_present': len(present), 'old_outside': len(missing), 'repaired_outside': repaired, 'harmed_present': harmed,
              'TRAIN_diagnostic_gate_passed': passed, 'seconds': time.monotonic()-tick,
              'TAG_diagnostic_only': True, 'new_model_or_service_improvement_proven': False,
              'source_selection_is_enriched_not_population_estimate': True}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'finished.json').write_text(json.dumps({'sha256': service.sha(out/'report.json')})+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
