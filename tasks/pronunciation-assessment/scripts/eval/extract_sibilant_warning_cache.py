"""Measure existing runtime warnings, without TAG-based selection or alignment.

Reproduce the complete original API response before observing any warning.
Unlinked/deleted tokens remain unavailable. Labels are attached only afterwards.
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
import shutil
import time
import numpy as np
from transformers import SeamlessM4TFeatureExtractor
import eval_sibilant_directional as service
from eval_sibilant_typed_margin import app_from, render
from eval_speech_rescue import observe
from audit_sibilant_anchor_coverage import retained_samples
from mdd.canonical_time import TimePhraseBuilder
from mdd.sibilant_transition import PAIR, VOWELS
from mdd.sibilant_margin import annotate_margin
from mdd.sibilant_observed_anchor import observed_tokens, linked_anchor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / 'protocol.json')
    shutil.copy2(__file__, out / Path(__file__).name)
    app, captured = app_from(p), {}
    app.S['cb'] = TimePhraseBuilder(app.S['cb'])
    fe = SeamlessM4TFeatureExtractor.from_pretrained(p['checkpoint'])
    forward, dec = app.S['fwd'], app.S['dec']

    def capture(decoder, wavs):
        lp, lengths = forward(decoder, wavs)
        captured.update(wav=wavs[0].copy(), lp=lp[0, :lengths[0]].numpy().copy())
        return lp, lengths

    app.S['fwd'] = capture
    arrays, features, labels, warnings = [], [], [], []
    counts, tick = collections.Counter(), time.monotonic()
    with (out / 'cal.responses.jsonl').open('x') as stream:
        for number, record in enumerate(service.read_jsonl(p['sources']), 1):
            row, prior = record['row'], record['prior']
            raw, by_q = prior['raw'], {}
            phones = service.phonemes(prior['body'])
            eligible = [q for q, token in enumerate(raw['canon'][:-1])
                        if token in PAIR and raw['canon'][q + 1] in VOWELS
                        and phones[q].get('errorType')]
            if not eligible:
                raise AssertionError('prepared warning source has no runtime warning')
            x = service.load_audio(row)[:320000]
            result, _, current = app.analyze(x, row['prompt'], '')
            wav, lp = captured['wav'], captured['lp']
            sites, _ = annotate_margin(lp, current['canon'], current['sites'], dec.d.index,
                                       dec.blank, p['pair_thresholds'])
            if render(app, row, current, sites) != prior['body'] or current['canon'] != raw['canon']:
                raise AssertionError('existing complete API response did not reproduce: ' + row['uid'])
            np.testing.assert_array_equal(x[retained_samples(x)], wav)
            tokens, spans = observed_tokens(lp, dec.d, dec.blank)
            if tokens != result['pred_jamo']:
                raise AssertionError('observed greedy sequence differs from API decoder')
            # Check every linked token, not just the warnings under study.
            for site in raw['sites']:
                linked_anchor(tokens, spans, site)
            digest = hashlib.sha256(wav.tobytes()).hexdigest()
            dt = len(wav) / (16000. * len(lp))
            for q in eligible:
                warning = {'uid': row['uid'], 'q_runtime': q, 'canonical': raw['canon'][q],
                           'baseline_pred': raw['sites'][q].get('pred'), 'feature_index': None}
                anchor = linked_anchor(tokens, spans, raw['sites'][q])
                if anchor is None:
                    warning['state'] = 'unlinked_preserved'
                else:
                    left = max(0, int((anchor['frame'] - 15) * dt * 16000))
                    right = min(len(wav), int((anchor['frame'] + 16) * dt * 16000))
                    clip = wav[left:right]
                    value = fe([clip], sampling_rate=16000, return_tensors='np', padding=True)
                    length = int(value['attention_mask'][0].sum())
                    if not 3 <= length <= p['max_frames']:
                        warning['state'] = 'unsupported_length_preserved'
                    else:
                        centers = left + (np.arange(length) + .5) * len(clip) / length
                        relative = np.clip((centers / 16000. - (anchor['frame'] + .5) * dt) / .31,
                                           -2, 2).astype(np.float32)
                        by_q[q] = len(arrays)
                        warning.update(feature_index=len(arrays), state='observed')
                        arrays.append((value['input_features'][0, :length].astype(np.float32), relative))
                        features.append(dict(warning, left=left, right=right, input_frames=length,
                                             anchor=anchor, frame_sec=dt, vad_pcm_sha256=digest,
                                             clip_sha256=hashlib.sha256(clip.tobytes()).hexdigest()))
                counts[warning['state']] += 1
                warnings.append(warning)
            viewed = observe(prior, row)
            mapping = service.all_optimal_mapping(row['canonical'], raw['canon'])
            for site in row['sites']:
                options = mapping[site['q']]
                rq = next(iter(options)) if len(options) == 1 and None not in options else None
                labels.append(dict(site, expected=site['canon'], uid=row['uid'], speaker=row['speaker'],
                                   item=row['item'], language=row.get('language'), q_runtime=rq,
                                   feature_index=by_q.get(rq), existing=viewed[site['q']]))
            service.write_line(stream, record)
            counts['complete_API_bodies_reproduced'] += 1
            if number % 20 == 0:
                print(number, dict(counts), 'seconds', round(time.monotonic() - tick, 1), flush=True)
    data = {'x': np.zeros((len(arrays), p['max_frames'], 160), np.float32),
            'relative': np.zeros((len(arrays), p['max_frames']), np.float32),
            'lengths': np.array([len(v[0]) for v in arrays], np.int64)}
    for i, (x, relative) in enumerate(arrays):
        data['x'][i, :len(x)] = x
        data['relative'][i, :len(x)] = relative
    np.savez(out / 'cal.npz', **data)
    for suffix, values in [('features', features), ('labels', labels), ('warnings', warnings)]:
        with (out / ('cal.' + suffix + '.jsonl')).open('x') as stream:
            for value in values:
                service.write_line(stream, value)
    report = {'counts': dict(counts), 'features': len(features), 'all_warning_sites': len(warnings),
              'no_TAG_model_inputs': True, 'automatic_anchor_is_not_acoustic_boundary': True,
              'full_denominators_source': p['complete_cal_prefix'], 'seconds': time.monotonic() - tick,
              'sha256': {v.name: service.sha(v) for v in out.glob('cal.*')}}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')


if __name__ == '__main__':
    main()
