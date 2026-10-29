"""Stream original runtime observations into small frozen TRAIN summaries.

TAG timing is consulted only after all runtime windows and summaries are fixed.
Rejected/unknown/mismatched sites remain in the output label ledger.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import hashlib
import json
import time
import numpy as np
import torch
from transformers import SeamlessM4TFeatureExtractor
import eval_sibilant_directional as service
from eval_sibilant_typed_margin import app_from, render
from eval_speech_rescue import observe
from audit_sibilant_anchor_coverage import retained_samples
from mdd.canonical_time import TimePhraseBuilder
from mdd.sibilant_margin import annotate_margin
from mdd.sibilant_observed_anchor import observed_tokens
from mdd.sibilant_source_view import source_relative
from mdd.sibilant_vowel_reference import original_window, annotated_intended_vowel_interval
from mdd.sibilant_tag_support import single_phone_intervals
from mdd.sibilant_frame_clock import stacked_window_centers
from mdd.sibilant_vowel_presence import (SCHEMA, next_reference, training_target,
    observed_vowel_support)
from mdd.sibilant_transition import PAIR, VOWELS
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_hubert_adaptation import load_native, encode_clip, tensor_digest
from mdd.sibilant_precision_tail import PrecisionTailHead, SUMMARY_SCHEMA


def native_grid(feature, source_relative_values):
    left, right, oldn = feature['left'], feature['right'], feature['input_frames']
    first, last = (left+319)//320, (right-400)//320
    indices = np.arange(first, last+1)
    if not len(indices) or len(indices) > 32:
        raise ValueError('unsupported original native grid length')
    centers = left+(np.arange(oldn)+.5)*(right-left)/oldn
    anchors = centers/16000.-source_relative_values[:oldn]*.31
    if np.ptp(anchors) > 1e-6:
        raise ValueError('observed reference not constant')
    reference = float(np.median(anchors))
    relative = np.clip(((indices*320+199.5)/16000.-reference)/.31, -2, 2).astype(np.float32)
    result = dict(feature, source_coordinate_schema=feature['coordinate_schema'],
        source_input_frames=oldn, input_frames=len(indices), encoder_left=left//320*320,
        global_frame_start=first, local_frame_start=first-left//320,
        frame_first_source_sample=first*320+199.5,
        observed_reference_seconds=reference,
        coordinate_schema='hubert_global_conv400_hop320_observedV_v1',
        feature_schema='frozen_hubert_ko_h6_h12_fp16_aligned_short_v1')
    return result, relative


def summarize_clip(encoder, processor, model, source, feature, relative):
    device = next(model.parameters()).device
    hidden = encode_clip(encoder, processor, source, feature)
    if len(hidden) != len(relative) or len(hidden) != feature['input_frames']:
        raise ValueError('native grid length differs')
    x = torch.nn.functional.pad(hidden, (0, 0, 0, 32-len(hidden)))[None]
    mask = torch.arange(32, device=device)[None] < len(hidden)
    rel = torch.zeros((1, 32), device=device)
    rel[0, :len(hidden)] = torch.as_tensor(relative, device=device)
    summary, moments, phase = model.summarize(x, mask, rel)
    original, original_phase = model.base(x, mask, rel)
    if not torch.equal(original, model.from_summary(summary, moments)) or not torch.equal(phase, original_phase):
        raise ValueError('frozen summary not exactly original readout')
    return (summary[0].cpu().numpy(), moments[0].cpu().numpy(),
            bool(observed_vowel_support(phase, mask)[0].cpu()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for name, digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen normal extraction source changed: '+name)
    torch.set_num_threads(2)
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    annotations = {z['stem']:z for z in service.read_jsonl(p['annotations'])}
    encoder, processor, _ = load_native(p['model_manifest'], device)
    artifact = torch.load(p['control_artifact'], map_location='cpu')
    model = PrecisionTailHead(load_probe(artifact, p['binding'], device), 'ce_full').to(device).eval().requires_grad_(False)
    before = tensor_digest(model.state_dict())
    app, captured = app_from(p), {}
    app.S['cb'] = TimePhraseBuilder(app.S['cb'])
    dec, forward = app.S['dec'], app.S['fwd']
    def capture(decoder, wavs):
        lp, lengths = forward(decoder, wavs)
        captured.update(wav=wavs[0].copy(), lp=lp[0, :lengths[0]].numpy().copy())
        return lp, lengths
    app.S['fwd'] = capture
    fe = SeamlessM4TFeatureExtractor.from_pretrained(p['checkpoint'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(a.protocol).read_text())
    features, labels, summaries, moments, availability, targets = [], [], [], [], [], []
    counts, elapsed = collections.Counter(), []
    tick = time.monotonic()
    with torch.no_grad(), (out/'runtime_provenance.jsonl').open('x') as provenance:
        for n, row in enumerate(service.read_jsonl(p['rows']), 1):
            started = time.monotonic()
            x = service.load_audio(row)[:320000]
            code, _, gate = app.check_audio(x, row['prompt'])
            by_q, source_features = {}, []
            record = {'rejected':code, 'raw':None, 'body':None}
            queries = []
            if code:
                counts['input_rejected_utterances'] += 1
            else:
                result, _, raw = app.analyze(x, row['prompt'], '')
                wav, lp = captured['wav'], captured['lp']
                sites, _ = annotate_margin(lp, raw['canon'], raw['sites'], dec.d.index, dec.blank, p['pair_thresholds'])
                record.update(raw=raw, body=render(app, row, raw, sites))
                indices = retained_samples(x)
                np.testing.assert_array_equal(x[indices], wav)
                tokens, spans = observed_tokens(lp, dec.d, dec.blank)
                if tokens != result['pred_jamo']:
                    raise ValueError('observed decoder tokens differ')
                optimal = service.all_optimal_mapping(raw['canon'], tokens)
                dt = len(wav)/(16000.*len(lp))
                phones = service.phonemes(record['body'])
                for q, token in enumerate(raw['canon'][:-1]):
                    if token not in PAIR or raw['canon'][q+1] not in VOWELS:
                        continue
                    anchor, state = next_reference(tokens, spans, raw['canon'], raw['sites'], q, optimal)
                    query = {'uid':row['uid'], 'q_runtime':q, 'state':state, 'feature_index':None,
                        'C_linked':raw['sites'][q].get('j') is not None,
                        'baseline_warning':bool(phones[q].get('errorType'))}
                    if anchor is not None:
                        left, right, shift = original_window(indices, len(x), anchor['frame'], dt)
                        clip = x[left:right]
                        values = fe([clip], sampling_rate=16000, return_tensors='np', padding=True)
                        length = int(values['attention_mask'][0].sum())
                        if not 3 <= length <= p['max_frames']:
                            query['state'] = 'unsupported_length'
                        else:
                            query['feature_index'] = len(features)
                            feature = dict(query, canonical=token, baseline_pred=raw['sites'][q].get('pred'),
                                baseline_actual=phones[q].get('actual_jamo'), expected_V=raw['canon'][q+1],
                                observed_V=anchor['token'], left=left, right=right, anchor=anchor,
                                frame_sec=dt, input_frames=length, coordinate_schema=SCHEMA,
                                clip_sha256=hashlib.sha256(clip.tobytes()).hexdigest())
                            relative = source_relative(left, right, length, shift, anchor['frame'], dt)
                            grid, native_relative = native_grid(feature, relative)
                            summary, moment, available = summarize_clip(encoder, processor, model, x, grid, native_relative)
                            features.append(grid)
                            summaries.append(summary)
                            moments.append(moment)
                            availability.append(available)
                            targets.append(-100)
                            by_q[q] = query['feature_index']
                            source_features.append(feature)
                    queries.append(query)
                    counts['query/'+query['state']] += 1
                counts['accepted_utterances'] += 1
            # Runtime observations are now fixed. Only this block reads TAG.
            raw = record['raw']
            mapping = service.all_optimal_mapping(row['canonical'], raw['canon']) if raw else None
            prior_views = observe(record, row)
            by_rq = collections.defaultdict(list)
            for site in row['sites']:
                options = mapping[site['q']] if mapping is not None else set()
                rq = next(iter(options)) if len(options) == 1 and None not in options else None
                label = dict(site, expected=site['canon'], uid=row['uid'], speaker=row['speaker'],
                    item=row['item'], language=row.get('language'), q_runtime=rq,
                    feature_index=by_q.get(rq), existing=prior_views[site['q']])
                labels.append(label)
                if rq is not None:
                    by_rq[rq].append(label)
            actualmap = service.all_optimal_mapping(row['canonical'], row['actual'])
            intervals = single_phone_intervals(annotations[row['uid']]['tag_phones'], row['actual'])
            source_by_q = {f['q_runtime']:f for f in source_features}
            for q, i in by_q.items():
                if len(by_rq[q]) != 1:
                    counts['target/label_mapping_unknown'] += 1
                    continue
                z, f = by_rq[q][0], source_by_q[q]
                opts = actualmap[z['q']+1]
                v = annotated_intended_vowel_interval(intervals, opts)
                c = intervals[z['a']] if intervals is not None and z.get('a') is not None else None
                centers = stacked_window_centers(f['left'], f['right'], f['input_frames'])/16000.
                y, _, state = training_target(z, c, v, f['left'], f['right'], centers,
                    z.get('a') is not None and opts == {z['a']+1})
                targets[i] = y
                counts['target/'+state] += 1
            service.write_line(provenance, {'uid':row['uid'], 'rejected':code, 'audio_gate':gate,
                'runtime_canonical':raw['canon'] if raw else None, 'queries':queries,
                'raw_sha256':hashlib.sha256(json.dumps(service.jsonable(raw), sort_keys=True).encode()).hexdigest(),
                'API_body_sha256':hashlib.sha256(json.dumps(service.jsonable(record['body']), sort_keys=True).encode()).hexdigest()})
            counts['utterances'] += 1
            elapsed.append(time.monotonic()-started)
            if n % 100 == 0:
                print('normal summaries', n, 'features', len(features), dict(counts),
                    'seconds', round(time.monotonic()-tick, 1), flush=True)
    if tensor_digest(model.state_dict()) != before:
        raise ValueError('frozen original model changed')
    summary, moment = np.asarray(summaries, np.float32), np.asarray(moments, np.float32)
    ps = []
    with torch.no_grad():
        for start in range(0, len(summary), 64):
            ps.extend(model.from_summary(torch.as_tensor(summary[start:start+64], device=device),
                torch.as_tensor(moment[start:start+64], device=device)).softmax(-1).cpu().numpy())
    np.savez(out/'summary.npz', summary=summary, moments=moment,
        available=np.asarray(availability, bool), y=np.asarray(targets, np.int64), probabilities=np.asarray(ps))
    for name, values in [('features', features), ('labels', labels)]:
        with (out/('data.'+name+'.jsonl')).open('x') as stream:
            for value in values:
                service.write_line(stream, value)
    report = {'scope':__doc__, 'partition':'train', 'schema':SUMMARY_SCHEMA,
        'counts':dict(counts), 'features':len(features), 'all_label_sites':len(labels),
        'known':int((np.asarray(targets) >= 0).sum()), 'no_input':sum(z['feature_index'] is None for z in labels),
        'all_original_summary_logits_and_roles_exact':True, 'original_parent_unchanged':True,
        'TAG_in_runtime_input':False, 'native_frames_saved':False,
        'seconds':time.monotonic()-tick, 'full_extraction_per_utterance_ms_p50_p95':(1000*np.quantile(elapsed,[.5,.95])).tolist(),
        'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'sha256':{name:service.sha(out/name) for name in ('summary.npz', 'data.features.jsonl', 'data.labels.jsonl', 'runtime_provenance.jsonl')},
        'goal_complete':False, 'deployment':False}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (out/'finished.json').write_text(json.dumps({'report_sha256':service.sha(out/'report.json')}))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
