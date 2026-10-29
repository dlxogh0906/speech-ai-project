"""Frozen M3 diagnostics on a frozen audit ledger, with all CAL denominators.

No parameter or threshold fitting. Human TAG is used only in explicitly named
oracle diagnostics. Fixed runtime candidates depend on the reference alone.
"""
import argparse
import collections
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'), str(Path(__file__).parent)]
import numpy as np
import eval_sibilant_directional as service
from eval_sibilant_rank_pilot import sequence_log_probs
from mdd.ctc_evidence import trellis, occurrence

TARGET_STAGE = 'decoder_expected_phone_instead_of_TAG_actual'
PAIR = {'ㅅ': 'ㅆ', 'ㅆ': 'ㅅ'}


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def finite(x):
    return float(x) if np.isfinite(x) else None


def difference(a, b):
    return finite(a - b) if np.isfinite(a) and np.isfinite(b) else None


def digest_order(z):
    return hashlib.sha256((z['uid'] + '/' + str(z['q'])).encode()).hexdigest()


def matched_controls(ledger):
    pools = collections.defaultdict(list)
    key = lambda z: (z['item'], z['q'], z['expected'], z['following'])
    for z in ledger:
        if (z['label_kind'] == 'correct' and z['view']['state'] == 'available'
                and not (z['view']['phone'] or {}).get('errorType')):
            pools[key(z)].append(z)
    matches = {}
    for z in ledger:
        if z.get('error_stage') == TARGET_STAGE:
            eligible = [x for x in pools[key(z)] if x['speaker'] != z['speaker']]
            matches[(z['uid'], z['q'])] = [(x['uid'], x['q']) for x in sorted(eligible, key=digest_order)[:3]]
    return matches


def diagnostic(lp, ids, q, actual_id, base_trellis, sample_seconds):
    candidate = list(ids)
    if actual_id is None:
        del candidate[q]
    else:
        candidate[q] = actual_id
    alt = trellis(lp, candidate)
    result = {'sum_delta': difference(alt['logsum'], base_trellis['logsum']),
              'best_path_delta': difference(alt['best'], base_trellis['best']),
              'expected_alignment': occurrence(base_trellis, q),
              'actual_alignment': occurrence(alt, q) if actual_id is not None else None}
    if result['expected_alignment'] is None or result['actual_alignment'] is None:
        return result
    exp, act = result['expected_alignment'], result['actual_alignment']
    ef, af = exp['frames'], act['frames']
    result['alignment_mean_shift_ms'] = (act['mean_frame'] - exp['mean_frame']) * sample_seconds * 1000
    for name, frames in [('expected_best_span', ef), ('candidate_best_span', af),
                         ('union_best_spans', sorted(set(ef + af)))]:
        result[name] = {'actual_peak_p': float(np.exp(lp[frames, actual_id]).max()),
                        'expected_peak_p': float(np.exp(lp[frames, ids[q]]).max()),
                        'peak_log_ratio': float(lp[frames, actual_id].max() - lp[frames, ids[q]].max()),
                        'actual_wins_any_frame': bool(np.any(lp[frames, actual_id] > lp[frames, ids[q]]))}
    lo, hi = min(ef + af), max(ef + af)
    if q + 1 < len(ids):
        nxt = occurrence(base_trellis, q + 1)
        if nxt:
            hi = max(hi, max(nxt['frames']))
    lo, hi = max(0, lo - 2), min(len(lp), hi + 3)
    result['consonant_next_token_window'] = {
        'frames': [lo, hi], 'coordinate': 'VAD-processed model grid; not physical boundaries',
        'actual_peak_p': float(np.exp(lp[lo:hi, actual_id]).max()),
        'expected_peak_p': float(np.exp(lp[lo:hi, ids[q]]).max()),
        'actual_wins_any_frame': bool(np.any(lp[lo:hi, actual_id] > lp[lo:hi, ids[q]]))}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='smoke only, incomplete denominators')
    args = ap.parse_args()
    p = json.loads(Path(args.protocol).read_text())
    for name, expected in p['sha256'].items():
        if service.sha(name) != expected:
            raise ValueError('frozen input changed: ' + name)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    (out / 'detail_log_probs').mkdir()
    dump(out / 'protocol.json', p)
    ledger = service.read_jsonl(p['ledger'])
    by_uid = collections.defaultdict(list)
    for z in ledger:
        by_uid[z['uid']].append(z)
    matches = matched_controls(ledger)
    controls = set(itertools.chain.from_iterable(matches.values()))
    targets = set(matches)
    detail = targets | controls | {(z['uid'], z['q']) for z in ledger if z.get('false_warning_stage')}
    dump(out / 'matched_controls.json', [{'target': list(k), 'controls': [list(v) for v in vv]} for k, vv in sorted(matches.items())])
    records = service.read_jsonl(p['responses'])
    records.sort(key=lambda r: (not any((r['row']['uid'], z['q']) in targets for z in by_uid[r['row']['uid']]), r['row']['uid']))
    if args.limit:
        records = records[:args.limit]
    meta = json.loads(Path(p['service_metadata']).read_text())['service_metadata']
    paths = {Path(k).name: k for k in meta['scoring_sha256']}
    config = argparse.Namespace(threads=2, scorer=paths['score_models_v2_w2vbert.joblib'],
        norms=paths['norms.json'], calib=paths['calibration.json'], flu_rule=paths['fluency_rule.json'],
        utt_pct=paths['utt_percentile.json'])
    app = service.app_module(config, p['checkpoint'])
    dec = app.S['dec']; captured = {}; original = app.S['fwd']

    def capture(decoder, wavs):
        lp, lengths = original(decoder, wavs)
        captured.update(lp=lp[0, :lengths[0]].numpy().copy(), wav=wavs[0].copy())
        return lp, lengths

    app.S['fwd'] = capture
    counts = collections.Counter(); started = time.monotonic()
    with (out / 'sites.jsonl').open('x') as fs, (out / 'utterances.jsonl').open('x') as fu:
        for n, rec in enumerate(records, 1):
            row = rec['row']; uid = row['uid']; cached = rec['existing']
            if service.sha(row['wav']) != row['wav_sha256']:
                raise ValueError('audio changed: ' + uid)
            wav = service.load_audio(row)[:320000]
            code = 'too_short' if len(wav) < 4800 else app.check_audio(wav, row['prompt'])[0]
            if code != cached['rejected']:
                raise AssertionError('input gate changed: ' + uid)
            counts[code or 'accepted'] += 1
            utterance = {'uid': uid, 'speaker': row['speaker'], 'item': row['item'], 'rejected': code}
            if code is not None:
                for z in by_uid[uid]:
                    service.write_line(fs, dict(uid=uid, q=z['q'], label_kind=z['label_kind'], expected=z['expected'],
                        actual=z.get('actual'), speaker=z['speaker'], item=z['item'], error_stage=z.get('error_stage'),
                        status='input_rejected', baseline_exact=False, baseline_warning=False))
            else:
                captured.clear()
                _, _, raw = app.analyze(wav, row['prompt'], '')
                body = service.jsonable(service.response(app, row, raw))
                if body != cached['body'] or service.jsonable(raw) != cached['raw']:
                    dump(out / 'reproduction_failure.json', {'uid': uid, 'raw': service.jsonable(raw), 'cached_raw': cached['raw'],
                                                           'body_same': body == cached['body']})
                    raise AssertionError('cached baseline did not reproduce: ' + uid)
                counts['body_and_raw_exactly_reproduced'] += 1
                lp, processed = captured['lp'], captured['wav']
                canon = raw['canon']; ids = [dec.d.index(t) for t in canon]
                actual_ids = [dec.d.index(t) for t in row['actual']]
                lab_ids = [dec.d.index(t) for t in row['canonical']]
                if dec.d.unk() in ids + actual_ids + lab_ids:
                    raise ValueError('unknown phone: ' + uid)
                sequences = []; lookup = {}

                def register(seq):
                    key = tuple(seq)
                    if key not in lookup:
                        lookup[key] = len(sequences); sequences.append(list(seq))
                    return lookup[key]

                base_i, tag_i, lab_i = [register(x) for x in (ids, actual_ids, lab_ids)]
                indices = {}; oracle_indices = {}
                for q, token in enumerate(canon):
                    if token not in PAIR:
                        continue
                    scores = {}
                    for alt in p['runtime_candidates']:
                        seq = list(ids)
                        if alt == '<delete>':
                            del seq[q]
                        else:
                            seq[q] = dec.d.index(alt)
                        scores[alt] = register(seq)
                    indices[q] = scores
                for z in by_uid[uid]:
                    q = z.get('q_runtime')
                    if q is None or q >= len(ids) or canon[q] != z['expected']:
                        continue
                    seq = list(ids)
                    if z['actual'] is None:
                        del seq[q]
                    else:
                        seq[q] = dec.d.index(z['actual'])
                    oracle = {'single_site_TAG': register(seq)}
                    a = z.get('a')
                    if a is not None and z['actual'] is not None and not z.get('alignment_ambiguous'):
                        if row['actual'][a] != z['actual']:
                            raise AssertionError('TAG site changed')
                        seq = list(actual_ids); seq[a] = dec.d.index(z['expected'])
                        oracle['TAG_background_expected'] = register(seq)
                    oracle_indices[z['q']] = oracle
                values = sequence_log_probs(lp, sequences, dec.blank)
                utterance.update(prompt=row['prompt'], canonical=canon, lab=row['canonical'], actual=row['actual'],
                    tag_minus_lab=difference(values[tag_i], values[lab_i]), tag_minus_runtime=difference(values[tag_i], values[base_i]),
                    logp_TAG=finite(values[tag_i]), logp_LAB=finite(values[lab_i]), logp_runtime=finite(values[base_i]),
                    vad_pcm_sha256=hashlib.sha256(processed.tobytes()).hexdigest(), frames=len(lp),
                    model_grid_seconds=len(processed) / 16000 / len(lp), baseline_exactly_reproduced=True)
                needs_detail = any((uid, z['q']) in detail for z in by_uid[uid])
                bt = trellis(lp, ids, dec.blank) if needs_detail else None
                if bt is not None:
                    if abs(bt['logsum'] - values[base_i]) > .02:
                        raise AssertionError('numpy vs torch CTC mismatch')
                    np.savez_compressed(out / 'detail_log_probs' / (uid + '.npz'), log_probs=lp,
                                        canonical_ids=np.array(ids), actual_ids=np.array(actual_ids),
                                        vad_pcm_sha256=np.array(utterance['vad_pcm_sha256']))
                for z in by_uid[uid]:
                    q = z.get('q_runtime'); key = (uid, z['q']); phone = z['view']['phone'] or {}
                    entry = {k: z.get(k) for k in ('uid', 'q', 'q_runtime', 'speaker', 'item', 'language', 'expected', 'actual',
                                                 'following', 'position', 'label_kind', 'error_stage')}
                    entry.update(target126=key in targets, matched_normal_control=key in controls,
                        baseline_warning=bool(phone.get('errorType')),
                        baseline_exact=z.get('error_stage') == 'exact_error_warning', baseline_pred=(z.get('raw_site') or {}).get('pred'),
                        whole_TAG_minus_LAB=utterance['tag_minus_lab'], whole_TAG_minus_runtime=utterance['tag_minus_runtime'])
                    if z['q'] not in oracle_indices or q not in indices:
                        entry['status'] = 'reference_mapping_unavailable_or_disagrees'
                    else:
                        scores = {k: difference(values[i], values[base_i]) for k, i in indices[q].items()}
                        entry.update(status='scored', runtime_candidate_deltas=scores,
                            pair_delta=scores[PAIR[z['expected']]],
                            runtime_winner=max((k for k in scores if scores[k] is not None), key=lambda k: scores[k]),
                            oracle_single_site_delta=difference(values[oracle_indices[z['q']]['single_site_TAG']], values[base_i]))
                        if 'TAG_background_expected' in oracle_indices[z['q']]:
                            entry['oracle_TAG_background_delta'] = difference(values[tag_i], values[oracle_indices[z['q']]['TAG_background_expected']])
                        if key in detail:
                            # Normals get exactly the same counterfactual error candidate
                            # as their matched miss; no actual phone used for runtime scores.
                            alt = z['actual'] if z['label_kind'] != 'correct' else PAIR[z['expected']]
                            entry['alignment_diagnostic_candidate'] = alt
                            entry['alignment_diagnostic'] = diagnostic(lp, ids, q,
                                dec.d.index(alt) if alt is not None else None, bt, utterance['model_grid_seconds'])
                            if abs(entry['alignment_diagnostic']['sum_delta'] -
                                   (scores[PAIR[z['expected']]] if z['label_kind'] == 'correct' else entry['oracle_single_site_delta'])) > .02:
                                raise AssertionError('diagnostic CTC mismatch')
                    service.write_line(fs, entry)
            service.write_line(fu, utterance)
            if n % 25 == 0 or n == len(records):
                progress = {'utterances': n, 'total': len(records), 'counts': dict(counts),
                            'seconds': time.monotonic() - started, 'smoke_limit': args.limit}
                dump(out / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    dump(out / 'finished.json', dict(progress, source_sha256={str(Path(__file__)): service.sha(__file__)},
         sites_sha256=service.sha(out / 'sites.jsonl'), utterances_sha256=service.sha(out / 'utterances.jsonl')))


if __name__ == '__main__':
    main()
