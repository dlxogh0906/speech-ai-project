"""Frozen H0 layer/window diagnostics; canonical inference and TAG oracle separated.

prepare freezes TRAIN IDs and all evaluation denominators before extraction.
extract never optimizes an encoder. Features are 1024-D for every condition.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import gc
import gzip
import hashlib
import heapq
import itertools
import json
import math
import os
import shutil
import time
import numpy as np
import torch
from train_h0_backbone_control import Backbone, file_sha
from train_w2vbert_ctc import read_split, read_wav, load_vocab
from audit_h0_backbone_protocol import calibration_role
from eval_actual_ctc import project_on_canon
from mdd.h0_phone_audit import optimal_mapping, surface_position

LAYERS = (6, 12, 24)
WINDOWS = ('auto', 'multi', 'tag')


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.%d.tmp' % os.getpid())
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def timing_windows(canon, actual, meta):
    """Unique TAG token correspondence only; no forced-aligned TAG proxy.

    A deletion can use an explicitly annotated '-' TAG interval only when it
    uniquely matches the corresponding LAB interval. Missing evidence stays null.
    Multi-token intervals do not establish individual phone boundaries.
    """
    tokens, spans = [], []
    skipped = {'', '.', '-', 'spn', '<unk>'}
    for start, end, value in meta['tag_phones']:
        chars = [] if value.strip() in skipped else [c for c in value if '\u3131' <= c <= '\u3163']
        tokens.extend(chars)
        spans.extend([(start, end) if len(chars) == 1 else None] * len(chars))
    if tokens != actual: raise ValueError('TAG normalization differs from CTC target')
    lab = [(c, (start, end) if len([c for c in value if '\u3131' <= c <= '\u3163']) == 1 else None)
           for start, end, value in meta['lab_phones'] for c in value if '\u3131' <= c <= '\u3163']
    if [c for c, _ in lab] != canon: raise ValueError('LAB normalization differs')
    mapping = optimal_mapping(tuple(canon), tuple(actual))
    windows, reasons = [], []
    for q, choices in enumerate(mapping):
        span = None; reason = 'ambiguous_token_mapping'
        if len(choices) == 1:
            idx = next(iter(choices))
            if idx is not None:
                span = spans[idx]; reason = 'tag_phone' if span is not None else 'multi_token_interval'
            else:
                reason = 'deletion_without_explicit_interval'
                base = lab[q][1]
                matches = [(a, b) for a, b, t in meta['tag_phones']
                           if t.strip() == '-' and base is not None and a is not None and b is not None
                           and base[0] is not None and base[1] is not None
                           and abs(a - base[0]) <= .002 and abs(b - base[1]) <= .002]
                if len(matches) == 1:
                    span = matches[0]; reason = 'explicit_tag_deletion_interval'
        if span is not None and (span[0] is None or span[1] is None or
                                 not np.isfinite(span).all() or span[0] < 0 or span[1] <= span[0]):
            span = None; reason = 'invalid_interval'
        windows.append(span); reasons.append(reason)
    return windows, reasons


def prepare(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    parent = json.loads(Path(a.parent_protocol).read_text())
    manifest = Path(parent['stage2']['manifest'])
    bindings = json.loads(Path(parent['input_audit']).read_text())['sha256']
    for name in ('train.tsv', 'train.ltr', 'dict.ltr.txt'):
        p = manifest / name
        if file_sha(p) != bindings[str(p)]: raise ValueError('changed input ' + str(p))
    # Ten clean and ten erroneous utterances per foreign training prompt, chosen
    # by UID hash. Duplicated training rows are not extra unique examples.
    heaps = collections.defaultdict(list); seen = set(); all_train_speakers = set(); train_items = set()
    with (manifest / 'train.tsv').open() as ft, (manifest / 'train.ltr').open() as fl, (manifest / 'train.source').open() as fs:
        root = Path(next(ft).strip())
        for row, (line, label, source) in enumerate(zip(ft, fl, fs)):
            if source.strip() not in ('foreign_clean', 'foreign_mis'): continue
            rel, frames = line.rstrip().split('\t'); uid = Path(rel).stem
            bits = uid.split('-'); item = bits[5]; speaker = bits[0]
            all_train_speakers.add(speaker); train_items.add(item)
            if uid in seen: continue
            seen.add(uid)
            rank = int(hashlib.sha256(('P1-20261027:' + uid).encode()).hexdigest(), 16)
            entry = (-rank, uid, dict(path=str(root / rel), frames=int(frames), row=row,
                     canonical=None, actual=label.split(), split='train', role='train'))
            h = heaps[item, source.strip()]
            if len(h) < 10: heapq.heappush(h, entry)
            elif entry[0] > h[0][0]: heapq.heapreplace(h, entry)
    selected = {uid: r for heap in heaps.values() for _, uid, r in heap}
    train_records = {}
    with gzip.open(a.cache, 'rt', encoding='utf-8') as f:
        for line in f:
            m = json.loads(line); uid = m.get('stem')
            if uid in selected:
                r = selected[uid]
                if r['actual'] != m['ltr']: raise ValueError('train target differs')
                r.update(canonical=m['canon'], meta=m)
                train_records[uid] = r
    if set(train_records) != set(selected): raise ValueError('selected TRAIN metadata missing')
    records = [train_records[k] for k in sorted(train_records)]
    for split in ('valid', 'valid_unseen'):
        for ext in ('tsv', 'ltr', 'canon', 'meta.jsonl'):
            p = manifest / (split + '.' + ext)
            if file_sha(p) != bindings[str(p)]: raise ValueError('changed validation input')
        audio, actual = read_split(str(manifest), split)
        canon = [s.split() for s in (manifest / (split + '.canon')).read_text().splitlines()]
        with (manifest / (split + '.meta.jsonl')).open() as f:
            for row, line in enumerate(f):
                meta = json.loads(line); role = calibration_role(meta['spk'])
                if split == 'valid_unseen' and role != 'selection': continue
                if meta['spk'] in all_train_speakers: raise ValueError('speaker leakage')
                if split == 'valid_unseen' and meta['item'] in train_items: raise ValueError('item leakage')
                records.append(dict(path=audio[row][0], frames=int(audio[row][1]), row=row,
                                    canonical=canon[row], actual=actual[row], meta=meta, split=split, role=role))
    total = 0; support = collections.Counter(); reasons = collections.Counter(); groups = collections.Counter()
    with (out / 'utterances.jsonl').open('x') as fu, (out / 'sites.jsonl').open('x') as fq:
        for r in records:
            m = r.pop('meta'); canon, actual = r['canonical'], r['actual']
            spans, why = timing_windows(canon, actual, m)
            truth = project_on_canon(canon, actual)
            r.update(uid=m['stem'], speaker=m['spk'], item=m['item'], task=m['task'],
                     prompt=m['prompt'], offset=total, tag_windows=spans,
                     whole_tag_normal=(canon == actual))
            fu.write(json.dumps(r, ensure_ascii=False) + '\n')
            groups[r['split'] + '/' + r['role']] += 1
            for q, (c, t, span, reason) in enumerate(zip(canon, truth, spans, why)):
                s = {k: r[k] for k in ('split', 'role', 'row', 'uid', 'speaker', 'item', 'task', 'whole_tag_normal')}
                s.update(id=total, q=q, expected=c, actual=t, tag_window=span, timing_status=reason,
                         position=surface_position(canon, q),
                         following_vowel=next((x for x in canon[q+1:] if x in 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'), '<EOS>'))
                fq.write(json.dumps(s, ensure_ascii=False) + '\n'); total += 1
                reasons[r['split'], reason] += 1
                if r['role'] == 'train' and span is not None: support[t or '<DEL>'] += 1
    protocol = dict(version='P1_frozen_layers_v1', created=time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        parent_protocol=str(Path(a.parent_protocol).resolve()), parent_sha256=file_sha(a.parent_protocol),
        experiments=str(Path(a.experiments).resolve()), cache_sha256=file_sha(a.cache),
        data_sha256={name: file_sha(out / name) for name in ('utterances.jsonl', 'sites.jsonl')},
        total_sites=total, utterances=dict(groups), layers=list(LAYERS), windows=list(WINDOWS),
        train_selection='10 clean + 10 erroneous unique foreign TRAIN UIDs per item by fixed SHA256; no native rows (no TAG timing)',
        training_support=dict(support), timing_support={'/'.join(k): v for k, v in reasons.items()},
        probe=dict(type='1024-to-47 linear softmax, no expected-phone input', seed=20261027,
                   updates=1000, batch=512, optimizer='AdamW', lr=.001, weight_decay=.01,
                   normalization='TRAIN-only per-dimension mean/std, floor .01',
                   supervision='same TRAIN sites with unambiguous TAG timing and valid auto alignment in every checkpoint',
                   classes=sorted(load_vocab(str(manifest / 'dict.ltr.txt'))) + ['<DEL>']),
        automatic='mean of canonical CTC Viterbi token frames; depends on reference but not TAG',
        multi='equal mean of central auto mean, +/-40ms expanded mean, +/-80ms expanded mean; still 1024D',
        tag='mean frames whose frontend center is inside annotated interval; nearest center if shorter than one frame; invalid/outside interval unavailable',
        frame_centers=dict(w2vbert='20ms*i + 17.5ms center of two stacked 25ms fbank frames spaced by 10ms',
                           xlsr300m='20ms*i + 12.5ms convolution receptive-field center (400 samples)'),
        calibration='one threshold per condition on seen CAL normals, budget floor(.01*N), frozen for selection; failure at >1% stays failure',
        scope='all original selection sites, no test; unavailable TAG windows kept in denominators and common-window comparison separate',
        primary='suri 475 whole-TAG-normal recordings: paired u/o margin and same TAG-window representation drift; AUC requires both actual classes',
        cross_checkpoint='2k probe and normalization applied unchanged to 4k same-backbone features, diagnostic coordinate transfer only',
        limitations=['9 reused unseen development prompts', 'TAG boundary not physical truth or runtime input',
                     'single probe capacity/budget, negative result not information-theoretic upper bound',
                     'layerwise cosine uses corresponding coordinates, no causal or cross-backbone similarity claim',
                     'no encoder optimization, production or label changes; insertion gaps outside this canonical-site probe'])
    write_json(out / 'protocol.json', protocol)
    print(json.dumps(dict(sites=total, utterances=dict(groups), estimated_feature_GiB=total*1024*2*36/1024**3), ensure_ascii=False), flush=True)


def frame_centers(kind, n):
    return np.arange(n, dtype=np.float64) * .02 + (.0175 if kind == 'w2vbert' else .0125)


def pool_indices(segment, span, centers):
    """All auto indices ignore span. TAG indices depend only on supplied time."""
    n = len(centers)
    auto = np.asarray(segment or [], dtype=int)
    multi = []
    if len(auto):
        multi = [auto] + [np.arange(max(0, int(auto.min()) - k), min(n, int(auto.max()) + k + 1)) for k in (2, 4)]
    tag = np.empty(0, dtype=int)
    if span is not None and span[0] < centers[-1] + .01 and span[1] > centers[0] - .01:
        tag = np.flatnonzero((centers >= span[0]) & (centers < span[1]))
        if not len(tag): tag = np.asarray([np.argmin(abs(centers - (span[0] + span[1]) / 2))])
    return auto, multi, tag


class LayerHooks:
    def __init__(self, model):
        self.values = {}; self.handles = []
        layers = model.net.wav2vec2_bert.encoder.layers if model.kind == 'w2vbert' else model.net.encoder.layers
        for layer in (6, 12):
            def hook(module, args, output, key=layer):
                h = output[0]
                self.values[key] = h[0] if model.kind == 'w2vbert' else h[:, 0]
            self.handles.append(layers[layer - 1].register_forward_hook(hook))
        head = model.net.lm_head if model.kind == 'w2vbert' else model.head
        def head_hook(module, args): self.values[24] = args[0][0]
        self.handles.append(head.register_forward_pre_hook(head_hook))


def extract(a):
    from mdd.gop_feats_v2 import viterbi_align
    base = Path(a.out); p = json.loads((base / 'protocol.json').read_text())
    for name, digest in p['data_sha256'].items():
        if file_sha(base / name) != digest: raise ValueError('changed P1 data')
    parent = json.loads(Path(p['parent_protocol']).read_text())
    if file_sha(p['parent_protocol']) != p['parent_sha256']: raise ValueError('changed parent')
    exp = Path(p['experiments']); name = '%s_%d' % (a.backbone, a.step)
    dest = base / ('smoke_' + name if a.smoke else name); dest.mkdir(exist_ok=False)
    records = [json.loads(line) for line in (base / 'utterances.jsonl').open()]
    if a.smoke:
        records = [next(r for r in records if r['role'] == role and (item is None or r['item'] == item))
                   for role, item in [('train', None), ('calibration', None), ('selection', 'RW054'), ('selection', 'RS012')]]
        offset = 0
        for r in records: r['offset'] = offset; offset += len(r['canonical'])
        n = offset
    else: n = p['total_sites']
    estimate = n * 1024 * 2 * 9
    if shutil.disk_usage(dest).free < estimate + 1024**3: raise RuntimeError('insufficient space for feature allocation')
    old_report = json.loads((exp / ('%s_eval_%d' % (a.backbone, a.step)) / 'report.json').read_text())
    checkpoint = exp / (a.backbone + '_stage2') / ('step_%d.pt' % a.step)
    if file_sha(checkpoint) != old_report['checkpoint_sha256']: raise ValueError('checkpoint changed')
    old = {}
    for line in (exp / ('%s_eval_%d' % (a.backbone, a.step)) / 'sites.jsonl').open():
        s = json.loads(line); old.setdefault((s['split'], s['row']), []).append(s['predicted'])
    ck = torch.load(str(checkpoint), map_location='cpu'); vocab = ck['vocab']; inv = {v:k for k,v in vocab.items()}
    if ck['protocol_sha256'] != p['parent_sha256']: raise ValueError('checkpoint parent mismatch')
    model = Backbone(a.backbone, parent['backbones'][a.backbone]['path'], max(vocab.values())+1, torch.device('cuda'))
    model.load_state_dict(ck['model'], strict=True); del ck
    model.eval(); model.requires_grad_(False); hooks = LayerHooks(model)
    arrays = {(layer, window): np.lib.format.open_memmap(str(dest / ('L%d_%s.npy' % (layer, window))),
              mode='w+', dtype=np.float16, shape=(n, 1024)) for layer in LAYERS for window in WINDOWS}
    mask = np.zeros((n, 3), dtype=bool); bounds = np.full((n, 6), -1, dtype=np.int32)
    started = time.time(); matched = 0; maximum_head_difference = 0.; audio_digest = hashlib.sha256()
    seq_file = (dest / 'sequences.jsonl').open('x')
    with torch.inference_mode():
        for i, r in enumerate(records):
            x = read_wav(r['path'], 20.)
            if len(x) != r['frames']: raise ValueError('audio truncation')
            audio_digest.update(r['uid'].encode()); audio_digest.update(x.tobytes())
            hooks.values.clear()
            with torch.cuda.amp.autocast(): logits, lengths = model([x])
            length = int(lengths[0]); lp = logits[0, :length].float().log_softmax(-1).cpu().numpy()
            hidden = {layer: hooks.values[layer][:length].float().cpu().numpy() for layer in LAYERS}
            if any(h.shape != (length, 1024) or not np.isfinite(h).all() for h in hidden.values()): raise ValueError('feature shape/value')
            # Final probe features must be the actual CTC readout input, including
            # any encoder final normalization. Verify the hook does not alter it.
            if i < 4:
                head = model.net.lm_head if a.backbone == 'w2vbert' else model.head
                with torch.cuda.amp.autocast(): recovered = head(hooks.values[24])[:length]
                difference = float((recovered - logits[0, :length]).abs().max())
                maximum_head_difference = max(maximum_head_difference, difference)
                if difference != 0: raise ValueError('final readout mismatch')
            pred = [inv[t] for t, _ in itertools.groupby(lp.argmax(-1)) if t in inv]
            projected = project_on_canon(r['canonical'], pred)
            if r['split'] != 'train':
                if projected != old[r['split'], r['row']]: raise ValueError('inference differs from original sites')
                matched += 1
            segments = viterbi_align(lp, [vocab[t] for t in r['canonical']], blank=0)
            centers = frame_centers(a.backbone, length)
            for q, span in enumerate(r['tag_windows']):
                ix = r['offset'] + q
                auto, multi, tag = pool_indices(segments[q] if segments is not None else [], span, centers)
                for w, indices in enumerate((auto, auto, tag)):
                    if len(indices): mask[ix, w] = True; bounds[ix, 2*w:2*w+2] = [int(indices.min()), int(indices.max())+1]
                for layer, h in hidden.items():
                    arrays[layer, 'auto'][ix] = h[auto].mean(0) if len(auto) else 0
                    arrays[layer, 'multi'][ix] = np.mean([h[j].mean(0) for j in multi], axis=0) if multi else 0
                    arrays[layer, 'tag'][ix] = h[tag].mean(0) if len(tag) else 0
            seq_file.write(json.dumps(dict(uid=r['uid'], split=r['split'], row=r['row'], predicted=pred), ensure_ascii=False)+'\n')
            if (i+1) % 100 == 0 or i+1 == len(records):
                status = dict(condition=name, utterances=i+1, target=len(records), elapsed_seconds=time.time()-started,
                              estimated_remaining_seconds=(time.time()-started)/(i+1)*(len(records)-i-1))
                write_json(dest / 'live_status.json', status); write_json(base / 'live_status.json', status)
                print(json.dumps(status), flush=True)
    seq_file.close()
    for array in arrays.values(): array.flush()
    np.save(dest / 'available.npy', mask); np.save(dest / 'window_bounds.npy', bounds)
    write_json(dest / 'finished.json', dict(status='features_extracted', encoder_updates=0, smoke=a.smoke,
        utterances=len(records), sites=n, validation_projection_matches=matched, maximum_head_difference=maximum_head_difference,
        available=mask.sum(0).tolist(), checkpoint_sha256=old_report['checkpoint_sha256'],
        audio_order_and_pcm_sha256=audio_digest.hexdigest(),
        protocol_sha256=file_sha(base / 'protocol.json'), source_sha256=file_sha(__file__),
        elapsed_seconds=time.time()-started))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['prepare', 'extract'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--parent-protocol'); ap.add_argument('--experiments'); ap.add_argument('--cache')
    ap.add_argument('--backbone', choices=['w2vbert', 'xlsr300m']); ap.add_argument('--step', type=int, choices=[2000,4000])
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.stage == 'prepare': prepare(a)
    else:
        # Independent checkpoints may be extracted concurrently. The sequential
        # launcher joins a running extraction through this lock and accepts only
        # a complete receipt tied to the unchanged protocol.
        import fcntl
        name = ('smoke_' if a.smoke else '') + '%s_%d' % (a.backbone, a.step)
        with (Path(a.out) / ('.' + name + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            done = Path(a.out) / name / 'finished.json'
            if done.exists():
                receipt = json.loads(done.read_text())
                if receipt['protocol_sha256'] != file_sha(Path(a.out) / 'protocol.json'):
                    raise ValueError('completed extraction uses a different protocol')
                print('joined completed extraction: ' + name, flush=True)
            else: extract(a)


if __name__ == '__main__': main()
