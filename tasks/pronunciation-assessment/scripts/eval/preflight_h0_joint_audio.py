"""Bind planned fold audio and original partial annotations before model fitting."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import json
import wave
from train_h0_backbone_control import file_sha, stream
from train_w2vbert_ctc import read_split, load_vocab
from mdd.h0_partial_reference import partial_tokens, partial_mapping, classify_truth
from mdd.h0_partial_scoring import mapping_indices, gap_value, annotation_mask
from audit_h0_p5_readiness import write


def inspect_audio(pair):
    path, expected = pair
    with wave.open(path, 'rb') as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()) != (16000,1,2,expected):
            raise ValueError('audio header mismatch: '+path)
        pcm = f.readframes(expected)
    if len(pcm) != 2*expected or not 0 < expected <= 320000:
        raise ValueError('truncated, empty, or >20s planned audio: '+path)
    return dict(path=path, samples=expected, pcm_sha256=hashlib.sha256(pcm).hexdigest())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(); out = a.root/'audio_annotation_preflight'; out.mkdir(exist_ok=False)
    p = json.loads((a.root/'protocol.json').read_text())
    mr = json.loads((a.root/'manifests/report.json').read_text())
    for name, binding in mr['bindings'].items():
        assert file_sha(name) == binding['sha256'], name
    files = {}; roles = collections.defaultdict(set); exposures = {}
    for stage, updates in [('stage1',10000), ('stage2',4000)]:
        rows, labels = read_split(str(a.root/'manifests'/stage), 'train')
        batches = stream(rows, p['training']['batch_samples'], p['training']['seed'])
        h = hashlib.sha256(); draws = samples = 0; unique = set(); receipts = []
        for micro in range(updates*p['training']['accum']):
            ids = next(batches); h.update((' '.join(map(str,ids))+'\n').encode())
            draws += len(ids); samples += sum(rows[i][1] for i in ids); unique.update(ids)
            if (micro+1) in (2000*p['training']['accum'], updates*p['training']['accum']):
                receipts.append(dict(row_draws=draws, audio_samples=samples, batch_order_sha256=h.hexdigest()))
        if stage == 'stage2':
            expected = json.loads((a.root/'matrices/report.json').read_text())['receipts']
            assert receipts == expected
        exposures[stage] = dict(receipts=receipts, unique_manifest_rows=len(unique))
        for i in unique:
            path, n = rows[i]
            assert labels[i]
            if path in files:
                assert files[path] == n
            files[path] = n; roles[path].add('train')
    vocab = load_vocab(str(a.root/'manifests/stage2/dict.ltr.txt'))
    support = {}; role_paths = {}
    for role in ('cal','inner','outer'):
        manifest = a.root/'manifests/stage2'
        rows, legacy_actual = read_split(str(manifest), role)
        canon = (manifest/(role+'.canon')).read_text().splitlines()
        count = collections.Counter(); role_paths[role] = set()
        with gzip.open(out/(role+'_annotations.jsonl.gz'), 'wt') as dest:
            for i, line in enumerate((manifest/(role+'.meta.jsonl')).open()):
                meta = json.loads(line); path, n = rows[i]
                assert Path(path).stem == meta['stem']
                if path in files:
                    assert files[path] == n
                files[path] = n; roles[path].add(role); role_paths[role].add(path)
                ref, ref_owners, ref_ignored = partial_tokens(meta['lab_phones'], vocab, reference=True)
                actual, actual_owners, actual_ignored = partial_tokens(meta['tag_phones'], vocab)
                _, options = partial_mapping(ref, actual)
                sites = []
                for q, (phone, values) in enumerate(zip(ref, options)):
                    kind, truth = classify_truth(phone, values)
                    sites.append(dict(q=q, expected=phone, actual=truth, status=kind))
                    count[kind] += 1
                indices = mapping_indices(ref, actual)
                record = dict(role=role, row=i, uid=meta['stem'], speaker=meta['spk'], item=meta['item'],
                    task=meta['task'], language=meta['language'], reference=ref, actual=actual, sites=sites,
                    reference_owners=ref_owners, actual_owners=actual_owners,
                    reference_ignored=ref_ignored, actual_ignored=actual_ignored,
                    original_lab_phones=meta['lab_phones'], original_tag_phones=meta['tag_phones'],
                    legacy_canonical=canon[i].split(), legacy_actual=legacy_actual[i])
                record['annotation_mask'] = annotation_mask(record, indices)
                record['truth_gaps'] = [gap_value(ref, actual, indices, g) for g in range(len(ref)+1)]
                count['normal_gap'] += sum(g == () for g in record['truth_gaps'])
                count['insertion_gap'] += sum(bool(g) for g in record['truth_gaps'])
                count['recordings'] += 1
                dest.write(json.dumps(record, ensure_ascii=False)+'\n')
        assert i+1 == len(rows)
        support[role] = dict(count)
    write(out/'annotation_support.json', support)
    write(out/'planned_exposures.json', exposures)
    cross_role_paths = {path:sorted(rs) for path, rs in roles.items() if len(rs)>1}
    if cross_role_paths:
        write(out/'cross_role_paths.json', cross_role_paths)
        raise ValueError('same audio path across train/evaluation roles')
    paths = sorted(files.items()); hashes = collections.defaultdict(list); audio_samples = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool, gzip.open(out/'audio_bindings.jsonl.gz','wt') as fw:
        for start in range(0, len(paths), 512):
            for record in pool.map(inspect_audio, paths[start:start+512]):
                record['roles'] = sorted(roles[record['path']])
                hashes[record['pcm_sha256']].append(record['path'])
                audio_samples += record['samples']
                fw.write(json.dumps(record, ensure_ascii=False)+'\n')
            if start % 10240 == 0:
                print('audio checked', min(start+512,len(paths)), '/', len(paths), flush=True)
    duplicates = []
    for digest, ps in hashes.items():
        rs = set(r for path in ps for r in roles[path])
        if len(rs)>1:
            duplicates.append(dict(sha256=digest, roles=sorted(rs), paths=ps))
    write(out/'report.json', dict(status='passed' if not duplicates else 'cross_role_PCM_duplicates_require_review',
        unique_audio_files=len(paths), audio_samples=audio_samples, cross_role_PCM_duplicates=duplicates,
        planned_exposures=exposures, annotation_support=support, model_optimizer_updates=0,
        limits=['exact whole-PCM equality only; approximate copies and cross-corpus speaker identity unresolved',
                'original annotation mapping, not independent listening or repaired labels']))
    print('audio/annotation preflight finished; cross-role PCM duplicates', len(duplicates), flush=True)


if __name__ == '__main__':
    main()
