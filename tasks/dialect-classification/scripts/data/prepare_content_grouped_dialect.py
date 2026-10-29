"""Rebuild development splits using source IDs plus identical decoded audio.

Uses FLAC STREAMINFO PCM MD5 as a candidate fingerprint, and verifies every
duplicate group by decoding its files and comparing SHA256. No historical
predictions or trained checkpoints are used to assign groups or select rows.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import soundfile as sf

ROOT = Path('/data/speech-ai/out/dialect')
SOURCES = [ROOT / 'speakerid_20261011/manifests' / (s + '.csv') for s in ['train', 'valid', 'test']]
SOURCES += [ROOT / 'expanded_six_class_20261013/manifests' / (s + '.csv') for s in ['train', 'valid', 'test']]
LABELS = ['Gangwon', 'Gyeongsang', 'Jeolla', 'Jeju', 'Chungcheong', 'Standard']
SEED = '20261014-content-v1'


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def decoded_digest(path, algorithm='md5'):
    h = hashlib.new(algorithm)
    with sf.SoundFile(path) as audio:
        if audio.subtype != 'PCM_16':
            raise ValueError('Expected prepared PCM16: ' + path)
        info = [audio.samplerate, audio.channels, len(audio)]
        for block in audio.blocks(blocksize=65536, dtype='int16', always_2d=True):
            h.update(block.astype('<i2', copy=False).tobytes())
    return info, h.hexdigest()


def fingerprint(path):
    with open(path, 'rb') as stream:
        header = stream.read(42)
    if header[:4] == b'fLaC' and (header[4] & 127) == 0 and int.from_bytes(header[5:8], 'big') == 34:
        block = header[8:42]
        packed = int.from_bytes(block[10:18], 'big')
        rate, channels, bits, frames = packed >> 44, ((packed >> 41) & 7) + 1, ((packed >> 36) & 31) + 1, packed & ((1 << 36) - 1)
        md5 = block[18:34].hex()
        if bits == 16 and int(md5, 16):
            return path, '{}:{}:{}:{}'.format(rate, channels, frames, md5)
    info, md5 = decoded_digest(path)
    return path, ':'.join(map(str, info)) + ':' + md5


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def status(stage, **extra):
        value = {'status': 'running', 'stage': stage, 'updated_at': datetime.now().astimezone().isoformat(),
                 'elapsed_seconds': round(time.monotonic() - started, 1), **extra}
        write(args.out / 'preparation_status.json', value)
        print(json.dumps(value), flush=True)

    status('load_source_manifests')
    rows, file_people, parent = {}, defaultdict(set), {}

    def find(p):
        parent.setdefault(p, p)
        if parent[p] != p:
            parent[p] = find(parent[p])
        return parent[p]

    def merge(people):
        roots = sorted({find(p) for p in people})
        for root in roots[1:]:
            parent[root] = roots[0]

    source_hashes = {}
    for source in SOURCES:
        source_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        with source.open() as stream:
            for row in csv.DictReader(stream):
                if row['row_id'] in rows:
                    old = rows[row['row_id']]
                    assert all(old[k] == row[k] for k in ['audio_path', 'person_id', 'label_id', 'start_sec', 'duration_sec'])
                    continue
                rows[row['row_id']] = row
                file_people[row['audio_path']].add(row['person_id'])
                find(row['person_id'])
    for people in file_people.values():
        merge(people)
    fingerprints, content_files = {}, defaultdict(list)
    with (args.out / 'pcm_fingerprints.jsonl').open('w') as stream, ThreadPoolExecutor(max_workers=4) as pool:
        for n, (path, key) in enumerate(pool.map(fingerprint, sorted(file_people)), 1):
            fingerprints[path] = key
            content_files[key].append(path)
            stream.write(json.dumps({'path': path, 'pcm16_fingerprint': key}) + '\n')
            if n % 10000 == 0:
                stream.flush()
                status('fingerprint_source_audio', completed=n, total=len(file_people))
    duplicates = [(key, files) for key, files in content_files.items() if len(files) > 1]
    status('verify_duplicate_pcm', groups=len(duplicates))
    duplicate_groups = []
    for n, (key, files) in enumerate(duplicates, 1):
        decoded = [decoded_digest(path, 'sha256') for path in files]
        assert len({(tuple(info), value) for info, value in decoded}) == 1, 'FLAC metadata/hash inconsistency'
        people = set().union(*(file_people[p] for p in files))
        merge(people)
        duplicate_groups.append({'pcm16_fingerprint': key, 'verified_pcm_sha256': decoded[0][1],
                                 'files': files, 'source_people': sorted(people)})
        if n % 500 == 0:
            status('verify_duplicate_pcm', completed=n, total=len(duplicates))
    write(args.out / 'duplicate_audio_groups.json', duplicate_groups)
    labels_by_group, people_by_group = defaultdict(set), defaultdict(set)
    for row in rows.values():
        group = find(row['person_id'])
        labels_by_group[group].add(int(row['label_id']))
        people_by_group[group].add(row['person_id'])
    ambiguous = {group for group, labels in labels_by_group.items() if len(labels) != 1}
    # One deterministic window per distinct source waveform; never select on errors.
    chosen = {}
    for row in rows.values():
        group, content = find(row['person_id']), fingerprints[row['audio_path']]
        if group in ambiguous:
            continue
        rank = sha(SEED + ':window:' + row['row_id'])
        if content not in chosen or rank < chosen[content][0]:
            chosen[content] = (rank, row, group)
    buckets = {s: {i: defaultdict(list) for i in range(6)} for s in ['train', 'valid', 'test']}
    split_of = {}
    for group in people_by_group:
        h = int(sha(SEED + ':split:' + group)[:8], 16) % 1000
        split_of[group] = 'train' if h < 800 else 'valid' if h < 900 else 'test'
    for _, row, group in chosen.values():
        new = dict(row)
        new['original_person_id'] = row['person_id']
        new['person_id'] = group
        new['evaluation_group_id'] = group
        new['source_dataset'] = row.get('source_dataset') or row['record_key'].split(':')[0]
        new['pcm16_fingerprint'] = fingerprints[row['audio_path']]
        buckets[split_of[group]][int(row['label_id'])][group].append(new)
    output = args.out / 'manifests'
    output.mkdir(exist_ok=True)
    fields = sorted(set().union(*(set(r) for r in rows.values())) | {'original_person_id', 'evaluation_group_id', 'pcm16_fingerprint'})
    selected, counts = {}, {}
    for split, target in [('train', args.train_per_class), ('valid', 600), ('test', 1000)]:
        selected[split] = []
        for label in range(6):
            groups = buckets[split][label]
            order = sorted(groups, key=lambda p: sha(SEED + ':person:' + p))
            for group in order:
                groups[group].sort(key=lambda r: sha(SEED + ':record:' + r['row_id']))
            values, offset = [], 0
            while len(values) < target:
                added = 0
                for group in order:
                    if offset < len(groups[group]):
                        values.append(groups[group][offset]); added += 1
                        if len(values) == target:
                            break
                if not added:
                    break
                offset += 1
            assert values, 'No rows for class/split'
            selected[split].extend(values)
        with (output / (split + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(selected[split])
        counts[split] = {label: {'rows': sum(r['label_name'] == label for r in selected[split]),
                               'groups': len({r['person_id'] for r in selected[split] if r['label_name'] == label})} for label in LABELS}
    overlaps = {}
    for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        overlaps[a + '_' + b] = {}
        for key in ['person_id', 'original_person_id', 'record_key', 'audio_path', 'pcm16_fingerprint']:
            n = len({r[key] for r in selected[a]} & {r[key] for r in selected[b]})
            assert n == 0, (a, b, key, n)
            overlaps[a + '_' + b][key] = n
    summary = {'created_at': datetime.now().astimezone().isoformat(), 'seed': SEED, 'labels': LABELS,
               'counts': counts, 'source_manifest_sha256': source_hashes,
               'source_rows': len(rows), 'source_files': len(file_people), 'source_person_ids': len(parent),
               'identity_components': len(people_by_group), 'components_with_multiple_source_ids': sum(len(v) > 1 for v in people_by_group.values()),
               'duplicate_pcm_groups': len(duplicates), 'conflicting_label_components_excluded': len(ambiguous),
               'overlaps': overlaps, 'manifest_sha256': hashlib.sha256(b''.join((output / (s + '.csv')).read_bytes() for s in ['train', 'valid', 'test'])).hexdigest(),
               'split_policy': 'Union source person IDs connected by any identical decoded recording, then hash components before row sampling.',
               'selection_policy': 'One deterministic window per decoded source waveform; round robin across identity components, no predictions used.',
               'initialization_policy': 'Generic pretrained encoder only; prohibit all previous dialect-finetuned checkpoints, heads, teachers and fitted normalizers.',
               'evaluation_status': 'New split for development; original dataset has been used in prior development. Not an independent external benchmark.',
               'test_access_policy': 'Keep test predictions sealed until protocol and model selection are frozen. This preparation reads metadata/audio only.',
               'limitations': ['Near-duplicates with crop, shift, noise or resampling remain unaudited.',
                               'Different source IDs without identical recordings may still identify one actual person.',
                               'Standard label code 01 remains provisional and comes from a different recording style.',
                               'A new external speaker/recording set is required to substantiate a final booth accuracy claim.']}
    write(output / 'summary.json', summary)
    write(args.out / 'identity_components.json', {g: sorted(p) for g, p in people_by_group.items() if len(p) > 1})
    status('complete', status='complete', counts=counts, duplicate_pcm_groups=len(duplicates), components_with_multiple_source_ids=summary['components_with_multiple_source_ids'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-per-class', type=int, default=15000)
    args = parser.parse_args()
    try:
        prepare(args)
    except Exception as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        write(args.out / 'preparation_status.json', {'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)})
        raise
