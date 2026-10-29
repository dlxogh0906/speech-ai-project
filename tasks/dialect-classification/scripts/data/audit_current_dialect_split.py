"""Audit current validation/test against current and ancestral training manifests.

Checks source IDs and exact file bytes. This does not certify person identity
across providers, near-duplicate audio, or an untouched final evaluation set.
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

ROOT = Path('/data/speech-ai/out/dialect')
CURRENT = ROOT / 'expanded_six_class_20261013/manifests'
TRAIN_MANIFESTS = [ROOT / 'speakerid_20261011/manifests/train.csv',
                   ROOT / 'standard_six_class_20261013/manifests/train.csv',
                   CURRENT / 'train.csv']


def write(path, value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def file_digest(path):
    sha = hashlib.sha256()
    size = 0
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
            size += len(block)
    return sha.hexdigest(), size


def audit(args):
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    keys = ['person_id', 'record_key', 'row_id', 'audio_path', 'source_audio_sha256']
    values = {s: {k: set() for k in keys} for s in ['train_lineage', 'valid', 'test']}
    file_splits, file_people = defaultdict(set), defaultdict(set)
    provenance = []
    for split, path in [('train_lineage', p) for p in TRAIN_MANIFESTS] + [(s, CURRENT / (s + '.csv')) for s in ['valid', 'test']]:
        counts, people = Counter(), defaultdict(set)
        with path.open() as stream:
            for row in csv.DictReader(stream):
                for key in keys:
                    if row.get(key):
                        values[split][key].add(row[key])
                counts[row['label_name']] += 1
                people[row['label_name']].add(row['person_id'])
                file_splits[row['audio_path']].add(split)
                file_people[row['audio_path']].add(row['person_id'])
        provenance.append({'split': split, 'path': str(path), 'sha256': file_digest(path)[0],
                           'rows_per_class': dict(counts), 'people_per_class': {k: len(v) for k, v in people.items()}})
    pairs = [('train_lineage', 'valid'), ('train_lineage', 'test'), ('valid', 'test')]
    overlaps = {a + '__' + b: {k: len(values[a][k] & values[b][k]) for k in keys} for a, b in pairs}
    report = {
        'created_at': datetime.now().astimezone().isoformat(), 'training_provenance': provenance,
        'overlaps': overlaps, 'unique_source_files': len(file_splits),
        'source_counts': {s: {k: len(v) for k, v in groups.items()} for s, groups in values.items()},
        'exact_file_bytes': {'status': 'pending'},
        'strict_leakage_free_certified': False,
        'limitations': [
            'Source person IDs cannot exclude one real person registered under different IDs or across providers.',
            'Exact file hashes cannot detect differently encoded, trimmed, shifted, or augmented copies.',
            'The existing test was inspected in prior development; it is a development comparison set, not an untouched final test.',
            'Standard uses a different reading corpus, and Dialect 01 is still a provisional label mapping.',
            'This audit does not verify the generic pretrained encoders\' entire pretraining corpora.',
        ],
    }
    write(args.out / 'split_audit.json', report)
    print(json.dumps({'metadata_overlaps': overlaps, 'files_to_hash': len(file_splits)}, ensure_ascii=False), flush=True)
    progress = args.out / 'audit_status.json'
    write(progress, {'status': 'running', 'stage': 'exact_file_hash', 'completed': 0, 'total': len(file_splits)})
    groups = defaultdict(list)
    total_bytes, missing, failures = 0, [], []

    def hash_one(path):
        try:
            sha, size = file_digest(path)
            return {'path': path, 'sha256': sha, 'bytes': size, 'splits': sorted(file_splits[path])}
        except OSError as exc:
            return {'path': path, 'error': type(exc).__name__, 'message': str(exc)}

    with (args.out / 'file_hashes.jsonl').open('w') as stream, ThreadPoolExecutor(max_workers=4) as pool:
        for number, item in enumerate(pool.map(hash_one, sorted(file_splits)), 1):
            stream.write(json.dumps(item) + '\n')
            if 'error' in item:
                (missing if item['error'] == 'FileNotFoundError' else failures).append(item)
            else:
                groups[item['sha256']].append(item['path'])
                total_bytes += item['bytes']
            if number % 5000 == 0:
                stream.flush()
                state = {'status': 'running', 'stage': 'exact_file_hash', 'completed': number, 'total': len(file_splits),
                         'elapsed_seconds': round(time.monotonic() - started, 1), 'bytes_read': total_bytes}
                write(progress, state)
                print(json.dumps(state), flush=True)
    duplicated = []
    for sha, paths in groups.items():
        splits = set().union(*(file_splits[p] for p in paths))
        if len(splits) > 1:
            duplicated.append({'sha256': sha, 'paths': paths, 'splits': sorted(splits)})
    report['exact_file_bytes'] = {'status': 'complete', 'hashed_files': sum(len(v) for v in groups.values()),
                                  'bytes_read': total_bytes, 'cross_split_duplicate_groups': len(duplicated),
                                  'cross_split_duplicates': duplicated, 'missing_files': missing, 'errors': failures}
    report['elapsed_seconds'] = time.monotonic() - started
    report['completed_at'] = datetime.now().astimezone().isoformat()
    write(args.out / 'split_audit.json', report)
    status = 'complete' if not missing and not failures else 'incomplete'
    write(progress, {'status': status, 'stage': 'finished', 'elapsed_seconds': report['elapsed_seconds'],
                     'completed': len(file_splits), 'total': len(file_splits), 'cross_split_duplicate_groups': len(duplicated)})
    print(json.dumps({'status': status, 'duplicate_groups': len(duplicated), 'missing': len(missing), 'errors': len(failures)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args())
