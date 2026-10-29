"""Prepare a six-class training pilot and preserve verified five-region evaluation.

AI Hub 130 exposes dialect labels but no person IDs in its released text
metadata. Consequently all selected 130 recordings are training-only. They
are NEVER randomly split into validation/test or assigned invented person IDs.
The existing person-disjoint five-region validation/test sets are retained as
regression diagnostics. Independent Seoul/Gyeonggi evaluation remains missing.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

LABELS = ['Gangwon', 'Gyeongsang', 'Jeolla', 'Jeju', 'Chungcheong', 'Seoul_Gyeonggi']
LABELS_KO = ['강원', '경상', '전라', '제주', '충청', '서울·경기(표준어 계열)']
DIALECT = {'1': 5, '2': 0, '3': 4, '4': 1, '5': 2, '6': 3}
FIELDS = ['row_id', 'audio_path', 'start_sec', 'duration_sec', 'label_id', 'label_name',
          'speaker_key', 'record_key', 'topic', 'original_split', 'person_id', 'source_dataset',
          'person_identity_verified', 'transcript', 'source_audio_sha256']


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def download(info, root, reserve_gib=60):
    """AI Hub outer TAR contains .tar.gz parts for dataset 130 (not ZIPs)."""
    folder = root / 'archives' / str(info['fileSn'])
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / Path(info['fileStreCours']).name
    expected = int(info['fileSize'])
    marker = folder / 'download.json'
    if marker.is_file() and target.is_file() and target.stat().st_size == expected:
        return target
    if shutil.disk_usage(root).free < expected * 2 + reserve_gib * 2**30:
        raise RuntimeError('Disk reserve reached before downloading archive')
    key = os.environ.get('AIHUB_APIKEY', '').strip() or (Path.home() / '.aihub_apikey').read_text().strip()
    request = Request('https://api.aihub.or.kr/down/0.6/130.do?fileSn=' + str(info['fileSn']),
                      headers={'apikey': key, 'User-Agent': 'speech-ai-dialect/1.0'})
    print('[download130] file=%s expected_bytes=%s' % (info['fileSn'], expected), flush=True)
    parts = []
    started = time.time()
    try:
        with urlopen(request, timeout=120) as response:
            if 'tar' not in response.headers.get('Content-Type', ''):
                raise RuntimeError('Non-TAR API response: ' + response.read(1000).decode('utf-8', 'replace').replace(key, '<redacted>'))
            with tarfile.open(fileobj=response, mode='r|*') as bundle:
                for member in bundle:
                    if not member.isfile():
                        continue
                    name = Path(member.name.replace('\\', '/')).name
                    if name != target.name and not name.startswith(target.name + '.part'):
                        raise RuntimeError('Unexpected API archive member: ' + name)
                    part = folder / (name + '.download')
                    with bundle.extractfile(member) as source, part.open('wb') as sink:
                        shutil.copyfileobj(source, sink, 4 * 1024**2)
                    parts.append((name, part))
    except HTTPError as error:
        raise RuntimeError('AI Hub HTTP %s: %s' % (error.code, error.read(2000).decode('utf-8', 'replace').replace(key, '<redacted>'))) from None
    if not parts or len({name for name, _ in parts}) != len(parts):
        raise RuntimeError('Missing or duplicated download parts')
    parts.sort(key=lambda item: int(item[0].rsplit('.part', 1)[1]) if '.part' in item[0] else 0)
    temporary = target.with_name(target.name + '.tmp')
    with temporary.open('wb') as sink:
        for _, part in parts:
            with part.open('rb') as source:
                shutil.copyfileobj(source, sink, 4 * 1024**2)
    if temporary.stat().st_size != expected:
        raise RuntimeError('Downloaded archive size does not match catalog')
    os.replace(temporary, target)
    for _, part in parts:
        part.unlink()
    write_json(marker, {'filekey': info['fileSn'], 'bytes': expected, 'elapsed_seconds': time.time() - started})
    return target


def label_metadata(path):
    metadata, texts, counts = {}, {}, Counter()
    with tarfile.open(path, 'r:gz') as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not name.endswith(('_metadata.txt', '_scripts.txt')):
                raise RuntimeError('Unrecognized label schema, inspect before training: ' + name)
            handle = io.TextIOWrapper(bundle.extractfile(member), encoding='utf-8-sig')
            for line in handle:
                if name.endswith('_metadata.txt'):
                    parts = [value.strip() for value in line.split('|')]
                    if len(parts) != 9:
                        raise RuntimeError('Unexpected 130 metadata field count')
                    rid = Path(parts[0]).stem
                    if rid in metadata:
                        raise RuntimeError('Duplicated source recording name')
                    metadata[rid] = {'topic': parts[1], 'detail_topic': parts[2], 'gender': parts[3],
                                     'age': parts[4], 'residence': parts[5], 'dialect': parts[6],
                                     'source': parts[7], 'quality': parts[8]}
                    counts[parts[6]] += 1
                else:
                    parts = line.split(' :: ', 1)
                    if len(parts) == 2:
                        texts[Path(parts[0]).stem] = parts[1].strip()
    return metadata, texts, counts


def convert(info, label_info, args):
    root = args.root
    directory = root / 'archives' / str(info['fileSn'])
    marker, manifest = directory / 'converted.json', directory / 'rows.csv'
    if marker.is_file() and manifest.is_file():
        return manifest
    labels = download(label_info, root)
    metadata, texts, counts = label_metadata(labels)
    print('[labels130] ' + json.dumps({'filekey': label_info['fileSn'], 'recordings': len(metadata), 'dialect_counts': dict(counts), 'person_id_field': False}), flush=True)
    archive = download(info, root)
    audio_dir = root / 'audio' / str(info['fileSn'])
    audio_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    temporary = manifest.with_suffix('.tmp.csv')
    with tarfile.open(archive, 'r|gz') as bundle, temporary.open('w', newline='') as sink:
        writer = csv.DictWriter(sink, fieldnames=FIELDS)
        writer.writeheader()
        for member in bundle:
            if not member.isfile() or not member.name.lower().endswith('.wav'):
                continue
            rid = Path(member.name).stem
            profile = metadata.get(rid)
            if profile is None:
                raise RuntimeError('Missing label for audio: ' + rid)
            stats['source_recordings'] += 1
            # Crowdsourced adults with normal quality. Use dialect, not residence.
            if profile['source'] != '3' or profile['quality'] != '1' or profile['age'] not in ('A', 'S') or profile['dialect'] not in DIALECT:
                stats['excluded_metadata'] += 1
                continue
            raw = bundle.extractfile(member).read()
            wave, rate = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
            digest = hashlib.sha256(raw).hexdigest()
            wave = wave.mean(axis=1)
            if rate != 16000:
                divisor = math.gcd(rate, 16000)
                wave = resample_poly(wave, 16000 // divisor, rate // divisor).astype(np.float32)
            duration = len(wave) / 16000
            if duration < 3 or not np.isfinite(wave).all():
                stats['short_or_nonfinite'] += 1
                continue
            if shutil.disk_usage(root).free < 60 * 2**30:
                raise RuntimeError('Disk reserve reached during conversion')
            path = audio_dir / (rid + '.flac')
            sf.write(path, wave, 16000, subtype='PCM_16')
            label = DIALECT[profile['dialect']]
            length = min(duration, 12.)
            # A single center window per recording prevents repeated-file inflation.
            start = max(0., (duration - length) / 2)
            key = '130:' + rid
            writer.writerow({'row_id': key, 'audio_path': str(path), 'start_sec': start,
                             'duration_sec': length, 'label_id': label, 'label_name': LABELS[label],
                             'speaker_key': '', 'record_key': key, 'topic': profile['topic'],
                             'original_split': 'Training', 'person_id': '', 'source_dataset': '130',
                             'person_identity_verified': False, 'transcript': texts.get(rid, ''),
                             'source_audio_sha256': digest})
            stats['converted_recordings'] += 1
            stats[LABELS[label]] += 1
            if stats['converted_recordings'] % 1000 == 0:
                print('[audio130] ' + json.dumps(dict(stats)), flush=True)
    os.replace(temporary, manifest)
    write_json(marker, {'stats': dict(stats), 'metadata_dialect_counts': dict(counts),
                        'identity': 'No speaker ID present. Training-only.', 'source_filekey': info['fileSn']})
    archive.unlink()  # This job's source archive after complete gzip decode and conversion.
    return manifest


def choose(rows, cap, person_cap=None):
    rows = sorted(rows, key=lambda row: hashlib.sha256(row['row_id'].encode()).hexdigest())
    count = Counter()
    chosen = []
    for row in rows:
        pid = row.get('person_id')
        if person_cap and count[pid] >= person_cap:
            continue
        chosen.append(row)
        count[pid] += 1
        if len(chosen) >= cap:
            break
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/speech-ai/corpus/dialect_six_class_20261012'))
    parser.add_argument('--out', type=Path, default=Path('/data/speech-ai/out/dialect/six_class_20261012'))
    parser.add_argument('--catalog', type=Path, default=Path('/data/speech-ai/out/dialect/aihub130_review_20261012/files.json'))
    parser.add_argument('--old-manifests', type=Path, default=Path('/data/speech-ai/out/dialect/speakerid_20261011/manifests'))
    parser.add_argument('--audio-filekeys', type=int, nargs='+', default=[39440, 39446, 39448])
    parser.add_argument('--old-train-per-class', type=int, default=2000)
    parser.add_argument('--old-eval-per-class', type=int, default=500)
    parser.add_argument('--new-train-per-class', type=int, default=1000)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    files = json.loads(args.catalog.read_text())['list']
    jobs = []
    for key in args.audio_filekeys:
        audio = next(f for f in files if f['fileSn'] == key)
        if '/1.Training' not in audio['fileStreCours']:
            raise RuntimeError('Only original Training data is authorized for this preparation path')
        wanted = audio['fileStreCours'].replace('/2.원천데이터/', '/1.라벨링데이터/')
        label = next(f for f in files if f['fileStreCours'] == wanted)
        jobs.append((audio, label))
    write_json(args.out / 'status.json', {'status': 'running', 'stage': 'prepare_130', 'audio_filekeys': args.audio_filekeys})
    try:
        new_rows = []
        for audio, labels in jobs:
            path = convert(audio, labels, args)
            with path.open(newline='') as handle:
                new_rows.extend(csv.DictReader(handle))
        manifests = args.out / 'manifests'
        manifests.mkdir(exist_ok=True)
        all_rows, original_people = {}, {}
        selected_new = []
        for label in range(6):
            candidates = [r for r in new_rows if int(r['label_id']) == label]
            if not candidates:
                raise RuntimeError('No 130 training examples for ' + LABELS[label])
            selected_new.extend(choose(candidates, args.new_train_per_class))
        known_audio = set()
        for row in selected_new:
            if row['source_audio_sha256'] in known_audio:
                raise RuntimeError('Duplicate source audio in new training selection')
            known_audio.add(row['source_audio_sha256'])
        for split in ['train', 'valid', 'test']:
            with (args.old_manifests / (split + '.csv')).open(newline='') as handle:
                old_rows = list(csv.DictReader(handle))
            original_people[split] = {r['person_id'] for r in old_rows}
            selected = []
            for label in range(5):
                candidates = [r for r in old_rows if int(r['label_id']) == label]
                selected.extend(choose(candidates, args.old_train_per_class if split == 'train' else args.old_eval_per_class, 20))
            for r in selected:
                r.update(source_dataset=r['row_id'].split(':', 1)[0], person_identity_verified=True)
            if split == 'train':
                selected.extend(selected_new)
            all_rows[split] = selected
            with (manifests / (split + '.csv')).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows({field: r.get(field, '') for field in FIELDS} for r in selected)
        overlap = {a + '_' + b: len(original_people[a] & original_people[b]) for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]}
        if any(overlap.values()):
            raise RuntimeError('Person overlap in original regional manifests')
        h = hashlib.sha256()
        for split in ['train', 'valid', 'test']:
            h.update((manifests / (split + '.csv')).read_bytes())
        summary = {'labels': LABELS, 'labels_ko': LABELS_KO,
                   'counts': {split: dict(Counter(r['label_name'] for r in rows)) for split, rows in all_rows.items()},
                   'source_counts': {split: dict(Counter(r['source_dataset'] for r in rows)) for split, rows in all_rows.items()},
                   'original_regional_person_overlap': overlap,
                   'person_disjoint_verified': False,
                   'verified_regional_evaluation_only': True,
                   'seoul_evaluation_available': False,
                   'unverified_identity_records': {'train': len(selected_new), 'valid': 0, 'test': 0},
                   'manifest_sha256': h.hexdigest(),
                   'limitations': ['130 has no persistent person IDs. It is used for training only; no 130 validation/test split is invented.',
                                   'Seoul/Gyeonggi dialect label is a proxy for standard-like speech; it is not a phonetic standard-language certification.',
                                   'Only the old five regional classes can currently be assessed on verified held-out people.',
                                   'A full six-class accuracy or Seoul recall cannot be measured until independent identified Seoul speakers are supplied.',
                                   'This is a capped training pilot using an encoder already adapted on the original regional training set.']}
        write_json(manifests / 'summary.json', summary)
        write_json(args.out / 'status.json', {'status': 'prepared', 'stage': 'data_complete', 'summary': str(manifests / 'summary.json'), 'counts': summary['counts']})
        print('[six-class-data] ' + json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as error:
        write_json(args.out / 'status.json', {'status': 'failed', 'error': str(error), 'error_type': type(error).__name__})
        raise


if __name__ == '__main__':
    main()
