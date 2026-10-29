"""Prepare natural-answer dialect audio with provider global speaker IDs.

AI Hub 71517/71558 document speakerId as speaker[region][serial]. The old
recording-only dataset is deliberately not mixed into this person evaluation.
Original Training/Validation recordings are resplit by connected speaker groups.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from download_aihub_dialect import download, write_json
from prepare_dialect_subset import LABELS, LABELS_KO, FIELDS, windows, zip_name
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

FIELDS = FIELDS + ['person_id']
PROVINCES = {'gw': 0, 'gs': 1, 'gb': 1, 'gn': 1, 'jl': 2, 'jr': 2, 'jb': 2, 'jn': 2, 'jj': 3, 'cc': 4, 'cb': 4, 'cn': 4}


def person_id(value):
    match = re.fullmatch(r'speaker([a-z]+)(\d+)', str(value).strip().lower())
    if not match:
        raise RuntimeError('Unrecognized provider global speaker ID format')
    key = 'aihub-2022-dialect:' + match[1] + ':' + str(int(match[2]))
    return hashlib.sha256(key.encode()).hexdigest()


def seconds(value):
    if isinstance(value, (int, float)):
        return float(value)
    parts = [float(p) for p in str(value).split(':')]
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def plan_jobs(datasets, catalog):
    jobs = []
    for dataset in datasets:
        files = json.loads((catalog / ('%s-files.json' % dataset)).read_text())['list']
        by_path = {f['fileStreCours']: f for f in files}
        for info in files:
            path = info['fileStreCours']
            if '/01.원천데이터/' not in path or '1인발화 질문에답하기' not in path:
                continue
            label_path = path.replace('/01.원천데이터/', '/02.라벨링데이터/').replace('TS_', 'TL_').replace('VS_', 'VL_')
            if label_path not in by_path:
                raise RuntimeError('Missing paired source labels')
            jobs.append({'dataset': dataset, 'audio': info, 'labels': by_path[label_path],
                         'original_split': 'Validation' if '/Validation/' in path else 'Training'})
    return sorted(jobs, key=lambda j: (j['original_split'] != 'Validation', int(j['audio']['fileSize'])))


def index_labels(connection, job, archive):
    key = int(job['labels']['fileSn'])
    if connection.execute('SELECT 1 FROM imported WHERE filekey=?', (key,)).fetchone():
        return
    count = 0
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if not info.filename.lower().endswith('.json'):
                continue
            record = json.loads(bundle.read(info).decode('utf-8-sig'))
            rid = Path(record['fileName']).stem
            if rid != Path(zip_name(info)).stem:
                raise RuntimeError('Source audio and label filenames disagree')
            speakers = {}
            for speaker in record['speaker']:
                source_id = str(speaker['speakerId'])
                pid = person_id(source_id)
                label = PROVINCES.get(str(speaker.get('residenceProvince', '')).lower())
                speakers[source_id] = {'person_id': pid, 'label_id': label}
                profile = json.dumps({k: speaker.get(k) for k in ['gender', 'birthYear']}, sort_keys=True)
                connection.execute('INSERT OR IGNORE INTO profiles VALUES (?,?)', (pid, hashlib.sha256(profile.encode()).hexdigest()))
            utterances = []
            for number, sentence in enumerate(record['transcription']['sentences']):
                start, end = seconds(sentence['startTime']), seconds(sentence['endTime'])
                if not (math.isfinite(start) and math.isfinite(end) and end > start >= 0):
                    continue
                utterances.append([number, str(sentence['speakerId']), start, end])
            topic = str(record.get('script', {}).get('domain', ''))
            connection.execute('INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?)',
                               (job['dataset'], rid, job['original_split'], topic, json.dumps(speakers), json.dumps(utterances), key))
            count += 1
    if not count:
        raise RuntimeError('No recognized label records')
    connection.execute('INSERT INTO imported VALUES (?)', (key,))
    connection.commit()
    print('[speaker labels] dataset=%s file=%s records=%s' % (job['dataset'], key, count), flush=True)


def convert(job, args, connection):
    root = Path(args.root)
    folder = root / 'archives' / str(job['dataset']) / str(job['audio']['fileSn'])
    marker, rows_path = folder / 'processed.json', folder / 'rows.csv'
    if marker.exists() and rows_path.exists():
        return rows_path
    label_archive = download(job['dataset'], job['labels'], root / 'archives', args.reserve_gib)
    index_labels(connection, job, label_archive)
    source_archive = download(job['dataset'], job['audio'], root / 'archives', args.reserve_gib)
    audio_root = root / 'audio' / str(job['dataset'])
    audio_root.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    started = time.time()
    temp_rows = rows_path.with_suffix('.csv.tmp')
    with zipfile.ZipFile(source_archive) as bundle, temp_rows.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for info in bundle.infolist():
            if not info.filename.lower().endswith('.wav'):
                continue
            stats['source_recordings'] += 1
            rid = Path(zip_name(info)).stem
            row = connection.execute('SELECT original_split,topic,speakers,utterances FROM records WHERE dataset=? AND record_id=?', (job['dataset'], rid)).fetchone()
            if row is None:
                raise RuntimeError('No paired source label for audio recording: ' + rid)
            original_split, topic, speakers, utterances = row
            speakers, utterances = json.loads(speakers), json.loads(utterances)
            if shutil.disk_usage(str(audio_root)).free < args.reserve_gib * 2 ** 30:
                raise RuntimeError('Disk reserve reached')
            raw = bundle.read(info)  # Reading full members verifies their source ZIP CRC.
            wave, rate = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
            del raw
            wave = wave.mean(axis=1)
            if rate != 16000:
                divisor = math.gcd(int(rate), 16000)
                wave = resample_poly(wave, 16000 // divisor, int(rate) // divisor).astype(np.float32)
            duration = len(wave) / 16000
            clips = windows(utterances, speakers, duration)
            if not clips:
                stats['no_usable_windows'] += 1
                continue
            path = audio_root / (rid + '.flac')
            temp = path.with_name(path.stem + '.tmp.flac')
            sf.write(str(temp), wave, 16000, format='FLAC', subtype='PCM_16')
            del wave
            saved = sf.info(str(temp))
            if saved.samplerate != 16000 or saved.channels != 1 or abs(saved.duration - duration) > 1 / 16000:
                raise RuntimeError('Converted audio metadata mismatch')
            os.replace(temp, path)
            key = str(job['dataset']) + ':' + rid
            for segment, sid, start, length, label in clips:
                pid = speakers[sid]['person_id']
                writer.writerow({'row_id': key + ':' + sid + ':' + segment, 'audio_path': str(path),
                                 'start_sec': '%.6f' % start, 'duration_sec': '%.6f' % length,
                                 'label_id': label, 'label_name': LABELS[label], 'speaker_key': key + ':' + sid,
                                 'record_key': key, 'topic': topic, 'original_split': original_split, 'person_id': pid})
                stats['windows'] += 1
                stats['speech_seconds'] += length
            stats['converted_recordings'] += 1
            if stats['converted_recordings'] % 200 == 0:
                print('[speaker audio] dataset=%s file=%s %s' % (job['dataset'], job['audio']['fileSn'], json.dumps(dict(stats))), flush=True)
    if not stats['converted_recordings']:
        raise RuntimeError('No usable globally identified dialect audio')
    os.replace(temp_rows, rows_path)
    write_json(marker, {'statistics': dict(stats), 'elapsed_seconds': time.time() - started,
                        'source_archive': str(source_archive), 'labels_archive': str(label_archive)})
    source_archive.unlink()  # Only this job's newly downloaded, verified source ZIP.
    return rows_path


def assemble(args, connection, manifests):
    parent = {}
    labels_by_person = collections.defaultdict(set)
    def find(value):
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]
    for (raw,) in connection.execute('SELECT speakers FROM records'):
        speakers = json.loads(raw)
        ids = sorted({speaker['person_id'] for speaker in speakers.values()})
        for speaker in speakers.values():
            if speaker['label_id'] is not None:
                labels_by_person[speaker['person_id']].add(speaker['label_id'])
        for pid in ids[1:]:
            a, b = find(ids[0]), find(pid)
            parent[max(a, b)] = min(a, b)
        if ids:
            find(ids[0])
    ambiguous = {row[0] for row in connection.execute('SELECT person_id FROM profiles GROUP BY person_id HAVING COUNT(*)>1')}
    ambiguous.update(pid for pid, labels in labels_by_person.items() if len(labels) > 1)
    component_keys = collections.defaultdict(list)
    for pid in list(parent):
        component_keys[find(pid)].append(pid)
    def split_for(pid):
        group = component_keys[find(pid)]
        value = int(hashlib.sha256((str(args.seed) + ':' + min(group)).encode()).hexdigest()[:8], 16) % 1000
        return 'train' if value < 800 else 'valid' if value < 900 else 'test'
    out = Path(args.out) / 'manifests'
    out.mkdir(parents=True, exist_ok=True)
    counts, durations, people, records = {}, {}, {}, {}
    handles = {s: (out / (s + '.csv.tmp')).open('w', newline='') for s in ['train', 'valid', 'test']}
    writers = {s: csv.DictWriter(h, fieldnames=FIELDS) for s, h in handles.items()}
    seen, excluded = set(), 0
    for split, writer in writers.items():
        writer.writeheader()
        counts[split], durations[split], people[split], records[split] = collections.Counter(), collections.Counter(), set(), set()
    try:
        for path in manifests:
            with path.open(newline='') as source:
                for row in csv.DictReader(source):
                    if row['row_id'] in seen:
                        continue
                    seen.add(row['row_id'])
                    if row['person_id'] in ambiguous:
                        excluded += 1
                        continue
                    split = split_for(row['person_id'])
                    writers[split].writerow(row)
                    people[split].add(row['person_id']); records[split].add(row['record_key'])
                    counts[split][row['label_name']] += 1
                    durations[split][row['label_name']] += float(row['duration_sec'])
    finally:
        for handle in handles.values():
            handle.close()
    overlap = {a + '_' + b: len(people[a] & people[b]) for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]}
    if any(overlap.values()) or any(records[a] & records[b] for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]):
        raise RuntimeError('Person or recording leakage after grouping')
    for split in handles:
        os.replace(out / (split + '.csv.tmp'), out / (split + '.csv'))
    report = {'labels': LABELS, 'labels_ko': LABELS_KO, 'counts': {s: dict(c) for s, c in counts.items()},
              'speech_hours': {s: {k: round(v / 3600, 3) for k, v in d.items()} for s, d in durations.items()},
              'recordings': {s: len(v) for s, v in records.items()}, 'people': {s: len(v) for s, v in people.items()},
              'person_overlap': overlap, 'person_disjoint_verified': True,
              'person_id_source': 'AI Hub 71517/71558 speaker[].speakerId; documented global speaker[region][serial] scheme, canonicalized and hashed',
              'split_rule': 'Connected components of all annotated recording participants; SHA256(seed:minimum_person_id), 80/10/10',
              'label_source': 'speaker.residenceProvince', 'ambiguous_person_ids_excluded': len(ambiguous), 'excluded_windows': excluded,
              'seed': args.seed, 'datasets_prepared': args.datasets,
              'limitations': ['Source participant IDs establish this split; different IDs assigned to the same actual person cannot be excluded independently.',
                              'Middle/older-age dialect corpus and spontaneous answers only; results do not establish all-age performance.',
                              'Official Training/Validation audio is resplit by people, not an untouched official benchmark.',
                              'Old recording-only dialect data is not included in classifier training or evaluation.']}
    combined = hashlib.sha256()
    report['manifest_file_sha256'] = {}
    for split in handles:
        hashed = hashlib.sha256()
        with (out / (split + '.csv')).open('rb') as source:
            for chunk in iter(lambda: source.read(4 * 1024 ** 2), b''):
                hashed.update(chunk); combined.update(chunk)
        report['manifest_file_sha256'][split] = hashed.hexdigest()
    report['manifest_sha256'] = combined.hexdigest()
    write_json(out / 'summary.json', report)
    print('[person manifests] ' + json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=int, nargs='+', default=[71558])
    parser.add_argument('--catalog', default='/data/speech-ai/tmp/aihub-dialect-catalog')
    parser.add_argument('--root', default='/data/speech-ai/corpus/dialect_speakerid_20261011')
    parser.add_argument('--out', default='/data/speech-ai/out/dialect/speakerid_20261011')
    parser.add_argument('--reserve-gib', type=float, default=40)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(args.datasets, Path(args.catalog))
    write_json(out / 'download_plan.json', {'datasets': args.datasets, 'speech_type': 'one-person spontaneous answers', 'jobs': jobs})
    connection = sqlite3.connect(str(out / 'labels.sqlite'))
    connection.execute('CREATE TABLE IF NOT EXISTS records(dataset INTEGER,record_id TEXT,original_split TEXT,topic TEXT,speakers TEXT,utterances TEXT,filekey INTEGER,PRIMARY KEY(dataset,record_id))')
    connection.execute('CREATE TABLE IF NOT EXISTS profiles(person_id TEXT,profile_sha256 TEXT,PRIMARY KEY(person_id,profile_sha256))')
    connection.execute('CREATE TABLE IF NOT EXISTS imported(filekey INTEGER PRIMARY KEY)')
    write_json(out / 'status.json', {'status': 'preparing', 'datasets': args.datasets, 'five_region_classifier_ready': False})
    try:
        manifests = [convert(job, args, connection) for job in jobs]
        assemble(args, connection, manifests)
        summary = json.loads((out / 'manifests/summary.json').read_text())
        ready = all(all(summary['counts'][split].get(label, 0) > 0 for label in LABELS) for split in ['train', 'valid', 'test'])
        write_json(out / 'status.json', {'status': 'data_complete', 'datasets': args.datasets, 'five_region_classifier_ready': ready,
                                       'next_step': 'Compare encoders on the new verified-ID split' if ready else '71517 access and preparation needed for Gangwon/Gyeongsang'})
    except Exception as exc:
        write_json(out / 'status.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc), 'five_region_classifier_ready': False})
        raise


if __name__ == '__main__':
    main()
