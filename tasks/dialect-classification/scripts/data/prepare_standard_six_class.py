"""Add AI Hub 537 standard-dialect recordings to the existing person split.

Provider person numbers are cross-checked against folders and filenames. The
official archive split is discarded: it shares people. AI Hub 130 is excluded.
Dialect 01 is treated as the standard-dialect candidate, NOT as residence.
The public example and corpus distribution support this interpretation; the
separate downloadable codebook is still unavailable without a web login.
"""
import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import zipfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

LABELS = ['Gangwon', 'Gyeongsang', 'Jeolla', 'Jeju', 'Chungcheong', 'Standard']
FIELDS = ['row_id', 'audio_path', 'start_sec', 'duration_sec', 'label_id',
          'label_name', 'speaker_key', 'record_key', 'topic', 'original_split',
          'person_id', 'source_dataset', 'source_person_id', 'source_dialect_code',
          'transcript', 'source_audio_sha256']


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def balanced_records(rows, cap):
    # One window per original recording, then rotate between people.
    people, records = defaultdict(list), set()
    for row in sorted(rows, key=lambda r: digest(r['row_id'])):
        if row['record_key'] in records:
            continue
        records.add(row['record_key'])
        people[row['person_id']].append(row)
    queues = [deque(people[p]) for p in sorted(people, key=digest)]
    chosen = []
    while queues and len(chosen) < cap:
        remaining = []
        for queue in queues:
            if len(chosen) >= cap:
                break
            chosen.append(queue.popleft())
            if queue:
                remaining.append(queue)
        queues = remaining
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/speech-ai/corpus/dialect_standard_speaker_20261013'))
    parser.add_argument('--out', type=Path, default=Path('/data/speech-ai/out/dialect/standard_six_class_20261013'))
    parser.add_argument('--old', type=Path, default=Path('/data/speech-ai/out/dialect/speakerid_20261011/manifests'))
    parser.add_argument('--train-per-class', type=int, default=5000)
    parser.add_argument('--valid-per-class', type=int, default=600)
    parser.add_argument('--test-per-class', type=int, default=1000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'data_progress.json', {'status': 'running', 'stage': 'audit_labels', 'pid': os.getpid()})
    profiles, records, source_people = defaultdict(set), {}, {}
    counts, excluded = Counter(), Counter()
    for key, filename in [(55147, 'TL_common_01.zip'), (55159, 'VL_common_01.zip')]:
        archive = args.root / 'archives' / '537' / str(key) / filename
        people = set()
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                if not name.endswith('.json'):
                    continue
                j = json.loads(bundle.read(name))
                sid, profile = str(j['Basic']['NumberOfSpeaker']), j['Speaker']
                wavname = j['File']['FileName']
                if Path(name).parts[-2] != sid or not wavname.split('-')[1].startswith(sid):
                    raise RuntimeError('Provider person number does not match paths')
                people.add(sid)
                profiles[sid].add(tuple(str(profile[k]) for k in ['SpeakerName', 'Gender', 'Age', 'Dialect', 'Region']))
                counts[profile['Dialect']] += 1
                if profile['Dialect'] != '01' or j['Other']['QualityStatus'] != 'Good':
                    excluded['nonstandard_or_quality'] += 1
                    continue
                start, end = float(j['Other']['SpeechStart']), float(j['Other']['SpeechEnd'])
                if not math.isfinite(start + end) or start < 0 or end - start < 3:
                    excluded['short_or_invalid_speech'] += 1
                    continue
                pid = digest('aihub537:person:' + sid)
                partition = int(digest('standard-six:42:' + sid)[:8], 16) / 2**32
                split = 'train' if partition < .8 else 'valid' if partition < .9 else 'test'
                rid = '537:' + Path(wavname).stem
                length = min(end - start, 12.)
                row = {'row_id': rid, 'audio_path': str(args.root / 'audio' / (Path(wavname).stem + '.flac')),
                       'start_sec': 0., 'duration_sec': length, 'label_id': 5, 'label_name': LABELS[5],
                       'speaker_key': pid, 'record_key': rid, 'topic': 'commonSentence',
                       'original_split': 'Training' if key == 55147 else 'Validation', 'person_id': pid,
                       'source_dataset': '537', 'source_person_id': sid, 'source_dialect_code': '01',
                       'transcript': j['Transcription']['LabelText'], '_wavname': wavname,
                       '_speech_start': start + (end - start - length) / 2, '_split': split}
                if rid in records:
                    previous = records[rid]
                    if previous['person_id'] != pid or previous['transcript'] != row['transcript']:
                        raise RuntimeError('Contradictory duplicate recording')
                    excluded['duplicate_recording'] += 1
                    continue
                records[rid] = row
        source_people[key] = people
        print('[standard-labels] file=%s people=%s eligible_records=%s' % (key, len(people), len(records)), flush=True)
    conflicts = {sid for sid, values in profiles.items() if len(values) != 1}
    if conflicts:
        raise RuntimeError('Conflicting provider person profiles: %d' % len(conflicts))
    caps = {'train': args.train_per_class, 'valid': args.valid_per_class, 'test': args.test_per_class}
    selected = {split: balanced_records([r for r in records.values() if r['_split'] == split], cap)
                for split, cap in caps.items()}
    wanted = {r['_wavname']: r for values in selected.values() for r in values}
    (args.root / 'audio').mkdir(exist_ok=True)
    converted = set()
    for key, filename in [(55163, 'VS_common_01.zip'), (55154, 'TS_common_01.zip')]:
        path = args.root / 'archives' / '537' / str(key) / filename
        if not path.is_file():
            write_json(args.out / 'data_progress.json', {'status': 'waiting', 'stage': 'audio_archive', 'filekey': key})
            raise RuntimeError('Required audio download is not finished: ' + str(path))
        with zipfile.ZipFile(path) as bundle:
            for name in bundle.namelist():
                wavname = Path(name).name
                if wavname not in wanted or wavname in converted:
                    continue
                row = wanted[wavname]
                raw = bundle.read(name)
                wave, rate = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
                wave = wave.mean(axis=1)
                begin = round(row['_speech_start'] * rate)
                length = round(float(row['duration_sec']) * rate)
                if begin + length > len(wave) + round(.05 * rate):
                    raise RuntimeError('Speech interval exceeds waveform')
                wave = wave[begin:begin + length]
                if rate != 16000:
                    factor = math.gcd(rate, 16000)
                    wave = resample_poly(wave, 16000 // factor, rate // factor).astype(np.float32)
                if len(wave) < 3 * 16000 or not np.isfinite(wave).all():
                    raise RuntimeError('Selected waveform is invalid')
                row['duration_sec'] = len(wave) / 16000
                row['source_audio_sha256'] = hashlib.sha256(raw).hexdigest()
                sf.write(row['audio_path'], wave, 16000, subtype='PCM_16')
                converted.add(wavname)
                if len(converted) % 500 == 0:
                    print('[standard-audio] %s/%s' % (len(converted), len(wanted)), flush=True)
        print('[standard-archive] file=%s converted=%s' % (key, len(converted)), flush=True)
    if converted != set(wanted):
        raise RuntimeError('Missing selected audio: %d' % len(set(wanted) - converted))
    old_summary = json.loads((args.old / 'summary.json').read_text())
    if not old_summary['person_disjoint_verified']:
        raise RuntimeError('Existing regional split is unverified')
    original_people = {}
    for split, cap in caps.items():
        with (args.old / (split + '.csv')).open(newline='') as stream:
            regional = list(csv.DictReader(stream))
        original_people[split] = {r['person_id'] for r in regional}
        for label in range(5):
            chosen = balanced_records([r for r in regional if int(r['label_id']) == label], cap)
            for r in chosen:
                r['source_dataset'] = r['record_key'].split(':')[0]
            selected[split].extend(chosen)
        selected[split].sort(key=lambda r: (int(r['label_id']), digest(r['row_id'])))
    root = args.out / 'manifests'
    root.mkdir(exist_ok=True)
    people = {s: {r['person_id'] for r in rows} for s, rows in selected.items()}
    recording = {s: {r['record_key'] for r in rows} for s, rows in selected.items()}
    for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        if people[a] & people[b] or recording[a] & recording[b]:
            raise RuntimeError('Person or recording overlap')
        if original_people[a] & original_people[b]:
            raise RuntimeError('Existing regional person split changed')
    for split, rows in selected.items():
        if {int(r['label_id']) for r in rows} != set(range(6)):
            raise RuntimeError('A class is absent from ' + split)
        with (root / (split + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore')
            writer.writeheader(); writer.writerows(rows)
    manifest_hash = hashlib.sha256()
    for split in caps:
        manifest_hash.update((root / (split + '.csv')).read_bytes())
    summary = {'labels': LABELS, 'counts': {s: dict(Counter(r['label_name'] for r in rows)) for s, rows in selected.items()},
               'people': {s: {label: len({r['person_id'] for r in rows if r['label_name'] == label}) for label in LABELS} for s, rows in selected.items()},
               'source_person_disjoint_verified': True, 'person_overlap': {'train_valid': 0, 'train_test': 0, 'valid_test': 0},
               'person_id_source': '537 Basic.NumberOfSpeaker, checked against folder/filename and profiles; original 71517/71558 speakerId split retained.',
               'original_regional_manifest_sha256': old_summary['manifest_sha256'], 'manifest_sha256': manifest_hash.hexdigest(),
               'standard_available_all_splits': True, 'standard_label_source': '537 Speaker.Dialect == 01; residence is not the class label.',
               'standard_label_mapping_status': 'Candidate mapping: public standard-speech example and observed 01/02 binary distribution; downloadable codebook has not been obtained.',
               'official_537_train_valid_person_overlap': len(source_people[55147] & source_people[55159]),
               'source_537_records': len(records), 'source_537_dialect_counts': dict(counts), 'excluded': dict(excluded),
               'limitations': ['Source person identifiers do not independently rule out people registered under different IDs or across providers.',
                              'Standard means spoken standard dialect, not birthplace or current residence.',
                              '537 short read commands versus original spontaneous regional speech may introduce corpus/style bias.',
                              'Dialect 01 semantic mapping remains provisional until the provider codebook is obtained.',
                              'AI Hub 130 and 71349 are excluded from classifier training and scoring.']}
    write_json(root / 'summary.json', summary)
    write_json(args.out / 'data_progress.json', {'status': 'complete', 'stage': 'prepared', 'summary': summary})
    print('[standard-ready] ' + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
