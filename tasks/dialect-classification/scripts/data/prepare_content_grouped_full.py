"""Expand training windows while retaining the audited content/person split."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main(args):
    out = args.out / 'manifests'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'summary.json').exists(), 'Use a new output directory'
    original = args.base / 'manifests'
    summary = json.loads((original / 'summary.json').read_text())
    assert summary['conflicting_label_components_excluded'] == 0
    components = json.loads((args.base / 'identity_components.json').read_text())
    canonical = {p: group for group, people in components.items() for p in people}
    fingerprint = {}
    with (args.base / 'pcm_fingerprints.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            fingerprint[row['path']] = row['pcm16_fingerprint']
    base_rows = read(original / 'train.csv')
    selected_path = {}
    for path, content in sorted(fingerprint.items()):
        selected_path.setdefault(content, path)
    for row in base_rows:
        selected_path[row['pcm16_fingerprint']] = row['audio_path']

    def assigned_train(group):
        key = summary['seed'] + ':split:' + group
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 1000 < 800

    train = {r['row_id']: r for r in base_rows}
    window_keys = {(r['pcm16_fingerprint'], round(float(r['start_sec']) * 16000),
                    round(float(r['duration_sec']) * 16000)) for r in base_rows}
    discarded_duplicates = 0
    for filename, expected in summary['source_manifest_sha256'].items():
        path = Path(filename)
        assert digest(path) == expected, 'Source manifest changed: ' + filename
        print('Loading ' + filename, flush=True)
        with path.open() as stream:
            for row in csv.DictReader(stream):
                group = canonical.get(row['person_id'], row['person_id'])
                if not assigned_train(group) or row['row_id'] in train:
                    continue
                content = fingerprint[row['audio_path']]
                if row['audio_path'] != selected_path[content]:
                    discarded_duplicates += 1
                    continue
                key = (content, round(float(row['start_sec']) * 16000),
                       round(float(row['duration_sec']) * 16000))
                if key in window_keys:
                    discarded_duplicates += 1
                    continue
                window_keys.add(key)
                row.update(original_person_id=row['person_id'], person_id=group,
                           evaluation_group_id=group, pcm16_fingerprint=content,
                           source_dataset=row.get('source_dataset') or row['record_key'].split(':')[0])
                train[row['row_id']] = row
    assert {r['row_id'] for r in base_rows} <= set(train)
    assert all(assigned_train(r['person_id']) for r in train.values())
    selected = {'train': list(train.values())}
    fields = sorted(set().union(*(set(r) for r in selected['train'])))
    with (out / 'train.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected['train'])
    heldout_hashes = {}
    for split in ['valid', 'test']:
        shutil.copyfile(original / (split + '.csv'), out / (split + '.csv'))
        assert digest(original / (split + '.csv')) == digest(out / (split + '.csv'))
        heldout_hashes[split] = digest(out / (split + '.csv'))
        selected[split] = read(out / (split + '.csv'))
    overlaps = {}
    for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        overlaps[a + '_' + b] = {}
        for key in ['person_id', 'original_person_id', 'record_key', 'audio_path', 'pcm16_fingerprint']:
            n = len({r[key] for r in selected[a]} & {r[key] for r in selected[b]})
            assert n == 0, (a, b, key, n)
            overlaps[a + '_' + b][key] = n
    counts = {}
    for split, rows in selected.items():
        counts[split] = {}
        for label in summary['labels']:
            values = [r for r in rows if r['label_name'] == label]
            counts[split][label] = dict(rows=len(values), groups=len({r['person_id'] for r in values}),
                                       recordings=len({r['pcm16_fingerprint'] for r in values}),
                                       hours=sum(float(r['duration_sec']) for r in values) / 3600)
    h = hashlib.sha256()
    for split in ['train', 'valid', 'test']:
        with (out / (split + '.csv')).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
    summary.update(created_at=datetime.now().astimezone().isoformat(), base_manifest_root=str(original),
                   base_manifest_sha256=summary['manifest_sha256'], manifest_sha256=h.hexdigest(),
                   fixed_heldout_sha256=heldout_hashes, counts=counts, overlaps=overlaps,
                   retained_baseline_training_rows=len(base_rows), discarded_duplicate_rows=discarded_duplicates,
                   selection_policy='All existing eligible windows in fixed training components; one canonical file per identical decoded recording; exact window deduplication. Heldouts copied byte-for-byte.')
    write(out / 'summary.json', summary)
    write(args.out / 'preparation_status.json', dict(status='complete', counts=counts, overlaps=overlaps))
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
