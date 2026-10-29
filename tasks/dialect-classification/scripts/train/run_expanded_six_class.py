"""Prepare an isolated expansion, preserve held-out CSVs, and train EAT."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from collections import Counter

REPO = Path(__file__).resolve().parents[4]
RUN = Path('/data/speech-ai/out/dialect/expanded_six_class_20261013')
BASE = Path('/data/speech-ai/out/dialect/standard_six_class_20261013')
DATA = Path('/data/speech-ai/corpus/dialect_expanded_six_20261013')
SOURCE = Path('/data/speech-ai/corpus/dialect_standard_speaker_20261013')
PYTHON = sys.executable


def write(path, value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def status(stage, **extra):
    write(RUN / 'pipeline_status.json', dict(stage=stage, pid=os.getpid(),
          updated_at=datetime.datetime.now().astimezone().isoformat(), **extra))
    print(stage, extra, flush=True)


def run(command, log):
    status(log.stem, status='running', log=str(log))
    with log.open('a') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=str(REPO))


def load(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    if not (DATA / 'archives').exists():
        (DATA / 'archives').symlink_to(SOURCE / 'archives', target_is_directory=True)
    config = {'experiment': 'expanded_six_class', 'target_train_per_class': 15000,
              'regional_record_policy': 'one window per recording, person-balanced, preserve baseline training rows',
              'baseline': str(BASE), 'heldout_policy': 'valid/test byte-identical to baseline; no selection from test',
              'model': 'EAT Large', 'head_epochs': 40, 'adapt_epochs': 5,
              'selection': 'validation Top1 then macro F1; early stop after 2 non-improving encoder epochs',
              'north': 'No additional North data; six-class improvement experiment first',
              'session': 'dialect-expand-20261013', 'data_root': str(DATA)}
    write(RUN / 'experiment_config.json', config)
    ready = RUN / 'prepared_verified.json'
    if not ready.exists():
        run([PYTHON, '-u', str(REPO / 'tasks/dialect-classification/scripts/data/prepare_standard_six_class.py'),
             '--root', str(DATA), '--out', str(RUN), '--train-per-class', '15000'], RUN / 'preparation.log')
        manifest = RUN / 'manifests'
        # Retain every original training row, then add prepared recordings.
        old = load(BASE / 'manifests/train.csv')
        expanded = load(manifest / 'train.csv')
        merged = []
        for label in range(6):
            values = [r for r in old if int(r['label_id']) == label]
            records = {r['record_key'] for r in values}
            for row in expanded:
                if int(row['label_id']) == label and row['record_key'] not in records and len(values) < 15000:
                    values.append(row); records.add(row['record_key'])
            merged.extend(values)
        with (manifest / 'train.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(old[0]))
            writer.writeheader(); writer.writerows(merged)
        for split in ['valid', 'test']:
            shutil.copyfile(BASE / 'manifests' / (split + '.csv'), manifest / (split + '.csv'))
        rows = {s: load(manifest / (s + '.csv')) for s in ['train', 'valid', 'test']}
        people = {s: {r['person_id'] for r in values} for s, values in rows.items()}
        records = {s: {r['record_key'] for r in values} for s, values in rows.items()}
        for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
            if people[a] & people[b] or records[a] & records[b]:
                raise RuntimeError('Person/recording overlap')
        if not {r['row_id'] for r in old} <= {r['row_id'] for r in rows['train']}:
            raise RuntimeError('Baseline training rows were lost')
        for split in ['valid', 'test']:
            if (manifest / (split + '.csv')).read_bytes() != (BASE / 'manifests' / (split + '.csv')).read_bytes():
                raise RuntimeError('Held-out split changed')
        missing = [r['audio_path'] for values in rows.values() for r in values if not Path(r['audio_path']).is_file()]
        if missing:
            raise RuntimeError('Missing selected audio: ' + str(missing[:3]))
        summary = json.loads((manifest / 'summary.json').read_text())
        labels = summary['labels']
        summary['counts'] = {s: dict(Counter(r['label_name'] for r in values)) for s, values in rows.items()}
        summary['people'] = {s: {label: len({r['person_id'] for r in values if r['label_name'] == label}) for label in labels} for s, values in rows.items()}
        summary['manifest_sha256'] = hashlib.sha256(b''.join((manifest / (s + '.csv')).read_bytes() for s in ['train', 'valid', 'test'])).hexdigest()
        summary['expansion'] = config
        summary['speech_hours'] = {s: {label: sum(float(r['duration_sec']) for r in values if r['label_name'] == label) / 3600 for label in labels} for s, values in rows.items()}
        write(manifest / 'summary.json', summary)
        write(ready, {'verified': True, 'summary': summary, 'heldout_byte_identical': True, 'baseline_training_retained': True})
        status('data_verified', status='running', counts=summary['counts'])
    if not (RUN / 'eat_large/report.json').exists():
        run([PYTHON, '-u', str(REPO / 'tasks/dialect-classification/scripts/train/train_standard_six_class.py'),
             '--manifests', str(RUN / 'manifests'), '--out', str(RUN / 'eat_large'),
             '--adapt-epochs', '5'], RUN / 'training.log')
    baseline = json.loads((BASE / 'eat_large/report.json').read_text())
    candidate = json.loads((RUN / 'eat_large/report.json').read_text())
    comparison = {'baseline_test_top1': baseline['test']['top1'], 'candidate_test_top1': candidate['test']['top1'],
                  'delta_percentage_points': 100 * (candidate['test']['top1'] - baseline['test']['top1']),
                  'baseline_valid_top1': baseline['valid']['top1'], 'candidate_valid_top1': candidate['valid']['top1'],
                  'selection': candidate['selected_candidate'], 'per_class': candidate['test']['per_class'],
                  'no_deployment': True, 'north_generalization_not_measured': True}
    write(RUN / 'comparison.json', comparison)
    status('complete', status='complete', **comparison)


if __name__ == '__main__':
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except Exception as exc:
        status('failed', status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
