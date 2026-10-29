#!/usr/bin/env bash
set -euo pipefail
cd /home/btdevgpu02/project/speech-ai-project
mkdir -p /data/speech-ai/out/dialect/six_class_20261012
mkdir -p /data/speech-ai/tmp/dialect_six_class_20261012
export PYTHONPATH="/data/speech-ai/tools/dialect_compare_py38:/data/speech-ai/tools/hf_py38${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data/speech-ai/models/hf
export HF_MODULES_CACHE=/data/speech-ai/tmp/dialect_compare_20261011/hf_modules
export TMPDIR=/data/speech-ai/tmp/dialect_six_class_20261012
export LD_LIBRARY_PATH="/data/speech-ai/cuda11_shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
exec 9>/data/speech-ai/out/dialect/six_class_20261012/pipeline.lock
flock -n 9 || exit 0
/home/btdevgpu02/miniconda3/envs/speech-ai/bin/python -u - <<'PY'
from pathlib import Path
import datetime
import json
import os
import subprocess
import time

root = Path('/data/speech-ai/out/dialect/six_class_20261012')
def status(stage, **extra):
    p = root / 'pipeline_status.json'
    q = p.with_suffix('.tmp.json')
    q.write_text(json.dumps({'stage': stage, 'updated_at': datetime.datetime.now().astimezone().isoformat(), **extra}, indent=2))
    os.replace(q, p)
status('waiting_for_data', status='running')
while True:
    try:
        prepared = json.loads((root / 'status.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        prepared = {}
    if prepared.get('status') == 'failed':
        status('data_failed', status='failed', error=prepared)
        raise RuntimeError('Data preparation failed; inspect preparation.log')
    if prepared.get('status') == 'prepared':
        break
    time.sleep(5)
status('six_class_eat_training', status='running')
command = ['/home/btdevgpu02/miniconda3/envs/speech-ai/bin/python', '-u',
           'tasks/dialect-classification/scripts/train/train_six_class_dialect_pilot.py']
if not (root / 'eat_large/report.json').is_file():
    with (root / 'training.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        status('training_failed', status='failed', returncode=result.returncode)
        raise RuntimeError('Six-class pilot failed; inspect training.log')
status('preserved_regional_head_training', status='running')
protected = command + ['--preserve-regional-head', '--feature-cache', str(root / 'eat_large/features'),
                       '--out', str(root / 'eat_preserved_head')]
if not (root / 'eat_preserved_head/report.json').is_file():
    with (root / 'preserved_head.log').open('w') as log:
        result = subprocess.run(protected, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        status('preserved_head_failed', status='failed', returncode=result.returncode)
        raise RuntimeError('Preserved regional head pilot failed; inspect preserved_head.log')
status('six_class_encoder_adaptation', status='running')
adapted = command + ['--preserve-regional-head', '--feature-cache', str(root / 'eat_large/features'),
                     '--out', str(root / 'eat_adapted_six'), '--adapt-epochs', '1']
if not (root / 'eat_adapted_six/report.json').is_file():
    with (root / 'encoder_adaptation.log').open('w') as log:
        result = subprocess.run(adapted, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        status('encoder_adaptation_failed', status='failed', returncode=result.returncode)
        raise RuntimeError('Encoder adaptation failed; inspect encoder_adaptation.log')
status('complete', status='complete', report=str(root / 'eat_adapted_six/report.html'))
PY
