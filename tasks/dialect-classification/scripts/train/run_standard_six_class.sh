#!/usr/bin/env bash
set -euo pipefail
cd /home/btdevgpu02/project/speech-ai-project
export PYTHONPATH="/data/speech-ai/tools/dialect_compare_py38:/data/speech-ai/tools/hf_py38${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data/speech-ai/models/hf
export HF_MODULES_CACHE=/data/speech-ai/tmp/dialect_compare_20261011/hf_modules
export TMPDIR=/data/speech-ai/tmp/dialect_standard_six_20261013
export LD_LIBRARY_PATH="/data/speech-ai/cuda11_shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" /data/speech-ai/out/dialect/standard_six_class_20261013
exec 9>/data/speech-ai/out/dialect/standard_six_class_20261013/pipeline.lock
flock -n 9 || exit 0
/home/btdevgpu02/miniconda3/envs/speech-ai/bin/python -u - <<'PY'
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/data/speech-ai/out/dialect/standard_six_class_20261013')
data = Path('/data/speech-ai/corpus/dialect_standard_speaker_20261013/archives/537')
def status(stage, **extra):
    p = root / 'pipeline_status.json'
    q = p.with_suffix('.tmp.json')
    q.write_text(json.dumps({'stage': stage, 'pid': os.getpid(), 'updated_at': datetime.datetime.now().astimezone().isoformat(), **extra}, ensure_ascii=False, indent=2))
    os.replace(q, p)
def run(command, filename):
    with (root / filename).open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        status('failed', status='failed', log=str(root / filename), returncode=result.returncode)
        raise RuntimeError('Pipeline command failed: ' + filename)
status('waiting_for_audio', status='running')
for key, name in [(55163, 'VS_common_01.zip'), (55154, 'TS_common_01.zip')]:
    while not (data / str(key) / 'download.json').is_file():
        time.sleep(10)
status('prepare_person_splits', status='running')
python = '/home/btdevgpu02/miniconda3/envs/speech-ai/bin/python'
if not (root / 'manifests/summary.json').is_file():
    run([python, '-u', 'tasks/dialect-classification/scripts/data/prepare_standard_six_class.py'], 'preparation.log')
status('train_and_evaluate', status='running')
if not (root / 'eat_large/report.json').is_file():
    run([python, '-u', 'tasks/dialect-classification/scripts/train/train_standard_six_class.py'], 'training.log')
status('complete', status='complete', report=str(root / 'eat_large/report.html'))
PY
