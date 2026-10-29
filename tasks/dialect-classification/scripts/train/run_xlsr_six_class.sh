#!/usr/bin/env bash
set -euo pipefail
cd /home/btdevgpu02/project/speech-ai-project
export PYTHONPATH="/data/speech-ai/tools/dialect_compare_py38:/data/speech-ai/tools/hf_py38${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data/speech-ai/models/hf
export HF_MODULES_CACHE=/data/speech-ai/tmp/dialect_compare_20261011/hf_modules
export TMPDIR=/data/speech-ai/tmp/dialect_xlsr_six_20261014
export LD_LIBRARY_PATH="/data/speech-ai/cuda11_shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" /data/speech-ai/out/dialect/xlsr_six_class_20261014
exec 9>/data/speech-ai/out/dialect/xlsr_six_class_20261014/pipeline.lock
flock -n 9 || { echo 'XLSR six-class experiment is already running'; exit 1; }
record_exit() {
    local experiment_exit_code=$?
    if [ "$experiment_exit_code" -ne 0 ]; then
        DIALECT_EXPERIMENT_EXIT_CODE="$experiment_exit_code" python3 - <<'PY'
import datetime
import json
import os
from pathlib import Path
root = Path('/data/speech-ai/out/dialect/xlsr_six_class_20261014')
path = root / 'pipeline_status.json'
value = json.loads(path.read_text()) if path.exists() else {}
if os.environ.get('DIALECT_EXPERIMENT_STAGE') == 'extraction':
    extraction = root / 'extraction_status.json'
    if extraction.exists():
        value.update(json.loads(extraction.read_text()))
value.update(status='failed', process_exit_code=int(os.environ['DIALECT_EXPERIMENT_EXIT_CODE']),
             updated_at=datetime.datetime.now().astimezone().isoformat())
value.setdefault('error', 'Pipeline process exited before completion')
temporary = path.with_suffix('.tmp.json')
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
os.replace(temporary, path)
PY
    fi
    exit "$experiment_exit_code"
}
trap record_exit EXIT
export DIALECT_EXPERIMENT_STAGE=extraction
/home/btdevgpu02/miniconda3/envs/speech-ai/bin/python -u tasks/dialect-classification/scripts/train/extract_xlsr_six_class.py
export DIALECT_EXPERIMENT_STAGE=training
/home/btdevgpu02/miniconda3/envs/speech-ai/bin/python -u tasks/dialect-classification/scripts/train/train_xlsr_six_class.py
