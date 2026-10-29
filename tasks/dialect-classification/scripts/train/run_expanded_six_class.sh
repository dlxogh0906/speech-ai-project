#!/usr/bin/env bash
set -euo pipefail
cd /home/btdevgpu02/project/speech-ai-project
export PYTHONPATH="/data/speech-ai/tools/dialect_compare_py38:/data/speech-ai/tools/hf_py38${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data/speech-ai/models/hf
export HF_MODULES_CACHE=/data/speech-ai/tmp/dialect_compare_20261011/hf_modules
export TMPDIR=/data/speech-ai/tmp/dialect_expanded_six_20261013
export LD_LIBRARY_PATH="/data/speech-ai/cuda11_shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" /data/speech-ai/out/dialect/expanded_six_class_20261013
exec 9>/data/speech-ai/out/dialect/expanded_six_class_20261013/pipeline.lock
flock -n 9 || { echo 'Expanded dialect pipeline already running'; exit 1; }
exec /home/btdevgpu02/miniconda3/envs/speech-ai/bin/python -u tasks/dialect-classification/scripts/train/run_expanded_six_class.py
