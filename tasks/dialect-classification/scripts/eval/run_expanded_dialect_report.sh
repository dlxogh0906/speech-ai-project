#!/usr/bin/env bash
set -euo pipefail
cd /home/btdevgpu02/project/speech-ai-project
export PYTHONPATH="/data/speech-ai/tools/dialect_compare_py38:/data/speech-ai/tools/hf_py38${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data/speech-ai/models/hf HF_MODULES_CACHE=/data/speech-ai/tmp/dialect_compare_20261011/hf_modules
export LD_LIBRARY_PATH="/data/speech-ai/cuda11_shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
exec /home/btdevgpu02/miniconda3/envs/speech-ai/bin/python -u tasks/dialect-classification/scripts/eval/report_expanded_dialect.py
