#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
source tools/env.sh
export PYTHONPATH="/data/speech-ai/tools/hf_py38:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
PROTO="$1"
OUT="$2"
mkdir -p "$OUT"
trap 'code=$?; echo "$code" > "$OUT/run.exit"' EXIT
for ARM in B0 B1 B2 B3; do
    "$PY" tasks/pronunciation-assessment/scripts/train/train_h0_intermediate.py \
        --protocol "$PROTO" --condition "$ARM" --out "$OUT/$ARM" > "$OUT/$ARM.train.log" 2>&1
    for STEP in 3000 4000; do
        "$PY" tasks/pronunciation-assessment/scripts/eval/eval_h0_intermediate.py \
            --protocol "$PROTO" --checkpoint "$OUT/$ARM/step_$STEP.pt" \
            --out "$OUT/${ARM}_eval_$STEP" > "$OUT/${ARM}_eval_$STEP.log" 2>&1
    done
done
"$PY" tasks/pronunciation-assessment/scripts/eval/summarize_h0_intermediate.py \
    --protocol "$PROTO" --experiments "$OUT" > "$OUT/comparison.log" 2>&1
