#!/usr/bin/env bash
set -euo pipefail
source tools/env.sh
export PYTHONPATH="${2:?existing Transformers package directory}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
p4_out="${1:?output directory}"
trap 'code=$?; echo "$code" > "$p4_out/extract.exit"' EXIT
for p4_arm in B0 B3; do
  "$PY" tasks/pronunciation-assessment/scripts/eval/diagnose_h0_p4.py --stage extract \
    --out "$p4_out" --arm "$p4_arm" --smoke > "$p4_out/smoke_${p4_arm}.log" 2>&1
done
for p4_arm in B0 B3; do
  "$PY" tasks/pronunciation-assessment/scripts/eval/diagnose_h0_p4.py --stage extract \
    --out "$p4_out" --arm "$p4_arm" > "$p4_out/${p4_arm}.log" 2>&1
done
