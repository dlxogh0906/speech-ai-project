#!/usr/bin/env bash
# Run from repository root. All artifacts remain separate from production.
set -euo pipefail
source tools/env.sh
export PYTHONPATH="${2:?pass the existing Transformers Python38 package directory}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
diag_out="$1"
trap 'code=$?; echo "$code" > "$diag_out/run.exit"' EXIT
for diag_backbone in w2vbert xlsr300m; do
  for diag_step in 2000 4000; do
    "$PY" tasks/pronunciation-assessment/scripts/eval/diagnose_h0_representations.py \
      --stage extract --out "$diag_out" --backbone "$diag_backbone" --step "$diag_step" \
      > "$diag_out/${diag_backbone}_${diag_step}.log" 2>&1
  done
done
"$PY" tasks/pronunciation-assessment/scripts/eval/fit_h0_representation_probes.py \
  --root "$diag_out" > "$diag_out/probes.log" 2>&1
