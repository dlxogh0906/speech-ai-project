#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
source tools/env.sh
export PYTHONPATH="$(dirname "$SPEECH_AI_DATA")/tools/hf_py38${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SPEECH_AI_DATA/weights/external"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
threshold_out="${1:-$MDD_OUTPUTS/analysis/phone_threshold_2026-10-10/pilot}"
mkdir -p "$threshold_out"
exec 9>"$threshold_out/pipeline.lock"
flock -n 9 || { echo "Experiment already running" >&2; exit 1; }
exec > >(tee -a "$threshold_out/run.log") 2>&1
trap 'code=$?; echo "$code" > "$threshold_out/exit_code.txt"' EXIT
threshold_script=tasks/pronunciation-assessment/scripts/eval/calibrate_phone_threshold.py
if [ ! -f "$threshold_out/plan.json" ]; then
    "$PY" "$threshold_script" prepare --out "$threshold_out"
fi
"$PY" "$threshold_script" smoke --out "$threshold_out"
"$PY" "$threshold_script" calibrate --out "$threshold_out"
"$PY" "$threshold_script" test --out "$threshold_out"
"$PY" tests/test_imports.py
echo "[complete] $threshold_out/report.txt"
