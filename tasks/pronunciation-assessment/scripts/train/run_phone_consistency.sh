#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
source tools/env.sh
export PYTHONPATH="$(dirname "$SPEECH_AI_DATA")/tools/hf_py38${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SPEECH_AI_DATA/weights/external"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
pair_out="${1:-$MDD_OUTPUTS/analysis/phone_consistency_2026-10-10/pilot}"
mkdir -p "$pair_out"
exec 9>"$pair_out/pipeline.lock"
flock -n 9 || { echo "Experiment already running: $pair_out" >&2; exit 1; }
exec > >(tee -a "$pair_out/run.log") 2>&1
trap 'code=$?; echo "$code" > "$pair_out/exit_code.txt"' EXIT
if [ ! -f "$pair_out/plan.json" ]; then
  "$PY" tasks/pronunciation-assessment/scripts/train/train_phone_consistency.py prepare \
    --out "$pair_out" --exclude-robustness --exclude-prior \
    "$MDD_OUTPUTS/analysis/noise_preprocess_2026-10-09/full_v02/experiment.json" \
    "$MDD_OUTPUTS/analysis/noise_mixing_2026-10-09/full/valid/experiment.json" \
    "$MDD_OUTPUTS/analysis/noise_mixing_2026-10-09/full/test/experiment.json"
fi
if [ ! -f "$pair_out/reference_anchors.pt" ]; then
  "$PY" tasks/pronunciation-assessment/scripts/train/train_phone_consistency.py anchors --out "$pair_out"
fi
"$PY" tasks/pronunciation-assessment/scripts/train/train_phone_consistency.py train --out "$pair_out" --mode targeted
"$PY" tasks/pronunciation-assessment/scripts/eval/eval_phone_consistency.py valid --out "$pair_out"
pair_step="$("$PY" - "$pair_out" <<'PY'
import json, sys
from pathlib import Path
print(json.loads((Path(sys.argv[1])/'selection.json').read_text())['chosen']['step'])
PY
)"
"$PY" tasks/pronunciation-assessment/scripts/train/train_phone_consistency.py train \
  --out "$pair_out" --mode paired_control --updates "$pair_step"
"$PY" tasks/pronunciation-assessment/scripts/eval/eval_phone_consistency.py test --out "$pair_out"
"$PY" tests/test_imports.py
echo "[complete] $pair_out/report.txt"
