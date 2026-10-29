#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
source tools/env.sh
export PYTHONPATH="$(dirname "$SPEECH_AI_DATA")/tools/hf_py38${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SPEECH_AI_DATA/weights/external"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
pilot_out="${1:-$MDD_OUTPUTS/analysis/noise_training_2026-10-09/pilot}"
mkdir -p "$pilot_out"
exec > >(tee -a "$pilot_out/run.log") 2>&1
trap 'code=$?; echo "$code" > "$pilot_out/exit_code.txt"' EXIT
if [ ! -f "$pilot_out/plan.json" ]; then
  "$PY" tasks/pronunciation-assessment/scripts/train/train_noise_pilot.py prepare \
    --out "$pilot_out" --exclude-prior \
    "$MDD_OUTPUTS/analysis/noise_preprocess_2026-10-09/full_v02/experiment.json" \
    "$MDD_OUTPUTS/analysis/noise_mixing_2026-10-09/full/valid/experiment.json" \
    "$MDD_OUTPUTS/analysis/noise_mixing_2026-10-09/full/test/experiment.json"
fi
"$PY" tasks/pronunciation-assessment/scripts/train/train_noise_pilot.py train --out "$pilot_out" --mode noise
"$PY" tasks/pronunciation-assessment/scripts/eval/eval_noise_training.py valid --out "$pilot_out"
pilot_step="$("$PY" - "$pilot_out" <<'PY'
import json, sys
from pathlib import Path
print(json.loads((Path(sys.argv[1])/'selection.json').read_text())['chosen']['step'])
PY
)"
"$PY" tasks/pronunciation-assessment/scripts/train/train_noise_pilot.py train --out "$pilot_out" --mode clean --updates "$pilot_step"
"$PY" tasks/pronunciation-assessment/scripts/eval/eval_noise_training.py test --out "$pilot_out"
echo "[pilot complete] $pilot_out/report.txt"
