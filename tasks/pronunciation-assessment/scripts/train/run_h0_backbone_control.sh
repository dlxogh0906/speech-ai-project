#!/usr/bin/env bash
# New persistent output required. Invoke inside tmux for the full experiment.
# source tools/env.sh; export PYTHONPATH=<existing HF overlay>:$PYTHONPATH
# bash .../run_h0_backbone_control.sh <protocol.json> <persistent-output>
set -euo pipefail
cd "$(dirname "$0")/../../../.."
source tools/env.sh >/dev/null
protocol=${1:?protocol.json required}
output=${2:?new persistent output directory required}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$PY" - "$protocol" "$output" <<'PY'
import json, pathlib, shutil, sys
import transformers
p=json.load(open(sys.argv[1])); out=pathlib.Path(sys.argv[2]).resolve()
if out.exists(): raise SystemExit('output already exists; refusing overwrite')
parent=out.parent
while not parent.exists(): parent=parent.parent
if str(out).startswith('/dev/shm/') or shutil.disk_usage(str(parent)).free<p['minimum_free_bytes']:
    raise SystemExit('persistent storage with at least 30 GiB free required; no training started')
PY
mkdir -p "$output"
trap 'result=$?; echo "$result" > "$output/run.exit"' EXIT
task=tasks/pronunciation-assessment/scripts
for backbone in w2vbert xlsr300m; do
  "$PY" "$task/train/train_h0_backbone_control.py" --protocol "$protocol" \
    --backbone "$backbone" --stage stage1 --out "$output/${backbone}_stage1" \
    > "$output/${backbone}_stage1.log" 2>&1
  "$PY" "$task/train/train_h0_backbone_control.py" --protocol "$protocol" \
    --backbone "$backbone" --stage stage2 --out "$output/${backbone}_stage2" \
    --stage1-checkpoint "$output/${backbone}_stage1/step_10000.pt" \
    > "$output/${backbone}_stage2.log" 2>&1
  for step in 2000 4000; do
    "$PY" "$task/eval/eval_h0_backbone_control.py" --protocol "$protocol" \
      --checkpoint "$output/${backbone}_stage2/step_${step}.pt" \
      --out "$output/${backbone}_eval_${step}" > "$output/${backbone}_eval_${step}.log" 2>&1
  done
done
"$PY" - "$output" <<'PY'
import json,pathlib,sys
sys.path.insert(0,'tasks/pronunciation-assessment/scripts/eval')
from eval_h0_backbone_control import compare_trajectory
out=pathlib.Path(sys.argv[1])
load=lambda b:[json.load(open(out/('%s_eval_%d'%(b,s))/'report.json')) for s in (2000,4000)]
result=compare_trajectory(load('w2vbert'),load('xlsr300m'))
(out/'trajectory.json').write_text(json.dumps(result,indent=2))
PY
