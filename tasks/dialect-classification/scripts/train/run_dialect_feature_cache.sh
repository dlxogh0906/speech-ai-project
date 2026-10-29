#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
source tools/env.sh >/dev/null
DIALECT_FEATURE_ROOT="${1:-/data/speech-ai/out/dialect/compare_20261011}"
mkdir -p "$DIALECT_FEATURE_ROOT" /data/speech-ai/tmp/dialect_compare_20261011
export PYTHONPATH="/data/speech-ai/tools/dialect_compare_py38:/data/speech-ai/tools/hf_py38${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data/speech-ai/models/hf
export HF_MODULES_CACHE=/data/speech-ai/tmp/dialect_compare_20261011/hf_modules
export TMPDIR=/data/speech-ai/tmp/dialect_compare_20261011
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/data/speech-ai/cuda11_shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
feature_status() {
  "$PY" - "$DIALECT_FEATURE_ROOT/status.json" "$1" "$2" <<'PY'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]);t=p.with_suffix('.tmp')
t.write_text(json.dumps({'status':sys.argv[2],'stage':sys.argv[3],
    'classifier_training':'paused_awaiting_verified_person_split',
    'speaker_audit':str(p.parent/'speaker_audit.json'),
    'updated_at':datetime.datetime.now().astimezone().isoformat()},indent=2)+'\n')
os.replace(str(t),str(p))
PY
}
DIALECT_FEATURE_STAGE=initializing
trap 'feature_status feature_cache_failed "$DIALECT_FEATURE_STAGE"' ERR
for DIALECT_FEATURE_MODEL in w2vbert2 wavlm_large eat_large; do
  DIALECT_FEATURE_STAGE="$DIALECT_FEATURE_MODEL"
  feature_status feature_cache_running "$DIALECT_FEATURE_STAGE"
  if [[ ! -f "$DIALECT_FEATURE_ROOT/$DIALECT_FEATURE_MODEL/features_only_complete.json" ]]; then
    "$PY" -u tasks/dialect-classification/scripts/train/extract_dialect_comparison_features.py \
      --model "$DIALECT_FEATURE_MODEL" --out "$DIALECT_FEATURE_ROOT/$DIALECT_FEATURE_MODEL" \
      --batch-size 128 --workers 6 \
      >> "$DIALECT_FEATURE_ROOT/$DIALECT_FEATURE_MODEL.log" 2>&1
  fi
done
feature_status feature_cache_complete_awaiting_person_split all_encoders
