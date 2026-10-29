#!/usr/bin/env bash
# Sequential archive download/conversion, followed by frozen XLS-R 2B head training.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo 'Usage: bash tasks/dialect-classification/scripts/train/run_dialect_10pct.sh [run-dir] [--train-only]'
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
source tools/env.sh >/dev/null
DIALECT_RUN_DIR="${1:-/data/speech-ai/out/dialect/20261010}"
DIALECT_TRAIN_ONLY="${2:-}"
if [[ -n "$DIALECT_TRAIN_ONLY" && "$DIALECT_TRAIN_ONLY" != "--train-only" ]]; then
  echo "Unknown option: $DIALECT_TRAIN_ONLY" >&2
  exit 2
fi
mkdir -p "$DIALECT_RUN_DIR" /data/speech-ai/tmp/dialect_20261010
export TMPDIR=/data/speech-ai/tmp/dialect_20261010
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="/data/speech-ai/cuda11_shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

write_status() {
  "$PY" - "$DIALECT_RUN_DIR/status.json" "$1" "$2" <<'PY'
import datetime,json,os,pathlib,sys
path=pathlib.Path(sys.argv[1])
temporary=path.with_name(path.name+'.tmp')
temporary.write_text(json.dumps({'status':sys.argv[2],'stage':sys.argv[3],'updated_at_utc':datetime.datetime.utcnow().isoformat()+'Z'},indent=2)+'\n')
os.replace(str(temporary),str(path))
PY
}

DIALECT_STAGE=prepare
trap 'write_status failed "$DIALECT_STAGE"' ERR
if [[ "$DIALECT_TRAIN_ONLY" != "--train-only" ]]; then
  write_status running "$DIALECT_STAGE"
  "$PY" -u tasks/dialect-classification/scripts/data/prepare_dialect_subset.py \
    --plan "$DIALECT_RUN_DIR/download_plan.json" --out "$DIALECT_RUN_DIR" \
    > "$DIALECT_RUN_DIR/prepare.log" 2>&1
fi

DIALECT_STAGE=train_xlsr2b
write_status running "$DIALECT_STAGE"
"$PY" -u tasks/dialect-classification/scripts/train/train_xlsr2b_dialect.py \
  --manifests "$DIALECT_RUN_DIR/manifests" --out "$DIALECT_RUN_DIR/xlsr2b" \
  > "$DIALECT_RUN_DIR/train.log" 2>&1

write_status complete evaluation_report
