#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
source tools/env.sh >/dev/null
DIALECT_COMPARE_DIR="${1:-/data/speech-ai/out/dialect/compare_speakerid_20261011}"
DIALECT_COMPARE_MANIFESTS="${2:-/data/speech-ai/out/dialect/speakerid_20261011/manifests}"
DIALECT_COMPARE_REFERENCE="$DIALECT_COMPARE_DIR/xlsr2b/report.json"
mkdir -p "$DIALECT_COMPARE_DIR" /data/speech-ai/tmp/dialect_compare_20261011
export PYTHONPATH="/data/speech-ai/tools/dialect_compare_py38:/data/speech-ai/tools/hf_py38${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data/speech-ai/models/hf
export HF_MODULES_CACHE=/data/speech-ai/tmp/dialect_compare_20261011/hf_modules
export TMPDIR=/data/speech-ai/tmp/dialect_compare_20261011
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/data/speech-ai/cuda11_shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

status() {
  "$PY" - "$DIALECT_COMPARE_DIR/status.json" "$1" "$2" <<'PY'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]);t=p.with_suffix('.tmp')
t.write_text(json.dumps({'status':sys.argv[2],'stage':sys.argv[3],'updated_at':datetime.datetime.now().astimezone().isoformat()},indent=2)+'\n')
os.replace(str(t),str(p))
PY
}

DIALECT_COMPARE_STAGE=initializing
trap 'status failed "$DIALECT_COMPARE_STAGE"' ERR
"$PY" - "$DIALECT_COMPARE_MANIFESTS" "$DIALECT_COMPARE_DIR" <<'PY'
import json,sys
from pathlib import Path
summary=json.loads((Path(sys.argv[1])/'summary.json').read_text())
if summary.get('person_disjoint_verified') is not True:
    raise SystemExit('Classifier comparison requires verified person splits. Use run_dialect_feature_cache.sh for frozen extraction only.')
if any(summary['counts'][s].get(label,0)==0 for s in ['train','valid','test'] for label in summary['labels']):
    raise SystemExit('All five regions must have train/valid/test examples before comparing classifiers.')
for report_path in Path(sys.argv[2]).glob('*/report.json'):
    report=json.loads(report_path.read_text())
    if report.get('data_summary',{}).get('manifest_sha256') != summary.get('manifest_sha256'):
        raise SystemExit('Existing model report uses different manifest rows: '+str(report_path))
PY
DIALECT_COMPARE_STAGE=xlsr2b
status running "$DIALECT_COMPARE_STAGE"
if [[ ! -f "$DIALECT_COMPARE_REFERENCE" ]]; then
  "$PY" -u tasks/dialect-classification/scripts/train/train_xlsr2b_dialect.py \
    --manifests "$DIALECT_COMPARE_MANIFESTS" --out "$DIALECT_COMPARE_DIR/xlsr2b" \
    --device cuda:0 --batch-size 32 --workers 4 \
    > "$DIALECT_COMPARE_DIR/xlsr2b.log" 2>&1
fi
"$PY" tasks/dialect-classification/scripts/train/summarize_dialect_comparison.py --root "$DIALECT_COMPARE_DIR" --reference "$DIALECT_COMPARE_REFERENCE" > "$DIALECT_COMPARE_DIR/summary.log" 2>&1
for DIALECT_COMPARE_MODEL in w2vbert2 wavlm_large eat_large; do
  DIALECT_COMPARE_STAGE="$DIALECT_COMPARE_MODEL"
  status running "$DIALECT_COMPARE_STAGE"
  if [[ ! -f "$DIALECT_COMPARE_DIR/$DIALECT_COMPARE_MODEL/report.json" ]]; then
    "$PY" -u tasks/dialect-classification/scripts/train/train_dialect_comparison.py \
      --model "$DIALECT_COMPARE_MODEL" --out "$DIALECT_COMPARE_DIR/$DIALECT_COMPARE_MODEL" --manifests "$DIALECT_COMPARE_MANIFESTS" \
      --batch-size 128 --workers 6 \
      > "$DIALECT_COMPARE_DIR/$DIALECT_COMPARE_MODEL.log" 2>&1
  fi
  "$PY" tasks/dialect-classification/scripts/train/summarize_dialect_comparison.py --root "$DIALECT_COMPARE_DIR" --reference "$DIALECT_COMPARE_REFERENCE" > "$DIALECT_COMPARE_DIR/summary.log" 2>&1
done
DIALECT_COMPARE_STAGE=xlsr2b_latency
status running "$DIALECT_COMPARE_STAGE"
if [[ ! -f "$DIALECT_COMPARE_DIR/xlsr2b/benchmark.json" ]]; then
  "$PY" -u tasks/dialect-classification/scripts/train/train_dialect_comparison.py \
    --model xlsr2b --benchmark-only --out "$DIALECT_COMPARE_DIR/xlsr2b" --manifests "$DIALECT_COMPARE_MANIFESTS" \
    > "$DIALECT_COMPARE_DIR/xlsr2b_benchmark.log" 2>&1
fi
"$PY" tasks/dialect-classification/scripts/train/summarize_dialect_comparison.py --root "$DIALECT_COMPARE_DIR" --reference "$DIALECT_COMPARE_REFERENCE" > "$DIALECT_COMPARE_DIR/summary.log" 2>&1
status complete comparison_report
