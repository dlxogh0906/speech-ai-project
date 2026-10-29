#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
source tools/env.sh >/dev/null
DIALECT_FINE_DIR=/data/speech-ai/out/dialect/finetune_speakerid_20261012
DIALECT_BASE_DIR=/data/speech-ai/out/dialect/compare_speakerid_20261011
mkdir -p "$DIALECT_FINE_DIR" /data/speech-ai/tmp/dialect_finetune_20261012
exec 9> "$DIALECT_FINE_DIR/pipeline.lock"
flock -n 9
export PYTHONPATH="/data/speech-ai/tools/dialect_compare_py38:/data/speech-ai/tools/hf_py38${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data/speech-ai/models/hf
export HF_MODULES_CACHE=/data/speech-ai/tmp/dialect_compare_20261011/hf_modules
export TMPDIR=/data/speech-ai/tmp/dialect_finetune_20261012
export LD_LIBRARY_PATH="/data/speech-ai/cuda11_shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

status() {
  "$PY" - "$DIALECT_FINE_DIR/status.json" "$1" "$2" <<'PY'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]);t=p.with_suffix('.tmp')
t.write_text(json.dumps({'status':sys.argv[2],'stage':sys.argv[3],'updated_at':datetime.datetime.now().astimezone().isoformat()},indent=2)+'\n')
os.replace(str(t),str(p))
PY
}

DIALECT_FINE_STAGE=initializing
trap 'status failed "$DIALECT_FINE_STAGE"' ERR
for DIALECT_FINE_MODEL in eat_large xlsr2b; do
  DIALECT_FINE_STAGE="$DIALECT_FINE_MODEL"
  status running "$DIALECT_FINE_STAGE"
  if [[ "$DIALECT_FINE_MODEL" == eat_large ]]; then
    DIALECT_FINE_BLOCKS=4
    DIALECT_FINE_BATCH=16
    DIALECT_FINE_EVAL_BATCH=128
  else
    DIALECT_FINE_BLOCKS=8
    DIALECT_FINE_BATCH=8
    DIALECT_FINE_EVAL_BATCH=32
  fi
  if [[ ! -f "$DIALECT_FINE_DIR/$DIALECT_FINE_MODEL/report.json" ]]; then
    "$PY" -u tasks/dialect-classification/scripts/train/finetune_dialect_encoder.py \
      --model "$DIALECT_FINE_MODEL" --baseline "$DIALECT_BASE_DIR/$DIALECT_FINE_MODEL" \
      --out "$DIALECT_FINE_DIR/$DIALECT_FINE_MODEL" --last-blocks "$DIALECT_FINE_BLOCKS" \
      --batch-size "$DIALECT_FINE_BATCH" --accumulation 4 --epochs 3 --patience 2 \
      --eval-batch-size "$DIALECT_FINE_EVAL_BATCH" --workers 4 \
      > "$DIALECT_FINE_DIR/$DIALECT_FINE_MODEL.log" 2>&1
  fi
done
DIALECT_FINE_STAGE=summarize
status running "$DIALECT_FINE_STAGE"
"$PY" - "$DIALECT_FINE_DIR" "$DIALECT_BASE_DIR" <<'PY'
import csv,datetime,json,pathlib,sys
root,base=map(pathlib.Path,sys.argv[1:]);rows=[]
for name in ['eat_large','xlsr2b']:
    trial=json.loads((root/name/'report.json').read_text())
    frozen=json.loads((base/name/'report.json').read_text())
    row={'model':name,'encoder_adapted_selected':trial['selected_encoder_adapted'],
         'selected_epoch':trial['selected_epoch'],'trainable_encoder_parameters':trial['trainable_encoder_parameters'],
         'checkpoint_selection_metric':trial.get('selection_metric','macro_f1'),
         'baseline_validation_top1':trial['baseline_validation']['top1'],
         'validation_top1':trial['selected_validation']['top1'],
         'baseline_validation_macro_f1':trial['baseline_validation']['macro_f1'],
         'validation_macro_f1':trial['selected_validation']['macro_f1'],
         'baseline_test_top1':frozen['test_single_window']['top1'],
         'test_top1':trial['test_single_window']['top1'],'test_macro_f1':trial['test_single_window']['macro_f1'],
         'three_window_top1':trial['test_three_utterance_windows_mean_logits']['top1'],
         'elapsed_hours':trial['elapsed_seconds']/3600,'report':str(root/name/'report.json')}
    for region,metrics in trial['test_single_window']['per_region'].items():
        row[region+'_top1']=metrics['recall_top1']
    rows.append(row)
winner=max(rows,key=lambda r:(r['validation_top1'],r['validation_macro_f1']))
payload={'completed_at':datetime.datetime.now().astimezone().isoformat(),'models':rows,
         'selected_model_by_validation':winner['model'],
         'selection_rule':'Validation Top1, then Macro F1; individual checkpoint selection metric recorded per model'}
(root/'comparison.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
with (root/'comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(payload,ensure_ascii=False))
PY
status complete fine_tuning_reports
