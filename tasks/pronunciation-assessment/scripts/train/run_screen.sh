#!/usr/bin/env bash
# 짧은 스크리닝 한 개 — "나쁜 아이디어를 싸게 버리기" 용. 최종 성능 측정용이 아니다.
#
#   bash .../run_screen.sh S1_enclr0.3 --encoder-lr-scale 0.3
#   bash .../run_screen.sh S3_freeze12 --freeze-layers 12
#   MANIFEST=$MDD_OUTPUTS/manifest_p5a_clean bash .../run_screen.sh P5a_clean   # 학습 매니페스트만 바꿀 때
#
# 조건은 H0 와 같고(hold-out 매니페스트 · 가중치 1 · lr 1e-5 · 스케줄 40k 기준) 4k 에서 멈춘다.
# → H0 의 step_2000 · step_4000 이 그대로 기준선(S0)이 된다. 데이터를 줄이지 않는 이유:
#   이 트레이너는 step 당 시간이 데이터 크기와 무관하다(배치를 오디오 길이로 채움) — 줄여도 안 빨라지고
#   같은 발화를 여러 번 돌아 과적합 양상만 달라진다.
# 채점은 valid(seen) · valid_unseen 만. test 는 보지 않는다.
set -u
NAME=${1:?"실험 이름"}; shift
cd "$(dirname "$0")/../../../.." || exit 1
source tools/env.sh >/dev/null
export HF_HOME=/data/speech-ai/corpus/weights/external PYTHONPATH=/data/speech-ai/tools/hf_py38:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
T=tasks/pronunciation-assessment; M=$MDD_OUTPUTS/manifest_holdout_v1        # 채점(valid)은 항상 이 매니페스트
TM=${MANIFEST:-$M}                                                          # 학습 매니페스트
W=${WEIGHTS:-$( [[ -f $TM/weights_b0.5/weights_H0.txt ]] && echo $TM/weights_b0.5/weights_H0.txt || echo $TM/weights_ones.txt )}
CK=$SPEECH_AI_DATA/checkpoints/screen_v1/$NAME; R=$MDD_OUTPUTS/experiments/screen_v1/$NAME
STOP=${STOP_AT:-4000}; SCORE_STEPS=${SCORE_STEPS:-"2000 4000"}                # 연장 스크리닝: STOP_AT=8000 SCORE_STEPS="2000 4000 6000 8000"
mkdir -p "$R/sel"; echo "$* $( [[ $TM != $M ]] && echo "[train=$(basename $TM)]" )" > "$R/args.txt"
log(){ echo "[$(date '+%F %T')] [$NAME] $*"; }
if [[ ! -d $CK/step_$STOP ]]; then
  log "학습 (${STOP} 에서 멈춤) · 추가 인자: $*"
  $PY $T/scripts/train/train_w2vbert_ctc.py --manifest $TM --valid-split valid_unseen \
    --model $SPEECH_AI_DATA/checkpoints/w2vbert_stage1/step_10000 --save-dir $CK \
    --max-update 40000 --stop-at $STOP --lr 1e-5 --freeze-updates 0 --save-interval 2000 --valid-n 10000 \
    --weights $W "$@" 2>&1 | grep -v -E "Warning|warn|conv1d"
fi
[[ -d $CK/step_$STOP ]] || { log "학습 실패"; exit 1; }
for s in $SCORE_STEPS; do
  for sp in valid valid_unseen; do
    o=$R/sel/step_${s}_$sp
    [[ -f $o/summary.json ]] && continue
    [[ -f $o.preds.jsonl ]] || $PY $T/scripts/eval/decode_hf_ctc.py --model $CK/step_$s --manifest $M --split $sp \
        --out $o 2>&1 | grep -E "^\[preds\]"
    $PY $T/scripts/eval/mdd_error_analysis.py --manifest $M --split $sp --preds $o.preds.jsonl \
        --out $R/sel --name step_${s}_$sp --no-plots 2>&1 | grep -E "baseline" | sed "s/^/[$s $sp] /"
  done
done
log "완료"
