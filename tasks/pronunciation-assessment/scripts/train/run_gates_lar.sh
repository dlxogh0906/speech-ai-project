#!/bin/bash
# LAR 적응 학습 완료 대기 → 게이트①(TAG test F1) → 게이트②(단어누락 정밀도) 자동 실행
set -u
PY=$HOME/miniconda3/envs/fairseq_xlsr/bin/python
LOG=$MDD_OUTPUTS/train_lar.launch.log
M=$MDD_OUTPUTS
CKPT=$SPEECH_AI_DATA/mdd_ckpt_tag_all_lar/checkpoint_best.pt

echo "[wait] training to finish..."
for i in $(seq 1 120); do
  grep -q "done training" "$LOG" && break
  sleep 60
done
grep -q "done training" "$LOG" || { echo "[abort] training not finished after 2h wait"; exit 1; }
sleep 30
echo "[gate1] TAG test 178k re-eval with $CKPT"
cd scripts
$PY mdd_eval_tag71.py --manifest $M/manifest_tag_all --ckpt $CKPT \
    --preds $M/eval_tag_all_lar_preds.jsonl --out $M/eval_tag71_all_lar.json \
    > $M/gate1_tag.log 2>&1
echo "[gate1] exit=$?"
echo "[gate2] word-skip re-eval (record LAR 20k)"
$PY eval_word_skip_record.py --ckpt $CKPT \
    --preds $M/word_skip_preds_lar.jsonl --out $M/eval_word_skip_lar.json \
    > $M/gate2_wordskip.log 2>&1
echo "[gate2] exit=$?"
echo "ALL_GATES_DONE"
