#!/bin/bash
# 원어민 학습 종료 → omniASR ① 자동 기동 체인
LOG=$MDD_OUTPUTS/train_native.launch.log
until grep -q "Stopping training\|done training" "$LOG" 2>/dev/null; do sleep 60; done
echo "[chain] $(date '+%F %T') 원어민 학습 종료 신호 감지"
# 마지막 검증·체크포인트 저장·프로세스 종료 대기
while pgrep -f "mdd_ctc_300m_tag_all_native" >/dev/null 2>&1; do sleep 30; done
sleep 90
echo "[chain] $(date '+%F %T') GPU 정리 확인 — omniASR ① 기동"
cd $MDD_OUTPUTS
nohup $HOME/miniconda3/envs/fairseq_xlsr/bin/fairseq-hydra-train \
  --config-dir config --config-name mdd_ctc_300m_omniw2v_direct \
  > train_omniw2v.launch.log 2>&1 &
echo "[chain] omniASR pid $! — train_omniw2v.launch.log"
