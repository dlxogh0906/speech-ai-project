#!/bin/bash
set -u
PY=$HOME/miniconda3/envs/fairseq_xlsr/bin/python
M=$MDD_OUTPUTS
cd scripts
echo "[1/3] train 화자 특징 추출 (80k)"
$PY gop_ctc.py --split train --n 80000 --cache $M/gop_feats_train.jsonl --out $M/eval_gop_train_feats.json > $M/gop_feats_train.log 2>&1
echo "exit=$?"
echo "[2/3] test 화자 특징 추출 (전체)"
$PY gop_ctc.py --split test --n 200000 --cache $M/gop_feats_test.jsonl --out $M/eval_gop_test_feats.json > $M/gop_feats_test.log 2>&1
echo "exit=$?"
echo "[3/3] 보정 fit + 평가"
$PY gop_calibrate.py > $M/gop_calibrate.log 2>&1
echo "exit=$?"
echo GOP_CALIB_DONE
