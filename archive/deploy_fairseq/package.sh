#!/usr/bin/env bash
# Build a shippable code+head tarball for the fairseq XLS-R deployment.
# Does NOT include the 8 GB encoder checkpoint — ship that separately (see DEPLOY_GCP_FAIRSEQ.md).
set -euo pipefail

REPO=$HOME
OUT=${1:-$NATIONALITY_OUTPUTS/deploy_fairseq/deploy_fairseq_code.tar.gz}

cd "$REPO"
tar -czf "$OUT" \
  country_classifier \
  fairseq \
  scripts/create_env.sh \
  taeho/deploy_fairseq/app.py \
  taeho/deploy_fairseq/index.html \
  taeho/deploy_fairseq/head_fine62_16s.pt \
  taeho/deploy_fairseq/class_labels.csv \
  taeho/deploy_fairseq/DEPLOY_GCP_FAIRSEQ.md

echo "wrote $OUT"
du -h "$OUT"
echo "Encoder checkpoint to ship separately:"
ls -lh $SPEECH_AI_DATA/models/xlsr2_2B_1000k_inference.pt
