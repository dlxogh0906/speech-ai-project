#!/usr/bin/env bash

# 저장소 .env 를 읽어 경로 변수를 채운다
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$ROOT/tools/env.sh"

# XLS-R 300M 국적분류 데모 서버 (이 머신: A100 80GB)
# - 인코더: xlsr2_300m.pt (24층, 1024d)
# - head: 0705_country41_sent043_300m (layer 17, 영문 라벨, 043 병합 best)
# - GPU 메모리 하드 락: 40GB (이 프로세스가 그 이상 할당 시 OOM)
set -e
cd "$(dirname "$0")"

ENV=$HOME/miniconda3/envs/fairseq_xlsr
export PATH="$ENV/bin:$PATH"                 # ffmpeg/ffprobe 를 PATH에서 찾게

export XLSR_CKPT=$SPEECH_AI_MODELS/xlsr2_300m.pt
# FAIRSEQ_ROOT 은 .env 에서 온다
export HEAD_PT=$NATIONALITY_OUTPUTS/0705_country41_sent043_300m/head.pt
export GPU_MEM_GB=40                          # ← GPU 락 (40GB 초과 금지)
export USE_GPU=1
export USE_INT8=1
export PORT="${PORT:-8777}"
export HOST=0.0.0.0

exec "$ENV/bin/python" server.py
