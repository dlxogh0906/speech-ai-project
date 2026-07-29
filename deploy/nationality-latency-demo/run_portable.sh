#!/usr/bin/env bash
# XLS-R 2B 국적분류 — GPU/CPU/int8 속도 비교 + 동접 부하 테스트 데모 (이식용 런처)
#
# 사용 전 아래 두 경로를 대상 컴퓨터에 맞게 지정하세요 (환경변수 또는 이 파일 직접 편집):
#   XLSR_CKPT     : 32L 체크포인트(xlsr2_1p45B_32L.pt, 5.5GB) 경로  ← 이 번들에는 없음, 따로 복사
#   FAIRSEQ_ROOT  : 로컬 fairseq repo 경로
#
# 예)
#   XLSR_CKPT=/data/models/xlsr2_1p45B_32L.pt \
#   FAIRSEQ_ROOT=/data/fairseq \
#   bash run_portable.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

# head.pt는 번들에 포함되어 있어 자동 지정됨
export HEAD_PT="${HEAD_PT:-${NATIONALITY_OUTPUTS:?NATIONALITY_OUTPUTS 를 설정하십시오}/0704_country41_sent043_en/head.pt}"

# 아래 두 개는 반드시 지정 (미설정 시 에러로 안내)
export XLSR_CKPT="${XLSR_CKPT:?XLSR_CKPT 미설정 — 32L 체크포인트(xlsr2_1p45B_32L.pt) 경로를 지정하세요. SETUP.md 참조}"
export FAIRSEQ_ROOT="${FAIRSEQ_ROOT:?FAIRSEQ_ROOT 미설정 — fairseq repo 경로를 지정하세요. SETUP.md 참조}"

export USE_GPU="${USE_GPU:-1}"        # 0 = CPU 전용 (GPU 없는 컴퓨터)
export USE_INT8="${USE_INT8:-1}"      # 0 = int8 경로 비활성
export PORT="${PORT:-8777}"
export CPU_CONCURRENCY="${CPU_CONCURRENCY:-4}"

cd "$HERE"
echo "* http://localhost:${PORT}          → GPU vs CPU vs int8 비교"
echo "* http://localhost:${PORT}/loadtest → 동접 부하 테스트"
exec python server.py
