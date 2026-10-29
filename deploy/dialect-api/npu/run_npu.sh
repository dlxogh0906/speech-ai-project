#!/bin/bash
# NPU 서버의 사투리 API(:8907) 실행 — 포그라운드. systemd(dialect-api.service)가 부른다. 2026-10-28~
# 번들 루트(이 파일의 세 단계 위)에서 deploy/dialect-api/app.py --backend npu 로 models/*.rbln 을 돈다.
# 설정 ${DIALECT_API_CONF:-/root/dialect_serve/api.env}: DIALECT_API_KEY(필수) · DIALECT_PORT · DIALECT_NPU_DEVICES.
# 입력·결과 저장 폴더는 systemd drop-in(capture.conf)의 SPEECH_CAPTURE_DIR — 이 스크립트는 건드리지 않는다.
set -euo pipefail
B=$(cd "$(dirname "$0")/../../.." && pwd)
set -a
. "${DIALECT_API_CONF:-/root/dialect_serve/api.env}"
set +a
: "${DIALECT_API_KEY:?}"
export FFMPEG_BIN=${FFMPEG_BIN:-/root/nat_serve/bin/ffmpeg}
cd "$B"
exec "${DIALECT_PY:-/root/nat_serve/venv/bin/python}" -u deploy/dialect-api/app.py --backend npu --model-dir "$B/models" \
     --npu-devices "${DIALECT_NPU_DEVICES:-5,5}" --host 127.0.0.1 --port "${DIALECT_PORT:-8907}"
