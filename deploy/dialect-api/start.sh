#!/usr/bin/env bash
# 사투리 분류 API + 데모를 포그라운드로 띄운다 — tmux 안에서 실행한다.
#   tmux new-session -d -s dialect-api bash deploy/dialect-api/start.sh
# 설정은 ${DIALECT_API_CONF:-~/.config/dialect-api/api.env} (키 · 모델 폴더 · 추가 PYTHONPATH · 포트 · 로그 · 캡처 폴더).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CONF=${DIALECT_API_CONF:-$HOME/.config/dialect-api/api.env}
cd "$ROOT"
source tools/env.sh
set -a
. "$CONF"
set +a
: "${DIALECT_MODEL_DIR:?}" "${DIALECT_API_KEY:?}" "${DIALECT_LOG:?}"
mkdir -p "$(dirname "$DIALECT_LOG")"
echo "[start $(date '+%F %T')] port=${DIALECT_PORT:-8987}" >> "$DIALECT_LOG"
PYTHONPATH=${DIALECT_PYTHONPATH:-} exec "$PY" -u deploy/dialect-api/app.py --port "${DIALECT_PORT:-8987}" >> "$DIALECT_LOG" 2>&1
