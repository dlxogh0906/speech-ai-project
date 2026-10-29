#!/usr/bin/env bash
# 국적 모델 전부 비교 데모를 포그라운드로 띄운다 — tmux 안에서 실행한다.
#   tmux new-session -d -s nat-all-demo bash deploy/nationality-all-demo/start.sh
# 설정은 ${NAT_ALL_CONF:-~/.config/nat-all-demo/demo.env} (키 · 포트 · 로그 · 캡처 폴더).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CONF=${NAT_ALL_CONF:-$HOME/.config/nat-all-demo/demo.env}
cd "$ROOT"
source tools/env.sh
set -a
. "$CONF"
set +a
: "${NAT_ALL_KEY:?}" "${NAT_ALL_LOG:?}"
mkdir -p "$(dirname "$NAT_ALL_LOG")"
echo "[start $(date '+%F %T')] port=${NAT_ALL_PORT:-8988}" >> "$NAT_ALL_LOG"
exec "$PY" -u deploy/nationality-all-demo/app.py --port "${NAT_ALL_PORT:-8988}" >> "$NAT_ALL_LOG" 2>&1
