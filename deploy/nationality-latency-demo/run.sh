#!/usr/bin/env bash

# 저장소 .env 를 읽어 경로 변수를 채운다
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$ROOT/tools/env.sh"

# XLS-R 2B CPU latency 데모 서버 실행
# 사용법: bash run.sh          (기본 포트 8777)
#        PORT=9000 bash run.sh
set -e
cd "$(dirname "$0")"
exec conda run --no-capture-output -n fairseq_xlsr_5080 python server.py
