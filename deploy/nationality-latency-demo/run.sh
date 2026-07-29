#!/usr/bin/env bash
# XLS-R 2B CPU latency 데모 서버 실행
# 사용법: bash run.sh          (기본 포트 8777)
#        PORT=9000 bash run.sh
set -e
cd "$(dirname "$0")"
exec conda run --no-capture-output -n fairseq_xlsr_5080 python server.py
