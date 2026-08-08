#!/usr/bin/env bash
# 추론 API 서버 기동 스크립트. 포그라운드로 돈다 — 긴 세션은 tmux 안에서 띄울 것.
# 두 서버를 다 띄우려면 tmux 창 두 개에서 각각 실행한다.
set -euo pipefail

usage() {
    cat <<'EOF'
사용법:
  bash deploy/serve.sh mdd [--port 8914] [--cpu]    발음 진단 + 점수 3종 (deploy/mdd-api)
  bash deploy/serve.sh nat [--port 8905]            국적 분류 KD·ONNX·CPU (deploy/nationality-kd-demo)

호출 예시 (서버가 뜬 뒤):
  curl -s localhost:8914/health
  curl -s -X POST localhost:8914/diagnose -F "audio=@sample.wav" \
       -F "sentence=안녕하세요" -F "session_id=spk1"
  curl -s -X POST localhost:8905/predict  -F "audio=@sample.wav"

주의:
  - mdd 는 fairseq_xlsr 환경 + .env 경로 설정 필요. 첫 적재 수십 초~수 분.
  - nat 은 onnxruntime + ffmpeg 만으로 CPU 에서 돈다. 모델 위치가 기본값과 다르면
    KD_MODEL_DIR=<encoder_kd_l17.onnx 폴더> 를 앞에 붙인다.
  - API 에 인증이 없다. 외부 공개 시 앞단 접근 제어 필수 (deploy/README.md).
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/tools/env.sh"

TASK="${1:-}"; [ $# -gt 0 ] && shift
case "$TASK" in -h|--help) usage; exit 0 ;; esac
PORT="" CPU=0
while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --cpu)  CPU=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "알 수 없는 인자: $1" >&2; usage; exit 2 ;;
    esac
done

check_port() {   # 이미 쓰는 포트를 뺏지 않는다 (CLAUDE.md)
    if (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; then
        exec 3>&- 2>/dev/null || true
        echo "포트 $1 이 이미 사용 중이다. --port 로 빈 포트를 지정할 것." >&2
        exit 1
    fi
}

case "$TASK" in
    mdd)
        PORT="${PORT:-8914}"; check_port "$PORT"
        echo "[serve] 발음 진단 API → http://0.0.0.0:$PORT  (POST /diagnose · GET /health)"
        [ "$CPU" = 1 ] && export MDD_FORCE_CPU=1
        exec "$PY" "$ROOT/deploy/mdd-api/mdd_infer_api.py" --port "$PORT"
        ;;
    nat)
        PORT="${PORT:-8905}"; check_port "$PORT"
        echo "[serve] 국적 분류 API → http://0.0.0.0:$PORT  (POST /predict · GET /health)"
        exec "$PY" "$ROOT/deploy/nationality-kd-demo/app.py" --port "$PORT"
        ;;
    *)  usage; exit 2 ;;
esac
