#!/usr/bin/env bash
# 국적/모국어 분류 단건 추론 CLI — deploy/nationality-kd-demo(KD 인코더, ONNX)를 감싼다.
#
# 필요: .env 설정 + onnxruntime·fastapi (requirements-nationality.txt) + ffmpeg.
# torch·fairseq·GPU 불필요 — CPU 만으로 돈다. 모델 위치가 기본값과 다르면
# KD_MODEL_DIR=<encoder_kd_l17.onnx·head_np.npz 가 있는 폴더> 로 지정한다.
set -euo pipefail

usage() {
    cat <<'EOF'
사용법:
  bash deploy/infer_nationality.sh <오디오파일> [옵션]

옵션:
  --url URL   이미 떠 있는 서버를 쓴다 (서버를 새로 띄우지 않음)
  --port N    임시 서버 포트 (기본 8932)
  --keep      추론 후 서버를 끄지 않고 남긴다 (여러 건 돌릴 때 재적재 회피)

출력: top-5 후보와 확률 JSON. 오디오는 ffmpeg 이 읽는 포맷 전부 가능(최대 12초 사용).
예시:
  bash deploy/infer_nationality.sh sample.wav
  KD_MODEL_DIR=/path/to/deploy bash deploy/infer_nationality.sh sample.wav
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/tools/env.sh"

AUDIO="" URL="" PORT=8932 KEEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --url)  URL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)  if [ -z "$AUDIO" ]; then AUDIO="$1"
            else echo "알 수 없는 인자: $1" >&2; usage; exit 2; fi
            shift ;;
    esac
done
[ -n "$AUDIO" ] || { usage; exit 2; }
[ -f "$AUDIO" ] || { echo "오디오 파일이 없다: $AUDIO" >&2; exit 2; }

SRV_PID=""
LOG="$(mktemp "${TMPDIR:-/tmp}/nat_infer_XXXXXX.log")"
cleanup() { [ -n "$SRV_PID" ] && [ "$KEEP" = 0 ] && kill "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [ -z "$URL" ]; then
    # 이미 쓰는 포트를 뺏지 않는다 (CLAUDE.md)
    if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
        exec 3>&- 2>/dev/null || true
        echo "포트 $PORT 가 이미 사용 중이다.  --url http://127.0.0.1:$PORT 또는 --port <빈 포트>" >&2
        exit 1
    fi
    echo "[serve] nationality-kd-demo 기동 중 (포트 $PORT · 로그 $LOG) — ONNX 적재 수십 초" >&2
    "$PY" "$ROOT/deploy/nationality-kd-demo/app.py" --port "$PORT" >"$LOG" 2>&1 &
    SRV_PID=$!
    URL="http://127.0.0.1:$PORT"
    READY=0
    for _ in $(seq 1 150); do          # 최대 5분
        if curl -sf "$URL/health" >/dev/null 2>&1; then READY=1; break; fi
        kill -0 "$SRV_PID" 2>/dev/null || { echo "서버가 기동 중 죽었다 — 로그: $LOG" >&2; exit 1; }
        sleep 2
    done
    [ "$READY" = 1 ] || { echo "서버가 5분 안에 준비되지 않았다 — 로그: $LOG" >&2; exit 1; }
fi

OUT="$(mktemp "${TMPDIR:-/tmp}/nat_resp_XXXXXX.json")"
HTTP="$(curl -s -X POST "$URL/predict" -F "audio=@$AUDIO" -o "$OUT" -w '%{http_code}')" || { echo "서버에 연결 실패: $URL" >&2; exit 1; }
if [ "$HTTP" != 200 ]; then
    echo "요청 실패 (HTTP $HTTP):" >&2; cat "$OUT" >&2; echo >&2; exit 1
fi
"$PY" -c 'import sys,json; print(json.dumps(json.load(open(sys.argv[1])), ensure_ascii=False, indent=2))' "$OUT"
rm -f "$OUT"

if [ -n "$SRV_PID" ] && [ "$KEEP" = 1 ]; then
    echo "[keep] 서버 유지 — PID $SRV_PID · $URL" >&2
    echo "       재사용:  bash deploy/infer_nationality.sh <오디오> --url $URL" >&2
    echo "       종료:    kill $SRV_PID" >&2
fi
