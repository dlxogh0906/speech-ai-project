#!/usr/bin/env bash
# 발음 진단(MDD + 점수 3종) 단건 추론 CLI — deploy/mdd-api 를 감싼다.
#
# 필요: .env 설정(체크포인트·산출물 경로) + fairseq_xlsr 파이썬 환경.
# 서버 첫 적재는 수십 초(GPU)~수 분(CPU) 걸린다. 여러 건을 돌릴 거면 --keep 으로
# 서버를 남겨 두고 --url 로 재사용할 것.
#
# ⚠️ 점수는 화자 단위로만 유효하다(발화 1개 PCC 0.43). 같은 --session 으로
#    5발화 이상 누적해야 점수가 나오고, 그 전에는 진단만 반환된다(score=pending).
set -euo pipefail

usage() {
    cat <<'EOF'
사용법:
  bash deploy/infer_mdd.sh <오디오파일> "<정답 문장>" [옵션]

옵션:
  --url URL      이미 떠 있는 mdd-api 를 쓴다 (서버를 새로 띄우지 않음)
  --port N       임시 서버 포트 (기본 8931)
  --keep         추론 후 서버를 끄지 않고 남긴다 (여러 건 돌릴 때 모델 재적재 회피)
  --session ID   세션 누적 — 같은 ID 로 5발화 이상이면 점수 3종이 나온다
  --cpu          GPU 없이 강제 CPU (MDD_FORCE_CPU=1)

예시:
  bash deploy/infer_mdd.sh sample.wav "안녕하세요 만나서 반갑습니다"
  bash deploy/infer_mdd.sh u1.wav "문장1" --keep --session spk1     # 서버·세션 유지
  bash deploy/infer_mdd.sh u2.wav "문장2" --url http://127.0.0.1:8931 --session spk1
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/tools/env.sh"

AUDIO="" SENTENCE="" URL="" PORT=8931 KEEP=0 SESSION="" FORCE_CPU=0
while [ $# -gt 0 ]; do
    case "$1" in
        --url)     URL="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --keep)    KEEP=1; shift ;;
        --cpu)     FORCE_CPU=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)  if   [ -z "$AUDIO" ];    then AUDIO="$1"
            elif [ -z "$SENTENCE" ]; then SENTENCE="$1"
            else echo "알 수 없는 인자: $1" >&2; usage; exit 2; fi
            shift ;;
    esac
done
[ -n "$AUDIO" ] && [ -n "$SENTENCE" ] || { usage; exit 2; }
[ -f "$AUDIO" ] || { echo "오디오 파일이 없다: $AUDIO" >&2; exit 2; }

SRV_PID=""
LOG="$(mktemp "${TMPDIR:-/tmp}/mdd_infer_XXXXXX.log")"
cleanup() { [ -n "$SRV_PID" ] && [ "$KEEP" = 0 ] && kill "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [ -z "$URL" ]; then
    # 이미 쓰는 포트를 뺏지 않는다 (CLAUDE.md) — 점유 중이면 띄우지 않고 안내만
    if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
        exec 3>&- 2>/dev/null || true
        echo "포트 $PORT 가 이미 사용 중이다." >&2
        echo "  그 서버가 mdd-api 면:  --url http://127.0.0.1:$PORT" >&2
        echo "  새로 띄우려면:        --port <빈 포트>" >&2
        exit 1
    fi
    echo "[serve] mdd-api 기동 중 (포트 $PORT · 로그 $LOG) — 모델 적재에 수십 초~수 분" >&2
    [ "$FORCE_CPU" = 1 ] && export MDD_FORCE_CPU=1
    "$PY" "$ROOT/deploy/mdd-api/mdd_infer_api.py" --port "$PORT" >"$LOG" 2>&1 &
    SRV_PID=$!
    URL="http://127.0.0.1:$PORT"
    READY=0
    for _ in $(seq 1 300); do          # 최대 10분 (uvicorn 은 모델 적재가 끝나야 뜬다)
        if curl -sf "$URL/health" >/dev/null 2>&1; then READY=1; break; fi
        kill -0 "$SRV_PID" 2>/dev/null || { echo "서버가 기동 중 죽었다 — 로그: $LOG" >&2; exit 1; }
        sleep 2
    done
    [ "$READY" = 1 ] || { echo "서버가 10분 안에 준비되지 않았다 — 로그: $LOG" >&2; exit 1; }
fi

CURL=( curl -s -X POST "$URL/diagnose" -F "audio=@$AUDIO" -F "sentence=$SENTENCE" )
[ -n "$SESSION" ] && CURL+=( -F "session_id=$SESSION" )

OUT="$(mktemp "${TMPDIR:-/tmp}/mdd_resp_XXXXXX.json")"
HTTP="$("${CURL[@]}" -o "$OUT" -w '%{http_code}')" || { echo "서버에 연결 실패: $URL" >&2; exit 1; }
if [ "$HTTP" != 200 ]; then
    echo "요청 실패 (HTTP $HTTP):" >&2; cat "$OUT" >&2; echo >&2; exit 1
fi
"$PY" -c 'import sys,json; print(json.dumps(json.load(open(sys.argv[1])), ensure_ascii=False, indent=2))' "$OUT"
rm -f "$OUT"

if [ -n "$SRV_PID" ] && [ "$KEEP" = 1 ]; then
    echo "[keep] 서버 유지 — PID $SRV_PID · $URL" >&2
    echo "       재사용:  bash deploy/infer_mdd.sh <오디오> \"<문장>\" --url $URL" >&2
    echo "       종료:    kill $SRV_PID" >&2
fi
