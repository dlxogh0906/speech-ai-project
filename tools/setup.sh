#!/usr/bin/env bash
# 클론 직후 **한 방에** — 환경 구축 + .env 작성 + 두 task 추론 자체검증까지.
#
# 사용법:
#   bash tools/setup.sh <가중치폴더>
#
# 예:
#   bash tools/setup.sh ~/speech-ai-weights
#
# <가중치폴더> 는 배포 번들의 `final/` 만 있으면 된다 (2.1 GB):
#   final/pronunciation/mdd_tag_all_native.pt          발음 진단 모델
#   final/pronunciation/dict.ltr.txt                   자모 사전 (없으면 디코딩 불가)
#   final/pronunciation/score_models_v2_native.joblib  점수 3종 회귀기
#   final/nationality/encoder_kd_l17.onnx              국적 인코더
#   final/nationality/head_np.npz                      국적 헤드
#
# 환경변수로 조정(선택):
#   ENV_NAME=speech-ai            conda env 이름
#   ENV_PREFIX=<경로>             이름 대신 그 경로에 만든다(홈 디스크가 빠듯할 때)
#   SKIP_ENV=1                    환경 구축을 건너뛴다(.env 작성 + 검증만)
#   SKIP_TEST=1                   자체검증을 건너뛴다
#
# 하는 일:
#   1. 가중치 폴더 검사        — 없는 파일이 있으면 여기서 멈춘다
#   2. bash tools/create_env.sh — conda env + torch + requirements + fairseq + KoG2P
#   3. .env 자동 작성           — 손으로 채울 게 없다 (기존 파일은 백업한다)
#   4. 자체검증                 — 두 추론 스크립트를 실제로 돌려 JSON 이 나오는지 본다
set -uo pipefail

case "${1:-}" in
    -h|--help|"")
        awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
        [ -z "${1:-}" ] && exit 2 || exit 0 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS="$(cd "$1" 2>/dev/null && pwd)" || { echo "[fatal] 가중치 폴더가 없다: $1" >&2; exit 1; }

ok()   { echo "  ✓ $*"; }
fail() { echo "  ✗ $*" >&2; }
step() { echo; echo "━━━ $* ━━━"; }

# ── 1. 가중치 검사 ────────────────────────────────────────────────────────
step "1/4 가중치 검사  ($WEIGHTS)"
MDD_CKPT_F="$WEIGHTS/final/pronunciation/mdd_tag_all_native.pt"
MDD_DICT_D="$WEIGHTS/final/pronunciation"
MDD_SCORER_F="$WEIGHTS/final/pronunciation/score_models_v2_native.joblib"
NAT_DIR="$WEIGHTS/final/nationality"

missing=0
for f in "$MDD_CKPT_F" "$MDD_DICT_D/dict.ltr.txt" "$MDD_SCORER_F" \
         "$NAT_DIR/encoder_kd_l17.onnx" "$NAT_DIR/head_np.npz"; do
    if [ -f "$f" ]; then ok "${f#$WEIGHTS/}"; else fail "없음: ${f#$WEIGHTS/}"; missing=1; fi
done
if [ "$missing" = 1 ]; then
    echo >&2
    echo "[fatal] 가중치가 부족하다. 배포 번들의 final/ 폴더를 통째로 두고 그 상위를 넘길 것." >&2
    exit 1
fi
[ -f "$WEIGHTS/final/SHA256SUMS.txt" ] && \
    ( cd "$WEIGHTS/final" && sha256sum -c --quiet SHA256SUMS.txt 2>/dev/null \
      && ok "SHA256 검증 통과" || fail "SHA256 불일치 — 전송이 깨졌을 수 있다" )

# ── 2. 환경 구축 ──────────────────────────────────────────────────────────
FAIRSEQ_DIR="${FAIRSEQ_ROOT:-$(dirname "$ROOT")/fairseq}"
G2P_DIR="${G2P_TOOLS:-$(dirname "$ROOT")/g2p_tools}"
OUT_DIR="${SPEECH_AI_OUT:-$(dirname "$ROOT")/speech-ai-out}"

if [ "${SKIP_ENV:-0}" = 1 ]; then
    step "2/4 환경 구축 — SKIP_ENV=1 이라 건너뛴다"
else
    step "2/4 환경 구축"
    FAIRSEQ_ROOT="$FAIRSEQ_DIR" G2P_TOOLS="$G2P_DIR" bash "$ROOT/tools/create_env.sh" || {
        echo "[fatal] 환경 구축 실패 — 위 로그를 볼 것." >&2; exit 1; }
fi

# 방금 만든 환경의 파이썬을 찾는다.
# `conda run` 출력은 파싱하지 않는다 — 경로 뒤에 빈 줄을 붙이는 버전이 있다(실측).
if [ -n "${ENV_PREFIX:-}" ]; then
    PY_BIN="$ENV_PREFIX/bin/python"
else
    _cb="$(conda info --base 2>/dev/null)" \
        && . "$_cb/etc/profile.d/conda.sh" \
        && conda activate "${ENV_NAME:-speech-ai}" \
        || { echo "[fatal] conda activate 실패: ${ENV_NAME:-speech-ai}" >&2; exit 1; }
    PY_BIN="$CONDA_PREFIX/bin/python"
fi
[ -x "$PY_BIN" ] || { echo "[fatal] 환경의 python 을 못 찾았다: [$PY_BIN]" >&2; exit 1; }
ok "python: $PY_BIN"

# ── 3. .env 작성 ──────────────────────────────────────────────────────────
step "3/4 .env 작성"
mkdir -p "$OUT_DIR/mdd/analysis" "$OUT_DIR/nationality/data" "$OUT_DIR/nationality/runs"
if [ -f "$ROOT/.env" ]; then
    bak="$ROOT/.env.bak.$(date +%Y%m%d%H%M%S)"
    cp "$ROOT/.env" "$bak" && ok "기존 .env 를 백업했다 → ${bak#$ROOT/}"
fi
cat > "$ROOT/.env" <<EOF
# tools/setup.sh 가 자동 생성했다. 손으로 고쳐도 된다.
# 가중치: $WEIGHTS

# ── 추론에 실제로 쓰이는 값 ────────────────────────────────
# 발음 진단 — 배포 체크포인트와 점수 회귀기. 이 둘은 **짝이다**(같은 특징으로 학습됨).
MDD_SERVE_CKPT=$MDD_CKPT_F
MDD_SERVE_SCORER=$MDD_SCORER_F
# 체크포인트에 박힌 data 경로는 학습 서버 것이라 여기 없다. dict.ltr.txt 위치를 알려준다.
MDD_DATA_DIR=$MDD_DICT_D
# 국적 — KD 인코더(ONNX) + 헤드
KD_MODEL_DIR=$NAT_DIR

# ── 도구 ───────────────────────────────────────────────────
PY=$PY_BIN
FAIRSEQ_ROOT=$FAIRSEQ_DIR
G2P_TOOLS=$G2P_DIR
FFMPEG_BIN=ffmpeg

# ── 산출물 ─────────────────────────────────────────────────
MDD_OUTPUTS=$OUT_DIR/mdd
MDD_ANALYSIS=$OUT_DIR/mdd/analysis
NATIONALITY_OUTPUTS=$OUT_DIR/nationality
NATIONALITY_DATA=$OUT_DIR/nationality/data
NATIONALITY_RUNS=$OUT_DIR/nationality/runs

# ── 학습·평가에만 필요 (추론만 할 거면 비워 둬도 된다) ─────
# SPEECH_AI_DATA=/path/to/corpus          AIHub 오디오 + mdd_ckpt_*/ 원본
# SPEECH_AI_MODELS=/path/to/models        XLS-R 백본 (xlsr2_300m.pt · xlsr2_2B_1000k.pt)
EOF
ok ".env 작성 완료 (손으로 채울 것 없음)"

# ── 4. 자체검증 ───────────────────────────────────────────────────────────
if [ "${SKIP_TEST:-0}" = 1 ]; then
    step "4/4 자체검증 — SKIP_TEST=1 이라 건너뛴다"; exit 0
fi
step "4/4 자체검증 — 두 추론 스크립트를 실제로 돌린다"

# conda 의 LD_LIBRARY_PATH 가 시스템 ffmpeg 를 깨뜨리는 머신이 있다(ENVIRONMENT.md 함정 10).
if ! ffmpeg -version >/dev/null 2>&1 && env -u LD_LIBRARY_PATH ffmpeg -version >/dev/null 2>&1; then
    echo "  ! LD_LIBRARY_PATH 가 ffmpeg 를 깨뜨린다 — 이 검증 동안만 걷어낸다."
    echo "    (셸 프로파일에서 빼지 않으면 서비스 기동 때도 같은 문제가 난다)"
    export -n LD_LIBRARY_PATH; unset LD_LIBRARY_PATH
fi
command -v ffmpeg >/dev/null || { echo "[fatal] ffmpeg 이 없다. apt install ffmpeg" >&2; exit 1; }

SMOKE_WAV="$OUT_DIR/smoke_4s.wav"
ffmpeg -hide_banner -loglevel error -f lavfi \
    -i "anoisesrc=d=4:c=pink:a=0.35,bandpass=f=700:width_type=h:w=600,tremolo=f=5:d=0.7" \
    -ar 16000 -ac 1 -y "$SMOKE_WAV" || { echo "[fatal] 테스트 오디오 생성 실패" >&2; exit 1; }
ok "테스트 오디오 생성 (합성음 — 결과값이 아니라 파이프라인 동작만 본다)"

free_port() {  # 쓰지 않는 포트를 고른다 (기존 서비스를 뺏지 않는다 — CLAUDE.md)
    local p; for p in $(seq "$1" "$(($1 + 40))"); do
        (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null || { echo "$p"; return; }
        exec 3>&- 2>/dev/null
    done; echo "$1"
}

rc_all=0
# 주의: `... | grep -q` 로 쓰지 말 것. grep -q 가 첫 매치에서 파이프를 닫으면 앞단이
# SIGPIPE 로 죽고, pipefail 때문에 **성공한 실행이 실패로 보고된다.** 파일로 받고 검사한다.
LOG_NAT="$(mktemp)"; LOG_MDD="$(mktemp)"
trap 'rm -f "$LOG_NAT" "$LOG_MDD"' EXIT

echo
echo "  [국적] bash deploy/infer_nationality.sh …"
bash "$ROOT/deploy/infer_nationality.sh" "$SMOKE_WAV" --port "$(free_port 8960)" >"$LOG_NAT" 2>&1
if grep -q '"top5"' "$LOG_NAT"; then
    ok "국적 분류 동작 — 1순위 $(grep -o '"label": "[^"]*"' "$LOG_NAT" | head -1 | cut -d'"' -f4)"
else
    fail "국적 분류 실패:"; tail -15 "$LOG_NAT" | sed 's/^/      /' >&2; rc_all=1
fi

echo
echo "  [발음] bash deploy/infer_mdd.sh …  (모델 적재에 수십 초~수 분)"
bash "$ROOT/deploy/infer_mdd.sh" "$SMOKE_WAV" "안녕하세요 만나서 반갑습니다" \
     --port "$(free_port 8970)" >"$LOG_MDD" 2>&1
if grep -q '"canonical"' "$LOG_MDD"; then
    ok "발음 진단 동작 — 음소 $(grep -c '"phone"' "$LOG_MDD")개 판정"
else
    fail "발음 진단 실패:"; tail -15 "$LOG_MDD" | sed 's/^/      /' >&2; rc_all=1
fi

echo
if [ "$rc_all" = 0 ]; then
    cat <<EOF
━━━ 완료 — 두 task 모두 동작한다 ━━━

  bash deploy/infer_nationality.sh <오디오>                  국적 top-5
  bash deploy/infer_mdd.sh <오디오> "<읽은 문장>"             발음 진단 + 점수
  bash deploy/serve.sh nat  --port 8905                      서버 기동
  bash deploy/serve.sh mdd  --port 8914

  ⚠️ 발음 **점수**는 화자 단위로만 유효하다. 같은 --session 으로 20~30발화를
     모아야 하고, 5발화 미만이면 진단만 나온다.
EOF
else
    echo "━━━ 일부 실패 — 위 ✗ 를 볼 것 ━━━" >&2
fi
exit "$rc_all"
