#!/usr/bin/env bash
# 국적 — Qwen3-TTS VoiceDesign 증강 음성 만들기 (2026-10-18). tmux nat-tts 에서 돌린다.
#
#   1. plan — 목소리 설명 · 문장 배정 (41클래스 × gate 6목소리×4문장 · aug 40목소리×15문장)
#   2. gate 984발화 생성 → run_zeroth 판정 · whisper 문자 오류율 → gate_gate.md (억양 신호가 있는지 먼저 본다)
#   3. WAIT_PID 가 끝나면 aug 24,600발화를 3개 프로세스로 생성 → 죽은 조각은 남은 것만 다시(최대 3번) → synth_aug.csv
#      full_encoder_chain.sh 의 TTS 단계가 synth_aug.csv 를 기다렸다가 학습한다
#
#   WAIT_PID=<학습률 비교 체인 pid> bash tasks/nationality-classification/experiments/tts_voicedesign_job.sh
#   로그: $(dirname $NATIONALITY_OUTPUTS)/logs/nat_tts.log · 마지막 줄 TTS_EXIT=<코드>
set -eo pipefail
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$ROOT/tools/env.sh" >/dev/null
T="$ROOT/tasks/nationality-classification"
W="$NATIONALITY_OUTPUTS/webmic_kd"
O="$NATIONALITY_OUTPUTS/tts_voicedesign_20261018"
S="$T/scripts/data/tts_voicedesign.py"
FT="$T/scripts/train/finetune_full_encoder.py"
LOGS="$(dirname "$NATIONALITY_OUTPUTS")/logs"
QPY="${QWENTTS_PY:-/data/speech-ai/envs/qwentts/bin/python}"      # qwen-tts · transformers 가 깔린 별도 환경
export HF_HOME="${HF_HOME:-/data/speech-ai/hf_cache}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$O" "$LOGS"
exec > >(tee -a "$LOGS/nat_tts.log") 2>&1
trap 'echo "TTS_EXIT=$?"' EXIT
say() { echo "[tts $(date '+%m-%d %H:%M:%S')] $*"; }
quiet() { grep --line-buffered -v -i -E "warn|pad_token|Fetching" || true; }
say "시작 (pid $$)${WAIT_PID:+ · aug 는 pid $WAIT_PID 가 끝난 뒤}"

# 1 · 2) 계획 → gate 생성 → 판정
[[ -f "$O/plan.csv" ]] || "$PY" "$S" plan
"$QPY" "$S" gen --split gate 2>&1 | quiet
"$PY" "$S" rows --split gate
[[ -f "$O/eval_gate_zeroth/logits.npz" ]] || \
  "$PY" "$FT" --eval_only --ckpt "$W/run_zeroth/ep0.pt" --test "$O/synth_gate.csv" --out "$O/eval_gate_zeroth" 2>&1 | tail -2
[[ -f "$O/asr_gate.csv" ]] || "$QPY" "$S" asr --split gate 2>&1 | quiet
"$PY" "$S" gate --split gate --models "run_zeroth=$O/eval_gate_zeroth/logits.npz"
say "gate 끝 — $O/gate_gate.md"

# 3) aug — 학습률 비교가 끝난 뒤 (GPU 를 덜 다툰다). 다른 작업에 GPU 메모리를 뺏겨 죽은 조각이 있으면
#    10분 뒤 남은 것만 다시 셋으로 나눠 돌린다(최대 3번). 1% 이하로 남으면 길이가 이상해 버린 것으로 본다
if [[ -n "${WAIT_PID:-}" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  say "앞 체인(pid $WAIT_PID) 끝 — aug 생성 시작"
fi
left() { "$PY" -c "import csv,os,sys; print(sum(1 for r in csv.DictReader(open(sys.argv[1], encoding='utf-8')) if r['split'] == 'aug' and not os.path.exists(r['out'])))" "$O/plan.csv"; }
max_left=$(( $(grep -c '^aug,' "$O/plan.csv") / 100 ))
for t in 1 2 3; do
  pids=()
  for k in 0 1 2; do "$QPY" "$S" gen --split aug --shard "$k" --nshards 3 >> "$O/gen_aug_$k.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || say "aug 조각 pid $p 실패 — $O/gen_aug_*.log"; done   # 맨 wait 는 exec >(tee) 까지 기다려 안 끝난다
  n=$(left); say "aug 남은 발화 ${n}개 (시도 $t/3)"
  (( n <= max_left )) && break
  sleep 600
done
(( n <= max_left )) || { say "aug 를 다 만들지 못했다 — $O/gen_aug_*.log"; exit 1; }
"$PY" "$S" rows --split aug
say "aug 끝 — $O/synth_aug.csv"
