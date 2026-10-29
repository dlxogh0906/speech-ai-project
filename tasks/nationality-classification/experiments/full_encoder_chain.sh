#!/usr/bin/env bash
# 국적 — XLS-R 300m vs 2B 인코더 전체 미세조정 + 클래스 불균형 · 다운샘플링 비교 체인 (2026-10-18). tmux 에서 돌린다.
#
#   1. 음성 복원 — 남은 zip 이 있으면 AIHub 승인부터 확인(probe)하고, 만료면 30분마다 다시 본다. 살아나면 받는다
#   2. 경로를 바꾼 매니페스트(lab131full_zeroth_r16k)를 만든다 — 음성이 없는 행이 0.1% 를 넘으면 멈춘다
#   3. 학습률 짧은 비교: 학습 행 30% · 1 epoch — 300m {1e-5, 3e-5} · 2B {3e-6, 1e-5} (1e-5 는 둘 다 → 같은 학습률 비교도 된다)
#   4. 300m 본 학습 (최대 5 epoch, 기준이 평가 10번 ≈ 1 epoch 동안 안 오르면 멈춤) → run_zeroth 와 비교 (compare_300m.md)
#   5. 클래스 불균형 보정(300m, 대조군 = 4): 모든 클래스를 가장 적은 클래스의 행 수로 맞춘 목록(capmin) ·
#      손실에 클래스 가중치 N/(C·n_c)(cwinv) → compare_imbalance.md — 2026-10-19 요청으로 2B 앞에
#   6. 2B 본 학습 → run_zeroth · 300m · 2B 비교 (compare.md) — 2026-10-19 요청으로 다운샘플링 앞으로 옮겼다
#   7. 다운샘플링: 클래스당 상한 1만 · 2천 학습 목록으로 300m 을 같은 학습률 · 같은 최대 스텝 · 같은 조기 종료로 학습
#      → 대조군(4의 300m)과 비교 (compare_downsampling.md)
#   8. Qwen3-TTS VoiceDesign 증강: 41클래스 모두에 새 목소리 600발화씩(tts_voicedesign_job.sh)을 train 에 더해 300m 을
#      대조군과 같은 설정으로 학습 → compare_tts.md · tts_gate_check.md. <출력>/SKIP_TTS 파일이 있으면 건너뛴다
#   평가는 모두 같은 시험지(test_koraug_zeroth, 사용자 녹음·kor001 제외) · 같은 경로 · 화자 부트스트랩 95% 구간
#
#   bash tasks/nationality-classification/experiments/full_encoder_chain.sh
#   SKIP_SWEEP=1 LR_300M=1e-5 LR_2B=3e-6 bash ...   # 비교 건너뛰기
#   IMB_ARMS="" bash ...       # 불균형 보정 건너뛰기 (기본 "capmin cwinv", cwsqrt 도 된다)
#   DS_CAPS="" bash ...        # 다운샘플링 건너뛰기 (기본 "10000 2000")
#   TTS_AUG="" bash ...        # TTS 증강 건너뛰기
#   WAIT_PID=<pid> bash ...    # 복원 뒤, 그 프로세스(앞 체인 — 매니페스트를 읽는 중)가 끝나야 2단계로
#   PARTIAL=1 bash ...         # 복원을 기다리지 않고 지금 있는 음성만으로 (빠진 행 허용 — run_zeroth 와 데이터가 달라진다)
#   DATA_DIR=<폴더> OUT_DIR=<폴더> bash ...   # 이미 만든 매니페스트 폴더로(2단계를 건너뛴다) — 예: 화자 8:1:1 분할
#                                           # (make_speaker_split.py → lab131full_zeroth_r16k_spk811, 2026-10-19)
#   로그: $(dirname $NATIONALITY_OUTPUTS)/logs/nat_fullenc_chain.log · 마지막 줄 CHAIN_EXIT=<코드>
set -eo pipefail
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$ROOT/tools/env.sh" >/dev/null
T="$ROOT/tasks/nationality-classification"
W="$NATIONALITY_OUTPUTS/webmic_kd"
D="${DATA_DIR:-$W/lab131full_zeroth_r16k}"
TEST="$D/test_koraug_zeroth.csv"
F="${OUT_DIR:-$W/full_encoder_20261018}"
LOGS="$(dirname "$NATIONALITY_OUTPUTS")/logs"
RESTORE="$T/scripts/data/restore_nationality_audio.py"
CAPPER="$T/scripts/data/make_capped_manifest.py"
TTSV="$T/scripts/data/tts_voicedesign.py"
TT="$NATIONALITY_OUTPUTS/tts_voicedesign_20261018"                 # 합성 음성 (tts_voicedesign_job.sh)
FT="$T/scripts/train/finetune_full_encoder.py"
CMP="$T/scripts/eval/compare_logits_bootstrap.py"
mkdir -p "$F" "$LOGS"
# 종료 코드는 여기서 남긴다 — 바깥에서 `| tee; echo $?` 로 받으면 tee 의 코드(늘 0)가 찍힌다
exec > >(tee -a "$LOGS/nat_fullenc_chain.log") 2>&1
trap 'echo "CHAIN_EXIT=$?"' EXIT
say() { echo "[chain $(date '+%m-%d %H:%M:%S')] $*"; }
say "시작 (pid $$)${WAIT_PID:+ · 앞 체인 pid $WAIT_PID}${PARTIAL:+ · PARTIAL}${DATA_DIR:+ · 데이터 $D}"

# 1) 음성 복원 — 승인이 만료된 데이터셋이 있으면 살아날 때까지 기다린다
if [[ -z "${PARTIAL:-}" ]]; then
  for t in 1 2 3; do
    waited=""
    until "$PY" "$RESTORE" probe >> "$LOGS/nat_restore.log" 2>&1; do
      [[ -n "$waited" ]] || say "AIHub 승인이 만료된 데이터셋이 있다 — 30분마다 다시 확인 ($LOGS/nat_restore.log)"
      waited=1; sleep 1800
    done
    if "$PY" "$RESTORE" run --workers 6 >> "$LOGS/nat_restore.log" 2>&1; then restored=1; break; fi
    say "복원 실패 (시도 $t/3) — $LOGS/nat_restore.log"; sleep 1800
  done
  [[ -n "${restored:-}" ]] || { say "복원을 끝내지 못했다"; exit 1; }
  say "복원 끝"
fi

# 2) 매니페스트 — 앞 체인이 아직 그 매니페스트로 학습률 비교를 돌리면 끝나기를 기다린다
if [[ -n "${WAIT_PID:-}" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  say "앞 체인(pid $WAIT_PID) 끝"
fi
if [[ -z "${DATA_DIR:-}" ]]; then
  "$PY" "$RESTORE" manifest --dst "$D" ${PARTIAL:+--max_missing_frac 1} | tee "$F/manifest.log"
else                                     # 따로 만든 폴더 — 다시 쓰지 않는다
  for f in train.csv valid.csv test_koraug_zeroth.csv; do
    [[ -f "$D/$f" ]] || { say "DATA_DIR 에 $f 가 없다 — $D"; exit 1; }
  done
fi

train() {   # 백본 학습률 출력폴더 [추가 인자...]
  local bb="$1" lr="$2" out="$3"; shift 3
  if [[ -f "$out/final_ema.pt" ]]; then say "건너뜀 $out (끝남)"; return 0; fi
  local extra=(); [[ -f "$out/last.pt" ]] && extra+=(--resume)
  say "학습 $bb lr $lr → $out"
  "$PY" "$FT" --backbone "$bb" --lr "$lr" --data_dir "$D" --out_dir "$out" "${extra[@]}" "$@" 2>&1 \
    | grep --line-buffered -v -e 'Warning' -e 'warnings.warn' -e 'return F.conv1d' >> "$out.log"
}
best_of() {   # 폴더들 → 기준이 가장 높은 폴더
  "$PY" - "$@" <<'EOF'
import json, sys
best = max(sys.argv[1:], key=lambda d: max(e["crit"] for e in json.load(open(d + "/history.json"))["evals"]))
print(best)
EOF
}
lr_of() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1] + '/history.json'))['cfg']['lr'])" "$1"; }
n_rows() { "$PY" -c "import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], encoding='utf-8'))))" "$1"; }
n_logits() { "$PY" -c "import numpy as np,sys; print(len(np.load(sys.argv[1], allow_pickle=True)['ok']))" "$1"; }
eval_one() {   # 이름 체크포인트 → $F/eval_<이름>/logits.npz. 시험지 행 수가 다른 옛 평가는 다시 한다
  local name="$1" ck="$2" lg="$F/eval_$1/logits.npz"
  if [[ -f "$lg" && "$(n_logits "$lg")" == "$(n_rows "$TEST")" ]]; then return 0; fi
  say "평가 $name"
  "$PY" "$FT" --eval_only --ckpt "$ck" --test "$TEST" --out "$F/eval_$name" 2>&1 | tail -2
}

# 3) 학습률 짧은 비교
if [[ -z "${SKIP_SWEEP:-}" ]]; then
  for x in "300m 1e-5" "300m 3e-5" "2b 3e-6" "2b 1e-5"; do
    set -- $x
    train "$1" "$2" "$F/sweep_$1_lr$2" --epochs 1 --limit 232000 --patience 0 --save_last_every 0
  done
  LR_300M="${LR_300M:-$(lr_of "$(best_of "$F/sweep_300m_lr1e-5" "$F/sweep_300m_lr3e-5")")}"
  LR_2B="${LR_2B:-$(lr_of "$(best_of "$F/sweep_2b_lr3e-6" "$F/sweep_2b_lr1e-5")")}"
fi
say "본 학습 학습률: 300m $LR_300M · 2B $LR_2B"

# 4) 300m 본 학습 — 폴더 자리에 파일이 있으면 앞 체인을 본 학습 직전에 멈추려고 둔 자리표시다
for p in "$F/full_300m" "$F/full_2b"; do
  if [[ -f "$p" ]]; then rm -f "$p" "$p.log"; say "자리표시 $p 지움"; fi
done
train 300m "$LR_300M" "$F/full_300m"
eval_one zeroth "$W/run_zeroth/ep0.pt"
eval_one full300m "$F/full_300m/best.pt"
"$PY" "$CMP" --test "$TEST" --train "$D/train.csv" --out "$F/compare_300m.md" \
  --models "zeroth=$F/eval_zeroth/logits.npz" "full300m=$F/eval_full300m/logits.npz" > /dev/null
say "300m 비교 — $F/compare_300m.md"

# 5) 클래스 불균형 보정 (2026-10-19 사용자 요청 — 2B 보다 먼저). 대조군 = full_300m(보정 없음). 300m · 학습률 · 조기 종료는 같다
#    capmin: 모든 클래스를 가장 적은 클래스의 행 수로 줄인다(클래스 안 출처 비율 유지). 최대 스텝은 대조군 5 epoch 와 같게
#    cwinv · cwsqrt: 데이터는 대조군 그대로, 학습 손실에 클래스 가중치 N/(C·n_c) · 그 제곱근
imb_models=("full300m=$F/eval_full300m/logits.npz")
for arm in ${IMB_ARMS-capmin cwinv}; do
  case "$arm" in
    capmin)
      Dc="${D}_capmin"
      if [[ ! -f "$F/imb_capmin/last.pt" && ! -f "$F/imb_capmin/final_ema.pt" ]]; then
        "$PY" "$CAPPER" --src "$D" --cap min --dst "$Dc" | tee "$F/manifest_capmin.log"
      fi
      ep=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['matched_epochs'])" "$Dc/cap_meta.json")
      train 300m "$LR_300M" "$F/imb_capmin" --data_dir "$Dc" --epochs "$ep" ;;
    cwinv)  train 300m "$LR_300M" "$F/imb_cwinv" --class_weight inv ;;
    cwsqrt) train 300m "$LR_300M" "$F/imb_cwsqrt" --class_weight sqrt ;;
    *) say "IMB_ARMS 에 모르는 항목 $arm"; exit 1 ;;
  esac
  eval_one "imb_$arm" "$F/imb_$arm/best.pt"
  imb_models+=("imb_$arm=$F/eval_imb_$arm/logits.npz")
done
if (( ${#imb_models[@]} > 1 )); then
  "$PY" "$CMP" --test "$TEST" --train "$D/train.csv" --out "$F/compare_imbalance.md" --models "${imb_models[@]}" > /dev/null
  say "불균형 보정 비교 — $F/compare_imbalance.md"
fi

# 6) 2B 본 학습 → 셋 비교 (2026-10-19 사용자 요청으로 다운샘플링보다 먼저)
train 2b "$LR_2B" "$F/full_2b"
eval_one full2b "$F/full_2b/best.pt"
"$PY" "$CMP" --test "$TEST" --train "$D/train.csv" --out "$F/compare.md" \
  --models "zeroth=$F/eval_zeroth/logits.npz" "full300m=$F/eval_full300m/logits.npz" "full2b=$F/eval_full2b/logits.npz" > /dev/null
say "2B 비교 — $F/compare.md"

# 7) 다운샘플링 — 클래스 비율만 바꾼다(클래스 안 출처 비율은 그대로, 화자는 번갈아 뽑아 최대한 남긴다).
#    학습률 · 최대 스텝(대조군 5 epoch 와 같은 스텝) · 평가 간격 · 조기 종료는 대조군 full_300m 과 같다
ds_models=("full300m=$F/eval_full300m/logits.npz")
for cap in ${DS_CAPS-10000 2000}; do
  tag="cap$((cap / 1000))k"; Dc="${D}_$tag"
  if [[ ! -f "$F/ds_$tag/last.pt" && ! -f "$F/ds_$tag/final_ema.pt" ]]; then   # 학습 전이면 지금 $D 에서 다시 만든다
    "$PY" "$CAPPER" --src "$D" --cap "$cap" --dst "$Dc" | tee "$F/manifest_$tag.log"
  fi
  ep=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['matched_epochs'])" "$Dc/cap_meta.json")
  train 300m "$LR_300M" "$F/ds_$tag" --data_dir "$Dc" --epochs "$ep"
  eval_one "ds_$tag" "$F/ds_$tag/best.pt"
  ds_models+=("ds_$tag=$F/eval_ds_$tag/logits.npz")
done
if (( ${#ds_models[@]} > 1 )); then
  "$PY" "$CMP" --test "$TEST" --train "$D/train.csv" --out "$F/compare_downsampling.md" --models "${ds_models[@]}" > /dev/null
  say "다운샘플링 비교 — $F/compare_downsampling.md"
fi

# 8) Qwen3-TTS VoiceDesign 증강 — 맨 뒤. 억양 점검(gate_gate.md)에서 억양 신호가 거의 없어(lift 1.15) 2B 뒤로 미뤘다.
#    합성 음성이 다 만들어지기(synth_aug.csv)를 기다린다. train 에만 더하고 나머지는 대조군과 같다.
#    $F/SKIP_TTS 파일을 만들면 체인을 다시 띄우지 않고 건너뛴다
tts_on() { [[ -n "${TTS_AUG-1}" && ! -f "$F/SKIP_TTS" ]]; }
if tts_on; then
  until [[ -f "$TT/synth_aug.csv" ]] || ! tts_on; do sleep 600; done
fi
if tts_on; then
  Dt="${D}_ttsvd"
  if [[ ! -f "$F/tts_vd/last.pt" && ! -f "$F/tts_vd/final_ema.pt" ]]; then
    "$PY" "$TTSV" build --src "$D" --synth "$TT/synth_aug.csv" --dst "$Dt" | tee "$F/manifest_ttsvd.log"
  fi
  train 300m "$LR_300M" "$F/tts_vd" --data_dir "$Dt"
  eval_one tts_vd "$F/tts_vd/best.pt"
  "$PY" "$CMP" --test "$TEST" --train "$D/train.csv" --out "$F/compare_tts.md" \
    --models "full300m=$F/eval_full300m/logits.npz" "tts_vd=$F/eval_tts_vd/logits.npz" > /dev/null
  # 학습에 안 쓴 합성 목소리(gate)를 두 모델로 판정 — 합성에서만 오르고 실제 사람에서 안 오르면 TTS 특유의 단서를 배운 것
  for m in "full300m=$F/full_300m/best.pt" "tts_vd=$F/tts_vd/best.pt"; do
    n="${m%%=*}"
    [[ -f "$F/gate_$n/logits.npz" ]] || \
      "$PY" "$FT" --eval_only --ckpt "${m#*=}" --test "$TT/synth_gate.csv" --out "$F/gate_$n" 2>&1 | tail -1
  done
  "$PY" "$TTSV" gate --split gate --out "$TT" --report "$F/tts_gate_check.md" \
    --models "full300m=$F/gate_full300m/logits.npz" "tts_vd=$F/gate_tts_vd/logits.npz" > /dev/null
  say "TTS 증강 비교 — $F/compare_tts.md · $F/tts_gate_check.md"
else
  say "TTS 증강 건너뜀"
fi
say "끝"
