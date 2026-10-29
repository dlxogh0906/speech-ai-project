#!/usr/bin/env bash
# AIHub 037 「교육용 한국인의 영어 음성 데이터」(datasetkey 71463) 전체 받기 — 영어 발음 평가(한국인 학습자)용.
#   PRN = 문장 낭독(발음 평가 · 조음/운율 점수) · SPK = 말하기(ED/OC/PS/PU × EV/IS/LC/OB/OP/PR/TX)
#
#   bash tasks/pronunciation-assessment/scripts/data/aihub_download_037.sh labels   # 라벨 60개 (약 0.2 GB) — 먼저
#   bash tasks/pronunciation-assessment/scripts/data/aihub_download_037.sh prn      # PRN 오디오 (약 33 GB)
#   bash tasks/pronunciation-assessment/scripts/data/aihub_download_037.sh spk      # SPK 오디오 (약 48 GB)
#   bash tasks/pronunciation-assessment/scripts/data/aihub_download_037.sh all      # 위 순서대로 (tmux 에서)
#
# API 키: 환경변수 AIHUB_APIKEY, 없으면 ~/.aihub_apikey 파일(권한 600)에서 읽는다. 명령줄·로그에 키를 쓰지 않는다.
# 저장 위치: $AIHUB_DEST (기본 $SPEECH_AI_DATA/aihub)/037.교육용_한국인의_영어_음성_데이터/01-1.정식개방데이터/…
# 이미 받은 zip 은 건너뛴다(재실행 = 이어받기). 파일 1개를 받을 때 조각(.part*)과 병합본이 잠시 같이 있어 2배 공간이 필요하다.
# ⚠️ AIHub 이용 조건: 재배포·제3자 제공 금지 — 이 오디오를 외부 API(SpeechSuper 등)로 보내지 않는다.

set -eo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi
WHAT="${1:-labels}"
case "$WHAT" in labels|prn|spk|all) ;; *) echo "인자는 labels | prn | spk | all 중 하나: $WHAT" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
source "$ROOT/tools/env.sh" >/dev/null
AIHUBSHELL="${AIHUBSHELL:-$(command -v aihubshell || echo "$HOME/bin/aihubshell")}"
[[ -x "$AIHUBSHELL" ]] || { echo "aihubshell 이 없다. 설치: curl -o ~/bin/aihubshell https://api.aihub.or.kr/api/aihubshell.do && chmod +x ~/bin/aihubshell" >&2; exit 1; }
if [[ -z "${AIHUB_APIKEY:-}" && -r "$HOME/.aihub_apikey" ]]; then AIHUB_APIKEY="$(cat "$HOME/.aihub_apikey")"; fi
[[ -n "${AIHUB_APIKEY:-}" ]] || { echo "AIHub API 키가 없다 — AIHUB_APIKEY 를 export 하거나 ~/.aihub_apikey 에 저장할 것" >&2; exit 1; }
export AIHUB_APIKEY

DEST="${AIHUB_DEST:-$SPEECH_AI_DATA/aihub}"
[[ "$DEST" != *"미설정"* ]] || { echo "SPEECH_AI_DATA 가 설정되지 않았다 (.env)" >&2; exit 1; }
mkdir -p "$DEST"
DS=71463
NUM=037

# 파일키 범위(2026-10-05 AIHub 화면 기준) — 이름은 화면과 같다
#   라벨: TL_PRN 485442 · TL_SPK 485443~485471 · VL_PRN 485502 · VL_SPK 485503~485531
#   PRN 오디오: VS_PRN 485472(3.63 GB) · TS_PRN 485412(29.18 GB)
#   SPK 오디오: VS_SPK 485473~485501 · TS_SPK 485413~485441 (가장 큰 것 TS_SPK_en_PS_OP 11.80 GB)
keys_labels() { echo 485442; seq 485443 485471; echo 485502; seq 485503 485531; }
keys_prn() { echo 485472; echo 485412; }
keys_spk() { seq 485473 485501; seq 485413 485441; }

# 파일키 → 이름은 목록 모드로 알아낸다(받기 전에 이미 있는지 판단하려고)
LIST="$("$AIHUBSHELL" -mode l -datasetkey "$DS" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g')"
name_of() { echo "$LIST" | grep -E "\| *$1 *\|?$|\| *$1\b" | grep -oE "[A-Z]{2}_(PRN|SPK)_en_[A-Z]{2}_[A-Z]{2}\.zip" | head -1; }
find_zip() { find "$DEST" -path "*/$NUM.*" -name "$1" -type f 2>/dev/null | head -1; }

case "$WHAT" in
  labels) KEYS="$(keys_labels)" ;;
  prn) KEYS="$(keys_prn)" ;;
  spk) KEYS="$(keys_spk)" ;;
  all) KEYS="$(keys_labels; keys_prn; keys_spk)" ;;
esac

cd "$DEST"
fail=0
for fk in $KEYS; do
  name="$(name_of "$fk")"
  # zip 을 풀고 지운 경우 — extract_aihub_zips.py 가 남긴 완료 표시(.extracted_ok)가 있으면 받지 않는다
  if [[ -n "$name" ]] && find "$DEST" -path "*/$NUM.*" -path "*/${name%.zip}/.extracted_ok" -type f 2>/dev/null | grep -q .; then
    echo "[skip] $fk $name — 압축 해제본 있음(.extracted_ok)"; continue
  fi
  if [[ -n "$name" && -n "$(find_zip "$name")" ]] && ! find "$DEST" -path "*/$NUM.*" -name "$name.part*" | grep -q .; then
    echo "[skip] $fk $name — 이미 있음"; continue
  fi
  free_gb=$(df -Pk "$DEST" | awk 'NR==2{printf "%d", $4/1024/1024}')
  if (( free_gb < 40 )); then echo "[중단] 디스크 여유 ${free_gb} GB < 40 GB" >&2; exit 1; fi
  echo "[get ] $fk ${name:-(이름 미상)} — $(date '+%F %T')"
  "$AIHUBSHELL" -mode d -datasetkey "$DS" -filekey "$fk" 2>&1 | grep -v -i "apikey" | grep -E "successful|완료|error|Error|실패|fail" || true
  got="$( [[ -n "$name" ]] && find_zip "$name" )"
  if [[ -n "$got" ]]; then
    echo "[ok  ] $got ($(du -h "$got" | cut -f1))"
  else
    echo "[FAIL] $fk ${name:-} — zip 이 생기지 않았다. 다시 실행하면 이어받는다" >&2; fail=1
  fi
done
exit $fail
