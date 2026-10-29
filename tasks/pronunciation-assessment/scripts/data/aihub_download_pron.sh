#!/usr/bin/env bash
# AIHub 040~043 「교육용 ○○ 모국어 사용자의 한국어 음성 데이터」에서 발음평가(pronunciation) 파일만 받는다.
# Speech_* (LAR·자유발화, 약 609 GB)는 받지 않는다 — 실현형 음소 라벨이 없어 CTC 학습에 못 쓴다.
#
#   bash tasks/pronunciation-assessment/scripts/data/aihub_download_pron.sh labels   # 라벨 12개 (약 2.6 GB) — 먼저
#   bash tasks/pronunciation-assessment/scripts/data/aihub_download_pron.sh audio    # 오디오 12개 (약 378 GB) — tmux 에서
#   bash tasks/pronunciation-assessment/scripts/data/aihub_download_pron.sh all
#
# API 키: 환경변수 AIHUB_APIKEY, 없으면 ~/.aihub_apikey 파일(권한 600)에서 읽는다. 명령줄에 키를 쓰지 않는다.
# 각 데이터셋은 AIHub 에서 이용 신청이 승인돼 있어야 받아진다.
# 저장 위치: $AIHUB_DEST (기본 $SPEECH_AI_DATA/aihub). 이미 받은 zip 은 건너뛴다(재실행 = 이어받기).
# aihubshell 은 파일 1개를 받을 때 download.tar 와 풀린 zip 이 잠시 같이 있어 그 파일 크기의 2배가 필요하다.

set -eo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi
WHAT="${1:-labels}"
case "$WHAT" in labels|audio|all) ;; *) echo "인자는 labels | audio | all 중 하나: $WHAT" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
source "$ROOT/tools/env.sh" >/dev/null
AIHUBSHELL="${AIHUBSHELL:-$(command -v aihubshell || echo "$HOME/bin/aihubshell")}"
[[ -x "$AIHUBSHELL" ]] || { echo "aihubshell 이 없다. 설치: curl -o ~/bin/aihubshell https://api.aihub.or.kr/api/aihubshell.do && chmod +x ~/bin/aihubshell" >&2; exit 1; }

if [[ -z "${AIHUB_APIKEY:-}" && -r "$HOME/.aihub_apikey" ]]; then AIHUB_APIKEY="$(tr -d '[:space:]' < "$HOME/.aihub_apikey")"; fi
[[ -n "${AIHUB_APIKEY:-}" ]] || { echo "AIHub API 키가 없다 — AIHUB_APIKEY 를 export 하거나 ~/.aihub_apikey 에 저장할 것" >&2; exit 1; }
export AIHUB_APIKEY      # aihubshell 이 -aihubapikey 가 없으면 이 환경변수를 읽는다

DEST="${AIHUB_DEST:-$SPEECH_AI_DATA/aihub}"
[[ "$DEST" != *"미설정"* ]] || { echo "SPEECH_AI_DATA 가 설정되지 않았다 (.env)" >&2; exit 1; }
mkdir -p "$DEST"

# 데이터셋키 파일키 파일명 GB(대략) 종류   — https://api.aihub.or.kr/info/<데이터셋키>.do 로 확인(2026-09-14)
FILES="
71469 475409 TL_pronunciation_lab.zip   0.29 labels
71469 475410 TL_pronunciation_tag.zip   0.18 labels
71469 475414 VL_pronunciation_lab.zip   0.04 labels
71469 475415 VL_pronunciation_tag.zip   0.03 labels
71489 475422 TL_pronunciation_lab.zip   0.46 labels
71489 475423 TL_pronunciation_tag.zip   0.29 labels
71489 475427 VL_pronunciation_lab.zip   0.06 labels
71489 475428 VL_pronunciation_tag.zip   0.04 labels
71490 475434 TL_pronunciation_lab.zip   0.25 labels
71490 475435 TL_pronunciation_tag.zip   0.16 labels
71490 475439 VL_pronunciation_lab.zip   0.03 labels
71490 475440 VL_pronunciation_tag.zip   0.02 labels
71479 475447 TL_pronunciation_lab.zip   0.44 labels
71479 475448 TL_pronunciation_tag.zip   0.28 labels
71479 475452 VL_pronunciation_lab.zip   0.05 labels
71479 475453 VL_pronunciation_tag.zip   0.04 labels
71469 475412 VS_pronunciation_sound.zip   8.02 audio
71469 475407 TS_pronunciation_sound_2.zip 5.55 audio
71469 475406 TS_pronunciation_sound_1.zip 59.51 audio
71489 475425 VS_pronunciation_sound.zip   13.19 audio
71489 475420 TS_pronunciation_sound_2.zip 44.42 audio
71489 475419 TS_pronunciation_sound_1.zip 60.51 audio
71490 475437 VS_pronunciation_sound.zip   7.56 audio
71490 475432 TS_pronunciation_sound_2.zip 0.64 audio
71490 475431 TS_pronunciation_sound_1.zip 60.08 audio
71479 475450 VS_pronunciation_sound.zip   13.17 audio
71479 475445 TS_pronunciation_sound_2.zip 43.45 audio
71479 475444 TS_pronunciation_sound_1.zip 62.17 audio
"
declare -A DSNAME=([71469]=040 [71489]=041 [71490]=042 [71479]=043)

# 받은 zip 위치 찾기: aihubshell 은 "<번호>.<데이터셋명>/01-1.정식개방데이터/<split>/<원천|라벨링>/<파일>" 로 푼다.
# 파일명이 데이터셋마다 같으므로 데이터셋 번호 폴더 아래에서 찾는다.
find_zip() { find "$DEST" -path "*/$1.*" -name "$2" -type f 2>/dev/null | head -1; }

cd "$DEST"
fail=0
while read -r ds fk name gb kind; do
  [[ -z "$ds" ]] && continue
  [[ "$WHAT" == all || "$WHAT" == "$kind" ]] || continue
  num="${DSNAME[$ds]}"
  # zip 을 풀고 지운 경우(2026-09-15 디스크 정리) — 압축 해제본에 extract_aihub_zips.py 의 완료 표시가 있으면 받지 않는다
  if find "$DEST" -path "*/$num.*" -path "*/${name%.zip}/.extracted_ok" -type f 2>/dev/null | grep -q .; then
    echo "[skip] $num $name — 압축 해제본 있음(.extracted_ok)"; continue
  fi
  if [[ -n "$(find_zip "$num" "$name")" ]] && ! find "$DEST" -path "*/$num.*" -name "$name.part*" | grep -q .; then
    echo "[skip] $num $name — 이미 있음"; continue
  fi
  need_kb=$(awk -v g="$gb" 'BEGIN{printf "%d", g*2.2*1024*1024 + 5*1024*1024}')
  free_kb=$(df -Pk "$DEST" | awk 'NR==2{print $4}')
  if (( free_kb < need_kb )); then echo "[중단] 디스크 부족 — $num $name 에 약 $((need_kb/1024/1024)) GB 필요, 여유 $((free_kb/1024/1024)) GB" >&2; exit 1; fi
  echo "[get ] $num $name (${gb} GB, key $fk) — $(date '+%F %T')"
  "$AIHUBSHELL" -mode d -datasetkey "$ds" -filekey "$fk" || true   # aihubshell 은 실패해도 0 을 돌려줄 때가 있어 결과 파일로 판정한다
  got="$(find_zip "$num" "$name")"
  if [[ -n "$got" ]]; then
    echo "[ok  ] $got ($(du -h "$got" | cut -f1))"
  else
    echo "[FAIL] $num $name — zip 이 생기지 않았다 (승인·키·네트워크 확인). 다시 실행하면 이어받는다" >&2; fail=1
  fi
done <<< "$FILES"

exit $fail
