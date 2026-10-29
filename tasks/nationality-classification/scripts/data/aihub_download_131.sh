#!/usr/bin/env bash
# AIHub 131 「인공지능 학습을 위한 외국인 한국어 발화 음성 데이터」(datasetkey 505) 받기 + 압축 해제.
# 국적 라벨(residence_info.country · skill_info.motherTongue)이 라벨링데이터 JSON 에 있다.
#
#   bash tasks/nationality-classification/scripts/data/aihub_download_131.sh labels   # 라벨 12개 (약 1.8 GB) — 먼저
#   bash tasks/nationality-classification/scripts/data/aihub_download_131.sh audio    # 없는 음성 7개 (약 101 GB) — tmux 에서
#   bash tasks/nationality-classification/scripts/data/aihub_download_131.sh all
#
# 이미 풀린 폴더(.extracted_ok)나 받은 zip 은 건너뛴다(재실행 = 이어받기).
# 음성 zip 은 풀기가 끝나면(.extracted_ok 확인) 지운다 — 디스크 절약. 라벨 zip 은 남긴다.
# API 키: AIHUB_APIKEY 또는 ~/.aihub_apikey(권한 600). 명령줄에 키를 쓰지 않는다.
# 저장 위치: $AIHUB_DEST (기본 $SPEECH_AI_DATA/aihub).

set -eo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi
WHAT="${1:-labels}"
case "$WHAT" in labels|audio|all) ;; *) echo "인자는 labels | audio | all 중 하나: $WHAT" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
source "$ROOT/tools/env.sh" >/dev/null
AIHUBSHELL="${AIHUBSHELL:-$(command -v aihubshell || echo "$HOME/bin/aihubshell")}"
[[ -x "$AIHUBSHELL" ]] || { echo "aihubshell 이 없다" >&2; exit 1; }
if [[ -z "${AIHUB_APIKEY:-}" && -r "$HOME/.aihub_apikey" ]]; then AIHUB_APIKEY="$(tr -d '[:space:]' < "$HOME/.aihub_apikey")"; fi
[[ -n "${AIHUB_APIKEY:-}" ]] || { echo "AIHub API 키가 없다" >&2; exit 1; }
export AIHUB_APIKEY

DEST="${AIHUB_DEST:-$SPEECH_AI_DATA/aihub}"
[[ "$DEST" != *"미설정"* ]] || { echo "SPEECH_AI_DATA 가 설정되지 않았다 (.env)" >&2; exit 1; }
EXTRACT="$ROOT/tasks/pronunciation-assessment/scripts/data/extract_aihub_zips.py"
DS=505

# 파일키  zip 이름(aihubshell 목록 그대로)  GB  종류    — aihubshell -mode l -datasetkey 505 (2026-09-17)
FILES="
57660|1. 베트남어.zip|0.26|labels
57661|2. 영어.zip|0.04|labels
57662|3. 일본어.zip|0.29|labels
57663|4. 중국어.zip|0.39|labels
57664|5. 태국어.zip|0.18|labels
57665|6. 기타.zip|0.38|labels
57672|1. 베트남어.zip|0.03|labels
57673|2. 영어.zip|0.01|labels
57674|3. 일본어.zip|0.04|labels
57675|4. 중국어.zip|0.05|labels
57676|5. 태국어.zip|0.02|labels
57677|6. 기타.zip|0.05|labels
57667|TS_2. 영어.zip|9|audio
57685|VS_2. 영어.zip|1|audio
57684|VS_1. 베트남어.zip|7|audio
57686|VS_3. 일본어.zip|7|audio
57687|VS_4. 중국어.zip|9|audio
66283|VS_5. 태국어_add.zip|5|audio
57683|VS_6. 기타.zip|9|audio
57666|TS_1. 베트남어.zip|54|audio
"
# 라벨 zip 은 Training·Validation 이 이름이 같다 → 파일키로 split 을 정한다
split_of() { case "$1" in 5766[0-7]) echo "1.Training" ;; *) echo "2.Validation" ;; esac; }

mkdir -p "$DEST"; cd "$DEST"
fail=0
while IFS='|' read -r fk name gb kind; do
  [[ -z "$fk" ]] && continue
  [[ "$WHAT" == all || "$WHAT" == "$kind" ]] || continue
  sp="$(split_of "$fk")"
  base="${name%.zip}"; under="${base// /_}"
  # 풀린 폴더가 있으면 건너뛴다 (aihubshell·추출기가 공백을 _ 로 바꾼 이름도 확인)
  if find "$DEST" -path "*/131.*/*$sp/*" \( -path "*/$base/.extracted_ok" -o -path "*/$under/.extracted_ok" \) -type f 2>/dev/null | grep -q .; then
    echo "[skip] $sp $name — 압축 해제본 있음"; continue
  fi
  got="$(find "$DEST" -path "*/131.*/*$sp/*" \( -name "$name" -o -name "$under.zip" \) -type f 2>/dev/null | head -1)"
  if [[ -z "$got" ]]; then
    need_kb=$(awk -v g="$gb" 'BEGIN{printf "%d", g*2.2*1024*1024 + 5*1024*1024}')
    free_kb=$(df -Pk "$DEST" | awk 'NR==2{print $4}')
    (( free_kb >= need_kb )) || { echo "[중단] 디스크 부족 — $name 에 약 $((need_kb/1024/1024)) GB 필요" >&2; exit 1; }
    echo "[get ] $sp $name (${gb} GB, key $fk) — $(date '+%F %T')"
    "$AIHUBSHELL" -mode d -datasetkey "$DS" -filekey "$fk" || true
    got="$(find "$DEST" -path "*/131.*/*$sp/*" \( -name "$name" -o -name "$under.zip" \) -type f 2>/dev/null | head -1)"
  fi
  if [[ -z "$got" ]]; then echo "[FAIL] $sp $name — zip 이 생기지 않았다 (승인·키·네트워크 확인)" >&2; fail=1; continue; fi
  echo "[unzip] $got — $(date '+%T')"
  if "$PY" "$EXTRACT" --root "$(dirname "$got")" --glob "$(basename "$got")" --workers 1; then
    tgt="${got%.zip}"
    if [[ "$kind" == audio && -f "$tgt/.extracted_ok" ]]; then rm -f "$got"; echo "[rm  ] $got (풀기 완료)"; fi
    echo "[ok  ] $tgt ($(find "$tgt" -type f | wc -l) 파일)"
  else
    echo "[FAIL] 압축 해제 실패: $got" >&2; fail=1
  fi
done <<< "$FILES"
exit $fail
