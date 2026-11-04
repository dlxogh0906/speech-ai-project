#!/usr/bin/env bash
# 운영 기록 → DB → 국적 추정 → 카탈로그(엑셀) + 히트맵(HTML) 한 번에. A100 에서 돌린다(NPU 는 ssh 로).
#   bash tools/capture_db/update_all.sh            # 결과: docs/db문서자료/운영기록/
# 녹음·개인 식별 값은 NPU 밖으로 나가지 않는다 — A100 으로 오는 건 집계용 json(rid·해시 없음)뿐이다.
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/env.sh >/dev/null
NPU=${NPU_HOST:-npu}; R=/root/capture_db; DB=/data/speech-logs/db/captures.sqlite
OUT=${CATALOG_OUT:-docs/db문서자료/운영기록}; EX="docs/db문서자료/국적별_한국어_발음오류_유형카탈로그.xlsx"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
scp -q tools/capture_db/{ingest_captures.py,pron_nationality.py,build_catalog.py} "$NPU:$R/"
ssh -o BatchMode=yes "$NPU" "cd $R && python3 ingest_captures.py --root /data/speech-logs/captures --db $DB \
  && cd /root/nat_serve/bundle && FFMPEG_BIN=/root/nat_serve/bin/ffmpeg nice -n 10 ../venv/bin/python $R/pron_nationality.py --db $DB 2>&1 | grep '^\[국적\]' \
  && cd $R && python3 build_catalog.py aggregate --db $DB --out /data/speech-logs/db/catalog_data.json && chmod 600 /data/speech-logs/db/catalog_data.json"
scp -q "$NPU:/data/speech-logs/db/catalog_data.json" "$TMP/catalog_data.json"
$PY tools/capture_db/build_catalog.py render --data "$TMP/catalog_data.json" --outdir "$OUT" --example-xlsx "$EX"
$PY tools/capture_db/build_catalog.py render --data "$TMP/catalog_data.json" --outdir "$OUT" --example-xlsx "$EX" --only-source "앱 서버(python-httpx)"
