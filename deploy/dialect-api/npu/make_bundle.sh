#!/usr/bin/env bash
# 사투리 NPU 번들의 코드 부분을 <번들 폴더> 에 저장소와 같은 경로로 복사한다 — 2026-10-28.
# models/ 는 따로 만든다: tasks/dialect-classification/scripts/export/export_npu_bundle.py --out <번들 폴더>/models
#   bash deploy/dialect-api/npu/make_bundle.sh <번들 폴더>
# 번들 루트에 pyproject.toml 이 있어서 각 스크립트의 부트스트랩이 tasks/*/src 를 그대로 찾는다.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
DST=${1:?번들 폴더}
FILES=(
  pyproject.toml
  shared/__init__.py shared/paths.py shared/request_capture.py
  tasks/dialect-classification/src/dialect_classifier/__init__.py
  tasks/dialect-classification/src/dialect_classifier/ensemble.py
  tasks/dialect-classification/src/dialect_classifier/npu_models.py
  tasks/dialect-classification/src/dialect_classifier/npu_ensemble.py
  tasks/dialect-classification/scripts/export/compile_rbln.py
  tasks/dialect-classification/scripts/eval/check_npu_parity.py
  deploy/dialect-api/app.py deploy/dialect-api/index.html deploy/dialect-api/README.md
  deploy/dialect-api/npu/run_npu.sh deploy/dialect-api/npu/dialect-api.service deploy/dialect-api/npu/capture.conf
  deploy/dialect-api/npu/README_NPU.md
)
cd "$ROOT"
for f in "${FILES[@]}"; do
  install -D -m 644 "$f" "$DST/$f"
done
chmod 755 "$DST/deploy/dialect-api/npu/run_npu.sh"
(cd "$DST" && sha256sum "${FILES[@]}" > SHA256SUMS.code.txt)
echo "[bundle] ${#FILES[@]} files → $DST"
