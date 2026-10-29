#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
source tools/env.sh >/dev/null
export TMPDIR=/data/speech-ai/tmp/dialect_speakerid_20261011
mkdir -p "$TMPDIR"
"$PY" -u tasks/dialect-classification/scripts/data/prepare_speakerid_dialect.py "$@"
