# 셸에서 경로 변수를 쓰고 싶을 때:  source tools/env.sh
# 저장소 루트의 .env 를 읽어 export 한다. .env 가 없으면 .env.example 을 복사해 채울 것.
#
# ⚠️ 파이썬 스크립트는 이걸 source 하지 않아도 된다 — shared/paths.py 가 같은 .env 를 읽는다.

_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SPEECH_AI_ROOT="$_root"

if [ -f "$_root/.env" ]; then
    set -a; . "$_root/.env"; set +a
    echo "[env] .env 적용됨 (SPEECH_AI_ROOT=$_root)"
else
    echo "[env] ⚠️ $_root/.env 가 없다.  cp .env.example .env  후 값을 채울 것" >&2
fi

export PY="${PY:-python}"
