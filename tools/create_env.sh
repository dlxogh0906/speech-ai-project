#!/usr/bin/env bash
# 통합 환경 구축 — 국적 분류 + 발음 진단을 **한 환경**에서 돌린다.
#
# 사용법:
#   bash tools/create_env.sh
#   conda activate speech-ai        # 또는 아래 ENV_PREFIX 를 쓴 경우 그 경로
#
# 환경변수로 조정:
#   ENV_NAME=speech-ai              # conda env 이름
#   ENV_PREFIX=<경로>               # 이름 대신 그 경로에 만든다(홈 디스크가 빠듯할 때)
#   CUDA_TAG=cu118                  # torch 휠 태그
#
# 만드는 것:
#   1. conda env (python 3.8.20)      — fairseq + omegaconf 2.0.6 조합이 고정한다
#   2. torch 2.1.2+cu118 + requirements.txt 전부 (두 task 합집합)
#   3. fairseq (editable, --no-deps)  — 아래 "왜 태그가 아니라 커밋인가" 참고
#   4. KoG2P 클론                     — pip 패키지가 아니다
#
# ── 왜 fairseq 을 태그가 아니라 커밋으로 고정하는가 ─────────────────────────
# 체크포인트에 저장된 task 설정에 `multi_corpus_keys` 가 들어 있는데, 이 필드는
# **v0.12.2 태그에 없다.** 태그로 설치하면 로드가 이렇게 죽는다:
#     ConfigKeyError: Key 'multi_corpus_keys' not in 'AudioFinetuningConfig'
# fairseq 은 v0.12.2 이후 버전을 올린 적이 없어서 main 브랜치도 "0.12.2" 로 보고한다.
# 학습에 쓴 건 main 브랜치 체크아웃이었고, 그게 그동안 "수정 사본" 으로 불리던 것이다.
# 소스 수정은 필요 없다 — 순정 upstream 으로 두 task 전부 검증했다(2026-08-08).
set -uo pipefail

case "${1:-}" in
    -h|--help)
        # 맨 위 주석 블록을 그대로 쓴다 (줄 번호를 박아두면 편집할 때마다 어긋난다).
        awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
        exit 0 ;;
    "") ;;
    *)  echo "알 수 없는 인자: $1  (설정은 환경변수로 한다 — --help 참고)" >&2; exit 2 ;;
esac

ENV_NAME="${ENV_NAME:-speech-ai}"
PY_VERSION="${PY_VERSION:-3.8.20}"
CUDA_TAG="${CUDA_TAG:-cu118}"
TORCH_VERSION="${TORCH_VERSION:-2.1.2}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.1.2}"

# 검증된 조합. 바꾸려면 위 주석을 먼저 읽을 것.
FAIRSEQ_REPO="https://github.com/facebookresearch/fairseq.git"
FAIRSEQ_COMMIT="${FAIRSEQ_COMMIT:-3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99}"
KOG2P_REPO="https://github.com/scarletcho/KoG2P.git"
KOG2P_COMMIT="${KOG2P_COMMIT:-cbc69bbe1439862364d4ffd1552f59b930f655c7}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 호출자가 이미 준 값이 우선한다 (.env 규약과 동일 — shared/paths.py 주석 참고).
_pre_fairseq="${FAIRSEQ_ROOT:-}"; _pre_g2p="${G2P_TOOLS:-}"
# .env 가 있으면 FAIRSEQ_ROOT / G2P_TOOLS 를 거기서 읽는다(없어도 죽지 않는다).
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
[ -n "$_pre_fairseq" ] && FAIRSEQ_ROOT="$_pre_fairseq"
[ -n "$_pre_g2p" ]     && G2P_TOOLS="$_pre_g2p"

FAIRSEQ_DIR="${FAIRSEQ_ROOT:-$(dirname "$ROOT")/fairseq}"
G2P_DIR="${G2P_TOOLS:-$(dirname "$ROOT")/g2p_tools}"

step() { echo; echo "=== $* ==="; }

step "0/5 사전 점검"
command -v conda >/dev/null || { echo "[fatal] conda 가 PATH 에 없다." >&2; exit 1; }
command -v git   >/dev/null || { echo "[fatal] git 이 없다." >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null \
  || echo "(GPU 없음 — 국적 추론은 CPU 로 돌지만 발음 진단은 매우 느리다)"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
# conda >=24 는 채널 ToS 동의를 요구한다.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true

step "1/5 conda env (python ${PY_VERSION})"
if [ -n "${ENV_PREFIX:-}" ]; then
    conda create -p "$ENV_PREFIX" "python=${PY_VERSION}" -y || exit 1
    PY="$ENV_PREFIX/bin/python"
else
    conda create -n "$ENV_NAME" "python=${PY_VERSION}" -y || exit 1
    PY="$(conda run -n "$ENV_NAME" python -c 'import sys;print(sys.executable)')" || exit 1
fi
echo "python: $PY"

step "2/5 pip 상한"
# omegaconf 2.0.6 이 legacy 메타데이터라 pip>=24.1 은 설치를 거부한다.
"$PY" -m pip install -q "pip<24.1" "setuptools<70" wheel || exit 1

step "3/5 torch ${TORCH_VERSION}+${CUDA_TAG} · requirements"
"$PY" -m pip install "torch==${TORCH_VERSION}+${CUDA_TAG}" "torchaudio==${TORCHAUDIO_VERSION}+${CUDA_TAG}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" || exit 1
"$PY" -m pip install -r "$ROOT/requirements.txt" \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" || exit 1

step "4/5 fairseq (editable, --no-deps) → $FAIRSEQ_DIR"
# --no-deps 가 필수다. 빼면 resolver 가 omegaconf 를 최신으로 올려 hydra 설정이 통째로 깨진다.
# --no-build-isolation 도 필수다. 빌드에 이 환경의 torch·Cython 이 필요하다.
if [ ! -d "$FAIRSEQ_DIR/.git" ]; then
    git clone "$FAIRSEQ_REPO" "$FAIRSEQ_DIR" || exit 1
fi
git -C "$FAIRSEQ_DIR" fetch origin "$FAIRSEQ_COMMIT" --depth 1 2>/dev/null || git -C "$FAIRSEQ_DIR" fetch origin
git -C "$FAIRSEQ_DIR" checkout -q "$FAIRSEQ_COMMIT" || exit 1
"$PY" -m pip install -e "$FAIRSEQ_DIR" --no-deps --no-build-isolation || exit 1

step "5/5 KoG2P → $G2P_DIR/KoG2P"
if [ ! -d "$G2P_DIR/KoG2P/.git" ]; then
    mkdir -p "$G2P_DIR" && git clone "$KOG2P_REPO" "$G2P_DIR/KoG2P" || exit 1
fi
git -C "$G2P_DIR/KoG2P" checkout -q "$KOG2P_COMMIT" || exit 1

step "검증"
# cwd 를 fairseq 소스 **밖**에 두고 확인한다 (namespace package 그림자 — ENVIRONMENT.md 함정 1).
cd "$ROOT"
"$PY" - <<'PY'
import importlib, sys
ok = True
try:
    import torch
    print(f"  [ok] torch {torch.__version__} · cuda_available={torch.cuda.is_available()}")
except Exception as e:
    ok = False; print(f"  [FAIL] torch: {e}")
sys.path.insert(0, ".")
try:
    from shared.fairseq_guard import ensure_fairseq
    ensure_fairseq()
except Exception as e:
    print(f"  [warn] ensure_fairseq 실패(.env 의 FAIRSEQ_ROOT 확인): {e}")
for m in ["fairseq", "numpy", "scipy", "sklearn", "joblib", "edlib", "editdistance",
          "onnxruntime", "fastapi", "uvicorn"]:
    try:
        mod = importlib.import_module(m)
        print(f"  [ok] {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        ok = False; print(f"  [FAIL] {m}: {e}")
sys.exit(0 if ok else 1)
PY
rc=$?

echo
if [ $rc -eq 0 ]; then
    echo "완료. 다음:"
    echo "  1) cp .env.example .env  후 경로를 채운다 (PY=$PY 도 함께)"
    echo "     FAIRSEQ_ROOT=$FAIRSEQ_DIR"
    echo "     G2P_TOOLS=$G2P_DIR"
    echo "  2) python tests/test_imports.py"
    echo "  3) bash deploy/infer_nationality.sh <오디오>"
    echo "     bash deploy/infer_mdd.sh <오디오> \"<문장>\""
else
    echo "일부 검증 실패 — 위 [FAIL] 을 확인할 것." >&2
fi
exit $rc
