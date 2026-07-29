#!/usr/bin/env bash
# Create the `fairseq_xlsr` conda env for the XLS-R country-classification baseline.
#
# Usage:
#   bash scripts/create_env.sh
#   conda activate fairseq_xlsr
#
# Notes:
#  * fairseq requires torch>=1.13 and is NOT compatible with Python 3.13, hence py3.8.
#  * torch 2.1.2 + cu118 has cp38 wheels and runs on this A100 server.
#  * The fairseq editable install builds C++/CUDA extensions and can fail on some
#    toolchains; that step is non-fatal here. If it fails, see README_BASELINE.md
#    ("fairseq install troubleshooting"). The `tiny` smoke-test backend does NOT need
#    fairseq -- only the `fairseq_xlsr` backend (which loads xlsr2_2B_1000k.pt) does.
set -uo pipefail

ENV_NAME="${ENV_NAME:-fairseq_xlsr}"
PY_VERSION="${PY_VERSION:-3.8}"
CUDA_TAG="${CUDA_TAG:-cu118}"          # override: CUDA_TAG=cu121 bash scripts/create_env.sh
TORCH_VERSION="${TORCH_VERSION:-2.1.2}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.1.2}"
FAIRSEQ_DIR="${FAIRSEQ_DIR:-$FAIRSEQ_ROOT}"

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null \
  || echo "(no nvidia-smi; CUDA wheels will still install but GPU may be unavailable)"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# conda >=24 requires accepting channel Terms of Service before env creation.
echo "=== accept channel ToS (idempotent) ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "=== create env ${ENV_NAME} (python=${PY_VERSION}) ==="
if ! conda create -n "${ENV_NAME}" "python=${PY_VERSION}" -y; then
  echo "[fatal] conda create failed (try: conda create -n ${ENV_NAME} python=${PY_VERSION} -c conda-forge --override-channels -y)"
  exit 1
fi

echo "=== activate ${ENV_NAME} ==="
conda activate "${ENV_NAME}"
# Guard: never fall back to base and pollute it with pip installs.
if [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
  echo "[fatal] failed to activate ${ENV_NAME}; aborting before any pip install."
  exit 1
fi

python -m pip install --upgrade pip setuptools wheel

echo "=== install PyTorch ${TORCH_VERSION} (${CUDA_TAG}) ==="
python -m pip install \
  "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "=== install python deps ==="
# numpy<2 keeps older torch / fairseq happy.
python -m pip install \
  "numpy>=1.21.3,<2" pandas scipy scikit-learn tqdm pyyaml soundfile cython

echo "=== editable install of fairseq (${FAIRSEQ_DIR}) -- non-fatal ==="
if python -m pip install -e "${FAIRSEQ_DIR}"; then
  echo "fairseq editable install OK"
else
  echo "WARNING: fairseq editable install FAILED."
  echo "  Fallback A (no build isolation): pip install -e ${FAIRSEQ_DIR} --no-build-isolation"
  echo "  Fallback B (skip install, use path): export PYTHONPATH=${FAIRSEQ_DIR}:\$PYTHONPATH"
  echo "  The 'tiny' backend works without fairseq; only XLS-R loading needs it."
fi

echo
echo "=== verification ==="
python - <<'PY'
import importlib
def check(mod, attr="__version__"):
    try:
        m = importlib.import_module(mod)
        print(f"  [ok] {mod} {getattr(m, attr, '')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {mod}: {exc}")
import torch
print(f"  [ok] torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
for m in ["numpy", "scipy", "sklearn", "pandas", "soundfile", "fairseq"]:
    check(m)
PY

echo
echo "Done. Next:"
echo "  conda activate ${ENV_NAME}"
echo "  python scripts/inspect_manifests.py --data_root $NATIONALITY_DATA --output $NATIONALITY_RUNS/manifest_inspection.json"
