#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-skeleicl}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
MODEL_ROOT="${SKELEICL_MODEL_ROOT:-/mnt/workspace/tusifan/wm/guood/simple-icl/icl_model}"

if [[ ! -d "${MODEL_ROOT}" ]]; then
    echo "Missing upstream model directory: ${MODEL_ROOT}" >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"
conda install -c conda-forge "ffmpeg>=6,<7" -y
python -m pip install --upgrade pip
python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e "${MODEL_ROOT}"
python -m pip install -e "${XPL_ROOT}"
