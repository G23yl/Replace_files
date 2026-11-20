#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NNODES="${NNODES:-2}"
export PREFLIGHT="${PREFLIGHT:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export OUTPUT_DIR="/mnt/workspace/tusifan/wm/guood/simple-icl/icl_model/runs/robotwin_skeleicl_lora"

bash "${SCRIPT_DIR}/train_zero1.sh" "${NPROC_PER_NODE}" task=skeleicl_humangen use_lora=true "$@"
