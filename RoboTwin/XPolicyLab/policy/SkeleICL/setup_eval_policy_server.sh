#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_conda_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
MODEL_ROOT="${SKELEICL_MODEL_ROOT:-/mnt/workspace/tusifan/wm/guood/simple-icl/icl_model}"
UPSTREAM_MODEL_TASK_CONFIG="${SKELEICL_UPSTREAM_MODEL_TASK_CONFIG:-}"
CHECKPOINT_ROOT="${SKELEICL_CKPT_ROOT:-${SCRIPT_DIR}/checkpoints}"
run_dir_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
run_dir="${CHECKPOINT_ROOT}/${run_dir_name}"
checkpoint_path="${SKELEICL_CHECKPOINT_PATH:-}"
if [[ -n "${checkpoint_path}" ]]; then
    if [[ ! -f "${checkpoint_path}" ]]; then
        echo "[SERVER] SKELEICL_CHECKPOINT_PATH is not a file: ${checkpoint_path}" >&2
        exit 1
    fi
else
    weights_dir="${run_dir}/checkpoints/weights"
    if [[ -d "${weights_dir}" ]]; then
        checkpoint_path=$(find "${weights_dir}" -maxdepth 1 -type f -name 'step_*.pt' | sort -V | tail -n 1)
    fi
    if [[ -z "${checkpoint_path}" ]]; then
        checkpoint_path="${weights_dir}/step_latest.pt"
    fi
fi
dataset_stats_path="${SKELEICL_DATASET_STATS_PATH:-${run_dir}/dataset_stats.json}"
server_overrides=(
    host="${policy_server_host}"
    port="${policy_server_port}"
    bench_name="${bench_name}"
    task_name="${task_name}"
    ckpt_name="${ckpt_name}"
    env_cfg_type="${env_cfg_type}"
    seed="${seed}"
    policy_name="SkeleICL"
    action_type="${action_type}"
    gpu_id="${policy_gpu_id}"
    model_root="${MODEL_ROOT}"
    checkpoint_path="${checkpoint_path}"
    dataset_stats_path="${dataset_stats_path}"
)
if [[ -n "${UPSTREAM_MODEL_TASK_CONFIG}" ]]; then
    server_overrides+=("upstream_model_task_config=${UPSTREAM_MODEL_TASK_CONFIG}")
fi

echo "[SERVER] policy=SkeleICL, task=${task_name}, ckpt=${ckpt_name}, port=${policy_server_port}"
echo "[SERVER] checkpoint_path: ${checkpoint_path}"
echo "[SERVER] dataset_stats_path: ${dataset_stats_path}"
if [[ -n "${UPSTREAM_MODEL_TASK_CONFIG}" ]]; then
    echo "[SERVER] upstream_model_task_config: ${UPSTREAM_MODEL_TASK_CONFIG}"
fi

# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate "${policy_conda_env}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${XPL_ROOT}/..:${MODEL_ROOT}/src:${MODEL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    DIFFSYNTH_MODEL_BASE_PATH="${MODEL_ROOT}/checkpoints" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${SCRIPT_DIR}/deploy.yml" \
        --overrides "${server_overrides[@]}"
