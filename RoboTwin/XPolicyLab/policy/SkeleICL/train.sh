#!/usr/bin/env bash
set -euo pipefail

# Usage: train.sh <bench> <ckpt> <env_cfg_type> <action_type> <seed> <gpu_ids> [model_type]
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_ids=$6
model_type=${7:-simpleicl}
if [[ "${action_type}" != "ee" ]]; then
    echo "SkeleICL Robotwin supports action_type=ee only" >&2
    exit 2
fi
if [[ "${model_type}" != "simpleicl" && "${model_type}" != "fastwam" ]]; then
    echo "model_type must be simpleicl or fastwam" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${SKELEICL_MODEL_ROOT:-/mnt/workspace/tusifan/wm/guood/simple-icl/icl_model}"
if [[ "${gpu_ids}" == *,* ]]; then
    IFS=',' read -r -a ids <<< "${gpu_ids}"
    nproc=${#ids[@]}
else
    nproc=1
fi
train_seed=${seed}
if (( train_seed <= 0 )); then train_seed=1; fi
task="${model_type}_humangen"
output="${SCRIPT_DIR}/checkpoints/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"

cd "${MODEL_ROOT}"
CUDA_VISIBLE_DEVICES="${gpu_ids}" bash scripts/train_zero1.sh "${nproc}" \
    "task=${task}" "seed=${train_seed}" "output_dir=${output}"
