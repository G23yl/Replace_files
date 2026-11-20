#!/usr/bin/env bash
set -euo pipefail

# HumanGen data is consumed directly by icl_model; no XPolicyLab conversion runs here.
bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
action_type=${4:?action_type required}
model_type=${5:-simpleicl}
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
printf 'HumanGen source config: %s/configs/task/%s_humangen.yaml\n' "${MODEL_ROOT}" "${model_type}"
printf 'Training output: %s/checkpoints/%s-%s-%s-%s-<seed>\n' "${SCRIPT_DIR}" "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}"
printf 'No LeRobot conversion is performed; prepare HumanGen robotwin data and paired manifests as configured in icl_model/configs/data/humangen.yaml.\n'
