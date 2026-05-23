#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_SCRIPT="${SCRIPT_DIR}/train_smolvla_pick_nanobanana.sh"
if [[ ! -x "${TRAIN_SCRIPT}" ]]; then
  echo "Could not find executable train script: ${TRAIN_SCRIPT}" >&2
  exit 2
fi

TOP_FEATURES='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.top":{"type":"VISUAL","shape":[3,480,640]}}'
TOP_WRIST_FEATURES='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.top":{"type":"VISUAL","shape":[3,480,640]},"observation.images.wrist":{"type":"VISUAL","shape":[3,480,640]}}'

BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LOG_DIR="${LOG_DIR:-outputs/smolvla_six_runs_logs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

run_job() {
  local gpu="$1"
  local dataset_repo="$2"
  local policy_repo="$3"
  local output_dir="$4"
  local job_name="$5"
  local wandb_project="$6"
  local input_features="$7"
  local log_file="${LOG_DIR}/${job_name}.log"

  echo "[launch] gpu=${gpu} job=${job_name} dataset=${dataset_repo} log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    DATASET_REPO="${dataset_repo}" \
    POLICY_REPO="${policy_repo}" \
    OUTPUT_DIR="${output_dir}" \
    JOB_NAME="${job_name}" \
    WANDB_PROJECT="${wandb_project}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    TRAIN_STEPS="${TRAIN_STEPS}" \
    SAVE_FREQ="${SAVE_FREQ}" \
    INPUT_FEATURES="${input_features}" \
    bash "${TRAIN_SCRIPT}" >"${log_file}" 2>&1
}

run_pair() {
  local left_pid
  local right_pid
  local status=0

  run_job "$1" "$2" "$3" "$4" "$5" "$6" "$7" &
  left_pid=$!
  run_job "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" &
  right_pid=$!

  if ! wait "${left_pid}"; then
    status=1
  fi
  if ! wait "${right_pid}"; then
    status=1
  fi

  return "${status}"
}

echo "[launch] TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[launch] BATCH_SIZE=${BATCH_SIZE} TRAIN_STEPS=${TRAIN_STEPS} SAVE_FREQ=${SAVE_FREQ}"
echo "[launch] LOG_DIR=${LOG_DIR}"

run_pair \
  0 "joon-stack/pick_place_nanobanana" "joon-stack/smolvla_pnp_nanobanana_top" "outputs/smolvla_pnp_nanobanana_top" "smolvla_pnp_nanobanana_top" "smolvla_pnp_nanobanana_top" "${TOP_FEATURES}" \
  1 "joon-stack/pick_place_nanobanana" "joon-stack/smolvla_pnp_nanobanana_top_wrist" "outputs/smolvla_pnp_nanobanana_top_wrist" "smolvla_pnp_nanobanana_top_wrist" "smolvla_pnp_nanobanana_top_wrist" "${TOP_WRIST_FEATURES}"

run_pair \
  0 "joon-stack/stack_cup_merged" "joon-stack/smolvla_stack_cup_top" "outputs/smolvla_stack_cup_top" "smolvla_stack_cup_top" "smolvla_stack_cup_top" "${TOP_FEATURES}" \
  1 "joon-stack/stack_cup_merged" "joon-stack/smolvla_stack_cup_top_wrist" "outputs/smolvla_stack_cup_top_wrist" "smolvla_stack_cup_top_wrist" "smolvla_stack_cup_top_wrist" "${TOP_WRIST_FEATURES}"

run_pair \
  0 "HWAN7919/put_banana_in_pot_merge" "joon-stack/smolvla_put_banana_in_pot_top" "outputs/smolvla_put_banana_in_pot_top" "smolvla_put_banana_in_pot_top" "smolvla_put_banana_in_pot_top" "${TOP_FEATURES}" \
  1 "HWAN7919/put_banana_in_pot_merge" "joon-stack/smolvla_put_banana_in_pot_top_wrist" "outputs/smolvla_put_banana_in_pot_top_wrist" "smolvla_put_banana_in_pot_top_wrist" "smolvla_put_banana_in_pot_top_wrist" "${TOP_WRIST_FEATURES}"

echo "[launch] all jobs finished"
