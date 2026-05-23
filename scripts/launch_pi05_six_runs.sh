#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x "${SCRIPT_DIR}/train_pi05_pick_nanobanana.sh" ]]; then
  TRAIN_SCRIPT="${SCRIPT_DIR}/train_pi05_pick_nanobanana.sh"
elif [[ -x "${SCRIPT_DIR}/train_pi05_pnp_nanobanana.sh" ]]; then
  TRAIN_SCRIPT="${SCRIPT_DIR}/train_pi05_pnp_nanobanana.sh"
else
  echo "Could not find train_pi05_pick_nanobanana.sh or train_pi05_pnp_nanobanana.sh" >&2
  exit 2
fi

TOP_FEATURES='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.top":{"type":"VISUAL","shape":[3,480,640]}}'
TOP_WRIST_FEATURES='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.top":{"type":"VISUAL","shape":[3,480,640]},"observation.images.wrist":{"type":"VISUAL","shape":[3,480,640]}}'

GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ID_ARRAY[@]}}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${BATCH_SIZE:-64}}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_DIR="${LOG_DIR:-outputs/pi05_six_runs_logs/$(date +%Y%m%d_%H%M%S)}"
ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-no}"
ACCELERATE_MAIN_PROCESS_PORT="${ACCELERATE_MAIN_PROCESS_PORT:-29500}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "${LOG_DIR}"

if (( NUM_PROCESSES < 1 )); then
  echo "NUM_PROCESSES must be >= 1" >&2
  exit 2
fi

if [[ -n "${PER_DEVICE_BATCH_SIZE:-}" ]]; then
  BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}"
  GLOBAL_BATCH_SIZE="$((BATCH_SIZE * NUM_PROCESSES))"
elif (( GLOBAL_BATCH_SIZE % NUM_PROCESSES == 0 )); then
  BATCH_SIZE="$((GLOBAL_BATCH_SIZE / NUM_PROCESSES))"
else
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} is not divisible by NUM_PROCESSES=${NUM_PROCESSES}. Set PER_DEVICE_BATCH_SIZE explicitly." >&2
  exit 2
fi

run_job() {
  local dataset_repo="$1"
  local policy_repo="$2"
  local output_dir="$3"
  local job_name="$4"
  local wandb_project="$5"
  local input_features="$6"
  local log_file="${LOG_DIR}/${job_name}.log"

  echo "[launch] job=${job_name} dataset=${dataset_repo} log=${log_file}"
  echo "[launch] gpu_ids=${GPU_IDS} num_processes=${NUM_PROCESSES} global_batch=${GLOBAL_BATCH_SIZE} per_device_batch=${BATCH_SIZE} steps=${TRAIN_STEPS}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] would run ${TRAIN_SCRIPT}"
    return 0
  fi

  local accelerate_multi_gpu=0
  if (( NUM_PROCESSES > 1 )); then
    accelerate_multi_gpu=1
  fi

  env \
    DATASET_REPO="${dataset_repo}" \
    POLICY_REPO="${policy_repo}" \
    OUTPUT_DIR="${output_dir}" \
    JOB_NAME="${job_name}" \
    WANDB_PROJECT="${wandb_project}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    TRAIN_STEPS="${TRAIN_STEPS}" \
    SAVE_FREQ="${SAVE_FREQ}" \
    INPUT_FEATURES="${input_features}" \
    USE_ACCELERATE=1 \
    ACCELERATE_MULTI_GPU="${accelerate_multi_gpu}" \
    ACCELERATE_NUM_PROCESSES="${NUM_PROCESSES}" \
    ACCELERATE_GPU_IDS="${GPU_IDS}" \
    ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION}" \
    ACCELERATE_MAIN_PROCESS_PORT="${ACCELERATE_MAIN_PROCESS_PORT}" \
    bash "${TRAIN_SCRIPT}" >"${log_file}" 2>&1
}

echo "[launch] TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[launch] GPU_IDS=${GPU_IDS} NUM_PROCESSES=${NUM_PROCESSES}"
echo "[launch] GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} PER_DEVICE_BATCH_SIZE=${BATCH_SIZE}"
echo "[launch] TRAIN_STEPS=${TRAIN_STEPS} SAVE_FREQ=${SAVE_FREQ}"
echo "[launch] LOG_DIR=${LOG_DIR}"
echo "[launch] mode=sequential distributed; one job uses all listed GPUs"

run_job "joon-stack/pick_place_nanobanana" "joon-stack/pi05_pnp_nanobanana_top" "outputs/pi05_pnp_nanobanana_top" "pi05_pnp_nanobanana_top" "pnp_nanobanana_top" "${TOP_FEATURES}"
run_job "joon-stack/stack_cup_merged" "joon-stack/pi05_stack_cup_top" "outputs/pi05_stack_cup_top" "pi05_stack_cup_top" "stack_cup_top" "${TOP_FEATURES}"
run_job "HWAN7919/put_banana_in_pot_merge" "joon-stack/pi05_put_banana_in_pot_top" "outputs/pi05_put_banana_in_pot_top" "pi05_put_banana_in_pot_top" "put_banana_in_pot_top" "${TOP_FEATURES}"
run_job "joon-stack/pick_place_nanobanana" "joon-stack/pi05_pnp_nanobanana_top_wrist" "outputs/pi05_pnp_nanobanana_top_wrist" "pi05_pnp_nanobanana_top_wrist" "pnp_nanobanana_top_wrist" "${TOP_WRIST_FEATURES}"
run_job "joon-stack/stack_cup_merged" "joon-stack/pi05_stack_cup_top_wrist" "outputs/pi05_stack_cup_top_wrist" "pi05_stack_cup_top_wrist" "stack_cup_top_wrist" "${TOP_WRIST_FEATURES}"
run_job "HWAN7919/put_banana_in_pot_merge" "joon-stack/pi05_put_banana_in_pot_top_wrist" "outputs/pi05_put_banana_in_pot_top_wrist" "pi05_put_banana_in_pot_top_wrist" "put_banana_in_pot_top_wrist" "${TOP_WRIST_FEATURES}"

echo "[launch] all jobs finished"
