#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Usage:
#   bash scripts/train_pi05_pnp_nanobanana.sh
# Optional:
#   POLICY_REPO=joon-stack/my_policy BATCH_SIZE=32 bash scripts/train_pi05_pnp_nanobanana.sh

DATASET_REPO="${DATASET_REPO:-joon-stack/pick_place_nanobanana}"
DATASET_ROOT="${DATASET_ROOT:-}"
DATASET_REVISION="${DATASET_REVISION:-}"
DATASET_VIDEO_BACKEND="pyav"
export DATASET_REPO DATASET_ROOT DATASET_REVISION DATASET_VIDEO_BACKEND

POLICY_REPO="${POLICY_REPO:-joon-stack/pi05_pnp_nanobanana}"
POLICY_PUSH_TO_HUB="${POLICY_PUSH_TO_HUB:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pi05_pnp_nanobanana_top}"
JOB_NAME="${JOB_NAME:-pi05_pnp_nanobanana}"
WANDB_PROJECT="${WANDB_PROJECT:-pnp_nanobanana}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
INPUT_FEATURES=${INPUT_FEATURES:-'{"observation.state":{"type":"STATE","shape":[7]},"observation.images.top":{"type":"VISUAL","shape":[3,480,640]}}'}

USE_ACCELERATE="${USE_ACCELERATE:-${ACCELERATE_LAUNCH:-0}}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-}"
ACCELERATE_GPU_IDS="${ACCELERATE_GPU_IDS:-}"
ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-no}"
ACCELERATE_MAIN_PROCESS_PORT="${ACCELERATE_MAIN_PROCESS_PORT:-}"
ACCELERATE_MULTI_GPU="${ACCELERATE_MULTI_GPU:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dataset.repo_id|--dataset.repo_id=*|--dataset.root|--dataset.root=*|--dataset.revision|--dataset.revision=*)
      echo "Do not override dataset repo/root/revision via CLI; set DATASET_REPO, DATASET_ROOT, or DATASET_REVISION env vars so preflight and train stay identical." >&2
      exit 2
      ;;
  esac
done

echo "== Dataset stats preflight =="
echo "DATASET_REPO=${DATASET_REPO}"
if [[ -n "${DATASET_ROOT}" ]]; then
  echo "DATASET_ROOT=${DATASET_ROOT}"
else
  echo "DATASET_ROOT=<HF cache resolved by LeRobotDataset>"
fi
if [[ -n "${DATASET_REVISION}" ]]; then
  echo "DATASET_REVISION=${DATASET_REVISION}"
fi

PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}" python - <<'PY'
import os
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.recompute_dataset_parquet_stats import recompute_dataset_parquet_stats

repo = os.environ["DATASET_REPO"]
root_env = os.environ.get("DATASET_ROOT") or None
root = Path(root_env) if root_env else None
revision = os.environ.get("DATASET_REVISION") or None
video_backend = os.environ["DATASET_VIDEO_BACKEND"]

print("Materializing dataset with LeRobotDataset...")
materialized = LeRobotDataset(repo, root=root, revision=revision, video_backend=video_backend)
resolved_root = materialized.root
resolved_revision = materialized.revision
print(f"RESOLVED_DATASET_ROOT={resolved_root}")
print(f"RESOLVED_DATASET_REVISION={resolved_revision}")

print("Patching meta/stats.json from parquet...")
recompute_dataset_parquet_stats(resolved_root, backup=True)

print("Reloading dataset and verifying loader-visible stats...")
dataset = LeRobotDataset(repo, root=root, revision=revision, video_backend=video_backend)
computed = recompute_dataset_parquet_stats(dataset.root, dry_run=True, backup=False)

for feature in ["action", "observation.state"]:
    loaded = dataset.meta.stats[feature]
    for key in ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]:
        loaded_value = np.asarray(loaded[key])
        computed_value = np.asarray(computed[feature][key])
        if key == "count":
            ok = np.array_equal(loaded_value, computed_value)
        else:
            ok = np.allclose(loaded_value, computed_value, rtol=1e-6, atol=1e-8)
        if not ok:
            raise SystemExit(
                f"BAD STATS: loader-visible {feature}.{key} does not match parquet "
                f"(loaded={loaded_value}, computed={computed_value})"
            )

    print(
        f"{feature} OK count={loaded['count'].tolist()} "
        f"q01={np.asarray(loaded['q01']).tolist()} q99={np.asarray(loaded['q99']).tolist()}"
    )

print("DATASET STATS PREFLIGHT OK")
PY

dataset_root_args=()
if [[ -n "${DATASET_ROOT}" ]]; then
  dataset_root_args+=(--dataset.root="${DATASET_ROOT}")
fi

dataset_revision_args=()
if [[ -n "${DATASET_REVISION}" ]]; then
  dataset_revision_args+=(--dataset.revision="${DATASET_REVISION}")
fi

cmd=(
  lerobot-train
  --policy.type=pi05
  --policy.pretrained_path=lerobot/pi05_base
  --dataset.repo_id="${DATASET_REPO}"
  "${dataset_root_args[@]}"
  "${dataset_revision_args[@]}"
  --dataset.video_backend="${DATASET_VIDEO_BACKEND}"
  --dataset.use_imagenet_stats=false
  --policy.device=cuda
  --policy.dtype=bfloat16
  --policy.gradient_checkpointing=true
  --policy.compile_model=true
  --policy.repo_id="${POLICY_REPO}"
  --policy.push_to_hub="${POLICY_PUSH_TO_HUB}"
  --policy.private=false
  --policy.n_obs_steps=1
  --policy.chunk_size=50
  --policy.n_action_steps=50
  --policy.image_resolution='[224,224]'
  --policy.empty_cameras=0
  --policy.input_features="${INPUT_FEATURES}"
  --policy.optimizer_lr=2.5e-5
  --policy.optimizer_betas='[0.9,0.95]'
  --policy.optimizer_weight_decay=0.01
  --policy.optimizer_grad_clip_norm=1.0
  --policy.scheduler_warmup_steps=1000
  --policy.scheduler_decay_steps=30000
  --policy.scheduler_decay_lr=2.5e-6
  --output_dir="${OUTPUT_DIR}"
  --job_name="${JOB_NAME}"
  --resume=false
  --seed=1000
  --num_workers=8
  --batch_size="${BATCH_SIZE}"
  --steps="${TRAIN_STEPS}"
  --eval_freq=0
  --log_freq=20
  --save_checkpoint=true
  --save_freq="${SAVE_FREQ}"
  --use_policy_training_preset=true
  --wandb.enable=true
  --wandb.disable_artifact=true
  --wandb.project="${WANDB_PROJECT}"
)

if [[ "${USE_ACCELERATE}" == "1" ]]; then
  lerobot_train_bin="$(command -v lerobot-train)"
  cmd[0]="${lerobot_train_bin}"

  launch_cmd=("${ACCELERATE_BIN}" launch)
  if [[ "${ACCELERATE_MULTI_GPU}" == "1" ]]; then
    launch_cmd+=(--multi_gpu)
  fi
  if [[ -n "${ACCELERATE_NUM_PROCESSES}" ]]; then
    launch_cmd+=(--num_processes="${ACCELERATE_NUM_PROCESSES}")
  fi
  if [[ -n "${ACCELERATE_GPU_IDS}" ]]; then
    launch_cmd+=(--gpu_ids="${ACCELERATE_GPU_IDS}")
  fi
  if [[ -n "${ACCELERATE_MIXED_PRECISION}" ]]; then
    launch_cmd+=(--mixed_precision="${ACCELERATE_MIXED_PRECISION}")
  fi
  if [[ -n "${ACCELERATE_MAIN_PROCESS_PORT}" ]]; then
    launch_cmd+=(--main_process_port="${ACCELERATE_MAIN_PROCESS_PORT}")
  fi

  echo "== Accelerate launch =="
  echo "ACCELERATE_NUM_PROCESSES=${ACCELERATE_NUM_PROCESSES:-<accelerate default>}"
  echo "ACCELERATE_GPU_IDS=${ACCELERATE_GPU_IDS:-<accelerate default>}"
  echo "PER_DEVICE_BATCH_SIZE=${BATCH_SIZE}"
  "${launch_cmd[@]}" "${cmd[@]}" "$@"
else
  "${cmd[@]}" "$@"
fi
