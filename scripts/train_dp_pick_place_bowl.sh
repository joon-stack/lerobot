#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_dp_pick_place_bowl.sh
# Optional:
#   bash scripts/train_dp_pick_place_bowl.sh --steps=50000 --batch_size=32

DATASET_REPO="nikriz/pick_place_bowl"
POLICY_REPO="nikriz/dp_so101_drawer_nvs3_ori"
OUTPUT_DIR="outputs/dp_pick_place_bowl"
JOB_NAME="dp_pick_place_bowl"
WANDB_PROJECT="pick_place_bowl"
WANDB_RUN_ID="azisy28x"

cmd=(
  lerobot-train
  --policy.type=diffusion
  --dataset.repo_id="${DATASET_REPO}"
  --dataset.video_backend=torchcodec
  --dataset.use_imagenet_stats=true
  --policy.device=cuda
  --policy.push_to_hub=true
  --policy.repo_id="${POLICY_REPO}"
  --policy.n_obs_steps=2
  --policy.horizon=16
  --policy.n_action_steps=8
  --policy.drop_n_last_frames=7
  --policy.vision_backbone=resnet18
  --policy.crop_shape='[224,224]'
  --policy.crop_is_random=true
  --policy.use_group_norm=true
  --policy.spatial_softmax_num_keypoints=32
  --policy.use_separate_rgb_encoder_per_camera=false
  --policy.noise_scheduler_type=DDIM
  --policy.num_train_timesteps=100
  --policy.beta_schedule=squaredcos_cap_v2
  --policy.beta_start=1e-4
  --policy.beta_end=2e-2
  --policy.prediction_type=epsilon
  --policy.clip_sample=true
  --policy.clip_sample_range=1.0
  --policy.do_mask_loss_for_padding=false
  --policy.optimizer_lr=1e-4
  --policy.optimizer_betas='[0.95,0.999]'
  --policy.optimizer_eps=1e-8
  --policy.optimizer_weight_decay=1e-6
  --policy.scheduler_name=cosine
  --policy.scheduler_warmup_steps=500
  --output_dir="${OUTPUT_DIR}"
  --job_name="${JOB_NAME}"
  --resume=false
  --seed=1000
  --num_workers=8
  --batch_size=64
  --steps=200000
  --eval_freq=20000
  --log_freq=200
  --save_checkpoint=true
  --save_freq=20000
  --use_policy_training_preset=true
  --wandb.enable=true
  --wandb.disable_artifact=true
  --wandb.project="${WANDB_PROJECT}"
  --wandb.run_id="${WANDB_RUN_ID}"
)

"${cmd[@]}" "$@"
