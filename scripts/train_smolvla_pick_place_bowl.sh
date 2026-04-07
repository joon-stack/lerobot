#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_smolvla_pick_place_bowl.sh
# Optional:
#   bash scripts/train_smolvla_pick_place_bowl.sh --steps=50000 --batch_size=32

DATASET_REPO="nikriz/pick_place_bowl"
POLICY_REPO="nikriz/smolvla_so101_drawer"
OUTPUT_DIR="outputs/smolvla_pick_place_bowl"
JOB_NAME="smolvla_pick_place_bowl"
WANDB_PROJECT="pick_place_bowl"
WANDB_RUN_ID="3zuj1pxc"

cmd=(
  lerobot-train
  --policy.path=lerobot/smolvla_base
  --dataset.repo_id="${DATASET_REPO}"
  --dataset.video_backend=torchcodec
  --dataset.use_imagenet_stats=true
  --policy.input_features=null
  --policy.output_features=null
  --policy.device=cuda
  --policy.push_to_hub=true
  --policy.repo_id="${POLICY_REPO}"
  --policy.n_obs_steps=1
  --policy.chunk_size=50
  --policy.n_action_steps=50
  --policy.max_state_dim=32
  --policy.max_action_dim=32
  --policy.resize_imgs_with_padding='[512,512]'
  --policy.empty_cameras=0
  --policy.adapt_to_pi_aloha=false
  --policy.use_delta_joint_actions_aloha=false
  --policy.tokenizer_max_length=48
  --policy.num_steps=10
  --policy.use_cache=true
  --policy.freeze_vision_encoder=true
  --policy.train_expert_only=true
  --policy.train_state_proj=true
  --policy.optimizer_lr=1e-4
  --policy.optimizer_betas='[0.9,0.95]'
  --policy.optimizer_eps=1e-8
  --policy.optimizer_weight_decay=1e-10
  --policy.optimizer_grad_clip_norm=10
  --policy.scheduler_warmup_steps=1000
  --policy.scheduler_decay_steps=30000
  --policy.scheduler_decay_lr=2.5e-6
  --policy.add_image_special_tokens=false
  --policy.attention_mode=cross_attn
  --policy.prefix_length=-1
  --policy.pad_language_to=longest
  --policy.num_expert_layers=-1
  --policy.num_vlm_layers=16
  --policy.self_attn_every_n_layers=2
  --policy.expert_width_multiplier=0.75
  --policy.min_period=0.004
  --policy.max_period=4.0
  --output_dir="${OUTPUT_DIR}"
  --job_name="${JOB_NAME}"
  --resume=false
  --seed=1000
  --num_workers=4
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
