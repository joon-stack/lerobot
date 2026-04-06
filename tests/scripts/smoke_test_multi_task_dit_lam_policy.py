#!/usr/bin/env python

from __future__ import annotations

import argparse

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.multi_task_dit_lam.configuration_multi_task_dit_lam import MultiTaskDiTLAMConfig
from lerobot.policies.multi_task_dit_lam.modeling_multi_task_dit_lam import MultiTaskDiTLAMPolicy
from lerobot.policies.multi_task_dit_lam.processor_multi_task_dit_lam import (
    make_multi_task_dit_lam_pre_post_processors,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for multi_task_dit_lam full policy forward.")
    parser.add_argument("--teacher-repo-root", required=True)
    parser.add_argument("--teacher-config", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--state-dim", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--latent-stride-k", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-train-timesteps", type=int, default=20)
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument("--obs-key", default=f"{OBS_IMAGES}.laptop")
    parser.add_argument("--task", default="pick up the cube")
    return parser.parse_args()


def create_train_batch(
    *,
    batch_size: int,
    n_obs_steps: int,
    horizon: int,
    state_dim: int,
    action_dim: int,
    height: int,
    width: int,
    task: str,
    obs_key: str,
) -> dict[str, torch.Tensor | list[str]]:
    return {
        OBS_STATE: torch.randn(batch_size, n_obs_steps, state_dim),
        obs_key: torch.rand(batch_size, n_obs_steps, 3, height, width),
        ACTION: torch.randn(batch_size, horizon, action_dim),
        "task": [task] * batch_size,
    }


def create_observation_batch(
    *,
    batch_size: int,
    state_dim: int,
    height: int,
    width: int,
    task: str,
    obs_key: str,
) -> dict[str, torch.Tensor | list[str]]:
    return {
        OBS_STATE: torch.randn(batch_size, state_dim),
        obs_key: torch.rand(batch_size, 3, height, width),
        "task": [task] * batch_size,
    }


def create_config(args: argparse.Namespace) -> MultiTaskDiTLAMConfig:
    config = MultiTaskDiTLAMConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(args.state_dim,)),
            args.obs_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, args.height, args.width)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(args.action_dim,))},
        n_obs_steps=args.n_obs_steps,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        objective="diffusion",
        noise_scheduler_type="DDPM",
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        latent_stride_k=args.latent_stride_k,
        teacher_repo_root=args.teacher_repo_root,
        teacher_config_path=args.teacher_config,
        teacher_checkpoint_path=args.teacher_checkpoint,
        teacher_obs_key_map={args.obs_key: "image"},
        device=args.device,
    )
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    config.validate_features()
    return config


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    config = create_config(args)
    preprocessor, postprocessor = make_multi_task_dit_lam_pre_post_processors(config=config, dataset_stats=None)

    policy = MultiTaskDiTLAMPolicy(config=config)
    policy.to(args.device)
    policy.train()

    train_batch = create_train_batch(
        batch_size=args.batch_size,
        n_obs_steps=args.n_obs_steps,
        horizon=args.horizon,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        height=args.height,
        width=args.width,
        task=args.task,
        obs_key=args.obs_key,
    )
    processed_train_batch = preprocessor(train_batch)

    loss, info = policy.forward(processed_train_batch)
    print("=== Train Forward ===")
    print(f"loss={loss.item():.6f}")
    for key in ("loss_action", "loss_latent", "loss_total", "n_segment_steps", "latent_memory_steps"):
        print(f"{key}={info[key]}")

    loss.backward()
    print("backward_ok=True")

    policy.eval()
    policy.reset()
    observation_batch = create_observation_batch(
        batch_size=args.batch_size,
        state_dim=args.state_dim,
        height=args.height,
        width=args.width,
        task=args.task,
        obs_key=args.obs_key,
    )
    processed_obs = preprocessor(observation_batch)

    with torch.no_grad():
        selected_action = policy.select_action(processed_obs)
        processed_action = postprocessor(selected_action)

    print("\n=== Inference ===")
    print(f"selected_action.shape={tuple(selected_action.shape)}")
    print(f"processed_action.shape={tuple(processed_action.shape)}")
    print(f"processed_action.dtype={processed_action.dtype}")
    print(f"processed_action.device={processed_action.device}")
    print(f"finite={torch.isfinite(processed_action).all().item()}")


if __name__ == "__main__":
    main()
