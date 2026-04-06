#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.policies.multi_task_dit_lam.configuration_multi_task_dit_lam import MultiTaskDiTLAMConfig
from lerobot.policies.multi_task_dit_lam.modeling_multi_task_dit_lam import (
    LatentActionTeacherAdapter,
    _resolve_teacher_latent_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test the multi_task_dit_lam DinoLAM teacher adapter with dummy endpoint images."
    )
    parser.add_argument("--teacher-repo-root", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--latent-stride-k", type=int, default=20)
    parser.add_argument("--obs-key", type=str, default="observation.images.image")
    parser.add_argument("--teacher-image-key", type=str, default="image")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> MultiTaskDiTLAMConfig:
    cfg = MultiTaskDiTLAMConfig(
        device=args.device,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        latent_stride_k=args.latent_stride_k,
        teacher_repo_root=str(args.teacher_repo_root.expanduser().resolve()),
        teacher_config_path=str(args.teacher_config.expanduser().resolve()),
        teacher_checkpoint_path=str(args.teacher_checkpoint.expanduser().resolve()),
        teacher_obs_key_map={args.obs_key: args.teacher_image_key},
    )
    _resolve_teacher_latent_spec(cfg)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    adapter = LatentActionTeacherAdapter(cfg)

    current = torch.rand(args.batch_size, 3, args.height, args.width, device=args.device)
    boundary = torch.rand(
        args.batch_size,
        cfg.n_segment_steps,
        3,
        args.height,
        args.width,
        device=args.device,
    )
    valid = torch.ones(args.batch_size, cfg.n_segment_steps, dtype=torch.bool, device=args.device)

    batch = {
        f"teacher.current.{args.obs_key}": current,
        f"teacher.boundary.{args.obs_key}": boundary,
        "teacher.boundary_valid": valid,
    }

    print("=== Config ===")
    print(f"device={cfg.device}")
    print(f"future_horizon={cfg.future_horizon}")
    print(f"n_segment_steps={cfg.n_segment_steps}")
    print(f"teacher_action_dim={cfg.teacher_action_dim}")
    print(f"teacher_num_action_tokens={cfg.teacher_num_action_tokens}")
    print(f"latent_token_dim={cfg.latent_token_dim}")
    print(f"latent_memory_steps={cfg.latent_memory_steps}")
    print(f"teacher_latent_target={cfg.teacher_latent_target}")

    teacher_latents, valid_mask = adapter(batch)

    print("\n=== Output ===")
    print(f"teacher_latents.shape={tuple(teacher_latents.shape)}")
    print(f"valid_mask.shape={tuple(valid_mask.shape)}")
    print(f"teacher_latents.dtype={teacher_latents.dtype}")
    print(f"teacher_latents.device={teacher_latents.device}")
    print(f"finite={bool(torch.isfinite(teacher_latents).all().item())}")


if __name__ == "__main__":
    main()
