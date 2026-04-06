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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig


@PreTrainedConfig.register_subclass("multi_task_dit_lam")
@dataclass
class MultiTaskDiTLAMConfig(MultiTaskDiTConfig):
    """Latent-first Multi-Task DiT configuration.

    This policy predicts a coarse latent sequence from observations and language,
    then predicts an action trajectory conditioned on that latent sequence.
    """

    latent_stride_k: int = 4
    lambda_latent: float = 1.0

    teacher_repo_root: str | None = None
    teacher_config_path: str | None = None
    teacher_checkpoint_path: str | None = None
    teacher_action_dim: int | None = None
    teacher_num_action_tokens: int | None = None
    teacher_latent_target: str = "z_t_tokens_raw"
    teacher_obs_key_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.drop_n_last_frames = max(
            int(self.drop_n_last_frames or 0),
            max(self.latent_boundary_offsets, default=0),
        )

    def _validate(self):
        super()._validate()

        if self.objective != "diffusion":
            raise ValueError(
                "multi_task_dit_lam currently supports only objective='diffusion'. "
                f"Got objective={self.objective!r}."
            )
        if self.latent_stride_k <= 0:
            raise ValueError(f"latent_stride_k must be positive, got {self.latent_stride_k}")
        if self.future_horizon <= 0:
            raise ValueError(
                "future_horizon must be positive. "
                f"Got horizon={self.horizon}, n_obs_steps={self.n_obs_steps}."
            )
        if self.teacher_action_dim is not None and self.teacher_action_dim <= 0:
            raise ValueError(
                f"teacher_action_dim must be positive when provided, got {self.teacher_action_dim}"
            )
        if self.teacher_num_action_tokens is not None and self.teacher_num_action_tokens <= 0:
            raise ValueError(
                "teacher_num_action_tokens must be positive when provided, "
                f"got {self.teacher_num_action_tokens}"
            )
        if (
            self.teacher_action_dim is not None
            and self.teacher_num_action_tokens is not None
            and self.teacher_action_dim % self.teacher_num_action_tokens != 0
        ):
            raise ValueError(
                "teacher_action_dim must be divisible by teacher_num_action_tokens, got "
                f"{self.teacher_action_dim} vs {self.teacher_num_action_tokens}."
            )
        if self.teacher_latent_target not in {"auto", "z_t_tokens_raw", "z_t_tokens_h_log0", "z_t_tokens"}:
            raise ValueError(
                "teacher_latent_target must be one of "
                "{'auto', 'z_t_tokens_raw', 'z_t_tokens_h_log0', 'z_t_tokens'}, "
                f"got {self.teacher_latent_target!r}."
            )

    @property
    def future_horizon(self) -> int:
        return self.horizon - (self.n_obs_steps - 1)

    @property
    def latent_boundaries(self) -> list[int]:
        boundaries = list(range(0, self.future_horizon, self.latent_stride_k))
        if not boundaries or boundaries[-1] != self.future_horizon:
            boundaries.append(self.future_horizon)
        return boundaries

    @property
    def latent_boundary_offsets(self) -> list[int]:
        return self.latent_boundaries[1:]

    @property
    def n_segment_steps(self) -> int:
        return len(self.latent_boundaries) - 1

    @property
    def n_latent_steps(self) -> int:
        return self.n_segment_steps

    @property
    def latent_token_dim(self) -> int:
        if self.teacher_action_dim is None or self.teacher_num_action_tokens is None:
            raise ValueError(
                "teacher_action_dim and teacher_num_action_tokens must be resolved "
                "before latent_token_dim can be used."
            )
        return self.teacher_action_dim // self.teacher_num_action_tokens

    @property
    def latent_memory_steps(self) -> int:
        if self.teacher_num_action_tokens is None:
            raise ValueError(
                "teacher_num_action_tokens must be resolved before latent_memory_steps can be used."
            )
        return self.n_segment_steps * self.teacher_num_action_tokens

    @property
    def observation_delta_indices(self) -> list[int]:
        history = list(range(1 - self.n_obs_steps, 1))
        return history + self.latent_boundary_offsets
