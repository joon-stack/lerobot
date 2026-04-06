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

import importlib
import sys
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor

from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (
    ObservationEncoder,
    SinusoidalPosEmb,
    TransformerBlock,
)
from lerobot.policies.multi_task_dit_lam.configuration_multi_task_dit_lam import MultiTaskDiTLAMConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_STATE,
)


def _resolve_repo_root(config: MultiTaskDiTLAMConfig) -> Path:
    if config.teacher_repo_root is None:
        raise ValueError("teacher_repo_root must be set for multi_task_dit_lam training.")
    return Path(config.teacher_repo_root).expanduser().resolve()


def _resolve_teacher_path(config: MultiTaskDiTLAMConfig, maybe_relative_path: str | None) -> Path:
    if maybe_relative_path is None:
        raise ValueError("teacher_config_path and teacher_checkpoint_path must be set for training.")
    path = Path(maybe_relative_path).expanduser()
    if path.is_absolute():
        return path
    return _resolve_repo_root(config) / path


def _load_teacher_model_cfg(config: MultiTaskDiTLAMConfig):
    from omegaconf import OmegaConf

    config_path = _resolve_teacher_path(config, config.teacher_config_path)
    teacher_cfg = OmegaConf.load(config_path)
    return teacher_cfg.get("model", teacher_cfg)


def _resolve_teacher_latent_spec(config: MultiTaskDiTLAMConfig) -> None:
    if config.teacher_action_dim is not None and config.teacher_num_action_tokens is not None:
        return

    model_cfg = _load_teacher_model_cfg(config)
    idm_cfg = model_cfg["idm"]
    config.teacher_action_dim = int(idm_cfg["action_dim"])
    config.teacher_num_action_tokens = int(idm_cfg.get("num_action_tokens", 4))


def _strip_state_dict_prefix(state_dict: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped[key[len(prefix) :]] = value
        else:
            stripped[key] = value
    return stripped


def _select_state_dict(payload: Any) -> dict[str, Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model", "module"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                payload = candidate
                break

    if not isinstance(payload, dict):
        raise ValueError("Teacher checkpoint payload is not a dictionary of tensors.")

    state_dict = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
    if not state_dict:
        raise ValueError("Teacher checkpoint did not contain any tensor weights.")
    state_dict = _strip_state_dict_prefix(state_dict, "module.")
    state_dict = _strip_state_dict_prefix(state_dict, "model.")
    return state_dict


class LatentConditionedTransformerBlock(nn.Module):
    """Action transformer block with latent-memory cross-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_features: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.self_block = TransformerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_features=num_features,
            dropout=dropout,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout)

    def forward(self, x: Tensor, features: Tensor, memory: Tensor | None = None) -> Tensor:
        x = self.self_block(x, features)
        if memory is not None and memory.numel() > 0:
            cross_input = self.cross_norm(x)
            cross_output, _ = self.cross_attn(cross_input, memory, memory)
            x = x + cross_output
        return x


class SequenceDiffusionTransformer(nn.Module):
    """Generic DiT-style diffusion transformer for arbitrary token sequences."""

    def __init__(
        self,
        config: MultiTaskDiTLAMConfig,
        conditioning_dim: int,
        token_dim: int,
        horizon: int,
    ):
        super().__init__()
        self.config = config
        self.token_dim = token_dim
        self.horizon = horizon
        self.hidden_size = config.hidden_dim
        self.timestep_embed_dim = config.timestep_embed_dim
        self.cond_dim = self.timestep_embed_dim + conditioning_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.timestep_embed_dim),
            nn.Linear(self.timestep_embed_dim, 2 * self.timestep_embed_dim),
            nn.GELU(),
            nn.Linear(2 * self.timestep_embed_dim, self.timestep_embed_dim),
            nn.GELU(),
        )
        self.input_proj = nn.Linear(token_dim, self.hidden_size)
        if config.use_positional_encoding:
            self.pos_embedding = nn.Parameter(torch.empty(1, horizon, self.hidden_size).normal_(std=0.02))
        else:
            self.pos_embedding = None

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_dim,
                    num_heads=config.num_heads,
                    num_features=self.cond_dim,
                    dropout=config.dropout,
                    use_rope=config.use_rope,
                    max_seq_len=horizon,
                    rope_base=config.rope_base,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_proj = nn.Linear(self.hidden_size, token_dim)

        for block in self.transformer_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        conditioning_vec: Tensor,
        memory: Tensor | None = None,
    ) -> Tensor:
        timestep_features = self.time_mlp(timestep)
        cond_features = torch.cat([timestep_features, conditioning_vec], dim=-1)
        hidden_seq = self.input_proj(x)
        if self.pos_embedding is not None:
            hidden_seq = hidden_seq + self.pos_embedding[:, : hidden_seq.shape[1]]
        for block in self.transformer_blocks:
            hidden_seq = block(hidden_seq, cond_features)
        return self.output_proj(hidden_seq)


class LatentConditionedDiffusionTransformer(nn.Module):
    """Action DiT that attends to a latent token sequence."""

    def __init__(
        self,
        config: MultiTaskDiTLAMConfig,
        conditioning_dim: int,
        action_dim: int,
        horizon: int,
        memory_dim: int,
    ):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.horizon = horizon
        self.hidden_size = config.hidden_dim
        self.timestep_embed_dim = config.timestep_embed_dim
        self.cond_dim = self.timestep_embed_dim + conditioning_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.timestep_embed_dim),
            nn.Linear(self.timestep_embed_dim, 2 * self.timestep_embed_dim),
            nn.GELU(),
            nn.Linear(2 * self.timestep_embed_dim, self.timestep_embed_dim),
            nn.GELU(),
        )
        self.input_proj = nn.Linear(action_dim, self.hidden_size)
        self.memory_proj = nn.Linear(memory_dim, self.hidden_size)
        if config.use_positional_encoding:
            self.pos_embedding = nn.Parameter(torch.empty(1, horizon, self.hidden_size).normal_(std=0.02))
        else:
            self.pos_embedding = None

        self.transformer_blocks = nn.ModuleList(
            [
                LatentConditionedTransformerBlock(
                    hidden_size=config.hidden_dim,
                    num_heads=config.num_heads,
                    num_features=self.cond_dim,
                    dropout=config.dropout,
                    use_rope=config.use_rope,
                    max_seq_len=horizon,
                    rope_base=config.rope_base,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_proj = nn.Linear(self.hidden_size, action_dim)

        for block in self.transformer_blocks:
            nn.init.constant_(block.self_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.self_block.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, timestep: Tensor, conditioning_vec: Tensor, memory: Tensor | None = None) -> Tensor:
        timestep_features = self.time_mlp(timestep)
        cond_features = torch.cat([timestep_features, conditioning_vec], dim=-1)
        hidden_seq = self.input_proj(x)
        if self.pos_embedding is not None:
            hidden_seq = hidden_seq + self.pos_embedding[:, : hidden_seq.shape[1]]

        projected_memory = self.memory_proj(memory) if memory is not None else None
        for block in self.transformer_blocks:
            hidden_seq = block(hidden_seq, cond_features, projected_memory)
        return self.output_proj(hidden_seq)


class SequenceDiffusionObjective(nn.Module):
    """Generic DDPM/DDIM objective over arbitrary sequences."""

    def __init__(self, config: MultiTaskDiTLAMConfig, token_dim: int, horizon: int):
        super().__init__()
        scheduler_kwargs = {
            "num_train_timesteps": config.num_train_timesteps,
            "beta_start": config.beta_start,
            "beta_end": config.beta_end,
            "beta_schedule": config.beta_schedule,
            "clip_sample": config.clip_sample,
            "clip_sample_range": config.clip_sample_range,
            "prediction_type": config.prediction_type,
        }
        if config.noise_scheduler_type == "DDPM":
            self.noise_scheduler: DDPMScheduler | DDIMScheduler = DDPMScheduler(**scheduler_kwargs)
        elif config.noise_scheduler_type == "DDIM":
            self.noise_scheduler = DDIMScheduler(**scheduler_kwargs)
        else:
            raise ValueError(f"Unsupported noise scheduler type {config.noise_scheduler_type}")

        self.token_dim = token_dim
        self.horizon = horizon
        self.prediction_type = config.prediction_type
        self.num_inference_steps = (
            config.num_inference_steps
            if config.num_inference_steps is not None
            else self.noise_scheduler.config.num_train_timesteps
        )

    def compute_loss(
        self,
        model: nn.Module,
        target_sequence: Tensor,
        conditioning_vec: Tensor,
        valid_mask: Tensor | None = None,
        memory: Tensor | None = None,
    ) -> Tensor:
        noise = torch.randn_like(target_sequence)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(target_sequence.shape[0],),
            device=target_sequence.device,
        ).long()
        noisy_sequence = self.noise_scheduler.add_noise(target_sequence, noise, timesteps)

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = target_sequence
        else:
            raise ValueError(f"Unsupported prediction type: {self.prediction_type}")

        predicted = model(noisy_sequence, timesteps, conditioning_vec=conditioning_vec, memory=memory)
        loss = F.mse_loss(predicted, target, reduction="none")
        if valid_mask is not None:
            loss = loss * valid_mask.unsqueeze(-1).to(loss.dtype)
        return loss.mean()

    @torch.no_grad()
    def conditional_sample(
        self,
        model: nn.Module,
        batch_size: int,
        conditioning_vec: Tensor,
        memory: Tensor | None = None,
    ) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        sample = torch.randn((batch_size, self.horizon, self.token_dim), dtype=dtype, device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output = model(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                conditioning_vec=conditioning_vec,
                memory=memory,
            )
            sample = self.noise_scheduler.step(model_output, t, sample).prev_sample
        return sample


class LatentActionTeacherAdapter:
    """Training-time adapter around the external DinoLAM teacher."""

    def __init__(self, config: MultiTaskDiTLAMConfig):
        self.config = config
        self.model = None
        self._model_cfg = None
        self._input_value_range = "zero_to_one"
        self._teacher_key = None
        self._source_key = None
        self._target_key = None

    def _resolve_mapping(self) -> tuple[str, str]:
        if self._source_key is not None and self._teacher_key is not None:
            return self._source_key, self._teacher_key

        if self.config.teacher_obs_key_map:
            source_key, teacher_key = next(iter(self.config.teacher_obs_key_map.items()))
        else:
            if not self.config.image_features:
                raise ValueError("multi_task_dit_lam requires at least one image feature for the DinoLAM teacher.")
            source_key = next(iter(self.config.image_features.keys()))
            teacher_key = "image"
            if self._model_cfg is not None:
                teacher_key = str(self._model_cfg.get("encoders", {}).get("image_key", teacher_key))

        self._source_key = source_key
        self._teacher_key = teacher_key
        return source_key, teacher_key

    def _resolve_target_key(self, outputs: dict[str, Tensor]) -> str:
        if self._target_key is not None:
            return self._target_key

        requested = self.config.teacher_latent_target
        if requested == "auto":
            target_key = "z_t_tokens_raw" if "z_t_tokens_raw" in outputs else "z_t_tokens_h_log0"
        else:
            target_key = requested
        if target_key not in outputs:
            raise KeyError(f"Teacher output {target_key!r} not found. Available keys: {sorted(outputs.keys())}")
        self._target_key = target_key
        return target_key

    def _load_model(self) -> None:
        if self.model is not None:
            return

        repo_root = _resolve_repo_root(self.config)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from omegaconf import OmegaConf

        config_path = _resolve_teacher_path(self.config, self.config.teacher_config_path)
        checkpoint_path = _resolve_teacher_path(self.config, self.config.teacher_checkpoint_path)

        dino_module = importlib.import_module("src.model.dino_lam")
        DinoLAM = getattr(dino_module, "DinoLAM")

        teacher_cfg = OmegaConf.load(config_path)
        model_cfg = teacher_cfg.get("model", teacher_cfg)
        self._model_cfg = model_cfg
        enc_cfg = model_cfg.get("encoders", {})
        self._input_value_range = str(enc_cfg.get("input_value_range", "zero_to_one")).strip().lower()
        model = DinoLAM(model_cfg)

        if checkpoint_path.suffix == ".safetensors":
            state_dict = load_safetensors_file(str(checkpoint_path), device="cpu")
        else:
            state_dict = _select_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        if checkpoint_path.suffix == ".safetensors":
            state_dict = _select_state_dict(state_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[multi_task_dit_lam] teacher missing keys: {sorted(missing)[:10]}")
        if unexpected:
            print(f"[multi_task_dit_lam] teacher unexpected keys: {sorted(unexpected)[:10]}")

        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.to(self.config.device)
        self.model = model

    def _prepare_teacher_tensor(self, tensor: Tensor) -> Tensor:
        tensor = tensor.to(self.config.device)
        if tensor.is_floating_point():
            if self._input_value_range == "minus_one_to_one":
                tensor = tensor * 2.0 - 1.0
            elif self._input_value_range != "zero_to_one":
                raise ValueError(
                    "Unsupported teacher encoder input_value_range "
                    f"{self._input_value_range!r}. Expected 'zero_to_one' or 'minus_one_to_one'."
                )
        return tensor

    @torch.no_grad()
    def __call__(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        self._load_model()
        source_key, teacher_key = self._resolve_mapping()

        current_key = f"teacher.current.{source_key}"
        boundary_key = f"teacher.boundary.{source_key}"
        if current_key not in batch or boundary_key not in batch:
            raise KeyError(
                f"Teacher inputs {current_key!r} and {boundary_key!r} were not found in the batch. "
                "Make sure the multi_task_dit_lam preprocessor is being used."
            )

        current = batch[current_key]
        future_boundaries = batch[boundary_key]
        batch_size = current.shape[0]
        num_steps = future_boundaries.shape[1]

        boundary_sequence = torch.cat([current.unsqueeze(1), future_boundaries], dim=1)
        obs_start = boundary_sequence[:, :-1].reshape(batch_size * num_steps, *boundary_sequence.shape[2:])
        obs_end = boundary_sequence[:, 1:].reshape(batch_size * num_steps, *boundary_sequence.shape[2:])

        outputs = self.model(
            obs={teacher_key: self._prepare_teacher_tensor(obs_start)},
            obs_future={teacher_key: self._prepare_teacher_tensor(obs_end)},
        )
        target_key = self._resolve_target_key(outputs)
        teacher_latents = outputs[target_key]
        if teacher_latents.ndim != 3:
            raise ValueError(
                f"Expected tokenized teacher latents with shape [B*S, Q, Dz], got {tuple(teacher_latents.shape)} "
                f"from {target_key!r}."
            )
        teacher_latents = teacher_latents.to(dtype=current.dtype)
        expected_q = self.config.teacher_num_action_tokens
        expected_dz = self.config.latent_token_dim
        if teacher_latents.shape[1] != expected_q or teacher_latents.shape[2] != expected_dz:
            raise ValueError(
                "Teacher token shape mismatch: expected [B*S, "
                f"{expected_q}, {expected_dz}], got {tuple(teacher_latents.shape)}."
            )
        teacher_latents = teacher_latents.reshape(batch_size, num_steps, expected_q, expected_dz)
        teacher_latents = teacher_latents.flatten(start_dim=1, end_dim=2)

        valid_mask = batch.get("teacher.boundary_valid")
        if valid_mask is None:
            valid_mask = torch.ones(
                (batch_size, num_steps),
                dtype=torch.bool,
                device=teacher_latents.device,
            )
        else:
            valid_mask = valid_mask.to(device=teacher_latents.device, dtype=torch.bool)
        valid_mask = valid_mask.repeat_interleave(expected_q, dim=1)
        return teacher_latents, valid_mask


class MultiTaskDiTLAMPolicy(PreTrainedPolicy):
    config_class = MultiTaskDiTLAMConfig
    name = "multi_task_dit_lam"

    def __init__(self, config: MultiTaskDiTLAMConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        _resolve_teacher_latent_spec(config)
        self.config = config

        self._queues = None
        self.observation_encoder = ObservationEncoder(config)
        conditioning_dim = self.observation_encoder.conditioning_dim

        self.latent_noise_predictor = SequenceDiffusionTransformer(
            config=config,
            conditioning_dim=conditioning_dim,
            token_dim=config.latent_token_dim,
            horizon=config.latent_memory_steps,
        )
        self.action_noise_predictor = LatentConditionedDiffusionTransformer(
            config=config,
            conditioning_dim=conditioning_dim,
            action_dim=config.action_feature.shape[0],
            horizon=config.horizon,
            memory_dim=config.latent_token_dim,
        )

        self.latent_objective = SequenceDiffusionObjective(
            config=config,
            token_dim=config.latent_token_dim,
            horizon=config.latent_memory_steps,
        )
        self.action_objective = SequenceDiffusionObjective(
            config=config,
            token_dim=config.action_feature.shape[0],
            horizon=config.horizon,
        )
        self.teacher_adapter = LatentActionTeacherAdapter(config)
        self.reset()

    def get_optim_params(self) -> list[dict[str, Any]]:
        non_vision_params = []
        vision_encoder_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "observation_encoder.vision_encoder" in name:
                vision_encoder_params.append(param)
            else:
                non_vision_params.append(param)

        return [
            {"params": non_vision_params},
            {
                "params": vision_encoder_params,
                "lr": self.config.optimizer_lr * self.config.vision_encoder_lr_multiplier,
            },
        ]

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        return batch

    @torch.no_grad()
    def _sample_predicted_latents(self, conditioning_vec: Tensor) -> Tensor:
        predicted_latents = self.latent_objective.conditional_sample(
            self.latent_noise_predictor,
            batch_size=conditioning_vec.shape[0],
            conditioning_vec=conditioning_vec,
        )
        return predicted_latents.detach()

    def _generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        conditioning_vec = self.observation_encoder.encode(batch)
        predicted_latents = self._sample_predicted_latents(conditioning_vec)
        actions = self.action_objective.conditional_sample(
            self.action_noise_predictor,
            batch_size=batch_size,
            conditioning_vec=conditioning_vec,
            memory=predicted_latents,
        )
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        for key in batch:
            if key in self._queues:
                batch[key] = torch.stack(list(self._queues[key]), dim=1)
        return self._generate_actions(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        batch = self._prepare_batch(batch)
        conditioning_vec = self.observation_encoder.encode(batch)

        teacher_latents, latent_valid_mask = self.teacher_adapter(batch)
        latent_loss = self.latent_objective.compute_loss(
            self.latent_noise_predictor,
            target_sequence=teacher_latents,
            conditioning_vec=conditioning_vec,
            valid_mask=latent_valid_mask,
        )

        predicted_latents = self._sample_predicted_latents(conditioning_vec)
        action_valid_mask = None
        if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
            action_valid_mask = ~batch["action_is_pad"].bool()

        action_loss = self.action_objective.compute_loss(
            self.action_noise_predictor,
            target_sequence=batch[ACTION],
            conditioning_vec=conditioning_vec,
            valid_mask=action_valid_mask,
            memory=predicted_latents,
        )

        total_loss = action_loss + self.config.lambda_latent * latent_loss
        info = {
            "loss_action": float(action_loss.detach().cpu()),
            "loss_latent": float(latent_loss.detach().cpu()),
            "loss_total": float(total_loss.detach().cpu()),
            "n_segment_steps": self.config.n_segment_steps,
            "latent_memory_steps": self.config.latent_memory_steps,
        }
        return total_loss, info
