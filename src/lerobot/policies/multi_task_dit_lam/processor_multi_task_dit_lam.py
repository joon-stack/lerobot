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
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.multi_task_dit_lam.configuration_multi_task_dit_lam import MultiTaskDiTLAMConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


@ProcessorStepRegistry.register("teacher_boundary_stash_trim_v1")
@dataclass
class TeacherBoundaryStashAndTrimProcessorStep(ProcessorStep):
    """Stash raw teacher boundary inputs while trimming model observations back to history only."""

    history_len: int
    boundary_offsets: list[int]
    teacher_obs_keys: list[str] = field(default_factory=list)

    def get_config(self) -> dict[str, Any]:
        return {
            "history_len": self.history_len,
            "boundary_offsets": self.boundary_offsets,
            "teacher_obs_keys": self.teacher_obs_keys,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def _infer_boundary_valid(self, action_is_pad: torch.Tensor, batch_size: int) -> torch.Tensor:
        if action_is_pad.ndim == 1:
            action_is_pad = action_is_pad.unsqueeze(0)

        indices = []
        for offset in self.boundary_offsets:
            action_idx = min(self.history_len - 1 + max(offset - 1, 0), action_is_pad.shape[-1] - 1)
            indices.append(action_idx)

        if not indices:
            return torch.ones((batch_size, 0), dtype=torch.bool, device=action_is_pad.device)

        index_tensor = torch.tensor(indices, dtype=torch.long, device=action_is_pad.device)
        boundary_is_pad = action_is_pad.index_select(dim=-1, index=index_tensor)
        return ~boundary_is_pad.bool()

    def __call__(self, transition):
        observation = transition.get(TransitionKey.OBSERVATION)
        complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})

        if not observation or not self.boundary_offsets:
            return transition

        expected_seq_len = self.history_len + len(self.boundary_offsets)
        teacher_key_filter = set(self.teacher_obs_keys)
        new_observation = {}
        batch_size = None
        trimmed_any = False

        for key, value in observation.items():
            if not isinstance(value, torch.Tensor) or value.ndim < 3 or value.shape[1] < expected_seq_len:
                new_observation[key] = value
                continue

            trimmed_any = True
            batch_size = value.shape[0]
            history = value[:, : self.history_len]
            future_boundaries = value[:, self.history_len : expected_seq_len]
            new_observation[key] = history

            if not teacher_key_filter or key in teacher_key_filter:
                complementary_data[f"teacher.current.{key}"] = history[:, self.history_len - 1]
                complementary_data[f"teacher.boundary.{key}"] = future_boundaries

        if not trimmed_any or batch_size is None:
            return transition

        if "action_is_pad" in complementary_data:
            complementary_data["teacher.boundary_valid"] = self._infer_boundary_valid(
                action_is_pad=complementary_data["action_is_pad"],
                batch_size=batch_size,
            )
        else:
            complementary_data["teacher.boundary_valid"] = torch.ones(
                (batch_size, len(self.boundary_offsets)),
                dtype=torch.bool,
            )

        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition


def make_multi_task_dit_lam_pre_post_processors(
    config: MultiTaskDiTLAMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    teacher_obs_keys = (
        list(config.teacher_obs_key_map.keys())
        if config.teacher_obs_key_map
        else list(config.image_features.keys())[:1]
    )
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.text_encoder_name,
            padding=config.tokenizer_padding,
            padding_side=config.tokenizer_padding_side,
            max_length=config.tokenizer_max_length,
            truncation=config.tokenizer_truncation,
        ),
        TeacherBoundaryStashAndTrimProcessorStep(
            history_len=config.n_obs_steps,
            boundary_offsets=config.latent_boundary_offsets,
            teacher_obs_keys=teacher_obs_keys,
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
