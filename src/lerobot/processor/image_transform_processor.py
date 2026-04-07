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

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.datasets.transforms import (
    FastImageTransforms,
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
    get_fast_image_transforms_incompatibility_reason,
)
from lerobot.types import RobotObservation, TransitionKey

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


def _build_image_transforms_config(cfg_dict: dict[str, Any]) -> ImageTransformsConfig:
    cfg_copy = dict(cfg_dict)
    tf_cfgs = cfg_copy.get("tfs", {})
    cfg_copy["tfs"] = {
        tf_name: tf_cfg if isinstance(tf_cfg, ImageTransformConfig) else ImageTransformConfig(**tf_cfg)
        for tf_name, tf_cfg in tf_cfgs.items()
    }
    return ImageTransformsConfig(**cfg_copy)


@ProcessorStepRegistry.register(name="gpu_image_transforms_processor")
@dataclass
class GPUImageTransformsProcessorStep(ObservationProcessorStep):
    """Apply image transforms on-device during training instead of in the dataset workers."""

    image_transforms_cfg: dict[str, Any]
    image_keys: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._cfg = _build_image_transforms_config(self.image_transforms_cfg)
        self._backend = self._cfg.backend

        fast_path_error = get_fast_image_transforms_incompatibility_reason(self._cfg)
        if self._backend == "gpu_fast":
            if fast_path_error is not None:
                raise ValueError(
                    "dataset.image_transforms.backend='gpu_fast' was requested but is unavailable: "
                    f"{fast_path_error}"
                )
            self._transforms = FastImageTransforms(self._cfg)
        elif self._backend == "auto" and fast_path_error is None:
            self._backend = "gpu_fast"
            self._transforms = FastImageTransforms(self._cfg)
        else:
            self._backend = "compatible"
            self._transforms = ImageTransforms(self._cfg)

    def _should_apply(self) -> bool:
        action = self.transition.get(TransitionKey.ACTION)
        return isinstance(action, torch.Tensor)

    def _apply_per_sample(self, value: torch.Tensor) -> torch.Tensor:
        if self._backend == "gpu_fast":
            return self._transforms(value)
        if value.ndim == 5:
            # [B, T, C, H, W]: one sampled transform plan per sample, shared across its temporal axis.
            return torch.stack([self._transforms(sample) for sample in value], dim=0)
        if value.ndim == 4:
            # [B, C, H, W]: one sampled transform plan per sample.
            return torch.stack([self._transforms(sample) for sample in value], dim=0)
        if value.ndim == 3:
            return self._transforms(value)
        return value

    def observation(self, observation: RobotObservation) -> RobotObservation:
        if not self._cfg.enable or not self._should_apply():
            return observation

        image_key_filter = set(self.image_keys)
        new_observation = dict(observation)
        for key, value in observation.items():
            if key not in image_key_filter or not isinstance(value, torch.Tensor):
                continue
            new_observation[key] = self._apply_per_sample(value)
        return new_observation

    def get_config(self) -> dict[str, Any]:
        return {
            "image_transforms_cfg": self.image_transforms_cfg,
            "image_keys": self.image_keys,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
