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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def test_from_pretrained_full_input_features_override_replaces_existing_dict(tmp_path):
    config = SmolVLAConfig(device="cpu")
    config.input_features = {
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.camera3": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    }
    config.save_pretrained(tmp_path)

    loaded = PreTrainedConfig.from_pretrained(
        tmp_path,
        cli_overrides=[
            '--input_features={"observation.state":{"type":"STATE","shape":[7]},'
            '"observation.images.top":{"type":"VISUAL","shape":[3,480,640]},'
            '"observation.images.wrist":{"type":"VISUAL","shape":[3,480,640]}}'
        ],
    )

    assert set(loaded.input_features) == {
        "observation.state",
        "observation.images.top",
        "observation.images.wrist",
    }
