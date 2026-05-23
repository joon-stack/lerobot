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

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import DEFAULT_QUANTILES
from lerobot.datasets.utils import STATS_PATH
from lerobot.scripts.recompute_dataset_parquet_stats import (
    recompute_dataset_parquet_stats,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def _write_parquet(path, action, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            ACTION: action.astype(np.float32).tolist(),
            OBS_STATE: state.astype(np.float32).tolist(),
            "episode_index": [0] * len(action),
        }
    )
    pq.write_table(table, path)


def _write_stats(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stale_feature_stats = {
        "min": [0.0, 0.0],
        "max": [1.0, 1.0],
        "mean": [0.5, 0.5],
        "std": [0.5, 0.5],
        "count": [1],
        "q01": [0.01, 0.01],
        "q10": [0.10, 0.10],
        "q50": [0.50, 0.50],
        "q90": [0.90, 0.90],
        "q99": [0.99, 0.99],
    }
    path.write_text(
        json.dumps(
            {
                ACTION: stale_feature_stats,
                OBS_STATE: stale_feature_stats,
                "observation.images.top": {"mean": [0.1], "std": [0.2], "count": [1]},
            }
        )
    )


def test_recompute_dataset_parquet_stats_updates_selected_features_exactly(tmp_path):
    action_0 = np.array([[0.0, 10.0], [1.0, 11.0]], dtype=np.float32)
    action_1 = np.array([[100.0, 20.0], [101.0, 21.0], [102.0, 22.0]], dtype=np.float32)
    state_0 = action_0 + 1000
    state_1 = action_1 + 1000

    _write_parquet(tmp_path / "data/chunk-000/file-000.parquet", action_0, state_0)
    _write_parquet(tmp_path / "data/chunk-000/file-001.parquet", action_1, state_1)
    _write_stats(tmp_path / STATS_PATH)

    recompute_dataset_parquet_stats(tmp_path)

    stats = json.loads((tmp_path / STATS_PATH).read_text())
    expected_action = np.concatenate([action_0, action_1]).astype(np.float64)
    expected_state = np.concatenate([state_0, state_1]).astype(np.float64)

    for feature, expected_values in [
        (ACTION, expected_action),
        (OBS_STATE, expected_state),
    ]:
        np.testing.assert_allclose(
            stats[feature]["min"], np.min(expected_values, axis=0)
        )
        np.testing.assert_allclose(
            stats[feature]["max"], np.max(expected_values, axis=0)
        )
        np.testing.assert_allclose(
            stats[feature]["mean"], np.mean(expected_values, axis=0)
        )
        np.testing.assert_allclose(
            stats[feature]["std"], np.std(expected_values, axis=0)
        )
        assert stats[feature]["count"] == [5]

        for quantile in DEFAULT_QUANTILES:
            q_key = f"q{int(quantile * 100):02d}"
            np.testing.assert_allclose(
                stats[feature][q_key], np.quantile(expected_values, quantile, axis=0)
            )

    assert stats["observation.images.top"] == {
        "mean": [0.1],
        "std": [0.2],
        "count": [1],
    }
    assert (tmp_path / "meta/stats.json.bak").exists()
