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

"""
Recompute selected LeRobot dataset statistics directly from data parquet files.

This is intended for repairing dataset-level normalization metadata when
`meta/stats.json` has stale or incorrectly aggregated vector statistics.
The script reads all `data/chunk-*/*.parquet` files, concatenates each selected
feature across all frames, recomputes exact per-dimension statistics, and
replaces only those selected entries in `meta/stats.json`.

Usage:

```bash
python src/lerobot/scripts/recompute_dataset_parquet_stats.py \
    --root=/path/to/dataset
```

Multiple dataset roots can be patched in one run:

```bash
python src/lerobot/scripts/recompute_dataset_parquet_stats.py \
    --root=/path/to/dataset_a /path/to/dataset_b /path/to/dataset_c
```
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import DEFAULT_QUANTILES
from lerobot.datasets.io_utils import load_stats, write_stats
from lerobot.datasets.utils import DATA_DIR, STATS_PATH
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.utils import init_logging

DEFAULT_FEATURES_TO_RECOMPUTE = [ACTION, OBS_STATE]


def _quantile_key(quantile: float) -> str:
    return f"q{int(quantile * 100):02d}"


def _find_data_parquet_files(root: Path) -> list[Path]:
    parquet_files = sorted((root / DATA_DIR).glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under {root / DATA_DIR}/chunk-*/*.parquet"
        )
    return parquet_files


def _column_to_2d_array(parquet_file: Path, feature: str) -> np.ndarray:
    table = pq.read_table(parquet_file, columns=[feature])
    rows = table.column(feature).to_pylist()
    if not rows:
        return np.empty((0, 0), dtype=np.float64)

    first = np.asarray(rows[0])
    if first.ndim == 0:
        return np.asarray(rows, dtype=np.float64).reshape(-1, 1)

    return np.stack([np.asarray(row, dtype=np.float64).reshape(-1) for row in rows])


def _load_feature_array(parquet_files: list[Path], feature: str) -> np.ndarray:
    arrays = []
    for parquet_file in parquet_files:
        try:
            feature_array = _column_to_2d_array(parquet_file, feature)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read feature '{feature}' from {parquet_file}"
            ) from exc

        if feature_array.shape[0] > 0:
            arrays.append(feature_array)

    if not arrays:
        raise ValueError(f"Feature '{feature}' has no rows in the dataset")

    feature_dim = arrays[0].shape[1]
    for array in arrays:
        if array.shape[1] != feature_dim:
            raise ValueError(
                f"Feature '{feature}' has inconsistent dimensions: expected {feature_dim}, got {array.shape[1]}"
            )

    return np.concatenate(arrays, axis=0)


def compute_exact_vector_stats(
    values: np.ndarray,
    quantiles: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute exact per-dimension stats for a vector feature."""
    if quantiles is None:
        quantiles = DEFAULT_QUANTILES

    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {values.shape}")
    if values.shape[0] == 0:
        raise ValueError("Cannot compute stats for an empty array")

    stats = {
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "count": np.array([values.shape[0]], dtype=np.int64),
    }

    for quantile in quantiles:
        stats[_quantile_key(quantile)] = np.quantile(values, quantile, axis=0)

    return stats


def _make_stats_backup(root: Path) -> Path:
    stats_path = root / STATS_PATH
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file does not exist: {stats_path}")

    backup_path = stats_path.with_name(f"{stats_path.name}.bak")
    suffix = 1
    while backup_path.exists():
        backup_path = stats_path.with_name(f"{stats_path.name}.bak.{suffix}")
        suffix += 1

    shutil.copy2(stats_path, backup_path)
    return backup_path


def recompute_dataset_parquet_stats(
    root: Path,
    features: list[str] | None = None,
    *,
    dry_run: bool = False,
    backup: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Recompute selected `meta/stats.json` entries from all data parquet rows."""
    if features is None:
        features = DEFAULT_FEATURES_TO_RECOMPUTE

    stats = load_stats(root)
    if stats is None:
        raise FileNotFoundError(f"Stats file does not exist: {root / STATS_PATH}")

    parquet_files = _find_data_parquet_files(root)
    logging.info(f"Found {len(parquet_files)} parquet files under {root / DATA_DIR}")

    new_feature_stats = {}
    for feature in features:
        if feature not in stats:
            raise KeyError(f"Feature '{feature}' is not present in {root / STATS_PATH}")

        values = _load_feature_array(parquet_files, feature)
        logging.info(
            f"Recomputed {feature}: rows={values.shape[0]}, dims={values.shape[1]}"
        )
        new_feature_stats[feature] = compute_exact_vector_stats(values)

    if dry_run:
        return new_feature_stats

    if backup:
        backup_path = _make_stats_backup(root)
        logging.info(f"Backed up existing stats to {backup_path}")

    stats.update(new_feature_stats)
    write_stats(stats, root)
    logging.info(f"Updated {root / STATS_PATH}")
    return new_feature_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute selected LeRobot dataset stats from all data parquet rows."
    )
    parser.add_argument(
        "--root",
        type=Path,
        nargs="+",
        required=True,
        help="One or more local LeRobot dataset roots containing meta/stats.json and data/.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES_TO_RECOMPUTE,
        help="Stats entries to recompute. Defaults to action and observation.state.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and log stats without writing meta/stats.json.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a stats.json.bak file before writing.",
    )

    args = parser.parse_args()
    init_logging()

    for root in args.root:
        logging.info(f"Processing dataset root: {root}")
        recompute_dataset_parquet_stats(
            root=root,
            features=args.features,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )


if __name__ == "__main__":
    main()
