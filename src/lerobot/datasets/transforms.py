#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import collections
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import (
    Transform,
    functional as F,  # noqa: N812
)

try:
    import kornia.enhance as KE
    import kornia.geometry.transform as KGT

    _KORNIA_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - covered by compatibility tests when kornia is installed
    KE = None
    KGT = None
    _KORNIA_IMPORT_ERROR = exc


class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms: list of transformations.
        p: represents the multinomial probabilities (with no replacement) used for sampling the transform.
            If the sum of the weights is not 1, they will be normalized. If ``None`` (default), all transforms
            have the same probability.
        n_subset: number of transformations to apply. If ``None``, all transforms are applied.
            Must be in [1, len(transforms)].
        random_order: apply transformations in a random order.
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video.

    Similar to a v2.RandomAdjustSharpness with p=1 and a sharpness_factor sampled randomly.
    While v2.RandomAdjustSharpness applies — with a given probability — a fixed sharpness_factor to an image,
    SharpnessJitter applies a random sharpness_factor each time. This is to have a more diverse set of
    augmentations as a result.

    A sharpness_factor of 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness
    by a factor of 2.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness: How much to jitter sharpness. sharpness_factor is chosen uniformly from
            [max(0, 1 - sharpness), 1 + sharpness] or the given
            [min, max]. Should be non negative numbers.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int | float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


@dataclass
class ImageTransformConfig:
    """
    For each transform, the following parameters are available:
      weight: This represents the multinomial probability (with no replacement)
            used for sampling the transform. If the sum of the weights is not 1,
            they will be normalized.
      type: The name of the class used. This is either a class available under torchvision.transforms.v2 or a
            custom transform defined here.
      kwargs: Lower & upper bound respectively used for sampling the transform's parameter
            (following uniform distribution) when it's applied.
    """

    weight: float = 1.0
    type: str = "Identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageTransformsConfig:
    """
    These transforms are all using standard torchvision.transforms.v2
    You can find out how these transformations affect images here:
    https://pytorch.org/vision/0.18/auto_examples/transforms/plot_transforms_illustrations.html
    We use a custom RandomSubsetApply container to sample them.
    """

    # Set this flag to `true` to enable transforms during training
    enable: bool = False
    # This is the maximum number of transforms (sampled from these below) that will be applied to each frame.
    # It's an integer in the interval [1, number_of_available_transforms].
    max_num_transforms: int = 3
    # By default, transforms are applied in Torchvision's suggested order (shown below).
    # Set this to True to apply them in a random order.
    random_order: bool = False
    # Controls where image transforms are executed during training.
    # - "compatible": keep the original transform implementation and semantics.
    # - "gpu_fast": use the batched Kornia fast path (requires CUDA + Kornia + supported transforms).
    # - "auto": pick "gpu_fast" only when it is safe to do so, otherwise fall back to "compatible".
    backend: str = "auto"
    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
            "brightness": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)},
            ),
            "contrast": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)},
            ),
            "saturation": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)},
            ),
            "hue": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)},
            ),
            "sharpness": ImageTransformConfig(
                weight=1.0,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
            "affine": ImageTransformConfig(
                weight=1.0,
                type="RandomAffine",
                kwargs={"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)},
            ),
        }
    )


FAST_IMAGE_TRANSFORM_BACKENDS = {"auto", "compatible", "gpu_fast"}


@dataclass(frozen=True)
class FastTransformSpec:
    name: str
    kind: str
    weight: float
    kwargs: dict[str, Any]


@dataclass
class FastAugPlan:
    selected_mask: torch.Tensor
    params: dict[str, dict[str, torch.Tensor]]

    def repeat_interleave(self, repeats: int) -> "FastAugPlan":
        if repeats == 1:
            return self
        return FastAugPlan(
            selected_mask=self.selected_mask.repeat_interleave(repeats, dim=0),
            params={
                name: {
                    key: value.repeat_interleave(repeats, dim=0)
                    for key, value in transform_params.items()
                }
                for name, transform_params in self.params.items()
            },
        )


def is_kornia_available() -> bool:
    return KE is not None and KGT is not None


def _make_bounds(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        scalar = float(value)
        if name in {"degrees"}:
            if scalar < 0:
                raise ValueError(f"{name} must be non-negative when provided as a scalar.")
            lower, upper = -scalar, scalar
        elif name in {"brightness", "contrast", "saturation", "sharpness"}:
            if scalar < 0:
                raise ValueError(f"{name} must be non-negative when provided as a scalar.")
            lower, upper = max(0.0, 1.0 - scalar), 1.0 + scalar
        elif name == "hue":
            if not -0.5 <= scalar <= 0.5:
                raise ValueError(f"{name} scalar must be in [-0.5, 0.5].")
            lower, upper = -abs(scalar), abs(scalar)
        else:
            raise TypeError(f"{name} scalar format is unsupported for the fast GPU backend.")
    elif isinstance(value, collections.abc.Sequence) and len(value) == 2:
        lower, upper = float(value[0]), float(value[1])
    else:
        raise TypeError(f"{name} must be provided as a scalar or [min, max] pair for the fast GPU backend.")
    if lower > upper:
        raise ValueError(f"Invalid bounds for {name}: ({lower}, {upper})")
    return lower, upper


def _make_affine_translation_bounds(value: Any) -> tuple[float, float]:
    if not isinstance(value, collections.abc.Sequence) or len(value) != 2:
        raise TypeError("RandomAffine translate must be a pair of fractions for the fast GPU backend.")
    tx, ty = float(value[0]), float(value[1])
    if tx < 0 or ty < 0:
        raise ValueError("RandomAffine translate bounds must be non-negative fractions.")
    return tx, ty


def _parse_fast_transform_spec(name: str, cfg: ImageTransformConfig) -> FastTransformSpec:
    if cfg.weight <= 0:
        raise ValueError(f"Transform '{name}' has non-positive weight and should not be parsed.")

    if cfg.type == "ColorJitter":
        supported_keys = [key for key in ("brightness", "contrast", "saturation", "hue") if key in cfg.kwargs]
        if len(supported_keys) != 1 or len(cfg.kwargs) != 1:
            raise ValueError(
                f"Fast GPU backend only supports single-attribute ColorJitter entries, got {cfg.kwargs}."
            )
        return FastTransformSpec(name=name, kind=supported_keys[0], weight=cfg.weight, kwargs=cfg.kwargs)

    if cfg.type == "SharpnessJitter":
        if set(cfg.kwargs) != {"sharpness"}:
            raise ValueError(
                f"Fast GPU backend only supports SharpnessJitter(sharpness=...), got {cfg.kwargs}."
            )
        return FastTransformSpec(name=name, kind="sharpness", weight=cfg.weight, kwargs=cfg.kwargs)

    if cfg.type == "RandomRotation":
        if set(cfg.kwargs) != {"degrees"}:
            raise ValueError(
                f"Fast GPU backend only supports RandomRotation(degrees=...), got {cfg.kwargs}."
            )
        return FastTransformSpec(name=name, kind="rotation", weight=cfg.weight, kwargs=cfg.kwargs)

    if cfg.type == "RandomAffine":
        supported_keys = {"degrees", "translate"}
        if not set(cfg.kwargs).issubset(supported_keys):
            raise ValueError(
                "Fast GPU backend only supports RandomAffine with degrees and optional translate."
            )
        if "degrees" not in cfg.kwargs:
            raise ValueError("Fast GPU backend requires RandomAffine degrees to be specified.")
        return FastTransformSpec(name=name, kind="affine", weight=cfg.weight, kwargs=cfg.kwargs)

    raise ValueError(
        f"Fast GPU backend does not support transform '{name}' with type '{cfg.type}'."
    )


def get_fast_image_transforms_incompatibility_reason(cfg: ImageTransformsConfig) -> str | None:
    if not cfg.enable:
        return "image transforms are disabled"
    if cfg.backend not in FAST_IMAGE_TRANSFORM_BACKENDS:
        return f"unknown image transform backend '{cfg.backend}'"
    active_cfgs = [tf_cfg for tf_cfg in cfg.tfs.values() if tf_cfg.weight > 0.0]
    if len(active_cfgs) == 0 or cfg.max_num_transforms <= 0:
        return None
    if not is_kornia_available():
        return f"kornia is not installed: {_KORNIA_IMPORT_ERROR}"
    if cfg.random_order:
        return "fast GPU backend does not support random_order=true"

    try:
        for name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue
            _parse_fast_transform_spec(name, tf_cfg)
    except (TypeError, ValueError) as exc:
        return str(exc)
    return None


class FastImageTransforms:
    """Batched GPU image transforms with the same public recipe as ImageTransforms."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        incompatibility = get_fast_image_transforms_incompatibility_reason(cfg)
        if incompatibility is not None:
            raise ValueError(f"Fast GPU image transforms are unavailable: {incompatibility}")

        self._cfg = cfg
        self.specs = [
            _parse_fast_transform_spec(name, tf_cfg)
            for name, tf_cfg in cfg.tfs.items()
            if tf_cfg.weight > 0.0
        ]
        self._weights = torch.tensor([spec.weight for spec in self.specs], dtype=torch.float32)
        self._n_subset = min(len(self.specs), cfg.max_num_transforms)

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        if not self._cfg.enable or self._n_subset == 0:
            return value
        if value.ndim == 5:
            batch_size, horizon, channels, height, width = value.shape
            images = value.reshape(batch_size * horizon, channels, height, width)
            plan = self._sample_plan(batch_size, height, width, value.device, value.dtype)
            output = self._apply_plan(images, plan.repeat_interleave(horizon))
            return output.reshape(batch_size, horizon, channels, height, width)
        if value.ndim == 4:
            batch_size, _, height, width = value.shape
            plan = self._sample_plan(batch_size, height, width, value.device, value.dtype)
            return self._apply_plan(value, plan)
        if value.ndim == 3:
            return self(value.unsqueeze(0)).squeeze(0)
        return value

    def _sample_uniform(
        self, bounds: tuple[float, float], batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        lower, upper = bounds
        return torch.empty(batch_size, device=device, dtype=dtype).uniform_(lower, upper)

    def _sample_plan(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> FastAugPlan:
        weights = self._weights.to(device=device)
        selected_indices = torch.multinomial(weights.expand(batch_size, -1), self._n_subset, replacement=False)
        selected_mask = torch.zeros(batch_size, len(self.specs), dtype=torch.bool, device=device)
        selected_mask.scatter_(1, selected_indices, True)

        params: dict[str, dict[str, torch.Tensor]] = {}
        for spec in self.specs:
            if spec.kind == "brightness":
                params[spec.name] = {
                    "factor": self._sample_uniform(
                        _make_bounds(spec.kwargs["brightness"], "brightness"),
                        batch_size,
                        device,
                        dtype,
                    )
                }
            elif spec.kind == "contrast":
                params[spec.name] = {
                    "factor": self._sample_uniform(
                        _make_bounds(spec.kwargs["contrast"], "contrast"),
                        batch_size,
                        device,
                        dtype,
                    )
                }
            elif spec.kind == "saturation":
                params[spec.name] = {
                    "factor": self._sample_uniform(
                        _make_bounds(spec.kwargs["saturation"], "saturation"),
                        batch_size,
                        device,
                        dtype,
                    )
                }
            elif spec.kind == "hue":
                hue_factor = self._sample_uniform(
                    _make_bounds(spec.kwargs["hue"], "hue"), batch_size, device, dtype
                )
                params[spec.name] = {"factor": hue_factor * math.tau}
            elif spec.kind == "sharpness":
                params[spec.name] = {
                    "factor": self._sample_uniform(
                        _make_bounds(spec.kwargs["sharpness"], "sharpness"),
                        batch_size,
                        device,
                        dtype,
                    )
                }
            elif spec.kind == "rotation":
                params[spec.name] = {
                    "angle": self._sample_uniform(
                        _make_bounds(spec.kwargs["degrees"], "degrees"),
                        batch_size,
                        device,
                        dtype,
                    )
                }
            elif spec.kind == "affine":
                angle = self._sample_uniform(
                    _make_bounds(spec.kwargs["degrees"], "degrees"), batch_size, device, dtype
                )
                if "translate" in spec.kwargs:
                    tx_bound, ty_bound = _make_affine_translation_bounds(spec.kwargs["translate"])
                    translations = torch.empty(batch_size, 2, device=device, dtype=dtype)
                    translations[:, 0].uniform_(-tx_bound * width, tx_bound * width)
                    translations[:, 1].uniform_(-ty_bound * height, ty_bound * height)
                else:
                    translations = torch.zeros(batch_size, 2, device=device, dtype=dtype)
                params[spec.name] = {"angle": angle, "translation": translations}
            else:  # pragma: no cover - guarded by compatibility checks
                raise ValueError(f"Unsupported fast transform kind '{spec.kind}'.")

        return FastAugPlan(selected_mask=selected_mask, params=params)

    @staticmethod
    def _broadcast_factor(factor: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        return factor.view(-1, *([1] * (images.ndim - 1)))

    def _apply_plan(self, images: torch.Tensor, plan: FastAugPlan) -> torch.Tensor:
        transformed = images
        height, width = transformed.shape[-2:]

        for idx, spec in enumerate(self.specs):
            active_mask = plan.selected_mask[:, idx]
            if not active_mask.any():
                continue

            params = plan.params[spec.name]
            if spec.kind == "brightness":
                candidate = (transformed * self._broadcast_factor(params["factor"], transformed)).clamp(0.0, 1.0)
            elif spec.kind == "contrast":
                candidate = KE.adjust_contrast_with_mean_subtraction(transformed, params["factor"]).clamp(0.0, 1.0)
            elif spec.kind == "saturation":
                candidate = KE.adjust_saturation(transformed, params["factor"]).clamp(0.0, 1.0)
            elif spec.kind == "hue":
                candidate = KE.adjust_hue(transformed, params["factor"]).clamp(0.0, 1.0)
            elif spec.kind == "sharpness":
                candidate = KE.sharpness(transformed, params["factor"]).clamp(0.0, 1.0)
            elif spec.kind == "rotation":
                candidate = KGT.rotate(
                    transformed,
                    params["angle"],
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
            elif spec.kind == "affine":
                center = transformed.new_tensor([(width - 1) / 2.0, (height - 1) / 2.0]).repeat(
                    transformed.shape[0], 1
                )
                scale = torch.ones(transformed.shape[0], 2, device=transformed.device, dtype=transformed.dtype)
                matrix = KGT.get_affine_matrix2d(
                    params["translation"],
                    center,
                    scale,
                    params["angle"],
                )
                candidate = KGT.warp_affine(
                    transformed,
                    matrix[:, :2, :],
                    dsize=(height, width),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
            else:  # pragma: no cover - guarded by compatibility checks
                raise ValueError(f"Unsupported fast transform kind '{spec.kind}'.")

            transformed = torch.where(active_mask.view(-1, 1, 1, 1), candidate, transformed)

        return transformed


def make_transform_from_config(cfg: ImageTransformConfig):
    if cfg.type == "SharpnessJitter":
        return SharpnessJitter(**cfg.kwargs)

    transform_cls = getattr(v2, cfg.type, None)
    if isinstance(transform_cls, type) and issubclass(transform_cls, Transform):
        return transform_cls(**cfg.kwargs)

    raise ValueError(
        f"Transform '{cfg.type}' is not valid. It must be a class in "
        f"torchvision.transforms.v2 or 'SharpnessJitter'."
    )


class ImageTransforms(Transform):
    """A class to compose image transforms based on configuration."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg

        self.weights = []
        self.transforms = {}
        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue

            self.transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

    def forward(self, *inputs: Any) -> Any:
        return self.tf(*inputs)
