#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from .configuration_multi_task_dit_lam import MultiTaskDiTLAMConfig
from .modeling_multi_task_dit_lam import MultiTaskDiTLAMPolicy
from .processor_multi_task_dit_lam import make_multi_task_dit_lam_pre_post_processors

__all__ = [
    "MultiTaskDiTLAMConfig",
    "MultiTaskDiTLAMPolicy",
    "make_multi_task_dit_lam_pre_post_processors",
]
