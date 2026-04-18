# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict
from typing import Any, Final, Mapping, Optional

try:
    import torch
    import torch.distributed as dist
except ModuleNotFoundError:  # pragma: no cover - lightweight test env
    torch = None
    dist = None

IGNORE_INDEX: Final = -100

TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}

SPECIAL_TOKENS_KEYS = [
    "prompt_start",
    "prompt_end",
    "image_start",
    "image_pre_tkn",
    "image_end",
    "traj_history_start",
    "traj_history_pre_tkn",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
    "traj_future_start",
    "traj_future_pre_tkn",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "vectorized_wm",
    "vectorized_wm_start",
    "vectorized_wm_end",
    "vectorized_wm_pre_tkn",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
]
SPECIAL_TOKENS = {key: f"<|{key}|>" for key in SPECIAL_TOKENS_KEYS}

CROSS_LEFT_CAMERA_NAME: Final = "camera_cross_left_120fov"
CROSS_RIGHT_CAMERA_NAME: Final = "camera_cross_right_120fov"
FRONT_TELE_CAMERA_NAME: Final = "camera_front_tele_30fov"
FRONT_WIDE_CAMERA_NAME: Final = "camera_front_wide_120fov"
REAR_LEFT_CAMERA_NAME: Final = "camera_rear_left_70fov"
REAR_RIGHT_CAMERA_NAME: Final = "camera_rear_right_70fov"
REAR_TELE_CAMERA_NAME: Final = "camera_rear_tele_30fov"

CAMERA_NAMES_TO_INDICES = {
    CROSS_LEFT_CAMERA_NAME: 0,
    FRONT_WIDE_CAMERA_NAME: 1,
    CROSS_RIGHT_CAMERA_NAME: 2,
    REAR_LEFT_CAMERA_NAME: 3,
    REAR_TELE_CAMERA_NAME: 4,
    REAR_RIGHT_CAMERA_NAME: 5,
    FRONT_TELE_CAMERA_NAME: 6,
}
CAMERA_INDICES_TO_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}

DEFAULT_PAI_CHUNK_BY_SPLIT = {
    "train": "0-99",
    "val": "99-100",
}

DEFAULT_SYSTEM_PROMPT: Final = (
    "You are a driving assistant that generates safe and accurate actions."
)
DEFAULT_USER_PROMPT: Final = "output the future trajectory."

_LOGGING_INITIALIZED = False


def setup_logging() -> None:
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return
    _LOGGING_INITIALIZED = True

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s"))
    root.addHandler(handler)


def get_global_rank() -> int:
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def rank_prefixed_message(message: str, rank: Optional[int] = None) -> str:
    if rank is None:
        return message
    return f"[rank: {rank}] {message}"


class RankedLogger(logging.LoggerAdapter):
    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        setup_logging()
        super().__init__(logger=logging.getLogger(name), extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, *args, rank: Optional[int] = None, **kwargs) -> None:
        if not self.isEnabledFor(level):
            return
        msg, kwargs = self.process(msg, kwargs)
        current_rank = get_global_rank()
        msg = rank_prefixed_message(msg, current_rank)
        if self.rank_zero_only and current_rank != 0:
            return
        if rank is not None and rank != current_rank:
            return
        self.logger.log(level, msg, *args, **kwargs)


def get_param_count(nn_model: Any, depth: int = 2) -> dict[str, int]:
    if depth < 1:
        raise ValueError("Provided depth must be greater than 0.")
    if torch is None:
        raise ModuleNotFoundError("torch is required to compute parameter counts.")
    param_counts = defaultdict(int, {"total_params": 0, "trainable_params": 0})
    for name, param in nn_model.named_parameters():
        bucket = ".".join(name.split(".")[:depth])
        param_counts[bucket] += param.numel()
        if param.requires_grad:
            param_counts["trainable_params"] += param.numel()
        param_counts["total_params"] += param.numel()
    return dict(param_counts)


def parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def require_env(name: str) -> str:
    value = get_env(name)
    if value is None:
        raise ValueError(
            f"Missing required runtime option `{name}`. Pass it with --model_kwargs '{{\"{name.lower()}\": ...}}'."
        )
    return value


__all__ = [
    "CAMERA_INDICES_TO_DISPLAY_NAMES",
    "CAMERA_NAMES_TO_INDICES",
    "DEFAULT_PAI_CHUNK_BY_SPLIT",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_PROMPT",
    "IGNORE_INDEX",
    "RankedLogger",
    "SPECIAL_TOKENS",
    "TRAJ_TOKEN",
    "get_env",
    "get_param_count",
    "parse_bool",
    "require_env",
]
