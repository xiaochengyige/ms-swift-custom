from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from swift.llm.argument import TrainArguments

from .register import (
    ALPAMAYO_STAGE1_MODEL_TYPE,
    ALPAMAYO_STAGE2_MODEL_TYPE,
    ALPAMAYO_TEMPLATE,
)


@dataclass
class AlpamayoTrainArguments(TrainArguments):
    stage: Optional[str] = None
    pai_local_dir: Optional[str] = None
    train_chunk_ids: Optional[str] = "0-99"
    val_chunk_ids: Optional[str] = "99-100"
    stage1_vlm_checkpoint_path: Optional[str] = None
    alpamayo_vlm_name_or_path: Optional[str] = None
    use_default_keyframe: bool = True
    include_camera_ids: bool = False
    include_frame_nums: bool = False

    def _normalize_stage(self) -> None:
        if self.stage is None:
            stage_by_model_type = {
                ALPAMAYO_STAGE1_MODEL_TYPE: "stage1",
                ALPAMAYO_STAGE2_MODEL_TYPE: "stage2",
            }
            self.stage = stage_by_model_type.get(self.model_type)
        if self.stage is None:
            raise ValueError("Please set `--stage stage1|stage2`.")
        self.stage = self.stage.lower()
        if self.stage not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported stage: {self.stage}")

    def _normalize_chunk_ids(self, value):
        if value in (None, "", "none", "null"):
            return None
        if isinstance(value, list):
            return value
        if not isinstance(value, str):
            return value

        text = value.strip()
        if not text:
            return None
        if text.startswith("["):
            parsed = json.loads(text)
            return [int(item) for item in parsed]
        if "," in text:
            return [int(item.strip()) for item in text.split(",") if item.strip()]
        return text

    def _inject_alpamayo_defaults(self) -> None:
        model_type_by_stage = {
            "stage1": ALPAMAYO_STAGE1_MODEL_TYPE,
            "stage2": ALPAMAYO_STAGE2_MODEL_TYPE,
        }
        model_type = model_type_by_stage[self.stage]
        if self.model_type is None:
            self.model_type = model_type
        elif self.model_type != model_type:
            raise ValueError(
                f"`--stage {self.stage}` expects `--model_type {model_type}`, got `{self.model_type}`."
            )

        if self.template is None:
            self.template = ALPAMAYO_TEMPLATE
        if not self.dataset and not self.cached_dataset:
            # We bypass ms-swift dataset loading in the custom pipeline, but TrainArguments
            # still requires a non-empty dataset field during initialization.
            self.dataset = ["__alpamayo_placeholder__"]
        if self.lazy_tokenize is None:
            self.lazy_tokenize = False
        self.remove_unused_columns = False
        self.train_type = "full"

    def _validate_alpamayo_args(self) -> None:
        if not self.pai_local_dir:
            raise ValueError("Please set `--pai_local_dir` to the local PAI dataset root.")
        self.pai_local_dir = os.path.abspath(os.path.expanduser(self.pai_local_dir))
        if self.stage == "stage2" and not self.resume_from_checkpoint and not self.stage1_vlm_checkpoint_path:
            raise ValueError(
                "`--stage1_vlm_checkpoint_path` is required for fresh stage2 training."
            )
        if self.stage1_vlm_checkpoint_path:
            self.stage1_vlm_checkpoint_path = os.path.abspath(
                os.path.expanduser(self.stage1_vlm_checkpoint_path)
            )

    def __post_init__(self) -> None:
        self._normalize_stage()
        self.train_chunk_ids = self._normalize_chunk_ids(self.train_chunk_ids)
        self.val_chunk_ids = self._normalize_chunk_ids(self.val_chunk_ids)
        self._inject_alpamayo_defaults()
        super().__post_init__()
        self._validate_alpamayo_args()

    def load_args_from_ckpt(self) -> None:
        super().load_args_from_ckpt()
        args_path = os.path.join(self.ckpt_dir, "args.json")
        with open(args_path, "r", encoding="utf-8") as f:
            old_args = json.load(f)

        default_values = {
            "stage": None,
            "pai_local_dir": None,
            "train_chunk_ids": "0-99",
            "val_chunk_ids": "99-100",
            "stage1_vlm_checkpoint_path": None,
            "alpamayo_vlm_name_or_path": None,
            "use_default_keyframe": True,
            "include_camera_ids": False,
            "include_frame_nums": False,
        }
        custom_keys = [
            "stage",
            "pai_local_dir",
            "train_chunk_ids",
            "val_chunk_ids",
            "stage1_vlm_checkpoint_path",
            "alpamayo_vlm_name_or_path",
            "use_default_keyframe",
            "include_camera_ids",
            "include_frame_nums",
        ]
        for key in custom_keys:
            old_value = old_args.get(key)
            if old_value is None:
                continue
            value = getattr(self, key, None)
            if value in (None, [], "") or value == default_values[key]:
                setattr(self, key, old_value)
