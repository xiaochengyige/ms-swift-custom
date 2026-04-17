from __future__ import annotations

import json
import os
from functools import partial
from typing import Any, Dict, Optional

import torch

from alpamayo_r1.data.pai import PAIDataset
from alpamayo_r1.processor.qwen_processor import QwenProcessor
from finetune.sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
from finetune.sft.models.sft_base_model import TrainableReasoningVLA
from swift.ray import RayHelper
from swift.utils import get_logger, is_master
from swift.llm.train.sft import SwiftSft

from .args import AlpamayoTrainArguments

logger = get_logger()


@RayHelper.worker(group=["default"])
class AlpamayoSwiftSft(SwiftSft):
    args_class = AlpamayoTrainArguments
    args: AlpamayoTrainArguments

    def _resolve_stage1_vlm_source(self) -> Optional[str]:
        args = self.args
        if args.stage != "stage2" or args.resume_from_checkpoint:
            return None
        return args.stage1_vlm_checkpoint_path

    def _resolve_vlm_name_or_path(self) -> str:
        args = self.args
        if args.alpamayo_vlm_name_or_path:
            return args.alpamayo_vlm_name_or_path

        config_path = os.path.join(args.model, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if config.get("vlm_name_or_path"):
                return config["vlm_name_or_path"]
        return "Qwen/Qwen3-VL-8B-Instruct"

    def _attach_swift_metadata(self, model, processor) -> None:
        model.model_dir = self.args.model_dir
        model.model_info = self.args.model_info
        model.model_meta = self.args.model_meta
        processor.model_info = self.args.model_info
        processor.model_meta = self.args.model_meta
        self.args.model_info.config = getattr(model, "config", None)

    @RayHelper.function(group="default")
    def _prepare_model_tokenizer(self, **kwargs):
        args = self.args
        if args.stage == "stage1":
            model = TrainableReasoningVLA.from_alpamayo_checkpoint(
                args.model,
                vlm_name_or_path=self._resolve_vlm_name_or_path(),
            )
        else:
            model = TrainableAlpamayoR1.from_pretrained(
                args.model,
                torch_dtype=args.torch_dtype,
                cotrain_vlm=False,
                stop_grad_from_vlm=True,
                stage1_vlm_checkpoint_path=self._resolve_stage1_vlm_source(),
            )

        self.alpamayo_processor = QwenProcessor(
            vlm_name_or_path=model.config.vlm_name_or_path,
            traj_vocab_size=getattr(model.config, "traj_vocab_size", None),
            min_pixels=getattr(model.config, "min_pixels", None),
            max_pixels=getattr(model.config, "max_pixels", None),
            include_camera_ids=args.include_camera_ids,
            include_frame_nums=args.include_frame_nums,
        )
        processor = self.alpamayo_processor.processor
        self._attach_swift_metadata(model, processor)
        self.model = model
        self.processor = processor

        if getattr(self.model, "generation_config", None) is None and getattr(self.model, "vlm", None) is not None:
            self.model.generation_config = getattr(self.model.vlm, "generation_config", None)
        if getattr(self.model, "generation_config", None) is not None:
            self._prepare_generation_config()

    def _get_vla_preprocess_args(self, generation_mode: bool) -> Dict[str, Any]:
        return {
            "_target_": "alpamayo_r1.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
            "components_order": ["image", "traj_history", "prompt", "traj_future"],
            "components_prompt": ["traj_future"],
            "label_components": ["traj_future"],
            "generation_mode": generation_mode,
            "include_camera_ids": self.args.include_camera_ids,
            "include_frame_nums": self.args.include_frame_nums,
        }

    def _build_dataset(self, chunk_ids, generation_mode: bool):
        if chunk_ids is None:
            return None
        return PAIDataset(
            local_dir=self.args.pai_local_dir,
            chunk_ids=chunk_ids,
            model_config=self.model.config,
            vla_preprocess_args=self._get_vla_preprocess_args(generation_mode),
            use_default_keyframe=self.args.use_default_keyframe,
        )

    @RayHelper.function(group="default")
    def _prepare_dataset(self):
        train_dataset = self._build_dataset(self.args.train_chunk_ids, generation_mode=False)
        val_dataset = self._build_dataset(self.args.val_chunk_ids, generation_mode=True)
        self._show_dataset(train_dataset, val_dataset)
        return train_dataset, val_dataset

    def _get_data_collator(self):
        padding_side = self.args.padding_side or "left"
        return partial(self.alpamayo_processor.collate_fn, padding_side=padding_side)

    @staticmethod
    def _describe_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, dict):
            return {k: AlpamayoSwiftSft._describe_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return {"type": "list", "length": len(value)}
        return type(value).__name__

    def _show_dataset(self, train_dataset, val_dataset):
        if train_dataset is not None:
            self.train_msg["train_dataset_size"] = len(train_dataset)
        if val_dataset is not None:
            self.train_msg["val_dataset_size"] = len(val_dataset)

        if not is_master() or train_dataset is None or len(train_dataset) == 0:
            return

        sample = train_dataset[0]
        logger.info(f"alpamayo train sample keys: {sorted(sample.keys())}")
        logger.info(
            "alpamayo train sample summary: "
            f"{json.dumps(self._describe_value(sample), ensure_ascii=False)}"
        )
        if val_dataset is not None and len(val_dataset) > 0:
            val_sample = val_dataset[0]
            logger.info(f"alpamayo val sample keys: {sorted(val_sample.keys())}")
            logger.info(
                "alpamayo val sample summary: "
                f"{json.dumps(self._describe_value(val_sample), ensure_ascii=False)}"
            )
