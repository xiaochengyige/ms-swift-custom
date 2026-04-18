# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import einops
import torch
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoModel, AutoProcessor, PreTrainedModel, PretrainedConfig
from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.utils import ModelOutput

from .support import IGNORE_INDEX, SPECIAL_TOKENS, TRAJ_TOKEN, RankedLogger, get_env, get_param_count, parse_bool
from .trajectory import (
    build_action_input_projector,
    build_action_output_projector,
    build_action_space,
    build_diffusion,
    build_trajectory_tokenizer,
)

logger = RankedLogger(__name__, rank_zero_only=True)

ALPAMAYO1_STAGE1_MODEL_TYPE = "alpamayo1_stage1"
ALPAMAYO1_STAGE2_MODEL_TYPE = "alpamayo1_stage2"

DEFAULT_TRAJ_TOKENIZER_CFG = {
    "type": "discrete",
    "action_space": {"type": "unicycle_accel_curvature", "n_waypoints": 64, "dt": 0.1},
    "dims_min": [-9.8, -0.2],
    "dims_max": [9.8, 0.2],
    "num_bins": 768,
}
DEFAULT_HIST_TRAJ_TOKENIZER_CFG = {
    "type": "discrete",
    "action_space": {"type": "unicycle_accel_curvature", "n_waypoints": 16, "dt": 0.1},
    "dims_min": [-9.8, -0.2],
    "dims_max": [9.8, 0.2],
    "num_bins": 768,
}
DEFAULT_ACTION_SPACE_CFG = {"type": "unicycle_accel_curvature", "n_waypoints": 64, "dt": 0.1}
DEFAULT_DIFFUSION_CFG = {"type": "flow_matching", "train_timestep_sampler": "beta"}
DEFAULT_ACTION_IN_PROJ_CFG = {"type": "per_waypoint_v2"}
DEFAULT_ACTION_OUT_PROJ_CFG = {"type": "linear"}


def _mark_hf_initialized(module: Any) -> None:
    setattr(module, "_is_hf_initialized", True)
    for child in getattr(module, "children", lambda: [])():
        _mark_hf_initialized(child)


def _bool_from_env(name: str, default: bool) -> bool:
    return parse_bool(get_env(name), default=default)


def _is_resume_from_checkpoint() -> bool:
    for index, arg in enumerate(sys.argv):
        if arg == "--resume_from_checkpoint":
            return index + 1 < len(sys.argv)
        if arg.startswith("--resume_from_checkpoint="):
            return True
    return False


def replace_pad_token(input_ids: torch.Tensor, new_ids: torch.Tensor, pad_idx: int) -> torch.Tensor:
    mask = input_ids == pad_idx
    return input_ids.masked_scatter(mask, new_ids)


def _build_processor_with_tokens(
    vlm_name_or_path: str,
    traj_vocab_size: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Any:
    processor_kwargs = {}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(vlm_name_or_path, **processor_kwargs)
    tokenizer = processor.tokenizer
    discrete_tokens = [f"<i{index}>" for index in range(traj_vocab_size)]
    tokenizer.add_tokens(discrete_tokens)
    tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")
    tokenizer.add_tokens(list(SPECIAL_TOKENS.values()), special_tokens=True)
    tokenizer.traj_token_ids = {
        key: tokenizer.convert_tokens_to_ids(value) for key, value in TRAJ_TOKEN.items()
    }
    return processor


def tokenize_history_trajectory(
    tokenizer: Any,
    traj_data: Mapping[str, Any],
    start_idx: int = 0,
) -> torch.Tensor:
    batch_size = traj_data["ego_history_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)
    hist_idx = tokenizer.encode(
        hist_xyz=hist_xyz[:, :1],
        hist_rot=hist_rot[:, :1],
        fut_xyz=hist_xyz,
        fut_rot=hist_rot,
    ) + start_idx
    return einops.rearrange(hist_idx, "(b n_traj) n -> b (n_traj n)", b=batch_size)


def tokenize_future_trajectory(
    tokenizer: Any,
    traj_data: Mapping[str, Any],
    start_idx: int = 0,
) -> torch.Tensor:
    batch_size = traj_data["ego_future_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)
    fut_xyz = traj_data["ego_future_xyz"].flatten(start_dim=0, end_dim=1)
    fut_rot = traj_data["ego_future_rot"].flatten(start_dim=0, end_dim=1)
    fut_idx = tokenizer.encode(
        hist_xyz=hist_xyz,
        hist_rot=hist_rot,
        fut_xyz=fut_xyz,
        fut_rot=fut_rot,
    ) + start_idx
    return einops.rearrange(fut_idx, "(b n_traj) n -> b (n_traj n)", b=batch_size)


def _to_tensor(value: Any, *, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim == 4 and tensor.shape[-1] == 3 and tensor.shape[-2] == 3:
        tensor = tensor.unsqueeze(1)
    return tensor


def _prepare_traj_data(
    *,
    input_ids: torch.Tensor,
    ego_history_xyz: Any,
    ego_history_rot: Any,
    ego_future_xyz: Any = None,
    ego_future_rot: Any = None,
) -> dict[str, Optional[torch.Tensor]]:
    device = input_ids.device
    traj_data = {
        "ego_history_xyz": _to_tensor(ego_history_xyz, device=device, dtype=torch.float32),
        "ego_history_rot": _to_tensor(ego_history_rot, device=device, dtype=torch.float32),
        "ego_future_xyz": _to_tensor(ego_future_xyz, device=device, dtype=torch.float32),
        "ego_future_rot": _to_tensor(ego_future_rot, device=device, dtype=torch.float32),
    }
    if traj_data["ego_history_xyz"] is None or traj_data["ego_history_rot"] is None:
        raise ValueError(
            "Missing trajectory supervision fields in the batch. Use `alpamayo1_pai` and set "
            "`--remove_unused_columns false`."
        )
    return traj_data


class TrajectoryFusionMixin:
    def _validate_mixin_requirements(self, require_future: bool = False) -> dict[str, Any]:
        if getattr(self, "hist_traj_tokenizer", None) is None:
            raise AttributeError("Missing `hist_traj_tokenizer`.")
        if getattr(self, "hist_token_start_idx", None) is None:
            raise AttributeError("Missing `hist_token_start_idx`.")
        if getattr(self, "config", None) is None or not hasattr(self.config, "traj_token_ids"):
            raise AttributeError("Missing `config.traj_token_ids`.")
        attrs = {
            "hist_traj_tokenizer": self.hist_traj_tokenizer,
            "hist_token_start_idx": self.hist_token_start_idx,
            "config": self.config,
        }
        if require_future:
            if getattr(self, "traj_tokenizer", None) is None:
                raise AttributeError("Missing `traj_tokenizer`.")
            if getattr(self, "future_token_start_idx", None) is None:
                raise AttributeError("Missing `future_token_start_idx`.")
            attrs["traj_tokenizer"] = self.traj_tokenizer
            attrs["future_token_start_idx"] = self.future_token_start_idx
        return attrs

    def fuse_traj_tokens(
        self,
        input_ids: torch.Tensor,
        traj_data: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        if (
            traj_data is None
            or traj_data.get("ego_history_xyz") is None
            or traj_data.get("ego_history_rot") is None
        ):
            return input_ids
        attrs = self._validate_mixin_requirements(require_future=False)
        history_ids = tokenize_history_trajectory(
            attrs["hist_traj_tokenizer"],
            traj_data,
            attrs["hist_token_start_idx"],
        )
        return replace_pad_token(input_ids, history_ids, attrs["config"].traj_token_ids["history"])


class FutureTrajectoryFusionMixin(TrajectoryFusionMixin):
    def fuse_traj_tokens(
        self,
        input_ids: torch.Tensor,
        traj_data: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        input_ids = super().fuse_traj_tokens(input_ids=input_ids, traj_data=traj_data)
        if traj_data is None or traj_data.get("ego_future_xyz") is None:
            return input_ids
        attrs = self._validate_mixin_requirements(require_future=True)
        future_ids = tokenize_future_trajectory(
            attrs["traj_tokenizer"],
            traj_data,
            attrs["future_token_start_idx"],
        )
        return replace_pad_token(input_ids, future_ids, attrs["config"].traj_token_ids["future"])


class Alpamayo1Stage1Config(PretrainedConfig):
    model_type = ALPAMAYO1_STAGE1_MODEL_TYPE

    def __init__(
        self,
        vlm_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        vlm_backend: str = "qwenvl3",
        traj_tokenizer_cfg: Optional[dict[str, Any]] = None,
        hist_traj_tokenizer_cfg: Optional[dict[str, Any]] = None,
        traj_vocab_size: int = 768,
        tokens_per_history_traj: int = 16,
        tokens_per_future_traj: int = 64,
        model_dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vlm_name_or_path = vlm_name_or_path
        self.vlm_backend = vlm_backend
        self.traj_tokenizer_cfg = traj_tokenizer_cfg or DEFAULT_TRAJ_TOKENIZER_CFG
        self.hist_traj_tokenizer_cfg = hist_traj_tokenizer_cfg or DEFAULT_HIST_TRAJ_TOKENIZER_CFG
        self.traj_vocab_size = traj_vocab_size
        self.tokens_per_history_traj = tokens_per_history_traj
        self.tokens_per_future_traj = tokens_per_future_traj
        self.model_dtype = model_dtype
        self.attn_implementation = attn_implementation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.add_special_tokens = add_special_tokens
        self._initialize_vlm_vocab_state()

    def _initialize_vlm_vocab_state(self) -> None:
        processor = _build_processor_with_tokens(
            vlm_name_or_path=self.vlm_name_or_path,
            traj_vocab_size=self.traj_vocab_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        tokenizer = processor.tokenizer
        self.vocab_size = len(tokenizer)
        self.traj_token_start_idx = tokenizer.traj_token_start_idx
        self.traj_token_ids = tokenizer.traj_token_ids


class Alpamayo1Stage2Config(Alpamayo1Stage1Config):
    model_type = ALPAMAYO1_STAGE2_MODEL_TYPE

    def __init__(
        self,
        diffusion_cfg: Optional[dict[str, Any]] = None,
        action_space_cfg: Optional[dict[str, Any]] = None,
        action_in_proj_cfg: Optional[dict[str, Any]] = None,
        action_out_proj_cfg: Optional[dict[str, Any]] = None,
        expert_cfg: Optional[dict[str, Any]] = None,
        keep_same_dtype: bool = True,
        expert_non_causal_attention: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.diffusion_cfg = diffusion_cfg or DEFAULT_DIFFUSION_CFG
        self.action_space_cfg = action_space_cfg or DEFAULT_ACTION_SPACE_CFG
        self.action_in_proj_cfg = action_in_proj_cfg or DEFAULT_ACTION_IN_PROJ_CFG
        self.action_out_proj_cfg = action_out_proj_cfg or DEFAULT_ACTION_OUT_PROJ_CFG
        self.expert_cfg = expert_cfg or {}
        self.keep_same_dtype = keep_same_dtype
        self.expert_non_causal_attention = expert_non_causal_attention


class Alpamayo1BaseModel(PreTrainedModel, TrajectoryFusionMixin):
    config_class = Alpamayo1Stage1Config
    base_model_prefix = "vlm"

    def __init__(
        self,
        config: Alpamayo1Stage1Config,
        pretrained_modules: Optional[dict[str, torch.nn.Module]] = None,
        original_vocab_size: Optional[int] = None,
        print_param_count: bool = True,
    ) -> None:
        super().__init__(config)
        pretrained_modules = pretrained_modules or {}
        for module in pretrained_modules.values():
            if isinstance(module, torch.nn.Module):
                _mark_hf_initialized(module)

        self._initialize_vlm_backbone(config, pretrained_modules, original_vocab_size)
        self._initialize_trajectory_tokenizers(config, pretrained_modules)
        self.tokenizer = _build_processor_with_tokens(
            vlm_name_or_path=config.vlm_name_or_path,
            traj_vocab_size=config.traj_vocab_size,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        ).tokenizer
        self.special_token_ids = {
            key: self.tokenizer.convert_tokens_to_ids(value)
            for key, value in SPECIAL_TOKENS.items()
        }
        self._maybe_resize_vlm_embeddings(config)

        if print_param_count:
            total_params = sum(param.numel() for param in self.parameters())
            trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")

    def _initialize_vlm_backbone(
        self,
        config: Alpamayo1Stage1Config,
        pretrained_modules: dict[str, Any],
        original_vocab_size: Optional[int],
    ) -> None:
        if "vlm" in pretrained_modules:
            self.vlm = pretrained_modules["vlm"]
            self.original_vocab_size = original_vocab_size
            return

        vlm_config = Qwen3VLConfig.from_pretrained(config.vlm_name_or_path)
        self.original_vocab_size = getattr(vlm_config.text_config, "vocab_size", None)
        vlm_config.text_config.vocab_size = config.vocab_size
        vlm_config.vocab_size = config.vocab_size
        if hasattr(vlm_config, "torch_dtype"):
            vlm_config.torch_dtype = config.model_dtype
        if config.attn_implementation:
            setattr(vlm_config, "_attn_implementation", config.attn_implementation)
            setattr(vlm_config, "attn_implementation", config.attn_implementation)
        self.vlm = Qwen3VLForConditionalGeneration(vlm_config)

    def _maybe_resize_vlm_embeddings(self, config: Alpamayo1Stage1Config) -> None:
        current_vocab_size = getattr(getattr(self.vlm.config, "text_config", None), "vocab_size", None)
        if current_vocab_size is not None and current_vocab_size == config.vocab_size:
            return
        self.original_vocab_size = current_vocab_size
        self.vlm.resize_token_embeddings(config.vocab_size)
        self.vlm.config.vocab_size = config.vocab_size
        if hasattr(self.vlm.config, "text_config"):
            self.vlm.config.text_config.vocab_size = config.vocab_size

    def _initialize_trajectory_tokenizers(
        self,
        config: Alpamayo1Stage1Config,
        pretrained_modules: dict[str, Any],
    ) -> None:
        self.traj_tokenizer = pretrained_modules.get("traj_tokenizer")
        if self.traj_tokenizer is None:
            self.traj_tokenizer = build_trajectory_tokenizer(config.traj_tokenizer_cfg)

        self.hist_traj_tokenizer = pretrained_modules.get("hist_traj_tokenizer")
        if self.hist_traj_tokenizer is None:
            history_config = config.hist_traj_tokenizer_cfg or config.traj_tokenizer_cfg
            self.hist_traj_tokenizer = build_trajectory_tokenizer(history_config)

        self.future_token_start_idx = config.traj_token_start_idx
        self.hist_token_start_idx = config.traj_token_start_idx

    def tie_weights(self) -> None:
        if hasattr(self.vlm, "tie_weights"):
            self.vlm.tie_weights()

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.vlm.get_output_embeddings()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.vlm.language_model.embed_tokens

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if not hasattr(self.vlm, "gradient_checkpointing_enable"):
            raise ValueError(f"{self.vlm.__class__.__name__} does not support gradient checkpointing.")
        self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        if not hasattr(self.vlm, "gradient_checkpointing_disable"):
            raise ValueError(f"{self.vlm.__class__.__name__} does not support gradient checkpointing.")
        self.vlm.gradient_checkpointing_disable()


def load_stage1_vlm_weights(checkpoint_path: str, model: Any) -> Any:
    checkpoint_dir = Path(checkpoint_path)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    single_path = checkpoint_dir / "model.safetensors"
    vlm_state_dict = {}

    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as file_obj:
            weight_map = json.load(file_obj).get("weight_map", {})
        shard_to_keys = defaultdict(list)
        for key, shard_name in weight_map.items():
            if key.startswith("vlm."):
                shard_to_keys[shard_name].append(key)
        for shard_name, keys in shard_to_keys.items():
            shard_state = load_safetensors_file(str(checkpoint_dir / shard_name), device="cpu")
            for key in keys:
                if key in shard_state:
                    vlm_state_dict[key] = shard_state[key]
    elif single_path.exists():
        shard_state = load_safetensors_file(str(single_path), device="cpu")
        for key, value in shard_state.items():
            if key.startswith("vlm."):
                vlm_state_dict[key] = value

    if not vlm_state_dict:
        raise ValueError(f"No vlm.* tensors found in checkpoint: {checkpoint_dir}")

    load_result = model.load_state_dict(vlm_state_dict, strict=False, assign=True)
    logger.info(
        "Loaded %d VLM tensors from %s (missing=%d, unexpected=%d)",
        len(vlm_state_dict),
        checkpoint_dir,
        len(load_result.missing_keys),
        len(load_result.unexpected_keys),
    )
    return model


@dataclass
class Alpamayo1ModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class Alpamayo1Stage1Model(Alpamayo1BaseModel, FutureTrajectoryFusionMixin):
    config_class = Alpamayo1Stage1Config

    @torch._dynamo.disable
    def _compute_next_token_loss(
        self,
        outputs: ModelOutput,
        labels: torch.Tensor,
        labels_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        labels_mask = labels_mask if labels_mask is not None else torch.ones_like(labels, dtype=torch.bool)
        if labels_mask[:, 1:].sum() == 0:
            return torch.tensor(0.0, device=labels.device)
        shift_labels = labels[..., 1:][labels_mask[:, 1:]].contiguous()
        shift_logits = outputs.logits[..., :-1, :].clone()[labels_mask[:, 1:]].contiguous().float()
        shift_labels = shift_labels.to(shift_logits.device)
        if token_mask is not None:
            shift_logits[..., ~token_mask] = torch.finfo(shift_logits.dtype).min
        return torch.nan_to_num(
            torch.nn.functional.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=IGNORE_INDEX,
                reduction="mean",
            ),
            nan=0.0,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        ego_history_xyz: Any = None,
        ego_history_rot: Any = None,
        ego_future_xyz: Any = None,
        ego_future_rot: Any = None,
        **kwargs: Any,
    ) -> Alpamayo1ModelOutput:
        traj_data = _prepare_traj_data(
            input_ids=input_ids,
            ego_history_xyz=ego_history_xyz,
            ego_history_rot=ego_history_rot,
            ego_future_xyz=ego_future_xyz,
            ego_future_rot=ego_future_rot,
        )
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)
        if labels is not None:
            labels = self.fuse_traj_tokens(labels, traj_data)

        outputs = self.vlm(input_ids=input_ids, labels=labels, **kwargs)
        if labels is None:
            return Alpamayo1ModelOutput(loss=getattr(outputs, "loss", None), logits=outputs.logits)

        traj_mask = (
            ((labels >= self.future_token_start_idx) & (labels < self.future_token_start_idx + self.config.traj_vocab_size))
            | (labels == self.special_token_ids["traj_future_start"])
            | (labels == self.special_token_ids["traj_future_end"])
        )
        future_loss = self._compute_next_token_loss(outputs, labels, traj_mask)
        labels = labels.clone()
        labels[traj_mask] = IGNORE_INDEX
        other_loss = self._compute_next_token_loss(outputs, labels, labels != IGNORE_INDEX)
        outputs.loss = future_loss + other_loss
        return Alpamayo1ModelOutput(loss=outputs.loss, logits=outputs.logits)


class Alpamayo1Stage2Model(Alpamayo1BaseModel):
    config_class = Alpamayo1Stage2Config

    def __init__(
        self,
        config: Alpamayo1Stage2Config,
        pretrained_modules: Optional[dict[str, torch.nn.Module]] = None,
        original_vocab_size: Optional[int] = None,
        cotrain_vlm: bool = False,
        stop_grad_from_vlm: bool = True,
    ) -> None:
        super().__init__(
            config=config,
            pretrained_modules=pretrained_modules,
            original_vocab_size=original_vocab_size,
            print_param_count=False,
        )
        self.cotrain_vlm = cotrain_vlm
        self.stop_grad_from_vlm = stop_grad_from_vlm

        expert_config = self.vlm.config.text_config.to_dict()
        for key, value in (config.expert_cfg or {}).items():
            expert_config[key] = value
        self.expert = AutoModel.from_config(self.vlm.config.text_config.__class__(**expert_config))
        if hasattr(self.expert, "embed_tokens"):
            del self.expert.embed_tokens

        self.action_space = build_action_space(config.action_space_cfg)
        self.diffusion = build_diffusion(config.diffusion_cfg, self.action_space.get_action_space_dims())
        self.action_in_proj = build_action_input_projector(
            config.action_in_proj_cfg,
            in_dims=self.action_space.get_action_space_dims(),
            out_dim=self.expert.config.hidden_size,
        )
        self.action_out_proj = build_action_output_projector(
            config.action_out_proj_cfg,
            in_features=self.expert.config.hidden_size,
            out_features=self.action_space.get_action_space_dims()[-1],
        )

        expert_dtype = self.expert.dtype
        if config.keep_same_dtype:
            self.diffusion = self.diffusion.to(dtype=expert_dtype)
            self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
            self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)

        if not self.cotrain_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False

        logger.info("Model parameter count:")
        for key, value in get_param_count(self).items():
            logger.info(f"{key}: {value:,}")

    def _prepare_future_training_data(self, traj_data: dict[str, Any]) -> dict[str, Any]:
        action = self.action_space.traj_to_action(
            traj_history_xyz=traj_data["ego_history_xyz"],
            traj_history_rot=traj_data["ego_history_rot"],
            traj_future_xyz=traj_data["ego_future_xyz"],
            traj_future_rot=traj_data["ego_future_rot"],
        )
        action = action.reshape(-1, *self.action_space.get_action_space_dims())
        return self.diffusion.construct_training_data(action)

    def _build_position_ids(
        self,
        vlm_outputs: Any,
        batch_size: int,
        num_expert_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        position_ids = torch.arange(num_expert_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=batch_size).clone()
        delta = vlm_outputs.rope_deltas + vlm_outputs.past_key_values.get_seq_length()
        position_ids += delta.to(position_ids.device)
        return position_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        ego_history_xyz: Any = None,
        ego_history_rot: Any = None,
        ego_future_xyz: Any = None,
        ego_future_rot: Any = None,
        **kwargs: Any,
    ) -> Alpamayo1ModelOutput:
        traj_data = _prepare_traj_data(
            input_ids=input_ids,
            ego_history_xyz=ego_history_xyz,
            ego_history_rot=ego_history_rot,
            ego_future_xyz=ego_future_xyz,
            ego_future_rot=ego_future_rot,
        )
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)
        if labels is not None:
            labels = self.fuse_traj_tokens(labels, traj_data)

        context = torch.no_grad() if not self.cotrain_vlm else torch.enable_grad()
        with context:
            vlm_outputs = self.vlm(
                input_ids=input_ids,
                labels=labels,
                use_cache=True,
                **kwargs,
            )

        future_start_token_id = self.config.traj_token_ids["future_start"]
        last_traj_future_start_idx = (input_ids == future_start_token_id).nonzero(as_tuple=False)
        last_traj_future_start_idx = last_traj_future_start_idx[-1, 1] + 1

        batch_size = input_ids.shape[0]
        future_traj_data = self._prepare_future_training_data(traj_data)
        action_embeds = self.action_in_proj(future_traj_data["noisy_x"], future_traj_data["timesteps"])
        kv_cache = vlm_outputs.past_key_values
        kv_cache.crop(last_traj_future_start_idx)
        if self.stop_grad_from_vlm:
            for layer in kv_cache.layers:
                layer.keys = layer.keys.detach()
                layer.values = layer.values.detach()

        position_ids = self._build_position_ids(
            vlm_outputs,
            batch_size,
            action_embeds.shape[1],
            action_embeds.device,
        )
        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False
        expert_outputs = self.expert(
            inputs_embeds=action_embeds,
            position_ids=position_ids,
            past_key_values=kv_cache,
            attention_mask=None,
            use_cache=True,
            **forward_kwargs,
        )
        diffusion_out = expert_outputs.last_hidden_state[:, -action_embeds.shape[1] :]
        pred = self.action_out_proj(diffusion_out)
        pred = pred.view(-1, *self.action_space.get_action_space_dims())
        future_loss = self.diffusion.compute_loss_from_pred(training_data=future_traj_data, pred=pred)
        loss = future_loss
        if self.cotrain_vlm:
            loss = loss + vlm_outputs.loss
        return Alpamayo1ModelOutput(loss=loss)


def _is_custom_checkpoint(model_dir: str) -> bool:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return any(
        key in config
        for key in (
            "traj_tokenizer_cfg",
            "hist_traj_tokenizer_cfg",
            "action_space_cfg",
            "diffusion_cfg",
            "model_type",
        )
    )


def _resolve_vlm_name_or_path(model_dir: str, config: Optional[PretrainedConfig] = None) -> str:
    return get_env("VLM_NAME_OR_PATH") or getattr(config, "vlm_name_or_path", None) or model_dir


def _resolve_torch_dtype(model_info) -> Optional[torch.dtype]:
    return getattr(model_info, "torch_dtype", None)


def _build_fresh_stage1_config(model_dir: str) -> Alpamayo1Stage1Config:
    config = Alpamayo1Stage1Config(
        vlm_name_or_path=_resolve_vlm_name_or_path(model_dir),
        min_pixels=int(get_env("MIN_PIXELS", "0")) or None,
        max_pixels=int(get_env("MAX_PIXELS", "0")) or None,
    )
    return config


def _build_fresh_stage2_config(model_dir: str) -> Alpamayo1Stage2Config:
    return Alpamayo1Stage2Config(
        vlm_name_or_path=_resolve_vlm_name_or_path(model_dir),
        min_pixels=int(get_env("MIN_PIXELS", "0")) or None,
        max_pixels=int(get_env("MAX_PIXELS", "0")) or None,
    )


def _load_pretrained_vlm(
    model_dir: str,
    config: PretrainedConfig,
    model_info,
    model_kwargs: dict[str, Any],
) -> tuple[torch.nn.Module, Optional[int]]:
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=_resolve_torch_dtype(model_info),
        attn_implementation=getattr(config, "attn_implementation", None),
        **model_kwargs,
    )
    original_vocab_size = getattr(getattr(vlm.config, "text_config", None), "vocab_size", None)
    return vlm, original_vocab_size


def _configure_runtime_env(config: Alpamayo1Stage1Config) -> None:
    os.environ["TOKENS_PER_HISTORY_TRAJ"] = str(config.tokens_per_history_traj)
    os.environ["TOKENS_PER_FUTURE_TRAJ"] = str(config.tokens_per_future_traj)
    if getattr(config, "min_pixels", None) is not None:
        os.environ["MIN_PIXELS"] = str(config.min_pixels)
    if getattr(config, "max_pixels", None) is not None:
        os.environ["MAX_PIXELS"] = str(config.max_pixels)


def _load_stage1_model(
    model_dir: str,
    model_info,
    model_kwargs: dict[str, Any],
    load_model: bool,
) -> tuple[Optional[Alpamayo1Stage1Model], Any]:
    config = Alpamayo1Stage1Config.from_pretrained(model_dir) if _is_custom_checkpoint(model_dir) else _build_fresh_stage1_config(model_dir)
    config.vlm_name_or_path = _resolve_vlm_name_or_path(model_dir, config)
    config._initialize_vlm_vocab_state()
    _configure_runtime_env(config)
    processor = _build_processor_with_tokens(
        vlm_name_or_path=config.vlm_name_or_path,
        traj_vocab_size=config.traj_vocab_size,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    if not load_model:
        return None, processor

    model_kwargs = dict(model_kwargs)
    if _is_custom_checkpoint(model_dir):
        model = Alpamayo1Stage1Model.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=_resolve_torch_dtype(model_info),
            **model_kwargs,
        )
    else:
        vlm, original_vocab_size = _load_pretrained_vlm(model_dir, config, model_info, model_kwargs)
        model = Alpamayo1Stage1Model(
            config=config,
            pretrained_modules={"vlm": vlm},
            original_vocab_size=original_vocab_size,
        )
    return model, processor


def _load_stage2_model(
    model_dir: str,
    model_info,
    model_kwargs: dict[str, Any],
    load_model: bool,
) -> tuple[Optional[Alpamayo1Stage2Model], Any]:
    config = Alpamayo1Stage2Config.from_pretrained(model_dir) if _is_custom_checkpoint(model_dir) else _build_fresh_stage2_config(model_dir)
    config.vlm_name_or_path = _resolve_vlm_name_or_path(model_dir, config)
    config._initialize_vlm_vocab_state()
    _configure_runtime_env(config)
    processor = _build_processor_with_tokens(
        vlm_name_or_path=config.vlm_name_or_path,
        traj_vocab_size=config.traj_vocab_size,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    if not load_model:
        return None, processor

    stage1_vlm_checkpoint_path = get_env("STAGE1_VLM_CHECKPOINT_PATH")
    if not _is_resume_from_checkpoint() and not stage1_vlm_checkpoint_path:
        raise ValueError(
            "Stage 2 fresh training requires `stage1_vlm_checkpoint_path`. Pass it with "
            "--model_kwargs '{\"stage1_vlm_checkpoint_path\": \"/path/to/stage1_ckpt\"}'."
        )

    model_kwargs = dict(model_kwargs)
    if _is_custom_checkpoint(model_dir):
        model = Alpamayo1Stage2Model.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=_resolve_torch_dtype(model_info),
            cotrain_vlm=_bool_from_env("COTRAIN_VLM", False),
            stop_grad_from_vlm=_bool_from_env("STOP_GRAD_FROM_VLM", True),
            **model_kwargs,
        )
    else:
        vlm, original_vocab_size = _load_pretrained_vlm(model_dir, config, model_info, model_kwargs)
        model = Alpamayo1Stage2Model(
            config=config,
            pretrained_modules={"vlm": vlm},
            original_vocab_size=original_vocab_size,
            cotrain_vlm=_bool_from_env("COTRAIN_VLM", False),
            stop_grad_from_vlm=_bool_from_env("STOP_GRAD_FROM_VLM", True),
        )

    if stage1_vlm_checkpoint_path and not _is_resume_from_checkpoint():
        load_stage1_vlm_weights(stage1_vlm_checkpoint_path, model)
    return model, processor


def get_model_tokenizer(
    stage: str,
    model_dir: str,
    model_info,
    model_kwargs: dict[str, Any],
    load_model: bool = True,
    **kwargs: Any,
) -> tuple[Optional[PreTrainedModel], Any]:
    del kwargs
    if stage == "stage1":
        return _load_stage1_model(model_dir, model_info, model_kwargs, load_model)
    if stage == "stage2":
        return _load_stage2_model(model_dir, model_info, model_kwargs, load_model)
    raise ValueError(f"Unsupported Alpamayo1 stage: {stage}")


__all__ = [
    "ALPAMAYO1_STAGE1_MODEL_TYPE",
    "ALPAMAYO1_STAGE2_MODEL_TYPE",
    "Alpamayo1Stage1Config",
    "Alpamayo1Stage1Model",
    "Alpamayo1Stage2Config",
    "Alpamayo1Stage2Model",
    "get_model_tokenizer",
    "load_stage1_vlm_weights",
]
