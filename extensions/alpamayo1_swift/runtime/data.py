# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import zipfile
from functools import lru_cache
from io import BytesIO
from typing import Any, Iterable, Optional, Sequence

from .support import (
    CAMERA_INDICES_TO_DISPLAY_NAMES,
    CAMERA_NAMES_TO_INDICES,
    DEFAULT_PAI_CHUNK_BY_SPLIT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    SPECIAL_TOKENS,
    get_env,
    parse_bool,
    require_env,
)

ALPAMAYO1_DATASET_NAME = "alpamayo1_pai"
ALPAMAYO1_IMAGE_SCHEME = "alpamayo1://"
DEFAULT_T0_US = 5_100_000
DEFAULT_TIME_STEP_SECONDS = 0.1
DEFAULT_NUM_HISTORY_STEPS = 16
DEFAULT_NUM_FUTURE_STEPS = 64
DEFAULT_NUM_IMAGE_FRAMES = 4

DEFAULT_CAMERA_NAMES = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]

_IMAGE_LOADER_INSTALLED = False


def parse_chunk_spec(spec: Optional[str]) -> Optional[list[int]]:
    if spec is None:
        return None
    spec = str(spec).strip()
    if not spec or spec.lower() == "all":
        return None
    if spec.startswith("["):
        values = json.loads(spec)
        return [int(value) for value in values]

    chunk_ids: list[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            chunk_ids.extend(range(int(start), int(end)))
        else:
            chunk_ids.append(int(part))
    return chunk_ids


def parse_subset_spec(subset: Optional[str]) -> tuple[str, str]:
    value = subset or "train"
    split, chunk_spec = value, None
    if "@" in value:
        split, chunk_spec = value.split("@", 1)
    split = split.strip() or "train"
    if chunk_spec is None or chunk_spec.strip() == "":
        chunk_spec = DEFAULT_PAI_CHUNK_BY_SPLIT.get(split, DEFAULT_PAI_CHUNK_BY_SPLIT["train"])
    return split, chunk_spec


def _payload_to_uri(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    token = base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")
    return f"{ALPAMAYO1_IMAGE_SCHEME}{token}"


def parse_image_uri(uri: str) -> dict[str, Any]:
    if not uri.startswith(ALPAMAYO1_IMAGE_SCHEME):
        raise ValueError(f"Unsupported image uri: {uri}")
    token = uri[len(ALPAMAYO1_IMAGE_SCHEME) :]
    token += "=" * (-len(token) % 4)
    return json.loads(base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8"))


def _build_component_text(
    start_token: str,
    end_token: str,
    *,
    padding: Optional[str] = None,
    content: Optional[str] = None,
) -> str:
    if (padding is None) == (content is None):
        raise ValueError("Exactly one of `padding` or `content` must be provided.")
    value = content if content is not None else padding
    return f"{start_token}{value}{end_token}"


def _select_camera_features(avdi: Any) -> list[str]:
    camera_namespace = getattr(getattr(avdi, "features", None), "CAMERA", None)
    if camera_namespace is None:
        return list(DEFAULT_CAMERA_NAMES)
    return [
        camera_namespace.CAMERA_CROSS_LEFT_120FOV,
        camera_namespace.CAMERA_FRONT_WIDE_120FOV,
        camera_namespace.CAMERA_CROSS_RIGHT_120FOV,
        camera_namespace.CAMERA_FRONT_TELE_30FOV,
    ]


def _select_label_egomotion(avdi: Any) -> Any:
    label_namespace = getattr(getattr(avdi, "features", None), "LABELS", None)
    if label_namespace is None:
        return "egomotion"
    return label_namespace.EGOMOTION


def _camera_name(camera_feature: Any) -> str:
    return str(camera_feature).split("/")[-1].lower()


def _sorted_camera_specs(camera_features: Sequence[Any]) -> list[tuple[Any, str, int]]:
    specs = []
    for feature in camera_features:
        name = _camera_name(feature)
        specs.append((feature, name, CAMERA_NAMES_TO_INDICES.get(name, 0)))
    return sorted(specs, key=lambda item: item[2])


def _build_messages_and_images(
    *,
    camera_specs: Sequence[tuple[Any, str, int]],
    image_timestamps: Any,
    clip_id: str,
    num_history_tokens: int,
    num_future_tokens: int,
    include_camera_ids: bool,
    include_frame_nums: bool,
) -> tuple[list[dict[str, str]], list[str]]:
    images: list[str] = []
    user_parts = [DEFAULT_USER_PROMPT]

    for feature, _, camera_id in camera_specs:
        if include_camera_ids:
            user_parts.append(f"\n{CAMERA_INDICES_TO_DISPLAY_NAMES[camera_id]}: ")
        else:
            user_parts.append("\n")

        for frame_index, timestamp_us in enumerate(image_timestamps.tolist()):
            if include_frame_nums:
                user_parts.append(f"frame {frame_index} ")
            user_parts.append("<image>\n")
            images.append(
                _payload_to_uri(
                    {
                        "clip_id": clip_id,
                        "camera_feature": str(feature),
                        "timestamp_us": int(timestamp_us),
                    }
                )
            )

    user_parts.append(
        _build_component_text(
            SPECIAL_TOKENS["traj_history_start"],
            SPECIAL_TOKENS["traj_history_end"],
            padding=SPECIAL_TOKENS["traj_history"] * num_history_tokens,
        )
    )
    assistant_text = _build_component_text(
        SPECIAL_TOKENS["traj_future_start"],
        SPECIAL_TOKENS["traj_future_end"],
        padding=SPECIAL_TOKENS["traj_future"] * num_future_tokens,
    )
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "".join(user_parts)},
        {"role": "assistant", "content": assistant_text},
    ]
    return messages, images


class PhysicalAIAVDatasetLocalInterface:
    def __init__(
        self,
        local_dir: str | pathlib.Path,
        chunk_ids: Optional[Sequence[int]] = None,
        features_metadata: str = "features.csv",
        clip_index_metadata: str = "clip_index.parquet",
        start_safe_margin_seconds: float = 1.6,
        end_safe_margin_seconds: float = 6.4,
    ) -> None:
        import pandas as pd
        from physical_ai_av.dataset import Features

        self.local_dir = str(local_dir)
        self.chunk_ids = list(chunk_ids) if chunk_ids is not None else None
        self.start_safe_margin_seconds = start_safe_margin_seconds
        self.end_safe_margin_seconds = end_safe_margin_seconds

        features_df = pd.read_csv(os.path.join(self.local_dir, features_metadata), index_col="feature")
        features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(json.loads, na_action="ignore")
        self.features = Features(features_df)
        self.clip_index = pd.read_parquet(os.path.join(self.local_dir, clip_index_metadata))
        self._filter_clips_by_event_t0s()
        self.sensor_presence = pd.read_parquet(os.path.join(self.local_dir, "metadata/feature_presence.parquet"))

    def _filter_clips_by_event_t0s(self) -> None:
        import numpy as np

        if "event_t0s" not in self.clip_index.columns:
            return
        start_margin_us = int(self.start_safe_margin_seconds * 1_000_000)
        end_margin_us = int(self.end_safe_margin_seconds * 1_000_000)
        has_end = "end_timestamp" in self.clip_index.columns

        def filter_events(row):
            event_t0s = row["event_t0s"]
            if event_t0s is None or len(event_t0s) == 0:
                return np.array([], dtype=np.int64)
            values = np.asarray(event_t0s, dtype=np.int64)
            mask = values >= start_margin_us
            if has_end:
                mask &= (values + end_margin_us) <= int(row["end_timestamp"])
            return values[mask]

        self.clip_index["event_t0s"] = self.clip_index.apply(filter_events, axis=1)
        self.clip_index = self.clip_index.loc[
            self.clip_index["event_t0s"].apply(lambda value: value is not None and len(value) > 0)
        ]

    def get_all_clip_ids(self) -> list[str]:
        if self.chunk_ids is None:
            return self.clip_index.index.tolist()
        return self.clip_index.loc[self.clip_index["chunk"].isin(self.chunk_ids)].index.tolist()

    def get_clip_chunk(self, clip_id: str) -> int:
        return int(self.clip_index.at[clip_id, "chunk"])

    def get_clip_key_frame(self, clip_id: str, sample_index_in_clip: int = 0):
        import numpy as np

        timestamp = self.clip_index.at[clip_id, "event_t0s"][sample_index_in_clip]
        return np.asarray(timestamp, dtype=np.int64)

    def get_clip_feature(self, clip_id: str, feature: str, maybe_stream: bool = False) -> Any:
        import pandas as pd
        from physical_ai_av import egomotion, video

        if feature not in self.features.features_df.index:
            return None
        chunk_filename = self.features.get_chunk_feature_filename(self.get_clip_chunk(clip_id), feature)
        chunk_filename = os.path.join(self.local_dir, chunk_filename)
        with open(chunk_filename, "rb") as file_obj:
            if chunk_filename.endswith(".parquet"):
                return pd.read_parquet(file_obj).loc[clip_id]
            if not chunk_filename.endswith(".zip"):
                raise ValueError(f"Unexpected feature path: {chunk_filename}")
            clip_files_in_zip = self.features.get_clip_files_in_zip(clip_id, feature)
            with zipfile.ZipFile(file_obj, "r") as zip_file:
                if feature == "egomotion":
                    egomotion_df = pd.read_parquet(io.BytesIO(zip_file.read(clip_files_in_zip["egomotion"])))
                    return egomotion.EgomotionState.from_egomotion_df(egomotion_df).create_interpolator(
                        egomotion_df["timestamp"].to_numpy()
                    )
                if feature.startswith("camera"):
                    return video.SeekVideoReader(
                        video_data=io.BytesIO(zip_file.read(clip_files_in_zip["video"])),
                        timestamps=pd.read_parquet(
                            io.BytesIO(zip_file.read(clip_files_in_zip["frame_timestamps"]))
                        )["timestamp"].to_numpy(),
                    )
                return {
                    key: (
                        pd.read_parquet(io.BytesIO(zip_file.read(value)))
                        if value.endswith(".parquet")
                        else io.BytesIO(zip_file.read(value))
                    )
                    for key, value in clip_files_in_zip.items()
                }


@lru_cache(maxsize=8)
def _get_local_interface(local_dir: str) -> PhysicalAIAVDatasetLocalInterface:
    return PhysicalAIAVDatasetLocalInterface(local_dir=local_dir)


@lru_cache(maxsize=2048)
def _get_cached_feature(local_dir: str, clip_id: str, feature: str) -> Any:
    avdi = _get_local_interface(local_dir)
    return avdi.get_clip_feature(clip_id, feature)


def _build_row(
    *,
    local_dir: str,
    avdi: PhysicalAIAVDatasetLocalInterface,
    clip_id: str,
    t0_us: int,
    include_camera_ids: bool,
    include_frame_nums: bool,
    num_history_steps: int,
    num_future_steps: int,
    time_step: float,
    num_image_frames: int,
    num_history_tokens: int,
    num_future_tokens: int,
) -> dict[str, Any]:
    import numpy as np
    import scipy.spatial.transform as spt

    egomotion = _get_cached_feature(local_dir, clip_id, str(_select_label_egomotion(avdi)))

    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)

    history_timestamps = t0_us + history_offsets_us
    future_timestamps = t0_us + future_offsets_us

    ego_history = egomotion(history_timestamps)
    ego_future = egomotion(future_timestamps)
    ego_history_xyz = ego_history.pose.translation
    ego_history_quat = ego_history.pose.rotation.as_quat()
    ego_future_xyz = ego_future.pose.translation
    ego_future_quat = ego_future.pose.rotation.as_quat()

    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()

    image_timestamps = np.array(
        [t0_us - (num_image_frames - 1 - index) * int(time_step * 1_000_000) for index in range(num_image_frames)],
        dtype=np.int64,
    )
    camera_specs = _sorted_camera_specs(_select_camera_features(avdi))
    messages, images = _build_messages_and_images(
        camera_specs=camera_specs,
        image_timestamps=image_timestamps,
        clip_id=clip_id,
        num_history_tokens=num_history_tokens,
        num_future_tokens=num_future_tokens,
        include_camera_ids=include_camera_ids,
        include_frame_nums=include_frame_nums,
    )
    return {
        "messages": messages,
        "images": images,
        "ego_history_xyz": [ego_history_xyz_local.astype(np.float32).tolist()],
        "ego_history_rot": [ego_history_rot_local.astype(np.float32).tolist()],
        "ego_future_xyz": [ego_future_xyz_local.astype(np.float32).tolist()],
        "ego_future_rot": [ego_future_rot_local.astype(np.float32).tolist()],
        "clip_id": clip_id,
        "t0_us": int(t0_us),
    }


def _resolve_runtime_options() -> dict[str, Any]:
    local_dir = require_env("PAI_LOCAL_DIR")
    return {
        "local_dir": os.path.abspath(os.path.expanduser(local_dir)),
        "use_default_keyframe": parse_bool(get_env("USE_DEFAULT_KEYFRAME"), default=True),
        "include_camera_ids": parse_bool(get_env("INCLUDE_CAMERA_IDS"), default=False),
        "include_frame_nums": parse_bool(get_env("INCLUDE_FRAME_NUMS"), default=False),
        "num_history_steps": int(get_env("NUM_HISTORY_STEPS", str(DEFAULT_NUM_HISTORY_STEPS))),
        "num_future_steps": int(get_env("NUM_FUTURE_STEPS", str(DEFAULT_NUM_FUTURE_STEPS))),
        "time_step": float(get_env("TIME_STEP", str(DEFAULT_TIME_STEP_SECONDS))),
        "num_image_frames": int(get_env("NUM_IMAGE_FRAMES", str(DEFAULT_NUM_IMAGE_FRAMES))),
        "num_history_tokens": int(get_env("TOKENS_PER_HISTORY_TRAJ", "16")),
        "num_future_tokens": int(get_env("TOKENS_PER_FUTURE_TRAJ", "64")),
    }


def _load_image_from_uri(uri: str) -> BytesIO:
    import numpy as np
    from PIL import Image

    payload = parse_image_uri(uri)
    local_dir = os.path.abspath(os.path.expanduser(require_env("PAI_LOCAL_DIR")))
    camera = _get_cached_feature(local_dir, payload["clip_id"], payload["camera_feature"])
    frames, _ = camera.decode_images_from_timestamps(np.array([payload["timestamp_us"]], dtype=np.int64))
    image = Image.fromarray(frames[0].astype(np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def install_alpamayo1_image_loader() -> None:
    global _IMAGE_LOADER_INSTALLED
    if _IMAGE_LOADER_INSTALLED:
        return

    from swift.llm.template import vision_utils

    original_load_file = vision_utils.load_file

    def load_file(path):
        if isinstance(path, str) and path.startswith(ALPAMAYO1_IMAGE_SCHEME):
            return _load_image_from_uri(path)
        return original_load_file(path)

    vision_utils.load_file = load_file
    _IMAGE_LOADER_INSTALLED = True


def load_alpamayo1_pai_dataset(
    dataset_syntax,
    dataset_meta=None,
    **kwargs: Any,
):
    from datasets import Dataset as HfDataset

    del dataset_meta, kwargs
    options = _resolve_runtime_options()
    split = "train"
    chunk_spec = DEFAULT_PAI_CHUNK_BY_SPLIT["train"]
    if dataset_syntax is not None and dataset_syntax.subsets:
        split, chunk_spec = parse_subset_spec(dataset_syntax.subsets[0])

    chunk_ids = parse_chunk_spec(chunk_spec)
    avdi = _get_local_interface(options["local_dir"])
    if chunk_ids is not None:
        avdi = PhysicalAIAVDatasetLocalInterface(local_dir=options["local_dir"], chunk_ids=chunk_ids)

    rows = []
    for clip_id in avdi.get_all_clip_ids():
        t0_us = DEFAULT_T0_US if options["use_default_keyframe"] else int(avdi.get_clip_key_frame(clip_id))
        rows.append(
            _build_row(
                local_dir=options["local_dir"],
                avdi=avdi,
                clip_id=clip_id,
                t0_us=t0_us,
                include_camera_ids=options["include_camera_ids"],
                include_frame_nums=options["include_frame_nums"],
                num_history_steps=options["num_history_steps"],
                num_future_steps=options["num_future_steps"],
                time_step=options["time_step"],
                num_image_frames=options["num_image_frames"],
                num_history_tokens=options["num_history_tokens"],
                num_future_tokens=options["num_future_tokens"],
            )
        )
    return HfDataset.from_list(rows)


__all__ = [
    "ALPAMAYO1_DATASET_NAME",
    "ALPAMAYO1_IMAGE_SCHEME",
    "PhysicalAIAVDatasetLocalInterface",
    "install_alpamayo1_image_loader",
    "load_alpamayo1_pai_dataset",
    "parse_chunk_spec",
    "parse_image_uri",
    "parse_subset_spec",
]
