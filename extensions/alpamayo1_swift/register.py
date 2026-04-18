from __future__ import annotations

from swift.llm import DatasetMeta, ModelMeta, register_dataset, register_model

from .runtime.data import ALPAMAYO1_DATASET_NAME, install_alpamayo1_image_loader, load_alpamayo1_pai_dataset
from .runtime.modeling import (
    ALPAMAYO1_STAGE1_MODEL_TYPE,
    ALPAMAYO1_STAGE2_MODEL_TYPE,
    get_model_tokenizer,
)

_REGISTERED = False


def register_all() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    install_alpamayo1_image_loader()

    register_dataset(
        DatasetMeta(
            dataset_name=ALPAMAYO1_DATASET_NAME,
            subsets=["train", "val"],
            load_function=load_alpamayo1_pai_dataset,
            help="Use syntax like alpamayo1_pai:train@0-99 or alpamayo1_pai:val@99-100.",
            tags=["chat", "multi-modal"],
        ),
        exist_ok=True,
    )

    register_model(
        ModelMeta(
            model_type=ALPAMAYO1_STAGE1_MODEL_TYPE,
            model_groups=[],
            template="qwen3_vl",
            get_function=lambda *args, **kwargs: get_model_tokenizer("stage1", *args, **kwargs),
            is_multimodal=True,
            task_type="causal_lm",
        ),
        exist_ok=True,
    )
    register_model(
        ModelMeta(
            model_type=ALPAMAYO1_STAGE2_MODEL_TYPE,
            model_groups=[],
            template="qwen3_vl",
            get_function=lambda *args, **kwargs: get_model_tokenizer("stage2", *args, **kwargs),
            is_multimodal=True,
            task_type="causal_lm",
        ),
        exist_ok=True,
    )

    _REGISTERED = True


register_all()

