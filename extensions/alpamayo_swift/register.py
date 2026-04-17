from __future__ import annotations

from swift.llm import ModelMeta, TemplateMeta, register_model, register_template
from swift.llm.template.base import Template

ALPAMAYO_STAGE1_MODEL_TYPE = "alpamayo_stage1"
ALPAMAYO_STAGE2_MODEL_TYPE = "alpamayo_stage2"
ALPAMAYO_TEMPLATE = "alpamayo_passthrough"


class AlpamayoPassthroughTemplate(Template):
    support_padding_free = False
    use_model = False


def _unsupported_get_model_tokenizer(*args, **kwargs):
    raise RuntimeError(
        "Alpamayo SFT is loaded through tools/alpamayo_sft.py and the runtime-patched SwiftSft."
    )


def register_alpamayo_components() -> None:
    register_template(
        TemplateMeta(
            template_type=ALPAMAYO_TEMPLATE,
            prefix=[],
            prompt=["{{QUERY}}"],
            chat_sep=None,
            template_cls=AlpamayoPassthroughTemplate,
        ),
        exist_ok=True,
    )

    for model_type in (ALPAMAYO_STAGE1_MODEL_TYPE, ALPAMAYO_STAGE2_MODEL_TYPE):
        register_model(
            ModelMeta(
                model_type=model_type,
                model_groups=[],
                template=ALPAMAYO_TEMPLATE,
                get_function=_unsupported_get_model_tokenizer,
                is_multimodal=True,
                task_type="causal_lm",
            ),
            exist_ok=True,
        )


__all__ = [
    "ALPAMAYO_STAGE1_MODEL_TYPE",
    "ALPAMAYO_STAGE2_MODEL_TYPE",
    "ALPAMAYO_TEMPLATE",
    "register_alpamayo_components",
]
