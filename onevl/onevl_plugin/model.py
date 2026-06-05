# Copyright (c) ModelScope Contributors. All rights reserved.
"""Non-invasive registration of the OneVL ``qwen3_vl_latent_cot`` model.

This module reproduces ``Qwen3VLLatentCoTLoader`` from the original OneVL
training fork *without* modifying any ms-swift source file.  It is imported via
``--external_plugins`` so that ``register_model`` runs at argument-parsing time
and the ``qwen3_vl_latent_cot`` model_type becomes available to ``swift sft`` /
``swift infer``.

The loader wraps the stock ``Qwen3VLLoader`` (so the full Qwen3-VL loading path
is preserved) and then patches the model in-place for latent CoT training,
reading every hyper-parameter from ``LATENT_COT_*`` environment variables -- the
exact same contract used by the original OneVL run scripts.
"""

from transformers import PreTrainedModel

from swift.model import Model, ModelArch, ModelGroup, ModelMeta, register_model
from swift.model.models.qwen import Qwen3VLLoader
from swift.utils import get_env_args, get_logger

logger = get_logger()

# The model_type / template string. We use the plain string instead of adding a
# constant to ``swift/model/constant.py`` (that would be an invasive change).
LATENT_COT_MODEL_TYPE = 'qwen3_vl_latent_cot'
LATENT_COT_TEMPLATE_TYPE = 'qwen3_vl_latent_cot'


class Qwen3VLLatentCoTLoader(Qwen3VLLoader):
    """Loader that wraps Qwen3-VL with latent CoT training support.

    Latent CoT configuration is read from ``LATENT_COT_*`` environment variables.
    """

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, config, processor, model_kwargs)

        from .latent_cot import LatentCoTConfig, patch_model_for_latent_cot, load_latent_cot_weights
        aux_path = get_env_args('LATENT_COT_AUX_MODEL_PATH', str, None)
        vis_aux_path = get_env_args('LATENT_COT_VISUAL_AUX_MODEL_PATH', str, None)
        latent_config = LatentCoTConfig(
            c_thought=get_env_args('LATENT_COT_C_THOUGHT', int, 2),
            c_thought_visual=get_env_args('LATENT_COT_C_THOUGHT_VISUAL', int, 2),
            aux_model_path=aux_path or None,
            visual_aux_model_path=vis_aux_path or None,
            explain_loss_weight=get_env_args('LATENT_COT_EXPLAIN_LOSS_WEIGHT', float, 1.0),
            visual_explain_loss_weight=get_env_args('LATENT_COT_VISUAL_EXPLAIN_LOSS_WEIGHT', float, 1.0),
            aux_visual_condition=get_env_args('LATENT_COT_AUX_VISUAL_CONDITION', bool, False),
            visual_aux_visual_condition=get_env_args('LATENT_COT_VISUAL_AUX_VISUAL_CONDITION', bool, False),
            use_separate_visual_latent_tokens=get_env_args(
                'LATENT_COT_USE_SEPARATE_VISUAL_LATENT_TOKENS', bool, False),
            freeze_visual_aux_decoder=get_env_args('LATENT_COT_FREEZE_VISUAL_AUX_DECODER', bool, False),
            freeze_aux_decoder=get_env_args('LATENT_COT_FREEZE_AUX_DECODER', bool, False),
            freeze_main_model=get_env_args('LATENT_COT_FREEZE_MAIN_MODEL', bool, False),
            latent_ce_loss=get_env_args('LATENT_COT_LATENT_CE_LOSS', bool, False),
            latent_use_all_subtokens=get_env_args('LATENT_COT_LATENT_USE_ALL_SUBTOKENS', bool, False),
            tokens_as_special=get_env_args('LATENT_COT_TOKENS_AS_SPECIAL', bool, True),
            use_original_vocab=get_env_args('LATENT_COT_USE_ORIGINAL_VOCAB', bool, False),
        )
        patch_model_for_latent_cot(model, processor, latent_config, model_dir=model_dir)
        load_latent_cot_weights(model, model_dir)
        return model


register_model(
    ModelMeta(
        LATENT_COT_MODEL_TYPE, [
            ModelGroup([
                Model('Qwen/Qwen3-VL-2B-Instruct', 'Qwen/Qwen3-VL-2B-Instruct'),
                Model('Qwen/Qwen3-VL-4B-Instruct', 'Qwen/Qwen3-VL-4B-Instruct'),
                Model('Qwen/Qwen3-VL-8B-Instruct', 'Qwen/Qwen3-VL-8B-Instruct'),
                Model('Qwen/Qwen3-VL-2B-Thinking', 'Qwen/Qwen3-VL-2B-Thinking'),
                Model('Qwen/Qwen3-VL-8B-Thinking', 'Qwen/Qwen3-VL-8B-Thinking'),
            ], LATENT_COT_TEMPLATE_TYPE),
        ],
        Qwen3VLLatentCoTLoader,
        model_arch=ModelArch.qwen3_vl,
        architectures=['Qwen3VLForConditionalGeneration'],
        requires=['transformers>=4.57', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'video']))

logger.info(f'[OneVL plugin] Registered model_type={LATENT_COT_MODEL_TYPE!r}.')
