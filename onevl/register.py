# Copyright (c) ModelScope Contributors. All rights reserved.
"""OneVL non-invasive plugin entry point for ms-swift 4.2.0.

Pass this file to ``--external_plugins`` so that, at argument-parsing time,
ms-swift imports it and the following non-invasive extensions take effect
WITHOUT editing any file under ``swift/``:

* ``register_model('qwen3_vl_latent_cot', ...)``      -- via ``onevl_plugin.model``
* ``register_template('qwen3_vl_latent_cot', ...)``   -- via ``onevl_plugin.template``
* ``register_template('qwen3_vl_latent_cot_explain')``-- via ``onevl_plugin.infer``
* ``loss_map['latent_cot'] = LatentCoTLoss``          -- runtime dict injection
* ``RowPreprocessor.standard_keys += [...]``          -- keep think_steps / future_image_tokens columns
* monkey-patch ``TunerMixin.prepare_model``           -- apply latent-CoT freeze after prepare

``import_external_file`` inserts this file's directory (``custom/onevl``) onto
``sys.path`` and imports this module top-level, so ``onevl_plugin`` is importable
as a package and its intra-package relative imports work correctly.

Usage:
    swift sft   --external_plugins onevl/register.py --model_type qwen3_vl_latent_cot ...
    swift infer --external_plugins onevl/register.py --model_type qwen3_vl_latent_cot ...
"""

from swift.utils import get_logger

logger = get_logger()

# 1) Model + templates (side-effect registration via register_model / register_template)
import onevl_plugin.model  # noqa: E402,F401  -> registers model_type 'qwen3_vl_latent_cot'
import onevl_plugin.template  # noqa: E402,F401  -> registers template 'qwen3_vl_latent_cot'
import onevl_plugin.infer  # noqa: E402,F401  -> registers template 'qwen3_vl_latent_cot_explain'

# 2) Loss: inject into the runtime loss_map dict (looked up lazily by the trainer)
from swift.loss.mapping import loss_map  # noqa: E402
from onevl_plugin.loss import LatentCoTLoss  # noqa: E402

loss_map['latent_cot'] = LatentCoTLoss
logger.info("[OneVL plugin] Registered loss_type='latent_cot'.")

# 3) Dataset: keep the extra columns alive through preprocessing so they reach
#    StdTemplateInputs.extra_kwargs (and then the template / patched forward).
from swift.dataset.preprocessor.core import RowPreprocessor  # noqa: E402

for _k in ('think_steps', 'future_image_tokens'):
    if _k not in RowPreprocessor.standard_keys:
        RowPreprocessor.standard_keys.append(_k)
logger.info(f'[OneVL plugin] standard_keys extended: {RowPreprocessor.standard_keys[-2:]}')

# 4) Tuner: apply the latent-CoT freeze AFTER ms-swift's prepare_model has reset
#    requires_grad / applied freeze_parameters (so our settings are not overridden).
from swift.pipelines.train.tuner import TunerMixin  # noqa: E402

if not getattr(TunerMixin, '_onevl_latent_cot_patched', False):
    _orig_prepare_model = TunerMixin.prepare_model.__func__

    def _patched_prepare_model(cls, args, model, **kwargs):
        model = _orig_prepare_model(cls, args, model, **kwargs)
        if getattr(model, '_latent_cot_config', None) is not None:
            from onevl_plugin.latent_cot import apply_latent_cot_freeze
            apply_latent_cot_freeze(model)
        return model

    TunerMixin.prepare_model = classmethod(_patched_prepare_model)
    TunerMixin._onevl_latent_cot_patched = True
    logger.info('[OneVL plugin] Patched TunerMixin.prepare_model for latent-CoT freeze.')

logger.info('[OneVL plugin] All non-invasive registrations complete.')
