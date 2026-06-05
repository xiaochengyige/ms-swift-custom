# Copyright (c) ModelScope Contributors. All rights reserved.
"""Non-invasive registration of the OneVL ``qwen3_vl_latent_cot`` template.

Reproduces ``Qwen3VLLatentCoTTemplate`` from the original OneVL training fork
without touching any ms-swift source file.  The template:

1. Masks the latent marker tokens in ``labels`` (so they do not contribute to
   the main CE loss) -- unless ``LATENT_COT_LATENT_CE_LOSS`` is set.
2. Passes through the ``think_steps`` / ``future_image_tokens`` dataset columns
   (collected automatically into ``StdTemplateInputs.extra_kwargs``) into the
   encoded sample, and aggregates them per-batch in ``_data_collator`` so the
   patched ``model.forward`` receives them as keyword arguments.

It is registered under the plain string ``'qwen3_vl_latent_cot'`` via
``register_template`` (invoked at import time through ``--external_plugins``).
"""

from typing import Any, Dict, List, Optional

from swift.template import StdTemplateInputs, register_template
from swift.template.templates.qwen import Qwen3VLTemplate, QwenTemplateMeta
from swift.utils import get_env_args, get_logger

logger = get_logger()

LATENT_COT_TEMPLATE_TYPE = 'qwen3_vl_latent_cot'


class Qwen3VLLatentCoTTemplate(Qwen3VLTemplate):
    """Qwen3-VL template with latent CoT support.

    Masks latent token positions in labels and passes through think_steps /
    future_image_tokens from dataset extra_kwargs so that the patched model
    forward can compute auxiliary decoder losses.
    """

    LATENT_TOKENS = {
        '<|latent|>', '<|start-latent|>', '<|end-latent|>',
        '<|latent-vis|>', '<|start-latent-vis|>', '<|end-latent-vis|>',
    }

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)

        input_ids = encoded['input_ids']
        labels = encoded.get('labels')
        latent_ce_loss = get_env_args('LATENT_COT_LATENT_CE_LOSS', bool, False)
        if labels is not None and not latent_ce_loss:
            tokenizer = self.tokenizer
            probe_id = tokenizer.convert_tokens_to_ids('<|latent|>')
            if probe_id != tokenizer.unk_token_id:
                latent_ids = set()
                for tok in self.LATENT_TOKENS:
                    tid = tokenizer.convert_tokens_to_ids(tok)
                    if tid != tokenizer.unk_token_id:
                        latent_ids.add(tid)
                if latent_ids:
                    for i, tid in enumerate(input_ids):
                        if tid in latent_ids:
                            labels[i] = -100
            else:
                self._mask_latent_region_by_pattern(input_ids, labels, tokenizer)
            encoded['labels'] = labels

        for key in ('think_steps', 'future_image_tokens'):
            val = inputs.extra_kwargs.get(key)
            if val is not None:
                encoded[key] = val

        return encoded

    def _mask_latent_region_by_pattern(self, input_ids, labels, tokenizer):
        """Mask labels for the full latent region when markers are sub-tokenized.

        Finds the ``latent`` keyword token, then expands outward through
        contiguous marker-component tokens to cover the whole
        ``<|start-latent|>...<|end-latent|>`` block.
        """
        from .latent_cot import (
            find_latent_mask_region, _get_marker_component_ids, _get_latent_pattern_ids)

        pat = _get_latent_pattern_ids(tokenizer)
        lkw = pat['latent_keyword_id']
        if lkw is None:
            return
        marker_ids = _get_marker_component_ids(tokenizer)
        mask_positions = find_latent_mask_region(input_ids, marker_ids, lkw)
        for i in mask_positions:
            labels[i] = -100

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        latent_fields = {}
        for key in ('think_steps', 'future_image_tokens'):
            values = [b.pop(key, None) for b in batch]
            if any(v is not None for v in values):
                latent_fields[key] = values

        res = super()._data_collator(batch, padding_to=padding_to)
        res.update(latent_fields)
        return res

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs


register_template(
    QwenTemplateMeta(
        LATENT_COT_TEMPLATE_TYPE,
        template_cls=Qwen3VLLatentCoTTemplate,
        default_system=None,
        thinking_prefix='<think>\n'))

logger.info(f'[OneVL plugin] Registered template={LATENT_COT_TEMPLATE_TYPE!r}.')
