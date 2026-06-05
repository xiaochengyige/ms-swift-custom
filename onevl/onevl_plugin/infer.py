# Copyright (c) ModelScope Contributors. All rights reserved.
"""Non-invasive OneVL inference integration for ``swift infer``.

Two inference modes are supported, both running through the stock
``TransformersEngine`` (``--infer_backend pt``):

1. Trajectory / answer-only (prefill) inference -- use the plain
   ``qwen3_vl_latent_cot`` template plus ``--response_prefix`` to inject the
   latent block in front of ``<answer>``.  Nothing in this file is required for
   that path; ``build_latent_response_prefix`` is provided as a convenience to
   construct the prefix string used by the scripts.

2. Inference *with* auxiliary-decoder explanations (text CoT and/or future
   visual tokens).  This requires hidden states of the latent positions, which
   are decoded by the auxiliary decoders attached to the model by
   ``Qwen3VLLatentCoTLoader``.  We expose this through a dedicated template
   ``qwen3_vl_latent_cot_explain`` that overrides ``Template.generate`` (to run
   one forward pass for hidden states and decode the explanations) and
   ``Template.decode`` (to attach them to the response, behind explicit
   delimiters so the trajectory remains parseable as a prefix).

The auxiliary-decode routines are ported from ``OneVL/infer/infer_onevl.py`` but
reuse the latent-position finders from :mod:`latent_cot` to avoid duplication.
"""

import os
from typing import Any, Dict, List, Optional

import torch

from swift.template import register_template
from swift.template.templates.qwen import QwenTemplateMeta
from swift.utils import get_env_args, get_logger

from . import latent_cot as lc
from .template import Qwen3VLLatentCoTTemplate

logger = get_logger()

EXPLAIN_TEMPLATE_TYPE = 'qwen3_vl_latent_cot_explain'

TEXT_EXPLAIN_DELIMITER = '\n<onevl_text_explain>\n'
VISUAL_EXPLAIN_DELIMITER = '\n<onevl_visual_explain>\n'


# ---------------------------------------------------------------------------
# Latent response-prefix builder (matches OneVL/infer/infer_onevl.py)
# ---------------------------------------------------------------------------

def build_latent_response_prefix(num_latent: int = 2, num_latent_vis: int = 4, answer_prefix: str = '[') -> str:
    """Build the assistant latent prefix injected before generation.

    Mirrors the ``assistant_prefix`` constructed in infer_onevl.py: a visual
    latent block (optional) followed by a text latent block and ``<answer>``.
    """
    if num_latent_vis > 0:
        return ('<|start-latent-vis|>' + '<|latent-vis|>' * num_latent_vis
                + '<|end-latent-vis|><|start-latent|>' + '<|latent|>' * num_latent
                + f'<|end-latent|><answer>{answer_prefix}')
    return ('<|start-latent|>' + '<|latent|>' * num_latent
            + f'<|end-latent|><answer>{answer_prefix}')


# ---------------------------------------------------------------------------
# Inference-time latent-position detection (reuses latent_cot finders)
# ---------------------------------------------------------------------------

def compute_inference_latent_positions(input_ids_single, tokenizer, pattern_ids, marker_component_ids):
    """Return ``(text_positions, visual_positions)`` for one sequence using the
    original-vocab sub-token pattern matching (same logic as training)."""
    ids_list = (input_ids_single.tolist() if hasattr(input_ids_single, 'tolist') else input_ids_single)
    lkw = pattern_ids['latent_keyword_id']
    pipe = pattern_ids['pipe_id']
    vis_suffix_id = pattern_ids.get('vis_suffix_id')

    text_kw = lc.find_latent_positions_from_pattern(ids_list, lkw, pipe)
    vis_kw = (lc.find_visual_latent_positions_from_pattern(ids_list, lkw, pipe, vis_suffix_id)
              if vis_suffix_id else [])

    text_block_start = lc._find_text_latent_block_start(ids_list, pipe, vis_suffix_id, tokenizer)
    stop_txt = set(text_kw)
    vis_pos_full = (lc._expand_keyword_positions_with_stop(ids_list, vis_kw, marker_component_ids, stop_txt)
                    if vis_kw else [])
    vis_pos = [p for p in vis_pos_full if p < text_block_start]
    text_pos = lc._expand_keyword_positions_with_stop(ids_list, text_kw, marker_component_ids, vis_pos)
    text_pos = [p for p in text_pos if p >= text_block_start]
    return text_pos, vis_pos


# ---------------------------------------------------------------------------
# Aux-decoder helpers (ported from infer_onevl.py)
# ---------------------------------------------------------------------------

def _get_aux_input_embeddings(aux_decoder):
    if hasattr(aux_decoder, 'model') and hasattr(aux_decoder.model, 'get_input_embeddings'):
        return aux_decoder.model.get_input_embeddings()
    return aux_decoder.get_input_embeddings()


def _call_aux_decoder_lm(aux_decoder, inputs_embeds, use_cache=False, past_key_values=None):
    """Forward through the aux decoder's language_model + lm_head directly."""
    if (hasattr(aux_decoder, 'model') and hasattr(aux_decoder.model, 'language_model')
            and hasattr(aux_decoder, 'lm_head')):
        lm_out = aux_decoder.model.language_model(
            inputs_embeds=inputs_embeds, use_cache=use_cache, past_key_values=past_key_values)
        hidden = (lm_out.last_hidden_state if hasattr(lm_out, 'last_hidden_state') else lm_out[0])
        logits = aux_decoder.lm_head(hidden)
        new_past = (lm_out.past_key_values if use_cache and hasattr(lm_out, 'past_key_values') else None)
        return logits, new_past
    out = aux_decoder(inputs_embeds=inputs_embeds, use_cache=use_cache, past_key_values=past_key_values)
    logits = out.logits if hasattr(out, 'logits') else out[0]
    new_past = (out.past_key_values if use_cache and hasattr(out, 'past_key_values') else None)
    return logits, new_past


def _extract_visual_embeds(student_embeds, input_ids, image_token_id, video_token_id=None):
    vis_mask = (input_ids == image_token_id)
    if video_token_id is not None:
        vis_mask = vis_mask | (input_ids == video_token_id)
    if not vis_mask.any():
        return None
    return student_embeds[vis_mask]


@torch.no_grad()
def decode_latent_with_aux(model, aux_decoder, latent_proj, input_ids, last_hidden, tokenizer,
                           text_positions_list, use_visual_condition=False, image_token_id=None,
                           video_token_id=None, max_explain_tokens=512, vit_embeds=None):
    """Autoregressively decode the text auxiliary explanation per batch item."""
    batch_size = input_ids.size(0)
    aux_embedding = _get_aux_input_embeddings(aux_decoder)
    results = []
    for b in range(batch_size):
        positions = text_positions_list[b]
        if not positions:
            results.append('')
            continue
        latent_embeds = last_hidden[b, positions, :]
        if latent_proj is not None:
            latent_embeds = latent_proj(latent_embeds)

        parts = []
        if use_visual_condition and image_token_id is not None:
            if vit_embeds is not None:
                student_embeds_b = vit_embeds[b]
            else:
                embed_fn = (model.model.get_input_embeddings() if hasattr(model, 'model')
                            else model.get_input_embeddings())
                student_embeds_b = embed_fn(input_ids[b])
            vit_cond = _extract_visual_embeds(student_embeds_b, input_ids[b], image_token_id, video_token_id)
            if vit_cond is not None:
                parts.append(vit_cond)
        parts.append(latent_embeds)
        cur_embeds = torch.cat(parts, dim=0).unsqueeze(0)

        generated_ids = []
        past_kv = None
        for _ in range(max_explain_tokens):
            logits, past_kv = _call_aux_decoder_lm(aux_decoder, cur_embeds, use_cache=True, past_key_values=past_kv)
            next_id = logits[:, -1, :].argmax(dim=-1)
            generated_ids.append(next_id.item())
            if next_id.item() == tokenizer.eos_token_id:
                break
            cur_embeds = aux_embedding(next_id).unsqueeze(1)
        results.append(tokenizer.decode(generated_ids, skip_special_tokens=True))
    return results


@torch.no_grad()
def decode_latent_with_visual_aux(model, visual_aux_decoder, visual_latent_proj, input_ids, last_hidden,
                                  visual_positions_list, use_visual_condition=False, image_token_id=None,
                                  video_token_id=None, max_visual_tokens=512, vit_embeds=None,
                                  vis_aux_tokenizer=None):
    """Autoregressively decode the visual auxiliary explanation per batch item."""
    batch_size = input_ids.size(0)
    aux_embedding = _get_aux_input_embeddings(visual_aux_decoder)
    eos_id = vis_aux_tokenizer.eos_token_id if vis_aux_tokenizer is not None else None
    results = []
    for b in range(batch_size):
        positions = visual_positions_list[b]
        if not positions:
            results.append('')
            continue
        latent_embeds = last_hidden[b, positions, :]
        if visual_latent_proj is not None:
            latent_embeds = visual_latent_proj(latent_embeds)

        parts = []
        if use_visual_condition and image_token_id is not None:
            if vit_embeds is not None:
                student_embeds_b = vit_embeds[b]
            else:
                embed_fn = (model.model.get_input_embeddings() if hasattr(model, 'model')
                            else model.get_input_embeddings())
                student_embeds_b = embed_fn(input_ids[b])
            vit_cond = _extract_visual_embeds(student_embeds_b, input_ids[b], image_token_id, video_token_id)
            if vit_cond is not None:
                parts.append(vit_cond)
        parts.append(latent_embeds)
        cur_embeds = torch.cat(parts, dim=0).unsqueeze(0)

        generated_ids = []
        past_kv = None
        for _ in range(max_visual_tokens):
            logits, past_kv = _call_aux_decoder_lm(
                visual_aux_decoder, cur_embeds, use_cache=True, past_key_values=past_kv)
            next_id = logits[:, -1, :].argmax(dim=-1)
            generated_ids.append(next_id.item())
            if eos_id is not None and next_id.item() == eos_id:
                break
            cur_embeds = aux_embedding(next_id).unsqueeze(1)
        tok = vis_aux_tokenizer if vis_aux_tokenizer is not None else None
        results.append(tok.decode(generated_ids, skip_special_tokens=True) if tok is not None else '')
    return results


# ---------------------------------------------------------------------------
# Explain template (overrides generate / decode)
# ---------------------------------------------------------------------------

_FORWARD_KEYS = (
    'input_ids', 'attention_mask', 'position_ids', 'pixel_values', 'pixel_values_videos',
    'image_grid_thw', 'video_grid_thw',
)


class Qwen3VLLatentCoTExplainTemplate(Qwen3VLLatentCoTTemplate):
    """Latent CoT template that also emits auxiliary-decoder explanations.

    The explanations are computed in ``generate`` (one extra forward pass to get
    latent hidden states, then autoregressive aux decoding) and appended to the
    response in ``decode`` behind explicit delimiters.
    """

    def _explain_flags(self):
        return {
            'text': get_env_args('LATENT_COT_DECODER_EXPLAIN', bool, False),
            'visual': get_env_args('LATENT_COT_VISUAL_DECODER_EXPLAIN', bool, False),
            'text_vis_cond': get_env_args('LATENT_COT_AUX_VISUAL_CONDITION', bool, False),
            'visual_vis_cond': get_env_args('LATENT_COT_VISUAL_AUX_VISUAL_CONDITION', bool, False),
            'max_explain_tokens': get_env_args('LATENT_COT_MAX_EXPLAIN_TOKENS', int, 512),
            'max_visual_tokens': get_env_args('LATENT_COT_MAX_VISUAL_TOKENS', int, 1024),
        }

    def generate(self, model, *args, **kwargs):
        self._onevl_text_explains = None
        self._onevl_visual_explains = None
        self._onevl_explain_idx = 0
        try:
            self._run_explain(model, kwargs)
        except Exception as e:  # explanation is best-effort; never break generation
            logger.warning_once(f'[OneVL explain] skipped due to: {e}')
        return super().generate(model, *args, **kwargs)

    @torch.no_grad()
    def _run_explain(self, model, gen_kwargs):
        flags = self._explain_flags()
        aux_decoder = getattr(model, '_latent_cot_aux_decoder', None)
        visual_aux_decoder = getattr(model, '_latent_cot_visual_aux_decoder', None)
        want_text = flags['text'] and aux_decoder is not None
        want_visual = flags['visual'] and visual_aux_decoder is not None
        if not (want_text or want_visual):
            return

        input_ids = gen_kwargs.get('input_ids')
        if input_ids is None:
            return
        tokenizer = self.tokenizer
        pattern_ids = lc._get_latent_pattern_ids(tokenizer)
        marker_ids = lc._get_marker_component_ids(tokenizer)

        fwd_inputs = {k: gen_kwargs[k] for k in _FORWARD_KEYS if k in gen_kwargs and gen_kwargs[k] is not None}

        # Capture ViT-injected embeddings for the visual-condition prefix.
        captured = {}
        hook = None
        lm = getattr(getattr(model, 'model', None), 'language_model', None)
        if (flags['text_vis_cond'] or flags['visual_vis_cond']) and lm is not None:
            def _hook(module, args, kwargs_):
                ie = kwargs_.get('inputs_embeds')
                if ie is not None:
                    captured['embeds'] = ie.detach()
                return None
            hook = lm.register_forward_pre_hook(_hook, with_kwargs=True)

        origin_forward = getattr(model, '_origin_forward_for_latent_cot', model.forward)
        out = origin_forward(**fwd_inputs, output_hidden_states=True, return_dict=True)
        if hook is not None:
            hook.remove()
        last_hidden = out.hidden_states[-1]
        vit_embeds = captured.get('embeds')

        bs = input_ids.size(0)
        text_positions_list, visual_positions_list = [], []
        for b in range(bs):
            tp, vp = compute_inference_latent_positions(input_ids[b], tokenizer, pattern_ids, marker_ids)
            text_positions_list.append(tp)
            visual_positions_list.append(vp)

        image_token_id = getattr(model.config, 'image_token_id', None)
        video_token_id = getattr(model.config, 'video_token_id', None)

        if want_text:
            self._onevl_text_explains = decode_latent_with_aux(
                model, aux_decoder, getattr(model, '_latent_cot_latent_proj', None),
                input_ids, last_hidden, tokenizer, text_positions_list,
                use_visual_condition=flags['text_vis_cond'], image_token_id=image_token_id,
                video_token_id=video_token_id, max_explain_tokens=flags['max_explain_tokens'],
                vit_embeds=vit_embeds)
        if want_visual:
            self._onevl_visual_explains = decode_latent_with_visual_aux(
                model, visual_aux_decoder, getattr(model, '_latent_cot_visual_latent_proj', None),
                input_ids, last_hidden, visual_positions_list,
                use_visual_condition=flags['visual_vis_cond'], image_token_id=image_token_id,
                video_token_id=video_token_id, max_visual_tokens=flags['max_visual_tokens'],
                vit_embeds=vit_embeds,
                vis_aux_tokenizer=getattr(model, '_latent_cot_visual_aux_tokenizer', None))

    def decode(self, generate_ids, **kwargs):
        response = super().decode(generate_ids, **kwargs)
        idx = getattr(self, '_onevl_explain_idx', 0)
        text_explains = getattr(self, '_onevl_text_explains', None)
        visual_explains = getattr(self, '_onevl_visual_explains', None)
        suffix = ''
        if text_explains is not None and idx < len(text_explains) and text_explains[idx]:
            suffix += TEXT_EXPLAIN_DELIMITER + text_explains[idx]
        if visual_explains is not None and idx < len(visual_explains) and visual_explains[idx]:
            suffix += VISUAL_EXPLAIN_DELIMITER + visual_explains[idx]
        self._onevl_explain_idx = idx + 1
        return response + suffix


register_template(
    QwenTemplateMeta(
        EXPLAIN_TEMPLATE_TYPE,
        template_cls=Qwen3VLLatentCoTExplainTemplate,
        default_system=None,
        thinking_prefix='<think>\n'))

logger.info(f'[OneVL plugin] Registered explain template={EXPLAIN_TEMPLATE_TYPE!r}.')
