"""
Latent Chain-of-Thought (CoT) support for Qwen3-VL in ms-swift.

Ports the CODI/SIM-CoT approach from veomni:
- Latent tokens replace explicit reasoning in the input
- Auxiliary decoder(s) reconstruct original reasoning from latent hidden states
- Visual auxiliary decoder reconstructs future image tokens from latent states

The model is patched in-place: aux decoders and projection layers are added
as submodules, and forward is monkey-patched to compute the combined loss.
"""

from dataclasses import dataclass
import os
from types import MethodType
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from swift.utils import get_logger

logger = get_logger()

LATENT_TOKEN = '<|latent|>'
START_LATENT_TOKEN = '<|start-latent|>'
END_LATENT_TOKEN = '<|end-latent|>'
LATENT_VIS_TOKEN = '<|latent-vis|>'
START_LATENT_VIS_TOKEN = '<|start-latent-vis|>'
END_LATENT_VIS_TOKEN = '<|end-latent-vis|>'

LATENT_SPECIAL_TOKENS = [
    START_LATENT_TOKEN, LATENT_TOKEN, END_LATENT_TOKEN,
    START_LATENT_VIS_TOKEN, LATENT_VIS_TOKEN, END_LATENT_VIS_TOKEN,
]


@dataclass
class LatentCoTConfig:
    c_thought: int = 2
    c_thought_visual: int = 2
    aux_model_path: Optional[str] = None
    visual_aux_model_path: Optional[str] = None
    explain_loss_weight: float = 1.0
    visual_explain_loss_weight: float = 1.0
    aux_visual_condition: bool = False
    visual_aux_visual_condition: bool = False
    use_separate_visual_latent_tokens: bool = False
    freeze_visual_aux_decoder: bool = False
    freeze_aux_decoder: bool = False
    freeze_main_model: bool = False
    latent_ce_loss: bool = False
    latent_use_all_subtokens: bool = False
    tokens_as_special: bool = True
    use_original_vocab: bool = False


def add_latent_tokens_to_tokenizer(processor, as_special_tokens: bool = True):
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in LATENT_SPECIAL_TOKENS if t not in existing_tokens]
    if new_tokens:
        tokenizer.add_tokens(new_tokens, special_tokens=as_special_tokens)
        kind = 'special' if as_special_tokens else 'regular'
        logger.info(f'Added {len(new_tokens)} latent tokens as {kind} tokens: {new_tokens}')
    return tokenizer


def get_latent_token_ids(tokenizer):
    return {
        'latent_token_id': tokenizer.convert_tokens_to_ids(LATENT_TOKEN),
        'start_latent_id': tokenizer.convert_tokens_to_ids(START_LATENT_TOKEN),
        'end_latent_id': tokenizer.convert_tokens_to_ids(END_LATENT_TOKEN),
        'latent_visual_token_id': tokenizer.convert_tokens_to_ids(LATENT_VIS_TOKEN),
        'start_latent_visual_id': tokenizer.convert_tokens_to_ids(START_LATENT_VIS_TOKEN),
        'end_latent_visual_id': tokenizer.convert_tokens_to_ids(END_LATENT_VIS_TOKEN),
    }


def _resolve_hidden_size(model):
    cfg = model.config if hasattr(model, 'config') else None
    if cfg is None:
        raise ValueError('Cannot determine hidden_size: model has no config')
    if hasattr(cfg, 'text_config'):
        return cfg.text_config.hidden_size
    return cfg.hidden_size


def build_aux_decoder(model_path, torch_dtype=None, device='cpu'):
    from transformers import AutoConfig, AutoModelForCausalLM
    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    # device_map is incompatible with DeepSpeed Zero-3; pass None and let
    # DeepSpeed handle parameter placement via the Zero-3 init context.
    try:
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
        use_device_map = None if is_deepspeed_zero3_enabled() else device
    except ImportError:
        use_device_map = device

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, 'model_type', '')

    if 'qwen3_vl' in model_type:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=use_device_map, trust_remote_code=True)
    if 'qwen2_vl' in model_type:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=use_device_map, trust_remote_code=True)
    return AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map=use_device_map, trust_remote_code=True)


def _checkpoint_has_visual_aux(model_dir: str) -> bool:
    """Return True if the checkpoint at model_dir contains _latent_cot_visual_aux_decoder weights."""
    import json
    import os
    index_path = os.path.join(model_dir, 'model.safetensors.index.json')
    single_path = os.path.join(model_dir, 'model.safetensors')
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f).get('weight_map', {})
        return any(k.startswith('_latent_cot_visual_aux_decoder.') for k in weight_map)
    if os.path.exists(single_path):
        from safetensors.torch import load_file
        keys = load_file(single_path).keys()
        return any(k.startswith('_latent_cot_visual_aux_decoder.') for k in keys)
    return False


def _get_aux_input_embeddings(aux_decoder):
    if hasattr(aux_decoder, 'model') and hasattr(aux_decoder.model, 'get_input_embeddings'):
        return aux_decoder.model.get_input_embeddings()
    return aux_decoder.get_input_embeddings()


def _extract_visual_embeds(student_embeds, batch_idx, input_ids, image_token_id, video_token_id=None):
    dev = student_embeds.device
    vis_mask = (input_ids[batch_idx] == image_token_id)
    if video_token_id is not None:
        vis_mask = vis_mask | (input_ids[batch_idx] == video_token_id)
    vis_mask = vis_mask.to(dev)
    if not vis_mask.any():
        return None
    return student_embeds[batch_idx][vis_mask]


def _latent_cot_debug_aux_vit_embed() -> bool:
    v = os.environ.get('LATENT_COT_DEBUG_AUX_VIT_EMBED', '')
    return str(v).lower() in ('1', 'true', 'yes', 'on')


def _latent_cot_debug_aux_vit_max_prints() -> int:
    try:
        return max(0, int(os.environ.get('LATENT_COT_DEBUG_AUX_VIT_EMBED_MAX', '5')))
    except ValueError:
        return 5


_latent_cot_debug_aux_vit_print_count = 0


def _record_aux_vit_for_debug(
    holder: Optional[dict], key: str, batch_idx: int, vis_embeds: Optional[torch.Tensor],
) -> None:
    if holder is None or batch_idx != 0:
        return
    holder[key] = None if vis_embeds is None else vis_embeds.detach()


def _flush_aux_vit_debug_compare(holder: Optional[dict]) -> None:
    global _latent_cot_debug_aux_vit_print_count
    if not holder or not _latent_cot_debug_aux_vit_embed():
        return
    mx = _latent_cot_debug_aux_vit_max_prints()
    if mx > 0 and _latent_cot_debug_aux_vit_print_count >= mx:
        return
    missing = object()
    t = holder.pop('text_aux_vit', missing)
    v = holder.pop('visual_aux_vit', missing)
    lim = str(mx) if mx > 0 else 'inf'

    if t is missing or v is missing:
        _latent_cot_debug_aux_vit_print_count += 1
        print(
            f'[LatentCoT DEBUG AUX-VIT] batch0 compare skipped: '
            f'text_recorded={t is not missing}, visual_recorded={v is not missing} '
            f'(enable both aux decoders + both visual conditions to compare) '
            f'[{_latent_cot_debug_aux_vit_print_count}/{lim}]')
        holder.clear()
        return
    if t is None and v is None:
        _latent_cot_debug_aux_vit_print_count += 1
        print(
            f'[LatentCoT DEBUG AUX-VIT] batch0: text_aux and visual_aux ViT prefix both None '
            f'[{_latent_cot_debug_aux_vit_print_count}/{lim}]')
        holder.clear()
        return
    if (t is None) != (v is None):
        _latent_cot_debug_aux_vit_print_count += 1
        print(
            f'[LatentCoT DEBUG AUX-VIT] batch0 MISMATCH: text_ve is None={t is None}, '
            f'visual_ve is None={v is None} [{_latent_cot_debug_aux_vit_print_count}/{lim}]')
        holder.clear()
        return
    same_shape = t.shape == v.shape
    max_diff = (t.float() - v.float()).abs().max().item() if same_shape else None
    all_eq = same_shape and bool(torch.equal(t, v))
    _latent_cot_debug_aux_vit_print_count += 1
    print(
        f'[LatentCoT DEBUG AUX-VIT] train batch0 text_aux vs visual_aux ViT rows: '
        f'shape={tuple(t.shape)}, identical={all_eq}, max_abs_diff={max_diff} '
        f'[{_latent_cot_debug_aux_vit_print_count}/{lim}]')
    holder.clear()


def compute_explain_loss(
    last_hidden_states, input_ids, latent_lists, explainable_ids_list,
    batch_size, aux_decoder, latent_proj, c_thought,
    student_embeds=None, use_visual_condition=False,
    image_token_id=None, video_token_id=None,
    n_markers_list=None,
    debug_aux_vit_holder: Optional[dict] = None,
):
    """Compute auxiliary decoder loss.

    Args:
        n_markers_list: optional list (one per batch sample) of actual marker
            counts.  When provided, grouping uses ``n_markers // c_thought``
            and positions are divided evenly across groups.  When *None*,
            falls back to the original ``len(positions) // c_thought``.
    """
    if aux_decoder is None or explainable_ids_list is None:
        return torch.tensor(0.0, device=last_hidden_states.device)

    aux_embedding = _get_aux_input_embeddings(aux_decoder)
    loss_fct = CrossEntropyLoss(reduction='sum')
    loss_all = torch.tensor(0.0, device=last_hidden_states.device)
    num_steps = 0

    for b in range(batch_size):
        if not latent_lists[b]:
            continue

        vis_embeds = None
        n_vis = 0
        if use_visual_condition and student_embeds is not None and image_token_id is not None:
            vis_embeds = _extract_visual_embeds(
                student_embeds, b, input_ids, image_token_id, video_token_id)
            if vis_embeds is not None:
                n_vis = vis_embeds.shape[0]

        _record_aux_vit_for_debug(debug_aux_vit_holder, 'text_aux_vit', b, vis_embeds)

        n_positions = len(latent_lists[b])
        if n_markers_list is not None:
            n_markers = n_markers_list[b]
            n_latent_groups = n_markers // c_thought if c_thought > 0 else 0
            positions_per_group = n_positions // n_latent_groups if n_latent_groups > 0 else n_positions
        else:
            n_latent_groups = n_positions // c_thought
            positions_per_group = c_thought
        step_ids = explainable_ids_list[b]

        for step_idx in range(min(n_latent_groups, len(step_ids))):
            start_pos = step_idx * positions_per_group
            end_pos = min(start_pos + positions_per_group, n_positions)
            latent_positions = latent_lists[b][start_pos:end_pos]
            latent_embeds = last_hidden_states[b, latent_positions, :]

            if latent_proj is not None:
                latent_embeds = latent_proj(latent_embeds)

            step_token_ids = step_ids[step_idx]
            if not step_token_ids or len(step_token_ids) == 0:
                continue

            step_tensor = torch.tensor(
                step_token_ids, device=last_hidden_states.device, dtype=torch.long)
            step_embeds = aux_embedding(step_tensor)

            parts = []
            if vis_embeds is not None:
                parts.append(vis_embeds)
            parts.append(latent_embeds)
            parts.append(step_embeds)
            combined_embeds = torch.cat(parts, dim=0).unsqueeze(0)
            logger.info_once(
                f'[AuxDecoder input] vis_cond={use_visual_condition}, '
                f'n_vis={n_vis}, n_latent={len(latent_positions)}, '
                f'n_step_tokens={len(step_token_ids)}, '
                f'combined_shape={combined_embeds.shape}')

            prefix_len = n_vis + len(latent_positions)
            labels_explain = torch.full(
                (1, combined_embeds.shape[1]), -100,
                dtype=torch.long, device=last_hidden_states.device)
            labels_explain[0, prefix_len:] = step_tensor

            attn_mask = torch.ones(
                (1, combined_embeds.shape[1]),
                dtype=torch.long, device=last_hidden_states.device)

            aux_outputs = aux_decoder(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                use_cache=False,
            )

            logits = aux_outputs.logits if hasattr(aux_outputs, 'logits') else aux_outputs[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_explain[..., 1:].contiguous()

            effective_tokens = (shift_labels != -100).sum()
            if effective_tokens > 0:
                step_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))
                loss_all += step_loss / effective_tokens
                num_steps += 1

    if num_steps > 0:
        loss_all = loss_all / num_steps
    return loss_all


def compute_visual_explain_loss(
    last_hidden_states, input_ids, latent_lists, visual_ids_list,
    batch_size, visual_aux_decoder, visual_latent_proj, c_thought_visual,
    student_embeds=None, use_visual_condition=False,
    image_token_id=None, video_token_id=None,
    n_markers_list=None,
    debug_aux_vit_holder: Optional[dict] = None,
):
    """Compute visual auxiliary decoder loss.

    Latent embeddings are directly concatenated (no pooling), consistent
    with ``compute_explain_loss``.  ``n_markers_list`` is accepted for
    API consistency but does not affect the computation here.
    """
    if visual_aux_decoder is None or visual_ids_list is None:
        return torch.tensor(0.0, device=last_hidden_states.device)

    vis_aux_embedding = _get_aux_input_embeddings(visual_aux_decoder)
    loss_fct = CrossEntropyLoss(reduction='sum')
    loss_all = torch.tensor(0.0, device=last_hidden_states.device)
    num_items = 0

    for b in range(batch_size):
        if not latent_lists[b]:
            continue
        vis_token_ids = visual_ids_list[b] if b < len(visual_ids_list) else None
        if not vis_token_ids or len(vis_token_ids) == 0:
            continue

        latent_positions = latent_lists[b]
        latent_embeds = last_hidden_states[b, latent_positions, :]

        if visual_latent_proj is not None:
            latent_embeds = visual_latent_proj(latent_embeds)

        vis_embeds = None
        n_vis = 0
        if use_visual_condition and student_embeds is not None and image_token_id is not None:
            vis_embeds = _extract_visual_embeds(
                student_embeds, b, input_ids, image_token_id, video_token_id)
            if vis_embeds is not None:
                n_vis = vis_embeds.shape[0]

        _record_aux_vit_for_debug(debug_aux_vit_holder, 'visual_aux_vit', b, vis_embeds)

        target_tensor = torch.tensor(
            vis_token_ids, device=last_hidden_states.device, dtype=torch.long)
        target_embeds = vis_aux_embedding(target_tensor)

        parts = []
        if vis_embeds is not None:
            parts.append(vis_embeds)
        parts.append(latent_embeds)
        parts.append(target_embeds)
        combined_embeds = torch.cat(parts, dim=0).unsqueeze(0)
        logger.info_once(
            f'[VisualAuxDecoder input] vis_cond={use_visual_condition}, '
            f'n_vis={n_vis}, n_latent={len(latent_positions)}, '
            f'n_target_tokens={len(vis_token_ids)}, '
            f'combined_shape={combined_embeds.shape}')

        n_latent_cond = latent_embeds.shape[0]
        prefix_len = n_vis + n_latent_cond
        labels = torch.full(
            (1, combined_embeds.shape[1]), -100,
            dtype=torch.long, device=last_hidden_states.device)
        labels[0, prefix_len:] = target_tensor

        attn_mask = torch.ones(
            (1, combined_embeds.shape[1]),
            dtype=torch.long, device=last_hidden_states.device)

        vis_aux_outputs = visual_aux_decoder(
            inputs_embeds=combined_embeds,
            attention_mask=attn_mask,
            use_cache=False,
        )

        logits = vis_aux_outputs.logits if hasattr(vis_aux_outputs, 'logits') else vis_aux_outputs[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        effective_tokens = (shift_labels != -100).sum()
        if effective_tokens > 0:
            item_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1))
            loss_all += item_loss / effective_tokens
            num_items += 1

    if num_items > 0:
        loss_all = loss_all / num_items
    return loss_all


def _tokenize_think_steps(think_steps_text, tokenizer):
    eos_id = tokenizer.eos_token_id
    if isinstance(think_steps_text, str):
        think_steps_text = [think_steps_text]
    result = []
    for text in think_steps_text:
        if text and str(text).strip():
            steps = [str(text)]
            step_ids = [
                tokenizer.encode(s, add_special_tokens=False) + [eos_id]
                for s in steps if s.strip()
            ]
            result.append(step_ids)
        else:
            result.append([])
    return result


def _tokenize_visual_targets(visual_text, visual_tokenizer):
    eos_id = visual_tokenizer.eos_token_id
    if isinstance(visual_text, str):
        visual_text = [visual_text]
    result = []
    for text in visual_text:
        if text and str(text).strip():
            ids = visual_tokenizer.encode(str(text), add_special_tokens=False)
            if eos_id is not None:
                ids = ids + [eos_id]
            result.append(ids)
        else:
            result.append([])
    return result


def _debug_decode_latent_span(tokenizer, input_ids_one, positions, max_decode_len=200):
    """Decode the span of input_ids at positions to text for debug logging."""
    if not positions or input_ids_one is None:
        return "(no positions)"
    ids = input_ids_one.tolist() if hasattr(input_ids_one, 'tolist') else input_ids_one
    lo, hi = min(positions), max(positions)
    span_ids = ids[lo:hi + 1]
    try:
        text = tokenizer.decode(span_ids, skip_special_tokens=False)
    except Exception:
        return "(decode error)"
    if len(text) > max_decode_len:
        text = text[:max_decode_len] + "..."
    return text.replace("\n", " ").strip()


def _get_latent_pattern_ids(tokenizer):
    """Pre-compute token IDs for pattern-matching latent markers in original vocab mode.

    Returns a dict with key anchor IDs used to locate latent regions:
      - latent_keyword_id: the single-token ID for the word "latent"
      - pipe_id: the single-token ID for "|"
      - vis_suffix_id: the single-token ID for "-vis"
    """
    def _single_id(text):
        enc = tokenizer.encode(text, add_special_tokens=False)
        return enc[0] if len(enc) == 1 else None

    return {
        'latent_keyword_id': _single_id('latent'),
        'pipe_id': _single_id('|'),
        'vis_suffix_id': _single_id('-vis'),
    }


def _get_marker_component_ids(tokenizer):
    """Get the set of token IDs that form latent marker strings.

    Used to expand from a ``latent`` keyword position outward to cover the
    full ``<|start-latent|>...<|end-latent|>`` region for label masking.
    """
    texts = ['<', '>', '|', '><', 'latent', 'start', 'end', '-lat', 'ent', '-vis']
    ids = set()
    for text in texts:
        enc = tokenizer.encode(text, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return ids


def _find_text_latent_block_start(ids_list, pipe_id, vis_suffix_id, tokenizer):
    """Return the first index of the text latent block ``<|start-latent|>``.

    Pattern: pipe, start, -lat, ent, pipe with previous token != -vis
    (so we do not match inside ``<|start-latent-vis|>``).  Used to split
    visual vs text latent spans when use_separate_visual_latent_tokens.
    """
    def _first_id(text):
        enc = tokenizer.encode(text, add_special_tokens=False)
        return enc[0] if len(enc) == 1 else None
    start_id = _first_id('start')
    neglat_id = _first_id('-lat')
    ent_id = _first_id('ent')
    if start_id is None or neglat_id is None or ent_id is None:
        return len(ids_list)
    n = len(ids_list)
    for i in range(1, n - 4):
        if (ids_list[i] == pipe_id
                and ids_list[i + 1] == start_id
                and ids_list[i + 2] == neglat_id
                and ids_list[i + 3] == ent_id
                and ids_list[i + 4] == pipe_id
                and ids_list[i - 1] != vis_suffix_id):
            return i
    return len(ids_list)


def find_latent_positions_from_pattern(ids_list, latent_keyword_id, pipe_id):
    """Find positions of ``<|latent|>`` (not ``<|latent-vis|>``) via the
    ``| latent |`` sub-token pattern.  Returns the index of the *keyword*
    token for each match, giving exactly one position per ``<|latent|>``.
    """
    positions = []
    n = len(ids_list)
    for i in range(1, n - 1):
        if (ids_list[i] == latent_keyword_id
                and ids_list[i - 1] == pipe_id
                and ids_list[i + 1] == pipe_id):
            positions.append(i)
    return positions


def _expand_keyword_positions_with_stop(
    ids_list, keyword_positions, marker_component_ids, stop_before,
):
    """Expand from each keyword position through contiguous marker tokens.

    Stops expanding so we never include any index in *stop_before* (e.g. the
    other latent type's keyword positions when separating text vs visual).
    """
    stop_set = set(stop_before)
    all_positions = []
    used: set = set()
    for kw_pos in keyword_positions:
        start = kw_pos
        while (start > 0
               and (start - 1) not in stop_set
               and ids_list[start - 1] in marker_component_ids
               and (start - 1) not in used):
            start -= 1
        end = kw_pos
        n = len(ids_list)
        while (end < n - 1
               and (end + 1) not in stop_set
               and ids_list[end + 1] in marker_component_ids
               and (end + 1) not in used):
            end += 1
        for p in range(start, end + 1):
            if p not in used:
                all_positions.append(p)
                used.add(p)
    return all_positions


def find_latent_all_positions_from_pattern(
    ids_list, latent_keyword_id, pipe_id, marker_component_ids,
    stop_before=None,
):
    """Find ALL sub-token positions for each ``<|latent|>`` marker region.

    Unlike ``find_latent_positions_from_pattern`` (which returns one keyword
    position per marker), this returns every sub-token position that belongs
    to each ``<|latent|>`` marker.  If *stop_before* is provided (set of
    indices), expansion does not include those positions (for separating
    text vs visual latent spans).
    """
    keyword_positions = find_latent_positions_from_pattern(
        ids_list, latent_keyword_id, pipe_id)
    if not keyword_positions:
        return [], 0
    if stop_before is not None:
        all_positions = _expand_keyword_positions_with_stop(
            ids_list, keyword_positions, marker_component_ids, stop_before)
        return all_positions, len(keyword_positions)

    all_positions = []
    used: set = set()
    for kw_pos in keyword_positions:
        start = kw_pos
        while start > 0 and ids_list[start - 1] in marker_component_ids and (start - 1) not in used:
            start -= 1
        end = kw_pos
        while end < len(ids_list) - 1 and ids_list[end + 1] in marker_component_ids and (end + 1) not in used:
            end += 1
        for p in range(start, end + 1):
            if p not in used:
                all_positions.append(p)
                used.add(p)
    return all_positions, len(keyword_positions)


def find_visual_latent_positions_from_pattern(ids_list, latent_keyword_id, pipe_id, vis_suffix_id):
    """Find positions of ``<|latent-vis|>`` via the ``| latent -vis`` pattern."""
    if vis_suffix_id is None:
        return []
    positions = []
    n = len(ids_list)
    for i in range(1, n - 1):
        if (ids_list[i] == latent_keyword_id
                and ids_list[i - 1] == pipe_id
                and ids_list[i + 1] == vis_suffix_id):
            positions.append(i)
    return positions


def find_visual_latent_all_positions_from_pattern(
    ids_list, latent_keyword_id, pipe_id, vis_suffix_id, marker_component_ids,
    stop_before=None,
):
    """Find ALL sub-token positions for each ``<|latent-vis|>`` marker region.

    If *stop_before* is provided, expansion does not include those positions
    (for separating visual vs text latent spans).
    """
    keyword_positions = find_visual_latent_positions_from_pattern(
        ids_list, latent_keyword_id, pipe_id, vis_suffix_id)
    if not keyword_positions:
        return [], 0
    if stop_before is not None:
        all_positions = _expand_keyword_positions_with_stop(
            ids_list, keyword_positions, marker_component_ids, set(stop_before))
        return all_positions, len(keyword_positions)

    all_positions = []
    used: set = set()
    for kw_pos in keyword_positions:
        start = kw_pos
        while start > 0 and ids_list[start - 1] in marker_component_ids and (start - 1) not in used:
            start -= 1
        end = kw_pos
        while end < len(ids_list) - 1 and ids_list[end + 1] in marker_component_ids and (end + 1) not in used:
            end += 1
        for p in range(start, end + 1):
            if p not in used:
                all_positions.append(p)
                used.add(p)
    return all_positions, len(keyword_positions)


def find_latent_mask_region(ids_list, marker_component_ids, latent_keyword_id):
    """Return the set of positions to mask (label=-100) for all latent marker
    regions.  Strategy: find every ``latent`` keyword occurrence, expand
    outward through contiguous marker-component tokens.
    """
    keyword_positions = [i for i, tid in enumerate(ids_list) if tid == latent_keyword_id]
    if not keyword_positions:
        return set()

    mask = set()
    for pos in keyword_positions:
        if pos in mask:
            continue
        start = pos
        while start > 0 and ids_list[start - 1] in marker_component_ids:
            start -= 1
        end = pos
        while end < len(ids_list) - 1 and ids_list[end + 1] in marker_component_ids:
            end += 1
        mask.update(range(start, end + 1))
    return mask


def patch_model_for_latent_cot(model, processor, config: LatentCoTConfig, model_dir: Optional[str] = None):
    """Patch a Qwen3-VL model in-place for latent CoT training.

    When model_dir is a checkpoint that already contains _latent_cot_visual_aux_decoder
    weights, the visual aux decoder is built from that checkpoint (same config) and
    weights are loaded from the checkpoint only — LATENT_COT_VISUAL_AUX_MODEL_PATH
    is not used for weights in that case.
    """
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    if config.use_original_vocab:
        pattern_ids = _get_latent_pattern_ids(tokenizer)
        model._latent_pattern_ids = pattern_ids
        model._latent_marker_component_ids = _get_marker_component_ids(tokenizer)
        logger.info(
            f'Using original vocab mode — no token additions. '
            f'Pattern IDs: {pattern_ids}')
        token_ids = get_latent_token_ids(tokenizer)
    else:
        add_latent_tokens_to_tokenizer(tokenizer, as_special_tokens=config.tokens_as_special)
        model.resize_token_embeddings(len(tokenizer))
        token_ids = get_latent_token_ids(tokenizer)

    model._latent_cot_config = config
    model._latent_token_ids = token_ids
    model._processor_ref = processor

    base_hidden_size = _resolve_hidden_size(model)

    if config.aux_model_path:
        logger.info(f'Building aux_decoder from: {config.aux_model_path}')
        aux_decoder = build_aux_decoder(config.aux_model_path, device='cpu')
        aux_hidden = _resolve_hidden_size(aux_decoder)
        latent_proj = nn.Sequential(
            nn.Linear(base_hidden_size, base_hidden_size),
            nn.GELU(),
            nn.Linear(base_hidden_size, aux_hidden),
            nn.LayerNorm(aux_hidden),
        )
        model.add_module('_latent_cot_aux_decoder', aux_decoder)
        model.add_module('_latent_cot_latent_proj', latent_proj)

    else:
        model._latent_cot_aux_decoder = None
        model._latent_cot_latent_proj = None

    if config.visual_aux_model_path:
        if model_dir and _checkpoint_has_visual_aux(model_dir):
            logger.info(
                f'Building visual_aux_decoder from: {config.visual_aux_model_path} (architecture only). '
                f'Weights will be restored from checkpoint (model_path={model_dir}).')
        else:
            logger.info(f'Building visual_aux_decoder from: {config.visual_aux_model_path}')
        vis_aux_decoder = build_aux_decoder(config.visual_aux_model_path, device='cpu')
        vis_aux_hidden = _resolve_hidden_size(vis_aux_decoder)
        visual_latent_proj = nn.Sequential(
            nn.Linear(base_hidden_size, base_hidden_size),
            nn.GELU(),
            nn.Linear(base_hidden_size, vis_aux_hidden),
            nn.LayerNorm(vis_aux_hidden),
        )
        model.add_module('_latent_cot_visual_aux_decoder', vis_aux_decoder)
        model.add_module('_latent_cot_visual_latent_proj', visual_latent_proj)

        from transformers import AutoTokenizer
        model._latent_cot_visual_aux_tokenizer = AutoTokenizer.from_pretrained(
            config.visual_aux_model_path, trust_remote_code=True)
    else:
        model._latent_cot_visual_aux_decoder = None
        model._latent_cot_visual_latent_proj = None
        model._latent_cot_visual_aux_tokenizer = None

    model._origin_forward_for_latent_cot = model.forward
    model.forward = MethodType(_latent_cot_forward, model)

    logger.info('Model patched for latent CoT training.')
    return model


def apply_latent_cot_freeze(model) -> None:
    """Apply latent CoT freeze settings. Must be called AFTER ms-swift's
    prepare_model (which resets requires_grad via model.requires_grad_(True))
    so that our freeze settings are not overridden."""
    config = getattr(model, '_latent_cot_config', None)
    if config is None:
        return

    if config.freeze_main_model:
        for name, param in model.named_parameters():
            if not name.startswith('_latent_cot_'):
                param.requires_grad = False

    if config.freeze_aux_decoder:
        aux = getattr(model, '_latent_cot_aux_decoder', None)
        if aux is not None:
            for param in aux.parameters():
                param.requires_grad = False

    if config.freeze_visual_aux_decoder:
        vis_aux = getattr(model, '_latent_cot_visual_aux_decoder', None)
        if vis_aux is not None:
            for param in vis_aux.parameters():
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flags = (f'freeze_main={config.freeze_main_model}, '
             f'freeze_aux={config.freeze_aux_decoder}, '
             f'freeze_vis_aux={config.freeze_visual_aux_decoder}')
    logger.info(
        f'Latent CoT freeze applied ({flags}). '
        f'Trainable: {trainable_params:,} / {total_params:,} '
        f'({100 * trainable_params / total_params:.1f}%)')


def load_latent_cot_weights(model, model_dir: str) -> None:
    """Load _latent_cot_* weights from a checkpoint directory.

    When loading a latent-CoT checkpoint via from_pretrained, the base model
    architecture (Qwen3VLForConditionalGeneration) does not contain
    _latent_cot_* modules, so those weights are silently dropped. This function
    restores them after patch_model_for_latent_cot has created the modules.
    """
    import json
    import os

    latent_state = {}

    index_path = os.path.join(model_dir, 'model.safetensors.index.json')
    single_path = os.path.join(model_dir, 'model.safetensors')

    if os.path.exists(index_path):
        from safetensors.torch import load_file
        with open(index_path) as f:
            weight_map = json.load(f).get('weight_map', {})
        shards: dict[str, list[str]] = {}
        for key, shard_file in weight_map.items():
            if key.startswith('_latent_cot_'):
                shards.setdefault(shard_file, []).append(key)
        for shard_file, keys in shards.items():
            shard_path = os.path.join(model_dir, shard_file)
            shard_weights = load_file(shard_path, device='cpu')
            for k in keys:
                if k in shard_weights:
                    latent_state[k] = shard_weights[k]
    elif os.path.exists(single_path):
        from safetensors.torch import load_file
        all_weights = load_file(single_path, device='cpu')
        latent_state = {k: v for k, v in all_weights.items()
                        if k.startswith('_latent_cot_')}

    if not latent_state:
        logger.info(f'No _latent_cot_* weights found in {model_dir}, using freshly initialised modules.')
        return

    vis_aux_keys = [k for k in latent_state if k.startswith('_latent_cot_visual_aux_decoder.') or k.startswith('_latent_cot_visual_latent_proj.')]
    if vis_aux_keys:
        logger.info(
            f'[LatentCoT] visual_aux_decoder weights: RESTORED from checkpoint (model_path={model_dir}); '
            f'{len(vis_aux_keys)} tensors.')
    else:
        logger.info(
            '[LatentCoT] visual_aux_decoder weights: NOT in checkpoint; '
            'using initialisation from LATENT_COT_VISUAL_AUX_MODEL_PATH.')

    from transformers.integrations import is_deepspeed_zero3_enabled
    if is_deepspeed_zero3_enabled():
        import deepspeed
        loaded_count = 0
        for name, param in model.named_parameters():
            if name in latent_state:
                with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        param.data.copy_(latent_state[name].to(param.device).to(param.dtype))
                loaded_count += 1
        logger.info(
            f'Restored {loaded_count}/{len(latent_state)} latent CoT weight tensors '
            f'from checkpoint (DeepSpeed zero3 mode).')
    else:
        missing, unexpected = model.load_state_dict(latent_state, strict=False)
        loaded_count = len(latent_state) - len(unexpected)
        logger.info(
            f'Restored {loaded_count} latent CoT weight tensors from checkpoint. '
            f'(missing in ckpt: {len(missing)}, unexpected: {len(unexpected)})')
        if unexpected:
            logger.warning(f'Unexpected keys when loading latent CoT weights: {unexpected}')


def _latent_cot_forward(
    self, input_ids=None, attention_mask=None, position_ids=None,
    labels=None, pixel_values=None, pixel_values_videos=None,
    image_grid_thw=None, video_grid_thw=None,
    think_steps=None, future_image_tokens=None,
    **kwargs,
):
    """Patched forward that computes latent CoT aux losses.

    Works in two modes depending on whether labels are present:
    - With labels (no --loss_type): model computes CE internally, we add aux losses.
    - Without labels (--loss_type latent_cot): we compute hidden states + aux losses,
      store them on outputs for the loss function to combine with CE loss.
    """
    config = self._latent_cot_config
    token_ids = self._latent_token_ids
    processor = getattr(self, '_processor_ref', None)
    use_orig = config.use_original_vocab

    if use_orig:
        pat = self._latent_pattern_ids
        lkw, pipe = pat['latent_keyword_id'], pat['pipe_id']
        has_latent = (input_ids is not None and lkw is not None
                      and (input_ids == lkw).any())
    else:
        has_latent = (input_ids is not None
                      and (input_ids == token_ids['latent_token_id']).any())

    if not has_latent:
        return self._origin_forward_for_latent_cot(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )

    need_hidden = ((think_steps is not None and
                    getattr(self, '_latent_cot_aux_decoder', None) is not None)
                   or (future_image_tokens is not None and
                       getattr(self, '_latent_cot_visual_aux_decoder', None) is not None))

    _vit_hook_handle = None
    self._captured_vit_embeds = None
    if (config.aux_visual_condition or config.visual_aux_visual_condition) and need_hidden:
        def _capture_vit_embeds_hook(module, args, kwargs):
            ie = kwargs.get('inputs_embeds')
            if ie is not None:
                self._captured_vit_embeds = ie.detach()
            return None
        _lm = getattr(getattr(self, 'model', None), 'language_model', None)
        if _lm is not None:
            _vit_hook_handle = _lm.register_forward_pre_hook(
                _capture_vit_embeds_hook, with_kwargs=True)

    outputs = self._origin_forward_for_latent_cot(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        output_hidden_states=need_hidden,
        **kwargs,
    )

    if _vit_hook_handle is not None:
        _vit_hook_handle.remove()

    dev = (outputs.loss.device if outputs.loss is not None
           else (input_ids.device if input_ids is not None else 'cuda'))
    student_ce_loss = outputs.loss if outputs.loss is not None else None

    explain_loss = torch.tensor(0.0, device=dev)
    visual_explain_loss = torch.tensor(0.0, device=dev)

    aux_decoder = getattr(self, '_latent_cot_aux_decoder', None)
    latent_proj = getattr(self, '_latent_cot_latent_proj', None)
    visual_aux_decoder = getattr(self, '_latent_cot_visual_aux_decoder', None)
    visual_latent_proj = getattr(self, '_latent_cot_visual_latent_proj', None)

    if need_hidden and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        batch_size = input_ids.size(0)
        last_hidden = outputs.hidden_states[-1]

        n_markers_list = None
        use_all_sub = use_orig and config.latent_use_all_subtokens
        separate_vis = use_orig and config.use_separate_visual_latent_tokens
        if use_orig:
            latent_lists = []
            if separate_vis:
                marker_comp_ids = self._latent_marker_component_ids
                pat = self._latent_pattern_ids
                vis_suffix_id = pat.get('vis_suffix_id')
                tokenizer = (processor.tokenizer if processor and hasattr(processor, 'tokenizer')
                            else getattr(processor, 'tokenizer', None))
                vis_latent_lists_sep = []
                vis_n_markers_list_sep = []
                for b in range(batch_size):
                    ids_list = input_ids[b].tolist()
                    text_kw = find_latent_positions_from_pattern(ids_list, lkw, pipe)
                    vis_kw = find_visual_latent_positions_from_pattern(
                        ids_list, lkw, pipe, vis_suffix_id) if vis_suffix_id is not None else []
                    stop_vis = set(vis_kw)
                    stop_txt = set(text_kw)
                    if use_all_sub:
                        text_block_start = len(ids_list)
                        if tokenizer is not None and vis_suffix_id is not None:
                            text_block_start = _find_text_latent_block_start(
                                ids_list, pipe, vis_suffix_id, tokenizer)
                        vis_pos_full = (
                            _expand_keyword_positions_with_stop(
                                ids_list, vis_kw, marker_comp_ids, stop_txt)
                            if vis_suffix_id is not None and vis_kw else [])
                        vis_pos = [p for p in vis_pos_full if p < text_block_start]
                        text_pos = _expand_keyword_positions_with_stop(
                            ids_list, text_kw, marker_comp_ids, vis_pos)
                        text_pos = [p for p in text_pos if p >= text_block_start]
                        latent_lists.append(text_pos)
                        vis_latent_lists_sep.append(vis_pos)
                        n_markers_list = (n_markers_list or [])
                        n_markers_list.append(len(text_kw))
                        vis_n_markers_list_sep.append(len(vis_kw))
                    else:
                        latent_lists.append(text_kw)
                        vis_latent_lists_sep.append(vis_kw)
                        n_markers_list = (n_markers_list or [])
                        n_markers_list.append(len(text_kw))
                        vis_n_markers_list_sep.append(len(vis_kw))
                self._latent_cot_vis_latent_lists = vis_latent_lists_sep
                self._latent_cot_vis_n_markers_list = vis_n_markers_list_sep
            elif use_all_sub:
                marker_comp_ids = self._latent_marker_component_ids
                n_markers_list = []
                for b in range(batch_size):
                    ids_list = input_ids[b].tolist()
                    positions, n_mk = find_latent_all_positions_from_pattern(
                        ids_list, lkw, pipe, marker_comp_ids)
                    latent_lists.append(positions)
                    n_markers_list.append(n_mk)
            else:
                for b in range(batch_size):
                    ids_list = input_ids[b].tolist()
                    positions = find_latent_positions_from_pattern(ids_list, lkw, pipe)
                    latent_lists.append(positions)
            logger.info_once(
                f'[LatentCoT] original_vocab mode, '
                f'all_subtokens={use_all_sub}, separate_visual={separate_vis}, '
                f'latent_keyword_id={lkw}, pipe_id={pipe}, '
                f'batch_size={batch_size}, '
                f'latent_positions_per_sample={[len(p) for p in latent_lists]}, '
                f'n_markers_per_sample={n_markers_list}, '
                f'think_steps={think_steps is not None}, '
                f'aux_decoder={aux_decoder is not None}')
        else:
            latent_id = token_ids['latent_token_id']
            latent_lists = []
            for b in range(batch_size):
                positions = (input_ids[b] == latent_id).nonzero(as_tuple=True)[0].tolist()
                latent_lists.append(positions)
            logger.info_once(
                f'[LatentCoT] single_token mode, '
                f'latent_token_id={latent_id}, '
                f'batch_size={batch_size}, '
                f'latent_positions_per_sample={[len(p) for p in latent_lists]}, '
                f'think_steps={think_steps is not None}, '
                f'aux_decoder={aux_decoder is not None}')

        txt_vis_cond = config.aux_visual_condition
        vis_vis_cond = config.visual_aux_visual_condition
        image_token_id = getattr(self.config, 'image_token_id', None)
        video_token_id = getattr(self.config, 'video_token_id', None)

        debug_aux_vit_holder = {}
        if not _latent_cot_debug_aux_vit_embed():
            debug_aux_vit_holder = None  # type: ignore

        student_embeds = None
        if txt_vis_cond or vis_vis_cond:
            student_embeds = self._captured_vit_embeds
            if student_embeds is not None:
                _img_tid = getattr(self.config, 'image_token_id', None)
                if _img_tid is not None and input_ids is not None:
                    _vis_mask = (input_ids[0] == _img_tid)
                    if _vis_mask.any():
                        _vit_std = student_embeds[0][_vis_mask].float().std(dim=0).mean().item()
                        logger.info_once(
                            f'[LatentCoT] Using ViT-injected embeddings for visual condition '
                            f'(shape={student_embeds.shape}, '
                            f'n_img_tokens={_vis_mask.sum().item()}, '
                            f'vit_embed_std_across_positions={_vit_std:.6f})')
                    else:
                        logger.info_once(
                            f'[LatentCoT] Using ViT-injected embeddings '
                            f'(shape={student_embeds.shape}, no image tokens)')
            else:
                embed_fn = (self.model.get_input_embeddings()
                            if hasattr(self, 'model')
                            else self.get_input_embeddings())
                student_embeds = embed_fn(input_ids)
                logger.warning_once(
                    '[LatentCoT] Hook failed to capture ViT embeddings, '
                    'falling back to placeholder embeddings')

        if aux_decoder is not None and think_steps is not None:
            tokenizer = (processor.tokenizer
                         if processor and hasattr(processor, 'tokenizer')
                         else processor)
            explainable_ids_list = _tokenize_think_steps(think_steps, tokenizer)

            explain_loss = compute_explain_loss(
                last_hidden, input_ids, latent_lists, explainable_ids_list,
                batch_size, aux_decoder, latent_proj, config.c_thought,
                student_embeds=student_embeds,
                use_visual_condition=txt_vis_cond,
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                n_markers_list=n_markers_list,
                debug_aux_vit_holder=debug_aux_vit_holder,
            )
            explain_loss = explain_loss * config.explain_loss_weight

        if visual_aux_decoder is not None and future_image_tokens is not None:
            vis_tokenizer = getattr(self, '_latent_cot_visual_aux_tokenizer', None)
            if vis_tokenizer is not None:
                visual_ids_list = _tokenize_visual_targets(
                    future_image_tokens, vis_tokenizer)

                vis_latent_lists = latent_lists
                vis_n_markers_list = n_markers_list
                if config.use_separate_visual_latent_tokens:
                    if use_orig and hasattr(self, '_latent_cot_vis_latent_lists'):
                        vis_latent_lists = self._latent_cot_vis_latent_lists
                        vis_n_markers_list = getattr(
                            self, '_latent_cot_vis_n_markers_list', None)
                    else:
                        vis_latent_lists = []
                        vis_n_markers_list = None
                        for b in range(batch_size):
                            if use_orig:
                                ids_list = input_ids[b].tolist()
                                if use_all_sub:
                                    marker_comp_ids = self._latent_marker_component_ids
                                    positions, n_mk = find_visual_latent_all_positions_from_pattern(
                                        ids_list, lkw, pipe, pat['vis_suffix_id'],
                                        marker_comp_ids)
                                    vis_latent_lists.append(positions)
                                    if vis_n_markers_list is None:
                                        vis_n_markers_list = []
                                    vis_n_markers_list.append(n_mk)
                                else:
                                    positions = find_visual_latent_positions_from_pattern(
                                        ids_list, lkw, pipe, pat['vis_suffix_id'])
                                    vis_latent_lists.append(positions)
                            else:
                                vis_latent_id = token_ids['latent_visual_token_id']
                                positions = (input_ids[b] == vis_latent_id).nonzero(
                                    as_tuple=True)[0].tolist()
                                vis_latent_lists.append(positions)

                    # Debug: decode latent spans for batch 0 to verify text vs visual
                    _tok = (processor.tokenizer if processor and hasattr(processor, 'tokenizer')
                            else getattr(processor, 'tokenizer', None))
                    if _tok is not None and batch_size > 0:
                        txt_pos = latent_lists[0]
                        vis_pos = vis_latent_lists[0]
                        txt_span = _debug_decode_latent_span(_tok, input_ids[0], txt_pos)
                        vis_span = _debug_decode_latent_span(_tok, input_ids[0], vis_pos)
                        n_txt = len(txt_pos)
                        n_vis = len(vis_pos)
                        n_txt_mk = n_markers_list[0] if n_markers_list else None
                        n_vis_mk = vis_n_markers_list[0] if vis_n_markers_list else None
                        logger.info_once(
                            f'[LatentCoT separate] text_aux: n_positions={n_txt}, '
                            f'n_markers={n_txt_mk}, span_decode=[{txt_span!r}]')
                        logger.info_once(
                            f'[LatentCoT separate] visual_aux: n_positions={n_vis}, '
                            f'n_markers={n_vis_mk}, span_decode=[{vis_span!r}]')

                visual_explain_loss = compute_visual_explain_loss(
                    last_hidden, input_ids, vis_latent_lists, visual_ids_list,
                    batch_size, visual_aux_decoder, visual_latent_proj,
                    config.c_thought_visual,
                    student_embeds=student_embeds,
                    use_visual_condition=vis_vis_cond,
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                    n_markers_list=vis_n_markers_list,
                    debug_aux_vit_holder=debug_aux_vit_holder,
                )
                visual_explain_loss = (visual_explain_loss
                                       * config.visual_explain_loss_weight)

        _flush_aux_vit_debug_compare(debug_aux_vit_holder)

    # Store aux losses on the model itself because accelerate's convert_to_fp32
    # reconstructs ModelOutput from declared fields only, dropping dynamic attrs.
    self._latent_cot_cache = {
        'student_ce_loss': student_ce_loss,
        'explain_loss': explain_loss,
        'visual_explain_loss': visual_explain_loss,
    }

    if student_ce_loss is not None:
        outputs.loss = student_ce_loss + explain_loss + visual_explain_loss
        logger.info_once(
            f'[LatentCoT] First loss breakdown: '
            f'student_ce={student_ce_loss.item():.4f}, '
            f'explain={explain_loss.item():.4f}, '
            f'visual_explain={visual_explain_loss.item():.4f}, '
            f'total={outputs.loss.item():.4f}')

    return outputs
