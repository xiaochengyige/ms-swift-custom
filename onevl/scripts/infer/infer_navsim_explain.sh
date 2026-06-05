#!/bin/bash
set -x
# ============================================================
# OneVL Latent CoT - inference WITH auxiliary-decoder explanations.
#
# Uses the `qwen3_vl_latent_cot_explain` template, which runs one forward pass
# for latent hidden states and decodes the text / visual auxiliary explanations,
# appending them to the response behind <onevl_text_explain> /
# <onevl_visual_explain> delimiters.
#
# The auxiliary decoders are rebuilt + restored from the checkpoint by the
# plugin loader, so LATENT_COT_AUX_MODEL_PATH / LATENT_COT_VISUAL_AUX_MODEL_PATH
# must point to the architecture sources used at training time.
# ============================================================

# source /path/to/.venv/bin/activate
ONEVL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWIFT_ROOT="$(cd "${ONEVL_DIR}/../ms-swift-4.2.0" && pwd)"
export PYTHONPATH="${SWIFT_ROOT}:${PYTHONPATH}"

# ---------- Paths (edit these) ----------
MODEL_PATH=${MODEL_PATH:-"${SWIFT_ROOT}/outputs/navsim/onevl_stage2_vis4_txt2/vX-xxxx/checkpoint-XXXX"}
VAL_DATASET=${VAL_DATASET:-"${ONEVL_DIR}/data/navsim_test.jsonl"}
RESULT_PATH=${RESULT_PATH:-"${MODEL_PATH}/infer_results/onevl_navsim_explain.jsonl"}
AUX_MODEL_PATH=${AUX_MODEL_PATH:-"Qwen/Qwen3-VL-4B-Instruct"}
VISUAL_AUX_MODEL_PATH=${VISUAL_AUX_MODEL_PATH:-"models/visual_aux_decoder/qwen3_vl_visual_aux_decoder_ad_512/hf_ckpt"}

# ---------- Latent prefix (same as training) ----------
NUM_LATENT=${NUM_LATENT:-2}
NUM_LATENT_VIS=${NUM_LATENT_VIS:-4}
ANSWER_PREFIX=${ANSWER_PREFIX:-"["}
build_latent_prefix() {
    local nl=$1 nlv=$2 ap=$3 vis="" txt=""
    if [ "${nlv}" -gt 0 ]; then
        vis="<|start-latent-vis|>"
        for _ in $(seq 1 "${nlv}"); do vis="${vis}<|latent-vis|>"; done
        vis="${vis}<|end-latent-vis|>"
    fi
    txt="<|start-latent|>"
    for _ in $(seq 1 "${nl}"); do txt="${txt}<|latent|>"; done
    txt="${txt}<|end-latent|>"
    printf '%s%s<answer>%s' "${vis}" "${txt}" "${ap}"
}
RESPONSE_PREFIX=$(build_latent_prefix "${NUM_LATENT}" "${NUM_LATENT_VIS}" "${ANSWER_PREFIX}")

# ---------- Aux decoder build + explain config ----------
export LATENT_COT_AUX_MODEL_PATH="${AUX_MODEL_PATH}"
export LATENT_COT_VISUAL_AUX_MODEL_PATH="${VISUAL_AUX_MODEL_PATH}"
export LATENT_COT_USE_ORIGINAL_VOCAB=true
export LATENT_COT_USE_SEPARATE_VISUAL_LATENT_TOKENS=true
# Explanation switches consumed by the explain template
export LATENT_COT_DECODER_EXPLAIN=true
export LATENT_COT_VISUAL_DECODER_EXPLAIN=true
export LATENT_COT_AUX_VISUAL_CONDITION=true
export LATENT_COT_VISUAL_AUX_VISUAL_CONDITION=true
export LATENT_COT_MAX_EXPLAIN_TOKENS=1024
export LATENT_COT_MAX_VISUAL_TOKENS=2560

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
swift infer \
    --external_plugins "${ONEVL_DIR}/register.py" \
    --model "${MODEL_PATH}" \
    --model_type qwen3_vl_latent_cot \
    --template qwen3_vl_latent_cot_explain \
    --infer_backend pt \
    --val_dataset "${VAL_DATASET}" \
    --result_path "${RESULT_PATH}" \
    --response_prefix "${RESPONSE_PREFIX}" \
    --max_new_tokens 1024 \
    --temperature 0 \
    --max_batch_size 1
