#!/bin/bash
set -x
# ============================================================
# OneVL Latent CoT - trajectory / answer-only inference via `swift infer`.
#
# Non-invasive: the latent prefix is injected through `--response_prefix`, and
# the custom model_type/template are provided by `--external_plugins`.
# This is the fast "prefill" path (no auxiliary decoders involved).
# ============================================================

# source /path/to/.venv/bin/activate
ONEVL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWIFT_ROOT="$(cd "${ONEVL_DIR}/../ms-swift-4.2.0" && pwd)"
export PYTHONPATH="${SWIFT_ROOT}:${PYTHONPATH}"

# ---------- Paths (edit these) ----------
MODEL_PATH=${MODEL_PATH:-"${SWIFT_ROOT}/outputs/navsim/onevl_stage2_vis4_txt2/vX-xxxx/checkpoint-XXXX"}
VAL_DATASET=${VAL_DATASET:-"${ONEVL_DIR}/data/navsim_test.jsonl"}
RESULT_PATH=${RESULT_PATH:-"${MODEL_PATH}/infer_results/onevl_navsim.jsonl"}

# ---------- OneVL latent prefix ----------
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
echo "RESPONSE_PREFIX = ${RESPONSE_PREFIX}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
swift infer \
    --external_plugins "${ONEVL_DIR}/register.py" \
    --model "${MODEL_PATH}" \
    --model_type qwen3_vl_latent_cot \
    --template qwen3_vl_latent_cot \
    --infer_backend pt \
    --val_dataset "${VAL_DATASET}" \
    --result_path "${RESULT_PATH}" \
    --response_prefix "${RESPONSE_PREFIX}" \
    --max_new_tokens 1024 \
    --temperature 0 \
    --max_batch_size 8
