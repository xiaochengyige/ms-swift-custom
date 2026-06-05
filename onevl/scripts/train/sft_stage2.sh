#!/bin/bash
set -x
# ============================================================
# OneVL Latent CoT - Stage 2 (joint fine-tuning), NON-INVASIVE.
#
# Stage 2 unfreezes everything and jointly fine-tunes the main model together
# with the auxiliary decoders. MODEL_PATH should be the Stage 1 checkpoint
# (which already contains the trained _latent_cot_* weights; they are restored
# by load_latent_cot_weights inside the plugin loader).
# ============================================================

# ---------- Environment ----------
# source /path/to/.venv/bin/activate
ONEVL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWIFT_ROOT="$(cd "${ONEVL_DIR}/../ms-swift-4.2.0" && pwd)"
export PYTHONPATH="${SWIFT_ROOT}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3

# ---------- Distributed settings ----------
nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NNODES=${WORKER_NUM:-1}
NODE_RANK=${ROLE_INDEX:-0}
MASTER_ADDR=${WORKER_0_HOST:-127.0.0.1}
MASTER_PORT=${WORKER_0_PORT:-29500}

# ---------- Paths (edit these) ----------
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-4B-Instruct"}              # ideally the Stage 1 checkpoint
AUX_MODEL_PATH=${AUX_MODEL_PATH:-"Qwen/Qwen3-VL-4B-Instruct"}
VISUAL_AUX_MODEL_PATH=${VISUAL_AUX_MODEL_PATH:-"models/visual_aux_decoder/qwen3_vl_visual_aux_decoder_ad_512/hf_ckpt"}
DATASET_PATH=${DATASET_PATH:-"${ONEVL_DIR}/data/navsim_vis4_text2_demo.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SWIFT_ROOT}/outputs/navsim/onevl_stage2_vis4_txt2"}

# ---------- Latent CoT configuration ----------
export LATENT_COT_C_THOUGHT=2
export LATENT_COT_C_THOUGHT_VISUAL=4
export LATENT_COT_AUX_MODEL_PATH="${AUX_MODEL_PATH}"
export LATENT_COT_VISUAL_AUX_MODEL_PATH="${VISUAL_AUX_MODEL_PATH}"
export LATENT_COT_EXPLAIN_LOSS_WEIGHT=1.0
export LATENT_COT_VISUAL_EXPLAIN_LOSS_WEIGHT=0.1
export LATENT_COT_AUX_VISUAL_CONDITION=true
export LATENT_COT_VISUAL_AUX_VISUAL_CONDITION=true
export LATENT_COT_USE_SEPARATE_VISUAL_LATENT_TOKENS=true
export LATENT_COT_FREEZE_VISUAL_AUX_DECODER=false
export LATENT_COT_FREEZE_AUX_DECODER=false
export LATENT_COT_FREEZE_MAIN_MODEL=false          # <-- Stage 2: train everything jointly
export LATENT_COT_LATENT_CE_LOSS=true
export LATENT_COT_LATENT_USE_ALL_SUBTOKENS=true
export LATENT_COT_USE_ORIGINAL_VOCAB=true

mkdir -p "$(dirname "${OUTPUT_DIR}")"

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((nproc_per_node-1))) \
NPROC_PER_NODE=$nproc_per_node \
NNODES=$NNODES \
NODE_RANK=$NODE_RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
swift sft \
    --external_plugins "${ONEVL_DIR}/register.py" \
    --model "${MODEL_PATH}" \
    --model_type qwen3_vl_latent_cot \
    --template qwen3_vl_latent_cot \
    --train_type full \
    --dataset "${DATASET_PATH}" \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4 \
    --loss_type latent_cot \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 2 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_steps 100 \
    --weight_decay 0.05 \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --dataloader_num_workers 4 \
    --output_dir "${OUTPUT_DIR}" \
    --gradient_checkpointing true \
    --deepspeed zero2
