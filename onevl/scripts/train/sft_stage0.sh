#!/bin/bash
set -x
# ============================================================
# OneVL Latent CoT - Stage 0 (answer warmup), NON-INVASIVE reproduction.
#
# Stage 0 trains the base Qwen3-VL on latent-formatted data WITHOUT the
# auxiliary decoders: model_type is the stock `qwen3_vl`, so the model is not
# patched and `--loss_type latent_cot` degenerates to plain per-token CE
# (LatentCoTLoss falls back when model._latent_cot_cache is absent).
# Goal: teach the model to emit the answer that follows the latent prefix.
#
# Difference vs the original OneVL fork: instead of editing swift source code,
# we load the plugin via `--external_plugins`.
# ============================================================

# ---------- Environment ----------
# Activate your venv if needed, e.g.: source /path/to/.venv/bin/activate
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
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-4B-Instruct"}
DATASET_PATH=${DATASET_PATH:-"${ONEVL_DIR}/data/navsim_vis4_text2_demo.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SWIFT_ROOT}/outputs/navsim/onevl_stage0_vis4_txt2"}

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
    --model_type qwen3_vl \
    --train_type full \
    --dataset "${DATASET_PATH}" \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --learning_rate 4e-5 \
    --loss_type latent_cot \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 1 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_steps 100 \
    --weight_decay 0.05 \
    --freeze_aligner False \
    --freeze_llm False \
    --freeze_vit False \
    --dataloader_num_workers 8 \
    --output_dir "${OUTPUT_DIR}" \
    --deepspeed zero2
