#!/usr/bin/env bash
# Multi-task LoRA fine-tuning with the settings of our run (one node, 8 GPUs).
# Run from the repository root: bash scripts/train/multitask.sh [extra arguments]
# Set the paths below, or pass them in the environment.

MODEL_PATH="${MODEL_PATH:-ckpts/omniweaving}"
DATA_CONFIG="${DATA_CONFIG:-configs/data.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/univedit_multitask}"

# Optional: resume a run (weights, optimizer, step and data progress)
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

CHECKPOINT_ARGS=""
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    CHECKPOINT_ARGS="--resume_from_checkpoint $RESUME_FROM_CHECKPOINT"
fi

# Log the training metrics to Weights & Biases: true or false
USE_WANDB="${USE_WANDB:-false}"

export PYTHONPATH="src:${PYTHONPATH}"

torchrun --standalone --nproc_per_node 8 -m univedit.training.trainer \
    --pretrained_model_root "$MODEL_PATH" \
    --data_config "$DATA_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps 10000 \
    --save_interval 100 \
    --log_interval 1 \
    --dtype bf16 \
    --sp_size 1 \
    --dp_replicate 4 \
    --enable_fsdp true \
    --enable_gradient_checkpointing true \
    --enable_text_encoder_fsdp true \
    --enable_text_encoder_gradient_checkpointing true \
    --use_lora true \
    --lora_r 32 \
    --lora_alpha 32 \
    --use_muon true \
    --learning_rate 5e-5 \
    --lr_scheduler constant \
    --warmup_steps 0 \
    --weight_decay 0.05 \
    --train_timestep_shift 3 \
    --train_modality_token_embeddings true \
    --modality_token_learning_rate 1e-5 \
    --modality_token_weight_decay 0.01 \
    --modality_token_warmup_steps 50 \
    --modality_token_lr_scheduler constant_with_warmup \
    --use_wandb "$USE_WANDB" \
    --wandb_project univedit \
    $CHECKPOINT_ARGS \
    "$@"
