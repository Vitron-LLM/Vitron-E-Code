#!/usr/bin/env bash
# Geometry post-training with the settings of our run (one node, 8 GPUs):
# LoRA fine-tuning continued from a multi-task checkpoint with depth and
# camera-pose losses, modality tokens frozen.
# Needs VIST3A (scripts/setup_vist3a.sh) and requirements-3d.txt.
# Run from the repository root: bash scripts/train/geo_posttrain.sh [extra arguments]
# Set the paths below, or pass them in the environment.

MODEL_PATH="${MODEL_PATH:-ckpts/omniweaving}"
DATA_CONFIG="${DATA_CONFIG:-configs/data.geo.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/univedit_geo}"

# Multi-task checkpoint to start from, and the checkpoint whose modality tokens are used
INIT_CHECKPOINT="${INIT_CHECKPOINT:-ckpts/univedit/geo3d}"
TOKEN_CHECKPOINT="${TOKEN_CHECKPOINT:-$INIT_CHECKPOINT}"
# Optional: resume a run of this stage
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

# 3D decoder
DECODER_CKPT="${DECODER_CKPT:-ckpts/univedit/3d_decoder/stitched_model.pth}"
ANYSPLAT_PATH="${ANYSPLAT_PATH:-ckpts/anysplat}"
export VIST3A_ROOT="${VIST3A_ROOT:-third_party/VIST3A}"

if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    CHECKPOINT_ARGS="--resume_from_checkpoint $RESUME_FROM_CHECKPOINT"
else
    CHECKPOINT_ARGS="--pretrained_lora_path $INIT_CHECKPOINT"
fi

# Log the training metrics to Weights & Biases: true or false
USE_WANDB="${USE_WANDB:-false}"

export PYTHONPATH="src:${PYTHONPATH}"

torchrun --standalone --nproc_per_node 8 -m univedit.training.geometry_trainer \
    --pretrained_model_root "$MODEL_PATH" \
    --data_config "$DATA_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps 1000 \
    --save_interval 50 \
    --log_interval 1 \
    --dtype bf16 \
    --sp_size 1 \
    --dp_replicate 4 \
    --enable_fsdp true \
    --enable_gradient_checkpointing true \
    --enable_text_encoder_fsdp false \
    --enable_text_encoder_gradient_checkpointing false \
    --use_lora true \
    --lora_r 32 \
    --lora_alpha 32 \
    --use_muon true \
    --learning_rate 1e-5 \
    --lr_scheduler constant \
    --warmup_steps 0 \
    --weight_decay 0.05 \
    --train_timestep_shift 3 \
    --train_modality_token_embeddings false \
    --geo_modality_token_path "$TOKEN_CHECKPOINT" \
    --decoder_checkpoint_path "$DECODER_CKPT" \
    --anysplat_path "$ANYSPLAT_PATH" \
    --geo_loss_weight 0.08 \
    --geo_task_weight_scale 3d23d:2.5,v2v:0.7,tiv2v:0.7,i2i:2.6,ti_i2i:2.6 \
    --geo_task_filter i2i,ti_i2i,v2v,tiv2v,3d23d \
    --geo_camera_loss_weight 10.3 \
    --geo_camera_rot_weight 0.9 \
    --geo_camera_trans_beta 0.001 \
    --geo_camera_trans_weight_scale 12.7 \
    --geo_camera_trans_anchor_weight 1.3 \
    --geo_camera_trans_anchor_beta 0.001 \
    --geo_depth_scale_beta 0.02 \
    --geo_depth_scale_weight_scale 2.27 \
    --geo_rollout_steps_low 40 \
    --geo_rollout_steps_high 50 \
    --geo_rollout_grad_steps 2 \
    --geo_extractor_resolution 448 \
    --geo_extractor_max_frames 17 \
    --geo_frame_stride 4 \
    --use_wandb "$USE_WANDB" \
    --wandb_project univedit \
    $CHECKPOINT_ARGS \
    "$@"
