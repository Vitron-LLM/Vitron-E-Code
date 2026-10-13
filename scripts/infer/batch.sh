#!/usr/bin/env bash
# Batch inference for i2i / ti_i2i / v2v / tiv2v / 3d23d.
# Run from the repository root: bash scripts/infer/batch.sh [extra arguments]
# Set the paths below, or pass them in the environment.

MODEL_PATH="${MODEL_PATH:-ckpts/omniweaving}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-ckpts/univedit/geo3d}"
IMAGE_METADATA="${IMAGE_METADATA:-examples/inference/image_metadata.jsonl}"
VIDEO_METADATA="${VIDEO_METADATA:-examples/inference/video_metadata.jsonl}"
THREED_METADATA="${THREED_METADATA:-examples/inference/threed_metadata.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/univedit_batch}"

# Only used with --enable_3d_decoder true (3D Gaussian export for 3d23d)
DECODER_CKPT="${DECODER_CKPT:-ckpts/univedit/3d_decoder/stitched_model.pth}"
ANYSPLAT_PATH="${ANYSPLAT_PATH:-ckpts/anysplat}"
export VIST3A_ROOT="${VIST3A_ROOT:-third_party/VIST3A}"

export PYTHONPATH="src:${PYTHONPATH}"

torchrun --standalone --nproc_per_node 1 -m univedit.inference.batch_infer \
    --model_path "$MODEL_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --image_metadata "$IMAGE_METADATA" \
    --video_metadata "$VIDEO_METADATA" \
    --threed_metadata "$THREED_METADATA" \
    --output_dir "$OUTPUT_DIR" \
    --num_inference_steps 50 \
    --guidance_scale 6.0 \
    --flow_shift 5.0 \
    --enable_3d_decoder false \
    --decoder_checkpoint_path "$DECODER_CKPT" \
    --anysplat_path "$ANYSPLAT_PATH" \
    "$@"
