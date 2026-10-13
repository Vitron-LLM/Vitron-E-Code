#!/usr/bin/env python3
"""Batch inference for UniVEdit on its five editing tasks: i2i and ti_i2i (image,
optionally with a reference image), v2v and tiv2v (video, optionally with a
reference image) and 3d23d (multi-view video, optionally exported as 3D
Gaussians). Input formats, checkpoint layout and output layout: see README.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import einops
import imageio
import loguru
import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from PIL import Image
from torch import distributed as dist
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torchvision import transforms

from hyvideo.commons import auto_offload_model
from hyvideo.commons.infer_state import initialize_infer_state
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline

from univedit.data.preprocessing import IMAGE_CLIP_FRAMES, resize_and_center_crop_image, resize_and_pad_image


BATCH_TASKS = ("i2i", "ti_i2i", "v2v", "tiv2v", "3d23d")
IMAGE_TASKS = ("i2i", "ti_i2i")
VIDEO_TASKS = ("v2v", "tiv2v")

IMAGE_EDIT_BUCKETS = (
    (352, 624),
    (432, 576),
    (480, 480),
    (576, 432),
    (624, 352),
)

VIDEO_EDIT_360P_BUCKETS = {
    "1:1": (480, 480),
    "16:9": (352, 624),
    "9:16": (624, 352),
}

MAX_INFER_VIDEO_LENGTH = 81
THREED_MAX_SIDE = 448  # longest side a 3d23d source video is resized to
DEFAULT_FPS = 12  # output fps when the source video does not report one

# Square resolutions of the 3D decoder inputs: VAE frames and feed-forward frames.
DECODER_VAE_RESOLUTION = 512
DECODER_FEEDFORWARD_RESOLUTION = 448


parallel_dims = None


def ensure_parallel_state():
    global parallel_dims
    if parallel_dims is None:
        parallel_dims = initialize_parallel_state(sp=int(os.environ.get("WORLD_SIZE", "1")))
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    return parallel_dims


def rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def rank0_log(message: str, level: str = "INFO") -> None:
    if rank0():
        loguru.logger.log(level, message)


def str_to_bool(value: Any) -> bool:
    """Convert a CLI value to bool: true/false, 1/0, yes/no, on/off; None gives True."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    value = str(value).lower().strip()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


# ── Result I/O: tensor <-> PIL / file conversion ──────────────────────────────

def save_first_frame(video, path: Path) -> None:
    """Save the first frame of a [0, 1] image / video tensor as an image file."""
    if video.ndim == 5:
        frame = video[0, :, 0]
    elif video.ndim == 4:
        frame = video[0]
    elif video.ndim == 3:
        frame = video
    else:
        raise ValueError(f"Unsupported output tensor shape for image save: {tuple(video.shape)}")
    frame = (frame * 255).clamp(0, 255).to(torch.uint8)
    frame = einops.rearrange(frame, "c h w -> h w c").cpu().numpy()
    Image.fromarray(frame).save(path)


def save_video(video, path: Path, fps: int) -> None:
    """Save a [0, 1] video tensor ([C, F, H, W] or batched) as a video file."""
    if video.ndim == 5:
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Unsupported output tensor shape for video save: {tuple(video.shape)}")
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, "c f h w -> f h w c").cpu().numpy()
    imageio.mimwrite(path, vid, fps=fps)


def input_video_fps(video_path: str, fallback: int) -> int:
    """Rounded average fps of a video file, or ``fallback`` if it cannot be read."""
    try:
        from decord import VideoReader

        vr = VideoReader(video_path)
        fps = round(vr.get_avg_fps())
        del vr
        return fps or fallback
    except Exception:
        return fallback


def tensor_video_to_pil_frames(video: torch.Tensor) -> list[Image.Image]:
    """Convert a [0, 1] tensor video to PIL frames."""
    if video.ndim == 5:
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Unsupported output tensor shape for PIL conversion: {tuple(video.shape)}")
    frames = einops.rearrange(video, "c f h w -> f h w c")
    frames = (frames * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


def center_crop_to_square(frame: Image.Image, target_size: int) -> Image.Image:
    """Center-crop a frame to a square of side ``target_size``, resizing if needed."""
    width, height = frame.size
    if width == height == target_size:
        return frame
    crop_size = min(width, height, target_size)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    frame = frame.crop((left, top, left + crop_size, top + crop_size))
    if crop_size != target_size:
        frame = frame.resize((target_size, target_size), Image.LANCZOS)
    return frame


def pil_frames_to_video_tensor(frames: list[Image.Image], target_h: int, target_w: int) -> torch.Tensor:
    """Convert PIL frames to [1, C, T, H, W] in [-1, 1]."""
    tensors = []
    for frame in frames:
        if frame.size != (target_w, target_h):
            frame = frame.resize((target_w, target_h), Image.LANCZOS)
        arr = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        tensors.append(tensor * 2.0 - 1.0)
    return torch.stack(tensors, dim=1).unsqueeze(0)


def checkpoint_root(path: str | None) -> Path | None:
    if not path:
        return None
    normalized = Path(path)
    return normalized.parent if normalized.name == "transformer" else normalized


def transformer_checkpoint_path(path: str | None) -> Path | None:
    if not path:
        return None
    root = checkpoint_root(path)
    if Path(path).name == "transformer":
        return Path(path)
    candidate = root / "transformer"
    return candidate if candidate.is_dir() else None


def default_lora_path(path: str | None) -> Path | None:
    root = checkpoint_root(path)
    if root is None:
        return None
    candidate = root / "lora" / "default"
    if candidate.is_dir():
        return candidate
    if (root / "pytorch_lora_weights.safetensors").is_file():
        return root
    return None


def load_checkpoint_to_transformer(pipe, checkpoint_path: str | None) -> bool:
    transformer_path = transformer_checkpoint_path(checkpoint_path)
    if transformer_path is None:
        rank0_log(f"No transformer checkpoint found under {checkpoint_path}; skipping transformer load.", "WARNING")
        return False

    rank0_log(f"Loading transformer checkpoint from {transformer_path}")
    model_state_dict = get_model_state_dict(pipe.transformer)
    dcp.load(state_dict={"model": model_state_dict}, checkpoint_id=str(transformer_path))
    rank0_log("Transformer checkpoint loaded")
    return True


def load_modality_tokens(pipe, token_checkpoint_path: str | None) -> bool:
    if not token_checkpoint_path:
        rank0_log("No modality token path provided; skipping modality token load.", "WARNING")
        return False
    loaded = pipe.load_modality_token_embeddings(token_checkpoint_path, strict=True)
    if loaded:
        rank0_log(f"Modality token embeddings loaded from {token_checkpoint_path}")
    return loaded


def load_lora_adapter(pipe, lora_path: str | None) -> bool:
    if not lora_path:
        rank0_log("No LoRA adapter path provided/found; skipping LoRA load.", "WARNING")
        return False
    if not os.path.isdir(lora_path):
        raise FileNotFoundError(f"LoRA adapter path does not exist: {lora_path}")
    rank0_log(f"Loading LoRA adapter from {lora_path}")
    pipe.transformer.load_lora_adapter(
        pretrained_model_name_or_path_or_dict=lora_path,
        prefix=None,
        adapter_name="default",
        weight_name="pytorch_lora_weights.safetensors",
        use_safetensors=True,
        hotswap=False,
    )
    rank0_log("LoRA adapter loaded")
    return True


def build_pipeline(args):
    ensure_parallel_state()
    infer_state = initialize_infer_state(args)

    if args.sparse_attn and args.use_sageattn:
        raise ValueError("sparse_attn and use_sageattn cannot both be enabled.")

    if args.dtype == "bf16":
        transformer_dtype = torch.bfloat16
    elif args.dtype == "fp32":
        transformer_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    enable_group_offloading = args.group_offloading
    if enable_group_offloading is None:
        offloading_config = HunyuanVideo_1_5_Pipeline.get_offloading_config()
        enable_group_offloading = offloading_config["enable_group_offloading"]

    device = torch.device("cpu") if args.offloading else torch.device("cuda")
    transformer_init_device = torch.device("cpu") if enable_group_offloading else device

    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_dtype=transformer_dtype,
        device=device,
        transformer_init_device=transformer_init_device,
        pipeline_config="omniweaving",
    )
    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=args.offloading,
        enable_group_offloading=enable_group_offloading,
        overlap_group_offloading=args.overlap_group_offloading,
    )

    if args.checkpoint_path:
        load_checkpoint_to_transformer(pipe, args.checkpoint_path)

    modality_path = args.modality_token_path or args.checkpoint_path
    load_modality_tokens(pipe, modality_path)

    lora_path = args.lora_path or default_lora_path(args.checkpoint_path)
    load_lora_adapter(pipe, str(lora_path) if lora_path else None)

    return pipe


def safe_stem(value: Any, fallback: str) -> str:
    text = str(value or fallback)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text[:120] or fallback


def resolve_path(root: Path, value: str | None) -> str | None:
    """Path of a metadata field; a relative path is taken from ``root``."""
    if not value:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def replace_reference_placeholder(text: str) -> str:
    """Replace the ``[v*]`` placeholder of a prompt with "the reference image"."""
    text = re.sub(
        r"\b(?:a|an|the)\s*\[\s*v\s*\*\s*\]",
        "the reference image",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\[\s*v\s*\*\s*\]", "the reference image", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def image_bucket_size(source_image: Image.Image) -> tuple[int, int, str]:
    """Image-edit bucket closest to the aspect ratio of the source image, as
    (height, width, name)."""
    ratio = source_image.width / source_image.height
    bucket_width, bucket_height = min(
        IMAGE_EDIT_BUCKETS,
        key=lambda size: abs((size[0] / size[1]) - ratio),
    )
    return bucket_height, bucket_width, f"{bucket_width}x{bucket_height}"


def video_bucket_size(video_path: str) -> tuple[int, int, str]:
    """Video-edit bucket closest to the aspect ratio of a video, as (height, width, name)."""
    from decord import VideoReader

    vr = VideoReader(video_path)
    if len(vr) <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    frame = vr.get_batch([0]).asnumpy()[0]
    height, width = frame.shape[:2]
    ratio = width / height
    bucket_key, (target_height, target_width) = min(
        VIDEO_EDIT_360P_BUCKETS.items(),
        key=lambda item: abs((item[1][1] / item[1][0]) - ratio),
    )
    del vr
    return target_height, target_width, bucket_key


def video_longest_side_size(video_path: str, max_side: int, multiple: int = 16) -> tuple[int, int]:
    """Target (height, width) for 3d23d: the longest side scaled to ``max_side``,
    aspect ratio kept, both sides rounded to a multiple of ``multiple``."""
    from decord import VideoReader

    vr = VideoReader(video_path)
    if len(vr) <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    frame = vr.get_batch([0]).asnumpy()[0]
    height, width = frame.shape[:2]
    del vr
    scale = max_side / max(height, width)
    target_height = max(multiple, round(height * scale / multiple) * multiple)
    target_width = max(multiple, round(width * scale / multiple) * multiple)
    return target_height, target_width


def build_frame_indices(total_frames: int, num_frames: int, mode: str) -> list[int]:
    """Indices of the ``num_frames`` frames to read from a video; a shorter video
    repeats its last frame."""
    if total_frames <= 0:
        raise ValueError("Video has no frames.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if total_frames >= num_frames:
        if mode == "first":
            return list(range(num_frames))
        if mode == "center_clip":
            start = (total_frames - num_frames) // 2
            return list(range(start, start + num_frames))
        if mode == "uniform":
            if num_frames == 1:
                return [total_frames // 2]
            return [round(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
        raise ValueError(f"Unsupported frame_sample_mode: {mode}")

    indices = list(range(total_frames))
    indices.extend([total_frames - 1] * (num_frames - total_frames))
    return indices


def encode_frames_to_latents(pipe, frames: torch.Tensor, *, scale: bool) -> torch.Tensor:
    """VAE latents [1, C, F_lat, H_lat, W_lat] of frames [F, 3, H, W] in [0, 1].
    ``scale``: apply the VAE shift / scaling factor (image tasks)."""
    pixel_values = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(frames)
    pixel_values = pixel_values.unsqueeze(0).transpose(1, 2).to(pipe.execution_device)
    with (
        torch.no_grad(),
        torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True),
        auto_offload_model(pipe.vae, pipe.execution_device, enabled=pipe.enable_offloading),
        pipe.vae.memory_efficient_context(),
    ):
        latents = pipe.vae.encode(pixel_values).latent_dist.mode()
        if scale:
            if hasattr(pipe.vae.config, "shift_factor") and pipe.vae.config.shift_factor:
                latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            else:
                latents = latents * pipe.vae.config.scaling_factor
    return latents


def encode_video_to_latents(
    pipe,
    video_path: str,
    *,
    num_frames: int,
    target_height: int,
    target_width: int,
    frame_sample_mode: str,
    save_resized_path: Path | None = None,
    save_fps: int | None = None,
    dup_views: int = 0,
    frame_indices: list[int] | None = None,
):
    """Encode a source video into VAE latents. ``save_resized_path``: file the resized
    clip is also saved to; ``dup_views``: number of views in the 3d23d frame layout;
    ``frame_indices``: the frames to read instead of ``frame_sample_mode``."""
    from decord import VideoReader

    rank0_log(
        f"Encoding source video: {video_path} -> {target_width}x{target_height}, frames={num_frames}"
    )
    video_reader = VideoReader(video_path)
    if frame_indices is None:
        frame_indices = build_frame_indices(len(video_reader), num_frames, frame_sample_mode)
    video_images = video_reader.get_batch(frame_indices)

    pixel_values = torch.from_numpy(video_images.asnumpy())
    pixel_values = pixel_values.permute(0, 3, 1, 2).contiguous() / 255.0

    height, width = pixel_values.size(-2), pixel_values.size(-1)
    if target_height / height > target_width / width:
        resize_size = target_height, int(width * target_height / height)
    else:
        resize_size = int(height * target_width / width), target_width

    resize_crop = transforms.Compose(
        [
            transforms.Resize(resize_size, antialias=True),
            transforms.CenterCrop((target_height, target_width)),
        ]
    )
    pixel_values = resize_crop(pixel_values)  # [F, C, H, W] in [0, 1]

    if dup_views > 0:
        # 3d23d frame layout: view 0 once, every other view 4 times.
        views = pixel_values[:dup_views]
        pixel_values = torch.cat(
            [views[0:1]] + [views[i:i + 1].repeat(4, 1, 1, 1) for i in range(1, views.shape[0])],
            dim=0,
        )
        rank0_log(
            f"[dup] {views.shape[0]} views -> {pixel_values.shape[0]} pixel frames "
            f"({views.shape[0]} latent frames, each carrying ONE clean view)"
        )

    if save_resized_path is not None:
        save_video(einops.rearrange(pixel_values, "f c h w -> c f h w"), save_resized_path, fps=save_fps or DEFAULT_FPS)

    return encode_frames_to_latents(pipe, pixel_values, scale=False)


def filter_dark_gaussians(gaussians, brightness_thresh: float = 0.02):
    """Zero the opacity of very dark Gaussians."""
    dc_rgb = gaussians.harmonics[..., 0].float()
    brightness = dc_rgb.abs().mean(dim=-1)
    mask = (brightness > brightness_thresh).to(gaussians.opacities.dtype)

    while mask.ndim < gaussians.opacities.ndim:
        mask = mask.unsqueeze(-1)
    gaussians.opacities = gaussians.opacities * mask
    return gaussians


def collect_jobs(metadata_path: str, plain_task: str, reference_task: str | None, tasks) -> list[dict[str, Any]]:
    """Jobs of one metadata jsonl: ``reference_task`` for a record with a
    ``reference`` image, else ``plain_task``; only the tasks in ``tasks``."""
    metadata_path = Path(metadata_path)
    root = metadata_path.parent
    source_key = "source_image" if plain_task in IMAGE_TASKS else "source_video"
    jobs = []
    for rec in load_jsonl(metadata_path):
        task = reference_task if reference_task and rec.get("reference") else plain_task
        if task not in tasks:
            continue
        if not rec.get("source"):
            raise ValueError(f"{metadata_path}: record without a source: {rec}")
        prompt = rec.get("text")
        if not (isinstance(prompt, str) and prompt.strip()):
            rank0_log(f"Skipping a record of {metadata_path} without text: {rec}", "WARNING")
            continue
        if task == "ti_i2i":
            prompt = replace_reference_placeholder(prompt)
        index = rec.get("index", len(jobs))
        jobs.append(
            {
                "task": task,
                "index": index,
                "sample_id": rec.get("id") or index,
                "prompt": prompt,
                source_key: resolve_path(root, rec["source"]),
                "reference_image": resolve_path(root, rec.get("reference")) if task == reference_task else None,
                "raw_record": rec,
            }
        )
    return jobs


def sample_dir_for_job(output_dir: Path, job: dict[str, Any]) -> Path:
    stem = f"{int(job['index']):06d}_{safe_stem(job['sample_id'], str(job['index']))}"
    return output_dir / job["task"] / stem


def materialize_sample_dir(sample_dir: Path, job: dict[str, Any]) -> None:
    """Create the sample directory and write instruction.txt and sample_metadata.json."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    instruction_path = sample_dir / "instruction.txt"
    instruction_path.write_text(job["prompt"].strip() + "\n", encoding="utf-8")

    sample_manifest = {
        "task": job["task"],
        "index": job["index"],
        "sample_id": job["sample_id"],
        "instruction": job["prompt"],
        "source_image": str(sample_dir / "source_image.png") if job.get("source_image") else None,
        "source_video": str(sample_dir / "source_video.mp4") if job.get("source_video") else None,
        "reference_image": str(sample_dir / "reference_image.png") if job.get("reference_image") else None,
        "original_source_image": job.get("source_image"),
        "original_source_video": job.get("source_video"),
        "original_reference_image": job.get("reference_image"),
        "raw_record": job.get("raw_record"),
    }
    with (sample_dir / "sample_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(sample_manifest, f, indent=2, ensure_ascii=False, default=str)


def _load_3d_decoder(args):
    """Load the stitched 3D decoder (VAE latents -> AnySplat). Returns None if disabled."""
    if not args.enable_3d_decoder:
        return None
    from univedit.decoder3d import load_decoder3d

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    rank0_log("Loading 3D decoder …")
    decoder = load_decoder3d(
        checkpoint_path=args.decoder_checkpoint_path,
        anysplat_path=args.anysplat_path,
        device=device,
    )
    rank0_log("3D decoder loaded")
    return decoder


def select_3d_views(video_path: str, num_views: int) -> list[int]:
    """Frame indices of the views of a 3d23d source video: uniform over its
    first 4 (num_views - 1) + 1 frames, or over the whole video if it is shorter."""
    from decord import VideoReader

    span = min(len(VideoReader(video_path)), 1 + 4 * (num_views - 1))
    return build_frame_indices(span, num_views, "uniform")


def write_first_frames(video_path: str, length: int, out_path: Path, fps: float) -> Path:
    """Write the first ``length`` frames of a video to a lossless mp4."""
    from decord import VideoReader

    frames = VideoReader(video_path).get_batch(list(range(length))).asnumpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        str(out_path), list(frames), fps=fps, codec="libx264rgb", pixelformat="rgb24",
        macro_block_size=1, quality=None, ffmpeg_params=["-crf", "0", "-preset", "veryfast"],
    )
    return out_path


def run_3d23d_job(
    pipe,
    decoder,
    args,
    job: dict[str, Any],
    sample_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run 3d23d editing on ``--dup_views`` views of the source video; with a
    decoder, also reconstruct 3D Gaussians from the edited frames."""
    target_height, target_width = video_longest_side_size(job["source_video"], THREED_MAX_SIDE)
    fps = args.fps or input_video_fps(job["source_video"], DEFAULT_FPS)

    k_views = args.dup_views
    video_length = 1 + 4 * (k_views - 1)
    view_indices = select_3d_views(job["source_video"], k_views)
    source_video_latents = encode_video_to_latents(
        pipe,
        job["source_video"],
        num_frames=k_views,
        target_height=target_height,
        target_width=target_width,
        frame_sample_mode="uniform",
        save_resized_path=(sample_dir / "source_video.mp4") if rank0() else None,
        save_fps=fps,
        dup_views=k_views,
        frame_indices=view_indices,
    )
    # The text encoder reads its frames from the same span as the views.
    from decord import VideoReader

    semantic_video = job["source_video"]
    if len(VideoReader(job["source_video"])) > video_length:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        name = "source_window.mp4" if rank == 0 else f"source_window.rank{rank}.mp4"
        semantic_video = str(write_first_frames(job["source_video"], video_length, sample_dir / name, fps))
        rank0_log(f"[3d23d] views from the first {video_length} frames of the source video")
    kwargs = {
        "prompt": job["prompt"],
        "negative_prompt": "",
        "aspect_ratio": f"{target_width}:{target_height}",
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "video_length": video_length,
        "seed": args.seed + int(job["index"]),
        "output_type": "pt",
        "return_dict": True,
        "deepstack": args.deepstack,
        "setclip": args.setclip,
        "task_type": "3d23d",
        "source_video": [semantic_video],
        "source_video_latents": source_video_latents,
    }
    with torch.no_grad():
        out = pipe(**kwargs)

    edited_video = out.videos
    # Saved preview: one frame per view.
    edited_video_full = edited_video
    take = torch.tensor([4 * i for i in range(k_views)], device=edited_video.device)
    edited_video = edited_video.index_select(-3, take)  # F axis of [..., C, F, H, W]
    rank0_log(f"[dup] decimated preview output to {edited_video.shape[-3]} views")
    if rank0():
        save_video(edited_video, sample_dir / "edited_video.mp4", fps=fps)

    result = {"video_length": video_length, "height": target_height, "width": target_width}

    if decoder is None:
        if rank0():
            shutil.copy2(sample_dir / "edited_video.mp4", output_path)
        return result

    ff_res = DECODER_FEEDFORWARD_RESOLUTION
    vae_res = DECODER_VAE_RESOLUTION

    # 3D decoder inputs: square crops of all edited frames.
    edited_frames = tensor_video_to_pil_frames(edited_video_full)
    stitch_indices = list(range(len(edited_frames)))
    cropped_frames = [center_crop_to_square(frame, ff_res) for frame in edited_frames]

    if rank0():
        square_preview = (pil_frames_to_video_tensor(cropped_frames, ff_res, ff_res) + 1.0) / 2.0
        save_video(square_preview, sample_dir / "decoder_square_input.mp4", fps=fps)
        rank0_log(
            "3D decoder input: "
            f"edited_frames={len(edited_frames)}, stitch_frames={len(cropped_frames)}, "
            f"vae_res={vae_res}, feedforward_res={ff_res}, indices={stitch_indices}"
        )

    video_tensor = pil_frames_to_video_tensor(cropped_frames, vae_res, vae_res)
    video_tensor = video_tensor.to(device=pipe.execution_device, dtype=torch.float32)

    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        auto_offload_model(pipe.vae, pipe.execution_device, enabled=pipe.enable_offloading),
        pipe.vae.memory_efficient_context(),
    ):
        decoder_latents = pipe.vae.encode(video_tensor).latent_dist.mode()

    feedforward = pil_frames_to_video_tensor(cropped_frames, ff_res, ff_res)
    feedforward = feedforward.to(device=decoder.device, dtype=torch.bfloat16)
    rank0_log(f"3D decoder VAE latents shape: {tuple(decoder_latents.shape)}")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        dec_out = decoder.forward_with_latent(
            decoder_latents.to(decoder.device),
            feedforward_image=feedforward,
            train=False,
        )

    if rank0():
        from univedit.decoder3d import import_vist3a

        vist3a = import_vist3a()
        save_interpolated_video = vist3a.save_interpolated_video
        export_ply = vist3a.export_ply

        gaussians = filter_dark_gaussians(dec_out.gaussians)
        pred_pose = dec_out.pred_context_pose

        torch.save(
            {
                "extrinsic": pred_pose["extrinsic"].cpu().float(),
                "intrinsic": pred_pose["intrinsic"].cpu().float(),
                "stitch_frame_indices": stitch_indices,
            },
            sample_dir / "pred_camera_poses.pt",
        )
        poses_json = {
            "num_views": int(pred_pose["extrinsic"].shape[1]),
            "stitch_frame_indices": stitch_indices,
            "extrinsic_c2w": [pred_pose["extrinsic"][0, i].cpu().float().tolist() for i in range(pred_pose["extrinsic"].shape[1])],
            "intrinsic": [pred_pose["intrinsic"][0, i].cpu().float().tolist() for i in range(pred_pose["intrinsic"].shape[1])],
        }
        with (sample_dir / "pred_camera_poses.json").open("w", encoding="utf-8") as f:
            json.dump(poses_json, f, indent=2)

        gs_path, _ = save_interpolated_video(
            pred_pose["extrinsic"],
            pred_pose["intrinsic"],
            1,
            ff_res,
            ff_res,
            gaussians,
            str(sample_dir),
            decoder.stitched_3d_model.decoder,
        )
        shutil.copy2(gs_path, output_path)

        export_ply(
            gaussians.means[0],
            gaussians.scales[0],
            gaussians.rotations[0],
            gaussians.harmonics[0],
            gaussians.opacities[0],
            sample_dir / "gaussians.ply",
            save_sh_dc_only=True,
        )

    return result


def run_image_job(pipe, args, job: dict[str, Any], sample_dir: Path, output_path: Path) -> dict[str, Any]:
    source_image = Image.open(job["source_image"]).convert("RGB")
    target_height, target_width, bucket = image_bucket_size(source_image)

    resized = resize_and_center_crop_image(source_image, target_height, target_width)
    if rank0():
        resized.save(sample_dir / "source_image.png")
    # The image is edited as a static clip of IMAGE_CLIP_FRAMES identical frames.
    clip = transforms.ToTensor()(resized).unsqueeze(0).repeat(IMAGE_CLIP_FRAMES, 1, 1, 1)
    source_image_latents = encode_frames_to_latents(pipe, clip, scale=True)

    kwargs = {
        "prompt": job["prompt"],
        "negative_prompt": "",
        "aspect_ratio": f"{target_width}:{target_height}",
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "video_length": IMAGE_CLIP_FRAMES,
        "seed": args.seed + int(job["index"]),
        "output_type": "pt",
        "return_dict": True,
        "deepstack": args.deepstack,
        "setclip": args.setclip,
        "task_type": job["task"],
        "source_image": source_image,
        "source_image_latents": source_image_latents,
        "target_height": target_height,
        "target_width": target_width,
    }
    if job["task"] == "ti_i2i":
        reference_image = Image.open(job["reference_image"]).convert("RGB")
        kwargs["reference_image"] = reference_image
        if rank0():
            resize_and_pad_image(reference_image, target_height, target_width).save(
                sample_dir / "reference_image.png"
            )

    with torch.no_grad():
        out = pipe(**kwargs)
    if rank0():
        save_first_frame(out.videos, output_path)
    return {"bucket": bucket, "height": target_height, "width": target_width}


def run_video_job(
    pipe,
    args,
    job: dict[str, Any],
    sample_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run v2v/tiv2v editing (tiv2v adds a reference image)."""
    target_height, target_width, bucket = video_bucket_size(job["source_video"])
    fps = args.fps or input_video_fps(job["source_video"], DEFAULT_FPS)
    source_video_latents = encode_video_to_latents(
        pipe,
        job["source_video"],
        num_frames=args.video_length,
        target_height=target_height,
        target_width=target_width,
        frame_sample_mode=args.frame_sample_mode,
        save_resized_path=(sample_dir / "source_video.mp4") if rank0() else None,
        save_fps=fps,
    )
    kwargs = {
        "prompt": job["prompt"],
        "negative_prompt": "",
        "aspect_ratio": f"{target_width}:{target_height}",
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "video_length": args.video_length,
        "seed": args.seed + int(job["index"]),
        "output_type": "pt",
        "return_dict": True,
        "deepstack": args.deepstack,
        "setclip": args.setclip,
        "task_type": job["task"],
        "source_video": [job["source_video"]],
        "source_video_latents": source_video_latents,
    }
    if job["task"] == "tiv2v":
        kwargs["reference_image"] = job["reference_image"]
        if rank0():
            reference_image = Image.open(job["reference_image"]).convert("RGB")
            resize_and_pad_image(reference_image, target_height, target_width).save(
                sample_dir / "reference_image.png"
            )

    with torch.no_grad():
        out = pipe(**kwargs)
    if rank0():
        save_video(out.videos, output_path, fps=fps)
    return {"bucket": bucket, "height": target_height, "width": target_width}


def resolve_tasks(args) -> list[str]:
    """Tasks to run: those of --tasks, or every task whose input is given."""
    inputs = {
        IMAGE_TASKS: ("--image_metadata", args.image_metadata),
        VIDEO_TASKS: ("--video_metadata", args.video_metadata),
        ("3d23d",): ("--threed_metadata", args.threed_metadata),
    }
    if args.tasks is None:
        tasks = [task for group, (_, value) in inputs.items() if value for task in group]
        if not tasks:
            raise ValueError(
                "No inputs given: pass --image_metadata (i2i/ti_i2i), --video_metadata "
                "(v2v/tiv2v) and/or --threed_metadata (3d23d)."
            )
        return tasks
    for group, (flag, value) in inputs.items():
        requested = [task for task in args.tasks if task in group]
        if requested and not value:
            raise ValueError(f"--tasks {' '.join(requested)} requires {flag}.")
    return list(args.tasks)


def check_3d_decoder_args(args) -> None:
    if not args.enable_3d_decoder:
        return
    missing = [
        flag
        for flag, value in (
            ("--decoder_checkpoint_path", args.decoder_checkpoint_path),
            ("--anysplat_path", args.anysplat_path),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"3d23d with --enable_3d_decoder requires {' and '.join(missing)} "
            "(or pass --enable_3d_decoder false to only produce the edited video)."
        )


def check_view_count(source_video: str, num_views: int) -> None:
    """Raise if a 3d23d source video has fewer frames than views to sample."""
    from decord import VideoReader

    total = len(VideoReader(source_video))
    if total < num_views:
        raise ValueError(
            f"{source_video} has {total} frames, fewer than --dup_views {num_views}: the last frame would be "
            f"repeated as several identical views. Use --dup_views {total} or less, or a longer video."
        )


def run_batch(args) -> None:
    jobs = []
    if "i2i" in args.tasks or "ti_i2i" in args.tasks:
        jobs.extend(collect_jobs(args.image_metadata, "i2i", "ti_i2i", args.tasks))
    if "v2v" in args.tasks or "tiv2v" in args.tasks:
        jobs.extend(collect_jobs(args.video_metadata, "v2v", "tiv2v", args.tasks))
    if "3d23d" in args.tasks:
        jobs.extend(collect_jobs(args.threed_metadata, "3d23d", None, args.tasks))

    if any(job["task"] == "3d23d" for job in jobs):
        check_3d_decoder_args(args)
        for source_video in sorted({job["source_video"] for job in jobs if job["task"] == "3d23d"}):
            check_view_count(source_video, args.dup_views)

    if rank0():
        counts = Counter(job["task"] for job in jobs)
        rank0_log(f"Collected {len(jobs)} jobs: {dict(counts)}")

    pipe = build_pipeline(args)
    decoder = _load_3d_decoder(args) if any(job["task"] == "3d23d" for job in jobs) else None

    output_dir = Path(args.output_dir)
    result_records = []

    for ordinal, job in enumerate(jobs, start=1):
        sample_dir = sample_dir_for_job(output_dir, job)
        if job["task"] in ("i2i", "ti_i2i"):
            output_suffix = ".png"
        elif job["task"] == "3d23d":
            output_suffix = ".ply.mp4" if decoder is not None else ".mp4"
        else:
            output_suffix = ".mp4"
        output_path = sample_dir / f"result{output_suffix}"
        if rank0():
            materialize_sample_dir(sample_dir, job)
            rank0_log(f"[{ordinal}/{len(jobs)}] {job['task']} {job['sample_id']} -> {output_path}")

        skipped = output_path.exists() and not args.overwrite
        bucket_info = {}
        if not skipped:
            if job["task"] in ("i2i", "ti_i2i"):
                bucket_info = run_image_job(pipe, args, job, sample_dir, output_path)
            elif job["task"] == "3d23d":
                bucket_info = run_3d23d_job(pipe, decoder, args, job, sample_dir, output_path)
            else:
                bucket_info = run_video_job(pipe, args, job, sample_dir, output_path)
        elif rank0():
            rank0_log(f"Output exists, skipping: {output_path}", "WARNING")

        if dist.is_initialized():
            dist.barrier()

        if rank0():
            result_records.append(
                {
                    "task": job["task"],
                    "index": job["index"],
                    "sample_id": job["sample_id"],
                    "prompt": job["prompt"],
                    "sample_dir": str(sample_dir),
                    "output_path": str(output_path),
                    "skipped": skipped,
                    **bucket_info,
                    "source_image": job.get("source_image"),
                    "source_video": job.get("source_video"),
                    "reference_image": job.get("reference_image"),
                    "raw_record": job.get("raw_record"),
                }
            )
        torch.cuda.empty_cache()

    if rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = output_dir / "batch_metadata.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for rec in result_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        run_config = {
            "timestamp": dt.datetime.now().isoformat(),
            "args": vars(args),
            "counts": dict(Counter(rec["task"] for rec in result_records)),
        }
        with (output_dir / "batch_config.json").open("w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2, ensure_ascii=False, default=str)
        rank0_log(f"Wrote metadata to {meta_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        "univedit.inference.batch_infer",
        description="Batch UniVEdit inference for i2i / ti_i2i / v2v / tiv2v / 3d23d. "
                    "See the README, section 1, for input formats and the output layout.",
    )
    # ── model / checkpoint ────────────────────────────────────────────────
    parser.add_argument("--model_path", type=str, required=True,
                        help="Base model directory.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Training checkpoint dir (checkpoint-N/ or its transformer/ subdir). Loads "
                             "transformer/ (DCP) if present, the modality tokens, and lora/default "
                             "unless --lora_path is given.")
    parser.add_argument("--modality_token_path", type=str, default=None,
                        help="Where to load modality_token_embeddings from (default: --checkpoint_path).")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Explicit LoRA adapter dir (default: <checkpoint_path>/lora/default).")

    # ── inputs / outputs ──────────────────────────────────────────────────
    parser.add_argument("--tasks", nargs="+", default=None, choices=list(BATCH_TASKS),
                        help="Tasks to run (default: every task whose input below is given).")
    parser.add_argument("--image_metadata", type=str, default=None,
                        help="i2i/ti_i2i jsonl (see the README, section 1).")
    parser.add_argument("--video_metadata", type=str, default=None,
                        help="v2v/tiv2v jsonl (see the README, section 1).")
    parser.add_argument("--threed_metadata", type=str, default=None,
                        help="3d23d jsonl (see the README, section 1).")
    parser.add_argument("--output_dir", type=str, default="./outputs/univedit_batch")
    parser.add_argument("--overwrite", type=str_to_bool, nargs="?", const=True, default=False)

    # ── sampling ──────────────────────────────────────────────────────────
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--flow_shift", type=float, default=5.0,
                        help="Timestep shift of the sampling schedule.")
    parser.add_argument("--seed", type=int, default=123,
                        help="Per-sample seed is seed + index.")
    parser.add_argument("--fps", type=int, default=None,
                        help=f"Output fps (default: the source video's fps, else {DEFAULT_FPS}).")

    # ── video tasks ───────────────────────────────────────────────────────
    parser.add_argument("--video_length", type=int, default=65,
                        help="v2v/tiv2v only: number of frames (4n+1, <= 81). 3d23d uses "
                             "--dup_views instead.")
    parser.add_argument("--frame_sample_mode", choices=["first", "center_clip", "uniform"], default="first",
                        help="v2v/tiv2v only: how --video_length frames are taken from the source video.")

    # ── 3d23d ─────────────────────────────────────────────────────────────
    parser.add_argument("--dup_views", type=int, default=26,
                        help="3d23d only: sample K views uniformly from the source video and repeat each "
                             "4x before the VAE. The VAE compresses 4 consecutive frames into "
                             "one latent, which is lossy for wide-baseline multi-view input; with 4 "
                             "identical frames per group it is lossless. K views -> 1+4(K-1) pixel "
                             "frames -> K latent frames; the output is decimated back to K views. "
                             "Default 26 matches training.")

    # ── runtime / acceleration ────────────────────────────────────────────
    parser.add_argument("--sparse_attn", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--offloading", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--group_offloading", type=str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--overlap_group_offloading", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--use_sageattn", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--sage_blocks_range", type=str, default="0-53")
    parser.add_argument("--enable_torch_compile", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--enable_cache", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--cache_type", type=str, default="deepcache")
    parser.add_argument("--no_cache_block_id", type=str, default="53")
    parser.add_argument("--cache_start_step", type=int, default=11)
    parser.add_argument("--cache_end_step", type=int, default=45)
    parser.add_argument("--total_steps", type=int, default=50)
    parser.add_argument("--cache_step_interval", type=int, default=4)
    parser.add_argument("--use_fp8_gemm", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--quant_type", type=str, default="fp8-per-token-sgl")
    parser.add_argument("--include_patterns", type=str, default="double_blocks")
    parser.add_argument("--deepstack", type=int, nargs="*", default=[8, 16, 24])
    parser.add_argument("--setclip", type=str_to_bool, nargs="?", const=True, default=True)

    # ── 3D decoder (AnySplat reconstruction, 3d23d only) ──────────────────
    parser.add_argument("--enable_3d_decoder", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Reconstruct a 3DGS scene from the edited 3d23d video (stitched VAE-latent "
                             "decoder + AnySplat). Requires --decoder_checkpoint_path and --anysplat_path "
                             "and the VIST3A code. Off by default: only the edited video is produced.")
    parser.add_argument("--decoder_checkpoint_path", type=str, default=os.environ.get("DECODER_CKPT"),
                        help="Stitched decoder checkpoint (stitched_model_epoch_*.pth); "
                             "default: $DECODER_CKPT.")
    parser.add_argument("--anysplat_path", type=str, default=os.environ.get("ANYSPLAT_PATH"),
                        help="Local AnySplat checkpoint directory; default: $ANYSPLAT_PATH.")

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.tasks = resolve_tasks(args)
    if args.dup_views < 2:
        raise ValueError(f"--dup_views must be at least 2, got {args.dup_views}")
    if args.video_length > MAX_INFER_VIDEO_LENGTH:
        raise ValueError(f"--video_length must be <= {MAX_INFER_VIDEO_LENGTH}, got {args.video_length}")
    if (args.video_length - 1) % 4 != 0:
        raise ValueError(f"--video_length must be 4n+1, got {args.video_length}")
    run_batch(args)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
