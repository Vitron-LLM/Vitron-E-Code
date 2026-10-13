"""Sample preprocessing: resolution buckets, resize / crop / pad, frame
selection, tensor conversion and the per-task sample builders."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torchvision import transforms

# Resolution buckets as (height, width).
IMAGE_BUCKETS = ((624, 352), (576, 432), (480, 480), (432, 576), (352, 624))
VIDEO_BUCKETS = ((480, 480), (352, 624), (624, 352))

# Frames of the static clip an image sample is presented as.
IMAGE_CLIP_FRAMES = 5

CLIP_SAMPLING_MODES = ("first", "center_clip", "random_clip", "uniform")
FRAME_SAMPLING_MODES = CLIP_SAMPLING_MODES + ("uniform_dup4",)

Bucket = Tuple[int, int]
Sample = Dict[str, Any]


def choose_bucket(source_size: Tuple[int, int], target_size: Tuple[int, int], buckets: Sequence[Bucket]) -> Bucket:
    """Bucket (height, width) whose aspect ratio is closest to the mean of the
    source and target ratios; the two sizes are (width, height)."""
    ratio = (source_size[0] / source_size[1] + target_size[0] / target_size[1]) / 2.0
    return min(buckets, key=lambda bucket: abs(bucket[1] / bucket[0] - ratio))


def _rgb(image: PIL.Image.Image) -> PIL.Image.Image:
    if not isinstance(image, PIL.Image.Image):
        raise TypeError(f"Expected a PIL image, got {type(image)}")
    return image.convert("RGB")


def resize_and_center_crop_image(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Scale an image to cover the target size and crop the center (source and target)."""
    width, height = image.size
    if target_height / height > target_width / width:
        resize_size = (target_height, int(width * target_height / height))
    else:
        resize_size = (int(height * target_width / width), target_width)
    resize_crop = transforms.Compose([
        transforms.Resize(resize_size, antialias=True),
        transforms.CenterCrop((target_height, target_width)),
    ])
    return resize_crop(image)


def resize_and_pad_image(
    image: Image.Image, target_height: int, target_width: int
) -> Image.Image:
    """Scale an image to fit the target size and pad it with a blurred extension
    of its edges (reference)."""
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    resize_width = max(1, round(width * scale))
    resize_height = max(1, round(height * scale))
    resized = transforms.Resize((resize_height, resize_width), antialias=True)(image)
    if resize_width == target_width and resize_height == target_height:
        return resized

    left = (target_width - resize_width) // 2
    top = (target_height - resize_height) // 2
    pad_right = target_width - resize_width - left
    pad_bottom = target_height - resize_height - top

    arr = np.array(resized.convert("RGB"))
    padded = np.pad(arr, ((top, pad_bottom), (left, pad_right), (0, 0)), mode="edge")
    canvas = Image.fromarray(padded)

    blur_radius = max(4.0, 0.03 * max(target_height, target_width))
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    canvas.paste(resized, (left, top))
    return canvas


def image_to_tensor(image: PIL.Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor [3, H, W] in [-1, 1]."""
    return TF.normalize(TF.to_tensor(image), [0.5], [0.5])


def video_to_tensor(frames: Sequence[PIL.Image.Image]) -> torch.Tensor:
    """PIL RGB frames -> float tensor [3, F, H, W] in [-1, 1]."""
    return torch.stack([image_to_tensor(frame) for frame in frames], dim=1)


def num_frames_for_views(num_views: int) -> int:
    """Frame count of the 3d23d layout: view 0 once, every other view 4 times."""
    return 4 * (num_views - 1) + 1


@dataclass(frozen=True)
class FrameSampling:
    """Which ``num_frames`` frames of a video form a sample. ``uniform_dup4`` is the
    3d23d layout; ``window`` is the number of consecutive frames its views span."""

    num_frames: int
    mode: str = "random_clip"
    window: Optional[int] = None

    def __post_init__(self):
        if self.num_frames < 1 or (self.num_frames - 1) % 4 != 0:
            raise ValueError(f"num_frames must be 4n+1 for the VAE, got {self.num_frames}")
        if self.mode not in FRAME_SAMPLING_MODES:
            raise ValueError(f"Unknown frame sampling mode '{self.mode}'. Expected one of {FRAME_SAMPLING_MODES}.")

    def indices(self, total_frames: int, rng=random) -> List[int]:
        """Frame indices for a video of ``total_frames`` frames; a video shorter
        than ``num_frames`` returns all of its frames (except ``uniform_dup4``)."""
        if total_frames <= 0:
            raise ValueError("Video has no frames.")
        if self.mode == "uniform_dup4":
            return self._dup4_indices(total_frames, rng)
        if total_frames < self.num_frames:
            return list(range(total_frames))
        if self.mode == "first":
            return list(range(self.num_frames))
        if self.mode == "center_clip":
            start = (total_frames - self.num_frames) // 2
            return list(range(start, start + self.num_frames))
        if self.mode == "random_clip":
            start = rng.randint(0, total_frames - self.num_frames)
            return list(range(start, start + self.num_frames))
        if self.num_frames == 1:
            return [total_frames // 2]
        return [round(i * (total_frames - 1) / (self.num_frames - 1)) for i in range(self.num_frames)]

    def _dup4_indices(self, total_frames: int, rng) -> List[int]:
        num_views = (self.num_frames - 1) // 4 + 1
        span = self.window or self.num_frames
        if total_frames > span:
            base = rng.randint(0, total_frames - span)
            length = span
        else:
            base, length = 0, total_frames
        if length < num_views:
            offsets = list(range(length)) + [length - 1] * (num_views - length)
        elif num_views == 1:
            offsets = [length // 2]
        else:
            offsets = [round(i * (length - 1) / (num_views - 1)) for i in range(num_views)]
        indices = [base + offsets[0]]
        for offset in offsets[1:]:
            indices.extend([base + offset] * 4)
        return indices


def sample_aligned_frames(source: Sequence, target: Sequence, sampling: FrameSampling, rng=random):
    """Pick the same frames from two frame-aligned videos; a clip shorter than
    ``sampling.num_frames`` is padded by repeating its last frame."""
    if not source or not target:
        raise ValueError("source or target video has no frames.")
    if len(source) != len(target):
        raise ValueError(
            f"source and target are not frame-aligned: {len(source)} vs {len(target)} frames."
        )
    indices = sampling.indices(len(source), rng)
    source_frames = [source[i] for i in indices]
    target_frames = [target[i] for i in indices]
    padding = sampling.num_frames - len(source_frames)
    source_frames.extend([source_frames[-1]] * padding)
    target_frames.extend([target_frames[-1]] * padding)
    return source_frames, target_frames


def semantic_inputs(source_frames: Sequence[PIL.Image.Image]):
    """Text-encoder inputs of a source clip: 4 / 6 / 8 uniformly sampled frames
    (Qwen2.5-VL) and the first frame as uint8 [H, W, 3] (SigLIP)."""
    total = min(len(source_frames), 161)
    count = 8 if total >= 24 * 4 else 6 if total >= 24 * 3 else 4
    indices = np.linspace(0, total - 1, count).astype(int)
    source_sampled_frames = [source_frames[i] for i in indices]
    source_first_frame = np.array(source_frames[0], dtype=np.uint8)
    return source_sampled_frames, source_first_frame


def build_image_sample(
    text: str,
    source: PIL.Image.Image,
    target: PIL.Image.Image,
    *,
    reference: Optional[PIL.Image.Image] = None,
) -> Sample:
    """i2i sample, or ti_i2i when ``reference`` is given: the image pair as a
    static clip of ``IMAGE_CLIP_FRAMES`` identical frames."""
    height, width = choose_bucket(source.size, target.size, IMAGE_BUCKETS)
    target_pixels = image_to_tensor(resize_and_center_crop_image(_rgb(target), height, width))
    sample: Sample = {
        "text": text,
        "target_pixels": target_pixels.unsqueeze(1).expand(-1, IMAGE_CLIP_FRAMES, -1, -1),
        "source_sequence": [resize_and_center_crop_image(_rgb(source), height, width)] * IMAGE_CLIP_FRAMES,
    }
    if reference is not None:
        sample["reference_image"] = resize_and_pad_image(_rgb(reference), height, width)
    return sample


def build_video_sample(
    text: str,
    source: Sequence[PIL.Image.Image],
    target: Sequence[PIL.Image.Image],
    sampling: FrameSampling,
    *,
    reference: Optional[PIL.Image.Image] = None,
    rng=random,
) -> Sample:
    """v2v / 3d23d sample, or tiv2v when ``reference`` is given, from two
    frame-aligned videos."""
    source_frames, target_frames = sample_aligned_frames(source, target, sampling, rng)
    height, width = choose_bucket(source_frames[0].size, target_frames[0].size, VIDEO_BUCKETS)
    source_frames = [resize_and_center_crop_image(_rgb(frame), height, width) for frame in source_frames]
    target_frames = [resize_and_center_crop_image(_rgb(frame), height, width) for frame in target_frames]
    source_sampled_frames, source_first_frame = semantic_inputs(source_frames)
    sample: Sample = {
        "text": text,
        "target_pixels": video_to_tensor(target_frames),
        "source_sequence": source_frames,
        "source_sampled_frames": source_sampled_frames,
        "source_first_frame": source_first_frame,
    }
    if reference is not None:
        sample["reference_image"] = resize_and_pad_image(_rgb(reference), height, width)
    return sample
