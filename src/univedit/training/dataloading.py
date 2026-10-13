# Modified by the UniVEdit authors from Tencent-Hunyuan/OmniWeaving (HunyuanVideo-1.5).
# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""Data loading for the trainer: collation, normalisation of the PIL values of
a batch, DataLoader construction for the datasets of the data config (with the
dataset shuffle seed), the per-dataset batch settings and the multi-loader sampler.
"""

import queue
import random
import threading
from typing import Any, Dict, Optional

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
import torchvision.transforms as T
from loguru import logger
from PIL import Image

from univedit.data.config import get_dataset_spec
from univedit.training.config import TrainingConfig


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def training_collate_fn(batch):
    """Collate samples into a batch: tensors are stacked, numbers become 1-D tensors,
    other values stay lists."""
    result = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values)
        else:
            result[key] = values
    return result


# ---------------------------------------------------------------------------
# Batch values: PIL images / frame sequences as per-sample lists, semantic inputs
# ---------------------------------------------------------------------------

def normalize_batched_pil_images(images, batch_size, field_name):
    """Batch value of single images as one [image, ...] list per sample (None stays None)."""
    if images is None:
        return None
    if isinstance(images, PIL.Image.Image):
        return [[images]]
    if len(images) == batch_size:
        return [
            image_list if isinstance(image_list, list) else [image_list]
            for image_list in images
        ]
    if batch_size == 1:
        return [images if isinstance(images, list) else [images]]
    raise ValueError(
        f"{field_name} must contain one image/list per batch sample, "
        f"got {len(images)} entries for batch size {batch_size}"
    )


def normalize_batched_pil_sequences(videos, batch_size, field_name):
    """Batch value of frame sequences as one list of PIL frames per sample (None stays None)."""
    if videos is None:
        return None

    def is_frame_list(value):
        return (
            isinstance(value, (list, tuple))
            and len(value) > 0
            and all(isinstance(frame, PIL.Image.Image) for frame in value)
        )

    if is_frame_list(videos) and batch_size == 1:
        return [list(videos)]

    if isinstance(videos, (list, tuple)) and len(videos) == batch_size:
        normalized = []
        for idx, sample_frames in enumerate(videos):
            if not is_frame_list(sample_frames):
                raise ValueError(
                    f"{field_name}[{idx}] must be a non-empty list of PIL frames, "
                    f"got {type(sample_frames)}"
                )
            normalized.append(list(sample_frames))
        return normalized

    raise ValueError(
        f"{field_name} must contain one frame-list per batch sample, "
        f"got {len(videos)} entries for batch size {batch_size}"
    )


def get_semantic_images_np(input_videos, nframes=8):
    """Return the first frames of the videos (uint8 ``[B, H, W, C]``) and ``nframes`` uniformly
    sampled PIL frames per video. A video is a file path, a decord reader or a list of PIL frames."""
    from decord import VideoReader as _VideoReader

    if input_videos is None or len(input_videos) == 0:
        return None, None

    first_frames = []
    sampled_frames = []
    for video in input_videos:
        if (
            isinstance(video, (list, tuple))
            and len(video) > 0
            and all(isinstance(frame, PIL.Image.Image) for frame in video)
        ):
            max_frames = min(len(video), 161)
            sample_indices = np.linspace(0, max_frames - 1, nframes).astype(int)
            sampled_frames.append([video[i].convert("RGB") for i in sample_indices])
            first_frames.append(np.array(video[0].convert("RGB"), dtype=np.uint8))
            continue

        if isinstance(video, str):
            vr = _VideoReader(video)
        else:
            vr = video
        if len(vr) == 0:
            raise ValueError(f"Video {video} has no frames")

        # Uniformly sampled frames
        max_frames = min(len(vr), 161)
        sample_indices = np.linspace(0, max_frames - 1, nframes).astype(int)
        frames = vr.get_batch(sample_indices)
        if hasattr(frames, 'asnumpy'):
            frames = frames.asnumpy()
        else:
            frames = frames.numpy()
        sampled_frames.append([Image.fromarray(f) for f in frames])

        # First frame
        first_frame = vr.get_batch([0])
        if hasattr(first_frame, 'asnumpy'):
            first_frame_np = first_frame.asnumpy()
        else:
            first_frame_np = first_frame.numpy()
        first_frames.append(first_frame_np[0])
        del vr

    first_images_np = np.stack(first_frames, axis=0).astype(np.uint8)
    return first_images_np, sampled_frames


def normalize_sampled_frames(source_sampled_frames, batch_size):
    """Batch value of sampled frames as one frame list per sample, or None if it is not well-formed."""
    if source_sampled_frames is None:
        return None
    if (
        isinstance(source_sampled_frames, (list, tuple))
        and len(source_sampled_frames) > 0
        and all(isinstance(frame, PIL.Image.Image) for frame in source_sampled_frames)
    ):
        return [list(source_sampled_frames)] if batch_size == 1 else None
    if isinstance(source_sampled_frames, (list, tuple)) and len(source_sampled_frames) == batch_size:
        normalized = []
        for frames in source_sampled_frames:
            if not (
                isinstance(frames, (list, tuple))
                and len(frames) > 0
                and all(isinstance(frame, PIL.Image.Image) for frame in frames)
            ):
                return None
            normalized.append(list(frames))
        return normalized
    return None


def normalize_semantic_images_np(source_first_frame, batch_size):
    """Batch value of first frames as a uint8 array [B, H, W, 3], or None if it is not well-formed."""
    if source_first_frame is None:
        return None
    if isinstance(source_first_frame, np.ndarray):
        if source_first_frame.ndim == 3 and batch_size == 1:
            return source_first_frame[None, ...]
        if source_first_frame.ndim == 4 and source_first_frame.shape[0] == batch_size:
            return source_first_frame
        return None
    if isinstance(source_first_frame, (list, tuple)) and len(source_first_frame) == batch_size:
        arrays = []
        for image in source_first_frame:
            array = np.asarray(image, dtype=np.uint8)
            if array.ndim != 3:
                return None
            arrays.append(array)
        return np.stack(arrays, axis=0)
    return None


def resize_and_center_crop_pil(
    frame: Image.Image, target_width: int, target_height: int,
) -> Image.Image:
    """Resize a PIL image so it covers the target size, then center-crop."""
    original_width, original_height = frame.size
    scale_factor = max(
        target_width / original_width, target_height / original_height,
    )
    resize_width = int(round(original_width * scale_factor))
    resize_height = int(round(original_height * scale_factor))
    resize_transform = T.Compose([
        T.Resize(
            (resize_height, resize_width),
            interpolation=T.InterpolationMode.LANCZOS,
        ),
        T.CenterCrop((target_height, target_width)),
    ])
    return resize_transform(frame)


# ---------------------------------------------------------------------------
# Loader construction
# ---------------------------------------------------------------------------

def dataset_seed(config: TrainingConfig) -> int:
    """Shuffle seed handed to the datasets: ``seed`` plus ``dataset_seed_offset``."""
    return config.seed + (config.dataset_seed_offset or 0)


def _make_dataloader(dataset, batch_size: int, dp_rank: int = 0, dp_size: int = 1):
    """Wrap an iterable dataset in a DataLoader that reads this data-parallel rank's share."""
    if not isinstance(dataset, torch.utils.data.IterableDataset):
        raise TypeError(
            f"{type(dataset).__name__} is not an iterable dataset: the trainer batches consecutive "
            "samples of a torch.utils.data.IterableDataset."
        )
    if dp_size > 1:
        if not hasattr(dataset, "shard"):
            raise TypeError(
                f"{type(dataset).__name__} cannot be split across {dp_size} data-parallel ranks: "
                "every rank would read the same samples. It needs a shard(num_shards, index) method."
            )
        dataset = dataset.shard(num_shards=dp_size, index=dp_rank)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=training_collate_fn,
    )


def build_dataloader(
    config: TrainingConfig,
    loader_name: str,
    dp_rank: int = 0,
    dp_size: int = 1,
):
    """Build the training DataLoader of one configured dataset."""
    spec = get_dataset_spec(loader_name)
    batch_size = batch_size_for_loader(config, loader_name)
    dataset = spec.factory(
        config=config,
        batch_size=batch_size,
        seed=dataset_seed(config),
    )
    return _make_dataloader(dataset, batch_size, dp_rank, dp_size)


def build_multi_dataloader(
    config: TrainingConfig,
    probabilities: Dict[str, float],
    dp_rank: int = 0,
    dp_size: int = 1,
) -> "MultiDataloader":
    """Build one training DataLoader per name in ``probabilities`` and wrap them."""
    dataloaders = {
        name: build_dataloader(config, name, dp_rank=dp_rank, dp_size=dp_size)
        for name in probabilities
    }
    return MultiDataloader(
        dataloaders=dataloaders,
        probabilities=probabilities,
        prefetch_batches=config.prefetch_batches,
        one_epoch_per_loader=config.one_epoch_per_loader,
    )


# ---------------------------------------------------------------------------
# Per-dataset batch settings
# ---------------------------------------------------------------------------

def batch_size_for_loader(config: TrainingConfig, loader_name: str) -> int:
    """Per-GPU batch size of a dataset: its data-config value, else ``config.batch_size``."""
    batch_size = get_dataset_spec(loader_name).batch_size
    return config.batch_size if batch_size is None else batch_size


def gradient_accumulation_steps_for_loader(config: TrainingConfig, loader_name: Optional[str]) -> int:
    """Gradient-accumulation steps of a dataset: its data-config value, else
    ``config.gradient_accumulation_steps``."""
    steps = None if loader_name is None else get_dataset_spec(loader_name).gradient_accumulation_steps
    return config.gradient_accumulation_steps if steps is None else steps


# ---------------------------------------------------------------------------
# MultiDataloader
# ---------------------------------------------------------------------------

def _is_multi_rank(group=None) -> bool:
    """True when torch.distributed is initialised and ``group`` has more than one rank."""
    if not (dist.is_available() and dist.is_initialized()):
        return False
    try:
        return dist.get_world_size(group=group) > 1
    except RuntimeError:
        return False


class _BackgroundPrefetchIterator:
    """Pull batches from an iterator in a background thread, a few ahead of the consumer."""

    _END = object()

    def __init__(self, iterator, max_prefetch: int = 1):
        self.iterator = iterator
        self.queue = queue.Queue(maxsize=max(1, max_prefetch))
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            for batch in self.iterator:
                self.queue.put(batch)
            self.queue.put(self._END)
        except BaseException as exc:
            self.queue.put(exc)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is self._END:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item


class MultiDataloader:
    """Several named DataLoaders sampled by probability; ``epochs[name]`` counts the completed
    passes of a loader."""

    def __init__(
        self,
        dataloaders: Dict[str, Any],
        probabilities: Dict[str, float],
        prefetch_batches: int = 1,
        one_epoch_per_loader: bool = False,
    ):
        assert set(dataloaders.keys()) == set(probabilities.keys()), (
            f"Dataloader names must match probability names. "
            f"Dataloaders: {set(dataloaders.keys())}, Probabilities: {set(probabilities.keys())}"
        )
        total_prob = sum(probabilities.values())
        assert abs(total_prob - 1.0) < 1e-6, (
            f"Probabilities must sum to 1.0, got {total_prob}"
        )

        self.dataloaders = dataloaders
        self.probabilities = probabilities
        self.names = list(dataloaders.keys())
        self.prefetch_batches = prefetch_batches
        self.epochs: Dict[str, int] = {name: 0 for name in self.names}
        self._one_epoch_per_loader = one_epoch_per_loader
        # Process group whose ranks roll epochs over on the same step
        self._epoch_sync_group = None
        self._epoch_sync_enabled = False
        self._exhausted: set = set()
        self._iterators: Dict[str, Any] = {}
        self._primed_batch: Dict[str, Any] = {}

    def warm_up(self, loader_names=None):
        """Pull one batch from each named loader (default: all) and keep it for the next
        ``next_batch`` call."""
        for name in (loader_names if loader_names is not None else self.names):
            if name in self._exhausted:
                continue
            self._primed_batch[name] = self.next_batch(name)

    def set_epoch_sync_group(self, group) -> None:
        """Make all ranks of ``group`` roll a loader's epoch over on the same step.
        Call it after ``warm_up``."""
        self._epoch_sync_group = group
        self._epoch_sync_enabled = group is not None

    def sample_name(
        self,
        sync_group=None,
    ):
        """Pick a loader name by probability; rank 0 of ``sync_group`` decides for every rank."""
        group = sync_group

        if self._one_epoch_per_loader:
            self._sync_exhausted(group)

        available_names = [n for n in self.names if n not in self._exhausted]
        if not available_names:
            raise RuntimeError(
                f"All dataloaders have completed one epoch and are exhausted: "
                f"{sorted(self._exhausted)}. Training will stop."
            )
        available_probs = [self.probabilities[n] for n in available_names]
        total_prob = sum(available_probs)
        available_probs = [p / total_prob for p in available_probs]

        if _is_multi_rank(group):
            rank_in_group = dist.get_rank(group=group)
            name = random.choices(available_names, weights=available_probs, k=1)[0] if rank_in_group == 0 else None
            name_list = [name]
            src_rank = dist.get_global_rank(group, 0) if group is not None else 0
            dist.broadcast_object_list(name_list, src=src_rank, group=group)
            name = name_list[0]
        else:
            name = random.choices(available_names, weights=available_probs, k=1)[0]

        return name

    def next_batch(self, name: str):
        """Return the next batch of the named loader, starting its next epoch when it runs out."""
        if name in self._primed_batch:
            return self._primed_batch.pop(name)
        if name not in self._iterators:
            self._iterators[name] = self._make_iterator(name)

        batch = None
        exhausted_locally = False
        try:
            batch = next(self._iterators[name])
        except StopIteration:
            exhausted_locally = True

        if not self._any_rank_exhausted(exhausted_locally):
            return batch

        if not exhausted_locally:
            logger.info(
                f"[MultiDataloader] '{name}' rolled over on a peer rank; "
                "restarting here too to keep ranks in step."
            )
        return self._rollover_and_get_batch(name)

    def _make_iterator(self, name: str):
        iterator = iter(self.dataloaders[name])
        if self.prefetch_batches > 0:
            return _BackgroundPrefetchIterator(iterator, max_prefetch=self.prefetch_batches)
        return iterator

    def _sync_exhausted(self, group=None):
        """All-reduce the exhausted flags so every rank agrees on which loaders are done."""
        if not _is_multi_rank(group):
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local = torch.tensor(
            [1 if n in self._exhausted else 0 for n in self.names],
            dtype=torch.int32, device=device,
        )
        dist.all_reduce(local, op=dist.ReduceOp.MAX, group=group)
        for i, n in enumerate(self.names):
            if local[i].item() > 0 and n not in self._exhausted:
                logger.info(f"[MultiDataloader] '{n}' marked exhausted from peer rank; excluding.")
                self._exhausted.add(n)

    def _any_rank_exhausted(self, exhausted_locally: bool) -> bool:
        """True if the loader ran out on this rank or on any rank of the epoch-sync group."""
        if not self._epoch_sync_enabled:
            return exhausted_locally
        group = self._epoch_sync_group
        if not _is_multi_rank(group):
            return exhausted_locally
        flag = torch.tensor([1 if exhausted_locally else 0], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        return bool(flag.item() > 0)

    def _rollover_and_get_batch(self, name: str):
        """Start the next epoch of a loader and return its first batch."""
        self.epochs[name] += 1
        if self._one_epoch_per_loader:
            logger.info(
                f"[MultiDataloader] '{name}' completed epoch {self.epochs[name]}; "
                "marking exhausted — will be excluded from next step onward."
            )
            self._exhausted.add(name)
        else:
            logger.info(
                f"[MultiDataloader] '{name}' completed epoch {self.epochs[name]}; "
                "restarting from the beginning of its shard/sample list."
            )
        self._iterators[name] = self._make_iterator(name)
        try:
            return next(self._iterators[name])
        except StopIteration:
            raise RuntimeError(
                f"Dataloader '{name}' is empty even after reset (epoch {self.epochs[name]}); "
                "its dataset yielded no samples."
            ) from None
