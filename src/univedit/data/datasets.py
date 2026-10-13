"""The sample format and the training dataset: a streaming reader of a
``metadata.jsonl`` manifest, tar shards (WebDataset layout) or parquet shards."""

from __future__ import annotations

import json
import random
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, TypedDict

import numpy as np
import PIL.Image
import torch
from loguru import logger

from univedit.data.preprocessing import FrameSampling, build_image_sample, build_video_sample
from univedit.data.sources import (
    IMAGE_EXTENSIONS,
    SHARD_BUILDERS,
    VIDEO_EXTENSIONS,
    Location,
    decode_image,
    decode_text,
    open_video,
    read_aligned_frames,
)

IMAGE_TASKS = ("i2i", "ti_i2i")
REFERENCE_TASKS = ("ti_i2i", "tiv2v")
MANIFEST_FILE = "metadata.jsonl"
FORMATS = ("jsonl",) + tuple(SHARD_BUILDERS)

# Default names of the sample fields in a row.
DEFAULT_KEYS = {"source": "source", "target": "target", "reference": "reference"}
DEFAULT_TEXT_KEYS = ("txt", "text", "json.text")


class EditSample(TypedDict, total=False):
    """Keys a training sample may carry."""

    # all tasks (an image is a static clip of IMAGE_CLIP_FRAMES frames)
    text: str
    target_pixels: torch.Tensor
    target_latents: torch.Tensor
    byt5_text_ids: torch.Tensor
    byt5_text_mask: torch.Tensor
    source_sequence: List[PIL.Image.Image]
    source_sequence_latents: torch.Tensor
    # v2v / tiv2v / 3d23d: text-encoder inputs from the source
    source_sampled_frames: List[PIL.Image.Image]
    source_first_frame: np.ndarray
    # ti_i2i / tiv2v: the reference
    reference_image: PIL.Image.Image
    reference_image_latents: torch.Tensor


class _SkipLog:
    """Warn about samples left out: the first ``first_n`` per reason, then every ``every_n``-th."""

    def __init__(self, label: str, first_n: int = 20, every_n: int = 100):
        self.label = label
        self.first_n = first_n
        self.every_n = every_n
        self.counts: Dict[str, int] = {}

    def __call__(self, reason: str, *, sample_id=None, details: Optional[str] = None) -> None:
        count = self.counts[reason] = self.counts.get(reason, 0) + 1
        if count > self.first_n and count % self.every_n != 0:
            return
        message = f"[data_skip][{self.label}] reason={reason} count={count}"
        if sample_id is not None:
            message += f" sample={str(sample_id)[:120]}"
        if details:
            message += f" details={details[:240]}"
        logger.warning(message)


def shuffle_buffer(samples: Iterable, buffer_size: int, seed: int) -> Iterator:
    """Shuffle a stream with a buffer of ``buffer_size`` items (0 keeps the order)."""
    if not buffer_size:
        yield from samples
        return
    rng = random.Random(seed)
    buffer: List = []
    for sample in samples:
        if len(buffer) < buffer_size:
            buffer.append(sample)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = sample
    rng.shuffle(buffer)
    yield from buffer


def iter_bucketed_samples(samples: Iterable[dict], batch_size: int, skip_log: Optional[_SkipLog] = None) -> Iterator[dict]:
    """Regroup a stream so every run of ``batch_size`` samples shares one
    ``target_pixels`` shape. Samples left in a bucket when the stream ends are dropped."""
    if batch_size <= 1:
        yield from samples
        return
    buckets: Dict[tuple, List[dict]] = {}
    for sample in samples:
        bucket = buckets.setdefault(tuple(sample["target_pixels"].shape), [])
        bucket.append(sample)
        if len(bucket) >= batch_size:
            yield from bucket
            bucket.clear()
    if skip_log is not None:
        for shape, bucket in buckets.items():
            if bucket:
                skip_log("incomplete_bucket", details=f"shape={shape}, dropped={len(bucket)}, batch_size={batch_size}")


class EditDataset(torch.utils.data.IterableDataset):
    """Streaming dataset of one task, one sample per row of ``<root>/metadata.jsonl``
    or of tar / parquet shard files; yields runs of ``batch_size`` samples of one bucket."""

    def __init__(
        self,
        path: str,
        task: str,
        *,
        format: Optional[str] = None,
        shards: Optional[str] = None,
        keys: Optional[Dict[str, str]] = None,
        sampling: Optional[FrameSampling] = None,
        shuffle_buffer: int = 0,
        batch_size: int = 1,
        seed: int = 0,
        name: str = "",
    ):
        super().__init__()
        if (task in IMAGE_TASKS) != (sampling is None):
            raise ValueError(f"Task '{task}': frame sampling is required for video / 3D tasks and unused for image tasks.")
        self.location = Location(path)
        self.task = task
        self.keys = {**DEFAULT_KEYS, **(keys or {})}
        self.sampling = sampling
        self.shuffle_buffer = shuffle_buffer
        self.batch_size = batch_size
        self.seed = seed
        self.num_shards = 1
        self.shard_index = 0
        self._label = name or path
        self._skip_log = _SkipLog(self._label)
        self._completed_shards: Set[str] = set()
        self._on_shard_completed: Optional[Callable[[str], None]] = None
        if format is None and not shards and self.location.exists(MANIFEST_FILE):
            format = "jsonl"
        self.records: List[dict] = self._read_manifest() if format == "jsonl" else []
        self.shard_paths: List[str] = [] if format == "jsonl" else self._list_shards(format, shards)
        self.format = format or self.shard_paths[0].rsplit(".", 1)[-1].lower()

    # ---- what the dataset consists of ----

    def _read_manifest(self) -> List[dict]:
        """Rows of ``metadata.jsonl``: one JSON object per line."""
        records = []
        lines = self.location.read_bytes(MANIFEST_FILE).decode("utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                record = exc
            if not isinstance(record, dict):
                raise ValueError(f"{self.location.path}/{MANIFEST_FILE} line {number}: expected a JSON object ({record}).")
            records.append(record)
        if not records:
            raise ValueError(f"{self.location.path}/{MANIFEST_FILE} lists no samples.")
        return records

    def _list_shards(self, format: Optional[str], shards: Optional[str]) -> List[str]:
        """Sorted shard files matching the glob (default: every file of the format), all of one format."""
        extensions = (format,) if format else tuple(SHARD_BUILDERS)
        globs = (shards,) if shards else tuple(f"**/*.{extension}" for extension in extensions)
        paths = sorted({path for pattern in globs for path in self.location.glob(pattern)})
        if not paths:
            expected = f"shard files matching {list(globs)}"
            if not format and not shards:
                expected = f"{MANIFEST_FILE} and no {expected}"
            raise FileNotFoundError(f"'{self.location.path}' has no {expected}.")
        found = sorted({path.rsplit(".", 1)[-1].lower() for path in paths})
        if len(found) > 1 or found[0] not in extensions:
            raise ValueError(
                f"Shard files below '{self.location.path}' must all be "
                f"{' or all be '.join('.' + extension for extension in extensions)}, found {found}; "
                "select one kind with 'format' or 'shards'."
            )
        return paths

    def shard(self, num_shards: int, index: int) -> "EditDataset":
        """Make this dataset read share ``index`` of ``num_shards`` (one share per rank)."""
        self.num_shards = num_shards
        self.shard_index = index
        return self

    # ---- resume hooks (shard files only) ----

    def set_completed_shards(self, completed_shards: Set[str]) -> "EditDataset":
        """Set the shard files to skip (paths relative to the root); the set is updated in place."""
        self._completed_shards = completed_shards if completed_shards is not None else set()
        return self

    def set_shard_completion_callback(self, callback: Callable[[str], None]) -> "EditDataset":
        """``callback(shard)`` is called each time a shard file has been read to its end."""
        self._on_shard_completed = callback
        return self

    # ---- rows of one pass ----

    def _iter_rows(self) -> Iterator[dict]:
        """This rank's rows: every ``num_shards``-th manifest row, or the rows of its shard files."""
        if self.format == "jsonl":
            rows = self.records[self.shard_index::self.num_shards]
            random.Random(self.seed + self.shard_index * 9973).shuffle(rows)
            yield from rows
            return
        pending = [shard for shard in self.shard_paths if shard not in self._completed_shards]
        if not pending:
            self._completed_shards.clear()
            pending = list(self.shard_paths)
        split_by_row = len(pending) < self.num_shards
        if split_by_row:
            random.Random(self.seed).shuffle(pending)
        else:
            pending = pending[self.shard_index::self.num_shards]
            random.Random(self.seed + self.shard_index * 9973).shuffle(pending)
        rows_before = 0
        for shard in pending:
            rows_in_shard = 0
            for index, row in enumerate(self.location.stream_shard(shard)):
                rows_in_shard = index + 1
                if not split_by_row or (rows_before + index) % self.num_shards == self.shard_index:
                    yield row
            rows_before += rows_in_shard
            self._completed_shards.add(shard)
            if self._on_shard_completed is not None:
                self._on_shard_completed(shard)

    # ---- row -> sample ----

    def _media(self, row: dict, field: str, extensions) -> Any:
        """Value of the field ``<name>``, or ``<name>.<ext>`` as tar members are named."""
        name = self.keys[field]
        for candidate in (name, *(f"{name}.{extension}" for extension in extensions)):
            value = _lookup(row, candidate)
            if value is not None:
                return value
        raise KeyError(f"no '{name}' field for {field}; the row has {_fields(row)}")

    def _text(self, row: dict) -> str:
        """Instruction text from the configured field, or from the usual ones."""
        names = (self.keys["text"],) if "text" in self.keys else DEFAULT_TEXT_KEYS
        for name in names:
            text = decode_text(_lookup(row, name))
            if text:
                return text
        raise KeyError(f"no instruction in {names}; the row has {_fields(row)}")

    def _image(self, value) -> PIL.Image.Image:
        """Image from a path below the root (str) or from an encoded value."""
        return self.location.read_image(value) if isinstance(value, str) else decode_image(value)

    def _frames(self, value):
        """Frame sequence from a path below the root (str) or from an encoded value."""
        return self.location.open_frames(value) if isinstance(value, str) else open_video(value)

    def _build_sample(self, row: dict, rng: random.Random) -> EditSample:
        text = self._text(row)
        reference = None
        if self.task in REFERENCE_TASKS:
            reference = self._image(self._media(row, "reference", IMAGE_EXTENSIONS))
        if self.task in IMAGE_TASKS:
            source = self._image(self._media(row, "source", IMAGE_EXTENSIONS))
            target = self._image(self._media(row, "target", IMAGE_EXTENSIONS))
            return build_image_sample(text, source, target, reference=reference)
        source = self._frames(self._media(row, "source", VIDEO_EXTENSIONS))
        target = self._frames(self._media(row, "target", VIDEO_EXTENSIONS))
        with source as source_reader, target as target_reader:
            source_frames, target_frames = read_aligned_frames(
                source_reader, target_reader, lambda total: self.sampling.indices(total, rng),
            )
        return build_video_sample(text, source_frames, target_frames, self.sampling, reference=reference, rng=rng)

    def _iter_samples(self) -> Iterator[EditSample]:
        """Samples of one pass; a row that cannot be turned into a sample is logged and left out."""
        rng = random.Random(self.seed + self.shard_index * 9973)
        for row in self._iter_rows():
            try:
                yield self._build_sample(row, rng)
            except Exception as exc:
                source = row.get(self.keys["source"])
                sample_id = row.get("__key__") or (source if isinstance(source, str) else None)
                self._skip_log(type(exc).__name__, sample_id=sample_id, details=str(exc))

    def __iter__(self) -> Iterator[EditSample]:
        shuffled = shuffle_buffer(self._iter_samples(), self.shuffle_buffer, self.seed)
        return iter_bucketed_samples(shuffled, self.batch_size, self._skip_log)


def _lookup(row: dict, name: str) -> Any:
    """Value of the field ``name``, or of the dotted path ``field.subfield...``
    into a JSON / struct field; None when missing."""
    if row.get(name) is not None:
        return row[name]
    value = row
    for part in name.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _fields(row: dict) -> List[str]:
    return sorted(key for key in row if not key.startswith("__"))
