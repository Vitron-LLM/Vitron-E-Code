"""Where datasets are read from: a local directory or a Hugging Face dataset
repo. File listing, byte reading, shard streaming and media decoding."""

from __future__ import annotations

import contextlib
import io
import os
import posixpath
import re
import tempfile
from typing import Any, Callable, Iterable, Iterator, List, Sequence, Tuple, Union

import PIL.ExifTags
import PIL.Image
import PIL.ImageOps


HF_PREFIX = "hf://datasets/"
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp")
VIDEO_EXTENSIONS = ("mp4", "mov", "webm")
# Shard file extension -> ``datasets`` builder that streams it.
SHARD_BUILDERS = {"tar": "webdataset", "parquet": "parquet"}

_REPO_ID = re.compile(r"[\w.-]+/[\w.-]+")

def has_extension(name: str, extensions: Sequence[str]) -> bool:
    return name.rsplit(".", 1)[-1].lower() in extensions if "." in name else False


class Location:
    """Root of a dataset: a local directory, or a Hugging Face dataset repo written
    ``hf://datasets/<user>/<repo>[/<subdir>]``. Method paths are relative to the root."""

    def __init__(self, path: str):
        self.path = path
        local = os.path.expanduser(path)
        if path.startswith(HF_PREFIX):
            repo_path = path[len(HF_PREFIX):].strip("/")
        elif os.path.exists(local):
            repo_path = None
        elif _REPO_ID.fullmatch(path):
            repo_path = path
        else:
            raise FileNotFoundError(
                f"'{path}' is neither an existing local path nor a Hugging Face dataset "
                f"('{HF_PREFIX}<user>/<repo>[/<subdir>]')."
            )
        if repo_path is None:
            import fsspec

            self.repo_id = None
            self.fs = fsspec.filesystem("file")
            self.root = os.path.abspath(local)
        else:
            from huggingface_hub import HfFileSystem

            parts = repo_path.split("/")
            if len(parts) < 2:
                raise ValueError(f"'{path}' does not name a dataset repo: expected {HF_PREFIX}<user>/<repo>.")
            self.repo_id = "/".join(parts[:2])
            self.fs = HfFileSystem()
            self.root = f"datasets/{repo_path}"

    def join(self, relative: str) -> str:
        return f"{self.root}/{relative}"

    def exists(self, relative: str) -> bool:
        return self.fs.exists(self.join(relative))

    def glob(self, pattern: str) -> List[str]:
        """Sorted relative paths of the files matching a glob pattern (``**`` crosses directories)."""
        return sorted(posixpath.relpath(path, self.root) for path in self.fs.glob(self.join(pattern)))

    def list_images(self, relative: str) -> List[str]:
        """Relative paths of the image files in a directory, sorted by name."""
        names = sorted(posixpath.basename(path) for path in self.fs.ls(self.join(relative), detail=False))
        return [f"{relative.rstrip('/')}/{name}" for name in names if has_extension(name, IMAGE_EXTENSIONS)]

    def read_bytes(self, relative: str) -> bytes:
        path = self.join(relative)
        return self.fs.cat_file(path)

    def read_image(self, relative: str) -> PIL.Image.Image:
        return decode_image(self.read_bytes(relative))

    def open_frames(self, relative: str):
        """Context manager over the frames of a video file, or of a directory of frame images sorted by name."""
        if not has_extension(relative, VIDEO_EXTENSIONS):
            return open_video(self.list_images(relative), load_frame=self.read_image)
        return open_video(self.join(relative) if self.repo_id is None else self.read_bytes(relative))

    def stream_shard(self, shard: str) -> Iterable[dict]:
        """Rows of one shard file (tar in WebDataset layout, or parquet), streamed
        with ``datasets``; media columns arrive encoded, as bytes or ``{"bytes", "path"}``."""
        from datasets import load_dataset

        builder = SHARD_BUILDERS[shard.rsplit(".", 1)[-1].lower()]
        if builder == "parquet":
            _accept_list_feature()
        if self.repo_id is None:
            name, data_file = builder, self.join(shard)
        else:
            name, data_file = self.repo_id, posixpath.relpath(self.join(shard), f"datasets/{self.repo_id}")
        return load_dataset(name, data_files={"train": data_file}, split="train", streaming=True).decode(False)


def _accept_list_feature() -> None:
    """Let ``datasets`` 3.x read parquet files whose schema names the ``List`` feature type (an alias of ``Sequence``)."""
    from datasets.features import features

    features._FEATURE_TYPES.setdefault("List", features.Sequence)


def _payload(value):
    """The bytes of a ``{"bytes": ..., "path": ...}`` value; any other value unchanged."""
    if isinstance(value, dict) and "bytes" in value:
        if value["bytes"] is None:
            raise ValueError(f"media value has no bytes (path: {value.get('path')!r}).")
        return value["bytes"]
    return value


def decode_text(value) -> str:
    """Stripped text from a str or UTF-8 bytes value ("" for None)."""
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8")
    return str(value).strip()


def decode_image(value) -> PIL.Image.Image:
    """RGB image from encoded image bytes (also as ``{"bytes": ...}``) or a PIL
    image; an EXIF orientation is applied to decoded bytes."""
    value = _payload(value)
    if isinstance(value, PIL.Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        image = PIL.Image.open(io.BytesIO(value))
        image.load()
        if image.getexif().get(PIL.ExifTags.Base.Orientation) is not None:
            image = PIL.ImageOps.exif_transpose(image)
        return image.convert("RGB")
    raise TypeError(f"Expected an image (PIL, bytes or a dict with bytes), got {type(value)}")


class VideoFrames:
    """Frames of an encoded video, read by index with decord."""

    def __init__(self, reader):
        self._reader = reader

    def __len__(self) -> int:
        return len(self._reader)

    def read(self, indices: Sequence[int]) -> List[PIL.Image.Image]:
        frames = self._reader.get_batch(list(indices)).asnumpy()
        return [PIL.Image.fromarray(frame).convert("RGB") for frame in frames]


class ImageFrames:
    """Frames given one image each; ``load_frame`` turns an item into an RGB image."""

    def __init__(self, items: Sequence, load_frame: Callable[[Any], PIL.Image.Image]):
        self._items = items
        self._load_frame = load_frame

    def __len__(self) -> int:
        return len(self._items)

    def read(self, indices: Sequence[int]) -> List[PIL.Image.Image]:
        decoded = {index: self._load_frame(self._items[index]) for index in set(indices)}
        return [decoded[index] for index in indices]


FrameSequence = Union[VideoFrames, ImageFrames]


@contextlib.contextmanager
def open_video(video, load_frame: Callable[[Any], PIL.Image.Image] = decode_image) -> Iterator[FrameSequence]:
    """Frames of a video value: a local file path, encoded video bytes (also as
    ``{"bytes": ...}``) or a list of frame images."""
    video = _payload(video)
    if isinstance(video, (list, tuple)):
        yield ImageFrames(video, load_frame)
        return
    from decord import VideoReader

    with contextlib.ExitStack() as stack:
        if isinstance(video, (bytes, bytearray)):
            tmp = stack.enter_context(tempfile.NamedTemporaryFile(suffix=".mp4"))
            tmp.write(video)
            tmp.flush()
            video = tmp.name
        elif not isinstance(video, str):
            raise TypeError(f"Expected a video (path, bytes, a dict with bytes or a list of frames), got {type(video)}")
        yield VideoFrames(VideoReader(video))


def read_aligned_frames(
    source: FrameSequence,
    target: FrameSequence,
    select: Callable[[int], List[int]],
) -> Tuple[List[PIL.Image.Image], List[PIL.Image.Image]]:
    """Read the frames ``select(total_frames)`` picks from two frame-aligned sequences."""
    if len(source) != len(target):
        raise ValueError(f"source and target are not frame-aligned: {len(source)} vs {len(target)} frames.")
    indices = select(len(source))
    return source.read(indices), target.read(indices)
