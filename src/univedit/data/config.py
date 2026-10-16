"""YAML data config: schema, validation, the table of configured datasets and
their construction (see ``configs/data.example.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import yaml

from univedit.data.datasets import DEFAULT_KEYS, FORMATS, EditDataset
from univedit.data.preprocessing import CLIP_SAMPLING_MODES, FrameSampling, num_frames_for_views
from univedit.data.sources import Location
from univedit.tokens import SUPPORTED_TASKS

SAMPLE_FIELDS = tuple(DEFAULT_KEYS) + ("text",)

# Top-level groups of a data config and the tasks of their datasets.
GROUP_TASKS = {"image": ("i2i", "ti_i2i"), "video": ("v2v", "tiv2v"), "3d": ("3d23d",)}

# Options a group may set as defaults for its datasets, and the per-dataset ones.
_GROUP_OPTIONS = {"image": set(), "video": {"num_frames", "frame_sampling"}, "3d": {"num_views", "view_window"}}
_DATASET_OPTIONS = {"task", "path", "prob", "format", "shards", "keys"}

DEFAULT_NUM_FRAMES = 65
DEFAULT_FRAME_SAMPLING = "random_clip"
DEFAULT_NUM_VIEWS = 26
# Decoded samples held by the shuffle buffer of a dataset, per group.
SHUFFLE_BUFFER = {"image": 150, "video": 50, "3d": 50}

DatasetFactory = Callable[..., torch.utils.data.Dataset]


@dataclass(frozen=True)
class DatasetSpec:
    """A configured dataset: its task, how to build it and its sampling weight."""

    name: str
    task: str
    factory: DatasetFactory
    prob: float = 1.0


_DATASETS: Dict[str, DatasetSpec] = {}


def group_of_task(task: str) -> str:
    """The data-config group (image / video / 3d) of a task."""
    return next(group for group, tasks in GROUP_TASKS.items() if task in tasks)


def register_dataset(
    name: str,
    task: str,
    factory: DatasetFactory,
    *,
    prob: float = 1.0,
) -> None:
    """Add ``factory(*, config, batch_size, seed) -> IterableDataset`` to the
    table under ``name``, replacing a dataset of that name."""
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unknown task '{task}' for dataset '{name}'. Expected one of {SUPPORTED_TASKS}.")
    _DATASETS[name] = DatasetSpec(name, task, factory, prob)


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return _DATASETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset name '{name}'. Configured: {sorted(_DATASETS)} (dataset names come from --data_config)."
        ) from None


def list_datasets() -> List[str]:
    return sorted(_DATASETS)


def sampling_probabilities() -> Dict[str, float]:
    """Sampling probability of every configured dataset with a positive weight:
    its ``prob`` divided by the sum of all weights."""
    weights = {name: spec.prob for name, spec in _DATASETS.items() if spec.prob > 0}
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


@dataclass(frozen=True)
class DatasetEntry:
    """One validated dataset of a data config."""

    name: str
    task: str
    path: str
    prob: float
    format: str
    shards: Optional[str]
    keys: Dict[str, str]
    sampling: Optional[FrameSampling]


def _fail(name: str, message: str):
    raise ValueError(f"Data config, dataset '{name}': {message}")


def _integer(name: str, options: Dict[str, Any], key: str, minimum: int, default=None) -> Optional[int]:
    value = options.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(name, f"'{key}' must be an integer >= {minimum}, got {value!r}.")
    return value


def _parse_entry(name: str, group: str, options: Dict[str, Any]) -> DatasetEntry:
    """Validate one dataset; ``options`` are its own options over the defaults of its group."""
    tasks = GROUP_TASKS[group]
    task = options.get("task", tasks[0] if len(tasks) == 1 else None)
    if task not in tasks:
        _fail(name, f"'task' must be one of {tasks} in group '{group}', got {task!r}.")
    path = options.get("path")
    if not isinstance(path, str) or not path:
        _fail(name, "'path' is required: a local directory or hf://datasets/<user>/<repo>[/<subdir>].")
    data_format = options.get("format", "auto")
    if data_format not in ("auto",) + FORMATS:
        _fail(name, f"'format' must be one of {('auto',) + FORMATS}, got {data_format!r}.")
    if data_format == "jsonl" and "shards" in options:
        _fail(name, "'shards' does not apply to format 'jsonl'.")

    prob = options.get("prob", 1.0)
    if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not prob >= 0:
        _fail(name, f"'prob' must be a number >= 0, got {prob!r}.")
    keys = options.get("keys", {})
    if not isinstance(keys, dict) or not all(
        key in SAMPLE_FIELDS and isinstance(value, str) and value for key, value in keys.items()
    ):
        _fail(name, f"'keys' must map some of {SAMPLE_FIELDS} to field names, got {keys!r}.")
    shards = options.get("shards")
    if shards is not None and (not isinstance(shards, str) or not shards):
        _fail(name, f"'shards' must be a glob pattern, got {shards!r}.")

    sampling = None
    if group == "3d":
        num_frames = num_frames_for_views(_integer(name, options, "num_views", 1, DEFAULT_NUM_VIEWS))
        window = _integer(name, options, "view_window", (num_frames - 1) // 4 + 1)
        sampling = FrameSampling(num_frames, "uniform_dup4", window)
    elif group == "video":
        mode = options.get("frame_sampling", DEFAULT_FRAME_SAMPLING)
        num_frames = _integer(name, options, "num_frames", 1, DEFAULT_NUM_FRAMES)
        if mode not in CLIP_SAMPLING_MODES:
            _fail(name, f"'frame_sampling' must be one of {CLIP_SAMPLING_MODES}, got {mode!r}.")
        if (num_frames - 1) % 4 != 0:
            _fail(name, f"'num_frames' must be 4n+1, got {num_frames}.")
        sampling = FrameSampling(num_frames, mode)
    try:
        Location(path)
    except (FileNotFoundError, ValueError) as exc:
        _fail(name, str(exc))

    return DatasetEntry(
        name=name,
        task=task,
        path=path,
        prob=float(prob),
        format=data_format,
        shards=shards,
        keys=dict(keys),
        sampling=sampling,
    )


def read_data_config(path: str) -> List[DatasetEntry]:
    """Read and validate a YAML data config: the groups ``image`` / ``video`` /
    ``3d``, each with defaults and a ``datasets`` mapping."""
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if not isinstance(content, dict) or not set(content) <= set(GROUP_TASKS):
        raise ValueError(f"Data config '{path}': the top-level keys must be among {tuple(GROUP_TASKS)}.")
    entries: Dict[str, DatasetEntry] = {}
    for group, section in content.items():
        defaults_allowed = _GROUP_OPTIONS[group]
        if not isinstance(section, dict) or not set(section) <= defaults_allowed | {"datasets"}:
            expected = f"optional defaults among {sorted(defaults_allowed)}" if defaults_allowed else "no other option"
            raise ValueError(f"Data config '{path}', group '{group}': expected 'datasets' and {expected}.")
        datasets = section.get("datasets") or {}
        if not isinstance(datasets, dict):
            raise ValueError(f"Data config '{path}', group '{group}': 'datasets' must map dataset names to their options.")
        defaults = {key: value for key, value in section.items() if key != "datasets"}
        for name, options in datasets.items():
            name = str(name)
            if name in entries:
                _fail(name, "the name is used more than once; dataset names are unique across groups.")
            if not isinstance(options, dict):
                _fail(name, "expected a mapping of options (path, prob, ...).")
            unknown = sorted(set(options) - _DATASET_OPTIONS - defaults_allowed)
            if unknown:
                _fail(name, f"option(s) {unknown} are unknown or do not apply to group '{group}'. "
                            f"Allowed: {sorted(_DATASET_OPTIONS | defaults_allowed)}.")
            entries[name] = _parse_entry(name, group, {**defaults, **options})
    if not any(entry.prob > 0 for entry in entries.values()):
        raise ValueError(f"Data config '{path}': no dataset with prob > 0.")
    return list(entries.values())


def build_dataset(entry: DatasetEntry, *, batch_size: int, seed: int) -> EditDataset:
    """Build the dataset of a data config entry."""
    return EditDataset(
        entry.path,
        entry.task,
        format=None if entry.format == "auto" else entry.format,
        shards=entry.shards,
        keys=entry.keys,
        sampling=entry.sampling,
        shuffle_buffer=SHUFFLE_BUFFER[group_of_task(entry.task)],
        batch_size=batch_size,
        seed=seed,
        name=entry.name,
    )


def load_data_config(path: str) -> None:
    """Make the datasets of a YAML data config the table of configured datasets."""
    entries = read_data_config(path)
    _DATASETS.clear()
    for entry in entries:
        def factory(*, config, batch_size, seed, entry=entry):
            return build_dataset(entry, batch_size=batch_size, seed=seed)

        register_dataset(entry.name, entry.task, factory, prob=entry.prob)
