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

"""UniVEdit trainer: multi-task LoRA training for image, video and 3D editing.

Contains :class:`HunyuanVideoTrainer` and the :func:`main` entry point. A subclass extends the
trainer through the ``add_cli_args`` / ``trainer_kwargs_from_args`` classmethods and the
``_begin_auxiliary_loss`` / ``_auxiliary_loss`` loss hooks.
"""

import os
import random
import math
import argparse
import time
import json
import datetime
from enum import Enum
from types import SimpleNamespace
from typing import Dict, Any, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
)
from diffusers.optimization import get_scheduler
from loguru import logger
import numpy as np
import torchvision.transforms as T
import PIL.Image

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import get_parallel_state, initialize_parallel_state
from hyvideo.optim.muon import get_muon_optimizer
from hyvideo.utils.data_utils import resize_and_center_crop
from univedit.data.config import GROUP_TASKS, get_dataset_spec, list_datasets, load_data_config, sampling_probabilities
from univedit.data.datasets import EditSample
from univedit.data.preprocessing import IMAGE_CLIP_FRAMES
from univedit.tokens import SPECIAL_TOKENS, SUPPORTED_TASKS, get_special_token_from_task
from univedit.training.config import SNRType, TrainingConfig, build_arg_parser, config_from_args
from univedit.training.dataloading import (
    MultiDataloader,
    batch_size_for_loader,
    build_multi_dataloader,
    dataset_seed,
    get_semantic_images_np,
    gradient_accumulation_steps_for_loader,
    normalize_batched_pil_images,
    normalize_batched_pil_sequences,
    normalize_sampled_frames,
    normalize_semantic_images_np,
    resize_and_center_crop_pil,
)

from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)


# Keys a batch may carry.
_SAMPLE_KEYS = frozenset(EditSample.__annotations__)
# Settings that hold local paths; they are left out of the config sent to Weights & Biases.
_LOCAL_PATH_FIELDS = frozenset({
    "pretrained_model_root", "output_dir", "data_config", "resume_from_checkpoint",
    "pretrained_lora_path", "pretrained_modality_token_path",
})


# ---- torch.distributed helpers: rank-0 broadcasts that keep sequence-parallel ranks in sync ----

try:
    from torch.distributed.tensor import DTensor, distribute_tensor
except ImportError:
    DTensor = None
    distribute_tensor = None


def is_src(src, group_src, group):
    """True on the rank that is the broadcast source (``src`` global rank or ``group_src`` rank within ``group``)."""
    assert src is not None or group_src is not None
    assert src is None or group_src is None
    if src is not None:
        return dist.get_rank() == src
    if group_src is not None:
        return dist.get_rank() == dist.get_global_rank(group, group_src)
    raise RuntimeError("src and group_src cannot be both None")

def _resolve_src(src, group_src, group):
    """Global rank of the broadcast source."""
    if src is not None:
        return src
    if group_src is not None and group is not None:
        return dist.get_global_rank(group, group_src)
    if group_src is not None:
        return group_src
    raise RuntimeError("src and group_src cannot be both None")


def broadcast_object(
        obj,
        src = None,
        group = None,
        device = None,
        group_src = None,
):
    """Broadcast a picklable object from the source rank to the group."""
    resolved_src = _resolve_src(src, group_src, group)
    buffer = [obj] if is_src(src, group_src, group) else [None]

    dist.broadcast_object_list(buffer, src=resolved_src, group=group, device=device)
    return buffer[0]

def broadcast_tensor(
        tensor,
        src  = None,
        group = None,
        async_op: bool = False,
        group_src = None,
):
    """Broadcast a tensor from the source rank to the group."""
    resolved_src = _resolve_src(src, group_src, group)
    if is_src(src, group_src, group):
        tensor = tensor.cuda().contiguous()
    if is_src(src, group_src, group):
        shape, dtype = tensor.shape, tensor.dtype
    else:
        shape, dtype = None, None
    shape = broadcast_object(shape, src=src, group_src=group_src, group=group)
    dtype = broadcast_object(dtype, src=src, group_src=group_src, group=group)

    buffer = tensor if is_src(src, group_src, group) else torch.empty(shape, device='cuda', dtype=dtype)
    dist.broadcast(buffer, src=resolved_src, group=group, async_op=async_op)
    return buffer


def sync_tensor_for_sp(tensor: torch.Tensor, sp_group) -> torch.Tensor:
    """Broadcast a tensor (or a picklable object) from rank 0 of the sequence-parallel group."""
    if sp_group is None:
        return tensor
    if not isinstance(tensor, torch.Tensor):
        obj_list = [tensor]
        dist.broadcast_object_list(obj_list, src=dist.get_global_rank(sp_group, 0), group=sp_group)
        return obj_list[0]
    return broadcast_tensor(tensor, group_src=0, group=sp_group)


# ---- LoRA weight loading into an existing (possibly FSDP2-sharded) PEFT adapter ----

def contains_dtensor(value) -> bool:
    """True if a (nested) value holds a DTensor."""
    if DTensor is not None and isinstance(value, DTensor):
        return True
    if isinstance(value, dict):
        return any(contains_dtensor(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_dtensor(v) for v in value)
    return False


def normalize_activation_checkpoint_key(key: str) -> str:
    """Parameter name without the activation-checkpoint wrapper segment."""
    return key.replace("._checkpoint_wrapped_module", "")


def load_lora_weights_into_dtensor_adapter(
    transformer,
    state_dict: Dict[str, torch.Tensor],
    adapter_name: str,
    is_main_process: bool = True,
):
    """Copy LoRA weights into adapter parameters that are FSDP2-sharded DTensors."""
    if DTensor is None or distribute_tensor is None:
        raise RuntimeError("Cannot load LoRA weights into DTensor parameters: DTensor is unavailable.")

    from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
    from peft.utils.save_and_load import _insert_adapter_name_into_state_dict

    config = transformer.peft_config[adapter_name]
    parameter_prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(config.peft_type)
    if parameter_prefix is None:
        raise RuntimeError(f"Unsupported PEFT type for DTensor LoRA loading: {config.peft_type}")

    peft_state_dict = _insert_adapter_name_into_state_dict(
        state_dict,
        adapter_name=adapter_name,
        parameter_prefix=parameter_prefix,
    )
    target_params = {}
    adapter_marker = f".{adapter_name}."
    for name, param in transformer.named_parameters():
        normalized_name = normalize_activation_checkpoint_key(name)
        if parameter_prefix in normalized_name and adapter_marker in normalized_name:
            target_params[normalized_name] = (name, param)

    loaded_keys = set()
    unexpected_keys = []
    with torch.no_grad():
        for key, value in peft_state_dict.items():
            normalized_key = normalize_activation_checkpoint_key(key)
            target = target_params.get(normalized_key)
            if target is None:
                unexpected_keys.append(key)
                continue

            target_name, param = target
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(
                    "LoRA weight shape mismatch: "
                    f"checkpoint {key}={tuple(value.shape)}, "
                    f"model {target_name}={tuple(param.shape)}"
                )

            source = value.detach().to(dtype=param.dtype).contiguous()
            if isinstance(param, DTensor):
                source = distribute_tensor(
                    source,
                    device_mesh=param.device_mesh,
                    placements=param.placements,
                )
            else:
                source = source.to(device=param.device)
            param.copy_(source)
            loaded_keys.add(normalized_key)

    missing_keys = [
        target_name for normalized_name, (target_name, _) in target_params.items()
        if normalized_name not in loaded_keys
    ]
    if is_main_process:
        logger.info(
            f"Loaded {len(loaded_keys)} LoRA DTensor parameter(s) "
            f"for adapter '{adapter_name}'"
        )
    return SimpleNamespace(missing_keys=missing_keys, unexpected_keys=unexpected_keys)


def load_lora_weights_into_existing_adapter(transformer, lora_dir: str, adapter_name: str, is_main_process: bool = True):
    """Load LoRA weights from ``lora_dir`` into an adapter that already exists on the transformer."""
    safe_path = os.path.join(lora_dir, "pytorch_lora_weights.safetensors")
    bin_path = os.path.join(lora_dir, "pytorch_lora_weights.bin")
    if os.path.exists(safe_path):
        from safetensors.torch import load_file as safe_load_file

        state_dict = safe_load_file(safe_path, device="cpu")
        source_path = safe_path
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        source_path = bin_path
    else:
        raise FileNotFoundError(
            f"LoRA weights not found in {lora_dir}; expected "
            f"{os.path.basename(safe_path)} or {os.path.basename(bin_path)}"
        )

    has_dtensor_lora = (
        DTensor is not None
        and any(
            isinstance(param, DTensor)
            and "lora_" in name
            and f".{adapter_name}." in name
            for name, param in transformer.named_parameters()
        )
    )
    if has_dtensor_lora:
        incompatible_keys = load_lora_weights_into_dtensor_adapter(
            transformer, state_dict, adapter_name, is_main_process=is_main_process,
        )
    else:
        from peft.utils import set_peft_model_state_dict

        incompatible_keys = set_peft_model_state_dict(
            transformer,
            state_dict,
            adapter_name=adapter_name,
        )
    missing_keys = [
        key for key in (getattr(incompatible_keys, "missing_keys", None) or [])
        if "lora_" in key and adapter_name in key
    ]
    unexpected_keys = [
        key for key in (getattr(incompatible_keys, "unexpected_keys", None) or [])
        if "lora_" in key and adapter_name in key
    ]
    if is_main_process:
        if missing_keys or unexpected_keys:
            logger.warning(
                "LoRA adapter loaded with incompatible keys: "
                f"missing={missing_keys}, unexpected={unexpected_keys}"
            )
        logger.info(f"LoRA adapter '{adapter_name}' weights loaded from {source_path}")


class LinearInterpolationSchedule:
    """Linear interpolation schedule for flow matching."""
    def __init__(self, T: int = 1000):
        self.T = T
    
    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return ``(1 - t/T) * x0 + (t/T) * x1`` (x0: clean latents, x1: noise)."""
        t_normalized = t / self.T
        t_normalized = t_normalized.view(-1, *([1] * (x0.ndim - 1)))
        return (1 - t_normalized) * x0 + t_normalized * x1


class TimestepSampler:
    """Samples training timesteps in [0, T] with the configured strategy."""

    TRAIN_EPS = 1e-5
    SAMPLE_EPS = 1e-3
    
    def __init__(
        self, 
        T: int = 1000, 
        device: torch.device = None,
        snr_type: SNRType = SNRType.LOGNORM,
    ):
        self.T = T
        self.device = device
        self.snr_type = SNRType(snr_type) if isinstance(snr_type, str) else snr_type
    
    def _check_interval(self, eval: bool = False):
        eps = self.SAMPLE_EPS if eval else self.TRAIN_EPS
        t0 = eps
        t1 = 1.0 - eps
        return t0, t1
    
    def sample(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = self.device if self.device is not None else torch.device("cuda")
        
        t0, t1 = self._check_interval(eval=False)
        
        if self.snr_type == SNRType.UNIFORM:
            t = torch.rand((batch_size,), device=device) * (t1 - t0) + t0
            
        elif self.snr_type == SNRType.LOGNORM:
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            t = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0
            
        elif self.snr_type == SNRType.MIX:
            # 70% lognorm, 30% uniform
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            t_lognorm = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0
            
            delta = 0.0
            t0_clip = t0 + delta
            t1_clip = t1 - delta
            t_clip_uniform = torch.rand((batch_size,), device=device) * (t1_clip - t0_clip) + t0_clip
            
            mask = (torch.rand((batch_size,), device=device) > 0.3).float()
            t = mask * t_lognorm + (1 - mask) * t_clip_uniform
            
        elif self.snr_type == SNRType.MODE:
            mode_scale = 1.29
            u = torch.rand(size=(batch_size,), device=device)
            t = 1.0 - u - mode_scale * (torch.cos(math.pi * u / 2.0) ** 2 - 1.0 + u)
            # Scale to [t0, t1] range
            t = t * (t1 - t0) + t0
        else:
            raise ValueError(f"Unknown SNR type: {self.snr_type}")
        
        # Scale to [0, T] range
        timesteps = t * self.T
        return timesteps


def timestep_transform(timesteps: torch.Tensor, T: int, shift: float = 1.0) -> torch.Tensor:
    """Transform timesteps with shift"""
    if shift == 1.0:
        return timesteps
    timesteps_normalized = timesteps / T
    timesteps_transformed = shift * timesteps_normalized / (1 + (shift - 1) * timesteps_normalized)
    return timesteps_transformed * T


class HunyuanVideoTrainer:
    """Trainer for multi-task image, video and 3D editing."""

    # ---- Subclass hooks used by main(trainer_cls=...) ----
    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Subclass hook: add command-line arguments to ``parser``."""

    @classmethod
    def trainer_kwargs_from_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Subclass hook: extra constructor keyword arguments from the parsed command line."""
        return {}

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_main_process = self.rank == 0
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.is_main_process = True
        
        if config.sp_size > self.world_size:
            raise ValueError(
                f"sp_size ({config.sp_size}) cannot be greater than world_size ({self.world_size})"
            )
        if self.world_size % config.sp_size != 0:
            raise ValueError(
                f"sp_size ({config.sp_size}) must evenly divide world_size ({self.world_size}). "
                f"world_size % sp_size = {self.world_size % config.sp_size}"
            )

        torch.cuda.set_device(self.local_rank)

        # Main process group
        if self.world_size > 1 and dist.is_available() and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                timeout=datetime.timedelta(minutes=config.nccl_timeout_minutes),
            )

        initialize_parallel_state(sp=config.sp_size, dp_replicate=config.dp_replicate)
        self.parallel_state = get_parallel_state()
        self.dp_rank = self.parallel_state.world_mesh['dp'].get_local_rank()
        self.dp_size = self.parallel_state.world_mesh['dp'].size()
        self.sp_enabled = self.parallel_state.sp_enabled
        self.sp_group = self.parallel_state.sp_group if self.sp_enabled else None

        # Gloo process group for data coordination
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            self.data_ctrl_group = dist.new_group(
                backend="gloo",
                timeout=datetime.timedelta(hours=config.data_ctrl_timeout_hours),
            )
        else:
            self.data_ctrl_group = None

        self._set_seed(config.seed + self.dp_rank)
        self.modality_token_optimizer = None
        self.modality_token_lr_scheduler = None
        self.modality_token_embedding = None
        self.modality_token_ids = None
        self.modality_token_embedding_hook = None
        self.text_encoder_visual_merger_dtype_hook = None
        self.wandb_run = None
        self._build_models()
        self._build_optimizer()
        
        self.noise_schedule = LinearInterpolationSchedule(T=config.num_train_timesteps)
        self.timestep_sampler = TimestepSampler(
            T=config.num_train_timesteps, 
            device=self.device,
            snr_type=config.snr_type,
        )
        
        self.global_step = 0
        self.micro_step = 0
        self.current_epoch = 0
        self._last_save_bucket = 0
        self._last_checkpoint_step = 0
        self._pending_update_metrics = []
        self.completed_data_shards: Dict[str, set[str]] = {}
        # {loader_name: epoch_count} of the loaded checkpoint
        self.data_epoch_counts: Dict[str, int] = {}
        # The MultiDataloader being trained on
        self._active_multi_dataloader = None
        
        if self.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)

        self._init_wandb()
    
    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _normalize_completed_data_shards_state(state) -> Dict[str, set[str]]:
        if not state:
            return {}
        normalized = {}
        for loader_name, shards in dict(state).items():
            normalized[str(loader_name)] = {str(shard) for shard in shards}
        return normalized

    def _completed_data_shards_state(self) -> Dict[str, List[str]]:
        return {
            loader_name: sorted(str(shard) for shard in shards)
            for loader_name, shards in sorted(self.completed_data_shards.items())
        }

    def _mark_data_shard_completed(self, loader_name: str, shard_id: str):
        if not loader_name or not shard_id:
            return
        self.completed_data_shards.setdefault(str(loader_name), set()).add(str(shard_id))

    def _merge_completed_data_shards(self, state):
        for loader_name, shards in self._normalize_completed_data_shards_state(state).items():
            self.completed_data_shards.setdefault(loader_name, set()).update(shards)

    def _gather_completed_data_shards(self):
        local_state = self._completed_data_shards_state()
        if self.world_size > 1:
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, local_state)
            for state in gathered:
                self._merge_completed_data_shards(state)
        else:
            self._merge_completed_data_shards(local_state)
        return self._completed_data_shards_state()

    def _gather_data_epoch_counts(self) -> Dict[str, int]:
        """Collect each loader's epoch count from the live MultiDataloader (max over ranks)."""
        local: Dict[str, int] = {}
        if self._active_multi_dataloader is not None:
            local = dict(self._active_multi_dataloader.epochs)
        if self.world_size > 1:
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, local)
            return merge_data_epoch_counts(gathered)
        return local

    def _attach_shard_progress_to_dataset(self, loader_name: str, dataset):
        """Hand a dataset the shards it completed and a callback to report the next ones."""
        if dataset is None:
            return
        completed = self.completed_data_shards.setdefault(loader_name, set())
        if hasattr(dataset, "set_completed_shards"):
            dataset.set_completed_shards(completed)
        if hasattr(dataset, "set_shard_completion_callback"):
            dataset.set_shard_completion_callback(
                lambda shard_id, loader_name=loader_name: self._mark_data_shard_completed(loader_name, shard_id)
            )

    def _attach_shard_progress(self, dataloader: "MultiDataloader"):
        for loader_name, loader in dataloader.dataloaders.items():
            self._attach_shard_progress_to_dataset(
                loader_name,
                getattr(loader, "dataset", None),
            )
        self._active_multi_dataloader = dataloader
        for loader_name, count in self.data_epoch_counts.items():
            if loader_name in dataloader.epochs:
                dataloader.epochs[loader_name] = count

    def _init_wandb(self):
        if not self.config.use_wandb or not self.is_main_process:
            return

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb logging is enabled, but wandb is not installed. "
                "Install it with `pip install wandb` or disable `--use_wandb`."
            ) from exc

        tags = [tag.strip() for tag in self.config.wandb_tags.split(",") if tag.strip()]
        run_config = {
            key: (value.value if isinstance(value, Enum) else value)
            for key, value in self.config.__dict__.items()
            if key not in _LOCAL_PATH_FIELDS
        }
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity or None,
            name=self.config.wandb_run_name or None,
            tags=tags or None,
            mode=self.config.wandb_mode,
            config=run_config,
        )
        logger.info(
            f"W&B enabled: project={self.config.wandb_project}, "
            f"run={self.wandb_run.name if self.wandb_run is not None else self.config.wandb_run_name}"
        )

    def _log_metrics_to_wandb(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if self.wandb_run is None or not self.is_main_process:
            return

        task_name = _metric_name(metrics.get("task_type"))
        loader_name = _metric_name(metrics.get("loader_name"))
        log_data = {
            "train/loss": metrics["loss"],
            "train/grad_norm": metrics["grad_norm"],
            "train/lr": metrics["lr"],
            "train/accumulation_steps": metrics["accumulation_steps"],
            "train/optimizer_step": int(metrics["optimizer_step"]),
            "train/micro_step": metrics.get("micro_step", self.micro_step),
            "train/micro_batch_count": metrics.get("micro_batch_count", 1),
            "train/sample_count": metrics.get("sample_count", 0),
            "train/timestep_mean": metrics["timestep_mean"],
            "train/timestep_min": metrics["timestep_min"],
            "train/timestep_max": metrics["timestep_max"],
            f"task/{task_name}/loss": metrics["loss"],
            f"task/{task_name}/timestep_mean": metrics["timestep_mean"],
        }

        if loader_name != "unknown":
            log_data[f"loader/{loader_name}/loss"] = metrics["loss"]
            log_data[f"loader/{loader_name}/timestep_mean"] = metrics["timestep_mean"]
        if metrics.get("modality_token_lr") is not None:
            log_data["train/modality_token_lr"] = metrics["modality_token_lr"]

        self.wandb_run.log(log_data, step=self.global_step if step is None else step)
    
    def _build_models(self):
        if self.config.dtype == "bf16":
            transformer_dtype = torch.bfloat16
        elif self.config.dtype == "fp32":
            transformer_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {self.config.dtype}")
        
        self.pipeline = HunyuanVideo_1_5_Pipeline.create_pipeline(
            pretrained_model_name_or_path=self.config.pretrained_model_root,
            transformer_dtype=transformer_dtype,
            enable_offloading=False,
            enable_group_offloading=False,
            overlap_group_offloading=False,
            create_sr_pipeline=False,
            flow_shift=self.config.sampling_timestep_shift,
            device=self.device,
        )
        
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.vision_encoder = self.pipeline.vision_encoder
        self.byt5_kwargs = {
            "byt5_model": self.pipeline.byt5_model,
            "byt5_tokenizer": self.pipeline.byt5_tokenizer,
        }
        
        self.transformer.train()

        # Freeze the VAE and the encoders
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.eval()
            self.text_encoder_2.requires_grad_(False)
        if self.vision_encoder is not None and isinstance(self.vision_encoder, nn.Module):
            self.vision_encoder.eval()
            self.vision_encoder.requires_grad_(False)
        if self.byt5_kwargs["byt5_model"] is not None:
            self.byt5_kwargs["byt5_model"].eval()
            self.byt5_kwargs["byt5_model"].requires_grad_(False)

        self._align_text_encoder_visual_merger_dtype()

        if self.config.train_modality_token_embeddings:
            self._enable_modality_token_embedding_training()
            if self.config.pretrained_modality_token_path is not None:
                self.load_pretrained_modality_token_embeddings(
                    self.config.pretrained_modality_token_path
                )

        if self.config.enable_text_encoder_gradient_checkpointing:
            self._apply_text_encoder_gradient_checkpointing()

        if self.config.enable_text_encoder_fsdp and self.world_size > 1:
            self._apply_text_encoder_fsdp()

        if self.config.use_lora:
            self._apply_lora()

        if self.config.enable_gradient_checkpointing:
            self._apply_gradient_checkpointing()
        
        if self.config.enable_fsdp and self.world_size > 1:
            self._apply_fsdp()

        if self.is_main_process:
            logger.info(f"Models loaded. Transformer dtype: {transformer_dtype}")
            total_params = sum(p.numel() for p in self.transformer.parameters())
            trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            logger.info(f"Transformer parameters: {total_params:,} (trainable: {trainable_params:,})")
            logger.info(f"LoRA enabled: {self.config.use_lora}")
            logger.info(f"Train modality token embeddings: {self.config.train_modality_token_embeddings}")
            logger.info(f"Main optimizer LR scheduler: {self.config.lr_scheduler}")
            logger.info(f"FSDP enabled: {self.config.enable_fsdp and self.world_size > 1}")
            logger.info(f"Gradient checkpointing enabled: {self.config.enable_gradient_checkpointing}")
            logger.info(f"Text encoder FSDP enabled: {self.config.enable_text_encoder_fsdp and self.world_size > 1}")
            logger.info(f"Text encoder gradient checkpointing enabled: {self.config.enable_text_encoder_gradient_checkpointing}")
            logger.info(f"Modality token LR scheduler: {self.config.modality_token_lr_scheduler}")
            logger.info(f"Timestep sampling strategy: {self.config.snr_type.value}")

    def _enable_modality_token_embedding_training(self):
        tokenizer = self.text_encoder.tokenizer
        token_ids = tokenizer.convert_tokens_to_ids(list(SPECIAL_TOKENS))
        missing_tokens = [
            token for token, token_id in zip(SPECIAL_TOKENS, token_ids)
            if token_id is None or token_id == tokenizer.unk_token_id
        ]
        if missing_tokens:
            raise ValueError(f"Modality tokens are not registered in tokenizer: {missing_tokens}")

        embedding = self.text_encoder.model.get_input_embeddings()
        if embedding is None:
            raise ValueError("Text encoder does not expose input embeddings.")

        self.text_encoder.requires_grad_(False)
        token_ids_tensor = torch.tensor(token_ids, device=embedding.weight.device, dtype=torch.long)
        self.modality_token_embedding = nn.Parameter(
            embedding.weight.detach().index_select(0, token_ids_tensor).float().clone()
        )
        self.modality_token_ids = token_ids_tensor

        def replace_modality_token_outputs(module, inputs, output):
            input_ids = inputs[0]
            if input_ids is None:
                return output

            patched_output = output.clone()
            token_vectors = self.modality_token_embedding.to(
                device=patched_output.device,
                dtype=patched_output.dtype,
            )
            token_ids_for_input = self.modality_token_ids.to(input_ids.device)
            for row_idx, token_id in enumerate(token_ids_for_input):
                patched_output = torch.where(
                    (input_ids == token_id).unsqueeze(-1),
                    token_vectors[row_idx].view(*([1] * (patched_output.ndim - 1)), -1),
                    patched_output,
                )
            return patched_output

        self.modality_token_embedding_hook = embedding.register_forward_hook(
            replace_modality_token_outputs
        )

        if self.is_main_process:
            token_pairs = ", ".join(f"{token}:{token_id}" for token, token_id in zip(SPECIAL_TOKENS, token_ids))
            logger.info(f"Training modality token embeddings with AdamW: {token_pairs}")
    
    def _apply_lora(self):
        if self.is_main_process:
            logger.info("Applying LoRA to transformer using PeftAdapterMixin...")

        if self.config.pretrained_lora_path is not None:
            if self.is_main_process:
                logger.info(f"Loading pretrained LoRA from {self.config.pretrained_lora_path}")
            self.load_pretrained_lora(self.config.pretrained_lora_path)
        else:
            from peft import LoraConfig
            
            if self.config.lora_target_modules is None:
                target_modules = "all-linear"
            else:
                target_modules = self.config.lora_target_modules
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            
            self.transformer.add_adapter(lora_config, adapter_name="default")

        
        if self.is_main_process:
            trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.transformer.parameters())
            logger.info(f"LoRA applied successfully. Trainable parameters: {trainable_params:,} / {total_params:,} "
                       f"({100 * trainable_params / total_params:.2f}%)")

    def _apply_fsdp(self):
        if self.is_main_process:
            logger.info("Applying FSDP2 to transformer...")
        
        param_dtype = torch.bfloat16
        reduce_dtype = torch.float32

        self.transformer = self.transformer.to(dtype=param_dtype)
        
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        )
        
        fsdp_config = {"mp_policy": mp_policy}
        if self.world_size > 1:
            try:
                fsdp_config["mesh"] = get_parallel_state().fsdp_mesh
            except Exception as e:
                if self.is_main_process:
                    logger.warning(f"Could not create DeviceMesh: {e}. FSDP will use process group instead.")
        
        for block in list(self.transformer.double_blocks) + list(self.transformer.single_blocks):
            if block is not None:
                fully_shard(block, **fsdp_config)
        
        fully_shard(self.transformer, **fsdp_config)
        
        if self.is_main_process:
            logger.info("FSDP2 applied successfully")

    def _text_encoder_fsdp_config(self):
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        fsdp_config = {"mp_policy": mp_policy}
        try:
            fsdp_config["mesh"] = get_parallel_state().fsdp_mesh
        except Exception as e:
            if self.is_main_process:
                logger.warning(f"Could not create text encoder DeviceMesh: {e}. FSDP will use process group instead.")
        return fsdp_config

    def _get_qwen_vl_model_parts(self):
        qwen = getattr(self.text_encoder, "model", None)
        qwen_core = getattr(qwen, "model", None)
        language_model = getattr(qwen_core, "language_model", None)
        visual_model = getattr(qwen_core, "visual", None)
        return qwen, qwen_core, language_model, visual_model

    def _align_text_encoder_visual_merger_dtype(self):
        _, _, _, visual_model = self._get_qwen_vl_model_parts()
        merger = getattr(visual_model, "merger", None) if visual_model is not None else None
        ln_q = getattr(merger, "ln_q", None)
        mlp = getattr(merger, "mlp", None)
        if ln_q is None or mlp is None:
            return

        def cast_merger_norm_output(module, inputs, output):
            try:
                target_dtype = next(mlp.parameters()).dtype
            except StopIteration:
                return output
            if torch.is_tensor(output) and output.is_floating_point() and output.dtype != target_dtype:
                return output.to(dtype=target_dtype)
            return output

        self.text_encoder_visual_merger_dtype_hook = ln_q.register_forward_hook(
            cast_merger_norm_output
        )

    def _apply_text_encoder_fsdp(self):
        if self.is_main_process:
            logger.info("Applying FSDP2 to Qwen-VL text encoder...")

        _, _, language_model, visual_model = self._get_qwen_vl_model_parts()
        if language_model is None:
            raise RuntimeError("Cannot enable text encoder FSDP: Qwen language_model was not found.")

        fsdp_config = self._text_encoder_fsdp_config()

        language_layers = list(getattr(language_model, "layers", []))
        for layer in language_layers:
            fully_shard(layer, **fsdp_config)

        visual_blocks = list(getattr(visual_model, "blocks", [])) if visual_model is not None else []
        for block in visual_blocks:
            fully_shard(block, **fsdp_config)

        if self.is_main_process:
            logger.info(
                "Qwen-VL text encoder block FSDP2 applied successfully "
                f"(language layers={len(language_layers)}, visual blocks={len(visual_blocks)}; "
                "embedding and root modules kept replicated for Qwen-VL input handling)"
            )

    def _apply_text_encoder_gradient_checkpointing(self):
        if self.is_main_process:
            logger.info("Applying activation checkpointing to Qwen-VL text encoder...")

        qwen, _, language_model, visual_model = self._get_qwen_vl_model_parts()
        if language_model is None:
            raise RuntimeError("Cannot enable text encoder checkpointing: Qwen language_model was not found.")

        for module in (qwen, getattr(qwen, "model", None), language_model):
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "use_cache"):
                config.use_cache = False

        modules_to_wrap = []
        language_layers = list(getattr(language_model, "layers", []))
        if language_layers:
            language_layer_type = type(language_layers[0])
            modules_to_wrap.append((language_model, language_layer_type))

        visual_blocks = list(getattr(visual_model, "blocks", [])) if visual_model is not None else []
        if visual_blocks:
            visual_block_type = type(visual_blocks[0])
            modules_to_wrap.append((visual_model, visual_block_type))

        def non_reentrant_wrapper(module):
            return checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )

        for root, layer_type in modules_to_wrap:
            apply_activation_checkpointing(
                root,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=lambda submodule, layer_type=layer_type: isinstance(submodule, layer_type),
            )

        if self.is_main_process:
            logger.info(
                "Qwen-VL text encoder activation checkpointing applied successfully "
                f"(language layers={len(language_layers)}, visual blocks={len(visual_blocks)})"
            )
    
    def _apply_gradient_checkpointing(self):
        if self.is_main_process:
            logger.info("Applying gradient checkpointing to transformer blocks...")
        
        no_split_module_type = None
        for block in self.transformer.double_blocks:
            if block is not None:
                no_split_module_type = type(block)
                break
        
        if no_split_module_type is None:
            for block in self.transformer.single_blocks:
                if block is not None:
                    no_split_module_type = type(block)
                    break
        
        if no_split_module_type is None:
            logger.warning("Could not find block type for gradient checkpointing. Using fallback.")
            if hasattr(self.transformer, "gradient_checkpointing_enable"):
                self.transformer.gradient_checkpointing_enable()
            return
        
        def non_reentrant_wrapper(module):
            return checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
        
        def selective_checkpointing(submodule):
            return isinstance(submodule, no_split_module_type)
        
        apply_activation_checkpointing(
            self.transformer,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=selective_checkpointing,
        )
        
        if self.is_main_process:
            logger.info("Gradient checkpointing applied successfully")
    
    def _build_optimizer(self):
        if self.config.use_muon:
            self.optimizer = get_muon_optimizer(
                model=self.transformer,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.config.weight_decay,
            )

        if self.modality_token_embedding is not None:
            self.modality_token_optimizer = torch.optim.AdamW(
                [self.modality_token_embedding],
                lr=self.config.modality_token_learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.config.modality_token_weight_decay,
            )
            modality_token_warmup_steps = (
                self.config.warmup_steps
                if self.config.modality_token_warmup_steps is None
                else self.config.modality_token_warmup_steps
            )
            self.modality_token_lr_scheduler = get_scheduler(
                self.config.modality_token_lr_scheduler,
                optimizer=self.modality_token_optimizer,
                num_warmup_steps=modality_token_warmup_steps,
                num_training_steps=self.config.max_steps,
            )
        
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps,
        )
        
        if self.is_main_process:
            logger.info(f"Optimizer and scheduler initialized")

    def _optimizer_uses_dtensor(self, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        if DTensor is None:
            return False
        optimizer = self.optimizer if optimizer is None else optimizer
        if optimizer is None:
            return False
        for group in optimizer.param_groups:
            for param in group["params"]:
                if isinstance(param, DTensor):
                    return True
        return contains_dtensor(optimizer.state)

    def _sync_modality_token_embedding_grads(self):
        if (
            self.modality_token_embedding is None
            or self.modality_token_embedding.grad is None
            or self.dp_size <= 1
        ):
            return

        dist.all_reduce(
            self.modality_token_embedding.grad,
            op=dist.ReduceOp.SUM,
            group=self.parallel_state.world_mesh["dp"].get_group(),
        )
        self.modality_token_embedding.grad.div_(self.dp_size)

    def _sync_modality_token_embedding_weights_for_sp(self):
        if self.modality_token_embedding is None or self.sp_group is None:
            return

        with torch.no_grad():
            token_weights = broadcast_tensor(
                self.modality_token_embedding.data.contiguous(),
                group_src=0,
                group=self.sp_group,
            )
            self.modality_token_embedding.data.copy_(token_weights)

    def _clip_trainable_grad_norms(self) -> torch.Tensor:
        dtensor_params = []
        tensor_params = []
        for p in self.transformer.parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if DTensor is not None and isinstance(p.grad, DTensor):
                dtensor_params.append(p)
            else:
                tensor_params.append(p)
        if (
            self.modality_token_embedding is not None
            and self.modality_token_embedding.grad is not None
        ):
            tensor_params.append(self.modality_token_embedding)

        total_norm_sq = torch.zeros((), device=self.device, dtype=torch.float32)

        for p in dtensor_params:
            local_grad = p.grad.detach().to_local()
            total_norm_sq += local_grad.float().pow(2).sum().to(self.device)

        if dtensor_params and dist.is_initialized():
            try:
                fsdp_group = get_parallel_state().fsdp_mesh.get_group()
            except Exception:
                fsdp_group = None
            dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM, group=fsdp_group)

        for p in tensor_params:
            grad = p.grad.detach()
            total_norm_sq += grad.float().pow(2).sum().to(self.device)

        total_norm = torch.sqrt(total_norm_sq)
        if not dtensor_params and not tensor_params:
            return total_norm

        _norm_val = total_norm.item()
        if math.isnan(_norm_val) or math.isinf(_norm_val):
            return total_norm

        clip_coef = self.config.max_grad_norm / (total_norm + 1e-6)
        clip_coef = torch.clamp(clip_coef, max=1.0)
        dtensor_clip_coef = float(clip_coef.detach().cpu().item())

        for p in dtensor_params:
            p.grad.mul_(dtensor_clip_coef)
        for p in tensor_params:
            p.grad.mul_(clip_coef.to(device=p.grad.device, dtype=p.grad.dtype))

        return total_norm
    
    def encode_byt5(self, text_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self.byt5_kwargs["byt5_model"] is None:
            return None, None
        with torch.no_grad():
            byt5_outputs = self.byt5_kwargs["byt5_model"](text_ids, attention_mask=attention_mask.float())
        byt5_emb = byt5_outputs[0]
        return byt5_emb, attention_mask
    
    def encode_pil_to_vision_states(self, pil_images_per_sample,
                                     target_height=None, target_width=None):
        """Encode each sample's list of PIL images with the vision encoder and average the states per sample.
        Returns ``[B, tokens, dim]``, or None without a vision encoder."""
        if self.vision_encoder is None:
            return None
        
        for cur_img_sampel in pil_images_per_sample:
            assert isinstance(cur_img_sampel, list), f"Each sample in pil_images_per_sample must be a list, but got {type(cur_img_sampel)}"
        
        vision_states_list = []
        for imgs in pil_images_per_sample:
            imgs_np = np.stack([np.array(img, dtype=np.uint8) for img in imgs])  # [N, H, W, C]
            if target_height is not None and target_width is not None:
                imgs_np = np.stack([
                    resize_and_center_crop(img, target_width, target_height)
                    for img in imgs_np
                ], axis=0)
            with torch.no_grad():
                encoder_output = self.vision_encoder.encode_images(imgs_np)
                current_vs = encoder_output.last_hidden_state  # [N, tokens, dim]
                current_vs = torch.mean(current_vs, dim=0)  # [tokens, dim]
            vision_states_list.append(current_vs)
        
        vision_states = torch.stack(vision_states_list, dim=0)  # [B, tokens, dim]
        return vision_states.to(device=self.device, dtype=self.transformer.dtype)

    def encode_text_for_task(
        self,
        prompts: List[str],
        task_type: str,
        batch: Dict[str, Any],
        modality_tokens: Optional[List[str]] = None,
    ):
        """Encode the prompts with the text encoder, together with the task's visual inputs from ``batch``.
        Returns ``(text_emb, text_mask, text_emb_2, text_mask_2, deepstack_hidden_states)``."""
        data_type = "video"
        deepstack = self.config.deepstack
        setclip = self.config.setclip

        def prepare_input(*args, **kwargs):
            if modality_tokens is not None:
                kwargs["modality_tokens"] = modality_tokens
            return self.text_encoder.prepare_input(*args, **kwargs)
        
        if task_type == "v2v":
            source_sampled_frames = batch.get("source_sampled_frames", None)
            if source_sampled_frames is not None:
                nframes = len(source_sampled_frames[0])
                after_process_inputs, crop_start = prepare_input(
                    prompts, videos=source_sampled_frames,
                    num_frames=nframes, prompt_mode=3,
                )
            else:
                raise ValueError("v2v batches need `source_sampled_frames` for the text encoder.")

        elif task_type == "3d23d":
            source_sampled_frames = batch.get("source_sampled_frames", None)
            if source_sampled_frames is not None:
                nframes = len(source_sampled_frames[0])
                after_process_inputs, crop_start = prepare_input(
                    prompts, videos=source_sampled_frames,
                    num_frames=nframes, prompt_mode=5,
                )
            else:
                raise ValueError("3d23d batches need `source_sampled_frames` for the text encoder.")

        elif task_type == "tiv2v":
            source_sampled_frames = batch.get("source_sampled_frames", None)
            reference_images = batch.get("reference_image", None)
            if source_sampled_frames is not None and reference_images is not None:
                newimgs = []
                for ps in reference_images:
                    if isinstance(ps, PIL.Image.Image):
                        ps = [ps]
                    subimgs = []
                    for p in ps:
                        p2 = p.copy()
                        p2.thumbnail((560, 560))
                        subimgs.append(p2)
                    newimgs.append(subimgs)
                nframes = len(source_sampled_frames[0])
                after_process_inputs, crop_start = prepare_input(
                    prompts, imgs=newimgs, videos=source_sampled_frames,
                    num_frames=nframes, prompt_mode=4,
                )
            else:
                raise ValueError("tiv2v batches need `source_sampled_frames` and `reference_image` for the text encoder.")
        
        elif task_type == "i2i":
            source_images = self._image_task_sources(batch, len(prompts))
            if source_images is not None:
                newimgs = []
                for imgs in source_images:
                    subimgs = []
                    for p in imgs:
                        p2 = p.copy()
                        p2.thumbnail((560, 560))
                        subimgs.append(p2)
                    newimgs.append(subimgs)
                after_process_inputs, crop_start = prepare_input(
                    prompts, newimgs, prompt_mode=1,
                )
            else:
                raise ValueError("i2i batches need `source_sequence` for the text encoder.")

        elif task_type == "ti_i2i":
            source_images = self._image_task_sources(batch, len(prompts))
            if source_images is not None:
                reference_images = normalize_batched_pil_images(
                    batch.get("reference_image", None),
                    len(prompts),
                    "reference_image",
                )
                if reference_images is None:
                    raise ValueError("ti_i2i requires `reference_image` in the batch")
                _require_one_image(reference_images, task_type, "reference image")
                newimgs = []
                for bi, imgs in enumerate(source_images):
                    visual_inputs = list(imgs) + reference_images[bi]
                    subimgs = []
                    for p in visual_inputs:
                        p2 = p.copy()
                        p2.thumbnail((560, 560))
                        subimgs.append(p2)
                    newimgs.append(subimgs)
                after_process_inputs, crop_start = prepare_input(
                    prompts, newimgs, prompt_mode=2,
                )
            else:
                raise ValueError("ti_i2i batches need `source_sequence` for the text encoder.")

        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        after_process_inputs = after_process_inputs.to(self.device)
        text_grad_context = (
            torch.enable_grad()
            if self.config.train_modality_token_embeddings
            else torch.no_grad()
        )
        text_encoder_dtype = getattr(self.text_encoder, "dtype", torch.float16)
        text_autocast_enabled = (
            self.device.type == "cuda"
            and text_encoder_dtype in (torch.float16, torch.bfloat16)
        )
        with text_grad_context, torch.autocast(
            device_type="cuda",
            dtype=text_encoder_dtype if text_autocast_enabled else torch.float16,
            enabled=text_autocast_enabled,
        ):
            text_outputs = self.text_encoder.encode(
                after_process_inputs,
                data_type=data_type,
                device=self.device,
                crop_start=crop_start,
                deepstack=deepstack,
                setclip=setclip,
            )
        text_emb = text_outputs.hidden_state
        text_mask = text_outputs.attention_mask
        deepstack_hidden_states = text_outputs.deepstack_hidden_states
        
        text_emb_2 = None
        text_mask_2 = None
        
        return text_emb, text_mask, text_emb_2, text_mask_2, deepstack_hidden_states
    
    def _vae_memory_context(self):
        """Context manager that enables the VAE's spatial tiling."""
        return self.vae.memory_efficient_context()

    def _scale_vae_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            return (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents * self.vae.config.scaling_factor

    def _encode_vae_tensor_once(self, pixel_values: torch.Tensor, sample_posterior: bool) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16), self._vae_memory_context():
            latent_dist = self.vae.encode(pixel_values).latent_dist
            latents = latent_dist.sample() if sample_posterior else latent_dist.mode()
        return self._scale_vae_latents(latents)

    def _encode_vae_tensor(
        self,
        pixel_values: torch.Tensor,
        sample_posterior: bool,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        if chunk_size <= 0 or pixel_values.shape[0] <= chunk_size:
            return self._encode_vae_tensor_once(pixel_values, sample_posterior=sample_posterior)

        latents = [
            self._encode_vae_tensor_once(chunk, sample_posterior=sample_posterior)
            for chunk in pixel_values.split(chunk_size, dim=0)
        ]
        return torch.cat(latents, dim=0)

    IMAGE_EDIT_TASKS = ("i2i", "ti_i2i")
    REFERENCE_TASKS = ("ti_i2i", "tiv2v")

    def encode_vae(self, images: torch.Tensor, task_type: Optional[str] = None) -> torch.Tensor:
        """Encode pixels in [-1, 1] to scaled VAE latents; image-editing tasks take the posterior mode,
        other tasks a sample."""
        if images.max() > 1.0 or images.min() < -1.0:
            raise ValueError(f"Images must be in the range [-1, 1], but got {images.min()} {images.max()}")

        if images.ndim == 4:
            images = images.unsqueeze(2)

        is_image_edit = task_type in self.IMAGE_EDIT_TASKS
        latents = self._encode_vae_tensor(
            images,
            sample_posterior=not is_image_edit,
            chunk_size=1,
        )

        return latents
    
    def encode_pil_images_to_latents(self, pil_images, target_size=None):
        """Encode a list of PIL images to VAE latents ``[N, C, 1, H, W]``.
        ``target_size``: optional pixel ``(height, width)`` to resize and center-crop the images to."""
        # ---- Resize + CenterCrop preprocessing ----
        if target_size is not None:
            target_h, target_w = target_size
            processed = []
            for img in pil_images:
                h, w = img.height, img.width
                if target_h / h > target_w / w:
                    resize_size = (target_h, int(w * target_h / h))
                else:
                    resize_size = (int(h * target_w / w), target_w)
                img = T.Resize(resize_size, antialias=True)(img)
                img = T.CenterCrop((target_h, target_w))(img)
                processed.append(img)
            pil_images = processed

        image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        pixel_values = torch.stack([image_transform(img) for img in pil_images])  # [N, C, H, W]
        pixel_values = pixel_values.unsqueeze(2).to(self.device)  # [N, C, 1, H, W]
        
        latents = self._encode_vae_tensor(
            pixel_values,
            sample_posterior=False,
            chunk_size=1,
        )
        
        return latents  # [N, C, 1, H, W]

    def encode_pil_sequences_to_latents(self, pil_sequences, target_size=None):
        """Encode one list of PIL frames per sample to VAE latents ``[B, C, F_lat, H_lat, W_lat]``.
        ``target_size``: optional pixel ``(height, width)`` to resize and center-crop the frames to."""
        if not pil_sequences:
            raise ValueError("pil_sequences must not be empty.")

        image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        video_tensors = []
        expected_frames = None
        for frames in pil_sequences:
            if not frames:
                raise ValueError("Each video must contain at least one frame.")
            if expected_frames is None:
                expected_frames = len(frames)
            elif len(frames) != expected_frames:
                raise ValueError(
                    "All videos in a batch must have the same number of frames, "
                    f"got {expected_frames} and {len(frames)}"
                )

            processed_frames = []
            for frame in frames:
                frame = frame.convert("RGB")
                if target_size is not None:
                    target_h, target_w = target_size
                    h, w = frame.height, frame.width
                    if target_h / h > target_w / w:
                        resize_size = (target_h, int(w * target_h / h))
                    else:
                        resize_size = (int(h * target_w / w), target_w)
                    frame = T.Resize(resize_size, antialias=True)(frame)
                    frame = T.CenterCrop((target_h, target_w))(frame)
                processed_frames.append(image_transform(frame))
            video_tensors.append(torch.stack(processed_frames, dim=1))  # [C, F, H, W]

        pixel_values = torch.stack(video_tensors, dim=0).to(self.device)  # [B, C, F, H, W]
        latents = self._encode_vae_tensor(
            pixel_values,
            sample_posterior=False,
            chunk_size=1,
        )
        return latents
    
    def get_condition(self, latents: torch.Tensor, task_type: str, batch: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Build the condition tensor ``[B, C+1, F, H, W]`` of a batch: condition latents plus a mask channel."""
        if task_type not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task type: {task_type}")
        assert batch is not None, f"`batch` must be provided for {task_type}"

        b, c, f, h, w = latents.shape

        # Pixel-space target size
        vae_spatial_ratio = getattr(self.pipeline, 'vae_spatial_compression_ratio', 16)
        pixel_target_size = (h * vae_spatial_ratio, w * vae_spatial_ratio)

        # Source clip latents and mask
        cond = torch.zeros([b, c + 1, f, h, w], device=latents.device, dtype=latents.dtype)
        if "source_sequence_latents" in batch and batch["source_sequence_latents"] is not None:
            source_sequence_latents = batch["source_sequence_latents"].to(device=latents.device, dtype=latents.dtype)
        else:
            source_sequences = normalize_batched_pil_sequences(
                batch.get("source_sequence", None), b, "source_sequence"
            )
            if source_sequences is None:
                raise ValueError(f"{task_type} requires `source_sequence_latents` or raw `source_sequence` frames.")
            source_sequence_latents = self.encode_pil_sequences_to_latents(
                source_sequences, target_size=pixel_target_size
            ).to(device=latents.device, dtype=latents.dtype)
        assert source_sequence_latents.shape == latents.shape, (
            f"{task_type}: source_sequence_latents shape {source_sequence_latents.shape} "
            f"must match target latents shape {latents.shape}"
        )
        cond[:, :-1, :] = source_sequence_latents
        cond[:, -1, :] = 1
        if task_type not in self.REFERENCE_TASKS:
            return cond

        # Reference image latents and mask, added at latent frame 1
        assert "reference_image" in batch or "reference_image_latents" in batch, \
            f"'reference_image' (or 'reference_image_latents') must be provided for {task_type}"
        assert 1 < f, f"{task_type}: the clip must have at least 2 latent frames for the reference image"
        pre_reference_latents = batch.get("reference_image_latents")
        if isinstance(pre_reference_latents, torch.Tensor):
            # Pre-encoded reference image
            ref_enc = pre_reference_latents.to(device=latents.device, dtype=latents.dtype)  # [B, C, 1, H, W]
            cond[:, :-1, 1] += ref_enc[:, :, 0]
            cond[:, -1, 1] += 1
        else:
            reference_images = normalize_batched_pil_images(batch["reference_image"], b, "reference_image")
            if reference_images is None:
                raise ValueError(f"{task_type} requires `reference_image` or `reference_image_latents` in the batch")
            for bi, imgs in enumerate(reference_images):
                assert len(imgs) == 1, (
                    f"{task_type} expects exactly one reference image per sample, "
                    f"got {len(imgs)} at batch index {bi}"
                )
            encoded = self.encode_pil_images_to_latents(
                [imgs[0] for imgs in reference_images],
                target_size=pixel_target_size,
            )
            encoded = encoded.squeeze(2)  # [B, C, H, W]
            for bi in range(b):
                cond[bi, :-1, 1] += encoded[bi].to(dtype=latents.dtype)
                cond[bi, -1, 1] += 1
        return cond

    @staticmethod
    def _image_task_sources(batch: Dict[str, Any], batch_size: int) -> Optional[List[List[PIL.Image.Image]]]:
        """i2i / ti_i2i: the source image of every sample, i.e. frame 0 of its static source clip."""
        source_sequences = normalize_batched_pil_sequences(batch.get("source_sequence", None), batch_size, "source_sequence")
        if source_sequences is None:
            return None
        return [[frames[0]] for frames in source_sequences]

    def _check_image_clip(self, batch: Dict[str, Any], task_type: str) -> None:
        """Raise unless an i2i / ti_i2i batch holds clips of ``IMAGE_CLIP_FRAMES`` frames (or their latents).
        With ``source_sequence_latents`` the source frames only feed the text and vision encoders
        (frame 0), so any number of them is accepted."""
        latent_frames = (IMAGE_CLIP_FRAMES - 1) // 4 + 1
        for key, frames in (
            ("target_pixels", IMAGE_CLIP_FRAMES),
            ("target_latents", latent_frames),
            ("source_sequence_latents", latent_frames),
        ):
            value = batch.get(key, None)
            if isinstance(value, torch.Tensor) and (value.ndim != 5 or value.shape[2] != frames):
                raise ValueError(
                    f"{task_type}: `{key}` must have {frames} frames ([B, C, {frames}, H, W]), "
                    f"got shape {tuple(value.shape)}."
                )
        source_sequences = batch.get("source_sequence", None)
        if source_sequences is not None and not isinstance(batch.get("source_sequence_latents"), torch.Tensor):
            for bi, frames in enumerate(normalize_batched_pil_sequences(source_sequences, len(batch["text"]), "source_sequence")):
                if len(frames) != IMAGE_CLIP_FRAMES:
                    raise ValueError(
                        f"{task_type}: `source_sequence` must have {IMAGE_CLIP_FRAMES} frames, "
                        f"got {len(frames)} for batch index {bi}."
                    )

    def sample_task(self, loader_name: Optional[str] = None) -> str:
        """Return the task of a configured dataset."""
        if loader_name is None:
            raise ValueError("loader_name is required to resolve the task type.")
        return get_dataset_spec(loader_name).task

    def prepare_batch(self, batch: Dict[str, Any], loader_name: Optional[str] = None) -> Dict[str, Any]:
        """Encode a collated batch into the transformer inputs and the flow-matching target.
        ``loader_name``: the configured dataset the batch came from; its task selects the encoding."""
        unknown = sorted(key for key in batch if key not in _SAMPLE_KEYS and not key.startswith("__"))
        if unknown:
            raise ValueError(
                f"Unknown sample key(s) {unknown}. Allowed keys: {sorted(_SAMPLE_KEYS)} "
                "(see univedit.data.datasets.EditSample)."
            )


        # Task of the batch
        task_type = self.sample_task(loader_name=loader_name)
        if task_type in self.IMAGE_EDIT_TASKS:
            self._check_image_clip(batch, task_type)

        target_pixels = batch.get("target_pixels", None)
        if target_pixels is not None:
            target_pixels = target_pixels.to(self.device)
        if "target_latents" in batch:
            latents = batch["target_latents"].to(self.device)
        elif target_pixels is not None:
            latents = self.encode_vae(target_pixels, task_type=task_type)
        else:
            raise ValueError(
                f"{task_type} batch has no target: provide `target_pixels` or `target_latents` "
                "(see univedit.data.datasets.EditSample)."
            )

        if self.sp_enabled:
            latents = sync_tensor_for_sp(latents, self.sp_group)

        if self.sp_enabled:
            task_type = sync_tensor_for_sp(task_type, self.sp_group)
        
        # ---- v2v / tiv2v / 3d23d: sampled frames and first frame of the source ----
        _is_sp_rank0 = (not self.sp_enabled) or (dist.get_rank(self.sp_group) == 0)
        source_first_frame = None
        source_sampled_frames = None
        if task_type in ("v2v", "tiv2v", "3d23d") and _is_sp_rank0:
            source_sampled_frames = normalize_sampled_frames(
                batch.get("source_sampled_frames", None), latents.shape[0],
            )
            source_first_frame = normalize_semantic_images_np(
                batch.get("source_first_frame", None), latents.shape[0],
            )
            vae_spatial_ratio = getattr(
                self.pipeline, 'vae_spatial_compression_ratio', 16,
            )
            target_height = int(latents.shape[-2] * vae_spatial_ratio)
            target_width = int(latents.shape[-1] * vae_spatial_ratio)

            if source_sampled_frames is None or source_first_frame is None:
                source_sequences_for_semantics = None
                if batch.get("source_sequence", None) is not None:
                    source_sequences_for_semantics = normalize_batched_pil_sequences(
                        batch.get("source_sequence", None), latents.shape[0], "source_sequence"
                    )
                    num_video_frames = min(len(frames) for frames in source_sequences_for_semantics)

                if source_sequences_for_semantics is not None:
                    if num_video_frames >= 24 * 4:
                        nframes = 8
                    elif num_video_frames >= 24 * 3:
                        nframes = 6
                    else:
                        nframes = 4

                    source_first_frame, source_sampled_frames = get_semantic_images_np(
                        source_sequences_for_semantics, nframes=nframes,
                    )

                    if source_sampled_frames is not None:
                        resized_frames = []
                        for frame_list in source_sampled_frames:
                            if frame_list is None:
                                resized_frames.append(frame_list)
                                continue
                            resized_frames.append([
                                resize_and_center_crop_pil(
                                    frame, target_width, target_height,
                                )
                                for frame in frame_list
                            ])
                        source_sampled_frames = resized_frames

                    if source_first_frame is not None:
                        source_first_frame = np.stack([
                            resize_and_center_crop(
                                image, target_width, target_height,
                            )
                            for image in source_first_frame
                        ], axis=0)

            if source_first_frame is None and source_sampled_frames is not None:
                source_first_frame = np.stack([
                    np.array(
                        resize_and_center_crop_pil(
                            frame_list[0], target_width, target_height,
                        ),
                        dtype=np.uint8,
                    )
                    for frame_list in source_sampled_frames
                ], axis=0)

            batch["source_sampled_frames"] = source_sampled_frames
            batch["source_first_frame"] = source_first_frame
        if _is_sp_rank0:
            cond_latents = self.get_condition(latents, task_type, batch=batch)
        else:
            b, c, f, h, w = latents.shape
            cond_latents = torch.zeros([b, c + 1, f, h, w], device=latents.device, dtype=latents.dtype)
        if self.sp_enabled:
            cond_latents = sync_tensor_for_sp(cond_latents, self.sp_group)

        prompts = batch["text"]
        # One modality token per sample
        modality_tokens = [get_special_token_from_task(task_type, strict=True)] * len(prompts)

        if self.sp_enabled:
            prompts = sync_tensor_for_sp(prompts, self.sp_group)
            modality_tokens = sync_tensor_for_sp(modality_tokens, self.sp_group)
        text_emb, text_mask, text_emb_2, text_mask_2, deepstack_hidden_states = self.encode_text_for_task(
            prompts,
            task_type,
            batch,
            modality_tokens=modality_tokens,
        )
        if self.sp_enabled:
            text_emb = sync_tensor_for_sp(text_emb, self.sp_group)
            text_mask = sync_tensor_for_sp(text_mask, self.sp_group)
            if text_emb_2 is not None:
                text_emb_2 = sync_tensor_for_sp(text_emb_2, self.sp_group)
            if text_mask_2 is not None:
                text_mask_2 = sync_tensor_for_sp(text_mask_2, self.sp_group)
            if deepstack_hidden_states is not None:
                deepstack_hidden_states = sync_tensor_for_sp(deepstack_hidden_states, self.sp_group)
        
        byt5_text_states = None
        byt5_text_mask = None
        if self.byt5_kwargs["byt5_model"] is not None:
            if "byt5_text_ids" in batch and batch["byt5_text_ids"] is not None:
                byt5_text_ids = batch["byt5_text_ids"].to(self.device)
                byt5_text_mask = batch["byt5_text_mask"].to(self.device)
                if self.sp_enabled:
                    byt5_text_ids = sync_tensor_for_sp(byt5_text_ids, self.sp_group)
                    byt5_text_mask = sync_tensor_for_sp(byt5_text_mask, self.sp_group)
                byt5_text_states, byt5_text_mask = self.encode_byt5(byt5_text_ids, byt5_text_mask)
            else:
                byt5_embeddings_list = []
                byt5_mask_list = []
                with torch.no_grad():
                    for prompt in prompts:
                        emb, mask = self.pipeline._process_single_byt5_prompt(prompt, self.device)
                        byt5_embeddings_list.append(emb)
                        byt5_mask_list.append(mask)
                
                byt5_text_states = torch.cat(byt5_embeddings_list, dim=0)
                byt5_text_mask = torch.cat(byt5_mask_list, dim=0)
                if self.sp_enabled:
                    byt5_text_states = sync_tensor_for_sp(byt5_text_states, self.sp_group)
                    byt5_text_mask = sync_tensor_for_sp(byt5_text_mask, self.sp_group)
        
        # ======================== Vision encoder (SigLIP) ========================
        vision_num_tokens = getattr(self.pipeline.config, 'vision_num_semantic_tokens', 257)
        vision_dim = getattr(self.pipeline.config, 'vision_states_dim', 1152)
        vision_states = torch.zeros(
            latents.shape[0], vision_num_tokens, vision_dim,
            device=self.device, dtype=self.transformer.dtype,
        )

        vae_spatial_ratio = getattr(self.pipeline, 'vae_spatial_compression_ratio', 16)
        vis_target_h = int(latents.shape[-2] * vae_spatial_ratio)
        vis_target_w = int(latents.shape[-1] * vae_spatial_ratio)

        if task_type in ("i2i", "ti_i2i"):
            source_images = self._image_task_sources(batch, latents.shape[0])
            if source_images is not None:
                reference_images = normalize_batched_pil_images(
                    batch.get("reference_image", None),
                    latents.shape[0],
                    "reference_image",
                )
                vision_images = []
                for bi, imgs in enumerate(source_images):
                    visual_inputs = list(imgs)
                    if task_type == "ti_i2i" and reference_images is not None:
                        visual_inputs += reference_images[bi]
                    vision_images.append(visual_inputs)
                vision_states = self.encode_pil_to_vision_states(
                    vision_images,
                    target_height=vis_target_h,
                    target_width=vis_target_w,
                )

        elif task_type in ("v2v", "3d23d"):
            # First frame of the source video
            _sem_np = batch.get("source_first_frame", None)
            if self.vision_encoder is not None and _sem_np is not None:
                with torch.no_grad():
                    encoder_output = self.vision_encoder.encode_images(_sem_np)
                vision_states = encoder_output.last_hidden_state.to(
                    device=self.device, dtype=self.transformer.dtype
                )

        elif task_type == "tiv2v":
            # Mean of the source first-frame states and the reference-image states
            vs_video = None
            _sem_np = batch.get("source_first_frame", None)
            if self.vision_encoder is not None and _sem_np is not None:
                with torch.no_grad():
                    encoder_output = self.vision_encoder.encode_images(_sem_np)
                vs_video = encoder_output.last_hidden_state.to(
                    device=self.device, dtype=self.transformer.dtype
                )
            
            vs_img = None
            if self.vision_encoder is not None and "reference_image" in batch and batch["reference_image"] is not None:
                reference_images = normalize_batched_pil_images(
                    batch["reference_image"],
                    latents.shape[0],
                    "reference_image",
                )
                vs_img = self.encode_pil_to_vision_states(
                    reference_images,
                    target_height=vis_target_h,
                    target_width=vis_target_w,
                )
            
            if vs_video is not None and vs_img is not None:
                vision_states = (vs_video + vs_img) / 2.0
            elif vs_video is not None:
                vision_states = vs_video
            elif vs_img is not None:
                vision_states = vs_img
        if self.sp_enabled and vision_states is not None:
            vision_states = sync_tensor_for_sp(vision_states, self.sp_group)
        
        noise = torch.randn_like(latents)
        timesteps = self.timestep_sampler.sample(latents.shape[0], device=self.device)
        timesteps = timestep_transform(timesteps, self.config.num_train_timesteps, self.config.train_timestep_shift)
        
        if self.sp_enabled:
            noise = sync_tensor_for_sp(noise, self.sp_group)
            timesteps = sync_tensor_for_sp(timesteps, self.sp_group)
        
        latents_noised = self.noise_schedule.forward(latents, noise, timesteps)
        target = noise - latents

        result = {
            "latents_noised": latents_noised,
            "cond_latents": cond_latents,
            "timesteps": timesteps,
            "target": target,
            "text_emb": text_emb,
            "text_emb_2": text_emb_2,
            "text_mask": text_mask,
            "text_mask_2": text_mask_2,
            "byt5_text_states": byt5_text_states,
            "byt5_text_mask": byt5_text_mask,
            "vision_states": vision_states,
            "task_type": task_type,
            "deepstack_hidden_states": deepstack_hidden_states,
        }
        return result
    
    def _begin_auxiliary_loss(self, inputs: Dict[str, Any]) -> Any:
        """Subclass hook, called after prepare_batch and before the forward pass.
        The return value is passed to _auxiliary_loss as ``state``."""
        return None

    def _auxiliary_loss(
        self,
        inputs: Dict[str, Any],
        state: Any,
        diffusion_loss: torch.Tensor,
        model_dtype: torch.dtype,
        extra_kwargs: Dict[str, Any],
        deepstack_hidden_states: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Subclass hook, called after the diffusion loss. Returns a loss that is
        added to it (or None) and the metrics to report with the step."""
        return None, {}

    def train_step(
        self,
        batch: Dict[str, Any],
        loader_name: Optional[str] = None,
        accumulation_steps: Optional[int] = None,
        optimizer_step: bool = True,
    ) -> Dict[str, Any]:
        """Run one forward/backward pass on a batch, step the optimizers if ``optimizer_step``,
        and return the metrics."""
        accumulation_steps = self.config.gradient_accumulation_steps if accumulation_steps is None else accumulation_steps
        inputs = self.prepare_batch(batch, loader_name=loader_name)
        aux_state = self._begin_auxiliary_loss(inputs)
        latents_input = torch.cat([inputs["latents_noised"], inputs["cond_latents"]], dim=1)
        model_dtype = torch.bfloat16 if self.config.dtype == "bf16" else torch.float32

        extra_kwargs = {}
        if inputs["byt5_text_states"] is not None:
            extra_kwargs["byt5_text_states"] = inputs["byt5_text_states"].to(dtype=model_dtype)
            extra_kwargs["byt5_text_mask"] = inputs["byt5_text_mask"]

        deepstack_hidden_states = inputs["deepstack_hidden_states"]
        if deepstack_hidden_states is not None:
            deepstack_hidden_states = deepstack_hidden_states.to(dtype=model_dtype)

        with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=(model_dtype == torch.bfloat16)):
            model_pred = self.transformer(
                latents_input.to(dtype=model_dtype),
                inputs["timesteps"],
                text_states=inputs["text_emb"].to(dtype=model_dtype),
                text_states_2=inputs["text_emb_2"].to(dtype=model_dtype) if inputs["text_emb_2"] is not None else None,
                encoder_attention_mask=inputs["text_mask"],
                vision_states=inputs["vision_states"].to(dtype=model_dtype) if inputs["vision_states"] is not None else None,
                mask_type=inputs["task_type"],
                extra_kwargs=extra_kwargs if extra_kwargs else None,
                return_dict=False,
                all_stack_text_states=deepstack_hidden_states,
            )[0]

        target = inputs["target"].to(dtype=model_pred.dtype)
        per_sample_loss = (model_pred.float() - target.float()).pow(2).flatten(1).mean(dim=1)
        raw_loss = per_sample_loss.mean()

        aux_loss, aux_metrics = self._auxiliary_loss(
            inputs, aux_state, raw_loss, model_dtype, extra_kwargs, deepstack_hidden_states,
        )
        if aux_loss is not None:
            raw_loss = raw_loss + aux_loss

        scaled_loss = raw_loss / accumulation_steps
        scaled_loss.backward()

        if optimizer_step:
            self._sync_modality_token_embedding_grads()
            if self.config.max_grad_norm > 0:
                grad_norm = self._clip_trainable_grad_norms()
            else:
                grad_norm = torch.tensor(0.0)

            _grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            _skip_optimizer_step = math.isnan(_grad_norm_val) or math.isinf(_grad_norm_val)
            if _skip_optimizer_step:
                logger.warning(f"Skipping optimizer step due to invalid grad norm ({_grad_norm_val}); zeroing gradients")

            if not _skip_optimizer_step:
                self.optimizer.step()
                if self.modality_token_optimizer is not None:
                    self.modality_token_optimizer.step()
                    self._sync_modality_token_embedding_weights_for_sp()
                self.lr_scheduler.step()
                if self.modality_token_lr_scheduler is not None:
                    self.modality_token_lr_scheduler.step()
            self.optimizer.zero_grad()
            if self.modality_token_optimizer is not None:
                self.modality_token_optimizer.zero_grad()
        else:
            grad_norm = torch.tensor(0.0)

        timesteps_cpu = inputs["timesteps"].detach().float().cpu()
        per_sample_loss_cpu = per_sample_loss.detach().float().cpu()

        metrics = {
            "loss": raw_loss.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, "get_last_lr") else self.config.learning_rate,
            "modality_token_lr": (
                self.modality_token_lr_scheduler.get_last_lr()[0]
                if self.modality_token_lr_scheduler is not None
                and hasattr(self.modality_token_lr_scheduler, "get_last_lr")
                else self.config.modality_token_learning_rate
            ) if self.modality_token_optimizer is not None else None,
            "task_type": inputs["task_type"],
            "loader_name": loader_name,
            "accumulation_steps": accumulation_steps,
            "optimizer_step": optimizer_step,
            "timestep_mean": timesteps_cpu.mean().item(),
            "timestep_min": timesteps_cpu.min().item(),
            "timestep_max": timesteps_cpu.max().item(),
            "sample_count": int(per_sample_loss_cpu.numel()),
        }
        metrics.update(aux_metrics)

        return metrics
    
    def save_checkpoint(self, step: int):
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        transformer_dir = os.path.join(checkpoint_dir, "transformer")
        save_full_checkpoint = not self.config.use_lora
        
        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
        if self.world_size > 1:
            dist.barrier()
        
        if self.config.use_lora and hasattr(self.transformer, "save_lora_adapter"):
            has_adapters = bool(getattr(self.transformer, "peft_config", None))
            if self.world_size > 1:
                has_adapters_tensor = torch.tensor(
                    int(has_adapters), device=self.device, dtype=torch.int32
                )
                dist.all_reduce(has_adapters_tensor, op=dist.ReduceOp.MIN)
                has_adapters = bool(has_adapters_tensor.item())

            if not has_adapters:
                raise RuntimeError(
                    "No LoRA adapter found in the model on at least one rank; "
                    "aborting checkpoint save to avoid distributed deadlock."
                )

            lora_dir = os.path.join(checkpoint_dir, "lora")
            os.makedirs(lora_dir, exist_ok=True)
            
            if hasattr(self.transformer, "peft_config") and self.transformer.peft_config:
                adapter_names = list(self.transformer.peft_config.keys())
                if self.is_main_process:
                    logger.info(f"Saving {len(adapter_names)} LoRA adapter(s): {adapter_names}")

                for adapter_name in adapter_names:
                    adapter_dir = os.path.join(lora_dir, adapter_name)
                    os.makedirs(adapter_dir, exist_ok=True)
                    self.transformer.save_lora_adapter(
                        save_directory=adapter_dir,
                        adapter_name=adapter_name,
                        safe_serialization=True,
                    )
                    if self.is_main_process:
                        logger.info(f"LoRA adapter '{adapter_name}' saved to {adapter_dir}")
            
            if self.world_size > 1:
                dist.barrier()
        
        if save_full_checkpoint:
            # Full transformer checkpoint
            model_state_dict = get_model_state_dict(self.transformer)
            dcp.save(
                state_dict={"model": model_state_dict},
                checkpoint_id=transformer_dir,
            )
        elif self.is_main_process:
            logger.info("Skipping full transformer checkpoint; saved lightweight LoRA/token state only")

        save_main_optimizer_dcp = (
            save_full_checkpoint
            or (self.config.use_lora and self._optimizer_uses_dtensor())
        )
        if save_main_optimizer_dcp:
            optimizer_state_dict = get_optimizer_state_dict(
                self.transformer,
                self.optimizer,
            )
            optimizer_dir = os.path.join(checkpoint_dir, "optimizer")
            dcp.save(
                state_dict={"optimizer": optimizer_state_dict},
                checkpoint_id=optimizer_dir,
            )
            if self.is_main_process and not save_full_checkpoint:
                logger.info("LoRA optimizer state saved with distributed checkpointing")

        completed_data_shards = self._gather_completed_data_shards()
        data_epoch_counts = self._gather_data_epoch_counts()

        if self.is_main_process:
            training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
            training_state = {
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "global_step": step,
                "micro_step": self.micro_step,
                "completed_data_shards": completed_data_shards,
                "data_epoch_counts": data_epoch_counts,
                "dataset_shuffle_seed_offset": self.config.dataset_seed_offset or 0,
            }
            if not save_full_checkpoint and self.config.use_lora and not save_main_optimizer_dcp:
                training_state["optimizer"] = self.optimizer.state_dict()
            if self.modality_token_embedding is not None:
                modality_token_state_path = os.path.join(checkpoint_dir, "modality_token_embeddings.pt")
                modality_token_state = {
                    "tokens": list(SPECIAL_TOKENS),
                    "token_ids": self.modality_token_ids.detach().cpu(),
                    "weight": self.modality_token_embedding.detach().cpu(),
                }
                if self.modality_token_optimizer is not None:
                    training_state["modality_token_optimizer"] = self.modality_token_optimizer.state_dict()
                if self.modality_token_lr_scheduler is not None:
                    training_state["modality_token_lr_scheduler"] = self.modality_token_lr_scheduler.state_dict()
                torch.save(modality_token_state, modality_token_state_path)
            torch.save(training_state, training_state_path)
            with open(os.path.join(checkpoint_dir, "completed_data_shards.json"), "w", encoding="utf-8") as f:
                json.dump(completed_data_shards, f, indent=2, sort_keys=True)
            with open(os.path.join(checkpoint_dir, "data_epoch_counts.json"), "w", encoding="utf-8") as f:
                json.dump(data_epoch_counts, f, indent=2, sort_keys=True)
        
        if self.world_size > 1:
            dist.barrier()
        
        if self.is_main_process:
            logger.info(f"Checkpoint saved at step {step} to {checkpoint_dir}")

    def load_pretrained_lora(self, lora_dir: str):
        checkpoint_lora_dir = os.path.join(lora_dir, "lora", "default")
        if os.path.isdir(checkpoint_lora_dir):
            lora_dir = checkpoint_lora_dir
        adapter_name = "default"
        if adapter_name in getattr(self.transformer, "peft_config", {}):
            load_lora_weights_into_existing_adapter(
                self.transformer, lora_dir, adapter_name, is_main_process=self.is_main_process,
            )
            return
        self.transformer.load_lora_adapter(
            pretrained_model_name_or_path_or_dict=lora_dir,
            prefix=None,
            adapter_name=adapter_name,
            weight_name="pytorch_lora_weights.safetensors",
            use_safetensors=True,
            hotswap=False,
        )

    def load_pretrained_modality_token_embeddings(self, modality_token_path: str):
        if os.path.isdir(modality_token_path):
            modality_token_path = os.path.join(modality_token_path, "modality_token_embeddings.pt")
        if not os.path.exists(modality_token_path):
            raise FileNotFoundError(
                f"Pretrained modality token embedding checkpoint not found: {modality_token_path}"
            )
        if self.modality_token_embedding is None:
            raise RuntimeError(
                "Cannot load pretrained modality token embeddings unless "
                "train_modality_token_embeddings is enabled."
            )

        modality_token_state = torch.load(
            modality_token_path, map_location=self.device, weights_only=True
        )
        state_token_ids = modality_token_state.get("token_ids")
        if state_token_ids is not None and list(state_token_ids.cpu().tolist()) != list(self.modality_token_ids.cpu().tolist()):
            logger.warning(
                "Pretrained modality token ids differ from current tokenizer ids; "
                f"checkpoint={state_token_ids.cpu().tolist()}, "
                f"current={self.modality_token_ids.cpu().tolist()}. "
                "Using current tokenizer ids."
            )
        weights = modality_token_state["weight"].to(
            device=self.modality_token_embedding.device,
            dtype=self.modality_token_embedding.dtype,
        )
        if weights.shape != self.modality_token_embedding.shape:
            raise ValueError(
                "Pretrained modality token embedding shape mismatch: "
                f"checkpoint={tuple(weights.shape)}, "
                f"current={tuple(self.modality_token_embedding.shape)}"
            )
        with torch.no_grad():
            self.modality_token_embedding.copy_(weights)
        if self.is_main_process:
            logger.info(f"Pretrained modality token embeddings loaded from {modality_token_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

        if self.is_main_process:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        if self.world_size > 1:
            dist.barrier()
        
        if self.config.use_lora:
            lora_dir = os.path.join(checkpoint_path, "lora", "default")
            if os.path.exists(lora_dir):
                self.load_pretrained_lora(lora_dir)
                if self.is_main_process:
                    logger.info(f"LoRA adapter loaded from {lora_dir}")
            elif self.is_main_process:
                logger.warning(f"LoRA adapter checkpoint not found from {lora_dir}")
        
        
        transformer_dir = os.path.join(checkpoint_path, "transformer")
        if os.path.exists(transformer_dir):
            model_state_dict = get_model_state_dict(self.transformer)
            dcp.load(
                state_dict={"model": model_state_dict},
                checkpoint_id=transformer_dir,
            )
            if self.is_main_process:
                logger.info("Transformer model state loaded")
        else:
            logger.warning(f"Transformer dcp checkpoint not found from {checkpoint_path}")

        optimizer_dir = os.path.join(checkpoint_path, "optimizer")
        main_optimizer_loaded_from_dcp = False
        if os.path.exists(optimizer_dir):
            optimizer_state_dict = get_optimizer_state_dict(
                self.transformer,
                self.optimizer,
            )
            dcp.load(
                state_dict={"optimizer": optimizer_state_dict},
                checkpoint_id=optimizer_dir,
            )
            main_optimizer_loaded_from_dcp = True
            if self.is_main_process:
                logger.info("Optimizer state loaded")
        
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.device, weights_only=True)
            self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])
            self.global_step = training_state.get("global_step", 0)
            self.micro_step = training_state.get("micro_step", 0)
            completed_data_shards = training_state.get("completed_data_shards")
            if completed_data_shards is None:
                completed_data_shards_path = os.path.join(checkpoint_path, "completed_data_shards.json")
                if os.path.exists(completed_data_shards_path):
                    with open(completed_data_shards_path, "r", encoding="utf-8") as f:
                        completed_data_shards = json.load(f)
            self.completed_data_shards = self._normalize_completed_data_shards_state(
                completed_data_shards or {}
            )
            data_epoch_counts = training_state.get("data_epoch_counts")
            if data_epoch_counts is None:
                data_epoch_counts_path = os.path.join(checkpoint_path, "data_epoch_counts.json")
                if os.path.exists(data_epoch_counts_path):
                    with open(data_epoch_counts_path, "r", encoding="utf-8") as f:
                        data_epoch_counts = json.load(f)
            self.data_epoch_counts = {
                str(lname): int(count) for lname, count in (data_epoch_counts or {}).items()
            }
            if (
                self.config.use_lora
                and "optimizer" in training_state
            ):
                if main_optimizer_loaded_from_dcp:
                    if self.is_main_process:
                        logger.info("Skipping training_state optimizer because DCP optimizer was loaded")
                elif contains_dtensor(training_state["optimizer"]):
                    if self.is_main_process:
                        logger.warning(
                            "Skipping training_state optimizer because it contains DTensor "
                            "state saved from a rank-local mesh. Continuing with fresh optimizer "
                            "momenta while preserving scheduler/global_step."
                        )
                else:
                    self.optimizer.load_state_dict(training_state["optimizer"])
            if (
                self.modality_token_optimizer is not None
                and "modality_token_optimizer" in training_state
            ):
                self.modality_token_optimizer.load_state_dict(training_state["modality_token_optimizer"])
            if (
                self.modality_token_lr_scheduler is not None
                and "modality_token_lr_scheduler" in training_state
            ):
                self.modality_token_lr_scheduler.load_state_dict(training_state["modality_token_lr_scheduler"])
            if self.is_main_process:
                completed_count = sum(len(shards) for shards in self.completed_data_shards.values())
                logger.info(
                    f"Training state loaded: global_step={self.global_step}, "
                    f"micro_step={self.micro_step}, completed_data_shards={completed_count}, "
                    f"data_epoch_counts={self.data_epoch_counts}"
                )

        if self.modality_token_embedding is not None:
            modality_token_state_path = os.path.join(checkpoint_path, "modality_token_embeddings.pt")
            if not os.path.exists(modality_token_state_path):
                raise FileNotFoundError(
                    f"Modality token embedding checkpoint not found: {modality_token_state_path}"
                )
            modality_token_state = torch.load(
                modality_token_state_path, map_location=self.device, weights_only=True
            )
            state_token_ids = modality_token_state.get("token_ids")
            if state_token_ids is not None and list(state_token_ids.cpu().tolist()) != list(self.modality_token_ids.cpu().tolist()):
                logger.warning(
                    "Checkpoint modality token ids differ from current tokenizer ids; "
                    f"checkpoint={state_token_ids.cpu().tolist()}, "
                    f"current={self.modality_token_ids.cpu().tolist()}. "
                    "Using current tokenizer ids."
                )
            weights = modality_token_state["weight"].to(
                device=self.modality_token_embedding.device,
                dtype=self.modality_token_embedding.dtype,
            )
            if weights.shape != self.modality_token_embedding.shape:
                raise ValueError(
                    "Modality token embedding shape mismatch: "
                    f"checkpoint={tuple(weights.shape)}, "
                    f"current={tuple(self.modality_token_embedding.shape)}"
                )
            with torch.no_grad():
                self.modality_token_embedding.copy_(weights)
            if self.is_main_process:
                logger.info(f"Modality token embeddings loaded from {modality_token_state_path}")
        
        if self.world_size > 1:
            global_step_tensor = torch.tensor(self.global_step, device=self.device)
            dist.broadcast(global_step_tensor, src=0)
            self.global_step = global_step_tensor.item()
            micro_step_tensor = torch.tensor(self.micro_step, device=self.device)
            dist.broadcast(micro_step_tensor, src=0)
            self.micro_step = micro_step_tensor.item()
            completed_shards_object = [
                self._completed_data_shards_state() if self.is_main_process else None
            ]
            dist.broadcast_object_list(completed_shards_object, src=0)
            self.completed_data_shards = self._normalize_completed_data_shards_state(
                completed_shards_object[0]
            )
        
        if self.world_size > 1:
            dist.barrier()
        
        if self.is_main_process:
            logger.info(f"Checkpoint loaded successfully. Resuming from step {self.global_step}")

    def _sync_interval_trackers_to_global_step(self):
        if self.config.save_interval > 0:
            self._last_save_bucket = self.global_step // self.config.save_interval
        else:
            self._last_save_bucket = 0
        self._last_checkpoint_step = self.global_step

    @staticmethod
    def _aggregate_update_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics_list:
            raise ValueError("Cannot aggregate an empty metrics list.")

        final_metrics = metrics_list[-1]
        total_samples = sum(max(int(metrics.get("sample_count", 0)), 0) for metrics in metrics_list)
        if total_samples <= 0:
            total_samples = len(metrics_list)

        aggregated = dict(final_metrics)
        aggregated["sample_count"] = total_samples
        aggregated["loss"] = sum(
            float(metrics["loss"]) * max(int(metrics.get("sample_count", 0)), 1)
            for metrics in metrics_list
        ) / total_samples
        aggregated["timestep_mean"] = sum(
            float(metrics["timestep_mean"]) * max(int(metrics.get("sample_count", 0)), 1)
            for metrics in metrics_list
        ) / total_samples
        aggregated["timestep_min"] = min(float(metrics["timestep_min"]) for metrics in metrics_list)
        aggregated["timestep_max"] = max(float(metrics["timestep_max"]) for metrics in metrics_list)
        aggregated["micro_batch_count"] = len(metrics_list)
        return aggregated

    def _train_step_and_log(
        self,
        batch,
        loader_name: Optional[str] = None,
        accumulation_steps: Optional[int] = None,
        accumulation_index: Optional[int] = None,
        optimizer_step: bool = True,
    ):
        """Execute one training step and handle logging/checkpointing."""
        metrics = self.train_step(
            batch,
            loader_name=loader_name,
            accumulation_steps=accumulation_steps,
            optimizer_step=optimizer_step,
        )
        self.micro_step += 1
        metrics["micro_step"] = self.micro_step
        self._pending_update_metrics.append(metrics)

        if not metrics["optimizer_step"]:
            return

        metrics = self._aggregate_update_metrics(self._pending_update_metrics)
        metrics["micro_step"] = self.micro_step
        self._pending_update_metrics.clear()

        completed_step = self.global_step + 1
        should_log = (
            self.config.log_interval > 0
            and completed_step % self.config.log_interval == 0
        )
        if should_log and self.is_main_process:
            loader_info = f"Loader: {loader_name} | " if loader_name else ""
            accum_info = ""
            if accumulation_index is not None and accumulation_steps is not None:
                accum_info = f"Accum: {accumulation_index + 1}/{accumulation_steps} | "
            logger.info(
                f"Step {completed_step}/{self.config.max_steps} | "
                f"Micro Step: {self.micro_step} | "
                f"{loader_info}"
                f"{accum_info}"
                f"Task: {metrics['task_type']} | "
                f"Loss: {metrics['loss']:.6f} | "
                f"Grad Norm: {metrics['grad_norm']:.4f} | "
                f"LR: {metrics['lr']:.2e}"
            )
            self._log_metrics_to_wandb(metrics, step=completed_step)

        self.global_step = completed_step

        if self.config.save_interval > 0:
            save_bucket = completed_step // self.config.save_interval
            if save_bucket > self._last_save_bucket:
                self._last_save_bucket = save_bucket
                self.save_checkpoint(completed_step)
                self._last_checkpoint_step = completed_step
                if self.world_size > 1:
                    dist.barrier()

    def train(self, dataloader: MultiDataloader):
        if not isinstance(dataloader, MultiDataloader):
            raise TypeError(
                "train() expects a MultiDataloader (one loader per configured dataset name), "
                f"got {type(dataloader).__name__}."
            )
        if self.is_main_process:
            logger.info("Starting training...")
            logger.info(f"Max optimizer update steps: {self.config.max_steps}")
            logger.info(f"Default per-GPU batch size: {self.config.batch_size}")
            logger.info(f"Learning rate: {self.config.learning_rate}")
        
        if self.config.resume_from_checkpoint is not None:
            self.load_checkpoint(self.config.resume_from_checkpoint)
            self._sync_interval_trackers_to_global_step()
            self.optimizer.zero_grad()
            if self.modality_token_optimizer is not None:
                self.modality_token_optimizer.zero_grad()

        self._attach_shard_progress(dataloader)
        if self.is_main_process:
            completed_count = sum(len(shards) for shards in self.completed_data_shards.values())
            logger.info(f"Attached dataloader shard progress; completed_data_shards={completed_count}")

        # Warm up the loaders and wait until every rank is ready
        if self.data_ctrl_group is not None:
            if self.is_main_process:
                logger.info("Warming up dataloaders before entering the training loop...")
            warm_up_start = time.time()
            dataloader.warm_up()
            if self.is_main_process:
                logger.info(f"Local warm_up done in {time.time() - warm_up_start:.1f}s; waiting on data-readiness barrier...")
            dist.barrier(group=self.data_ctrl_group)
            if self.is_main_process:
                logger.info("All ranks ready; entering training loop.")
            dataloader.set_epoch_sync_group(self.data_ctrl_group)
            if self.is_main_process:
                logger.info("Epoch-rollover sync armed on the data-control group.")

        self.transformer.train()

        if self.is_main_process:
            logger.info(f"Using MultiDataloader with {len(dataloader.names)} dataloaders:")
            for name in dataloader.names:
                logger.info(f"  - {name}: update_prob={dataloader.probabilities[name]:.2f}")
        
        while self.global_step < self.config.max_steps:
            try:
                loader_name = dataloader.sample_name()
            except RuntimeError as _e:
                if "completed one epoch" in str(_e):
                    if self.is_main_process:
                        logger.info(f"[train] Stopping early: {_e}")
                    break
                raise
            configured_accumulation_steps = gradient_accumulation_steps_for_loader(
                self.config,
                loader_name,
            )
            accumulation_steps = configured_accumulation_steps
            if self.is_main_process:
                per_gpu_batch_size = batch_size_for_loader(self.config, loader_name)
                logger.info(
                    f"Step dataloader | Step: {self.global_step + 1}/{self.config.max_steps} | "
                    f"Loader: {loader_name} | "
                    f"Per-GPU Batch Size: {per_gpu_batch_size} | "
                    f"Gradient Accumulation: {accumulation_steps}"
                )
            for accumulation_index in range(accumulation_steps):
                batch = dataloader.next_batch(loader_name)
                self._train_step_and_log(
                    batch,
                    loader_name=loader_name,
                    accumulation_steps=accumulation_steps,
                    accumulation_index=accumulation_index,
                    optimizer_step=(accumulation_index + 1 == accumulation_steps),
                )

        if self.global_step > 0 and self._last_checkpoint_step != self.global_step:
            self.save_checkpoint(self.global_step)
            self._last_checkpoint_step = self.global_step
        
        if self.is_main_process:
            logger.info("Training completed!")
            if self.wandb_run is not None:
                self.wandb_run.finish()
        
        if self.world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
    


def _metric_name(value: Optional[str]) -> str:
    """A task / loader name as a W&B key component ("unknown" for None)."""
    if value is None:
        return "unknown"
    return str(value).replace("/", "_").replace(" ", "_")


def _require_one_image(images: List[List[Any]], task_type: str, what: str) -> None:
    """Raise unless every sample of an image batch carries exactly one image."""
    for bi, imgs in enumerate(images):
        if len(imgs) != 1:
            raise ValueError(
                f"{task_type} expects exactly one {what} per sample, "
                f"got {len(imgs)} for batch index {bi}"
            )


def merge_data_epoch_counts(
    gathered: List[Optional[Dict[str, int]]],
) -> Dict[str, int]:
    """Merge per-rank ``{loader: epochs}`` counts, keeping the maximum per loader."""
    merged: Dict[str, int] = {}
    for rank_state in gathered:
        for lname, count in (rank_state or {}).items():
            merged[lname] = max(merged.get(lname, 0), count)
    return merged


def _load_training_state(checkpoint_dir: str) -> Optional[dict]:
    """Contents of the checkpoint's training_state.pt, or None if missing or unreadable."""
    if not checkpoint_dir:
        return None
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if not os.path.exists(state_path):
        return None
    try:
        return torch.load(state_path, map_location="cpu", weights_only=True)
    except Exception:
        return None


def _next_seed_offset(state: Optional[dict]) -> int:
    """The checkpoint's seed offset plus one (0 without a usable training state)."""
    try:
        return int(state.get("dataset_shuffle_seed_offset", 0)) + 1
    except Exception:
        return 0


def resolve_dataset_seed_offset(config: TrainingConfig) -> None:
    """Set ``config.dataset_seed_offset`` when it is None: 0 for a fresh run, the
    checkpoint's offset plus one for a resumed run."""
    if config.dataset_seed_offset is not None:
        return
    config.dataset_seed_offset = 0
    if not config.resume_from_checkpoint:
        return
    config.dataset_seed_offset = _next_seed_offset(_load_training_state(config.resume_from_checkpoint))
    logger.info(
        f"Resumed run: dataset seed offset {config.dataset_seed_offset} "
        f"(effective seed {dataset_seed(config)})."
    )


def main(trainer_cls: Optional[type] = None, argv: Optional[List[str]] = None) -> None:
    """Parse ``argv``, build the loaders and train with ``trainer_cls`` (default :class:`HunyuanVideoTrainer`)."""
    if trainer_cls is None:
        trainer_cls = HunyuanVideoTrainer
    parser = build_arg_parser()
    trainer_cls.add_cli_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)

    # Datasets and their sampling probabilities
    if config.data_config:
        load_data_config(config.data_config)
    prob_dict = sampling_probabilities()
    if not prob_dict:
        parser.error("--data_config is required.")
    known_names = set(GROUP_TASKS) | set(list_datasets())
    for option, values in (
        ("--dataloader_batch_sizes", config.dataloader_batch_sizes),
        ("--dataloader_gradient_accumulation_steps", config.dataloader_gradient_accumulation_steps),
    ):
        unknown = sorted(set(values) - known_names)
        if unknown:
            parser.error(
                f"{option}: {unknown} are neither data-config groups {sorted(GROUP_TASKS)} "
                f"nor datasets of --data_config {list_datasets()}."
            )
    resolve_dataset_seed_offset(config)

    trainer = trainer_cls(config, **trainer_cls.trainer_kwargs_from_args(args))

    dataloader = build_multi_dataloader(
        config, prob_dict, dp_rank=trainer.dp_rank, dp_size=trainer.dp_size,
    )

    if trainer.is_main_process:
        logger.info(f"Created MultiDataloader with probabilities: {prob_dict}")
        logger.info(
            "Train tasks: "
            f"{ {name: get_dataset_spec(name).task for name in prob_dict} }"
        )
        effective_batch_sizes = {
            name: batch_size_for_loader(config, name)
            for name in prob_dict
        }
        effective_accumulation_steps = {
            name: gradient_accumulation_steps_for_loader(config, name)
            for name in prob_dict
        }
        logger.info(f"Train per-GPU batch sizes: {effective_batch_sizes}")
        logger.info(f"Train gradient accumulation steps: {effective_accumulation_steps}")

    trainer.train(dataloader)


if __name__ == "__main__":
    main()
