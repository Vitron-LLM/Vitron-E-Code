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

"""Training configuration and its command-line interface: :class:`TrainingConfig`, the
argument parser, and the options of geometry post-training (:class:`GeometryArgs`).
"""

import argparse
import math
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional


def str_to_bool(value: Any) -> bool:
    """Convert a command-line value (true/false, 1/0, yes/no, on/off) to bool; None gives True."""
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


class SNRType(str, Enum):
    """Timestep sampling strategy."""

    UNIFORM = "uniform"
    LOGNORM = "lognorm"
    MIX = "mix"
    MODE = "mode"


@dataclass
class TrainingConfig:
    # Model paths
    pretrained_model_root: str
    
    # Training parameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    lr_scheduler: str = "constant"
    max_steps: int = 10000
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_muon: bool = True
    train_modality_token_embeddings: bool = False
    modality_token_learning_rate: float = 3e-5
    modality_token_weight_decay: float = 0.0
    modality_token_warmup_steps: Optional[int] = None
    modality_token_lr_scheduler: str = "cosine"

    # Diffusion parameters
    num_train_timesteps: int = 1000
    train_timestep_shift: float = 3.0
    sampling_timestep_shift: float = 5.0  # Flow shift of the pipeline's sampling scheduler
    snr_type: SNRType = SNRType.LOGNORM  # Timestep sampling strategy: uniform, lognorm, mix, or mode
    
    # Text encoder deepstack and setclip settings
    deepstack: List[int] = field(default_factory=lambda: [8, 16, 24])  # Deepstack layer indices; [] disables
    setclip: bool = True  # Enable CLIP features in text encoder
    
    # FSDP configuration
    enable_fsdp: bool = True  # Enable FSDP for distributed training
    enable_gradient_checkpointing: bool = True  # Enable gradient checkpointing
    enable_text_encoder_fsdp: bool = False  # Shard frozen Qwen-VL text encoder parameters
    enable_text_encoder_gradient_checkpointing: bool = False  # Activation checkpointing for the Qwen-VL text encoder
    sp_size: int = 8  # Sequence parallelism size (must divide world_size evenly)
    dp_replicate: int = 1  # Data parallelism replicate size (must divide world_size evenly)
    
    # Data configuration
    batch_size: int = 1
    prefetch_batches: int = 1
    data_ctrl_timeout_hours: float = 6.0  # Timeout of the data-readiness barrier
    nccl_timeout_minutes: float = 30.0  # Timeout of the main compute process group

    # Output configuration
    output_dir: str = "./outputs"
    save_interval: int = 100
    log_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = "univedit"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: str = ""
    wandb_mode: str = "online"
    
    # Device configuration
    dtype: str = "bf16"  # "bf16" or "fp32"
    
    # Seed
    seed: int = 42
    
    # Resume training configuration
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint directory to resume from
    
    # LoRA configuration
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None  # Target modules for LoRA (default: all Linear layers)
    pretrained_lora_path: Optional[str] = None
    pretrained_modality_token_path: Optional[str] = None

    # Multi-dataloader configuration
    data_config: str = ""  # YAML file declaring the datasets (univedit.data.config)
    one_epoch_per_loader: bool = False  # Stop sampling a loader once it completes one full epoch
    dataset_seed_offset: Optional[int] = None  # Offset added to seed for the dataset shuffle order (None: set at start-up)


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the trainer's argument parser."""
    parser = argparse.ArgumentParser(
        description="UniVEdit: multi-task training for image, video and 3D editing"
    )

    # Model paths
    parser.add_argument("--pretrained_model_root", type=str, default='ckpts', help="Path to pretrained model")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial"],
        help="LR scheduler for the main optimizer (default: constant).",
    )
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum optimizer update steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps of the datasets that set none in the data config")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--train_timestep_shift", type=float, default=3.0, help="Train Timestep shift")
    parser.add_argument("--sampling_timestep_shift", type=float, default=5.0,
                        help="Flow shift of the pipeline's sampling scheduler (used when a trainer samples "
                             "from the model, e.g. the geometry rollout); the diffusion loss does not use it.")
    parser.add_argument("--flow_snr_type", type=str, default="lognorm", 
                        choices=["uniform", "lognorm", "mix", "mode"],
                        help="SNR type for flow matching: uniform, lognorm, mix, or mode (default: lognorm)")
    
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-GPU batch size of the datasets that set none in the data config")
    parser.add_argument("--prefetch_batches", type=int, default=1,
                        help="Batches each dataset's background thread reads ahead of training "
                             "(data is read and decoded in the main process); 0 disables the thread.")
    parser.add_argument(
        "--data_ctrl_timeout_hours",
        type=float,
        default=6.0,
        help=(
            "Timeout (hours) for the data-readiness barrier that gates entry into the "
            "training loop after warm_up(), kept separate from NCCL's compute-collective "
            "timeout so a slow first batch (e.g. shard download) is not killed by the "
            "compute watchdog."
        ),
    )
    parser.add_argument(
        "--nccl_timeout_minutes",
        type=float,
        default=30.0,
        help=(
            "Timeout (minutes) for the main compute process group "
            "(default_pg / mesh_fsdp_shard / mesh_dp_replicate / sp); "
            "defaults to torch's NCCL default of 30 minutes."
        ),
    )
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--save_interval", type=int, default=100, help="Checkpoint save interval in optimizer update steps")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval in optimizer update steps")
    parser.add_argument("--use_wandb", type=str_to_bool, nargs='?', const=True, default=False,
                        help="Upload training metrics to Weights & Biases (default: false)")
    parser.add_argument("--wandb_project", type=str, default="univedit",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name")
    parser.add_argument("--wandb_tags", type=str, default="",
                        help="Comma-separated Weights & Biases tags")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"],
                        help="Weights & Biases mode")
    
    # Other parameters
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"], help="Data type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deepstack", type=int, nargs='*', default=[8, 16, 24],
                        help="Deepstack layer indices for text encoder (default: 8 16 24). "
                             "Must match the value used during inference. Use --deepstack (empty) to disable.")
    parser.add_argument("--setclip", type=str_to_bool, nargs='?', const=True, default=True,
                        help="Enable CLIP features in text encoder (default: true). "
                             "Must match the value used during inference.")
    parser.add_argument("--use_muon", type=str_to_bool, nargs='?', const=True, default=True,
        help="Use Muon optimizer for training (default: true). "
             "Use --use_muon or --use_muon true/1 to enable, --use_muon false/0 to disable"
    )
    parser.add_argument(
        "--train_modality_token_embeddings",
        type=str_to_bool,
        nargs='?',
        const=True,
        default=False,
        help="Train only the registered modality special-token embedding rows with a separate AdamW optimizer.",
    )
    parser.add_argument(
        "--modality_token_learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the separate AdamW optimizer used by modality token embeddings.",
    )
    parser.add_argument(
        "--modality_token_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for the separate AdamW optimizer used by modality token embeddings.",
    )
    parser.add_argument(
        "--modality_token_warmup_steps",
        type=int,
        default=None,
        help="Warmup steps for the separate modality token embedding scheduler. Defaults to --warmup_steps.",
    )
    parser.add_argument(
        "--modality_token_lr_scheduler",
        type=str,
        default="cosine",
        choices=["constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial"],
        help="LR scheduler for modality token embeddings (default: cosine warmup + cosine decay).",
    )
    # FSDP and gradient checkpointing
    parser.add_argument(
        "--enable_fsdp", type=str_to_bool, nargs='?', const=True, default=True,
        help="Enable FSDP for distributed training (default: true). "
             "Use --enable_fsdp or --enable_fsdp true/1 to enable, --enable_fsdp false/0 to disable"
    )
    parser.add_argument(
        "--enable_gradient_checkpointing", type=str_to_bool, nargs='?', const=True, default=True,
        help="Enable gradient checkpointing (default: true). "
             "Use --enable_gradient_checkpointing or --enable_gradient_checkpointing true/1 to enable, "
             "--enable_gradient_checkpointing false/0 to disable"
    )
    parser.add_argument(
        "--enable_text_encoder_fsdp", type=str_to_bool, nargs='?', const=True, default=False,
        help="Enable FSDP2 sharding for the frozen Qwen-VL text encoder (default: false). "
             "Useful when training modality token embeddings with visual prompts."
    )
    parser.add_argument(
        "--enable_text_encoder_gradient_checkpointing", type=str_to_bool, nargs='?', const=True, default=False,
        help="Enable activation checkpointing for the Qwen-VL text encoder (default: false). "
             "This reduces activation memory when modality token embeddings require gradients."
    )
    parser.add_argument(
        "--sp_size", type=int, default=8,
        help="Sequence parallelism size (default: 8). Must evenly divide world_size. "
             "For example, if world_size=8, valid sp_size values are 1, 2, 4, 8."
    )
    parser.add_argument(
        "--dp_replicate", type=int, default=1,
        help="Data parallelism replicate size (default: 1). "
    )
    
    # Resume training parameters
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from (e.g., ./outputs/checkpoint-1000)")
    
    # LoRA parameters
    parser.add_argument("--use_lora", type=str_to_bool, nargs='?', const=True, default=False,
                        help="Enable LoRA training (default: false). "
                             "Use --use_lora or --use_lora true/1 to enable, --use_lora false/0 to disable")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha scaling parameter (default: 16)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout rate (default: 0.0)")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                        help="Target modules for LoRA (default: all Linear layers). "
                             "Example: --lora_target_modules img_attn_q img_attn_v img_mlp.fc1")
    parser.add_argument("--pretrained_lora_path", type=str, default=None,
                        help="Path to pretrained LoRA adapter to load. Accepts an adapter dir or a checkpoint dir containing lora/default.")
    parser.add_argument("--pretrained_modality_token_path", type=str, default=None,
                        help="Path to pretrained modality_token_embeddings.pt, or a checkpoint dir containing it.")
    # Data / multi-dataloader parameters
    parser.add_argument(
        "--data_config",
        type=str,
        default="",
        help=(
            "Required. YAML file that declares the datasets by path (a local directory or "
            "hf://datasets/<user>/<repo>), each a metadata.jsonl manifest with media files, "
            "tar shards in WebDataset layout or parquet shards, with its task, its sampling "
            "weight and optionally its batch size and accumulation steps "
            "(see configs/data.example.yaml and \"Data\" in the README). One dataset is "
            "sampled per optimizer update and run for its accumulation window."
        ),
    )
    parser.add_argument(
        "--one_epoch_per_loader",
        type=str_to_bool,
        nargs='?',
        const=True,
        default=False,
        help=(
            "Stop sampling a dataloader once it completes one full epoch (all shards seen once). "
            "When all loaders are exhausted training stops early. Default: false (loaders cycle indefinitely)."
        ),
    )
    parser.add_argument(
        "--dataset_seed_offset",
        type=int,
        default=None,
        help=(
            "Offset added to --seed for the dataset shuffle order. "
            "Default: 0 for a fresh run; a resumed run takes the offset stored in the "
            "checkpoint plus 1, so it reads the data in a new order (shards a dataset "
            "reported as completed are skipped in any order). "
            "Set an integer to use exactly that offset."
        ),
    )

    return parser


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Build a :class:`TrainingConfig` from the namespace parsed by :func:`build_arg_parser`."""
    return TrainingConfig(
        pretrained_model_root=args.pretrained_model_root,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        prefetch_batches=args.prefetch_batches,
        data_ctrl_timeout_hours=args.data_ctrl_timeout_hours,
        nccl_timeout_minutes=args.nccl_timeout_minutes,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        wandb_mode=args.wandb_mode,
        dtype=args.dtype,
        seed=args.seed,
        deepstack=args.deepstack if args.deepstack else [],
        setclip=args.setclip,
        enable_fsdp=args.enable_fsdp,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        enable_text_encoder_fsdp=args.enable_text_encoder_fsdp,
        enable_text_encoder_gradient_checkpointing=args.enable_text_encoder_gradient_checkpointing,
        sp_size=args.sp_size,
        use_muon=args.use_muon,
        train_modality_token_embeddings=args.train_modality_token_embeddings,
        modality_token_learning_rate=args.modality_token_learning_rate,
        modality_token_weight_decay=args.modality_token_weight_decay,
        modality_token_warmup_steps=args.modality_token_warmup_steps,
        modality_token_lr_scheduler=args.modality_token_lr_scheduler,
        dp_replicate=args.dp_replicate,
        train_timestep_shift=args.train_timestep_shift,
        sampling_timestep_shift=args.sampling_timestep_shift,
        snr_type=SNRType(args.flow_snr_type),
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        pretrained_lora_path=args.pretrained_lora_path,
        pretrained_modality_token_path=args.pretrained_modality_token_path,
        data_config=args.data_config,
        one_epoch_per_loader=args.one_epoch_per_loader,
        dataset_seed_offset=args.dataset_seed_offset,
    )


# ============================================================================
# Geometry post-training options (univedit.training.geometry_trainer)
# ============================================================================

@dataclass
class GeometryArgs:
    """Geometry-loss hyper-parameters (see add_geometry_cli_args)."""

    # ---- switch ----
    geo_enable: bool = True

    # ---- geometry extractor (the stitched 3D decoder) ----
    decoder_checkpoint_path: Optional[str] = None     # stitched 3D decoder checkpoint
    anysplat_path: Optional[str] = None               # AnySplat weights
    geo_extractor_grad_checkpointing: bool = True     # gradient checkpointing inside the decoder
    geo_extractor_resolution: int = 448               # resolution of the decoded frames fed to the extractor
    geo_extractor_max_frames: int = 17                # max frames entering the loss, 0 = no limit

    # ---- frames entering the loss ----
    # Frames 0, stride, 2*stride, ... of the decoder output enter the loss.
    geo_frame_stride: int = 4

    # ---- loss weights ----
    geo_loss_weight: float = 0.05        # weight of the whole geometry loss in the total loss
    geo_camera_loss_weight: float = 0.1  # camera pose term, relative to the depth term
    geo_use_camera_loss: bool = True     # False: depth term only

    # Per-task multiplier on geo_loss_weight, e.g. "3d23d:2.0,v2v:1.2" (default 1; 0 = off).
    geo_task_weight_scale: str = ""

    # ---- depth term ----
    # SmoothL1 beta and multiplier of the depth term (depth / scene scale).
    geo_depth_scale_beta: float = 0.02
    geo_depth_scale_weight_scale: float = 1.0

    # ---- camera term ----
    # camera = w * rot + (1 - w) * trans.
    geo_camera_rot_weight: float = 0.9
    # SmoothL1 beta and multiplier of the translation term.
    geo_camera_trans_beta: float = 1e-3
    geo_camera_trans_weight_scale: float = 1.0
    # Anchor term (positions relative to frame 0); 0 = off.
    geo_camera_trans_anchor_weight: float = 0.0
    geo_camera_trans_anchor_beta: float = 1e-3

    # Trained modality-token embeddings (directory or file), written into the frozen
    # text encoder. Needed when the tokens are not trained in this run.
    geo_modality_token_path: str = ""

    # ---- schedule ----
    geo_start_step: int = 0
    geo_interval: int = 1

    # ---- rollout used to estimate the predicted clean latent ----
    # Steps drawn uniformly in [low, high]; `grad_steps` random ones keep gradients.
    geo_rollout_steps_low: int = 40
    geo_rollout_steps_high: int = 50
    geo_rollout_grad_steps: int = 2

    # ---- tasks with a geometry loss ----
    # Single-view tasks (i2i / ti_i2i) have no camera term.
    geo_task_filter: str = "i2i,ti_i2i,v2v,tiv2v,3d23d"


GEO_WEIGHT_SCALE_TASKS = ("i2i", "ti_i2i", "v2v", "tiv2v", "3d23d")


def _parse_task_weight_scale(spec: str) -> Dict[str, float]:
    """'3d23d:1.5,v2v:0.27' -> {'3d23d': 1.5, 'v2v': 0.27}; '' -> {} (all 1.0)."""
    out: Dict[str, float] = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"--geo_task_weight_scale: '{item}' must be task:scale")
        task, val = (x.strip() for x in item.split(":", 1))
        if task not in GEO_WEIGHT_SCALE_TASKS:
            raise ValueError(f"--geo_task_weight_scale: unknown task '{task}', expected one of {GEO_WEIGHT_SCALE_TASKS}")
        scale = float(val)
        if not math.isfinite(scale) or scale < 0:
            raise ValueError(f"--geo_task_weight_scale: scale for {task} must be a finite value >= 0, got {val}")
        if task in out:
            raise ValueError(f"--geo_task_weight_scale: task '{task}' given twice")
        out[task] = scale
    return out


def add_geometry_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add the geometry-loss options to `parser` (as an argument group)."""
    parser = parser.add_argument_group("geometry consistency")
    parser.add_argument("--geo_enable", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--decoder_checkpoint_path", default=None,
                        help="Stitched 3D decoder checkpoint (required)")
    parser.add_argument("--anysplat_path", default=None,
                        help="AnySplat weights, local directory or HF repo id (required)")
    parser.add_argument("--geo_extractor_grad_checkpointing", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--geo_extractor_resolution", type=int, default=448)
    parser.add_argument("--geo_extractor_max_frames", type=int, default=17)
    parser.add_argument("--geo_frame_stride", type=int, default=4)
    parser.add_argument("--geo_loss_weight", type=float, default=0.05)
    parser.add_argument("--geo_camera_loss_weight", type=float, default=0.1)
    parser.add_argument("--geo_use_camera_loss", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--geo_task_weight_scale", default="")
    parser.add_argument("--geo_camera_trans_beta", type=float, default=1e-3)
    parser.add_argument("--geo_camera_trans_weight_scale", type=float, default=1.0)
    parser.add_argument("--geo_camera_trans_anchor_weight", type=float, default=0.0)
    parser.add_argument("--geo_camera_trans_anchor_beta", type=float, default=1e-3)
    parser.add_argument("--geo_depth_scale_beta", type=float, default=0.02)
    parser.add_argument("--geo_depth_scale_weight_scale", type=float, default=1.0)
    parser.add_argument("--geo_camera_rot_weight", type=float, default=0.9)
    parser.add_argument("--geo_modality_token_path", default="")
    parser.add_argument("--geo_start_step", type=int, default=0)
    parser.add_argument("--geo_interval", type=int, default=1)
    parser.add_argument("--geo_rollout_steps_low", type=int, default=40)
    parser.add_argument("--geo_rollout_steps_high", type=int, default=50)
    parser.add_argument("--geo_rollout_grad_steps", type=int, default=2)
    parser.add_argument("--geo_task_filter", default="i2i,ti_i2i,v2v,tiv2v,3d23d")


def geometry_args_from_namespace(ns: argparse.Namespace) -> GeometryArgs:
    """Build and validate GeometryArgs from a parsed namespace."""
    args = GeometryArgs(
        **{f.name: getattr(ns, f.name) for f in fields(GeometryArgs)}
    )

    if args.geo_extractor_max_frames < 0:
        raise ValueError("--geo_extractor_max_frames must be >= 0 (0 disables subsampling)")
    _parse_task_weight_scale(args.geo_task_weight_scale)   # validates the spec
    if not (math.isfinite(args.geo_camera_trans_beta) and args.geo_camera_trans_beta > 0):
        raise ValueError("--geo_camera_trans_beta must be a finite value > 0")
    if not (math.isfinite(args.geo_camera_trans_weight_scale) and args.geo_camera_trans_weight_scale >= 0):
        raise ValueError("--geo_camera_trans_weight_scale must be a finite value >= 0")
    if not (math.isfinite(args.geo_camera_trans_anchor_weight) and args.geo_camera_trans_anchor_weight >= 0):
        raise ValueError("--geo_camera_trans_anchor_weight must be a finite value >= 0")
    if not (math.isfinite(args.geo_camera_trans_anchor_beta) and args.geo_camera_trans_anchor_beta > 0):
        raise ValueError("--geo_camera_trans_anchor_beta must be a finite value > 0")
    if not (math.isfinite(args.geo_depth_scale_beta) and args.geo_depth_scale_beta > 0):
        raise ValueError("--geo_depth_scale_beta must be a finite value > 0")
    if not (math.isfinite(args.geo_depth_scale_weight_scale) and args.geo_depth_scale_weight_scale >= 0):
        raise ValueError("--geo_depth_scale_weight_scale must be a finite value >= 0")
    if args.geo_rollout_steps_low <= 0 or args.geo_rollout_steps_high <= 0:
        raise ValueError("--geo_rollout_steps_low/high must be positive")
    if args.geo_rollout_steps_low > args.geo_rollout_steps_high:
        raise ValueError("--geo_rollout_steps_low cannot be greater than --geo_rollout_steps_high")
    if args.geo_rollout_grad_steps <= 0:
        raise ValueError("--geo_rollout_grad_steps must be positive")
    if args.geo_frame_stride < 1:
        raise ValueError("--geo_frame_stride must be >= 1")
    if args.geo_enable:
        if not args.decoder_checkpoint_path:
            raise ValueError("geometry post-training requires --decoder_checkpoint_path")
        if not args.anysplat_path:
            raise ValueError("geometry post-training requires --anysplat_path")

    return args
