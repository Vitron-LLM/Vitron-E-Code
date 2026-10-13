"""
Geometry-consistency post-training for UniVEdit: the base trainer plus a depth and
camera-pose loss between the model's edit and the ground-truth edit.

  total_loss    = diffusion_loss + geo_loss_weight * task_scale * geometry_loss
  geometry_loss = depth + geo_camera_loss_weight * camera

The 3D decoder needs VIST3A, see "Installation" in the README.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from univedit.decoder3d import load_decoder3d
from univedit.tokens import SPECIAL_TOKENS
from univedit.training import trainer as base_trainer
from univedit.training.config import (
    GeometryArgs,
    TrainingConfig,
    _parse_task_weight_scale,
    add_geometry_cli_args,
    geometry_args_from_namespace,
)


# ============================================================================
# Helpers
# ============================================================================

def _choose_grad_steps(num_steps: int, num_grad_steps: int, device: torch.device) -> torch.Tensor:
    """Choose `num_grad_steps` distinct indices in [0, num_steps) on rank 0 and broadcast them."""
    idx = torch.empty(num_grad_steps, dtype=torch.long, device=device)
    is_rank0 = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    if is_rank0:
        idx.copy_(torch.randperm(num_steps, device=device)[:num_grad_steps])
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(idx, src=0)
    return idx


def _choose_rollout_step_count(low: int, high: int, device: torch.device) -> int:
    """Pick a step count uniformly in [low, high] on rank 0 and broadcast it."""
    value = torch.empty((), dtype=torch.long, device=device)
    is_rank0 = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    if is_rank0:
        value.copy_(torch.randint(low, high + 1, (), device=device, dtype=torch.long))
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(value, src=0)
    return int(value.item())


def _as_bvhw(depth: torch.Tensor) -> torch.Tensor:
    """Bring a depth-like tensor of any supported layout to (B, V, H, W)."""
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    if depth.ndim == 5 and depth.shape[2] == 1:
        depth = depth.squeeze(2)
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if depth.ndim != 4:
        raise ValueError(f"Expected depth-like tensor [B,V,H,W], got {tuple(depth.shape)}")
    return depth


def _resize_bvhw(x: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    """Spatially resize a (B, V, H, W) tensor."""
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    b, v, h, w = x.shape
    # Fold B and V into the batch dimension.
    y = x.reshape(b * v, 1, h, w).float()
    if mode == "nearest":
        y = F.interpolate(y, size=size, mode=mode)
    else:
        y = F.interpolate(y, size=size, mode=mode, align_corners=False)
    return y.reshape(b, v, size[0], size[1]).to(dtype=x.dtype)


# ============================================================================
# Geometry extraction (depth + camera poses)
# ============================================================================

@dataclass
class GeometryExtraction:
    """
    Output of one geometry extraction.

    depth     : (B, T, H, W), up to scale.
    extrinsic : (B, T, 4, 4) camera-to-world, frame 0 as reference.
    """

    depth: torch.Tensor
    extrinsic: torch.Tensor


class Decoder3DGeometryExtractor(nn.Module):
    """
    Geometry extractor: the 3D decoder (AnySplat + LoRA, stitched in VAE
    latent space).
    """

    def __init__(self, checkpoint_path: str, anysplat_path: str, device: torch.device):
        super().__init__()
        self.decoder = load_decoder3d(
            checkpoint_path=checkpoint_path,
            anysplat_path=anysplat_path,
            device=device,
        )

    @property
    def grad_checkpointing(self) -> bool:
        return self.decoder.stitched_3d_model.grad_checkpointing

    @grad_checkpointing.setter
    def grad_checkpointing(self, value: bool) -> None:
        self.decoder.stitched_3d_model.grad_checkpointing = value

    def extract(self, latent: torch.Tensor, feedforward_image: torch.Tensor) -> GeometryExtraction:
        """
        Depth and camera poses for (T_vae-1)*4+1 frames, from the unscaled VAE latent
        (B, C, T_vae, H, W) and the decoded frames (B, 3, T, res, res).
        """
        decoder_device = next(self.decoder.parameters()).device
        latent = latent.to(decoder_device, dtype=torch.float32)
        feedforward_image = feedforward_image.to(decoder_device, dtype=torch.bfloat16)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=latent.is_cuda):
            output = self.decoder.forward_with_latent(latent, feedforward_image=feedforward_image, train=False)

        return GeometryExtraction(
            depth=_as_bvhw(output.depth_dict["depth"]),
            extrinsic=output.pred_context_pose["extrinsic"],
        )


# ============================================================================
# Camera pose loss
# ============================================================================

def _align_extrinsic_to_frame0(ext: torch.Tensor) -> torch.Tensor:
    """Re-express a c2w trajectory relative to its own frame 0: T_k' = T_0^{-1} T_k."""
    squeeze = ext.ndim == 3
    E = (ext.unsqueeze(0) if squeeze else ext).float()
    out = torch.linalg.inv(E[:, :1].detach()) @ E
    return out[0] if squeeze else out


def _translation_term(
    t_pred: torch.Tensor, t_gt: torch.Tensor, depth_pred: float, depth_gt: float,
    beta: float, eps: float,
    anchor_weight: float = 0.0, anchor_beta: float = 1e-3,
) -> torch.Tensor:
    """
    Translation loss: SmoothL1 on Δt_pred / depth_pred − Δt_gt / depth_gt, where Δt
    are the frame-to-frame steps. With anchor_weight > 0, adds the same loss on the
    positions relative to frame 0. Returns 0 if a depth is non-finite or <= eps.
    """
    dp, dg = float(depth_pred), float(depth_gt)
    if not (math.isfinite(dp) and math.isfinite(dg) and dp > eps and dg > eps):
        return t_pred.sum() * 0.0
    step_pred = t_pred[1:] - t_pred[:-1]
    step_gt = (t_gt[1:] - t_gt[:-1]).detach()
    r = step_pred / dp - step_gt / dg
    loss = F.smooth_l1_loss(r, torch.zeros_like(r), reduction="mean", beta=beta)
    if anchor_weight > 0:
        a = (t_pred[1:] - t_pred[:1].detach()) / dp - (t_gt[1:] - t_gt[:1]).detach() / dg
        loss = loss + anchor_weight * F.smooth_l1_loss(a, torch.zeros_like(a), reduction="mean", beta=anchor_beta)
    return loss


def _camera_pose_loss(
    pred_extrinsic: torch.Tensor,
    gt_extrinsic: torch.Tensor,
    eps: float = 1e-6,
    rot_weight: float = 0.9,
    depth_pred: float = float("nan"),
    depth_gt: float = float("nan"),
    trans_beta: float = 1e-3,
    trans_weight_scale: float = 1.0,
    anchor_weight: float = 0.0,
    anchor_beta: float = 1e-3,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Camera pose loss of one trajectory of N >= 2 frames (c2w, frame 0 first):
    rot_weight * chordal rotation distance + (1 - rot_weight) * translation term.
    Returns the loss and the two terms as numbers.
    """
    pred_extrinsic = pred_extrinsic.reshape(-1, 4, 4)
    gt_extrinsic = gt_extrinsic.reshape(-1, 4, 4)

    # ---- rotation (frames 1..N-1) ----
    r_pred = pred_extrinsic[1:, :3, :3]
    r_gt = gt_extrinsic[1:, :3, :3]
    rot_loss = (((r_pred - r_gt).pow(2).sum(dim=(-2, -1)) + 1e-12).sqrt() / math.sqrt(2.0)).mean()

    # ---- translation ----
    trans_loss = trans_weight_scale * _translation_term(
        pred_extrinsic[:, :3, 3], gt_extrinsic[:, :3, 3], depth_pred, depth_gt, trans_beta, eps,
        anchor_weight=anchor_weight, anchor_beta=anchor_beta,
    )

    loss = rot_weight * rot_loss + (1.0 - rot_weight) * trans_loss
    return loss, {"rot_loss": float(rot_loss.detach()), "trans_loss": float(trans_loss.detach())}


# ============================================================================
# Geometry losses
# ============================================================================

def _pair_geometry_losses(
    pred_geo: GeometryExtraction,
    gt_geo: GeometryExtraction,
    args: GeometryArgs,
) -> Dict[str, Any]:
    """Depth and camera losses of one sample (predicted vs GT extraction)."""
    pred_depth = _as_bvhw(pred_geo.depth)
    gt_depth = _as_bvhw(gt_geo.depth).detach()
    pred_extrinsic = pred_geo.extrinsic
    gt_extrinsic = gt_geo.extrinsic.detach()

    if tuple(pred_depth.shape[-2:]) != tuple(gt_depth.shape[-2:]):
        gt_depth = _resize_bvhw(gt_depth, pred_depth.shape[-2:], mode="bilinear")

    # Frames 0, stride, 2*stride, ..., at most geo_extractor_max_frames of them.
    t = pred_depth.shape[1]
    frame_index = None
    idx = torch.arange(0, t, max(args.geo_frame_stride, 1), device=pred_depth.device)
    if args.geo_extractor_max_frames > 0 and idx.numel() > args.geo_extractor_max_frames:
        sel = torch.linspace(0, idx.numel() - 1, args.geo_extractor_max_frames, device=idx.device).round().long()
        idx = idx.index_select(0, sel.unique())
    if idx.numel() < t:
        frame_index = idx

    if frame_index is not None:
        pred_depth = pred_depth.index_select(1, frame_index)
        gt_depth = gt_depth.index_select(1, frame_index)
        pred_extrinsic = pred_extrinsic.index_select(1, frame_index)
        gt_extrinsic = gt_extrinsic.index_select(1, frame_index)

    # Valid pixels: finite, positive depth on both sides. The others are set to 1
    # and masked out.
    valid = torch.isfinite(pred_depth) & torch.isfinite(gt_depth) & (pred_depth > 0) & (gt_depth > 0)
    has_valid = bool(valid.any())
    pred_depth = torch.where(valid, pred_depth.float(), 1.0)
    gt_depth = torch.where(valid, gt_depth.float(), 1.0)

    # Depth loss: SmoothL1 on depth divided by the scene scale (mean valid depth).
    scale_pred = pred_depth[valid].mean() if has_valid else pred_depth.new_ones(())
    scale_gt = gt_depth[valid].mean() if has_valid else gt_depth.new_ones(())
    pred_norm = pred_depth / scale_pred
    gt_norm = gt_depth / scale_gt
    valid_f = valid.to(dtype=pred_norm.dtype)
    depth_loss = F.smooth_l1_loss(pred_norm, gt_norm, reduction="none", beta=args.geo_depth_scale_beta) * valid_f
    depth_loss = depth_loss.sum() / valid_f.sum().clamp_min(1.0)
    depth_loss = args.geo_depth_scale_weight_scale * depth_loss

    # Camera loss (2 frames or more).
    camera_loss = None
    camera_stats: Dict[str, float] = {}
    if args.geo_use_camera_loss and pred_depth.shape[1] >= 2:
        pred_extrinsic = _align_extrinsic_to_frame0(pred_extrinsic)
        gt_extrinsic = _align_extrinsic_to_frame0(gt_extrinsic)
        # The scene scales of the depth term divide the translations.
        camera_loss, camera_stats = _camera_pose_loss(
            pred_extrinsic, gt_extrinsic,
            rot_weight=args.geo_camera_rot_weight,
            depth_pred=float(scale_pred.detach()) if has_valid else float("nan"),
            depth_gt=float(scale_gt) if has_valid else float("nan"),
            trans_beta=args.geo_camera_trans_beta,
            trans_weight_scale=args.geo_camera_trans_weight_scale,
            anchor_weight=args.geo_camera_trans_anchor_weight,
            anchor_beta=args.geo_camera_trans_anchor_beta,
        )

    return {"depth_loss": depth_loss, "camera_loss": camera_loss, "camera_stats": camera_stats}


def _aggregate_pair_losses(
    stats_list: list[Dict[str, Any]],
    args: GeometryArgs,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Batch loss and its terms from the per-sample results."""
    depth_loss = torch.stack([s["depth_loss"] for s in stats_list]).mean()
    cam_stats = [s["camera_stats"] for s in stats_list if s["camera_loss"] is not None]
    camera_terms = [s["camera_loss"] for s in stats_list if s["camera_loss"] is not None]
    camera_loss = torch.stack(camera_terms).mean() if camera_terms else torch.zeros_like(depth_loss)

    total = depth_loss + args.geo_camera_loss_weight * camera_loss

    metrics = {
        "geo_depth_loss": float(depth_loss.detach().cpu()),
        "geo_total_loss": float(total.detach().cpu()),
    }
    # Camera terms are logged only when a sample has one.
    if camera_terms:
        metrics["geo_camera_loss"] = float(camera_loss.detach().cpu())
        metrics["geo_camera_rot_loss"] = sum(c["rot_loss"] for c in cam_stats) / len(cam_stats)
        metrics["geo_camera_trans_loss"] = sum(c["trans_loss"] for c in cam_stats) / len(cam_stats)

    return total, metrics


def extract_geometry_list(
    extractor: Decoder3DGeometryExtractor,
    latents: torch.Tensor,        # (B, C, T, H, W) unscaled VAE latents
    feedforward: torch.Tensor,    # (B, 3, T, res, res) decoded frames
) -> list[GeometryExtraction]:
    """Run the extractor on each sample of the batch."""
    return [extractor.extract(latents[i : i + 1], feedforward[i : i + 1]) for i in range(latents.shape[0])]


def pair_losses_from_geo_lists(
    pred_geos: list[GeometryExtraction],
    gt_geos: list[GeometryExtraction],
    args: GeometryArgs,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Per-sample losses of predicted / GT extractions, aggregated."""
    stats_list = [_pair_geometry_losses(p, g, args) for p, g in zip(pred_geos, gt_geos)]
    return _aggregate_pair_losses(stats_list, args)


# ============================================================================
# Trainer
# ============================================================================

class GeometryTrainer(base_trainer.HunyuanVideoTrainer):
    """
    HunyuanVideoTrainer with the geometry loss, added through the base train_step
    hooks (_begin_auxiliary_loss / _auxiliary_loss).
    """

    # ---- CLI hooks of base_trainer.main ----
    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        super().add_cli_args(parser)
        add_geometry_cli_args(parser)

    @classmethod
    def trainer_kwargs_from_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        kwargs = super().trainer_kwargs_from_args(args)
        kwargs["geometry_args"] = geometry_args_from_namespace(args)
        return kwargs

    def __init__(
        self,
        config: TrainingConfig,
        geometry_args: Optional[GeometryArgs] = None,
    ):
        self.geometry_loss_args = geometry_args if geometry_args is not None else GeometryArgs()
        super().__init__(config)
        self._geo_task_weight_scale = _parse_task_weight_scale(self.geometry_loss_args.geo_task_weight_scale)
        self._load_modality_tokens_from(self.geometry_loss_args.geo_modality_token_path)
        self.geometry_task_filter = {
            item.strip()
            for item in self.geometry_loss_args.geo_task_filter.split(",")
            if item.strip()
        }

        self.geometry_extractor = None
        if self.geometry_loss_args.geo_enable:
            self._build_geometry_extractor()

    def _build_geometry_extractor(self) -> None:
        """Load the geometry extractor and freeze it."""
        args = self.geometry_loss_args
        if self.is_main_process:
            logger.info("Loading the frozen 3D decoder as geometry extractor")
        extractor = Decoder3DGeometryExtractor(args.decoder_checkpoint_path, args.anysplat_path, self.device)
        extractor.eval()
        extractor.requires_grad_(False)
        if args.geo_extractor_grad_checkpointing:
            extractor.grad_checkpointing = True
        self.geometry_extractor = extractor
        if self.is_main_process:
            logger.info(
                "Geometry loss enabled: weight={}, task_filter={}",
                args.geo_loss_weight,
                sorted(self.geometry_task_filter),
            )

    def _should_apply_geometry_loss(self, task_type: str) -> bool:
        """Whether the current step computes the geometry loss for this task."""
        args = self.geometry_loss_args
        if self.geometry_extractor is None:   # --geo_enable false
            return False
        if args.geo_loss_weight * self._geo_task_weight_scale.get(task_type, 1.0) <= 0:
            return False
        if self.global_step < args.geo_start_step:
            return False
        if args.geo_interval > 1 and self.global_step % args.geo_interval != 0:
            return False
        if self.geometry_task_filter and task_type not in self.geometry_task_filter:
            return False
        return True

    def _unscale_vae_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Undo the VAE latent scale / shift."""
        scaling = getattr(self.vae.config, "scaling_factor", 1.0)
        shift = getattr(self.vae.config, "shift_factor", None)
        latents = latents / scaling
        if shift:
            latents = latents + shift
        return latents

    @torch.no_grad()
    def _decode_feedforward_frames(self, unscaled_latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to frames at the extractor resolution, without gradient."""
        resolution = self.geometry_loss_args.geo_extractor_resolution
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=unscaled_latents.is_cuda), \
                self._vae_memory_context():
            video = self.vae.decode(unscaled_latents.to(self.vae.dtype), return_dict=False)[0]
        return F.interpolate(
            video.float(),
            size=(video.shape[2], resolution, resolution),
            mode="trilinear",
            align_corners=True,
        )

    def _exact_clean_latents(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """GT clean latent recovered from the noised latent: latents_noised - t * target."""
        t = inputs["timesteps"].float() / float(self.config.num_train_timesteps)
        t = t.view(-1, *([1] * (inputs["latents_noised"].ndim - 1)))
        return inputs["latents_noised"].float() - t * inputs["target"].float()

    def _rollout_latents_for_geometry_loss(
        self,
        inputs: Dict[str, Any],
        model_dtype: torch.dtype,
        extra_kwargs: Dict[str, Any],
        deepstack_hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Denoise from pure noise and return the final latent. `geo_rollout_grad_steps`
        randomly chosen steps keep gradients.
        """
        args = self.geometry_loss_args
        num_steps = _choose_rollout_step_count(args.geo_rollout_steps_low, args.geo_rollout_steps_high, self.device)

        scheduler = self.pipeline._create_scheduler(self.config.sampling_timestep_shift)
        scheduler.set_timesteps(num_steps, device=self.device)
        scheduler.sigmas = scheduler.sigmas.to(self.device)
        timesteps = scheduler.timesteps

        grad_indices = _choose_grad_steps(
            len(timesteps),
            min(args.geo_rollout_grad_steps, len(timesteps)),
            self.device,
        )
        grad_index_set = set(int(i) for i in grad_indices.detach().cpu().tolist())

        latents = torch.randn(inputs["latents_noised"].shape, device=self.device, dtype=model_dtype)
        if self.sp_enabled:
            # Same start noise on every rank of the sequence-parallel group.
            latents = base_trainer.sync_tensor_for_sp(latents, self.sp_group)
        cond_latents = inputs["cond_latents"].to(device=self.device, dtype=model_dtype)
        text_emb = inputs["text_emb"].to(dtype=model_dtype)
        text_emb_2 = inputs["text_emb_2"].to(dtype=model_dtype) if inputs["text_emb_2"] is not None else None
        vision_states = inputs["vision_states"].to(dtype=model_dtype) if inputs["vision_states"] is not None else None

        for i, t in enumerate(timesteps):
            backprop = i in grad_index_set
            latents_concat = torch.cat([latents, cond_latents], dim=1)
            latent_model_input = scheduler.scale_model_input(latents_concat, t)
            latent_model_input = latent_model_input.detach().to(dtype=model_dtype)
            t_expand = t.repeat(latent_model_input.shape[0])

            if self.pipeline.use_meanflow:
                if i == len(timesteps) - 1:
                    timesteps_r = torch.tensor([0.0], device=self.device)
                else:
                    timesteps_r = timesteps[i + 1]
                timesteps_r = timesteps_r.repeat(latent_model_input.shape[0])
            else:
                timesteps_r = None

            context = torch.enable_grad() if backprop else torch.no_grad()
            with context:
                with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=(model_dtype == torch.bfloat16)):
                    model_out = self.transformer(
                        latent_model_input,
                        t_expand,
                        text_states=text_emb,
                        text_states_2=text_emb_2,
                        encoder_attention_mask=inputs["text_mask"],
                        vision_states=vision_states,
                        mask_type=inputs["task_type"],
                        extra_kwargs=extra_kwargs if extra_kwargs else None,
                        return_dict=False,
                        all_stack_text_states=deepstack_hidden_states,
                        timestep_r=timesteps_r,
                    )[0]

            latents = scheduler.step(model_out, t, latents, return_dict=False)[0]

        return latents

    def _geo_latent_for_task(self, latent: torch.Tensor, task_type: Optional[str]) -> torch.Tensor:
        """Keep only the first latent frame for image tasks."""
        if task_type in self.IMAGE_EDIT_TASKS:
            return latent[:, :, :1]
        return latent

    # ---- GT geometry from the VAE posterior mode ----
    def _encode_vae_tensor_once(self, pixel_values: torch.Tensor, sample_posterior: bool) -> torch.Tensor:
        """Base encode; while capturing, also stores the posterior mode of a sampled encode."""
        capture = getattr(self, "_geo_gt_mode_capture", None)
        if capture is None or not sample_posterior:
            return super()._encode_vae_tensor_once(pixel_values, sample_posterior=sample_posterior)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16), self._vae_memory_context():
            latent_dist = self.vae.encode(pixel_values).latent_dist
            latents = latent_dist.sample()
            mode = latent_dist.mode()
        capture.append(self._scale_vae_latents(mode))
        return self._scale_vae_latents(latents)

    def encode_vae(self, images: torch.Tensor, task_type: Optional[str] = None) -> torch.Tensor:
        """Base encode_vae; keeps the posterior mode of the target in self._geo_gt_mode_latents."""
        self._geo_gt_mode_capture = []
        try:
            latents = super().encode_vae(images, task_type=task_type)
            captured = self._geo_gt_mode_capture
        finally:
            self._geo_gt_mode_capture = None
        mode = torch.cat(captured, dim=0) if captured else None
        self._geo_gt_mode_latents = mode if (mode is not None and tuple(mode.shape) == tuple(latents.shape)) else None
        return latents

    def prepare_batch(self, batch: Dict[str, Any], loader_name: Optional[str] = None) -> Dict[str, Any]:
        """Base prepare_batch; adds the posterior mode of the target as inputs["geo_gt_mode_latents"]."""
        self._geo_gt_mode_latents = None
        try:
            inputs = super().prepare_batch(batch, loader_name=loader_name)
            mode = self._geo_gt_mode_latents
        finally:
            self._geo_gt_mode_latents = None
        # Not added with sequence parallelism or on a shape mismatch.
        if (mode is not None and not getattr(self, "sp_enabled", False) and isinstance(inputs, dict)
                and "latents_noised" in inputs and tuple(mode.shape) == tuple(inputs["latents_noised"].shape)):
            inputs["geo_gt_mode_latents"] = mode
        return inputs

    def _gt_clean_latents_for_geometry(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """GT clean latent: the posterior mode if available, else the exact clean latent."""
        mode = inputs.get("geo_gt_mode_latents")
        if mode is not None:
            return mode.float().detach()
        return self._exact_clean_latents(inputs).detach()

    def _start_gt_geometry_extraction(
        self, inputs: Dict[str, Any],
    ) -> tuple[list[GeometryExtraction], torch.cuda.Stream]:
        """
        Extract the GT geometry on a side CUDA stream. Returns the extractions and the
        stream, which the caller waits for before reading them.
        """
        caller_stream = torch.cuda.current_stream()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(caller_stream)

        with torch.cuda.stream(stream):
            with torch.no_grad():
                gt_latents = self._gt_clean_latents_for_geometry(inputs)
                gt_unscaled = self._unscale_vae_latents(gt_latents.float()).detach()
                # First frame only for image tasks.
                gt_unscaled = self._geo_latent_for_task(gt_unscaled, inputs.get("task_type"))
                gt_feedforward = self._decode_feedforward_frames(gt_unscaled)
                gt_geos = extract_geometry_list(
                    self.geometry_extractor, gt_unscaled.to(self.device), gt_feedforward.to(self.device),
                )

            # Hand the tensors allocated on the side stream over to the caller stream.
            for geo in gt_geos:
                geo.depth.record_stream(caller_stream)
                geo.extrinsic.record_stream(caller_stream)

        return gt_geos, stream

    def _geometry_loss_from_prediction(
        self,
        *,
        clean_latents_pred: torch.Tensor,
        gt_geos: list[GeometryExtraction],
        gt_stream: torch.cuda.Stream,
        task_type: Optional[str] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Geometry loss between the extraction of the predicted latent and the GT extractions."""
        # Wait for the GT side stream.
        torch.cuda.current_stream().wait_stream(gt_stream)

        pred_unscaled = self._unscale_vae_latents(clean_latents_pred.float())
        # First frame only for image tasks.
        pred_unscaled = self._geo_latent_for_task(pred_unscaled, task_type)

        pred_feedforward = self._decode_feedforward_frames(pred_unscaled)
        pred_geos = extract_geometry_list(
            self.geometry_extractor, pred_unscaled.to(self.device), pred_feedforward.to(self.device),
        )
        return pair_losses_from_geo_lists(pred_geos, gt_geos, self.geometry_loss_args)

    # ---- train_step hooks of the base trainer ----
    def _begin_auxiliary_loss(self, inputs: Dict[str, Any]) -> Any:
        """Start the GT extraction; returns None when the step has no geometry loss."""
        if not self._should_apply_geometry_loss(inputs["task_type"]):
            return None
        return self._start_gt_geometry_extraction(inputs)

    def _auxiliary_loss(
        self,
        inputs: Dict[str, Any],
        state: Any,
        diffusion_loss: torch.Tensor,
        model_dtype: torch.dtype,
        extra_kwargs: Dict[str, Any],
        deepstack_hidden_states: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """The weighted geometry loss (None when this step has none) and its metrics."""
        metrics: Dict[str, Any] = {"diffusion_loss": diffusion_loss.item()}
        if state is None:
            return None, metrics
        gt_geos, gt_stream = state

        clean_latents_pred = self._rollout_latents_for_geometry_loss(
            inputs, model_dtype, extra_kwargs, deepstack_hidden_states,
        )
        geometry_loss, geometry_metrics = self._geometry_loss_from_prediction(
            clean_latents_pred=clean_latents_pred, gt_geos=gt_geos, gt_stream=gt_stream,
            task_type=inputs["task_type"],
        )
        metrics.update(geometry_metrics)
        geo_w = self.geometry_loss_args.geo_loss_weight * self._geo_task_weight_scale.get(inputs["task_type"], 1.0)
        return geo_w * geometry_loss, metrics

    def _load_modality_tokens_from(self, token_path: str) -> None:
        """Write trained modality-token embeddings into the text encoder's embedding table."""
        if not token_path:
            return
        path = token_path
        if os.path.isdir(path):
            path = os.path.join(path, "modality_token_embeddings.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"modality token file not found: {path}")

        tokenizer = self.text_encoder.tokenizer
        token_ids = tokenizer.convert_tokens_to_ids(list(SPECIAL_TOKENS))
        missing = [t for t, i in zip(SPECIAL_TOKENS, token_ids)
                   if i is None or i == tokenizer.unk_token_id]
        if missing:
            raise ValueError(f"modality tokens not registered in the tokenizer: {missing}")

        embedding = self.text_encoder.model.get_input_embeddings()
        state = torch.load(path, map_location="cpu", weights_only=True)
        weights = state["weight"]
        if weights.shape[0] != len(token_ids):
            raise ValueError(
                f"modality token row count mismatch: file {weights.shape[0]} vs tokenizer {len(token_ids)}"
            )
        saved_ids = state.get("token_ids")
        if saved_ids is not None and list(saved_ids.tolist()) != list(token_ids):
            logger.warning(
                "modality token ids differ from the file; writing with the current tokenizer ids: "
                "file={} current={}", saved_ids.tolist(), token_ids,
            )
        # Align rows by token name.
        saved_tokens = state.get("tokens")
        if saved_tokens is not None:
            saved_tokens = [str(t) for t in saved_tokens]
            if set(saved_tokens) != set(SPECIAL_TOKENS):
                raise ValueError(
                    f"modality token set mismatch: file {saved_tokens} vs code {list(SPECIAL_TOKENS)}"
                )
            order = [saved_tokens.index(t) for t in SPECIAL_TOKENS]
            weights = weights[order]
        with torch.no_grad():
            for row, tid in zip(weights, token_ids):
                embedding.weight[tid].copy_(
                    row.to(device=embedding.weight.device, dtype=embedding.weight.dtype)
                )
        if self.is_main_process:
            logger.info("Wrote trained modality-token embeddings into the embedding table: {} (tokens={})",
                        path, list(SPECIAL_TOKENS))

    @classmethod
    def _aggregate_update_metrics(cls, metrics_list: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Average each geo_* term over the micro-steps that logged it."""
        aggregated = super()._aggregate_update_metrics(metrics_list)
        for key in {k for m in metrics_list for k in m if k.startswith("geo_")}:
            vals = [float(m[key]) for m in metrics_list if key in m]
            aggregated[key] = sum(vals) / len(vals)
        return aggregated

    def _log_metrics_to_wandb(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Also log the diffusion loss and the geo_* terms under train/."""
        super()._log_metrics_to_wandb(metrics, step=step)
        if self.wandb_run is None or not self.is_main_process:
            return
        geometry_log = {
            f"train/{key}": value
            for key, value in metrics.items()
            if key.startswith("geo_") and isinstance(value, (int, float))
        }
        if "diffusion_loss" in metrics:
            geometry_log["train/diffusion_loss"] = metrics["diffusion_loss"]
        if geometry_log:
            self.wandb_run.log(geometry_log, step=self.global_step if step is None else step)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point: run the base trainer with GeometryTrainer."""
    base_trainer.main(trainer_cls=GeometryTrainer, argv=argv)


if __name__ == "__main__":
    main()
