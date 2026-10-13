"""
3D decoder: VAE latents -> 3D Gaussians, per-view depth and cameras.

A Conv3d stitching layer feeds the upsampled VAE latent into an intermediate layer of
AnySplat (fine-tuned with LoRA). Needs VIST3A / AnySplat, which import_vist3a()
imports on first use; see "Installation" in the README.
"""

import os
import sys
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


__all__ = [
    "Decoder3D",
    "import_vist3a",
    "load_decoder3d",
]


# ============================================================================
# VIST3A / AnySplat import
# ============================================================================

_VIST3A: Optional[SimpleNamespace] = None


def import_vist3a() -> SimpleNamespace:
    """
    Import the VIST3A / AnySplat symbols used by the 3D decoder (cached). VIST3A is
    taken from $VIST3A_ROOT if set, else from sys.path.
    """
    global _VIST3A
    if _VIST3A is not None:
        return _VIST3A

    root = os.environ.get("VIST3A_ROOT")
    if root:
        root = os.path.abspath(os.path.expanduser(root))
        if root not in sys.path:
            sys.path.insert(0, root)

    try:
        from third_party_model.anysplat.src.model.model.anysplat import AnySplat
        from third_party_model.anysplat.src.misc.image_io import save_interpolated_video
        from third_party_model.anysplat.src.model.ply_export import export_ply
        from models.anysplat_stitched import AnySplatStitched
        from models.stitching_layer_builder import parse_conv_spec
        from utils.lora_util.utils import add_lora, parse_lora_mode
    except ImportError as exc:
        raise ImportError(
            "The 3D decoder needs VIST3A (with AnySplat) and its dependencies. "
            "Clone VIST3A at the pinned commit, apply third_party/vist3a_32253c3.patch and set "
            "VIST3A_ROOT to the checkout (or add it to PYTHONPATH). "
            f"See \"Installation\" in the README. Original error: {exc!r}"
        ) from exc

    _VIST3A = SimpleNamespace(
        AnySplat=AnySplat,
        save_interpolated_video=save_interpolated_video,
        export_ply=export_ply,
        AnySplatStitched=AnySplatStitched,
        add_lora=add_lora,
        parse_lora_mode=parse_lora_mode,
        parse_conv_spec=parse_conv_spec,
    )
    return _VIST3A


# ============================================================================
# 3D decoder
# ============================================================================

# Configuration of the released decoder checkpoint.
STITCHING_LAYER_LOCATION = "enc_blocks_2"
STITCHING_LAYER_CONFIG = "conv3d_k5x3x3_o1024_s1x2x2_p2x1x1"
LORA_CONFIG = "r64,a32,d0.0,f0"


class Decoder3D(nn.Module):
    """
    Stitched decoder for VAE latents: trilinear upsampling of the latent,
    Conv3d stitching layer, AnySplatStitched (AnySplat + LoRA).
    """

    LATENT_CHANNELS = 32    # VAE latent channels
    TEMPORAL_DOWN = 4       # VAE temporal compression
    TARGET_SPATIAL = 64     # latent height / width fed to the stitching layer

    def __init__(self, stitched_3d_model: nn.Module, stitching_layer: nn.Module, device: torch.device):
        super().__init__()
        self.device = device
        self.stitched_3d_model = stitched_3d_model
        self.stitching_layer = stitching_layer

    def forward_with_latent(self, latent: torch.Tensor, feedforward_image: torch.Tensor, train: bool = False):
        """
        AnySplat output (Gaussians, depth, camera poses) for (T_vae-1)*4+1 views.

        latent            : (B, 32, T_vae, H_vae, W_vae) unscaled VAE latents
        feedforward_image : (B, 3, T, 448, 448) frames in [-1, 1]
        """
        t_out = (latent.shape[2] - 1) * self.TEMPORAL_DOWN + 1
        latent = F.interpolate(
            latent,
            size=[t_out, self.TARGET_SPATIAL, self.TARGET_SPATIAL],
            mode="trilinear",
            align_corners=True,
        )
        return self.stitched_3d_model(self.stitching_layer(latent), feedforward_image, train=train)


def load_decoder3d(checkpoint_path: str, anysplat_path: str, device: torch.device) -> Decoder3D:
    """Build the 3D decoder from the AnySplat weights and the stitched decoder checkpoint."""
    if not checkpoint_path:
        raise ValueError("load_decoder3d: checkpoint_path is required")
    if not anysplat_path:
        raise ValueError("load_decoder3d: anysplat_path is required")

    v = import_vist3a()

    logger.info(f"Loading AnySplat from {anysplat_path}")
    feedforward_model = v.AnySplat.from_pretrained(anysplat_path)
    feedforward_model.to(device)

    logger.info(f"Building AnySplatStitched at location: {STITCHING_LAYER_LOCATION}")
    stitched_3d_model = v.AnySplatStitched(feedforward_model, STITCHING_LAYER_LOCATION)

    lora_cfg = v.parse_lora_mode(LORA_CONFIG)
    v.add_lora(
        stitched_3d_model,
        target_modules=lora_cfg.target_modules,
        r=lora_cfg.r,
        alpha=lora_cfg.alpha,
        dropout=lora_cfg.dropout,
        fan_in_fan_out=lora_cfg.fan_in_fan_out,
    )

    conv_spec = v.parse_conv_spec(STITCHING_LAYER_CONFIG)
    stitching_layer = conv_spec.build(in_channels=Decoder3D.LATENT_CHANNELS)

    decoder = Decoder3D(stitched_3d_model, stitching_layer, device)

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    decoder.stitched_3d_model.load_state_dict(state_dict["lora"], strict=False)
    decoder.stitching_layer.weight.data = state_dict["stitching_layer"]["weight"]
    decoder.stitching_layer.bias.data = state_dict["stitching_layer"]["bias"]

    patch_embed = decoder.stitched_3d_model.encoder.aggregator.patch_embed
    patch_embed.mask_token.data = state_dict["mask_token"]
    patch_embed.cls_token.data = state_dict["cls_token"]
    patch_embed.register_tokens.data = state_dict["register_tokens"]

    decoder = decoder.to(device)
    decoder.eval()
    logger.info("3D decoder loaded successfully")
    return decoder
