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


import psutil
import inspect
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import PIL

import loguru

import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from torch import distributed as dist

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput, deprecate, logging

from hyvideo.commons import (
    PIPELINE_CONFIGS,
    auto_offload_model,
    get_gpu_memory,
    get_rank,
    is_flash3_available,
    is_angelslim_available,
)
from hyvideo.commons.parallel_states import get_parallel_state

from hyvideo.models.autoencoders import hunyuanvideo_15_vae
from hyvideo.models.text_encoders import PROMPT_TEMPLATE, TextEncoder
from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2
from hyvideo.models.text_encoders.byT5.format_prompt import MultilingualPromptFormat
from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import (
    HunyuanVideo_1_5_DiffusionTransformer,
)
from hyvideo.models.vision_encoder import VisionEncoder

from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

from hyvideo.utils.data_utils import (
    generate_crop_size_list,
    get_closest_ratio,
    resize_and_center_crop,
)
from hyvideo.utils.multitask_utils import (
    get_semantic_images_np,
    get_semantic_images_np2
)
from hyvideo.commons.infer_state import InferState
from univedit.tokens import SUPPORTED_TASKS, get_special_token_from_task

from .pipeline_utils import retrieve_timesteps, rescale_noise_cfg


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

IMAGE_EDIT_TASKS = ("i2i", "ti_i2i")
VIDEO_EDIT_TASKS = ("v2v", "tiv2v", "3d23d")
REFERENCE_VIDEO_EDIT_TASKS = ("tiv2v",)
REFERENCE_TASKS = ("ti_i2i", "tiv2v")

@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class HunyuanVideo_1_5_Pipeline(DiffusionPipeline):

    model_cpu_offload_seq = "text_encoder->text_encoder_2->byt5_model->transformer->vae"
    _optional_components = ["text_encoder_2"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: TextEncoder,
        transformer: HunyuanVideo_1_5_DiffusionTransformer,
        scheduler: KarrasDiffusionSchedulers,
        text_encoder_2: Optional[TextEncoder] = None,
        flow_shift: float = 7.0,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
        embedded_guidance_scale: Optional[float] = None,
        progress_bar_config: Dict[str, Any] = None,
        vision_num_semantic_tokens=729,
        vision_states_dim=1152,
        glyph_byT5_v2=True,
        byt5_model=None,
        byt5_tokenizer=None,
        byt5_max_length=256,
        prompt_format=None,
        execution_device=None,
        vision_encoder=None,
        enable_offloading=False,
    ):
        super().__init__()

        self.register_to_config(
            glyph_byT5_v2=glyph_byT5_v2,
            byt5_max_length=byt5_max_length,
            vision_num_semantic_tokens=vision_num_semantic_tokens,
            vision_states_dim=vision_states_dim,
            flow_shift=flow_shift,
            guidance_scale=guidance_scale,
            embedded_guidance_scale=embedded_guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        if glyph_byT5_v2:
            self.byt5_max_length = byt5_max_length
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
                text_encoder_2=text_encoder_2,
                byt5_model=byt5_model,
                byt5_tokenizer=byt5_tokenizer,
            )
            self.prompt_format = prompt_format
        else:
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
                text_encoder_2=text_encoder_2,
            )
            self.byt5_model = None
            self.byt5_tokenizer = None


        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.text_len = text_encoder.max_length
        self.target_dtype = torch.bfloat16
        self.vae_dtype = torch.float16
        self.autocast_enabled = True
        self.vae_autocast_enabled = True
        self.enable_offloading = enable_offloading
        self.execution_device = torch.device(execution_device)

        if vision_encoder:
            self.register_modules(vision_encoder=vision_encoder)
        else:
            self.vision_encoder = None

        # Resolution bucket configurations
        self.target_size_config = {
            "360p": {"bucket_hw_base_size": 480, "bucket_hw_bucket_stride": 16},
            "480p": {"bucket_hw_base_size": 640, "bucket_hw_bucket_stride": 16},
            "720p": {"bucket_hw_base_size": 960, "bucket_hw_bucket_stride": 16},
            "1080p": {"bucket_hw_base_size": 1440, "bucket_hw_bucket_stride": 16},
        }

        self.noise_init_device = torch.device('cuda')


    @classmethod
    def _create_scheduler(cls, flow_shift):
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=True,
            solver="euler",
        )
        return scheduler

    @classmethod
    def _load_byt5(cls, cached_folder, glyph_byT5_v2, byt5_max_length, device):
        if not glyph_byT5_v2:
            byt5_kwargs = None
            prompt_format = None
            return byt5_kwargs, prompt_format
        try:
            load_from = os.path.join(cached_folder, "text_encoder")
            glyph_root = os.path.join(load_from, "Glyph-SDXL-v2")
            if not os.path.exists(glyph_root):
                raise RuntimeError(
                    f"Glyph checkpoint not found from '{glyph_root}'. \n"
                    "Please download from https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2/files.\n\n"
                    "- Required files:\n"
                    "    Glyph-SDXL-v2\n"
                    "    ├── assets\n"
                    "    │   ├── color_idx.json\n"
                    "    │   └── multilingual_10-lang_idx.json\n"
                    "    └── checkpoints\n"
                    "        └── byt5_model.pt\n"
                )

            byT5_google_path = os.path.join(load_from, "byt5-small")
            if not os.path.exists(byT5_google_path):
                loguru.logger.warning(f"ByT5 google path not found from: {byT5_google_path}. Try downloading from https://huggingface.co/google/byt5-small.")
                byT5_google_path = "google/byt5-small"


            multilingual_prompt_format_color_path = os.path.join(glyph_root, "assets/color_idx.json")
            multilingual_prompt_format_font_path = os.path.join(glyph_root, "assets/multilingual_10-lang_idx.json")

            byt5_args = dict(
                byT5_google_path=byT5_google_path,
                byT5_ckpt_path=os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
                multilingual_prompt_format_color_path=multilingual_prompt_format_color_path,
                multilingual_prompt_format_font_path=multilingual_prompt_format_font_path,
                byt5_max_length=byt5_max_length
            )

            byt5_kwargs = load_glyph_byT5_v2(byt5_args, device=device)
            prompt_format = MultilingualPromptFormat(
                font_path=multilingual_prompt_format_font_path,
                color_path=multilingual_prompt_format_color_path
            )
            return byt5_kwargs, prompt_format
        except Exception as e:
            print(e)
            raise RuntimeError("Error loading byT5 glyph processor") from e


    def encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "image",
        task_type=None,
        video_sample_frames=None,
        nframes=None,
        semantic_images: Optional[torch.Tensor] = None,
        all_condition_pils=None,
        deepstack=[8,16,24],
        setclip=True,
        modality_tokens=None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`torch.Tensor`, *optional*):
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for negative prompt embeddings.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            text_encoder (TextEncoder, *optional*):
                Text encoder to use. If None, uses the pipeline's default text encoder.
            data_type (`str`, *optional*):
                Type of data being encoded. Defaults to "image".
            task_type (`str`):
                One of SUPPORTED_TASKS; selects the Qwen2.5-VL prompt mode
                (i2i: 7, ti_i2i: 8, v2v: 5, tiv2v: 6, 3d23d: 9; 1 = text only).
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        if prompt is not None and isinstance(prompt, str):
            prompt = [prompt]
        if negative_prompt is not None and isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        def prepare_input(*args, **kwargs):
            if modality_tokens is not None:
                kwargs["modality_tokens"] = modality_tokens
            return text_encoder.prepare_input(*args, **kwargs)

        if prompt_embeds is None:

            if task_type in VIDEO_EDIT_TASKS:
                sampled_frames = video_sample_frames
                if sampled_frames is not None:
                    if semantic_images is None:
                        prompt_mode = 5 if task_type == "3d23d" else 3
                        text_inputs, skip_token_num = prepare_input(prompt, videos=sampled_frames, num_frames=nframes, prompt_mode=prompt_mode)
                        text_inputs = text_inputs.to(device)
                    else:
                        newimgs = []
                        imgs = semantic_images
                        for p in imgs:
                            p2 = p.copy()
                            p2.thumbnail((560, 560))
                            newimgs.append(p2)
                        text_inputs, skip_token_num = prepare_input(prompt, imgs=newimgs, videos=sampled_frames, num_frames=nframes, prompt_mode=4)
                        text_inputs = text_inputs.to(device)
                else:
                    raise ValueError(f"task_type={task_type!r} needs its visual inputs for the text encoder.")
                
                prompt_outputs = text_encoder.encode(
                    text_inputs, data_type=data_type, device=device, deepstack=deepstack, crop_start=skip_token_num, setclip=setclip
                )
                prompt_embeds = prompt_outputs.hidden_state
                deepstack_hidden_states = prompt_outputs.deepstack_hidden_states

            elif task_type in ('i2i', 'ti_i2i'):
                image_edit_pils = all_condition_pils
                if image_edit_pils is not None:
                    newimgs = []
                    for ps in image_edit_pils:
                        if isinstance(ps, PIL.Image.Image):
                            ps = [ps]
                        subimgs = []
                        for p in ps:
                            p2 = p.copy()
                            p2.thumbnail((560, 560))
                            subimgs.append(p2)
                        newimgs.append(subimgs)
                    prompt_mode = 1 if task_type == 'i2i' else 2
                    text_inputs, skip_token_num = prepare_input(prompt, newimgs, prompt_mode=prompt_mode)
                    text_inputs = text_inputs.to(device)
                else:
                    raise ValueError(f"task_type={task_type!r} needs its visual inputs for the text encoder.")

                prompt_outputs = text_encoder.encode(
                    text_inputs, data_type=data_type, device=device, deepstack=deepstack, crop_start=skip_token_num, setclip=setclip
                )
                prompt_embeds = prompt_outputs.hidden_state
                deepstack_hidden_states = prompt_outputs.deepstack_hidden_states

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(
                    bs_embed * num_videos_per_prompt, seq_len
                )

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            if task_type in VIDEO_EDIT_TASKS:
                sampled_frames = video_sample_frames
                if sampled_frames is not None:
                    if semantic_images is None:
                        prompt_mode = 5 if task_type == "3d23d" else 3
                        uncond_input, skip_token_num = prepare_input(uncond_tokens, videos=sampled_frames, num_frames=nframes, prompt_mode=prompt_mode)
                        uncond_input = uncond_input.to(device)
                    else:
                        newimgs = []
                        imgs = semantic_images
                        for p in imgs:
                            p2 = p.copy()
                            p2.thumbnail((560, 560))
                            newimgs.append(p2)
                        uncond_input, skip_token_num = prepare_input(uncond_tokens, imgs=newimgs, videos=sampled_frames, num_frames=nframes, prompt_mode=4)
                        uncond_input = uncond_input.to(device)
                else:
                    raise ValueError(f"task_type={task_type!r} needs its visual inputs for the text encoder.")

                negative_prompt_outputs = text_encoder.encode(uncond_input, data_type=data_type, is_uncond=True, deepstack=deepstack, crop_start=skip_token_num, setclip=setclip)
                negative_prompt_embeds = negative_prompt_outputs.hidden_state
                negative_deepstack_hidden_states = negative_prompt_outputs.deepstack_hidden_states
            
            elif task_type in ('i2i', 'ti_i2i'):
                image_edit_pils = all_condition_pils
                if image_edit_pils is not None:
                    newimgs = []
                    for ps in image_edit_pils:
                        if isinstance(ps, PIL.Image.Image):
                            ps = [ps]
                        subimgs = []
                        for p in ps:
                            p2 = p.copy()
                            p2.thumbnail((560, 560))
                            subimgs.append(p2)
                        newimgs.append(subimgs)
                    prompt_mode = 1 if task_type == 'i2i' else 2
                    uncond_input, skip_token_num = prepare_input(uncond_tokens, newimgs, prompt_mode=prompt_mode)
                    uncond_input = uncond_input.to(device)
                else:
                    raise ValueError(f"task_type={task_type!r} needs its visual inputs for the text encoder.")

                negative_prompt_outputs = text_encoder.encode(uncond_input, data_type=data_type, is_uncond=True, deepstack=deepstack, crop_start=skip_token_num, setclip=setclip)
                negative_prompt_embeds = negative_prompt_outputs.hidden_state
                negative_deepstack_hidden_states = negative_prompt_outputs.deepstack_hidden_states


            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(1, num_videos_per_prompt)
                negative_attention_mask = negative_attention_mask.view(batch_size * num_videos_per_prompt, seq_len)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, -1)
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        
        if deepstack_hidden_states is not None or negative_deepstack_hidden_states is not None:
            assert num_videos_per_prompt == 1

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
            deepstack_hidden_states,
            negative_deepstack_hidden_states
        )


    def prepare_extra_func_kwargs(self, func, kwargs):
        """
        Prepare extra keyword arguments for scheduler functions.

        Filters kwargs to only include parameters that the function accepts.
        This is useful since not all schedulers have the same signature.
        """
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs


    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        latent_height,
        latent_width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """
        Prepare latents for video generation.

        Args:
            batch_size: Batch size for generation.
            num_channels_latents: Number of channels in latent space.
            latent_height: Height of latent tensors.
            latent_width: Width of latent tensors.
            video_length: Number of frames in the video.
            dtype: Data type for latents.
            device: Target device for latents.
            generator: Random number generator.
            latents: Pre-computed latents. If None, random latents are generated.

        Returns:
            torch.Tensor: Prepared latents tensor.
        """
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            latent_height,
            latent_width,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = torch.randn(shape, generator=generator, device=self.noise_init_device, dtype=dtype).to(device)
        else:
            latents = latents.to(device)

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self,
        w: torch.Tensor,
        embedding_dim: int = 512,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @staticmethod
    def get_byt5_text_tokens(byt5_tokenizer, byt5_max_length, text_prompt):
        """
        Tokenize text prompt for byT5 model.

        Args:
            byt5_tokenizer: The byT5 tokenizer.
            byt5_max_length: Maximum sequence length for tokenization.
            text_prompt: Text prompt string to tokenize.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - input_ids: Tokenized input IDs.
                - attention_mask: Attention mask tensor.
        """
        byt5_text_inputs = byt5_tokenizer(
            text_prompt,
            padding="max_length",
            max_length=byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return byt5_text_inputs.input_ids, byt5_text_inputs.attention_mask

    def _extract_glyph_texts(self, prompt):
        """
        Extract glyph texts from prompt using regex pattern.

        Args:
            prompt: Input prompt string containing quoted text.

        Returns:
            List[str]: List of extracted glyph texts (deduplicated if multiple).
        """
        pattern = r'\"(.*?)\"|“(.*?)”'
        matches = re.findall(pattern, prompt)
        result = [match[0] or match[1] for match in matches]
        result = list(dict.fromkeys(result)) if len(result) > 1 else result
        return result

    def _process_single_byt5_prompt(self, prompt_text, device):
        """
        Process a single prompt for byT5 encoding.

        Args:
            prompt_text: The prompt text to process.
            device: Target device for tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - byt5_embeddings: Encoded embeddings tensor.
                - byt5_mask: Attention mask tensor.
        """
        byt5_embeddings = torch.zeros((1, self.byt5_max_length, 1472), device=device)
        byt5_mask = torch.zeros((1, self.byt5_max_length), device=device, dtype=torch.int64)
        
        glyph_texts = self._extract_glyph_texts(prompt_text)
        
        if len(glyph_texts) > 0:
            text_styles = [{'color': None, 'font-family': None} for _ in range(len(glyph_texts))]
            formatted_text = self.prompt_format.format_prompt(glyph_texts, text_styles)
            
            text_ids, text_mask = self.get_byt5_text_tokens(
                self.byt5_tokenizer, self.byt5_max_length, formatted_text
            )
            text_ids = text_ids.to(device=device)
            text_mask = text_mask.to(device=device)
            
            byt5_outputs = self.byt5_model(text_ids, attention_mask=text_mask.float())
            byt5_embeddings = byt5_outputs[0]
            byt5_mask = text_mask
            
        return byt5_embeddings, byt5_mask

    def _prepare_byt5_embeddings(self, prompts, device):
        """
        Prepare byT5 embeddings for both positive and negative prompts.

        Args:
            prompts: List of prompt strings or single prompt string.
            device: Target device for tensors.

        Returns:
            dict: Dictionary containing:
                - "byt5_text_states": Combined embeddings tensor.
                - "byt5_text_mask": Combined attention mask tensor.
                Returns empty dict if glyph_byT5_v2 is disabled.
        """
        if not self.config.glyph_byT5_v2:
            return {}
            
        if isinstance(prompts, str):
            prompt_list = [prompts]
        elif isinstance(prompts, list):
            prompt_list = prompts
        else:
            raise ValueError("prompts must be str or list of str")

        positive_embeddings = []
        positive_masks = []
        negative_embeddings = []
        negative_masks = []

        for prompt in prompt_list:
            pos_emb, pos_mask = self._process_single_byt5_prompt(prompt, device)
            positive_embeddings.append(pos_emb)
            positive_masks.append(pos_mask)

            if self.do_classifier_free_guidance:
                neg_emb, neg_mask = self._process_single_byt5_prompt("", device)
                negative_embeddings.append(neg_emb)
                negative_masks.append(neg_mask)

        byt5_positive = torch.cat(positive_embeddings, dim=0)
        byt5_positive_mask = torch.cat(positive_masks, dim=0)
        
        if self.do_classifier_free_guidance:
            byt5_negative = torch.cat(negative_embeddings, dim=0)
            byt5_negative_mask = torch.cat(negative_masks, dim=0)
            
            byt5_embeddings = torch.cat([byt5_negative, byt5_positive], dim=0)
            byt5_masks = torch.cat([byt5_negative_mask, byt5_positive_mask], dim=0)
        else:
            byt5_embeddings = byt5_positive
            byt5_masks = byt5_positive_mask

        return {
            "byt5_text_states": byt5_embeddings,
            "byt5_text_mask": byt5_masks
        }

    def extract_image_features(self, reference_image):
        """
        Extract features from a reference image using VisionEncoder.

        Args:
            reference_image: numpy array of shape (H, W, 3) with dtype uint8.

        Returns:
            VisionEncoderModelOutput: Encoded image features.
        """
        assert isinstance(reference_image, np.ndarray)
        assert reference_image.ndim == 3 and reference_image.shape[2] == 3
        assert reference_image.dtype == np.uint8

        image_encoder_output = self.vision_encoder.encode_images(reference_image)

        return image_encoder_output

    def _prepare_vision_states(self, reference_image, target_resolution, latents, device, reference_image2=None):
        """
        Prepare vision states for multitask training.

        Args:
            reference_image: Image(s) for SigLIP semantic conditioning (None -> zero states).
            target_resolution: Target resolution bucket (e.g. "480p").
            latents: Latent tensors.
            device: Target device.
            reference_image2: Optional second reference image for extra semantic conditioning.

        Returns:
            torch.Tensor or None: Vision states tensor or None if vision encoder is unavailable.
        """
        vision_states = torch.zeros(
            latents.shape[0], self.config.vision_num_semantic_tokens, self.config.vision_states_dim
        ).to(latents.device)
        if reference_image is not None:
            if isinstance(reference_image, list):
                if self.vision_encoder is None:
                    vision_states = None
                else:
                    semantic_images = reference_image
                    is_batched = any(isinstance(item, list) for item in semantic_images)
                    if not is_batched:
                        semantic_images = [semantic_images]

                    per_sample_states = []
                    for sample_images in semantic_images:
                        if isinstance(sample_images, np.ndarray) and len(sample_images.shape) == 4:
                            sample_images = list(sample_images)
                        elif not isinstance(sample_images, list):
                            sample_images = [sample_images]

                        image_states = []
                        for semantic_image in sample_images:
                            semantic_image = np.array(semantic_image) if isinstance(semantic_image, Image.Image) else semantic_image
                            if len(semantic_image.shape) == 4:
                                semantic_image = semantic_image[0]
                            height, width = self.get_closest_resolution_given_reference_image(semantic_image, target_resolution)
                            input_image_np = resize_and_center_crop(
                                semantic_image, target_width=width, target_height=height
                            )
                            image_encoder_output = self.vision_encoder.encode_images(input_image_np)
                            image_vision_states = image_encoder_output.last_hidden_state.to(
                                device=device, dtype=self.target_dtype
                            )
                            if image_vision_states.ndim == 3 and image_vision_states.shape[0] == 1:
                                image_vision_states = image_vision_states[0]
                            image_states.append(
                                image_vision_states
                            )

                        sample_states = torch.stack(image_states, dim=0).mean(dim=0)
                        per_sample_states.append(sample_states)

                    vision_states = torch.stack(per_sample_states, dim=0)
                    if vision_states.shape[0] == 1 and latents.shape[0] != 1:
                        vision_states = vision_states.repeat(latents.shape[0], 1, 1)
            else:
                reference_image = np.array(reference_image) if isinstance(reference_image, Image.Image) else reference_image
                if len(reference_image.shape) == 4:
                    reference_image = reference_image[0]

                height, width = self.get_closest_resolution_given_reference_image(reference_image, target_resolution)

                # Encode reference image to vision states
                if self.vision_encoder is not None:
                    input_image_np = resize_and_center_crop(reference_image, target_width=width, target_height=height)
                    vision_states = self.vision_encoder.encode_images(input_image_np)
                    vision_states = vision_states.last_hidden_state.to(device=device, dtype=self.target_dtype)
                else:
                    vision_states = None

        if reference_image2 is not None:
            reference_image2 = np.array(reference_image2) if isinstance(reference_image2, Image.Image) else reference_image2
            if len(reference_image2.shape) == 4:
                reference_image2 = reference_image2[0]

            height2, width2 = self.get_closest_resolution_given_reference_image(reference_image2, target_resolution)

            if self.vision_encoder is not None:
                input_image_np2 = resize_and_center_crop(reference_image2, target_width=width2, target_height=height2)
                vision_states2 = self.vision_encoder.encode_images(input_image_np2)
                vision_states2 = vision_states2.last_hidden_state.to(device=device, dtype=self.target_dtype)
            else:
                vision_states2 = None

            if vision_states2 is not None and vision_states is not None:
                vision_states = (vision_states + vision_states2) / 2.0
        
        # Repeat image features for batch size if needed (for classifier-free guidance)
        if self.do_classifier_free_guidance and vision_states is not None:
            vision_states = vision_states.repeat(2, 1, 1)
        
        return vision_states

    def _prepare_cond_latents(self, task_type, cond_latents, latents):
        """
        Concatenate the condition latents with their mask channel.

        Every latent frame is conditioned, so the mask is all ones. Tasks with a
        reference image (ti_i2i, tiv2v) add the reference latents on latent
        frame 1 before this call and mark it with an extra 1 in the mask:

            i2i / v2v / 3d23d : mask = 1 on every frame
            ti_i2i / tiv2v    : mask = 1 on every frame, 2 on frame 1

        Args:
            task_type: Task type (one of SUPPORTED_TASKS).
            cond_latents: Condition latents, [B, C, F, H, W].
            latents: Noise latents, [B, C, F, H, W].

        Returns:
            torch.Tensor: [B, C + 1, F, H, W], condition latents plus the mask channel.
        """
        if task_type not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task_type={task_type!r}")
        if latents.shape[2] < 2:
            raise ValueError(f"{task_type} needs at least 2 latent frames, got {latents.shape[2]}")

        if task_type in IMAGE_EDIT_TASKS:
            if cond_latents is None or cond_latents.shape[2] != latents.shape[2]:
                raise ValueError(
                    f"{task_type} expects image-as-video condition latents covering all "
                    f"{latents.shape[2]} latent frames"
                )
            latents_concat = torch.zeros(
                latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3], latents.shape[4]
            ).to(device=latents.device, dtype=latents.dtype)
            latents_concat[:] = cond_latents.to(device=latents.device, dtype=latents.dtype)
            mask_dtype = latents.dtype
        else:
            latents_concat = cond_latents
            # The video-task mask is float32, so the concat below promotes the
            # condition to float32.
            mask_dtype = torch.float32

        mask_concat = torch.ones(
            latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4],
            device=latents.device, dtype=mask_dtype,
        )
        if task_type in ("ti_i2i",) + REFERENCE_VIDEO_EDIT_TASKS:
            mask_concat[:, :, 1] += 1.0

        return torch.concat([latents_concat, mask_concat], dim=1)

    def get_closest_resolution_given_reference_image(self, reference_image, target_resolution):
        """
        Get closest supported resolution for a reference image.

        Args:
            reference_image: PIL Image or numpy array.
            target_resolution: Target resolution string (e.g., "720p", "1080p").

        Returns:
            tuple[int, int]: (height, width) of closest supported resolution.
        """
        assert reference_image is not None

        if isinstance(reference_image, Image.Image):
            origin_size = reference_image.size
        elif isinstance(reference_image, np.ndarray):
            H, W, C = reference_image.shape
            origin_size = (W, H)
        else:
            raise ValueError(f"Unsupported reference_image type: {type(reference_image)}. Must be PIL Image or numpy array")

        return self.get_closest_resolution_given_original_size(origin_size, target_resolution)


    def get_closest_resolution_given_original_size(self, origin_size, target_size):
        """
        Get closest supported resolution for given original size and target resolution.

        Args:
            origin_size: Tuple of (width, height) of original image.
            target_size: Target resolution string (e.g., "720p", "1080p").

        Returns:
            tuple[int, int]: (height, width) of closest supported resolution.
        """
        bucket_hw_base_size = self.target_size_config[target_size]["bucket_hw_base_size"]
        bucket_hw_bucket_stride = self.target_size_config[target_size]["bucket_hw_bucket_stride"]

        assert bucket_hw_base_size in [128, 256, 480, 512, 640, 720, 960, 1440], \
            f"bucket_hw_base_size must be in [128, 256, 480, 512, 640, 720, 960, 1440], but got {bucket_hw_base_size}"

        crop_size_list = generate_crop_size_list(bucket_hw_base_size, bucket_hw_bucket_stride)
        aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
        closest_size, closest_ratio = get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)

        height = closest_size[0]
        width = closest_size[1]

        return height, width

    @property
    def vae_spatial_compression_ratio(self):
        if hasattr(self.vae.config, "ffactor_spatial"):
            return self.vae.config.ffactor_spatial
        else:
            return 16

    @property
    def vae_temporal_compression_ratio(self):
        if hasattr(self.vae.config, "ffactor_temporal"):
            return self.vae.config.ffactor_temporal
        else:
            return 4

    def get_latent_size(self, video_length, height, width):
        spatial_compression_ratio = self.vae_spatial_compression_ratio
        temporal_compression_ratio = self.vae_temporal_compression_ratio
        video_length = (video_length - 1) // temporal_compression_ratio + 1
        height, width = height // spatial_compression_ratio, width // spatial_compression_ratio

        assert height > 0 and width > 0 and video_length > 0, f"height: {height}, width: {width}, video_length: {video_length}"

        return video_length, height, width

    @staticmethod
    def _load_image_input(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        raise TypeError(f"Unsupported image input type: {type(image)}")

    @classmethod
    def _normalize_image_list(cls, images, field_name):
        if images is None:
            return None
        if isinstance(images, (str, Image.Image, np.ndarray)):
            return [cls._load_image_input(images)]
        if isinstance(images, tuple):
            images = list(images)
        if isinstance(images, list):
            return [cls._load_image_input(image) for image in images if image is not None]
        raise TypeError(f"{field_name} must be an image/path or a list of images/paths, got {type(images)}")

    @classmethod
    def _normalize_batched_image_list(cls, images, batch_size, field_name):
        if images is None:
            return None
        if isinstance(images, (str, Image.Image, np.ndarray)):
            return [[cls._load_image_input(images)]]
        if isinstance(images, tuple):
            images = list(images)
        if not isinstance(images, list):
            raise TypeError(f"{field_name} must be an image/path or a list, got {type(images)}")

        if len(images) == batch_size:
            batched = []
            for item in images:
                if isinstance(item, (list, tuple)):
                    batched.append(cls._normalize_image_list(list(item), field_name))
                else:
                    batched.append([cls._load_image_input(item)])
            return batched

        if batch_size == 1:
            return [cls._normalize_image_list(images, field_name)]

        raise ValueError(
            f"{field_name} must contain one image/list per prompt, "
            f"got {len(images)} entries for batch size {batch_size}"
        )

    @classmethod
    def _normalize_reference_images(cls, reference_image, batch_size, task_type):
        """Return one single-element RGB list per sample for `reference_image`."""
        reference_images = cls._normalize_batched_image_list(reference_image, batch_size, "reference_image")
        if reference_images is None:
            raise ValueError(f"{task_type} requires `reference_image`")
        for idx, imgs in enumerate(reference_images):
            if len(imgs) != 1:
                raise ValueError(
                    f"{task_type} expects exactly one reference image per sample, got {len(imgs)} for sample {idx}"
                )
        return reference_images

    @staticmethod
    def _resize_and_center_crop_pil(image, target_height, target_width):
        original_width, original_height = image.size
        if target_height / original_height > target_width / original_width:
            resize_height = target_height
            resize_width = int(original_width * target_height / original_height)
        else:
            resize_height = int(original_height * target_width / original_width)
            resize_width = target_width
        image_transform = transforms.Compose([
            transforms.Resize((resize_height, resize_width), antialias=True),
            transforms.CenterCrop((target_height, target_width)),
        ])
        return image_transform(image)

    @staticmethod
    def _resize_and_pad_pil(image, target_height, target_width):
        """Scale-to-fit (no cropping), then pad to the exact target size by
        stretching the resized image's border pixels outward (edge
        replication) and blurring just that padded border -- the resized
        image itself stays sharp in the center. Used for reference images
        (ti_i2i/tiv2v) where cropping could cut off the reference subject;
        a blurred edge-extended border is closer to natural image statistics
        than a hard solid-color border.
        """
        original_width, original_height = image.size
        scale = min(target_width / original_width, target_height / original_height)
        resize_width = max(1, round(original_width * scale))
        resize_height = max(1, round(original_height * scale))
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

    def _encode_pil_images_to_latents(
        self, pil_images, target_height, target_width, device, resize_mode="crop"
    ):
        """Encode PIL images as single-frame latents (the ti_i2i reference)."""
        resize_fn = self._resize_and_pad_pil if resize_mode == "pad" else self._resize_and_center_crop_pil
        processed = [
            resize_fn(image, target_height, target_width)
            for image in pil_images
        ]
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        pixel_values = torch.stack([image_transform(image) for image in processed])
        pixel_values = pixel_values.unsqueeze(2).to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True), self.vae.memory_efficient_context():
            latents = self.vae.encode(pixel_values).latent_dist.mode()
            if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            else:
                latents = latents * self.vae.config.scaling_factor
        return latents

    def get_task_specific_input(
        self,
        prompt,
        task_type=None,
        source_image=None,
        source_image_latents=None,
        reference_image=None,
        source_video=None,
        source_video_latents=None,

        video_length=None,
        target_resolution=None,
        target_height=None,
        target_width=None,
        device = '',
    ):
        if task_type not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task_type={task_type!r}; expected one of {SUPPORTED_TASKS}")

        if video_length is not None:
            latent_target_length = (video_length - 1) // self.vae_temporal_compression_ratio + 1
        else:
            latent_target_length = None

        if reference_image is not None and task_type not in REFERENCE_TASKS:
            raise ValueError(
                f"`reference_image` is only used by {REFERENCE_TASKS}, got task_type={task_type!r}"
            )

        modality_token = get_special_token_from_task(task_type)

        # Outputs. semantic_images / condition_images / video_sample_frames feed
        # the Qwen2.5-VL prompt; vision_images / vision_images2 feed SigLIP.
        semantic_images = None
        condition_images = None
        cond_latents = None
        vision_images = None
        vision_images2 = None
        video_sample_frames = None

        if task_type in IMAGE_EDIT_TASKS:
            # i2i / ti_i2i run on a static clip of the source image; the caller
            # passes its latents, which condition every latent frame.
            if source_image_latents is None:
                raise ValueError(f"{task_type} requires `source_image_latents`")
            clip_length = (source_image_latents.shape[2] - 1) * self.vae_temporal_compression_ratio + 1
            if video_length is None:
                video_length = clip_length

            batch_size = len(prompt) if isinstance(prompt, list) else 1

            input_images = self._normalize_batched_image_list(source_image, batch_size, "source_image")
            if input_images is None:
                raise ValueError(f"{task_type} requires `source_image`")
            for idx, imgs in enumerate(input_images):
                if len(imgs) != 1:
                    raise ValueError(f"{task_type} expects exactly one source image per sample, got {len(imgs)} for sample {idx}")

            reference_images = None
            if task_type == 'ti_i2i':
                reference_images = self._normalize_reference_images(reference_image, batch_size, task_type)

            first_input = input_images[0][0]
            if target_height is not None or target_width is not None:
                if target_height is None or target_width is None:
                    raise ValueError("target_height and target_width must be provided together")
                height, width = int(target_height), int(target_width)
            else:
                height, width = self.get_closest_resolution_given_reference_image(first_input, target_resolution)
            latent_target_length = (video_length - 1) // self.vae_temporal_compression_ratio + 1
            latent_target_length, latent_height, latent_width = self.get_latent_size(video_length, height, width)

            cond_latents = torch.zeros(
                batch_size,
                self.transformer.config.in_channels,
                latent_target_length,
                latent_height,
                latent_width,
                device=device,
                dtype=self.target_dtype,
            )

            prompt_condition_images = []
            vision_reference_images = []
            for idx, imgs in enumerate(input_images):
                input_latents = source_image_latents[idx:idx + 1].to(device)
                if tuple(input_latents.shape[2:]) != (latent_target_length, latent_height, latent_width):
                    raise ValueError(
                        f"{task_type}: `source_image_latents` has size {tuple(input_latents.shape[2:])}, but "
                        f"video_length={video_length} at {width}x{height} expects "
                        f"{(latent_target_length, latent_height, latent_width)}"
                    )
                cond_latents[idx] = input_latents[0].to(dtype=self.target_dtype)
                visual_inputs = [self._resize_and_center_crop_pil(imgs[0], height, width)]

                if task_type == 'ti_i2i':
                    ref_imgs = reference_images[idx]
                    ref_latents = self._encode_pil_images_to_latents(
                        ref_imgs[:1], height, width, device, resize_mode="pad"
                    )
                    # Reference goes on latent frame 1 (tiv2v's convention,
                    # matching the trainer's condition).
                    cond_latents[idx, :, 1] += ref_latents[0][:, 0].to(dtype=self.target_dtype)
                    visual_inputs.append(self._resize_and_pad_pil(ref_imgs[0], height, width))

                prompt_condition_images.append(visual_inputs)
                vision_reference_images.append(visual_inputs)

            semantic_images = None
            condition_images = prompt_condition_images
            vision_images = vision_reference_images
            vision_images2 = None
            video_sample_frames = None


        elif task_type in VIDEO_EDIT_TASKS:
            if source_video_latents is None:
                raise ValueError(f"{task_type} requires `source_video_latents`")
            # One entry per sample; a single path is treated as a batch of one.
            source_videos = source_video
            if source_videos is not None and not isinstance(source_videos, (list, tuple)):
                source_videos = [source_videos]

            num_video_frames = (source_video_latents.shape[2]-1) * 4 + 1
            if num_video_frames >= 24 * 4:
                nframes = 8
            elif num_video_frames >= 24 * 3:
                nframes = 6
            else:
                nframes = 4
            
            if task_type in REFERENCE_VIDEO_EDIT_TASKS:
                cond_latents = source_video_latents.to(device)
                video_length = (source_video_latents.shape[2] - 1) * self.vae_temporal_compression_ratio + 1
                vae_scale_factor = self.vae_spatial_compression_ratio
                height = int(cond_latents.shape[-2] * vae_scale_factor)
                width = int(cond_latents.shape[-1] * vae_scale_factor)

                # One RGB reference per source video (paths are opened with
                # Image.open(path).convert("RGB")).
                reference_batch_size = (
                    len(source_videos) if source_videos is not None
                    else (len(prompt) if isinstance(prompt, list) else 1)
                )
                condition_images = [
                    imgs[0]
                    for imgs in self._normalize_reference_images(reference_image, reference_batch_size, task_type)
                ]
                target_height, target_width = height, width
                # Reference image: scale-to-fit (no cropping), then pad.
                condition_images = [
                    self._resize_and_pad_pil(_img, target_height, target_width)
                    for _img in condition_images
                ]

                semantic_images_np, sampled_frames, cond_latents_img, input_img_pil_np = get_semantic_images_np2(video_path_list=source_videos, first_image_pil=condition_images, vae=self.vae, F=source_video_latents.shape[2], nframes=nframes)
                cond_latents = cond_latents + cond_latents_img

                if sampled_frames is not None:
                    target_height, target_width = height, width

                    def _resize_and_center_crop(frame):
                        original_width, original_height = frame.size
                        if target_height / original_height > target_width / original_width:
                            resize_height = target_height
                            resize_width = int(original_width * target_height / original_height)
                        else:
                            resize_height = int(original_height * target_width / original_width)
                            resize_width = target_width
                        resize_transform = transforms.Compose([
                            transforms.Resize((resize_height, resize_width), antialias=True),
                            transforms.CenterCrop((target_height, target_width)),
                        ])
                        return resize_transform(frame)

                    resized_frames = []
                    for frame_list in sampled_frames:
                        if frame_list is None:
                            resized_frames.append(frame_list)
                            continue
                        resized_frames.append([
                            _resize_and_center_crop(frame)
                            for frame in frame_list
                        ])
                    sampled_frames = resized_frames
                
                if semantic_images_np is not None:
                    semantic_images_np = np.stack([
                        resize_and_center_crop(image, width, height)
                        for image in semantic_images_np
                    ], axis=0)

                semantic_images = condition_images
                vision_images2 = None
                if input_img_pil_np is not None:
                    vision_images2 = np.stack([
                        resize_and_center_crop(image, width, height)
                        for image in input_img_pil_np
                    ], axis=0)
                condition_images = None
                vision_images = semantic_images_np
                video_sample_frames=sampled_frames
            
            else:
                semantic_images_np, sampled_frames = get_semantic_images_np(source_videos, nframes=nframes)
                cond_latents = source_video_latents.to(device)
                video_length = (source_video_latents.shape[2] - 1) * self.vae_temporal_compression_ratio + 1
                vae_scale_factor = self.vae_spatial_compression_ratio
                height = int(cond_latents.shape[-2] * vae_scale_factor)
                width = int(cond_latents.shape[-1] * vae_scale_factor)

                if sampled_frames is not None:
                    target_height, target_width = height, width

                    def _resize_and_center_crop(frame):
                        original_width, original_height = frame.size
                        if target_height / original_height > target_width / original_width:
                            resize_height = target_height
                            resize_width = int(original_width * target_height / original_height)
                        else:
                            resize_height = int(original_height * target_width / original_width)
                            resize_width = target_width
                        resize_transform = transforms.Compose([
                            transforms.Resize((resize_height, resize_width), antialias=True),
                            transforms.CenterCrop((target_height, target_width)),
                        ])
                        return resize_transform(frame)

                    resized_frames = []
                    for frame_list in sampled_frames:
                        if frame_list is None:
                            resized_frames.append(frame_list)
                            continue
                        resized_frames.append([
                            _resize_and_center_crop(frame)
                            for frame in frame_list
                        ])
                    sampled_frames = resized_frames

                if semantic_images_np is not None:
                    semantic_images_np = np.stack([
                        resize_and_center_crop(image, width, height)
                        for image in semantic_images_np
                    ], axis=0)
                
                vision_images = semantic_images_np
                vision_images2 = None
                condition_images = None
                semantic_images = None
                video_sample_frames=sampled_frames

        latent_target_length, latent_height, latent_width = self.get_latent_size(video_length, height, width)
        n_tokens = latent_target_length * latent_height * latent_width
        
        return {
            "task_type": task_type,
            "modality_token": modality_token,
            "prompt": prompt,
            "semantic_images": semantic_images,
            "condition_images": condition_images,
            "cond_latents": cond_latents,
            "vision_images": vision_images,
            "vision_images2": vision_images2,
            "video_sample_frames": video_sample_frames,
            "target_resolution": target_resolution,
            "height": height,
            "width": width,
            "video_length": video_length,
            "latent_target_length": latent_target_length,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "n_tokens": n_tokens,
            
        }


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        aspect_ratio: str,
        video_length: int,
        num_inference_steps: int = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        seed: Optional[int] = None,
        flow_shift: Optional[float] = None,
        embedded_guidance_scale: Optional[float] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        enable_vae_tile_parallelism: bool = True,
        task_type = None,
        deepstack=[8,16,24],
        setclip=True,
        source_image=None,
        source_image_latents=None,
        reference_image=None,
        source_video=None,
        source_video_latents=None,
        target_height=None,
        target_width=None,
        **kwargs,
    ):
        r"""
        Runs an editing task (i2i, ti_i2i, v2v, tiv2v or 3d23d) conditioned on text and visual inputs.

        Args:
            prompt (`str` or `List[str]`):
                Text instruction(s) guiding the edit.
            aspect_ratio (`str`):
                Aspect ratio string like "720:1280". Only used for logging; the output size follows the inputs.
            video_length (`int`):
                Number of frames in the generated video. For i2i/ti_i2i it must be 5: the source image is
                encoded as a static clip of 5 pixel frames that conditions every latent frame, the ti_i2i
                reference image is added on latent frame 1, and the edited image is read from the generated frames.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of denoising steps during generation. Larger values may improve video quality at the expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to value in config):
                Scale to encourage the model to better follow the prompt. `guidance_scale > 1` enables classifier-free guidance.
            negative_prompt (`str` or `List[str]`, *optional*):
                Negative prompt(s) that describe what should NOT be shown in the generated video.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                PyTorch random generator(s) for deterministic results.
            seed (`int`, *optional*):
                If specified, used to create the generator for reproducible sampling.
            flow_shift (`float`, *optional*):
                Flow shift parameter for the scheduler. Overrides the default pipeline configuration if provided.
            embedded_guidance_scale (`float`, *optional*):
                Additional control guidance scale, if supported.
            task_type (`str`):
                One of "i2i", "ti_i2i", "v2v", "tiv2v", "3d23d".
            source_image (PIL.Image, `str`, or a list with one entry per prompt, *optional*):
                i2i / ti_i2i: the image to edit, shown to the text and vision encoders.
            source_image_latents (`torch.Tensor`, *optional*):
                i2i / ti_i2i: scaled VAE latents of the source image as a static clip, shape
                (B, C, T, H, W). They are the condition and set the clip length.
            reference_image (PIL.Image or `str`, *optional*):
                ti_i2i / tiv2v: the reference image, exactly one per sample (a one-element list is
                accepted; for a batch pass one entry per sample). Not accepted by the other tasks.
            source_video (`str` or `List[str]`, *optional*):
                v2v / tiv2v / 3d23d: path(s) of the video to edit, one per sample. Read for the frames
                shown to the text encoder and the first frame given to the vision encoder.
            source_video_latents (`torch.Tensor`, *optional*):
                v2v / tiv2v / 3d23d: VAE latents of the source video, shape (B, C, T, H, W). They are the
                condition and set the output size and length.
            output_type (`str`, *optional*, defaults to "pt"):
                Output format of the returned video(s). Accepted values: `"pt"` for torch.Tensor or `"np"` for numpy.ndarray.
            return_dict (`bool`, *optional*, defaults to True):
                Whether to return a [`HunyuanVideoPipelineOutput`] or a tuple.
            **kwargs:
                Only `guidance_rescale`, `clip_skip` and `eta` are accepted; any other name raises a TypeError.

        Returns:
            HunyuanVideoPipelineOutput or `tuple`:
                If `return_dict` is True, returns a [`HunyuanVideoPipelineOutput`] with fields:
                    - `videos`: Generated video(s) as a tensor or numpy array.
                Otherwise, returns a tuple containing the outputs as above.

        Example:
            ```python
            pipe = HunyuanVideo_1_5_Pipeline.create_pipeline("your_model_dir")
            # Instruction-based image editing (image_latents: VAE latents of the image as a 5-frame clip)
            image = pipe(prompt="Make the sky orange", source_image="img.jpg", task_type="i2i",
                         aspect_ratio="1:1", source_image_latents=image_latents).videos
            # Reference-guided image editing
            image = pipe(prompt="Put this hat on the cat", source_image="cat.jpg",
                         reference_image="hat.jpg", task_type="ti_i2i",
                         aspect_ratio="1:1", source_image_latents=image_latents).videos
            # Video editing (source_video_latents: VAE latents of the source video)
            video = pipe(prompt="Turn it into winter", task_type="v2v", aspect_ratio="16:9",
                         video_length=121, source_video="in.mp4",
                         source_video_latents=latents).videos
            # Reference-guided video editing
            video = pipe(prompt="Replace the dog with this one", task_type="tiv2v", aspect_ratio="16:9",
                         video_length=121, source_video="in.mp4", source_video_latents=latents,
                         reference_image="dog.jpg").videos
            ```
        """
        unexpected = sorted(set(kwargs) - {"guidance_rescale", "clip_skip", "eta"})
        if unexpected:
            raise TypeError(f"__call__() got unexpected argument(s): {unexpected}")

        num_videos_per_prompt = 1
        target_resolution = '480p'

        if guidance_scale is None:
            guidance_scale = self.config.guidance_scale
        if embedded_guidance_scale is None:
            embedded_guidance_scale = self.config.embedded_guidance_scale
        if flow_shift is None:
            flow_shift = self.config.flow_shift
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps

        if embedded_guidance_scale is not None:
            assert not self.do_classifier_free_guidance
            assert self.transformer.config.guidance_embed
        else:
            assert not self.transformer.config.guidance_embed
        
        device = self.execution_device

        with auto_offload_model(self.vae, self.execution_device, enabled=self.enable_offloading):
            task_inputs = self.get_task_specific_input(
                prompt=prompt,
                task_type=task_type,

                source_image=source_image,
                source_image_latents=source_image_latents,
                reference_image=reference_image,
                source_video=source_video,
                source_video_latents=source_video_latents,
                video_length=video_length,
                target_resolution=target_resolution,
                target_height=target_height,
                target_width=target_width,
                device=device,
            )
        
        task_type = task_inputs["task_type"]
        modality_token = task_inputs["modality_token"]
        prompt = task_inputs["prompt"]

        semantic_images = task_inputs["semantic_images"]
        condition_images = task_inputs["condition_images"]
        cond_latents = task_inputs["cond_latents"]
        vision_images = task_inputs["vision_images"]
        vision_images2 = task_inputs["vision_images2"]
        video_sample_frames = task_inputs["video_sample_frames"]

        target_resolution = task_inputs["target_resolution"]
        height = task_inputs["height"]
        width = task_inputs["width"]
        video_length = task_inputs["video_length"]
        latent_target_length = task_inputs["latent_target_length"]
        latent_height = task_inputs["latent_height"]
        latent_width = task_inputs["latent_width"]
        n_tokens = task_inputs["n_tokens"]

        if self.ideal_task is not None and self.ideal_task != task_type:
            raise ValueError(
                f"The loaded pipeline is trained for '{self.ideal_task}' task, but received input for '{task_type}' task. "
                "Please load a pipeline trained for the correct task, or check and update your arguments accordingly."
            )

        if flow_shift is None:
            self.scheduler = self._create_scheduler(self.config.flow_shift)
        else:
            self.scheduler = self._create_scheduler(flow_shift)

        if seed is None or seed == -1:
            seed = random.randint(100000, 999999)

        if get_parallel_state().sp_enabled:
            assert seed is not None
            if dist.is_initialized():
                obj_list = [seed]
                group_src_rank = dist.get_global_rank(get_parallel_state().sp_group, 0)
                dist.broadcast_object_list(obj_list, src=group_src_rank, group=get_parallel_state().sp_group)
                seed = obj_list[0]

        if generator is None and seed is not None:
            generator = torch.Generator(device=self.noise_init_device).manual_seed(seed)

        self._guidance_scale = guidance_scale
        self._guidance_rescale = kwargs.get("guidance_rescale", 0.0)
        self._clip_skip = kwargs.get("clip_skip", None)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        if get_rank() == 0:
            print(
                '\n'
                f"{'=' * 60}\n"
                f"🎬  HunyuanVideo Generation Task\n"
                f"{'-' * 60}\n"
                f"User Prompt:               {prompt}\n"
                f"Negative Prompt:           {negative_prompt}\n"
                f"Aspect Ratio:              {f'{width}:{height}'}\n"
                f"Video Length:              {video_length}\n"
                f"Guidance Scale:            {guidance_scale}\n"
                f"Guidance Embedded Scale:   {embedded_guidance_scale}\n"
                f"Shift:                     {flow_shift}\n"
                f"Seed:                      {seed}\n"
                f"Video Resolution:          {width} x {height}\n"
                f'Attn mode:                 {self.transformer.attn_mode}\n'
                f"Transformer dtype:         {self.transformer.dtype}\n"
                f"Sampling Steps:            {num_inference_steps}\n"
                f"Use Meanflow:              {self.use_meanflow}\n"
                f"Deepstack:                 {deepstack}\n"
                f"Setclip:                   {setclip}\n"
                f"{'=' * 60}"
                '\n'
            )

        with auto_offload_model(self.text_encoder, self.execution_device, enabled=self.enable_offloading):
            if task_type in VIDEO_EDIT_TASKS:
                nframes = None
                if source_video_latents is not None:
                    num_video_frames = (source_video_latents.shape[2] - 1) * self.vae_temporal_compression_ratio + 1
                    if num_video_frames >= 24 * 4:
                        nframes = 8
                    elif num_video_frames >= 24 * 3:
                        nframes = 6
                    else:
                        nframes = 4
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    prompt_mask,
                    negative_prompt_mask,
                    deepstack_hidden_states,
                    negative_deepstack_hidden_states,
                ) = self.encode_prompt(
                    prompt,
                    device,
                    num_videos_per_prompt,
                    self.do_classifier_free_guidance,
                    negative_prompt,
                    clip_skip=None,
                    data_type="video",
                    task_type=task_type,
                    video_sample_frames=video_sample_frames,
                    nframes=nframes,
                    semantic_images=semantic_images,
                    deepstack=deepstack,
                    setclip=setclip,
                    modality_tokens=modality_token,
                )
            else:
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    prompt_mask,
                    negative_prompt_mask,
                    deepstack_hidden_states,
                    negative_deepstack_hidden_states,
                ) = self.encode_prompt(
                    prompt,
                    device,
                    num_videos_per_prompt,
                    self.do_classifier_free_guidance,
                    negative_prompt,
                    clip_skip=self.clip_skip,
                    data_type="video",
                    task_type=task_type,
                    all_condition_pils = condition_images,
                    deepstack=deepstack,
                    setclip=setclip,
                    modality_tokens=modality_token,
                )


        # there is no second encoder
        prompt_embeds_2 = None
        negative_prompt_embeds_2 = None
        prompt_mask_2 = None
        negative_prompt_mask_2 = None

        extra_kwargs = {}
        if self.config.glyph_byT5_v2:
            with auto_offload_model(self.byt5_model, self.execution_device, enabled=self.enable_offloading):
                extra_kwargs = self._prepare_byt5_embeddings(prompt, device)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])
            if deepstack_hidden_states is not None:
                deepstack_hidden_states = torch.cat([negative_deepstack_hidden_states, deepstack_hidden_states])

        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            **extra_set_timesteps_kwargs,
        )

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            latent_height,
            latent_width,
            latent_target_length,
            self.target_dtype,
            device,
            generator,
        )

        cond_latents = self._prepare_cond_latents(
            task_type, cond_latents, latents
        )
        with auto_offload_model(self.vision_encoder, self.execution_device, enabled=self.enable_offloading):
            vision_states = self._prepare_vision_states(
                vision_images, target_resolution, latents, device, reference_image2=vision_images2
            )

        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": generator, "eta": kwargs.get("eta", 0.0)},
        )

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        cache_helper = getattr(self, 'cache_helper', None)
        if cache_helper is not None:
            cache_helper.clear_states()

        with self.progress_bar(total=num_inference_steps) as progress_bar, auto_offload_model(self.transformer, self.execution_device, enabled=self.enable_offloading):
            for i, t in enumerate(timesteps):
                if cache_helper is not None:
                    cache_helper.cur_timestep = i
                latents_concat = torch.concat([latents, cond_latents], dim=1)
                latent_model_input = torch.cat([latents_concat] * 2) if self.do_classifier_free_guidance else latents_concat

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                t_expand = t.repeat(latent_model_input.shape[0])
                if self.use_meanflow:
                    if i == len(timesteps) - 1:
                        timesteps_r = torch.tensor([0.0], device=self.execution_device)
                    else:
                        timesteps_r = timesteps[i + 1]
                    timesteps_r = timesteps_r.repeat(latent_model_input.shape[0])
                else:
                    timesteps_r = None

                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(self.target_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
                    output = self.transformer(
                        latent_model_input,
                        t_expand,
                        prompt_embeds,
                        prompt_embeds_2,
                        prompt_mask,
                        timestep_r=timesteps_r,
                        vision_states=vision_states,
                        mask_type=task_type,
                        guidance=guidance_expand,
                        return_dict=False,
                        all_stack_text_states=deepstack_hidden_states,
                        extra_kwargs=extra_kwargs,
                    )
                    noise_pred = output[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    if progress_bar is not None:
                        progress_bar.update()

        if output_type == "latent":
            video_frames = latents
        else:
            if len(latents.shape) == 4:
                latents = latents.unsqueeze(2)
            elif len(latents.shape) != 5:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
                )

            if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            else:
                latents = latents / self.vae.config.scaling_factor

            if enable_vae_tile_parallelism and hasattr(self.vae, 'enable_tile_parallelism'):
                self.vae.enable_tile_parallelism()

            with torch.autocast(device_type="cuda", dtype=self.vae_dtype, enabled=self.vae_autocast_enabled), auto_offload_model(self.vae, self.execution_device, enabled=self.enable_offloading), self.vae.memory_efficient_context():
                video_frames = self.vae.decode(latents, return_dict=False, generator=generator)[0]

            if video_frames is not None:
                video_frames = (video_frames / 2 + 0.5).clamp(0, 1).cpu().float()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return video_frames

        return HunyuanVideoPipelineOutput(videos=video_frames)

    @property
    def ideal_resolution(self):
        return self.transformer.config.ideal_resolution

    @property
    def ideal_task(self):
        return self.transformer.config.ideal_task

    @property
    def use_meanflow(self):
        return self.transformer.config.use_meanflow

    def apply_infer_optimization(
        self,
        infer_state: Optional[InferState] = None,
        enable_offloading: bool = False,
        enable_group_offloading: bool = False,
        overlap_group_offloading: bool = True,
    ):
        """
        Apply inference optimizations to transformer based on infer_state.
        
        Args:
            infer_state: Optional InferState object containing optimization settings.
            enable_offloading: Whether to enable CPU offloading.
            enable_group_offloading: Whether to enable group offloading.
            overlap_group_offloading: Whether to use overlapping group offloading.
        """
        if infer_state is not None:

            if infer_state.use_fp8_gemm:
                from angelslim.compressor.diffusion import DynamicDiTQuantizer
                quant_type = infer_state.quant_type
                include_patterns = infer_state.include_patterns
                quantizer = DynamicDiTQuantizer(quant_type=quant_type, include_patterns=include_patterns)
                quantizer.convert_linear(self.transformer)

            if infer_state.enable_torch_compile:
                # block-wise compile
                for block in self.transformer.double_blocks:
                    block.forward = torch.compile(block.forward)

            # Apply sageattn if enabled
            if infer_state.enable_sageattn:
                self.transformer.set_attn_mode('sageattn')
            
            # Apply cache if enabled
            if infer_state.enable_cache:
                if not is_angelslim_available():
                    raise RuntimeError("Please install angelslim==0.2.1 via `pip install angelslim==0.2.1` to enable cache.")
                from angelslim.compressor.diffusion import DeepCacheHelper, TeaCacheHelper, TaylorCacheHelper
                no_cache_steps = list(range(0, infer_state.cache_start_step)) + list(range(infer_state.cache_start_step, infer_state.cache_end_step, infer_state.cache_step_interval)) + list(range(infer_state.cache_end_step, infer_state.total_steps))
                cache_type = infer_state.cache_type
                if cache_type == 'deepcache':
                    no_cache_block_id = {"double_blocks": infer_state.no_cache_block_id}
                    self.cache_helper = DeepCacheHelper(
                        double_blocks=self.transformer.double_blocks,
                        no_cache_steps=no_cache_steps,
                        no_cache_block_id=no_cache_block_id,
                    )
                elif cache_type == 'teacache':
                    self.cache_helper = TeaCacheHelper(
                        double_blocks=self.transformer.double_blocks,
                        no_cache_steps=no_cache_steps,
                    )
                elif cache_type == 'taylorcache':
                    self.cache_helper = TaylorCacheHelper(
                        double_blocks=self.transformer.double_blocks,
                        no_cache_steps=no_cache_steps,
                    )
                else:
                    raise ValueError(f"Unknown cache type: {cache_type}. Only 'deepcache', 'teacache', 'taylorcache' are supported.")
                self.cache_helper.enable()
            else:
                self.cache_helper = None
        else:
            self.cache_helper = None
        
        # Set enable_offloading
        self.enable_offloading = enable_offloading
        
        # Apply group offloading if enabled
        if enable_group_offloading:
            assert enable_offloading, "enable_group_offloading requires enable_offloading to be True"
            group_offloading_kwargs = {
                'onload_device': torch.device('cuda'),
                'num_blocks_per_group': 1 if overlap_group_offloading else 4,
            }
            if overlap_group_offloading:
                group_offloading_kwargs['use_stream'] = True
            self.transformer.enable_group_offload(**group_offloading_kwargs)


    @staticmethod
    def _load_text_encoder_ckpt(text_encoder, ckpt_path):
        """Load a fine-tuned text encoder checkpoint into the text encoder model."""
        if ckpt_path.endswith(".safetensors"):
            from safetensors import safe_open

            te_weights = {}
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    te_weights[key] = f.get_tensor(key)
        else:
            te_data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if isinstance(te_data, dict) and "state_dict" in te_data:
                te_data = te_data["state_dict"]
            te_weights = te_data
        if isinstance(te_weights, dict):
            te_weights = {k: v for k, v in te_weights.items() if k != "__metadata__"}
        text_encoder.model.load_state_dict(te_weights, strict=True)
        loguru.logger.info(f"Loaded text encoder checkpoint from {ckpt_path}")

    @staticmethod
    def _get_checkpoint_root(checkpoint_path):
        if not checkpoint_path:
            return None
        normalized_path = os.path.normpath(checkpoint_path)
        if os.path.basename(normalized_path) == "transformer":
            return os.path.dirname(normalized_path)
        return normalized_path

    @staticmethod
    def _get_modality_token_embeddings_path(checkpoint_path):
        normalized_path = os.path.normpath(checkpoint_path)
        if normalized_path.endswith(".pt"):
            return normalized_path
        checkpoint_root = HunyuanVideo_1_5_Pipeline._get_checkpoint_root(normalized_path)
        return os.path.join(checkpoint_root, "modality_token_embeddings.pt")

    @staticmethod
    def _load_modality_token_embeddings(text_encoder, checkpoint_path, *, strict=False):
        if not checkpoint_path:
            return False

        modality_token_embeddings_path = HunyuanVideo_1_5_Pipeline._get_modality_token_embeddings_path(
            checkpoint_path
        )
        if not os.path.exists(modality_token_embeddings_path):
            message = f"Modality token embedding checkpoint not found: {modality_token_embeddings_path}"
            if strict:
                raise FileNotFoundError(message)
            loguru.logger.warning(message)
            return False

        try:
            modality_state = torch.load(
                modality_token_embeddings_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            modality_state = torch.load(modality_token_embeddings_path, map_location="cpu")

        tokens = list(modality_state["tokens"])
        weights = modality_state["weight"]
        tokenizer = text_encoder.tokenizer
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        missing_tokens = [
            token for token, token_id in zip(tokens, token_ids)
            if token_id is None or token_id == tokenizer.unk_token_id
        ]
        if missing_tokens:
            raise ValueError(
                f"Modality tokens are not registered in text_encoder.tokenizer: {missing_tokens}"
            )

        processor_tokenizer = getattr(getattr(text_encoder, "processor", None), "tokenizer", None)
        if processor_tokenizer is not None:
            processor_token_ids = processor_tokenizer.convert_tokens_to_ids(tokens)
            if list(processor_token_ids) != list(token_ids):
                raise ValueError(
                    "Tokenizer id mismatch for modality tokens: "
                    f"text_encoder.tokenizer={dict(zip(tokens, token_ids))}, "
                    f"processor.tokenizer={dict(zip(tokens, processor_token_ids))}"
                )
            loguru.logger.info(f"Modality token ids verified: {dict(zip(tokens, token_ids))}")
        else:
            loguru.logger.warning(
                "text_encoder.processor.tokenizer is unavailable; skipped modality token id cross-check"
            )

        state_token_ids = modality_state.get("token_ids")
        if state_token_ids is not None:
            state_token_ids = state_token_ids.cpu().tolist()
            if list(state_token_ids) != list(token_ids):
                loguru.logger.warning(
                    "Checkpoint modality token ids differ from current tokenizer ids; "
                    f"checkpoint={state_token_ids}, current={token_ids}. "
                    "Using current tokenizer ids for merge."
                )

        embedding = text_encoder.model.get_input_embeddings()
        if embedding is None:
            raise ValueError("Text encoder does not expose input embeddings.")
        if weights.shape[0] != len(token_ids):
            raise ValueError(
                f"Expected {len(token_ids)} modality embedding rows, got {weights.shape[0]}"
            )
        if weights.shape[-1] != embedding.weight.shape[-1]:
            raise ValueError(
                "Modality embedding dim mismatch: "
                f"checkpoint={weights.shape[-1]}, model={embedding.weight.shape[-1]}"
            )

        token_ids_tensor = torch.tensor(token_ids, device=embedding.weight.device, dtype=torch.long)
        with torch.no_grad():
            embedding.weight.index_copy_(
                0,
                token_ids_tensor,
                weights.to(device=embedding.weight.device, dtype=embedding.weight.dtype),
            )
        loguru.logger.info(f"Merged modality token embeddings from {modality_token_embeddings_path}")
        return True

    def load_modality_token_embeddings(self, checkpoint_path, *, strict=False):
        return self._load_modality_token_embeddings(
            self.text_encoder, checkpoint_path, strict=strict
        )

    @classmethod
    def create_pipeline(cls, pretrained_model_name_or_path, transformer_dtype=torch.bfloat16, device=None, transformer_init_device=None, pipeline_config='omniweaving', modality_token_checkpoint_path=None, **kwargs):
        # use snapshot download here to get it working from from_pretrained

        if not os.path.isdir(pretrained_model_name_or_path):
            if pretrained_model_name_or_path.count("/") > 1:
                raise ValueError(
                    f'The provided pretrained_model_name_or_path "{pretrained_model_name_or_path}"'
                    " is neither a valid local path nor a valid repo id. Please check the parameter."
                )
            cached_folder = cls.download(
                pretrained_model_name_or_path,
                **kwargs,
            )
        else:
            cached_folder = pretrained_model_name_or_path

        if device is None:
            device = torch.device('cuda')
        
        if transformer_init_device is None:
            transformer_init_device = device

        from_pretrain_kwargs = {
            'pretrained_model_name_or_path': os.path.join(cached_folder, "transformer"),
        }

        vae_inference_config = cls.get_vae_inference_config()
        transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
            **from_pretrain_kwargs,
            torch_dtype=transformer_dtype, 
            low_cpu_mem_usage=True,
        ).to(transformer_init_device)

        vae = hunyuanvideo_15_vae.AutoencoderKLConv3D.from_pretrained(
            os.path.join(cached_folder, "vae"), 
            torch_dtype=vae_inference_config['dtype']
        ).to(device)
        vae.set_tile_sample_min_size(vae_inference_config['sample_size'], vae_inference_config['tile_overlap_factor'])
        scheduler = FlowMatchDiscreteScheduler.from_pretrained(os.path.join(cached_folder, "scheduler"))

        byt5_kwargs, prompt_format = cls._load_byt5(cached_folder, True, 256, device=device)
        text_encoder, text_encoder_2 = cls._load_text_encoders(cached_folder, device=device)
        
        text_encoder_ckpt = os.path.join(cached_folder, "text_encoder", "ckpt", "text_encoder_model.safetensors")
        if os.path.exists(text_encoder_ckpt):
            loguru.logger.info(f"Loading text encoder checkpoint from {text_encoder_ckpt}")
            cls._load_text_encoder_ckpt(text_encoder, text_encoder_ckpt)
            loguru.logger.info(f"Text encoder checkpoint loaded successfully.")
        else:
            loguru.logger.warning(f"Text encoder checkpoint not found at {text_encoder_ckpt}. Using default text encoder.")
        if modality_token_checkpoint_path is not None:
            cls._load_modality_token_embeddings(
                text_encoder, modality_token_checkpoint_path, strict=True
            )
        vision_encoder = cls._load_vision_encoder(cached_folder, device=device)

        pipeline = cls(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
            progress_bar_config=None,
            byt5_model=byt5_kwargs["byt5_model"],
            byt5_tokenizer=byt5_kwargs["byt5_tokenizer"],
            byt5_max_length=byt5_kwargs["byt5_max_length"],
            prompt_format=prompt_format,
            execution_device='cuda',
            vision_encoder=vision_encoder,
            enable_offloading=False,
            **PIPELINE_CONFIGS[pipeline_config],
        )

        return pipeline

    @staticmethod
    def get_offloading_config(memory_limitation=None):
        if memory_limitation is None:
            memory_limitation = get_gpu_memory()
        GB = 1024 * 1024 * 1024
        if memory_limitation < 60 * GB:
            return {
                'enable_offloading': True,
                'enable_group_offloading': True,
            }
        else:
            return {
                'enable_offloading': True,
                'enable_group_offloading': False,
            }

    @staticmethod
    def get_vae_inference_config(memory_limitation=None):
        if memory_limitation is None:
            memory_limitation = get_gpu_memory()
        GB = 1024 * 1024 * 1024
        if memory_limitation > 21 * GB:
            sample_size = 256
            tile_overlap_factor = 0.125
            dtype = torch.float16
        else:
            sample_size = 128
            tile_overlap_factor = 0.25
            dtype = torch.float16
        return {'sample_size': sample_size, 'tile_overlap_factor': tile_overlap_factor, 'dtype': dtype}


    @classmethod
    def _load_text_encoders(cls, pretrained_model_path, device):
        text_encoder_path = f'{pretrained_model_path}/text_encoder/llm'
        if not os.path.exists(text_encoder_path):
            msg = f"{text_encoder_path} not found. Please refer to checkpoints-download.md to download the text encoder checkpoints."
            loguru.logger.error(msg)
            raise FileNotFoundError(msg)
        text_encoder = TextEncoder(
            text_encoder_type='llm',
            tokenizer_type='llm',
            text_encoder_path=text_encoder_path,
            max_length=4096,
            text_encoder_precision="fp16",
            prompt_template=PROMPT_TEMPLATE['li-dit-encode-image-json'],
            prompt_template_video=PROMPT_TEMPLATE['li-dit-encode-video-json'],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=loguru.logger,
            device=device,
        )
        text_encoder_2 = None

        return text_encoder, text_encoder_2

    @classmethod
    def _load_vision_encoder(cls, pretrained_model_name_or_path, device):
        vision_encoder_path = f'{pretrained_model_name_or_path}/vision_encoder/siglip'
        if not os.path.exists(vision_encoder_path):
            msg = f"{vision_encoder_path} not found. Please refer to checkpoints-download.md to download the vision encoder checkpoints."
            loguru.logger.error(msg)
            raise FileNotFoundError(msg)
        vision_encoder = VisionEncoder(
            vision_encoder_type="siglip",
            vision_encoder_precision="fp16",
            vision_encoder_path=vision_encoder_path,
            processor_type=None,
            processor_path=None,
            output_key=None,
            logger=logger,
            device=device,
        )
        return vision_encoder
