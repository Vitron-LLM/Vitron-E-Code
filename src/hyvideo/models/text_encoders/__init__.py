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

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
)
from transformers.utils import ModelOutput

from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
def use_default(value, default):
    """Utility: return value if not None, else default."""
    return value if value is not None else default

# Prompt templates for different models and tasks


__all__ = [
    "C_SCALE", "PROMPT_TEMPLATE",
    "MODEL_BASE",
]

# =================== Constant Values =====================
# Computation scale factor, 1P = 1_000_000_000_000_000. Tensorboard will display the value in PetaFLOPS to avoid
# overflow error when tensorboard logging values.
C_SCALE = 1_000_000_000_000_000

PROMPT_TEMPLATE_ENCODE_IMAGE_JSON = [
    {"role": "system", "content": "You are a helpful assistant. Describe the image by detailing the following aspects: \
        1. The main content and theme of the image. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. The background environment, light, style and atmosphere."},
    {"role": "user", "content": "{}"}
]

PROMPT_TEMPLATE_ENCODE_VIDEO_JSON = [
    {"role": "system", "content": "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."},
    {"role": "user", "content": "{}"}
]

PROMPT_TEMPLATE = {
    "li-dit-encode-image-json": {"template": PROMPT_TEMPLATE_ENCODE_IMAGE_JSON, "crop_start": -1}, # auto-calculate crop_start
    "li-dit-encode-video-json": {"template": PROMPT_TEMPLATE_ENCODE_VIDEO_JSON, "crop_start": -1}, # auto-calculate crop_start
}


system_prompt_1 = {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant. Given a text instruction and an input image, generate an edited image that follows the user's instruction. The edited image should apply the requested changes to the objects, attributes, style, background, scene composition, camera trajectory transformation, or object moving and rotation, while preserving the irrelevant regions, spatial layout, and visual identity of the original image unless the instruction explicitly requires changing them."}]}
system_prompt_2 = {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant. Given a text instruction, an input image, and one or more reference images, generate an edited image by applying the relevant subject, style, object, identity, or scene attributes from the reference images to the input image. The edited image should follow the user's instruction, preserve the irrelevant regions and spatial layout of the input image, and incorporate the specified reference attributes accurately and consistently."}]}
system_prompt_3 = {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant. Given a text instruction and an input video, you need to analyze the visual content and temporal dynamics of the input video, and then explain how the user's text instruction should modify the video's visual style, objects, or scene composition. Generate an edited video that meets the user's requirements, ensuring the specified modifications are applied consistently across frames while preserving the original motion flow and coherence."}]}
system_prompt_4 = {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant. Given a text instruction, a reference image and an input video, you need to analyze the visual content and temporal dynamics of the input video, alongside the scene or subject characteristics of the reference image. Explain how the user's text instruction directs the application of the reference image's visual attributes onto the input video. Generate an edited video that meets the user's requirements, ensuring the specified modifications are applied consistently across frames while preserving the original motion flow and coherence."}]}
system_prompt_5 = {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant. Given an observation video of a 3D scene and a text editing instruction, generate an edited observation video of the same 3D scene. Apply the requested edit while preserving the scene’s static structure and 3D geometric consistency. Temporal changes should mainly come from camera motion or viewpoint changes, rather than object motion, unless explicitly required. Preserve object identity, layout, scale, relative positions, appearance, lighting, and geometry. Ensure the edited scene remains consistent across frames, with plausible perspective, parallax, and occlusion."}]}


from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from univedit.tokens import SPECIAL_TOKENS


MODEL_BASE = os.getenv("MODEL_BASE", "")
TEXT_ENCODER_PATH = {}
TOKENIZER_PATH = {}

PRECISION_TO_TYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


def load_text_encoder(
    text_encoder_type,
    text_encoder_precision=None,
    text_encoder_path=None,
    logger=None,
    device=None,
):
    if text_encoder_path is None:
        if text_encoder_type not in TEXT_ENCODER_PATH:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
        text_encoder_path = TEXT_ENCODER_PATH[text_encoder_type]

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(text_encoder_path, 
                                                 low_cpu_mem_usage=True)

    text_encoder.final_layer_norm = text_encoder.model.language_model.norm
    
    # from_pretrained will ensure that the model is in eval mode.
    if text_encoder_precision is not None:
        text_encoder = text_encoder.to(dtype=PRECISION_TO_TYPE[text_encoder_precision])

    text_encoder.requires_grad_(False)

    if device is not None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path


def load_tokenizer(
    tokenizer_type, tokenizer_path=None, padding_side="right", logger=None
):
    processor = None
    if tokenizer_path is None:
        if tokenizer_type not in TOKENIZER_PATH:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        tokenizer_path = TOKENIZER_PATH[tokenizer_type]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, padding_side=padding_side
    )

    return tokenizer, tokenizer_path, processor


def register_modality_tokens(tokenizer, model=None, logger=None):
    """Register task-modality tags as atomic special tokens."""
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(SPECIAL_TOKENS)},
        replace_additional_special_tokens=False,
    )

    if model is not None:
        embedding = model.get_input_embeddings()
        if embedding is not None and embedding.num_embeddings < len(tokenizer):
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    if logger is not None and num_added:
        logger.info(f"Registered modality special tokens: {SPECIAL_TOKENS}")

    return num_added


@dataclass
class TextEncoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
        hidden_states_list (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        text_outputs (`list`, *optional*, returned when `return_texts=True` is passed):
            List of decoded texts.
    """

    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    hidden_states_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_outputs: Optional[list] = None
    image_features: Optional[list] = None
    deepstack_hidden_states: torch.FloatTensor = None

class TextEncoder(nn.Module):
    def __init__(
        self,
        text_encoder_type: str,
        max_length: int,
        text_encoder_precision: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        tokenizer_type: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        output_key: Optional[str] = None,
        use_attention_mask: bool = True,
        prompt_template: Optional[dict] = None,
        prompt_template_video: Optional[dict] = None,
        hidden_state_skip_layer: Optional[int] = None,
        apply_final_norm: bool = False,
        reproduce: bool = False,
        logger=None,
        device=None,
    ):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.max_length = max_length
        self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.tokenizer_type = (
            tokenizer_type if tokenizer_type is not None else text_encoder_type
        )
        self.tokenizer_path = (
            tokenizer_path if tokenizer_path is not None else text_encoder_path
        )
        self.use_attention_mask = use_attention_mask
        if prompt_template_video is not None:
            assert (
                use_attention_mask is True
            ), "Attention mask is True required when training videos."
        self.prompt_template = prompt_template
        self.prompt_template_video = prompt_template_video
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.reproduce = reproduce
        self.logger = logger

        self.use_template = self.prompt_template is not None
        if self.use_template:
            assert (
                isinstance(self.prompt_template, dict)
                and "template" in self.prompt_template
            ), f"`prompt_template` must be a dictionary with a key 'template', got {self.prompt_template}"
            assert "{}" in str(self.prompt_template["template"]), (
                "`prompt_template['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template['template']}"
            )

        self.use_video_template = self.prompt_template_video is not None
        if self.use_video_template:
            if self.prompt_template_video is not None:
                assert (
                    isinstance(self.prompt_template_video, dict)
                    and "template" in self.prompt_template_video
                ), f"`prompt_template_video` must be a dictionary with a key 'template', got {self.prompt_template_video}"
            assert "{}" in str(self.prompt_template_video["template"]), (
                "`prompt_template_video['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template_video['template']}"
            )

        if text_encoder_type != "llm":
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
        self.output_key = output_key or "last_hidden_state"

        self.model, self.model_path = load_text_encoder(
            text_encoder_type=self.text_encoder_type,
            text_encoder_precision=self.precision,
            text_encoder_path=self.model_path,
            logger=self.logger,
            device=device,
        )

        self.tokenizer, self.tokenizer_path, self.processor = load_tokenizer(
            tokenizer_type=self.tokenizer_type,
            tokenizer_path=self.tokenizer_path,
            padding_side="right",
            logger=self.logger,
        )
        register_modality_tokens(self.tokenizer, self.model, self.logger)
        self.processor = AutoProcessor.from_pretrained(self.tokenizer_path)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            register_modality_tokens(self.processor.tokenizer, self.model, self.logger)
        self.model.requires_grad_(False)

        # pre-calculate crop_start for image and video
        if self.use_template and self.prompt_template is not None:
            self.text2tokens("a photo of a cat", data_type="image")
        if self.use_video_template and self.prompt_template_video is not None:
            self.text2tokens("a photo of a cat", data_type="video")

    @property
    def dtype(self):
        return self.model.dtype
    
    @property
    def device(self):
        return self.model.device

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        """
        Apply text to template.

        Args:
            text (str): Input text.
            template (str or list): Template string or list of chat conversation.
            prevent_empty_text (bool): If Ture, we will prevent the user text from being empty
                by adding a space. Defaults to True.
        """
        if isinstance(template, str):
            # Will send string to tokenizer. Used for llm
            return template.format(text)
        elif isinstance(template, list):
            # For JSON list template format (chat conversation)
            # Create a deep copy to avoid modifying the original template
            template_copy = deepcopy(template)
            for item in template_copy:
                if isinstance(item, dict) and "content" in item:
                    # Replace placeholder with text in the content field
                    item["content"] = item["content"].format(text if text else (" " if prevent_empty_text else ""))
            return template_copy
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    @staticmethod
    def _normalize_modality_tokens(modality_tokens, batch_size):
        if modality_tokens is None:
            return [None] * batch_size
        if isinstance(modality_tokens, str):
            modality_tokens = [modality_tokens] * batch_size
        else:
            modality_tokens = list(modality_tokens)
        if len(modality_tokens) != batch_size:
            raise ValueError(
                f"Expected {batch_size} modality tokens, got {len(modality_tokens)}."
            )
        invalid_tokens = [
            token for token in modality_tokens
            if token is not None and token not in SPECIAL_TOKENS
        ]
        if invalid_tokens:
            raise ValueError(
                f"Invalid modality tokens {invalid_tokens}. Expected one of {SPECIAL_TOKENS}."
            )
        return modality_tokens

    @staticmethod
    def _prepend_modality_token_to_serialized_text(text, token):
        if token is None:
            return text
        return f"{token}{text}"

    def _has_leading_modality_token(self, batch_encoding):
        input_ids = batch_encoding.get("input_ids", None)
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[1] == 0:
            return False
        for token in SPECIAL_TOKENS:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == self.tokenizer.unk_token_id:
                continue
            if bool((input_ids[:, 0] == token_id).all().item()):
                return True
        return False

    def calculate_crop_start(self, tokenized_input):
        """
        Automatically calculate the crop_start position based on identifying user tokens.
        
        Args:
            tokenized_input: The output from the tokenizer containing input_ids
            
        Returns:
            int: The position where the actual prompt content begins (after user markers)
        """
        input_ids = tokenized_input["input_ids"][0].tolist()  # Get the first example's tokens
        
        marker = "<|im_start|>user\n"
            
        # Tokenize just the marker to get its token IDs
        marker_tokens = self.tokenizer(marker, add_special_tokens=False)["input_ids"]
        
        # Find the end position of the marker in the input sequence
        for i in range(len(input_ids) - len(marker_tokens) + 1):
            if input_ids[i:i+len(marker_tokens)] == marker_tokens:
                # Return the position after the marker
                return i + len(marker_tokens)
                
        # If marker not found, try to find based on special tokens
        if hasattr(self.tokenizer, 'special_tokens_map'):
            # Check for user token or any other special token that might indicate user input start
            for token_name, token_value in self.tokenizer.special_tokens_map.items():
                if 'user' in token_name.lower():
                    user_token_id = self.tokenizer.convert_tokens_to_ids(token_value)
                    if user_token_id in input_ids:
                        return input_ids.index(user_token_id) + 1
        
        # Default fallback: return 0 (no cropping)
        return 0

    def text2tokens(self, text, data_type="image", max_length=300):
        """
        Tokenize the input text.

        Args:
            text (str or list): Input text.
        """
        tokenize_input_type = "str"
        if self.use_template or self.use_video_template:
            if data_type == "image":
                prompt_template = self.prompt_template["template"]
                crop_start = self.prompt_template.get("crop_start", -1)
            elif data_type == "video":
                prompt_template = self.prompt_template_video["template"]
                crop_start = self.prompt_template_video.get("crop_start", -1)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            if isinstance(text, (list, tuple)):
                text = [
                    self.apply_text_to_template(one_text, prompt_template)
                    for one_text in text
                ]
                if isinstance(text[0], list):
                    tokenize_input_type = "list"
            elif isinstance(text, str):
                text = self.apply_text_to_template(text, prompt_template)
                if isinstance(text, list):
                    tokenize_input_type = "list"
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")
        
            # First pass: tokenize with arbitrary max_length to find crop_start
            if crop_start == -1:
                # Use temporary max_length for the first pass (large enough)
                temp_kwargs = dict(
                    truncation=True,
                    max_length=256,  # Temporary large value
                    padding="max_length",
                    return_tensors="pt",
                )
                
                # First tokenization pass to calculate crop_start
                if tokenize_input_type == "str":
                    temp_tokenized = self.tokenizer(
                        text,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_attention_mask=True,
                        **temp_kwargs,
                    )
                elif tokenize_input_type == "list":
                    temp_tokenized = self.tokenizer.apply_chat_template(
                        text,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        **temp_kwargs,
                    )
                
                # Calculate the crop_start from this first pass
                crop_start = self.calculate_crop_start(temp_tokenized)
                
                # Store the calculated crop_start for future use
                if data_type == "image":
                    self.prompt_template["crop_start"] = crop_start
                else:
                    self.prompt_template_video["crop_start"] = crop_start
        else:
            crop_start = 0
        
        # Second pass: tokenize with the proper max_length using the found crop_start
        kwargs = dict(
            truncation=True,
            max_length=max_length + (crop_start if crop_start > 0 else 0),
            padding="max_length",
            return_tensors="pt",
        )
        
        if tokenize_input_type == "str":
            tokenized_output = self.tokenizer(
                text,
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=True,
                **kwargs,
            )
        elif tokenize_input_type == "list":
            tokenized_output = self.tokenizer.apply_chat_template(
                text,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")
                
        return tokenized_output

    def encode(
        self,
        batch_encoding,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=None,
        hidden_state_skip_layer=None,
        return_texts=False,
        data_type="image",
        device=None,
        is_uncond=False,
        deepstack=[],
        crop_start=None,
        setclip=False
    ):
        """
        Args:
            batch_encoding (dict): Batch encoding from tokenizer.
            use_attention_mask (bool): Whether to use attention mask. If None, use self.use_attention_mask.
                Defaults to None.
            output_hidden_states (bool): Whether to output hidden states. If False, return the value of
                self.output_key. If True, return the entire output. If set self.hidden_state_skip_layer,
                output_hidden_states will be set True. Defaults to False.
            do_sample (bool): Whether to sample from the model. Used for Decoder-Only LLMs. Defaults to None.
                When self.produce is False, do_sample is set to True by default.
            hidden_state_skip_layer (int): Number of hidden states to hidden_state_skip_layer. 0 means the last layer.
                If None, self.output_key will be used. Defaults to None.
            return_texts (bool): Whether to return the decoded texts. Defaults to False.
        """
        device = self.model.device if device is None else device
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(
            hidden_state_skip_layer, self.hidden_state_skip_layer
        )
        do_sample = use_default(do_sample, not self.reproduce)

        batch_encoding = batch_encoding.to(device)
        attention_mask = (
            batch_encoding["attention_mask"] if use_attention_mask else None
        )
        deepstack_hidden_state_map = {}
        deepstack_handles = []
        needs_full_hidden_states = output_hidden_states or hidden_state_skip_layer is not None
        if deepstack is not None and len(deepstack) > 0 and not needs_full_hidden_states:
            language_model = getattr(getattr(self.model, "model", None), "language_model", None)
            layers = getattr(language_model, "layers", None)
            if layers is None or any(iddx <= 0 or iddx >= len(layers) for iddx in deepstack):
                needs_full_hidden_states = True
            else:
                def make_deepstack_hook(stack_idx):
                    def capture_hidden_state(module, inputs):
                        deepstack_hidden_state_map[stack_idx] = inputs[0]
                    return capture_hidden_state

                for iddx in deepstack:
                    deepstack_handles.append(
                        layers[iddx].register_forward_pre_hook(make_deepstack_hook(iddx))
                    )
        elif deepstack is not None and len(deepstack) > 0:
            needs_full_hidden_states = True

        try:
            outputs = self.model.model(
                **batch_encoding,
                output_hidden_states=needs_full_hidden_states,
            )
        finally:
            for handle in deepstack_handles:
                handle.remove()
        if hidden_state_skip_layer is not None:
            last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
            # Real last hidden state already has layer norm applied. So here we only apply it
            # for intermediate layers.
            if hidden_state_skip_layer > 0 and self.apply_final_norm:
                last_hidden_state = self.model.final_layer_norm(last_hidden_state)
        else:
            last_hidden_state = outputs[self.output_key]

        # Remove hidden states of instruction tokens, only keep prompt tokens.
        input_tensor = batch_encoding["input_ids"]
        keep_leading_modality_token = self._has_leading_modality_token(batch_encoding)
        if self.use_template:
            if crop_start is None:
                if data_type == "image":
                    crop_start = self.prompt_template.get("crop_start", 0)
                elif data_type == "video":
                    crop_start = self.prompt_template_video.get("crop_start", 0)
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")
            if crop_start > 0:
                sequence_crop_start = (
                    crop_start + 1 if keep_leading_modality_token else crop_start
                )
                if keep_leading_modality_token:
                    last_hidden_state = torch.cat(
                        [
                            last_hidden_state[:, :1],
                            last_hidden_state[:, sequence_crop_start:],
                        ],
                        dim=1,
                    )
                    attention_mask = (
                        torch.cat(
                            [
                                attention_mask[:, :1],
                                attention_mask[:, sequence_crop_start:],
                            ],
                            dim=1,
                        )
                        if use_attention_mask else None
                    )
                    input_tensor = torch.cat(
                        [
                            input_tensor[:, :1],
                            input_tensor[:, sequence_crop_start:],
                        ],
                        dim=1,
                    )
                else:
                    last_hidden_state = last_hidden_state[:, crop_start:]
                    attention_mask = (
                        attention_mask[:, crop_start:] if use_attention_mask else None
                    )
                    input_tensor = input_tensor[:, crop_start:]
        if crop_start is None:
            crop_start = 0
        
        if deepstack is not None and len(deepstack)>0:
            if needs_full_hidden_states:
                deepstack_hidden_states = [outputs.hidden_states[iddx] for iddx in deepstack]
            else:
                missing_deepstack = [
                    iddx for iddx in deepstack
                    if iddx not in deepstack_hidden_state_map
                ]
                if missing_deepstack:
                    raise RuntimeError(f"Failed to capture deepstack hidden states: {missing_deepstack}")
                deepstack_hidden_states = [deepstack_hidden_state_map[iddx] for iddx in deepstack]
            deepstack_hidden_states = torch.stack(deepstack_hidden_states, dim=0)
            if crop_start > 0 and keep_leading_modality_token:
                sequence_crop_start = crop_start + 1
                deepstack_hidden_states = torch.cat(
                    [
                        deepstack_hidden_states[:, :, :1, :],
                        deepstack_hidden_states[:, :, sequence_crop_start:, :],
                    ],
                    dim=2,
                )
            else:
                deepstack_hidden_states = deepstack_hidden_states[:, :, crop_start:, :]
            deepstack_hidden_states = deepstack_hidden_states * attention_mask.unsqueeze(0).unsqueeze(-1)
        else:
            deepstack_hidden_states = None
        
        
        if setclip and 'pixel_values' in batch_encoding:
            B, L = input_tensor.shape
            mask = (input_tensor == 151653)

            index_range = torch.arange(L, device=input_tensor.device).expand(B, L)
            masked_indices = torch.where(mask, index_range, torch.tensor(-1, device=input_tensor.device))
            last_indices = masked_indices.max(dim=1).values  # (B,)

            all_last_hidden_state = []
            all_attention_mask = []
            all_deepstack_hidden_states = []
            for kkk in range(B):
                idx = last_indices[kkk].item()
                
                if idx != -1 and idx < L - 1:
                    cur_last_hidden_state = last_hidden_state[kkk, idx + 1:]
                    cur_attention_mask = attention_mask[kkk, idx+1:]
                    if deepstack_hidden_states is not None:
                        cur_deepstack_hidden_states = deepstack_hidden_states[:, kkk, idx+1:]
                    if keep_leading_modality_token:
                        cur_last_hidden_state = torch.cat(
                            [last_hidden_state[kkk, :1], cur_last_hidden_state],
                            dim=0,
                        )
                        cur_attention_mask = torch.cat(
                            [attention_mask[kkk, :1], cur_attention_mask],
                            dim=0,
                        )
                        if deepstack_hidden_states is not None:
                            cur_deepstack_hidden_states = torch.cat(
                                [
                                    deepstack_hidden_states[:, kkk, :1],
                                    cur_deepstack_hidden_states,
                                ],
                                dim=1,
                            )
                else:
                    cur_last_hidden_state = last_hidden_state[kkk]
                    cur_attention_mask = attention_mask[kkk]
                    if deepstack_hidden_states is not None:
                        cur_deepstack_hidden_states = deepstack_hidden_states[:, kkk]
                    
                    
                all_last_hidden_state.append(cur_last_hidden_state)
                all_attention_mask.append(cur_attention_mask)
                if deepstack_hidden_states is not None:
                    all_deepstack_hidden_states.append(cur_deepstack_hidden_states)
                
            
            max_len = max([t.size(0) for t in all_last_hidden_state])
            myhidden_size = last_hidden_state.size(-1)
            mydevice = last_hidden_state.device

            padded_last_hidden_state = torch.zeros((B, max_len, myhidden_size), device=mydevice, dtype=last_hidden_state.dtype)
            padded_attention_mask = torch.zeros((B, max_len), device=mydevice, dtype=attention_mask.dtype)
            
            padded_deepstack = None
            if deepstack_hidden_states is not None:
                # (Layers, B, L, H)
                num_layers = deepstack_hidden_states.size(0)
                padded_deepstack = torch.zeros((num_layers, B, max_len, myhidden_size), device=mydevice, dtype=deepstack_hidden_states.dtype)

            for i in range(B):
                curr_len = all_last_hidden_state[i].size(0)

                padded_last_hidden_state[i, :curr_len, :] = all_last_hidden_state[i]
                padded_attention_mask[i, :curr_len] = all_attention_mask[i]
                
                if deepstack_hidden_states is not None:
                    padded_deepstack[:, i, :curr_len, :] = all_deepstack_hidden_states[i]

            last_hidden_state = padded_last_hidden_state
            attention_mask = padded_attention_mask
            deepstack_hidden_states = padded_deepstack

        if output_hidden_states:
            return TextEncoderModelOutput(
                last_hidden_state, attention_mask, outputs.hidden_states, deepstack_hidden_states=deepstack_hidden_states
            )
        return TextEncoderModelOutput(last_hidden_state, attention_mask, deepstack_hidden_states=deepstack_hidden_states)


    def forward(
        self,
        text,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=False,
        hidden_state_skip_layer=None,
        return_texts=False,
    ):
        batch_encoding = self.text2tokens(text, max_length=self.max_length)
        return self.encode(
            batch_encoding,
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            do_sample=do_sample,
            hidden_state_skip_layer=hidden_state_skip_layer,
            return_texts=return_texts,
        )

    def get_task_prompt(self, prompt_mode):
        # prompt_mode -> (system prompt, number of system-prompt tokens to crop)
        #   1 = i2i, 2 = ti_i2i, 3 = v2v, 4 = tiv2v, 5 = 3d23d
        if prompt_mode == 1:
            return system_prompt_1, 89
        if prompt_mode == 2:
            return system_prompt_2, 90
        if prompt_mode == 3:
            return system_prompt_3, 90
        if prompt_mode == 4:
            return system_prompt_4, 104
        if prompt_mode == 5:
            return system_prompt_5, 121
        
        return None, None

    def prepare_input(
        self,
        texts,
        imgs=None,
        videos=None,
        num_frames=2,
        prompt_mode=None,
        token_per_image=None,
        add_generation_prompt=True,
        modality_tokens=None,
    ):
        if token_per_image is None:
            token_per_image = 400 
        select_prompt = None
        select_crop_size = None
        if not prompt_mode:
            raise ValueError("prompt_mode is required for prepare_input to select the correct system prompt and crop size.")
        else:
            select_prompt, select_crop_size = self.get_task_prompt(prompt_mode)

        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = list(texts)
        modality_tokens = self._normalize_modality_tokens(
            modality_tokens,
            len(texts),
        )
        modality_prefix_length = int(any(token is not None for token in modality_tokens))

        messages = []
        if prompt_mode == 4:
            assert imgs is not None and videos is not None, \
                "prompt_mode=4 expects both imgs and videos"
            crop_size = select_crop_size if select_crop_size is not None else 104

            num_frames = len(videos[0])
            img_len = token_per_image * (num_frames // 2) + token_per_image + 32
            for txt, curimg, video in zip(texts, imgs, videos):
                if txt is None or txt == '':
                    txt = ' '
                # Exactly one reference image per sample (a bare image or a 1-element list).
                if isinstance(curimg, (list, tuple)):
                    if len(curimg) != 1:
                        raise ValueError(
                            f"prompt_mode=4 expects exactly one reference image per sample, got {len(curimg)}"
                        )
                    curimg = curimg[0]
                messages.append([
                    select_prompt if select_prompt is not None else system_prompt_4,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": 'This is the reference image:'},
                            {"type": "image", "image": curimg},
                            {"type": "text", "text": 'This is the input video:'},
                            {"type": "video", "video": video, "max_pixels": 560 * 560},
                            {"type": "text", "text": txt},
                        ]
                    }
                ])

        elif prompt_mode in (3, 5):
            assert videos is not None, f"prompt_mode={prompt_mode} expects videos"
            assert imgs is None, "imgs and videos cannot be both not None"
            crop_size = select_crop_size if select_crop_size is not None else 90

            num_frames = len(videos[0])
            img_len = token_per_image * (num_frames // 2)
            fallback_prompt = system_prompt_5 if prompt_mode == 5 else system_prompt_3
            for txt, video in zip(texts, videos):
                if txt is None or txt == '':
                    txt = ' '
                messages.append([
                    select_prompt if select_prompt is not None else fallback_prompt,
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video, "max_pixels": 560 * 560},
                            {"type": "text", "text": txt},
                        ]
                    }
                ])

        elif prompt_mode == 1:
            assert imgs is not None, "prompt_mode=1 expects imgs"
            img_len = token_per_image
            for txt, img in zip(texts, imgs):
                if txt is None or txt == '':
                    txt = ' '
                crop_size = select_crop_size if select_crop_size is not None else 89
                messages.append([select_prompt if select_prompt is not None else system_prompt_1,
                    {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img[0]},
                        {"type": "text", "text": txt},
                    ]
                }])

        elif prompt_mode == 2:
            assert imgs is not None, "prompt_mode=2 expects imgs"
            img_len = token_per_image
            for txt, img in zip(texts, imgs):
                if txt is None or txt == '':
                    txt = ' '
                crop_size = select_crop_size if select_crop_size is not None else 90
                messages.append([select_prompt if select_prompt is not None else system_prompt_2,
                    {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "This is the reference image:"},
                        {"type": "image", "image": img[-1]},
                        {"type": "text", "text": "These are the input images:"},
                        {"type": "image", "image": img[0]},
                        {"type": "text", "text": txt},
                    ]
                }])

        else:
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

        texts = []
        for msg, modality_token in zip(messages, modality_tokens):
            serialized_text = self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            texts.append(
                self._prepend_modality_token_to_serialized_text(
                    serialized_text,
                    modality_token,
                )
            )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length + crop_size + img_len + modality_prefix_length,
            padding="max_length",
        )
        
        
        return inputs, crop_size
