# Modified by the UniVEdit authors from Tencent-Hunyuan/OmniWeaving (HunyuanVideo-1.5).
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from decord import VideoReader

def prepare_custom_video(vr, total_subset=161, nframes=8):
    if isinstance(vr, str):
        vr = VideoReader(vr)
    
    max_frames = min(len(vr), total_subset)
    sample_indices = np.linspace(0, max_frames - 1, nframes).astype(int)
    
    frames = vr.get_batch(sample_indices)
    if hasattr(frames, 'asnumpy'):
        frames = frames.asnumpy()  # decord NDArray
    else:
        frames = frames.numpy()  # PyTorch Tensor
    frame_list = [Image.fromarray(frame) for frame in frames]
    
    return frame_list

def get_semantic_images_np(video_path_list, nframes=8):
    """
    Extract first frame from each video in video_path_list and return batchified numpy array.
    
    Args:
        video_path_list: List of video paths
        
    Returns:
        first_images_np: numpy array of shape (B, H, W, C) with dtype uint8 (0-255)
            Returns None if video_path_list is None or empty
    """
    if video_path_list is None or len(video_path_list) == 0:
        return None
    
    first_frames = []
    sampled_frames = []
    for video_path in video_path_list:
        # Read first frame from video
        if isinstance(video_path, str):
            video_reader = VideoReader(video_path)
        else:
            assert isinstance(video_path, VideoReader), f"video_path must be either str or VideoReader, got {type(video_path)}"
            video_reader = video_path
        if len(video_reader) == 0:
            raise ValueError(f"Video {video_path} has no frames")
        
        sample_frame_list = prepare_custom_video(video_reader, nframes=nframes)
        sampled_frames.append(sample_frame_list)
        # Get first frame (index 0)
        first_frame = video_reader.get_batch([0])
        # Convert to numpy: shape is (1, H, W, C)
        if hasattr(first_frame, 'asnumpy'):
            first_frame_np = first_frame.asnumpy()  # decord NDArray
        else:
            first_frame_np = first_frame.numpy()  # PyTorch Tensor
        # Remove batch dimension: shape becomes (H, W, C)
        first_frame_np = first_frame_np[0]
        first_frames.append(first_frame_np)
        del video_reader
    
    # Stack all first frames into batch: shape becomes (B, H, W, C)
    first_images_np = np.stack(first_frames, axis=0)
    # Ensure uint8 format (0-255)
    first_images_np = first_images_np.astype(np.uint8)
    
    return first_images_np, sampled_frames

def get_semantic_images_np2(video_path_list, first_image_pil, vae, F, nframes=8, is_mode=True):
    """
    Extract first frame from each video in video_path_list and return batchified numpy array.
    Also encode first_image_pil list into cond_latents with values filled at F index=1.

    Args:
        video_path_list: List of video paths or VideoReader objects
        first_image_pil: List of PIL.Image.Image
        args: Arguments object containing vae_precision
        vae: VAE model
        F: Time dimension for cond_latents (must be >= 2)
        nframes: Number of frames to sample for each video (default: 8)
        is_mode: If True, use mode() for latent distribution, else use sample()

    Returns:
        first_images_np: numpy array of shape (B, H, W, C) with dtype uint8 (0-255)
        sampled_frames: list of sampled frame lists per video
        cond_latents: tensor of shape (B, C, F, H, W), values at index=1, others 0
        first_image_pil_np: numpy array of shape (B, H, W, C) with dtype uint8 (0-255)
    """
    first_images_np, sampled_frames = get_semantic_images_np(video_path_list, nframes=nframes)

    if first_image_pil is None or len(first_image_pil) == 0:
        raise ValueError("first_image_pil must be a non-empty list of PIL images")

    if F < 2:
        raise ValueError(f"F must be >= 2 to fill index=1, got {F}")

    first_image_pil_np = np.array([np.array(img, dtype=np.uint8) for img in first_image_pil])

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    first_images_pixel_values = [image_transform(image).unsqueeze(0) for image in first_image_pil]
    first_images_pixel_values = torch.cat(first_images_pixel_values).unsqueeze(2).to(vae.device)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        if is_mode:
            encoded_latents = vae.encode(first_images_pixel_values).latent_dist.mode()
        else:
            encoded_latents = vae.encode(first_images_pixel_values).latent_dist.sample()
        encoded_latents.mul_(vae.config.scaling_factor)

    B = encoded_latents.shape[0]
    C, H, W = encoded_latents.shape[1], encoded_latents.shape[3], encoded_latents.shape[4]
    cond_latents = torch.zeros(B, C, F, H, W, dtype=encoded_latents.dtype, device=encoded_latents.device)
    cond_latents[:, :, 1, :, :] = encoded_latents.squeeze(2)

    return first_images_np, sampled_frames, cond_latents, first_image_pil_np
