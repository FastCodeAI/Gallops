"""Frozen WA-JEPA encoder for Drive-JEPA (wa-jepa/CLAUDE.md, 16 Oct 2026).

AFARI-Research WA-JEPA (HF 15c0770e) online encoder: V-JEPA 2.1 ViT-L/16 built with WA-JEPA's own code
(third_party/vjepa2) and its own settings (the release's state.pt model config), `encoder.*` loaded strict.
Frozen: eval(), no_grad, bf16 autocast, no LoRA.

encode_wajepa(agent_input) -> float16 (4, 512, 1024): cameras b0, f0, l0, r0 (lidar2img's order), frames
cameras[-2] (t - 0.5 s) then cameras[-1] (t); each image img[28:-28] -> cv2.resize(img, (512, 256)) -> ToTensor ->
ImageNet normalisation; each camera one clip (3, 2, 256, 512); encoder(clip, training=False) -> (512, 1024).
"""
import os
import sys
from typing import List

import cv2
import numpy as np
import torch
from torchvision import transforms

# Site paths come from the environment (gallops' recipe sets both); the file as run defaulted to that machine's.
WAJEPA_CODE = os.environ.get("WAJEPA_CODE", "")
WAJEPA_WEIGHTS = os.environ.get("WAJEPA_WEIGHTS", "")
# WA-JEPA's ViT settings, from its state.pt model config (input_hw, patch_size, 4 history + 8 future frames,
# tubelet 2, interpolate_rope, modality_embedding, img_temporal_dim_size, pretrained_input_hw).
VIT_KWARGS = dict(
    img_size=(256, 512), patch_size=16, num_frames=12, tubelet_size=2, use_rope=True, use_sdpa=True, use_silu=False,
    wide_silu=True, interpolate_rope=True, modality_embedding=True, img_temporal_dim_size=1, pretrained_img_size=(256, 256),
    use_activation_checkpointing=False,
)
CAMERAS = ("cam_b0", "cam_f0", "cam_l0", "cam_r0")
_TO_TENSOR_NORM = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


def build_wajepa_encoder(device="cuda", weights=WAJEPA_WEIGHTS, code=WAJEPA_CODE):
    """WA-JEPA's online encoder, strict load, frozen. Returns (encoder, load message)."""
    if code not in sys.path:
        sys.path.insert(0, code)
    from third_party.vjepa2.models import vision_transformer as wa_vit

    enc = wa_vit.vit_large(**VIT_KWARGS)
    sd = torch.load(weights, map_location="cpu", mmap=True, weights_only=True)
    online = {k[len("encoder."):]: v.float() for k, v in sd.items() if k.startswith("encoder.")}
    msg = enc.load_state_dict(online, strict=True)
    enc.requires_grad_(False)
    return enc.to(device).eval(), msg


def preprocess_image(img: np.ndarray) -> torch.Tensor:
    """One RGB 1920x1080 camera image -> normalised (3, 256, 512)."""
    return _TO_TENSOR_NORM(cv2.resize(img[28:-28], (512, 256)))


def clips_from_agent_input(agent_input) -> torch.Tensor:
    """(4 cams, 3, 2 frames, 256, 512): b0 f0 l0 r0, frames [t - 0.5 s, t]."""
    cams = []
    for name in CAMERAS:
        frames = [preprocess_image(getattr(agent_input.cameras[k], name).image) for k in (-2, -1)]
        cams.append(torch.stack(frames, dim=1))
    return torch.stack(cams)


@torch.no_grad()
def encode_clips(encoder, clips: torch.Tensor) -> torch.Tensor:
    """clips (N, 3, 2, 256, 512) -> float16 (N, 512, 1024)."""
    dev = next(encoder.parameters()).device
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
        tok = encoder(clips.to(dev, non_blocking=True), training=False)
    return tok.float().half()


def encode_wajepa(agent_input, encoder) -> torch.Tensor:
    """float16 (4, 512, 1024) for one scene, on CPU."""
    return encode_clips(encoder, clips_from_agent_input(agent_input)).cpu()


def split_scenes(tokens: torch.Tensor, n: int) -> List[torch.Tensor]:
    """(n * 4, 512, 1024) -> n tensors (4, 512, 1024), each with its own storage (a view would pickle the whole batch)."""
    return [t.clone() for t in tokens.view(n, 4, 512, 1024)]
