"""Part C (+ the game used in Part E): learner, answer key, guesser, masks, loss.

learner    = the driving encoder (agent checkpoint, sha256 740335c8...) + LoRA
answer key = a copy of the learner that never gets gradients; its LoRA follows
             the learner's by momentum (lora.momentum_update)
guesser    = the V-JEPA predictor from vitl_merge_3dataset_e50.pt (sha256 46491827...)

Masks: upstream's own _MaskGenerator (vjepa2/src/masks/multiseq_multiblock3d.py:80-240),
instantiated for one 16x32 slice (crop (256, 512), 2 frames, tubelet 2 -> duration 1).
Loss: upstream forward_target / forward_context / loss_fn (vjepa2/app/vjepa/train.py:413-435):
answer-key output, layer-normed, gathered at the hidden positions, L1 (loss_exp 1).
"""

from __future__ import annotations

import copy
import functools
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange

import common as C

C.setup()

import vjepa2.src.models.vision_transformer as vit  # noqa: E402
from vjepa2.src.masks.multiseq_multiblock3d import _MaskGenerator  # noqa: E402
from vjepa2.src.masks.utils import apply_masks  # noqa: E402
from vjepa2.src.models.predictor import VisionTransformerPredictor  # noqa: E402

import lora as L  # noqa: E402

GRID_T, GRID_H, GRID_W = 1, 16, 32
ENC_PREFIX_IN_CKPT = "agent._pad_model._backbone.img_backbone."

# PLACEHOLDER defaults (pretrain-256px-16f.yaml:41-67), recorded for Sid's decision.
MASK_CFGS = [
    {"num_blocks": 8, "spatial_scale": (0.15, 0.15), "aspect_ratio": (0.75, 1.5)},
    {"num_blocks": 2, "spatial_scale": (0.7, 0.7), "aspect_ratio": (0.75, 1.5)},
]
# The pretraining run had 10 mask tokens (one per mask config x frame count);
# PredictorMultiSeqWrapper (vjepa2/src/utils/wrappers.py:35-41) picks the token
# by sequence index, which is 0 for our single input length.
MASK_TOKEN_INDEX = 0


# --------------------------------------------------------------------------- #
# Encoder (learner / answer key)
# --------------------------------------------------------------------------- #


def encoder_kwargs() -> dict:
    """Exactly what simple_image_encoder.py:24-35 feeds init_module."""
    with open(C.DEVKIT / "vjepa2/configs/eval/vitl/in1k.yaml") as fh:
        params = yaml.load(fh, Loader=yaml.FullLoader)
    kw = params["model_kwargs"]["pretrain_kwargs"]["encoder"]
    kw["model_name"] = "vit_large"
    return kw


def build_vit() -> nn.Module:
    """Same constructor call as vit_encoder.py:57-61 (input_size is swallowed by
    VisionTransformer **kwargs there too; non-square grids come from
    handle_nonsquare_inputs, vision_transformer.py:176-181)."""
    kw = dict(encoder_kwargs())
    name = kw["model_name"]
    return vit.__dict__[name](input_size=(256, 512), num_frames=2, **kw)


def agent_encoder_state(ckpt_state: dict) -> dict:
    return {k[len(ENC_PREFIX_IN_CKPT):]: v for k, v in ckpt_state.items() if k.startswith(ENC_PREFIX_IN_CKPT)}


def e50_state(e50: dict, key: str) -> dict:
    sd = {k.replace("module.", ""): v for k, v in e50[key].items()}
    return {k.replace("backbone.", ""): v for k, v in sd.items()}


def state_hash(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def relative_diff(a: dict, b: dict) -> dict:
    """||a-b|| / ||b||, over all shared tensors and the worst single tensor."""
    num = den = 0.0
    worst = 0.0
    for k in a:
        x, y = a[k].double(), b[k].double()
        d, n = float((x - y).norm()), float(y.norm())
        num += d * d
        den += n * n
        if n > 0:
            worst = max(worst, d / n)
    return {"global": (num ** 0.5) / (den ** 0.5), "worst_tensor": worst}


def encode(encoder: nn.Module, pairs: torch.Tensor, masks=None) -> torch.Tensor:
    """pairs (B, 2, 3, 256, 512) -> tokens.  The rearrange is the one at
    simple_image_encoder.py:42-46, minus GridMask."""
    B, N, Cc, H, W = pairs.shape
    x = rearrange(pairs.reshape(B * N, Cc, H, W), "(B N) C H W -> B C N H W", B=B)
    return encoder(x, masks=masks)


def make_learner_and_key(plain_encoder: nn.Module, rank: int = L.RANK, alpha: float = L.ALPHA):
    learner = copy.deepcopy(plain_encoder)
    L.wrap(learner, rank, alpha)
    key = copy.deepcopy(learner)
    for p in key.parameters():
        p.requires_grad_(False)
    return learner, key


# --------------------------------------------------------------------------- #
# Guesser
# --------------------------------------------------------------------------- #


class Guesser(VisionTransformerPredictor):
    """Upstream predictor with one change: the blocks are told the token grid.

    predictor.py:227-229 calls blk(x, mask=masks, attn_mask=None) without T /
    H_patches / W_patches, so RoPEAttention falls back to a square grid of side
    grid_size = img_size[0] // 16 (modules.py:291-321).  On our 16x32 grid that
    would read token 300 as frame 1, row 2, column 12 instead of frame 0, row 9,
    column 12.  The encoder already passes the grid (vision_transformer.py:200-203).
    Everything else is upstream predictor.py:166-246 verbatim (has_cls and the
    chop path are dropped: neither is used here)."""

    def forward(self, x, masks_x, masks_y, mask_index=1, has_cls=False):
        assert not has_cls and self.chop_last_n_tokens == 0
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]
        B = len(x) // len(masks_x)
        x = self.predictor_embed(x)
        _, N_ctxt, D = x.shape
        mask_index = mask_index % self.num_mask_tokens
        pred_tokens = self.mask_tokens[mask_index]
        pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
        pred_tokens = apply_masks(pred_tokens, masks_y)
        x = x.repeat(len(masks_x), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)
        masks_x = torch.cat(masks_x, dim=0)
        masks_y = torch.cat(masks_y, dim=0)
        masks = torch.cat([masks_x, masks_y], dim=1)
        argsort = torch.argsort(masks, dim=1)
        masks = torch.stack([masks[i, row] for i, row in enumerate(argsort)], dim=0)
        x = torch.stack([x[i, row, :] for i, row in enumerate(argsort)], dim=0)
        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, masks, None, T=GRID_T, H_patches=GRID_H, W_patches=GRID_W, use_reentrant=False
                )
            else:
                x = blk(x, mask=masks, attn_mask=None, T=GRID_T, H_patches=GRID_H, W_patches=GRID_W)
        x = self.predictor_norm(x)
        reverse_argsort = torch.argsort(argsort, dim=1)
        x = torch.stack([x[i, row, :] for i, row in enumerate(reverse_argsort)], dim=0)
        x = x[:, N_ctxt:]
        return self.predictor_proj(x)


def build_guesser(e50: dict) -> Guesser:
    """Settings from pretrain-256px-16f.yaml:69-78 and init_video_model
    (app/vjepa/utils.py:172-190); 10 mask tokens as stored in the checkpoint."""
    g = Guesser(
        img_size=(256, 512), patch_size=16, num_frames=2, tubelet_size=2,
        embed_dim=1024, predictor_embed_dim=384, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True, norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        uniform_power=True, use_mask_tokens=True, num_mask_tokens=10, zero_init_mask_tokens=True,
        use_rope=True, use_sdpa=True, use_activation_checkpointing=True,
    )
    sd = {k.replace("module.backbone.", ""): v for k, v in e50["predictor"].items()}
    g.load_state_dict(sd, strict=True)
    return g


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #


def mask_generators(cfgs=MASK_CFGS) -> list:
    return [
        _MaskGenerator(
            crop_size=(256, 512), num_frames=2, spatial_patch_size=(16, 16), temporal_patch_size=2,
            spatial_pred_mask_scale=c["spatial_scale"], temporal_pred_mask_scale=(1.0, 1.0),
            aspect_ratio=c["aspect_ratio"], npred=c["num_blocks"], max_context_frames_ratio=1.0,
            max_keep=None, full_complement=False, pred_full_complement=False, inv_block=False,
        )
        for c in cfgs
    ]


def sample_masks(gens, batch_size: int, device):
    """[(masks_enc (B, K_ctx), masks_pred (B, K_hid)) per mask config]."""
    out = []
    for g in gens:
        enc, pred = g(batch_size)
        if enc.ndim == 1:  # default_collate of a single 1-D tensor -> keep the batch dim
            enc, pred = enc[None], pred[None]
        out.append((enc.to(device), pred.to(device)))
    return out


# --------------------------------------------------------------------------- #
# The game
# --------------------------------------------------------------------------- #


def answer_key_targets(key: nn.Module, pairs: torch.Tensor) -> torch.Tensor:
    """train.py:413-417: no grad, layer-normed over the feature dim."""
    with torch.no_grad():
        h = encode(key, pairs)
        return F.layer_norm(h, (h.size(-1),))


def game_loss(learner, key, guesser, pairs, masks) -> tuple[torch.Tensor, list[float]]:
    """train.py:419-435 for one clip length and len(masks) mask configs."""
    h = answer_key_targets(key, pairs)
    per = []
    for enc_m, pred_m in masks:
        z = encode(learner, pairs, masks=[enc_m])
        z = guesser(z, [enc_m], [pred_m], mask_index=MASK_TOKEN_INDEX)
        hi = apply_masks(h, [pred_m], concat=False)[0]
        per.append(torch.mean(torch.abs(z - hi)))
    loss = sum(per) / len(per)
    return loss, [float(p) for p in per]
