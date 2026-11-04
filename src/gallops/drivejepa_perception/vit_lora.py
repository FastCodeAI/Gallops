"""B3: LoRA on the ViT's last four blocks, read by the side-car only (B3.1) or by everything (B3.2).

The release's ViT has 24 blocks, and every B3 variant keeps blocks 1-20 frozen. Their output for a
row is therefore a constant: `tokens.py --block20` caches it once, in fp16, for a seeded train
subset and the select and report logs. Training runs only blocks 21-24 and the final norm from that
cache, so no image is decoded and the first 20 blocks never run. The pass holds the cache to the
full forward's ViT output before writing a row (`frozen branch on the cache == vit_out`).

`LateBranch` copies blocks 21-24 and the norm. With `lora`, each block's qkv, proj, fc1 and fc2
carries rank-16 LoRA, B at zero, so the branch starts as the frozen ViT: 4 x 262,144 = 1.05 M
trained parameters. Without it, the branch is the frozen ViT's tail (B3.1's control).
"""

from __future__ import annotations

import copy

import torch.nn as nn

from gallops.drivejepa_perception.scorers import LoRALinear

#: The first block that runs from the cache; blocks[:BRANCH_FROM] are cached as their output.
BRANCH_FROM = 20
#: The RoPE grid of the release's input: two frames at tubelet 2, 256 x 512 at patch 16. These are
#: the values VisionTransformer.forward derives for it (handle_nonsquare_inputs is True).
GRID = {"T": 1, "H_patches": 16, "W_patches": 32}
LORA_TARGETS = (("attn", "qkv"), ("attn", "proj"), ("mlp", "fc1"), ("mlp", "fc2"))


class LateBranch(nn.Module):
    """Blocks BRANCH_FROM.. and the norm of the release's ViT, optionally with LoRA."""

    def __init__(self, vit: nn.Module, lora: bool, rank: int = 16):
        super().__init__()
        self.blocks = copy.deepcopy(nn.ModuleList(vit.blocks[BRANCH_FROM:]))
        self.norm = copy.deepcopy(vit.norm)
        for p in self.parameters():
            p.requires_grad_(False)
        if lora:
            for block in self.blocks:
                for parent, name in LORA_TARGETS:
                    owner = getattr(block, parent)
                    setattr(owner, name, LoRALinear(getattr(owner, name), rank))

    def forward(self, x):
        for block in self.blocks:
            x = block(x, mask=None, attn_mask=None, **GRID)
        return self.norm(x)
