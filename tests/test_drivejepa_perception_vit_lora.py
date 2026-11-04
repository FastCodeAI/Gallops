"""H15 B3's late branch (torch; it skips in gallops' own interpreter).

The claims B3 leans on: the branch is the ViT's last four blocks and its norm, LoRA at zero leaves
it exactly the frozen tail, only LoRA trains, and its size is the plan's 1.05 M at ViT-L's width.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from gallops.drivejepa_perception import vit_lora  # noqa: E402


class _Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        w = (q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5).softmax(-1)
        return self.proj(w @ v)


class _MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _Block(nn.Module):
    """The vjepa2 Block's names and call signature, small."""

    def __init__(self, d):
        super().__init__()
        self.norm1, self.attn, self.norm2, self.mlp = nn.LayerNorm(d), _Attention(d), nn.LayerNorm(d), _MLP(d)
        self.seen: dict = {}

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        self.seen = {"T": T, "H_patches": H_patches, "W_patches": W_patches}
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _ViT(nn.Module):
    def __init__(self, d=32, depth=24):
        super().__init__()
        self.blocks = nn.ModuleList(_Block(d) for _ in range(depth))
        self.norm = nn.LayerNorm(d)


def test_the_branch_is_the_last_four_blocks_and_starts_as_the_frozen_tail():
    torch.manual_seed(0)
    vit = _ViT()
    x = torch.randn(2, 10, 32)
    expected = x
    for block in vit.blocks[vit_lora.BRANCH_FROM :]:
        expected = block(expected)
    expected = vit.norm(expected)
    frozen, lora = vit_lora.LateBranch(vit, lora=False), vit_lora.LateBranch(vit, lora=True)
    assert len(frozen.blocks) == 4
    assert torch.equal(frozen(x), expected)
    assert torch.equal(lora(x), expected)  # B at zero: exactly the frozen tail
    assert frozen.blocks[0].seen == vit_lora.GRID


def test_only_lora_trains_and_it_moves_the_output():
    torch.manual_seed(0)
    vit = _ViT()
    source = {k: v.clone() for k, v in vit.state_dict().items()}
    branch = vit_lora.LateBranch(vit, lora=True)
    trainable = [n for n, p in branch.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith((".a", ".b")) for n in trainable)
    x = torch.randn(2, 10, 32)
    before = branch(x).detach()
    optimiser = torch.optim.SGD([p for p in branch.parameters() if p.requires_grad], lr=0.1)
    branch(x).pow(2).mean().backward()
    optimiser.step()
    assert not torch.equal(branch(x), before)
    # The branch is a copy: training it leaves the ViT the planner reads exactly as it was.
    assert all(torch.equal(v, source[k]) for k, v in vit.state_dict().items())


def test_lora_is_the_planned_size_at_vit_l_width():
    # Rank 16 on qkv, proj, fc1, fc2: 16 x (d + 3d) + 16 x (d + d) + 16 x (d + 4d) + 16 x (4d + d) per block.
    d = 1024
    per_block = 16 * (4 * d) + 16 * (2 * d) + 16 * (5 * d) + 16 * (5 * d)
    assert 4 * per_block == 1_048_576
    vit = _ViT(d=64)
    branch = vit_lora.LateBranch(vit, lora=True)
    count = sum(p.numel() for p in branch.parameters() if p.requires_grad)
    assert count == 4 * (16 * (4 * 64) + 16 * (2 * 64) + 16 * (5 * 64) + 16 * (5 * 64))
