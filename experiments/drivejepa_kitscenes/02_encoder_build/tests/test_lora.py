"""Part D unit tests on a small ViT of the same class (CPU, seconds).

Every check also runs against a planted mutation and must FAIL there (T6):
  non-zero B init -> T1, LoRA on a frozen non-target layer -> T3,
  sign error in merge -> T4, momentum update reversed -> T5.
The same check functions run on the real ViT-L / agent in src/gpu_checks.py.

    ./run.sh -m pytest tests -q
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

import common as C

C.setup()

import vjepa2.src.models.vision_transformer as vit  # noqa: E402

import lora as L  # noqa: E402


def small_vit() -> nn.Module:
    torch.manual_seed(0)
    return vit.VisionTransformer(img_size=(32, 64), patch_size=16, num_frames=2, tubelet_size=2,
                                 embed_dim=64, depth=2, num_heads=4, use_rope=True, uniform_power=True,
                                 use_sdpa=True)


def clip(b=2) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(b, 3, 2, 32, 64)


class AgentLike(nn.Module):
    """encoder + a frozen projector + generator/scorer stand-ins, like _pad_model."""

    def __init__(self, enc: nn.Module):
        super().__init__()
        self.enc = enc
        self.projector = nn.Linear(64, 16)
        self.generator = nn.Linear(16, 16)
        self.scorer = nn.Linear(16, 6)
        for p in self.parameters():
            p.requires_grad_(False)


class BadInitLoRA(L.LoRALinear):
    def __init__(self, base, rank, alpha):
        super().__init__(base, rank, alpha)
        nn.init.normal_(self.lora_B, std=0.02)


def _randomise(model, seed=5, std=0.05):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in L.lora_parameters(model):
            p.copy_(torch.randn(p.shape, generator=g) * std)


# --------------------------------------------------------------------------- T1


def test_t1_zero_lora_is_identity():
    plain = small_vit().eval()
    wrapped = copy.deepcopy(plain)
    assert len(L.wrap(wrapped, rank=4, alpha=4)) == 8
    ok, info = L.check_t1_zero_identity(wrapped.eval(), plain, clip())
    assert ok, info


def test_t6_nonzero_B_init_breaks_t1():
    plain = small_vit().eval()
    wrapped = copy.deepcopy(plain)
    L.wrap(wrapped, rank=4, alpha=4, layer_cls=BadInitLoRA)
    ok, info = L.check_t1_zero_identity(wrapped.eval(), plain, clip())
    assert not ok, info


# --------------------------------------------------------------------------- T3


def test_t3_only_lora_trainable():
    agent = AgentLike(small_vit())
    L.wrap(agent.enc, rank=4, alpha=4)
    ok, info = L.check_t3_trainable(agent, "enc", expected_wrapped=8)
    assert ok, info


def test_t6_lora_on_frozen_non_target_layer_breaks_t3():
    agent = AgentLike(small_vit())
    L.wrap(agent.enc, rank=4, alpha=4)
    agent.projector = L.LoRALinear(agent.projector, 4, 4)  # planted: projector must stay untouched
    ok, info = L.check_t3_trainable(agent, "enc", expected_wrapped=8)
    assert not ok, info


def test_t6_lora_on_non_target_layer_inside_encoder_breaks_t3():
    agent = AgentLike(small_vit())
    L.wrap(agent.enc, rank=4, alpha=4)
    # planted: an MLP layer outside the target list (blocks.N.mlp.fc1 renamed target set)
    import re

    L.wrap(agent.enc.blocks[0].mlp, rank=4, alpha=4, target_re=re.compile(r"^fc1\.base$"))
    ok, info = L.check_t3_trainable(agent, "enc", expected_wrapped=8)
    assert not ok, info


# --------------------------------------------------------------------------- T4

MERGE_RTOL, MERGE_ATOL = 1e-5, 1e-5  # float32


def test_t4_merge_matches_unmerged():
    enc = small_vit()
    L.wrap(enc, rank=4, alpha=8)
    _randomise(enc)
    layer = enc.blocks[1].mlp.fc2
    x = torch.randn(3, 7, layer.base.in_features)
    ok, info = L.check_t4_merge(layer, x, MERGE_RTOL, MERGE_ATOL)
    assert ok, info
    # and on the whole encoder: swapping every layer for its merged Linear
    merged = copy.deepcopy(enc)
    for n, m in list(L.lora_modules(merged).items()):
        lin = nn.Linear(m.base.in_features, m.base.out_features)
        with torch.no_grad():
            lin.weight.copy_(m.merged_weight())
            lin.bias.copy_(m.base.bias)
        L._set_child(merged, n, lin)
    with torch.no_grad():
        a, b = enc.eval()(clip()), merged.eval()(clip())
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-4), float((a - b).abs().max())


def test_t6_merge_sign_error_breaks_t4():
    enc = small_vit()
    L.wrap(enc, rank=4, alpha=8)
    _randomise(enc)
    layer = enc.blocks[1].mlp.fc2
    x = torch.randn(3, 7, layer.base.in_features)
    wrong = lambda m: m.base.weight - m.scale * (m.lora_B @ m.lora_A)  # noqa: E731
    ok, info = L.check_t4_merge(layer, x, MERGE_RTOL, MERGE_ATOL, merged_fn=wrong)
    assert not ok, info


# --------------------------------------------------------------------------- T5


def _pair():
    learner = small_vit()
    L.wrap(learner, rank=4, alpha=4)
    key = copy.deepcopy(learner)
    for p in key.parameters():
        p.requires_grad_(False)
    _randomise(learner, seed=7)
    return key, learner


def test_t5_momentum_rule():
    key, learner = _pair()
    ok, info = L.check_t5_momentum(key, learner, m=0.99)
    assert ok, info


@torch.no_grad()
def _reversed_update(key, learner, m):
    pk, pq = L.lora_parameters(key), L.lora_parameters(learner)
    torch._foreach_mul_(pk, 1 - m)
    torch._foreach_add_(pk, pq, alpha=m)


def test_t6_reversed_momentum_breaks_t5():
    key, learner = _pair()
    ok, info = L.check_t5_momentum(key, learner, m=0.99, update_fn=_reversed_update)
    assert not ok, info


def test_lora_count_formula():
    enc = small_vit()
    L.wrap(enc, rank=4, alpha=4)
    c = L.count(enc)
    # per block: qkv 4*(64+192) + proj 4*(64+64) + fc1 4*(64+256) + fc2 4*(256+64)
    assert c["total"] == 2 * 4 * ((64 + 192) + (64 + 64) + (64 + 256) + (256 + 64))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
