"""Part D: hand-written LoRA for the Drive-JEPA ViT-L encoder.

    y = W x + b + scale * B(A(x)),   scale = alpha / rank

A is random (Kaiming-uniform, the usual LoRA choice), B starts at zero, so a
freshly wrapped encoder computes exactly what the unwrapped one does.  Base
weights are frozen.  No dropout.

Targets: blocks.N.attn.qkv, blocks.N.attn.proj, blocks.N.mlp.fc1, blocks.N.mlp.fc2
for N = 0..23 (vjepa2/src/models/utils/modules.py:67-81 MLP fc1/fc2, :261-282
RoPEAttention qkv/proj) -- 96 layers on ViT-L.

Each check_* function returns (ok, detail) so the same check runs on the real
encoder (gpu_checks.py) and on planted mutations (tests/test_lora.py).
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# PLACEHOLDER defaults -- open decision for Sid before stage 2.
RANK = 16
ALPHA = 16
TARGET_RE = re.compile(r"^blocks\.\d+\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)$")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = RANK, alpha: float = ALPHA):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * F.linear(F.linear(x, self.lora_A), self.lora_B)

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + self.scale * (self.lora_B @ self.lora_A)

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}"


def _set_child(root: nn.Module, dotted: str, module: nn.Module) -> None:
    parent_name, _, child = dotted.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, module)


def wrap(encoder: nn.Module, rank: int = RANK, alpha: float = ALPHA, target_re=TARGET_RE,
         layer_cls=LoRALinear) -> list[str]:
    """Freeze every encoder parameter, then wrap each target Linear.  Returns the
    wrapped names.  Only lora_A / lora_B require grad afterwards.  layer_cls
    exists so the tests can plant a broken layer."""
    for p in encoder.parameters():
        p.requires_grad_(False)
    names = [n for n, m in encoder.named_modules() if isinstance(m, nn.Linear) and target_re.match(n)]
    for n in names:
        _set_child(encoder, n, layer_cls(encoder.get_submodule(n), rank, alpha))
    return names


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    out = []
    for m in lora_modules(model).values():
        out += [m.lora_A, m.lora_B]
    return out


def set_lora_trainable(model: nn.Module, flag: bool) -> None:
    for p in lora_parameters(model):
        p.requires_grad_(flag)


@torch.no_grad()
def momentum_update(key: nn.Module, learner: nn.Module, m: float) -> None:
    """key <- m * key + (1 - m) * learner, on LoRA tensors only (upstream
    vjepa2/app/vjepa/train.py:456-465 does the same on all encoder tensors;
    here the base weights are frozen and identical in both, so only LoRA moves)."""
    pk = lora_parameters(key)
    pq = lora_parameters(learner)
    assert len(pk) == len(pq) and len(pk) > 0
    torch._foreach_mul_(pk, m)
    torch._foreach_add_(pk, pq, alpha=1 - m)


def count(model: nn.Module) -> dict:
    per_type: dict[str, int] = {}
    for n, mod in lora_modules(model).items():
        kind = n.split(".", 2)[-1]
        per_type[kind] = per_type.get(kind, 0) + mod.lora_A.numel() + mod.lora_B.numel()
    return {"layers": len(lora_modules(model)), "total": sum(per_type.values()), "per_type": per_type}


# --------------------------------------------------------------------------- #
# Checks (T1..T5); each returns (ok, detail)
# --------------------------------------------------------------------------- #


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same shape, dtype and the very same bytes (so -0.0 vs 0.0 counts as a difference)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a, b = a.detach().contiguous(), b.detach().contiguous()
    view = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16,
            torch.float64: torch.int64}.get(a.dtype)
    if view is None:
        return bool(torch.equal(a, b))
    return bool(torch.equal(a.view(view), b.view(view)))


def check_t1_zero_identity(wrapped: nn.Module, plain: nn.Module, x: torch.Tensor):
    with torch.no_grad():
        ya, yb = wrapped(x), plain(x)
    ok = bit_equal(ya, yb)
    return ok, {"bit_equal": ok, "max_abs_diff": float((ya.float() - yb.float()).abs().max())}


def check_t3_trainable(agent_like: nn.Module, encoder_prefix: str, expected_wrapped: int):
    """Only lora_A / lora_B under the encoder may require grad, and exactly the
    expected number of LoRA layers may exist anywhere in the model."""
    trainable = [n for n, p in agent_like.named_parameters() if p.requires_grad]
    bad = [n for n in trainable if not (n.startswith(encoder_prefix) and re.search(r"\.lora_[AB]$", n))]
    wrapped = [n for n in lora_modules(agent_like)]
    outside = [n for n in wrapped if not (n.startswith(encoder_prefix)
                                          and TARGET_RE.match(n[len(encoder_prefix):].lstrip(".")))]
    ok = not bad and not outside and len(wrapped) == expected_wrapped and len(trainable) == 2 * expected_wrapped
    return ok, {"trainable_tensors": len(trainable), "non_lora_trainable": bad[:5],
                "lora_layers": len(wrapped), "lora_layers_off_target": outside[:5]}


def check_t4_merge(layer: LoRALinear, x: torch.Tensor, rtol: float, atol: float, merged_fn=None):
    merged_fn = merged_fn or (lambda m: m.merged_weight())
    with torch.no_grad():
        y_unmerged = layer(x)
        y_merged = F.linear(x, merged_fn(layer), layer.base.bias)
    diff = float((y_unmerged - y_merged).abs().max())
    ok = bool(torch.allclose(y_unmerged, y_merged, rtol=rtol, atol=atol))
    return ok, {"max_abs_diff": diff, "rtol": rtol, "atol": atol, "dtype": str(x.dtype)}


def check_t5_momentum(key: nn.Module, learner: nn.Module, m: float, update_fn=momentum_update):
    before_key = [p.detach().clone() for p in lora_parameters(key)]
    learner_now = [p.detach().clone() for p in lora_parameters(learner)]
    update_fn(key, learner, m)
    after = lora_parameters(key)
    worst = max(float((a - (m * k + (1 - m) * q)).abs().max()) for a, k, q in zip(after, before_key, learner_now))
    base_same = all(
        bit_equal(pk, pq)
        for (nk, pk), (nq, pq) in zip(key.named_parameters(), learner.named_parameters())
        if "lora_" not in nk
    )
    ok = worst <= 1e-7 and base_same
    return ok, {"max_abs_err_vs_rule": worst, "base_weights_identical": base_same, "m": m}
