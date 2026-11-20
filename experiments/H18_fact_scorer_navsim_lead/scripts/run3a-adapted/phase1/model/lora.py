"""LoRA on Q and V of WA-JEPA's fused qkv (phase1-perception/CLAUDE.md "Encoder": r32 on Q and V of every block,
everything else frozen; B starts at zero so the encoder starts identical to run 3A's). Merge and strip for the plain
encoder checkpoint. From the reference phase1_perception.py."""
import math

import torch
import torch.nn as nn


class LoRAQKV(nn.Module):
    """Fused qkv nn.Linear(dim, 3 * dim), output order [q | k | v]: low-rank updates on Q and V only."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        assert base.out_features % 3 == 0, "expected a fused qkv projection"
        self.base, self.dim, self.scale = base, base.out_features // 3, alpha / r
        d_in = base.in_features
        self.A_q = nn.Parameter(torch.empty(r, d_in))
        self.B_q = nn.Parameter(torch.zeros(self.dim, r))
        self.A_v = nn.Parameter(torch.empty(r, d_in))
        self.B_v = nn.Parameter(torch.zeros(self.dim, r))
        nn.init.kaiming_uniform_(self.A_q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A_v, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if self.merged:
            return out
        dq = (x @ self.A_q.t().to(x.dtype)) @ self.B_q.t().to(x.dtype) * self.scale
        dv = (x @ self.A_v.t().to(x.dtype)) @ self.B_v.t().to(x.dtype) * self.scale
        return out + torch.cat([dq, torch.zeros_like(dq), dv], dim=-1).to(out.dtype)

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            W = self.base.weight
            W[: self.dim] += (self.B_q.float() @ self.A_q.float() * self.scale).to(W.dtype)
            W[2 * self.dim:] += (self.B_v.float() @ self.A_v.float() * self.scale).to(W.dtype)
            self.merged = True


class LoRALinear(nn.Module):
    """For blocks with separate q / v projections (not used by WA-JEPA)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        return out if self.merged else out + (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.scale

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += (self.B.float() @ self.A.float() * self.scale).to(self.base.weight.dtype)
            self.merged = True


def inject_lora(encoder: nn.Module, r: int, alpha: float, targets=("qkv",)):
    """Freeze the encoder and wrap every target projection. Returns the trainable LoRA parameters."""
    for p in encoder.parameters():
        p.requires_grad_(False)
    n = 0
    for module in list(encoder.modules()):
        for name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear) or name not in targets:
                continue
            setattr(module, name, LoRAQKV(child, r, alpha) if name == "qkv" else LoRALinear(child, r, alpha))
            n += 1
    if n == 0:
        raise RuntimeError(f"no LoRA targets {targets} found")
    return [p for p in encoder.parameters() if p.requires_grad]


def lora_modules(module: nn.Module):
    return [(n, m) for n, m in module.named_modules() if isinstance(m, (LoRAQKV, LoRALinear))]


@torch.no_grad()
def merge_lora(encoder: nn.Module) -> int:
    n = 0
    for _, m in lora_modules(encoder):
        m.merge()
        n += 1
    return n


def strip_lora(module: nn.Module) -> None:
    """After merging: put the plain nn.Linear back (for a clean encoder checkpoint, saved to a new path)."""
    for name, child in list(module.named_children()):
        if isinstance(child, (LoRAQKV, LoRALinear)):
            assert child.merged, "merge before stripping"
            setattr(module, name, child.base)
        else:
            strip_lora(child)
