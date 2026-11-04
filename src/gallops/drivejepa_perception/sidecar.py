"""The side-car (phase A): agents and each one's next second, from the frozen ViT's tokens.

Ported from H04's AgentDecoder and AgentLoss (Navvrat; branch drivejepa-wajepa/run3a at e5ca907,
recipes/drivejepa_wajepa/overlay/agent_decoder/.../agent_decoder.py), which read WA-JEPA's four
cameras. What changed here, and why:
- **one camera:** the release's CAM_F0 tokens, (512, d_in), with that camera's projection;
- **no input projection when the tap is already 256-d** (the post-projector tokens); a 1024-d tap
  (A0) keeps H04's Linear(1024, 256);
- **reference points start inside the front camera's view** (x 2-50 m, |y| < 30 m), not in a ring;
- **a path head on every layer:** each agent's offset from its own centre at t + 0.5 s and
  t + 1.0 s, trained with L1 where the label's track lasted (`path_ok`).

The targets and the camera geometry (`TO_512`) are in targets.py; a token's ray is the inverse of
that projection through the token's pixel centre.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from gallops.drivejepa_perception.targets import H_TOK, NONE, NUM_CLASSES, PATH_STEPS, QUERIES, W_TOK


def token_rays(projection: torch.Tensor):
    """(B, 4, 4) ego -> token-image projection -> camera centre (B, 3), unit rays (B, 512, 3).

    H04's `token_rays` (global_look.py, run 2), for one camera: tokens row-major over 16 x 32.
    """
    p = projection.float()
    m_inv = torch.linalg.inv(p[:, :3, :3])
    centre = -(m_inv @ p[:, :3, 3:4]).squeeze(-1)
    r, c = torch.meshgrid(
        torch.arange(H_TOK, device=p.device), torch.arange(W_TOK, device=p.device), indexing="ij"
    )
    pix = torch.stack(
        [16.0 * c.flatten() + 8.0, 16.0 * r.flatten() + 8.0, torch.ones(H_TOK * W_TOK, device=p.device)], -1
    )
    rays = torch.einsum("bij,pj->bpi", m_inv, pix)
    return centre, F.normalize(rays, dim=-1)


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class MHA(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.h, self.dropout = heads, dropout
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, q, k, v):
        b, nq, d = q.shape

        def split(x):
            return x.view(b, -1, self.h, d // self.h).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            split(self.q(q)),
            split(self.k(k)),
            split(self.v(v)),
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o(out.transpose(1, 2).reshape(b, nq, d))


class DecoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.ln_sa, self.ln_ca, self.ln_ff = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa, self.ca = MHA(d, heads, dropout), MHA(d, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)
        self.ref_mlp = mlp(d, d, 3)
        nn.init.zeros_(self.ref_mlp[-1].weight)
        nn.init.zeros_(self.ref_mlp[-1].bias)
        self.cls_head = nn.Linear(d, NUM_CLASSES + 1)
        nn.init.constant_(self.cls_head.bias, 0.0)
        with torch.no_grad():
            self.cls_head.bias[NONE] = 2.0  # start near p(none) ~ 0.7, as H04 did
        self.box_head = mlp(d, d, 5)  # log l, log w, log h, sin yaw, cos yaw
        self.vel_head = mlp(d, d, 2)
        self.path_head = mlp(d, d, 2 * PATH_STEPS)  # offsets from the agent's own centre

    def forward(self, a, pos, k, v):
        h = self.ln_sa(a)
        a = a + self.drop(self.sa(h + pos, h + pos, h))
        a = a + self.drop(self.ca(self.ln_ca(a) + pos, k, v))
        return a + self.drop(self.ffn(self.ln_ff(a)))


class SideCar(nn.Module):
    """Agents and their next second from one camera's frozen tokens; see the module docstring."""

    def __init__(
        self,
        d_in: int = 256,
        d: int = 256,
        heads: int = 8,
        layers: int = 3,
        queries: int = QUERIES,
        ffn: int = 1024,
        dropout: float = 0.1,
        n_depths: int = 16,
        min_depth: float = 1.0,
        max_depth: float = 60.0,
    ):
        super().__init__()
        self.proj = nn.Linear(d_in, d) if d_in != d else nn.Identity()
        self.ln_tok = nn.LayerNorm(d)
        self.ray_mlp = mlp(3 * n_depths, 256, d)
        depths = torch.logspace(math.log10(min_depth), math.log10(max_depth), n_depths)
        self.register_buffer("depths", depths, persistent=False)
        self.register_buffer("scale", torch.tensor([1 / 64, 1 / 64, 1 / 8]), persistent=False)
        self.query = nn.Parameter(torch.randn(queries, d) * 0.02)
        ref = torch.zeros(queries, 3)
        ref[:, 0] = torch.rand(queries) * 48.0 + 2.0
        ref[:, 1] = torch.rand(queries) * 60.0 - 30.0
        self.ref = nn.Parameter(ref)
        self.pos_mlp = mlp(3, 256, d)
        self.layers = nn.ModuleList([DecoderLayer(d, heads, ffn, dropout) for _ in range(layers)])

    def ray_pos(self, projection):
        with torch.autocast(device_type="cuda", enabled=False):
            centre, rays = token_rays(projection)
            points = centre[:, None, None, :] + self.depths[None, None, :, None] * rays[:, :, None, :]
            points = (points * self.scale).flatten(2)  # B, 512, 48
        return self.ray_mlp(points)

    def forward(self, tokens, projection):
        """tokens (B, 512, d_in), projection (B, 4, 4) -> per layer: logits, box (8), vel, path."""
        b = tokens.shape[0]
        x = tokens if torch.is_autocast_enabled() else tokens.to(self.ln_tok.weight.dtype)
        mem = self.ln_tok(self.proj(x))
        k = mem + self.ray_pos(projection).to(mem.dtype)
        a = self.query[None].expand(b, -1, -1)
        ref = self.ref[None].expand(b, -1, -1)
        outs = []
        for layer in self.layers:
            pos = self.pos_mlp(ref * self.scale)
            a = layer(a, pos, k, mem)
            ref_new = ref + layer.ref_mlp(a).float()
            outs.append(
                {
                    "logits": layer.cls_head(a).float(),
                    "box": torch.cat([ref_new, layer.box_head(a).float()], -1),
                    "vel": layer.vel_head(a).float(),
                    "path": layer.path_head(a).float().view(b, -1, PATH_STEPS, 2),
                }
            )
            ref = ref_new.detach()
        return outs


def box_targets(box):
    """(..., 7) x, y, z, l, w, h, yaw -> (..., 8) x, y, z, log l, log w, log h, sin yaw, cos yaw."""
    return torch.cat(
        [box[..., :3], box[..., 3:6].clamp(min=0.05).log(), box[..., 6:7].sin(), box[..., 6:7].cos()], -1
    )


def softmax_focal(logits, target, gamma: float = 2.0):
    logp = F.log_softmax(logits, -1).gather(-1, target[..., None])[..., 0]
    return -((1 - logp.exp()) ** gamma) * logp


@torch.no_grad()
def hungarian(out, cls, box, mask, w_cls: float = 2.0, w_ctr: float = 0.25):
    """Per sample: (pred slots, target slots) minimising w_cls * -p(class) + w_ctr * L1(centre, m)."""
    prob = out["logits"].softmax(-1)
    cost_cls = -prob.gather(-1, cls[:, None, :].expand(-1, prob.shape[1], -1))
    cost_ctr = torch.cdist(out["box"][..., :3], box[..., :3], p=1)
    cost = (w_cls * cost_cls + w_ctr * cost_ctr).cpu().numpy()
    valid = mask.cpu().numpy()
    result = []
    for b in range(len(cost)):
        t = np.flatnonzero(valid[b])
        if len(t) == 0:
            result.append((np.zeros(0, np.int64), np.zeros(0, np.int64)))
            continue
        i, j = linear_sum_assignment(cost[b][:, t])
        result.append((i, t[j]))
    return result


class SideCarLoss(nn.Module):
    """H04's AgentLoss (focal class, L1 box and velocity, every layer) plus L1 on the path."""

    def __init__(self, w_cls: float = 2.0, w_box: float = 0.25, w_vel: float = 0.25, w_path: float = 0.25):
        super().__init__()
        self.w_cls, self.w_box, self.w_vel, self.w_path = w_cls, w_box, w_vel, w_path

    def forward(self, outs, tgt):
        cls, box, vel, mask = tgt["cls"], tgt["box"], tgt["vel"], tgt["mask"]
        n = mask.sum().clamp(min=1).float()
        n_path = tgt["path_ok"].sum().clamp(min=1).float()
        box8 = box_targets(box)
        total, parts = 0.0, {}
        for li, out in enumerate(outs):
            idx = hungarian(out, cls, box, mask)
            bi = torch.cat([torch.full((len(i),), b, dtype=torch.long) for b, (i, _) in enumerate(idx)]).to(
                cls.device
            )
            pi = torch.cat([torch.as_tensor(i, dtype=torch.long) for i, _ in idx]).to(cls.device)
            ti = torch.cat([torch.as_tensor(j, dtype=torch.long) for _, j in idx]).to(cls.device)
            target = torch.full(out["logits"].shape[:2], NONE, dtype=torch.long, device=cls.device)
            target[bi, pi] = cls[bi, ti]
            l_cls = softmax_focal(out["logits"], target).sum() / n
            l_box = (out["box"][bi, pi] - box8[bi, ti]).abs().sum() / n
            l_vel = (out["vel"][bi, pi] - vel[bi, ti]).abs().sum() / n
            ok = tgt["path_ok"][bi, ti][..., None].float()
            l_path = ((out["path"][bi, pi] - tgt["path"][bi, ti]).abs() * ok).sum() / n_path
            total = (
                total + self.w_cls * l_cls + self.w_box * l_box + self.w_vel * l_vel + self.w_path * l_path
            )
            parts.update({f"cls{li}": float(l_cls), f"box{li}": float(l_box), f"vel{li}": float(l_vel)})
            parts[f"path{li}"] = float(l_path)
            if li == len(outs) - 1:
                parts["_match"] = (bi, pi, ti)
        return total, parts
