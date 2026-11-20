"""Global look (wa-jepa/CLAUDE.md 3.5): every proposal-pose query attends to all 4 x 512 WA-JEPA tokens.

One block, shared by the 4 refinement rounds (it lives in the shared Traj_refiner), applied right after the footprint
look (Bev_refiner); kv=None switches it off.
    k = LN(tokens) + ray_pos, v = LN(tokens)     tokens: 4 x 512 after Linear(1024, 256), before the camera embedding;
                                                 computed once per forward, before the round loop (kv())
    q = LN(x) + pose_pos(MLP 4-256-256 of [x/64, y/64, cos psi, sin psi] of this round's proposals, detached)
    x <- x + W_o MHA(q, k, v), 8 heads, dropout 0, W_o weight and bias zero-initialised
ray_pos: for lidar2img l2i converted into the 512x256 model image (A @ Drive-JEPA's matrix, in the model's forward),
M = l2i[:3, :3], m = l2i[:3, 3]; C = -M^-1 m; d = normalise(M^-1 [u, v, 1]) with u = 16 c + 8, v = 16 r + 8 for the
token in row r, column c;
16 points C + depth d, depths geometric 1..60 m, scaled (x/64, y/64, z/8), MLP 48-256-256 (ReLU).
Logged per round (last_stats): ||W_o MHA|| / ||x|| and the attention share per camera.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

H_TOK, W_TOK = 16, 32


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


def token_rays(lidar2img: torch.Tensor):
    """lidar2img (B, N, 4, 4) for the 512x256 image -> C (B, N, 3), d (B, N, 512, 3), fp32. Tokens row-major over 16 x 32."""
    l2i = lidar2img.float()
    M_inv = torch.linalg.inv(l2i[..., :3, :3])
    C = -(M_inv @ l2i[..., :3, 3:4]).squeeze(-1)
    r, c = torch.meshgrid(torch.arange(H_TOK, device=l2i.device), torch.arange(W_TOK, device=l2i.device), indexing="ij")
    u = 16.0 * c.flatten() + 8.0
    v = 16.0 * r.flatten() + 8.0
    pix = torch.stack([u, v, torch.ones_like(u)], -1)  # (512, 3)
    d = torch.einsum("bnij,pj->bnpi", M_inv, pix)
    return C, F.normalize(d, dim=-1)


class GlobalLook(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 8, n_depths: int = 16, min_depth: float = 1.0, max_depth: float = 60.0):
        super().__init__()
        self.d, self.h = d_model, num_heads
        self.ln_q, self.ln_k, self.ln_v = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.pose_mlp = mlp(4, 256, d_model)
        self.ray_mlp = mlp(3 * n_depths, 256, d_model)
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)  # W_o
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)
        self.register_buffer("depths", torch.logspace(math.log10(min_depth), math.log10(max_depth), n_depths), persistent=False)
        self.register_buffer("scale", torch.tensor([1 / 64, 1 / 64, 1 / 8]), persistent=False)
        self.last_stats = []  # per round: (ratio, attention share per camera)

    def ray_points(self, lidar2img):
        with torch.autocast(device_type="cuda", enabled=False):
            C, d = token_rays(lidar2img)
            P = C[:, :, None, None, :] + self.depths[None, None, None, :, None] * d[:, :, :, None, :]  # B, N, 512, 16, 3
            return (P * self.scale).flatten(-2)  # B, N, 512, 48

    def kv(self, tokens, lidar2img):
        """tokens (B, 4, 512, C), lidar2img (B, 4, 4, 4) -> projected k, v (B, heads, 2048, C / heads)."""
        self.last_stats = []
        B = tokens.shape[0]
        k = self.ln_k(tokens) + self.ray_mlp(self.ray_points(lidar2img))
        v = self.ln_v(tokens)
        k = self.k_proj(k.flatten(1, 2)).view(B, -1, self.h, self.d // self.h).transpose(1, 2)
        v = self.v_proj(v.flatten(1, 2)).view(B, -1, self.h, self.d // self.h).transpose(1, 2)
        return k, v

    def forward(self, x, proposals, kv):
        """x (B, Q, C) proposal-pose queries, proposals (B, P, T, 3) of this round -> x + W_o MHA(q, k, v)."""
        B, Q, _ = x.shape
        p = proposals.detach().reshape(B, Q, 3)
        pose = torch.cat([p[..., :2] / 64.0, torch.cos(p[..., 2:3]), torch.sin(p[..., 2:3])], -1)
        q = self.ln_q(x) + self.pose_mlp(pose)
        q = self.q_proj(q).view(B, Q, self.h, self.d // self.h).transpose(1, 2)
        k, v = kv
        out = self.o_proj(F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype)).transpose(1, 2).reshape(B, Q, self.d))
        with torch.no_grad():  # logging only: ||W_o MHA|| / ||x||, attention mass per camera (first 4 samples)
            att = torch.softmax((q[:4].float() @ k[:4].float().transpose(-1, -2)) / math.sqrt(self.d // self.h), -1)
            share = att.view(*att.shape[:-1], 4, -1).sum(-1).mean((0, 1, 2))  # (4,) b0 f0 l0 r0
            self.last_stats.append((float(out.float().norm() / x.float().norm().clamp(min=1e-12)), share.tolist()))
        return x + out
