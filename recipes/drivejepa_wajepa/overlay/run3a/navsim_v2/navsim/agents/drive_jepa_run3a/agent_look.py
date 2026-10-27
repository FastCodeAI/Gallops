"""Agent look (paddock-ops/run3a/CLAUDE.md, "Agent look"): every waypoint query attends to the k nearest detected agents
at its own time. One block, shared by the 4 rounds (it lives in the shared Traj_refiner), right after the footprint look.

Inputs  x (B, P*T, 256) waypoint queries with this round's poses (B, P, T, 3) (x, y, psi, rear axle, ego frame at t = 0;
        detached); agent features A (B, Q, 256) from the agent decoder's last layer; its boxes (B, Q, 8) = x, y, z,
        log l, log w, log h, sin, cos, velocities (B, Q, 2) and class probabilities (B, Q, 4), last = none; detached.
Select  slots with P(none) >= cut are dropped; each agent moves to the waypoint's time (centre + velocity * t, t = 0.5 ..
        4.0 s); the k nearest per waypoint are kept (B, P*T, k), missing places masked. Gather keeps gradients to A.
Code    in the waypoint's frame: (dx/32, dy/32, dvx/10, dvy/10, dist/32, l, w, 3 class probabilities); waypoint velocity
        from consecutive poses (the first from the origin). MLP 10 -> 64, then a score bias (64 -> 8 heads) and a value
        offset (64 -> 256).
Attend  s = (W_q LN(x)) . (W_k A) / sqrt(d_head) + bias + log(1 - P(none) + 1e-4), plus a learned "nothing" key and
        value; m = sum w (W_v A + offset); x <- x + W_o m, W_o zero-initialised.
Logged per round (last_stats): ||W_o m|| / ||x||, attention on "nothing", mean kept agents per waypoint.
"""
import math

import torch
import torch.nn as nn

DT = 0.5


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


def waypoint_velocity(poses, dt=DT):
    """poses (B, P, T, 3) -> (B, P, T, 2): (p_j - p_{j-1}) / dt, p_{-1} = origin."""
    xy = poses[..., :2]
    prev = torch.cat([torch.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], -2)
    return (xy - prev) / dt


def select_agents(poses, centre, vel, p_none, k, cut, dt=DT):
    """poses (B, P, T, 3), centre (B, Q, 2), vel (B, Q, 2), p_none (B, Q) -> idx (B, P, T, k), valid (B, P, T, k):
    the k agents nearest to each waypoint at its own time t_j = dt * (j + 1), slots with p_none >= cut never kept."""
    T = poses.shape[2]
    t = dt * torch.arange(1, T + 1, device=poses.device, dtype=centre.dtype)
    at_t = centre[:, :, None] + vel[:, :, None] * t[None, None, :, None]  # (B, Q, T, 2)
    d2 = ((poses[:, :, :, None, :2] - at_t.permute(0, 2, 1, 3)[:, None]) ** 2).sum(-1)  # (B, P, T, Q)
    d2 = d2.masked_fill((p_none >= cut)[:, None, None, :], float("inf"))
    kk = min(k, d2.shape[-1])
    dist, idx = torch.topk(d2, kk, dim=-1, largest=False)
    valid = torch.isfinite(dist)
    if kk < k:  # fewer slots than k: pad with masked places
        pad = k - kk
        idx = torch.cat([idx, idx.new_zeros(*idx.shape[:-1], pad)], -1)
        valid = torch.cat([valid, valid.new_zeros(*valid.shape[:-1], pad)], -1)
    return idx, valid


def relative_code(poses, centre, vel, length, width, probs, idx, dt=DT):
    """Relative code of the kept agents in each waypoint's frame -> (B, P, T, k, 10)."""
    B, P, T, K = idx.shape
    g = lambda x: torch.gather(x[:, None, None].expand(B, P, T, *x.shape[1:]), 3,  # noqa: E731
                               idx.reshape(B, P, T, K, *([1] * (x.dim() - 2))).expand(B, P, T, K, *x.shape[2:]))
    t = dt * torch.arange(1, T + 1, device=poses.device, dtype=centre.dtype)[None, None, :, None, None]
    c, v = g(centre), g(vel)  # (B, P, T, K, 2)
    pos = c + v * t
    psi = poses[..., 2][..., None]
    cs, sn = torch.cos(psi), torch.sin(psi)
    rel = pos - poses[..., None, :2]
    dx, dy = cs * rel[..., 0] + sn * rel[..., 1], -sn * rel[..., 0] + cs * rel[..., 1]
    dvw = v - waypoint_velocity(poses, dt)[..., None, :]
    dvx, dvy = cs * dvw[..., 0] + sn * dvw[..., 1], -sn * dvw[..., 0] + cs * dvw[..., 1]
    dist = torch.sqrt(dx ** 2 + dy ** 2)
    return torch.cat([torch.stack([dx / 32, dy / 32, dvx / 10, dvy / 10, dist / 32, g(length), g(width)], -1),
                      g(probs)], -1)


class AgentLook(nn.Module):
    def __init__(self, d_model=256, num_heads=8, k=16, cut=0.5, code_hidden=64):
        super().__init__()
        self.d, self.h, self.k, self.cut = d_model, num_heads, k, cut
        self.ln = nn.LayerNorm(d_model)
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(d_model, d_model) for _ in range(3))
        self.o_proj = nn.Linear(d_model, d_model)  # W_o
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)
        self.code_mlp = nn.Sequential(nn.Linear(10, code_hidden), nn.ReLU())
        self.bias_head = nn.Linear(code_hidden, num_heads)
        self.offset_head = nn.Linear(code_hidden, d_model)
        self.null_k = nn.Parameter(torch.randn(d_model) * 0.02)  # learned "nothing" key and value (projected space)
        self.null_v = nn.Parameter(torch.zeros(d_model))
        self.last_stats = []

    @staticmethod
    def agents_from_decoder(out, feat):
        """agent decoder's last layer dict + features -> the detached geometry the look needs."""
        box, vel = out["box"].detach().float(), out["vel"].detach().float()
        probs = out["logits"].detach().float().softmax(-1)
        return {"feat": feat, "centre": box[..., :2], "vel": vel, "length": box[..., 3].exp(), "width": box[..., 4].exp(),
                "probs": probs[..., :3], "p_none": probs[..., 3]}

    def forward(self, x, poses, agents, return_attn=False):
        B, N, D = x.shape
        P, T = poses.shape[1], poses.shape[2]
        H, dh = self.h, D // self.h
        poses = poses.detach().float()
        idx, valid = select_agents(poses, agents["centre"], agents["vel"], agents["p_none"], self.k, self.cut)
        code = relative_code(poses, agents["centre"], agents["vel"], agents["length"], agents["width"], agents["probs"], idx)
        K = idx.shape[-1]
        idx_f, valid_f = idx.reshape(B, N, K), valid.reshape(B, N, K)
        h = self.code_mlp(code.reshape(B, N, K, 10).to(x.dtype))
        bias, offset = self.bias_head(h), self.offset_head(h)  # (B, N, K, H), (B, N, K, D)

        feat = agents["feat"]
        kq, vq = self.k_proj(feat), self.v_proj(feat)  # (B, Q, D)
        gidx = idx_f.reshape(B, N * K, 1).expand(-1, -1, D)
        ks = torch.gather(kq, 1, gidx).reshape(B, N, K, H, dh)
        vs = torch.gather(vq, 1, gidx).reshape(B, N, K, H, dh) + offset.reshape(B, N, K, H, dh).to(vq.dtype)
        q = self.q_proj(self.ln(x)).reshape(B, N, H, dh)

        s = torch.einsum("bnhd,bnkhd->bnhk", q.float(), ks.float()) / math.sqrt(dh)
        pn = torch.gather(agents["p_none"], 1, idx_f.reshape(B, N * K)).reshape(B, N, K)
        s = s + bias.float().permute(0, 1, 3, 2) + torch.log(1 - pn + 1e-4)[:, :, None, :]
        s = s.masked_fill(~valid_f[:, :, None, :], float("-inf"))
        s_null = torch.einsum("bnhd,hd->bnh", q.float(), self.null_k.float().reshape(H, dh))[..., None] / math.sqrt(dh)
        w = torch.cat([s, s_null], -1).softmax(-1)  # (B, N, H, K + 1)
        m = torch.einsum("bnhk,bnkhd->bnhd", w[..., :K], vs.float()) + w[..., K:] * self.null_v.float().reshape(H, dh)
        upd = self.o_proj(m.reshape(B, N, D).to(x.dtype))
        with torch.no_grad():
            self.last_stats.append((float(upd.float().norm() / x.float().norm().clamp(min=1e-6)),
                                    float(w[..., K].mean()), float(valid_f.float().sum(-1).mean())))
        out = x + upd
        if return_attn:
            return out, {"w": w, "idx": idx, "valid": valid, "code": code}
        return out
