"""Agent decoder (agent decoder spec, 22 Oct): the agents around the car (class, box, velocity) from the frozen WA-JEPA
tokens of b0 f0 l0 r0. No planner here; the planner run starts from this checkpoint later.

    mem = LN(Linear(1024, 256)(tokens))                       tokens (B, 4, 512, 1024) -> (B, 2048, 256)
    ray_pos = MLP 48-256-256 of run 2's ray points           (global_look.token_rays: 16 points per token ray, depths
                                                              1..60 m geometric, scaled x/64, y/64, z/8), (B, 2048, 256)
    64 learned queries a (256-d) + 64 learned reference points ref (x, y, z) in metres, initialised uniformly over
    +-32 m in x and y with z = 0; pos(ref) = MLP 3-256-256 of (x/64, y/64, z/8)
    3 layers, each:
        a <- a + SA(q = k = LN(a) + pos(ref), v = LN(a))                       8 heads
        a <- a + CA(q = LN(a) + pos(ref), k = mem + ray_pos, v = mem)          8 heads, all 2048 tokens
        a <- a + FFN(LN(a))
        ref' = ref + MLP(a)                                                    metres, last layer zero-initialised
        heads: class (vehicle, pedestrian, cyclist, none); box = (ref', log l, log w, log h, sin yaw, cos yaw), i.e.
               the box centre is the updated reference point; velocity (vx, vy)
        the next layer starts from ref'.detach()
Loss (AgentLoss): per layer, Hungarian matching per sample on 2 * (-p(true class)) + 0.25 * L1(centre, m); softmax focal
loss (gamma 2) on all 64 slots, unmatched slots -> none; L1 on the 8 box terms and on velocity for matched slots;
weights 2 / 0.25 / 0.25, each normalised by the number of agents in the batch; summed over the 3 layers.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .global_look import token_rays

NUM_CLASSES = 3  # vehicle, pedestrian, cyclist (AGENT_CLASSES in drive_jepa_features.py); index 3 = none
NONE = NUM_CLASSES
TO_512 = [[2 / 3, 0, 0, 0], [0, 0.625, -7, 0], [0, 0, 1, 0], [0, 0, 0, 1]]  # drive_jepa_model.to_512


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class MHA(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.dropout = heads, dropout
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, q, k, v):
        B, Nq, D = q.shape
        sh = lambda x: x.view(B, -1, self.h, D // self.h).transpose(1, 2)  # noqa: E731
        out = F.scaled_dot_product_attention(sh(self.q(q)), sh(self.k(k)), sh(self.v(v)),
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(B, Nq, D))


class DecoderLayer(nn.Module):
    def __init__(self, d, heads, ffn, dropout):
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
            self.cls_head.bias[NONE] = 2.0  # start near p(none) ~ 0.7, about the share of empty slots
        self.box_head = mlp(d, d, 5)  # log l, log w, log h, sin yaw, cos yaw
        self.vel_head = mlp(d, d, 2)

    def forward(self, a, pos, k, v):
        h = self.ln_sa(a)
        a = a + self.drop(self.sa(h + pos, h + pos, h))
        a = a + self.drop(self.ca(self.ln_ca(a) + pos, k, v))
        return a + self.drop(self.ffn(self.ln_ff(a)))


class AgentDecoder(nn.Module):
    def __init__(self, d=256, heads=8, layers=3, queries=64, ffn=1024, dropout=0.1, n_depths=16, min_depth=1.0, max_depth=60.0):
        super().__init__()
        self.proj = nn.Linear(1024, d)
        self.ln_tok = nn.LayerNorm(d)
        self.ray_mlp = mlp(3 * n_depths, 256, d)
        self.register_buffer("depths", torch.logspace(math.log10(min_depth), math.log10(max_depth), n_depths), persistent=False)
        self.register_buffer("scale", torch.tensor([1 / 64, 1 / 64, 1 / 8]), persistent=False)
        self.register_buffer("to_512", torch.tensor(TO_512, dtype=torch.float32), persistent=False)
        self.query = nn.Parameter(torch.randn(queries, d) * 0.02)
        ref = torch.zeros(queries, 3)
        ref[:, :2] = torch.rand(queries, 2) * 64.0 - 32.0
        self.ref = nn.Parameter(ref)
        self.pos_mlp = mlp(3, 256, d)
        self.layers = nn.ModuleList([DecoderLayer(d, heads, ffn, dropout) for _ in range(layers)])

    def ray_pos(self, lidar2img):
        """run 2's ray points (GlobalLook.ray_points) for Drive-JEPA's lidar2img (B, 4, 4, 4) -> (B, 2048, 256)."""
        with torch.autocast(device_type="cuda", enabled=False):
            C, d = token_rays(self.to_512 @ lidar2img.float())
            P = C[:, :, None, None, :] + self.depths[None, None, None, :, None] * d[:, :, :, None, :]  # B, 4, 512, 16, 3
            pts = (P * self.scale).flatten(-2).flatten(1, 2)  # B, 2048, 48
        return self.ray_mlp(pts)

    def forward(self, tokens, lidar2img):
        """tokens (B, 4, 512, 1024), lidar2img (B, 4, 4, 4) as cached -> list over layers of dicts
        logits (B, Q, 4), box (B, Q, 8) = x, y, z, log l, log w, log h, sin, cos; vel (B, Q, 2)."""
        B = tokens.shape[0]
        x = tokens.flatten(1, 2)  # float16 from the cache; autocast runs the Linear in fp16 anyway
        if not torch.is_autocast_enabled():
            x = x.to(self.proj.weight.dtype)
        mem = self.ln_tok(self.proj(x))
        k = mem + self.ray_pos(lidar2img).to(mem.dtype)
        a = self.query[None].expand(B, -1, -1)
        ref = self.ref[None].expand(B, -1, -1)
        outs = []
        for layer in self.layers:
            pos = self.pos_mlp(ref * self.scale)
            a = layer(a, pos, k, mem)
            ref_new = ref + layer.ref_mlp(a).float()
            outs.append({"logits": layer.cls_head(a).float(),
                         "box": torch.cat([ref_new, layer.box_head(a).float()], -1),
                         "vel": layer.vel_head(a).float(),
                         "feat": a})  # run 3A: the agent look reads the last layer's slot features
            ref = ref_new.detach()
        return outs


def box_targets(box):
    """(..., 7) x, y, z, l, w, h, yaw -> (..., 8) x, y, z, log l, log w, log h, sin yaw, cos yaw."""
    return torch.cat([box[..., :3], box[..., 3:6].clamp(min=0.05).log(), box[..., 6:7].sin(), box[..., 6:7].cos()], -1)


def softmax_focal(logits, target, gamma=2.0):
    logp = F.log_softmax(logits, -1).gather(-1, target[..., None])[..., 0]
    return -((1 - logp.exp()) ** gamma) * logp


@torch.no_grad()
def hungarian(out, cls, box, mask, w_cls=2.0, w_ctr=0.25):
    """Per sample: indices (pred slots, target slots) minimising w_cls * -p(class) + w_ctr * L1(centre xyz, m)."""
    prob = out["logits"].softmax(-1)  # B, Q, 4
    B = prob.shape[0]
    cost_cls = -prob.gather(-1, cls[:, None, :].expand(-1, prob.shape[1], -1))  # B, Q, T
    cost_ctr = torch.cdist(out["box"][..., :3], box[..., :3], p=1)  # B, Q, T
    C = (w_cls * cost_cls + w_ctr * cost_ctr).cpu().numpy()
    m = mask.cpu().numpy()
    res = []
    for b in range(B):
        t = np.flatnonzero(m[b])
        if len(t) == 0:
            res.append((np.zeros(0, np.int64), np.zeros(0, np.int64)))
            continue
        i, j = linear_sum_assignment(C[b][:, t])
        res.append((i, t[j]))
    return res


class AgentLoss(nn.Module):
    def __init__(self, w_cls=2.0, w_box=0.25, w_vel=0.25, gamma=2.0):
        super().__init__()
        self.w_cls, self.w_box, self.w_vel, self.gamma = w_cls, w_box, w_vel, gamma

    def forward(self, outs, tgt):
        cls, box, vel, mask = tgt["agent_class"], tgt["agent_box"], tgt["agent_vel"], tgt["agent_mask"]
        n = mask.sum().clamp(min=1).float()
        box8 = box_targets(box)
        total, parts = 0.0, {}
        for li, out in enumerate(outs):
            idx = hungarian(out, cls, box, mask)
            bi = torch.cat([torch.full((len(i),), b, dtype=torch.long) for b, (i, _) in enumerate(idx)]).to(cls.device)
            pi = torch.cat([torch.as_tensor(i, dtype=torch.long) for i, _ in idx]).to(cls.device)
            ti = torch.cat([torch.as_tensor(j, dtype=torch.long) for _, j in idx]).to(cls.device)
            target = torch.full(out["logits"].shape[:2], NONE, dtype=torch.long, device=cls.device)
            target[bi, pi] = cls[bi, ti]
            l_cls = softmax_focal(out["logits"], target, self.gamma).sum() / n
            l_box = (out["box"][bi, pi] - box8[bi, ti]).abs().sum() / n
            l_vel = (out["vel"][bi, pi] - vel[bi, ti]).abs().sum() / n
            total = total + self.w_cls * l_cls + self.w_box * l_box + self.w_vel * l_vel
            parts.update({f"cls{li}": float(l_cls), f"box{li}": float(l_box), f"vel{li}": float(l_vel)})
            if li == len(outs) - 1:
                parts["_match"] = (bi, pi, ti)
        return total, parts
