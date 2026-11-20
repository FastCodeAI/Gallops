"""Run 3B v2 model (run3b/CLAUDE.md, revised 19 Nov ~02:15).

  encoder   WA-JEPA ViT-L, run 3A's (bevformer/wajepa.py build_wajepa_encoder: same weights, same VIT_KWARGS), base frozen;
            fresh LoRA r32 / alpha 32 on Q and V of all 24 blocks' fused qkv (phase1/model/lora.py LoRAQKV: A kaiming, B = 0,
            so at step 0 the encoder is run 3A's). Input: 4 cameras b0 f0 l0 r0 x 2 frames (t - 0.5 s, t; LEAD f-2, f),
            normalised (3, 256, 512) -> tokens (B, 4, 512, 1024), bf16 autocast as run 3A's cache builder.
  heads     run 3A's architecture from scratch (nothing loaded): agent decoder (300 slots, 3 layers; drive_jepa_run3a/
            agent_decoder.py), planner (token projector + camera embeddings, ego history, 4 shared Traj_refiner rounds with
            footprint look + agent look), run 3A's aux heads (score_module/scorer.Scorer: key agents, ego areas, map aux;
            its score logits are not used), the new path reader (fact heads per waypoint) and the expert's-metres head.
  teacher forcing (section 3): agents_tf() swaps, per teacher-forced sample, the matched slots' box / velocity for the true
            ones (+ noise) and sets P(none) 0 for matched slots, 1 for the rest; the slot features are kept.
Camera order and lidar2img are run 3A's (b0 f0 l0 r0; Drive-JEPA's 768 x 432 matrices, to_512 applied inside as run 3A)."""
import math
import sys

import torch
import torch.nn as nn

DEVKIT = "{data.drivejepa_tree}/navsim_v2"
for p in (DEVKIT, "{runs}/run3a-adapted", "{runs}/run3b/r3b"):
    if p not in sys.path:
        sys.path.insert(0, p)

from navsim.agents.drive_jepa_run3a.agent_decoder import AgentDecoder  # noqa: E402
from navsim.agents.drive_jepa_run3a.agent_look import AgentLook  # noqa: E402
from navsim.agents.drive_jepa_run3a.bevformer.spatial_cross_attention import SpatialCrossAttention  # noqa: E402
from navsim.agents.drive_jepa_run3a.bevformer.transformer_decoder import MLP  # noqa: E402
from navsim.agents.drive_jepa_run3a.drive_jepa_config import DriveJEPAConfig  # noqa: E402
from navsim.agents.drive_jepa_run3a.score_module.scorer import Scorer  # noqa: E402
from navsim.agents.drive_jepa_run3a.traj_refiner import Traj_refiner  # noqa: E402

import navsim.agents.drive_jepa_run3a.agent_look as _AL  # noqa: E402


def _agent_look_forward(self, x, poses, agents, return_attn=False):
    """run 3A's AgentLook.forward (agent_look.py:96-129), line for line, without its diagnostic last_stats (three float()
    host reads per call: a GPU sync and a torch.compile graph break each). Same arithmetic, same outputs. Swapped in for
    this process only; run 3A's file is not edited."""
    B, N, D = x.shape
    H, dh = self.h, D // self.h
    poses = poses.detach().float()
    idx, valid = _AL.select_agents(poses, agents["centre"], agents["vel"], agents["p_none"], self.k, self.cut)
    code = _AL.relative_code(poses, agents["centre"], agents["vel"], agents["length"], agents["width"], agents["probs"], idx)
    K = idx.shape[-1]
    idx_f, valid_f = idx.reshape(B, N, K), valid.reshape(B, N, K)
    h = self.code_mlp(code.reshape(B, N, K, 10).to(x.dtype))
    bias, offset = self.bias_head(h), self.offset_head(h)
    feat = agents["feat"]
    kq, vq = self.k_proj(feat), self.v_proj(feat)
    gidx = idx_f.reshape(B, N * K, 1).expand(-1, -1, D)
    ks = torch.gather(kq, 1, gidx).reshape(B, N, K, H, dh)
    vs = torch.gather(vq, 1, gidx).reshape(B, N, K, H, dh) + offset.reshape(B, N, K, H, dh).to(vq.dtype)
    q = self.q_proj(self.ln(x)).reshape(B, N, H, dh)
    s = torch.einsum("bnhd,bnkhd->bnhk", q.float(), ks.float()) / math.sqrt(dh)
    pn = torch.gather(agents["p_none"], 1, idx_f.reshape(B, N * K)).reshape(B, N, K)
    s = s + bias.float().permute(0, 1, 3, 2) + torch.log(1 - pn + 1e-4)[:, :, None, :]
    s = s.masked_fill(~valid_f[:, :, None, :], float("-inf"))
    s_null = torch.einsum("bnhd,hd->bnh", q.float(), self.null_k.float().reshape(H, dh))[..., None] / math.sqrt(dh)
    w = torch.cat([s, s_null], -1).softmax(-1)
    m = torch.einsum("bnhk,bnkhd->bnhd", w[..., :K], vs.float()) + w[..., K:] * self.null_v.float().reshape(H, dh)
    upd = self.o_proj(m.reshape(B, N, D).to(x.dtype))
    out = x + upd
    if return_attn:
        return out, {"w": w, "idx": idx, "valid": valid, "code": code}
    return out


AgentLook.forward = _agent_look_forward

YESNO = ("hit", "offroad", "left_route", "red", "stop", "wrong_way", "comfort")
NUMBERS = ("metres_sofar", "lane_off", "ttc", "cutin_brake")
N_CAMS, GRID = 4, (16, 32)
TO_512 = [[2 / 3, 0, 0, 0], [0, 0.625, -7, 0], [0, 0, 1, 0], [0, 0, 0, 1]]  # run 3A: Drive-JEPA's 768 x 432 -> 512 x 256


def run3a_config():
    c = DriveJEPAConfig()
    c.use_wajepa_tokens = True
    c.agent_decoder_ckpt = ""  # from scratch (nothing loaded)
    return c


# ----------------------------------------------------------------------------------------------------------------- encoder
def build_encoder(dev, r=32, alpha=32.0, checkpointing=True, compile_blocks=False):
    from navsim.agents.drive_jepa_perception_based.bevformer.wajepa import build_wajepa_encoder
    from phase1.model.lora import LoRAQKV
    enc, msg = build_wajepa_encoder("cpu")
    for blk in enc.blocks:
        blk.attn.qkv = LoRAQKV(blk.attn.qkv, r, alpha)
    enc.use_activation_checkpointing = checkpointing  # block inputs only kept; blocks recomputed in backward
    for n, p in enc.named_parameters():
        p.requires_grad_(any(k in n for k in ("A_q", "B_q", "A_v", "B_v")))
    if compile_blocks:  # as Phase 1's runs (train.py --compile-vit): every block the same shapes -> one graph, no recompiles
        torch._inductor.config.emulate_precision_casts = True
        for blk in enc.blocks:
            blk.compile()
    return enc.to(dev), msg


class Encoder(nn.Module):
    """clips (B * 4, 3, 2, 256, 512), normalised (v2/data run3b_data2.to_clips) -> tokens (B, 4, 512, 1024) (bf16 autocast, as
    the cache builder); `group` samples per encoder call (memory)."""

    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, clips, group=None):
        B = clips.shape[0] // N_CAMS
        G = (group or B) * N_CAMS
        outs = []
        from torch.nn.attention import SDPBackend, sdpa_kernel
        # FlashAttention-2 only (PyTorch's flash SDPA backend): an ineligible call errors instead of silently falling back
        with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for s in range(0, B * N_CAMS, G):
                outs.append(self.vit(clips[s:s + G], training=False))
        return torch.cat(outs).view(B, N_CAMS, 512, 1024)


# ------------------------------------------------------------------------------------------------------------------- looks
def image_feature(feat, l2i_512, foot):
    B = feat.shape[0]
    fp = list(foot)
    flat = feat[:, fp].permute(1, 2, 0, 3)
    shape = torch.as_tensor([list(GRID)], dtype=torch.long, device=feat.device)
    start = shape.new_zeros((1,))
    metas = {"lidar2img": l2i_512[:, fp].float(), "img_shape": l2i_512.new_tensor([256.0, 512.0, 3.0]).float().expand(B, len(fp), 3)}
    return flat, shape, start, {"img_metas": metas}


def footprint_points(poses, cfg):
    x, y, h = poses[..., 0], poses[..., 1], poses[..., 2]
    c, s = torch.cos(h)[..., None], torch.sin(h)[..., None]
    cx, cy = x[..., None] + cfg.rear_axle_to_center * c, y[..., None] + cfg.rear_axle_to_center * s
    hl, hw = cfg.half_length, cfg.half_width
    kx = poses.new_tensor([hl, hl, -hl, -hl])
    ky = poses.new_tensor([hw, -hw, -hw, hw])
    corners = torch.stack([c * kx - s * ky + cx, s * kx + c * ky + cy], -1)
    zs = torch.linspace(cfg.point_cloud_range[2] - cfg.lidar_height, cfg.point_cloud_range[5] - cfg.lidar_height,
                        cfg.num_points_in_pillar, device=poses.device, dtype=poses.dtype)
    P = corners[..., None, :, :].expand(*corners.shape[:2], len(zs), 4, 2)
    Z = zs[None, None, :, None, None].expand(*corners.shape[:2], len(zs), 4, 1)
    return torch.cat([P, Z, torch.ones_like(Z)], -1).reshape(*corners.shape[:2], len(zs) * 4, 4)


def point_sampling(pts, l2i):
    cam = torch.einsum("bcij,bndj->cbndi", l2i.float(), pts.float())
    eps = 1e-5
    mask = cam[..., 2] > eps
    uv = cam[..., :2] / torch.maximum(cam[..., 2:3], torch.full_like(cam[..., 2:3], eps))
    uv = torch.stack([uv[..., 0] / 512.0, uv[..., 1] / 256.0], -1)
    mask = mask & (uv[..., 1] > 0) & (uv[..., 1] < 1) & (uv[..., 0] < 1) & (uv[..., 0] > 0)
    return uv, torch.nan_to_num(mask.float()).bool()


class Planner(nn.Module):
    """run 3A's DriveJEPAModel token path (projector, camera embeddings, ego history, shared refiner x ref_num, Scorer for
    the aux heads) + the expert's-metres head; from scratch."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.tf_d_model
        self.cfg = cfg
        self.projector = nn.Linear(1024, d)
        self.cams_embeds = nn.Parameter(torch.randn([N_CAMS, d]))
        self.hist_encoding = nn.Linear(11, d)
        self.init_feature = nn.Embedding(cfg.num_poses * cfg.proposal_num, d)
        shared = Traj_refiner(cfg)
        self._trajectory_head = nn.ModuleList([shared for _ in range(cfg.ref_num)])
        self.scorer = Scorer(cfg)
        self.metres_head = MLP(d, cfg.tf_d_ffn, cfg.num_poses)

    def forward(self, tokens, l2i_512, ego_status, agents):
        x = tokens
        if not torch.is_autocast_enabled():
            x = x.to(self.projector.weight.dtype)
        feat = self.projector(x) + self.cams_embeds[None, :, None, :].to(x.dtype)
        imf = image_feature(feat, l2i_512, self.cfg.footprint_cams)
        ego = ego_status[:, -1].to(self.hist_encoding.weight.dtype)
        bev = self.hist_encoding(ego)[:, None] + self.init_feature.weight[None]
        plist = []
        self._trajectory_head[0].agent_look.last_stats = []
        for refine in self._trajectory_head:
            bev, props = refine(bev, imf, agents)
            plist.append(props)
        props = plist[-1]
        _, pred_agents_states, pred_area_logit, _, _, _, pred_map_aux = self.scorer(props, bev)
        return dict(proposals=props, proposal_list=plist, pred_agents_states=pred_agents_states, pred_area_logit=pred_area_logit,
                    pred_map_aux=pred_map_aux, expert_metres=self.metres_head(bev.amax(1).float()))


class PathReaderLayer(nn.Module):
    def __init__(self, d, ffn, cfg):
        super().__init__()
        nf = len(cfg.footprint_cams)
        self.sca = SpatialCrossAttention(embed_dims=d, num_cams=nf, pc_range=cfg.point_cloud_range, dropout=0.0,
                                         deformable_attention=dict(type="MSDeformableAttention3D", embed_dims=d,
                                                                   num_points=cfg.num_points_in_pillar * 4, num_levels=1))
        self.n1 = nn.LayerNorm(d)
        self.agent_look = AgentLook(d, 8, cfg.agent_look_k, cfg.agent_look_cut)
        self.sa = nn.MultiheadAttention(d, 8, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.ReLU(inplace=True), nn.Linear(ffn, d))
        self.n3 = nn.LayerNorm(d)

    def forward(self, x, poses, imf, ref_cam, mask, agents):
        B, P, T, D = x.shape
        flat, shape, start, _ = imf
        q = x.reshape(B, P * T, D)
        q = self.n1(self.sca(q, flat, flat, reference_points_cam=ref_cam, bev_mask=mask, spatial_shapes=shape,
                             level_start_index=start))
        q = self.agent_look(q, poses, agents)
        w = q.reshape(B * P, T, D)
        w = self.n2(w + self.sa(w, w, w, need_weights=False)[0])
        w = self.n3(w + self.ffn(w))
        return w.reshape(B, P, T, D)


class PathReader(nn.Module):
    """section 2: waypoint tokens (x, y, heading, step) -> footprint look -> agent look -> self-attention over the 8 -> fact
    heads per waypoint."""

    def __init__(self, cfg, layers=2):
        super().__init__()
        d = cfg.tf_d_model
        self.cfg = cfg
        self.projector = nn.Linear(1024, d)
        self.cams_embeds = nn.Parameter(torch.randn([N_CAMS, d]))
        self.wp_in = nn.Sequential(nn.Linear(4, d), nn.ReLU(inplace=True), nn.Linear(d, d))
        self.step_emb = nn.Embedding(8, d)
        self.layers = nn.ModuleList([PathReaderLayer(d, cfg.tf_d_ffn, cfg) for _ in range(layers)])
        self.yesno = nn.Linear(d, len(YESNO))
        self.numbers = nn.Linear(d, len(NUMBERS))

    def forward(self, paths, tokens, l2i_512, agents):
        B, P, T, _ = paths.shape
        x = tokens
        if not torch.is_autocast_enabled():
            x = x.to(self.projector.weight.dtype)
        feat = self.projector(x) + self.cams_embeds[None, :, None, :].to(x.dtype)
        imf = image_feature(feat, l2i_512, self.cfg.footprint_cams)
        p = paths.float()
        inp = torch.stack([p[..., 0] / 32.0, p[..., 1] / 32.0, torch.cos(p[..., 2]), torch.sin(p[..., 2])], -1)
        x = self.wp_in(inp) + self.step_emb.weight[None, None]
        pts = footprint_points(p.reshape(B, P * T, 3), self.cfg)
        ref_cam, mask = point_sampling(pts, l2i_512[:, list(self.cfg.footprint_cams)])
        for layer in self.layers:
            x = layer(x, p, imf, ref_cam, mask, agents)
        return self.yesno(x).float(), self.numbers(x).float()


# ---------------------------------------------------------------------------------------------------------- teacher forcing
def agents_tf(agents, match, gt, tf, noise=(0.3, math.radians(3.0), 0.5), gen=None):
    """section 3. agents: AgentLook.agents_from_decoder dict (B, Q, ...) (geometry detached; feat keeps its gradient).
    match: (bi, pi, ti) from AgentLoss on the last decoder layer (sample, slot, target). gt: dict agent_box (B, M, 7: x y z l
    w h yaw), agent_vel (B, M, 2), agent_class (B, M). tf: (B,) bool = teacher-forced samples. Matched slots of tf samples:
    centre / length / width from the true box (+ N(0, 0.3 m) on x and y), velocity = true speed + N(0, 0.5 m/s) along the
    true direction rotated by N(0, 3 deg), class probabilities one-hot on the true class, P(none) 0; their other slots
    P(none) 1 and class probabilities 0. Free samples: the decoder's own values. Returns a new dict."""
    if tf is None or not bool(tf.any()):
        return agents
    bi, pi, ti = match
    dev = agents["centre"].device
    out = {k: (v.clone() if k != "feat" else v) for k, v in agents.items()}
    tfm = tf.to(dev)
    out["p_none"] = torch.where(tfm[:, None], torch.ones_like(out["p_none"]), out["p_none"])
    out["probs"] = torch.where(tfm[:, None, None], torch.zeros_like(out["probs"]), out["probs"])
    keep = tfm[bi]
    bi, pi, ti = bi[keep], pi[keep], ti[keep]
    if len(bi) == 0:
        return out
    box = gt["agent_box"].to(dev).float()[bi, ti]
    vel = gt["agent_vel"].to(dev).float()[bi, ti]
    cls = gt["agent_class"].to(dev).long()[bi, ti]
    n = len(bi)
    sp, sh, sv = noise
    r = lambda *s: torch.randn(*s, device=dev, generator=gen)  # noqa: E731
    out["centre"][bi, pi] = box[:, :2] + sp * r(n, 2)
    out["length"][bi, pi] = box[:, 3]
    out["width"][bi, pi] = box[:, 4]
    speed = vel.norm(dim=-1)
    ang = torch.atan2(vel[:, 1], vel[:, 0]) + sh * r(n)
    s2 = (speed + sv * r(n)).clamp(min=0.0)
    out["vel"][bi, pi] = torch.stack([s2 * torch.cos(ang), s2 * torch.sin(ang)], -1)
    out["probs"][bi, pi] = torch.nn.functional.one_hot(cls.clamp(0, 2), 3).float()
    out["p_none"][bi, pi] = 0.0
    return out


# ------------------------------------------------------------------------------------------------------------------- model
class Run3Bv2(nn.Module):
    def __init__(self, vit, reader_layers=2):
        super().__init__()
        self.cfg = run3a_config()
        self.encoder = Encoder(vit)
        self.agent_decoder = AgentDecoder(layers=self.cfg.agent_decoder_layers, queries=self.cfg.agent_decoder_queries)
        self.planner = Planner(self.cfg)
        self.reader = PathReader(self.cfg, reader_layers)
        self.register_buffer("to_512", torch.tensor(TO_512, dtype=torch.float32), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.vit.eval()  # base ViT as at inference (run 3A's cache: training=False); its LoRA still learns
        return self

    def decode(self, tokens, lidar2img):
        """agent decoder: run 3A's call (its own to_512 inside) -> layer outputs, the look's agents (decoder's own)."""
        outs = self.agent_decoder(tokens, lidar2img)
        return outs, AgentLook.agents_from_decoder(outs[-1], outs[-1]["feat"])

    def plan(self, tokens, lidar2img, ego_status, agents):
        return self.planner(tokens, self.to_512 @ lidar2img.float(), ego_status, agents)

    def read(self, paths, tokens, lidar2img, agents):
        return self.reader(paths, tokens, self.to_512 @ lidar2img.float(), agents)
