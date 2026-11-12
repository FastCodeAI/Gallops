"""Build step 3: the memory model (Sid's design, 2026-11-08). Encoder frozen and cached (build 2); generator + scorer from 0.1.8b
pass 0.6 (drive_jepa_018b_p06.ckpt); three new blocks, all 256-d:

  memory M      16 slots x 256, zeros at the start of every window / clip / scene (MemoryModel.init_memory).
  generator read  GenRead: the 256 queries (hist_encoding(ego) + init_feature, drive_jepa_model.py:61-62) cross-attend into M
                  (one attention layer); the output projection is ZERO-initialised and its result is added to the queries, before
                  the first refiner pass (drive_jepa_model.py:64-67).
  scorer read   ScorerRead: a learned query attention-pools M into one 256-d summary. The summary feeds the RATING HEAD only:
                  its first layer (scorer.pred_score.mlp[0], Linear 256 -> 1024, transformer_decoder.py:71) is widened to 512 inputs as
                  W x + W_new s + b with W, b copied from 0.1.8b and W_new (256 new input columns) ZERO-initialised. W x and W_new s are
                  two matmuls whose sum is the widened layer; kept apart so that at step 0 the head is bit-identical to 0.1.8b
                  (a single 512-wide matmul with zero columns rounds differently). Collision and drivable-area heads unchanged, no M.
  update        Update, after the pick: M's slots (each plus a learned slot embedding, see Update) attend to [32 image tokens = the 4 x 8 block means of the 16 x 32 grid (pool_map's
                  blocks without the global mean, gallops memory/probe.py:66-72); motion token = (dx, dy, dheading) since the previous
                  moment in the current frame, embedded; pick token = the scorer's own argmax plan (8 x 3), embedded, DETACHED], then
                  an FFN -> candidate slots; gate per slot g = sigmoid(Linear([old; candidate])); M_new = (1 - g) old + g candidate.
  modes         carry: M_new is handed to the next moment. wiped: M is zeroed before EVERY moment (the trainer / check does it:
                  MemoryModel.step_memory). Same weights, same training; the mode is a flag in the checkpoint (save_checkpoint).

The forward of one moment (forward_moment) is drive_jepa_model.py:50-51, 59-73, 85, 90-91 and scorer.py:36-42 re-done here with the
cached tokens in place of the encoder (the refiner modules, hist_encoding, init_feature and the scorer heads are the 0.1.8b modules
themselves, called as the model calls them), plus the three blocks. Replay for the scorer lessons = pai_model.Model.replay
(pai_model.py:194-201): the refiner with its weights detached from the same ego/query init and the same four proposal lists; here also
the generator read with ITS weights detached and M detached, so scorer lessons reach M only through the scorer's summary (module
docstring of s6_train.py explains the choice). Model loading = s4_cache_sample.py:30-38 verbatim (strict load of p06, 0.1.7's batch
fix on, 495/495 tensors equal).

VERSION 2 (build step 4, Sid 2026-11-11, after the pilot's memory collapse: PILOT_SEED0.md flag 2): behind `version` so the pilot's
checkpoints (version 1, no "version" key) still load and evaluate with the old behaviour.
  start state   M starts every window / clip / scene from M0, a learned 16 x 256 parameter (StartState) instead of zeros; initialised as 16
                ORTHOGONAL rows (torch.nn.init.orthogonal_, seeded) scaled by sqrt(256) = 16 per row: the written candidate rows are
                LayerNorm outputs (unit variance per element, row norm ~16; the pilot's untrained M read had norm 63 over 16 slots ~ 16 per
                slot), so M0 rows and written rows have the same magnitude and the gate's Linear([old; candidate]) sees inputs on the
                pilot's scale; pairwise cosine between the 16 rows is exactly 0 at the start. "wiped" = M reset to M0 before EVERY moment.
  gate spread   each slot has its own gate bias, initialised to logit(p_i) with p_i spread evenly from 0.9 (slot 1) to 0.1 (slot 16); the
                gate's Linear weight is initialised N(0, 1e-3^2) (pre-activation std ~ sqrt(512) x 1e-3 ~ 0.02 on unit-scale inputs), so the
                gate starts at its bias; weight and biases stay learnable. The slot identity on the update query (slot_emb) is kept.
  step 0        the generator read's output projection and the rating head's new columns are zero, so M0 changes nothing at step 0 (T1).

VERSION 3 (build step 5, Sid 2026-11-11: the memory's own lesson, BUILD5_DESIGN_NOTE.md) = version 2 + a TRAINING-ONLY read-out (ReadOut):
  from the memory written at moment t (o["M_new"], after the update) it predicts the 32 block means (4 x 8, 256-d) of the moment 2 s ago
  (t - 4) and 4 s ago (t - 8), standardised per (position, dimension) with the training slice's statistics (buffers t_mu / t_sd, saved with
  the checkpoint). 32 questions per target = position embedding + lag embedding + Linear of the CURRENT standardised block mean at that
  position (a cached constant); one attention layer into M; LayerNorm; Linear(256 -> 256) ZERO-initialised (the lesson starts at "predict the
  average"). The read-out feeds nothing: forward_moment's plans, logits and picks are version 2's. The lesson (s6_train.recall_lesson) reaches
  the read-out, the update block (gate included) and M0 only: M_new = update(M_old, block means (constants), logged motion, pick DETACHED) and
  M_old is the previous M_new or M0, so no path leads to the generator, the scorer, the generator read or the scorer summary (asserted in
  s6_train.recall_lesson on the first training step of every process). export_state() drops the read-out; the checkpoint records it."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pai_common as C
import pai_model as PM

P06 = Path("{runs}/generator_joint_pai/018b/release/drive_jepa_018b_p06.ckpt")
P06_SHA = "3d566eb211b5ce0f556680196a46fc0a74da29091ae62ac588910a48d62b8e66"
N_SLOTS, D, HEADS, FFN = 16, 256, 4, 1024
GRID_H, GRID_W, POOL_H, POOL_W = 16, 32, 4, 8                 # probe.py:51-52
N_IMG = POOL_H * POOL_W                                        # 32 image tokens
MODES = ("carry", "wiped")
VERSIONS = (1, 2, 3); RECALL_LAGS = (4, 8)                      # version 3: targets 2 s and 4 s back (moments are 0.5 s apart)
TARGET_STATS = Path(__file__).resolve().parents[1] / "out" / "recall_target_stats.pt"     # src/s9_target_stats.py
GATE_P_FIRST, GATE_P_LAST, GATE_W_STD = 0.9, 0.1, 1e-3    # version 2 gate start (module docstring)


def load_agent():
    """s4_cache_sample.py:30-38 verbatim: 0.1.8b-p06 strict through 0.1.7's agent class, batch fix on, eval; 495/495 tensors re-checked."""
    assert C.sha256_file(P06) == P06_SHA, "p06 sha"
    from drive_jepa_agent_s10 import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    import batch_fix as BF
    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path="", vocab_path=str(C.VOCAB), index_kind="ext",
                           init_checkpoint=str(P06), freeze_encoder=True, scorer_impl="s3", pool_workers=0)
    agent.initialize(); found = BF.install(agent._pad_model); BF.set_fix(True, True, True); agent = agent.cuda().eval()
    sd = torch.load(P06, map_location="cpu", weights_only=False)["state_dict"]; cur = agent.state_dict()
    assert all(torch.equal(cur[k.replace("agent._pad_model", "_pad_model")].cpu(), v) for k, v in sd.items()) and len(sd) == 495; del sd
    for p in agent.parameters():
        p.requires_grad_(False)
    return agent, {"init_checkpoint": str(P06), "sha256": P06_SHA, "tensors": 495, "loaded_equal": "495/495", "batch_fix": {k: len(v) for k, v in found.items()}}


class Attn(nn.Module):
    """Plain multi-head attention, 256-d, 4 heads; separate q / k / v / o projections so the output projection can be zero-initialised."""

    def __init__(self, zero_out: bool = False):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))
        if zero_out:
            nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)

    def forward(self, q_in, kv_in):
        B, Nq, _ = q_in.shape; Nk = kv_in.shape[1]; dh = D // HEADS
        q = self.q(q_in).view(B, Nq, HEADS, dh).transpose(1, 2)
        k = self.k(kv_in).view(B, Nk, HEADS, dh).transpose(1, 2)
        v = self.v(kv_in).view(B, Nk, HEADS, dh).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, Nq, D))


class GenRead(nn.Module):
    """queries (B, 256, D) cross-attend into M (B, 16, D); returns the delta to ADD to the queries (zero at init)."""

    def __init__(self):
        super().__init__(); self.attn = Attn(zero_out=True)

    def forward(self, queries, M):
        return self.attn(queries, M)


class ScorerRead(nn.Module):
    """a learned query attention-pools M into one 256-d summary (B, D); mem_in = the rating head's 256 new input columns (zero at init)."""

    def __init__(self):
        super().__init__(); self.query = nn.Parameter(torch.randn(1, 1, D) * 0.02); self.attn = Attn()
        self.mem_in = nn.Linear(D, FFN, bias=False); nn.init.zeros_(self.mem_in.weight)

    def forward(self, M):
        return self.attn(self.query.expand(M.shape[0], -1, -1), M)[:, 0]


class Update(nn.Module):
    """old M (B, 16, D), img32 (B, 32, D), motion (B, 3) = (dx, dy, dheading) in the current frame, pick (B, 8, 3) detached -> (M_new, gate)."""

    def __init__(self, version: int = 1):
        super().__init__(); self.version = version
        self.img_in = nn.Linear(D, D); self.mot_in = nn.Linear(4, D); self.pick_in = nn.Linear(24, D)
        self.img_pos = nn.Parameter(torch.randn(1, N_IMG, D) * 0.02); self.type_emb = nn.Parameter(torch.randn(3, D) * 0.02)
        # slot identity (added to the attention query only): M starts at zeros, so without it all 16 slots form the same query, get the same
        # candidate and the same gate, and stay identical forever (seen in the first smoke run: gate mean equal on all 16 slots). +4,096 params.
        self.slot_emb = nn.Parameter(torch.randn(1, N_SLOTS, D) * 0.02)
        self.attn = Attn(); self.ln1 = nn.LayerNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D, FFN), nn.ReLU(), nn.Linear(FFN, D)); self.ln2 = nn.LayerNorm(D)
        if version == 1:
            self.gate = nn.Linear(2 * D, 1)
        else:                                                   # version 2: per-slot gate bias, spread 0.9 -> 0.1; small weight (module docstring)
            self.gate = nn.Linear(2 * D, 1, bias=False); nn.init.normal_(self.gate.weight, std=GATE_W_STD)
            p = torch.linspace(GATE_P_FIRST, GATE_P_LAST, N_SLOTS); self.gate_bias = nn.Parameter(torch.log(p / (1 - p)))

    def forward(self, M_old, img32, motion, pick):
        mot = torch.stack([motion[:, 0], motion[:, 1], torch.cos(motion[:, 2]), torch.sin(motion[:, 2])], -1)
        kv = torch.cat([self.img_in(img32) + self.img_pos + self.type_emb[0], (self.mot_in(mot) + self.type_emb[1])[:, None],
                        (self.pick_in(pick.reshape(pick.shape[0], -1)) + self.type_emb[2])[:, None]], 1)       # (B, 34, D)
        x = self.ln1(M_old + self.attn(M_old + self.slot_emb, kv))
        cand = self.ln2(x + self.ffn(x))
        z = self.gate(torch.cat([M_old, cand], -1))                                                             # (B, 16, 1)
        g = torch.sigmoid(z if self.version == 1 else z + self.gate_bias[None, :, None])
        return (1 - g) * M_old + g * cand, g[..., 0]


class StartState(nn.Module):
    """version 2: the learned start state M0 (16, 256), 16 orthogonal rows of norm sqrt(256) = 16 (module docstring)."""

    def __init__(self):
        super().__init__()
        M0 = torch.empty(N_SLOTS, D); nn.init.orthogonal_(M0)       # rows orthonormal (16 < 256)
        self.M0 = nn.Parameter(M0 * math.sqrt(D))


class ReadOut(nn.Module):
    """version 3, training-only (module docstring). M (B, 16, D) the memory WRITTEN at the moment, z_now (B, 32, D) the standardised current
    block means, lag_idx (B,) long (0 = 2 s, 1 = 4 s) -> the predicted standardised block means of that earlier moment (B, 32, D)."""

    def __init__(self, stats: dict | None = None):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, N_IMG, D) * 0.02); self.lag = nn.Parameter(torch.randn(len(RECALL_LAGS), D) * 0.02); self.now = nn.Linear(D, D)
        self.attn = Attn(); self.ln = nn.LayerNorm(D); self.out = nn.Linear(D, D)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.register_buffer("t_mu", torch.zeros(1, N_IMG, D)); self.register_buffer("t_sd", torch.ones(1, N_IMG, D))
        if stats is not None:
            self.t_mu.copy_(stats["mu"].reshape(1, N_IMG, D).float()); self.t_sd.copy_(stats["sd"].reshape(1, N_IMG, D).float())
        self.stats_info = {k: v for k, v in (stats or {}).items() if k not in ("mu", "sd")}

    def standardise(self, bm):
        return (bm.float() - self.t_mu) / self.t_sd

    def forward(self, M, z_now, lag_idx):
        q = self.pos + self.lag[lag_idx][:, None] + self.now(z_now)
        return self.out(self.ln(q + self.attn(q, M)))


def block_means(tokens):
    """(B, 512, D) tokens of the 16 x 32 grid -> (B, 32, D) 4 x 8 block means = probe.pool_map's blocks (probe.py:66-72) without the global mean."""
    B = tokens.shape[0]
    return tokens.reshape(B, POOL_H, GRID_H // POOL_H, POOL_W, GRID_W // POOL_W, D).mean((2, 4)).reshape(B, N_IMG, D)


def motion_from_prev_future(prev_traj):
    """(B, 8, 3) human future of the PREVIOUS moment -> (B, 3) car motion since then in the CURRENT frame. Waypoint 0 (t + 0.5 s,
    the current anchor) of the previous moment's future is the current pose in the previous frame (x1, y1, h1); the displacement
    expressed in the current frame is R(-h1) (x1, y1), the heading change is h1."""
    x1, y1, h1 = prev_traj[:, 0, 0], prev_traj[:, 0, 1], prev_traj[:, 0, 2]
    c, s = torch.cos(h1), torch.sin(h1)
    return torch.stack([c * x1 + s * y1, -s * x1 + c * y1, h1], -1)


def to_current_frame(prev_poses, prev_traj):
    """poses (B, K, 3) given in the previous frame -> the current frame, using the previous future's waypoint 0 (x1, y1, h1) as above."""
    x1, y1, h1 = prev_traj[:, 0:1, 0], prev_traj[:, 0:1, 1], prev_traj[:, 0:1, 2]
    c, s = torch.cos(h1), torch.sin(h1)
    dx, dy = prev_poses[..., 0] - x1, prev_poses[..., 1] - y1
    return torch.stack([c * dx + s * dy, -s * dx + c * dy, prev_poses[..., 2] - h1], -1)


class MemoryModel(nn.Module):
    def __init__(self, agent, mode: str, version: int = 1):
        super().__init__()
        assert mode in MODES and version in VERSIONS, (mode, version)
        self.agent = agent; self.m = agent._pad_model; self.cfg = agent._config; self.mode = mode; self.version = version
        self.gen_read, self.sc_read, self.update = GenRead(), ScorerRead(), Update(version)
        self.start = StartState() if version >= 2 else None
        if version >= 3:                                       # built LAST so the version-2 blocks draw the same random numbers as in version 2
            assert TARGET_STATS.exists(), f"{TARGET_STATS} missing: run src/s9_target_stats.py"
            self.read_out = ReadOut(torch.load(TARGET_STATS, map_location="cpu", weights_only=False))
        else:
            self.read_out = None
        self.register_buffer("spatial", torch.tensor([[GRID_H, GRID_W]], dtype=torch.long), persistent=False)
        self.register_buffer("lsi", torch.tensor([0], dtype=torch.long), persistent=False)
        heads = self.m._trajectory_head; assert len(heads) == 4 and len({id(h) for h in heads}) == 1
        self.bev_refiner = heads[0].Bev_refiner
        gen = [(n, p) for n, p in self.m.named_parameters() if n.startswith(PM.GEN_PREFIXES)]
        self.gen_params = [(n, p) for n, p in gen if not n.endswith(PM.FIXED_SUFFIXES)]
        assert sum(p.numel() for _, p in self.gen_params) == PM.GEN_VALUES and len(self.gen_params) == PM.N_GEN       # pai_model.py:30, 117
        self.score_params = [(n, p) for n, p in self.m.named_parameters() if n.startswith("scorer.")]; assert len(self.score_params) == 30
        for _, p in self.gen_params + self.score_params:
            p.requires_grad_(True)
        # parameter groups for routing (T2) and the optimisers
        self.groups = {"generator": [p for _, p in self.gen_params],
                       "generator_lookup": list(self.gen_read.parameters()),
                       "rating_head": [p for n, p in self.score_params if n.startswith("scorer.pred_score.")] + [self.sc_read.mem_in.weight],
                       "scorer_summary": [self.sc_read.query] + list(self.sc_read.attn.parameters()),
                       "collision_head": [p for n, p in self.score_params if n.startswith("scorer.pred_col_agent.")],
                       "drivable_head": [p for n, p in self.score_params if n.startswith("scorer.pred_area.")],
                       "update_block": list(self.update.parameters())}
        if version >= 2:
            self.groups["start_state"] = [self.start.M0]
        if version >= 3:
            self.groups["read_out"] = list(self.read_out.parameters())
        assert len(self.groups["rating_head"]) == 11 and len(self.groups["collision_head"]) == 10 and len(self.groups["drivable_head"]) == 10
        self.memory_params = self.groups["generator_lookup"] + [self.sc_read.mem_in.weight] + self.groups["scorer_summary"] + self.groups["update_block"] + self.groups.get("start_state", []) + self.groups.get("read_out", [])

    # ---- sizes --------------------------------------------------------------------------------------------------------
    def new_param_count(self) -> dict:
        n = lambda ps: int(sum(p.numel() for p in ps))
        out = {"generator_read": n(self.gen_read.parameters()), "scorer_read_summary": n(self.groups["scorer_summary"]),
               "rating_head_new_columns": int(self.sc_read.mem_in.weight.numel()), "update_block": n(self.update.parameters())}
        if self.version >= 2:
            out["start_state"] = int(self.start.M0.numel()); out["update_block_gate_biases"] = int(self.update.gate_bias.numel())
            out["total_new"] = out["generator_read"] + out["scorer_read_summary"] + out["rating_head_new_columns"] + out["update_block"] + out["start_state"]
            if self.version >= 3:
                out["read_out_training_only"] = n(self.read_out.parameters()); out["total_new_exported"] = out["total_new"]
                out["total_new"] = out["total_new"] + out["read_out_training_only"]
        else:
            out["total_new"] = sum(out.values())
        out["memory_state_values"] = N_SLOTS * D; out["version"] = self.version
        return out

    # ---- state --------------------------------------------------------------------------------------------------------
    def init_memory(self, B: int, device="cuda"):
        """the memory at the start of a window / clip / scene: zeros (version 1) or the learned M0 (version 2, attached so it trains)."""
        if self.version == 1:
            return torch.zeros(B, N_SLOTS, D, device=device, dtype=torch.float32)
        return self.start.M0.to(device).float()[None].expand(B, -1, -1)

    def step_memory(self, M_new):
        """the memory handed to the next moment: M_new in carry mode; the start state (reset before EVERY moment) in wiped mode."""
        return M_new.float() if self.mode == "carry" else self.init_memory(M_new.shape[0], M_new.device)

    # ---- the pieces of the forward ----------------------------------------------------------------------------------
    def image_feature(self, tokens, lidar2img, img_shape):
        """cached tokens (B, 512, D) -> what ImgEncoder.forward returns (simple_image_encoder.py:64-66) for the refiner:
        (feat_flatten (num_cam=1, 512, B, D), spatial_shape, level_start_index, kwargs) with kwargs["img_metas"] holding lidar2img
        sliced as drive_jepa_model.py:50 does and img_shape (encoder.py:123, 141-142)."""
        feat = tokens.permute(1, 0, 2)[None]
        return (feat, self.spatial, self.lsi, {"img_metas": {"lidar2img": lidar2img[:, 1:2], "img_shape": img_shape}})

    def rating(self, bev, proposals, summary):
        """scorer.py:41-42 with the widened first layer (module docstring): the last logit is the rating (drive_jepa_model.py:85)."""
        B, P, T = proposals.shape[:3]
        pf = bev.reshape(B, P, T, -1).amax(-2)
        mlp = self.m.scorer.pred_score.mlp
        h = mlp[0](pf) + self.sc_read.mem_in(summary)[:, None, :]
        for layer in list(mlp)[1:]:
            h = layer(h)
        return h.reshape(B, -1, self.m.scorer.score_num)

    def queries(self, ego):
        return self.m.hist_encoding(ego)[:, None] + self.m.init_feature.weight[None]                     # drive_jepa_model.py:61-62

    @torch.no_grad()
    def replay(self, ego, image_feature, proposal_list, M):
        """pai_model.py:194-201: the refiner from the same init and the same proposal lists with everything detached (the image tokens
        are constants here; the look-up's weights and M are detached too), so the scorer lessons see a bev that carries no gradient."""
        bev = self.queries(ego)
        bev = bev + self.gen_read(bev, M)
        for prop in proposal_list:
            bev = self.bev_refiner(prop.detach(), bev, image_feature)
        return bev

    def forward_moment(self, tokens, ego, lidar2img, img_shape, M, motion, replay: bool) -> dict:
        """one moment for B rows. tokens (B, 512, D) fp32 cached; ego (B, 11) newest row, acceleration zeroed; M (B, 16, D) the memory
        read at this moment (already zeroed by the caller in wiped mode); motion (B, 3) since the previous moment (zeros at the first)."""
        B = ego.shape[0]
        img_f = self.image_feature(tokens, lidar2img, img_shape)
        bev = self.queries(ego)
        bev = bev + self.gen_read(bev, M)                                                                 # the generator read
        proposal_list = []
        for refine in self.m._trajectory_head:                                                            # drive_jepa_model.py:64-67
            bev, proposals = refine(bev, img_f); proposal_list.append(proposals)
        proposals = proposal_list[-1]
        summary = self.sc_read(M)                                                                         # the scorer read
        pred_logit = self.rating(bev, proposals, summary)
        pdm_score = torch.sigmoid(pred_logit)[:, :, -1]                                                   # :85
        chosen = torch.argmax(pdm_score, dim=1)                                                           # :90 (no previous-plan re-score)
        picked = proposals[torch.arange(B, device=proposals.device), chosen].detach()                     # :91, DETACHED for the update
        out = {"proposals": proposals, "proposal_list": proposal_list, "pred_logit": pred_logit, "pdm_score": pdm_score, "chosen": chosen,
               "picked": picked, "bev": bev, "summary": summary, "M_read": M}
        if replay:
            bev_r = self.replay(ego, img_f, proposal_list, M)
            out["logit"] = self.rating(bev_r, proposals.detach(), summary)                               # summary attached: scorer lessons -> summary, update block
            pf = bev_r.reshape(B, proposals.shape[1], proposals.shape[2], -1).amax(-2)
            out["agents_state"] = self.m.scorer.pred_col_agent(pf).reshape(B, proposals.shape[1], proposals.shape[2], -1, 2, 9)   # scorer.py:50-51
            out["area_logit"] = self.m.scorer.pred_area(bev_r)                                                                     # scorer.py:47-48
            out["bev_replay"] = bev_r
        bm = block_means(tokens)                                                                          # (B, 32, D) constants: the update's image input and the recall targets
        M_new, gate = self.update(M, bm, motion, picked)                                                   # the update, after the pick
        out.update(M_new=M_new, gate=gate, block_means=bm)
        return out

    # ---- optimisers and checkpoints --------------------------------------------------------------------------------------
    def optimisers(self):
        """generator Adam 3e-5 and scorer Adam 1e-4 as 0.1.8b (pai_model.py:233-235); memory blocks Adam 1e-4."""
        return [torch.optim.Adam([p for _, p in self.gen_params], lr=3e-5, weight_decay=0),
                torch.optim.Adam([p for _, p in self.score_params], lr=1e-4, weight_decay=0),
                torch.optim.Adam(self.memory_params, lr=1e-4, weight_decay=0)]

    def trainable_state(self) -> dict:
        g = lambda ps: {n: p.detach().cpu().clone() for n, p in ps}
        return {"generator": g(self.gen_params), "scorer": g(self.score_params),
                "memory": {k: v.detach().cpu().clone() for k, v in {**{"gen_read." + n: p for n, p in self.gen_read.state_dict().items()},
                                                                      **{"sc_read." + n: p for n, p in self.sc_read.state_dict().items()},
                                                                      **{"update." + n: p for n, p in self.update.state_dict().items()},
                                                                      **({"start." + n: p for n, p in self.start.state_dict().items()} if self.version >= 2 else {}),
                                                                      **({"read_out." + n: p for n, p in self.read_out.state_dict().items()} if self.version >= 3 else {})}.items()}}

    def export_state(self) -> dict:
        """what a shipped model carries: everything trainable EXCEPT the training-only read-out (version 3)."""
        st = self.trainable_state(); st["memory"] = {k: v for k, v in st["memory"].items() if not k.startswith("read_out.")}
        return st

    def load_trainable(self, s: dict) -> None:
        with torch.no_grad():
            for key, ps in (("generator", self.gen_params), ("scorer", self.score_params)):
                cur = dict(ps); assert set(s[key]) == set(cur)
                for n, t in s[key].items():
                    cur[n].copy_(t.to(cur[n].device))
            for name, mod in (("gen_read", self.gen_read), ("sc_read", self.sc_read), ("update", self.update)) + ((("start", self.start),) if self.version >= 2 else ()) + ((("read_out", self.read_out),) if self.version >= 3 else ()):
                mod.load_state_dict({k[len(name) + 1:]: v for k, v in s["memory"].items() if k.startswith(name + ".")}, strict=True)

    def save_checkpoint(self, path: Path, meta: dict) -> None:
        """weights + the mode flag (never chosen by hand later: the loader reads it) + provenance; checkpoint.json beside it."""
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"mode": self.mode, "version": self.version, "start": {"checkpoint": str(P06), "sha256": P06_SHA}, "new_params": self.new_param_count(),
               "saved": time.strftime("%F %T"), **meta}
        if self.version >= 3:
            rec["read_out"] = {"training_only": True, "in_exported_model": False, "lags_moments": list(RECALL_LAGS), "target_stats": self.read_out.stats_info,
                               "note": "weights and the target statistics (t_mu, t_sd) are under memory['read_out.*']; export_state() drops them"}
        torch.save({**self.trainable_state(), **rec}, path)
        (path.with_suffix(".json")).write_text(json.dumps(rec, indent=1, default=str) + "\n")

    @staticmethod
    def load_checkpoint(agent, path: Path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = MemoryModel(agent, ck["mode"], ck.get("version", 1)).cuda(); model.load_trainable(ck)      # no "version" key = the pilot's version 1
        return model, {k: v for k, v in ck.items() if k not in ("generator", "scorer", "memory")}
