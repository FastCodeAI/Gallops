"""Approach 4 model (build 2, step 3): 0.1.8b-p06 wrapped, not edited, plus a motion encoder for the car-state history (PREREG_APPROACH4.md §1,
Amendment 1; Sid 2026-11-17: two-piece widened layer, the p06-wrapping plan, Model018c.scorer_lessons reused).

Start: p06 loaded strictly through 0.1.7's agent class with 0.1.7's batch fix on (approach 3's s6_model.load_agent, imported: 495/495 tensors
equal). The image encoder is never called: its tokens come from the cache (frozen, eval, GridMask off when cached).

New, all 256-d (a-e of the brief):
  a. motion encoder  history (8 x 6 = 48, oldest first, rows [x, y, heading, speed, turn_rate, valid]) -> Linear 256 -> GELU -> Linear 256
  b. no_history      Parameter (256), zeros
  c. hiding          per moment, with probability hide_prob (arm A 0.5, arm B 1.0), the summary is replaced by no_history. Training only,
                     drawn from the model's own torch.Generator seeded from the training seed (independent of every other random stream).
                     Evaluation: arm A never hides, arm B always hides.
  d. link 1          Linear(256 -> 256), weight and bias zero, its output ADDED to the 256 plan starting points
                     (UP/drive_jepa_model.py:61-62: hist_encoding(ego)[:, None] + init_feature.weight[None]) and to the scorer replay's copy
                     of the same sum (pai_model.Model.replay, pai_model.py:186-187, which builds it under torch.no_grad: so link 1's term in
                     the replay carries no gradient either, and the scorer lessons reach the motion encoder only through link 2)
  e. link 2          the rating head's first Linear (scorer.pred_score.mlp[0], 256 -> 1024, UP/score_module/scorer.py:41-42 via
                     transformer_decoder.MLP) widened to 512 inputs: p06's W (columns 0-255) and b unchanged, new columns 256-511 = W_new
                     (1024 x 256, zero). Computed as W x + b + W_new s (two matmuls summed; Sid's choice: a single 512-wide matmul with
                     zero columns rounds differently, approach 3 s6_model docstring); 6 outputs per plan unchanged. Collision and
                     drivable-area heads read the plan feature only, as in p06.
  f. everything else is p06's module and weights, called as the model calls them; the comfort re-score (drive_jepa_model.py:87-88,
     calibrate_score) is outside this forward and untouched: training and checks pass no previous plan, as 0.1.8b's did.
Trainable (Amendment 1): motion encoder, no_history, link 1, link 2 (W_new), the generator (41 tensors, pai_model.GEN_PREFIXES minus
the fixed reference buffers) and all 30 scorer tensors. Everything else, the encoder included, has requires_grad False.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

import pai_model as PM

D, HIST_ROWS, HIST_COLS = 256, 8, 6
ARMS = {"A": 0.5, "B": 1.0}


def load_agent():
    """p06, strict, batch fix on, eval, all frozen (approach 3 s6_model.load_agent, imported unchanged)."""
    import s6_model as SM
    return SM.load_agent()


class A4Model(nn.Module):
    w_human, w_pull = 0.5, 2.0                    # 0.1.8b attempt 2 (pai_model_c.Model018c; run.json settings w_human 0.5, w_pull_away 2.0)
    hard_rule = True

    def __init__(self, agent, arm: str, seed: int):
        super().__init__()
        assert arm in ARMS
        self.arm, self.hide_prob, self.seed = arm, ARMS[arm], seed
        self.agent = agent; self.m = agent._pad_model; self.cfg = agent._config
        dev = next(self.m.parameters()).device
        torch.manual_seed(seed)                   # the motion encoder's default init; link weights are zero regardless
        self.motion = nn.Sequential(nn.Linear(HIST_ROWS * HIST_COLS, D), nn.GELU(), nn.Linear(D, D))
        self.no_history = nn.Parameter(torch.zeros(D))
        self.link1 = nn.Linear(D, D)
        nn.init.zeros_(self.link1.weight); nn.init.zeros_(self.link1.bias)
        mlp0 = self.m.scorer.pred_score.mlp[0]
        assert isinstance(mlp0, nn.Linear) and mlp0.in_features == D and mlp0.out_features == self.cfg.tf_d_ffn
        self.link2 = nn.Linear(D, mlp0.out_features, bias=False)     # W_new: columns 256-511 of the widened layer
        nn.init.zeros_(self.link2.weight)
        self.to(dev)
        self.hide_gen = torch.Generator(device="cpu").manual_seed(seed + 4_000_000)
        self.register_buffer("spatial", torch.tensor([[16, 32]], dtype=torch.long, device=dev), persistent=False)
        self.register_buffer("lsi", torch.tensor([0], dtype=torch.long, device=dev), persistent=False)
        heads = self.m._trajectory_head
        assert len(heads) == 4 and len({id(h) for h in heads}) == 1
        self.bev_refiner = heads[0].Bev_refiner
        for p in self.m.parameters():
            p.requires_grad_(False)
        gen = [(n, p) for n, p in self.m.named_parameters() if n.startswith(PM.GEN_PREFIXES)]
        self.gen_params = [(n, p) for n, p in gen if not n.endswith(PM.FIXED_SUFFIXES)]
        assert len(self.gen_params) == PM.N_GEN and sum(p.numel() for _, p in self.gen_params) == PM.GEN_VALUES
        self.score_params = [(n, p) for n, p in self.m.named_parameters() if n.startswith("scorer.")]
        assert len(self.score_params) == 30
        for _, p in self.gen_params + self.score_params:
            p.requires_grad_(True)
        self.new_params = [("motion." + n, p) for n, p in self.motion.named_parameters()] + [("no_history", self.no_history)] + \
                          [("link1." + n, p) for n, p in self.link1.named_parameters()] + [("link2.weight", self.link2.weight)]
        self.heads = {"rating": [p for n, p in self.score_params if n.startswith("scorer.pred_score.")]}
        assert len(self.heads["rating"]) == 10

    # ---- p06 start --------------------------------------------------------------------------------------------------------------
    def load_from_p06(self) -> dict:
        """every p06 tensor present in the wrapped network and identical to the file; the new parameters listed with their count."""
        import s6_model as SM
        import pai_common as C
        assert C.sha256_file(SM.P06) == SM.P06_SHA
        sd = torch.load(SM.P06, map_location="cpu", weights_only=False)["state_dict"]
        cur = self.agent.state_dict()
        keys = [k.replace("agent._pad_model", "_pad_model") for k in sd]
        missing = [k for k in keys if k not in cur]
        assert not missing, missing[:5]
        same = sum(int(torch.equal(cur[k2].cpu(), v)) for k2, v in zip(keys, sd.values()))
        assert same == len(sd) == 495, f"{same}/{len(sd)}"
        new = {n: int(p.numel()) for n, p in self.new_params}
        return {"p06": str(SM.P06), "sha256": SM.P06_SHA, "p06_tensors_present_and_equal": f"{same}/{len(sd)}",
                "new_parameters": new, "new_parameter_count": int(sum(new.values())),
                "trainable_count": int(sum(p.numel() for p in self.trainable())),
                "frozen_count": int(sum(p.numel() for p in self.m.parameters() if not p.requires_grad))}

    def trainable(self) -> list:
        return [p for _, p in self.gen_params + self.score_params + self.new_params]

    # ---- pieces -----------------------------------------------------------------------------------------------------------------
    def summary(self, history, training: bool):
        """(B, 8, 6) -> (B, 256) and the hide mask (B,) bool."""
        B = history.shape[0]
        z = self.motion(history.reshape(B, -1).to(self.no_history.dtype))
        if training:
            hide = torch.rand(B, generator=self.hide_gen) < self.hide_prob
        else:
            hide = torch.full((B,), self.arm == "B")
        hide = hide.to(z.device)
        z = torch.where(hide[:, None], self.no_history[None].expand_as(z).to(z.dtype), z)
        return z, hide

    def image_feature(self, tokens, lidar2img, img_shape):
        """cached tokens (B, 512, D) -> what ImgEncoder.forward returns (approach 3 s6_model.image_feature, same reasoning)."""
        feat = tokens.permute(1, 0, 2)[None]
        return (feat, self.spatial, self.lsi, {"img_metas": {"lidar2img": lidar2img[:, 1:2], "img_shape": img_shape}})

    def starts(self, ego, z):
        """drive_jepa_model.py:61-62 plus link 1: (hist_encoding(ego)[:, None] + init_feature.weight[None]) + link1(z)[:, None]."""
        bev = self.m.hist_encoding(ego)[:, None] + self.m.init_feature.weight[None]
        return bev + self.link1(z)[:, None]

    def rating(self, bev, proposals, z):
        """scorer.py:41-42 with the widened first layer: mlp[0](pf) = W pf + b, then + W_new z; the rest of the MLP unchanged."""
        B, P, T = proposals.shape[:3]
        pf = bev.reshape(B, P, T, -1).amax(-2)
        mlp = self.m.scorer.pred_score.mlp
        h = mlp[0](pf) + self.link2(z)[:, None, :]
        for layer in list(mlp)[1:]:
            h = layer(h)
        return h.reshape(B, -1, self.m.scorer.score_num), pf

    @torch.no_grad()
    def replay(self, ego, z, image_feature, proposal_list):
        """pai_model.Model.replay (:182-191): the refiner from the same starting points (here with link 1's term) over the same four
        proposal lists, its weights detached; image features are constants here, so no_grad is the same thing."""
        bev = self.starts(ego, z)
        for prop in proposal_list:
            bev = self.bev_refiner(prop.detach(), bev, image_feature)
        return bev

    # ---- forward ----------------------------------------------------------------------------------------------------------------
    def forward_moment(self, tokens, ego, lidar2img, img_shape, history, training: bool, replay: bool) -> dict:
        """drive_jepa_model.py:50-51, 59-73, 85, 90-91 with cached tokens, the history summary and the two links."""
        B = ego.shape[0]
        img_f = self.image_feature(tokens, lidar2img, img_shape)
        z, hide = self.summary(history, training)
        bev = self.starts(ego, z)
        proposal_list = []
        for refine in self.m._trajectory_head:
            bev, proposals = refine(bev, img_f); proposal_list.append(proposals)
        proposals = proposal_list[-1]
        pred_logit, _ = self.rating(bev, proposals, z)
        pdm_score = torch.sigmoid(pred_logit)[:, :, -1]
        chosen = torch.argmax(pdm_score, dim=1)
        out = {"proposals": proposals, "proposal_list": proposal_list, "pred_logit": pred_logit, "pdm_score": pdm_score, "chosen": chosen,
               "hide": hide, "z": z}
        if replay:
            bev_r = self.replay(ego, z, img_f, proposal_list)
            logit, pf = self.rating(bev_r, proposals.detach(), z)                          # z attached: link 2 and the motion encoder
            out["logit"] = logit
            out["agents_state"] = self.m.scorer.pred_col_agent(pf).reshape(B, proposals.shape[1], proposals.shape[2], -1, 2, 9)  # scorer.py:50-51
            out["area_logit"] = self.m.scorer.pred_area(bev_r)                                                                  # scorer.py:47-48
            out["bev_replay"] = bev_r
        return out

    def train_mode(self) -> None:
        """pai_model.Model.train_mode (:177-183) minus the encoder pieces: whole network eval; hist_encoding, init_feature and the
        trajectory head train; the new blocks train (no dropout or norm in them, so the mode changes nothing)."""
        self.m.eval()
        for mod in (self.m.hist_encoding, self.m.init_feature, self.m._trajectory_head):
            mod.train()
        for mod in (self.motion, self.link1, self.link2):
            mod.train()

    def eval_mode(self) -> None:
        self.m.eval()
        for mod in (self.motion, self.link1, self.link2):
            mod.eval()

    def scorer_lessons(self, r, targets, graded):
        """pai_model_c.Model018c.scorer_lessons (:44-87), called unchanged on this wrapper (it reads w_human, w_pull, cfg, agent)."""
        import pai_model_c as MC
        return MC.Model018c.scorer_lessons(self, r, targets, graded)

    def optimisers(self) -> list:
        """generator Adam 3e-5 and scorer Adam 1e-4 (pai_model.py:233-235, the 30 scorer tensors as in pai_model_b); the new blocks Adam
        1e-4 (not in 0.1.8b: the scorer's rate, as approach 3 gave its new blocks). No encoder optimiser."""
        return [torch.optim.Adam([p for _, p in self.gen_params], lr=3e-5, weight_decay=0),
                torch.optim.Adam([p for _, p in self.score_params], lr=1e-4, weight_decay=0),
                torch.optim.Adam([p for _, p in self.new_params], lr=1e-4, weight_decay=0)]

    def zero_grad_all(self) -> None:
        for p in self.trainable():
            p.grad = None

    # ---- state ------------------------------------------------------------------------------------------------------------------
    def trainable_state(self) -> dict:
        g = lambda ps: {n: p.detach().cpu().clone() for n, p in ps}
        return {"generator": g(self.gen_params), "scorer": g(self.score_params), "new": g(self.new_params),
                "hide_gen_state": self.hide_gen.get_state().clone()}

    def load_trainable(self, s: dict) -> None:
        with torch.no_grad():
            for key, ps in (("generator", self.gen_params), ("scorer", self.score_params), ("new", self.new_params)):
                cur = dict(ps); assert set(s[key]) == set(cur)
                for n, t in s[key].items():
                    cur[n].copy_(t.to(cur[n].device))
        if "hide_gen_state" in s:
            self.hide_gen.set_state(s["hide_gen_state"])


def snapshot(model: A4Model) -> dict:
    return copy.deepcopy(model.trainable_state())
