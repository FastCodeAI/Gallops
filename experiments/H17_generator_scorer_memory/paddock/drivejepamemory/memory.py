# SPDX-License-Identifier: Apache-2.0
"""Worker half of the memory adapter: the build-5 memory model behind the 0.1.8b-p06 worker.

What runs (reports/BUILD5_RECALL_LESSON.md, model version 3, in the memory folder
Gallops-sid/memory/3_generator_scorer_memory_PAI):

    encoder     0.1.8b pass 0.6, merged, run LIVE on the two camera frames (no token cache)
    generator   build 5's weights (41 tensors) copied over the 0.1.8b ones, plus the generator
                read: the 256 queries cross-attend into the memory before the first refiner pass
    scorer      build 5's weights (30 tensors), plus the scorer read: a 256-d summary of the
                memory feeds 256 new input columns of the rating head's first layer
    memory      16 slots x 256, start state M0 (learned), update block after the pick
    read-out    NOT loaded. It is training-only; its tensors are skipped and counted.

The four blocks below (Attn, GenRead, ScorerRead, Update, StartState, block_means) are copies of
src/s6_model.py in the memory folder (version >= 2 branch only), so the checkpoint's keys load
strictly. The forward of one call repeats drive_jepa_model.py:50-94 line by line with the three
memory steps added where s6_model.MemoryModel.forward_moment adds them.

State. The worker keeps the memory per session (session_uuid). It is set to M0
  * when a session starts or closes (control message from the planner), and
  * when the previous call of the session is not 0.45-0.55 s back in simulated time
    (the chain rule of the earlier scorer-memory design); that is counted as a chain reset.
In mode "reset" the model reads M0 at every call (same weights; offline this was called
"wiped"). The mode comes from the sidecar file next to the checkpoint, never from a flag.

Which pick is written into the memory: the scorer's own argmax, BEFORE the comfort re-score,
because that is what the update block saw in training (s6_model.py:314-315) and what the offline
replay does (src/s8_short.py). The plan that is driven is the pick AFTER the comfort re-score,
exactly as in 0.1.8b-p06.

Shadow. Each call also runs the generator and scorer once more with the OTHER mode's memory on
the same input (the encoder output is shared), so every run records how often carry and reset
would have picked differently on identical input. The shadow never touches the driven plan: it
runs after the driving branch and its only lasting effect is, in reset mode, the carried memory
it keeps for the next shadow call.

**Runs only inside the Drive-JEPA conda env**, like real_model.py and momentum.py.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

from . import constants as K
from .momentum import MomentumDriveJEPA
from .real_model import RealDriveJEPAError

N_SLOTS, D, HEADS, FFN = 16, 256, 4, 1024
GRID_H, GRID_W, POOL_H, POOL_W = 16, 32, 4, 8
N_IMG = POOL_H * POOL_W
GEN_PREFIXES = ("hist_encoding.", "init_feature.", "_trajectory_head.")
GATE_P_FIRST, GATE_P_LAST, GATE_W_STD = 0.9, 0.1, 1e-3


def _blocks():
    """Build the block classes. Inside a function because torch is importable only in the worker."""
    import torch
    import torch.nn as nn

    class Attn(nn.Module):
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
        def __init__(self):
            super().__init__(); self.attn = Attn(zero_out=True)

        def forward(self, queries, M):
            return self.attn(queries, M)

    class ScorerRead(nn.Module):
        def __init__(self):
            super().__init__(); self.query = nn.Parameter(torch.randn(1, 1, D) * 0.02); self.attn = Attn()
            self.mem_in = nn.Linear(D, FFN, bias=False); nn.init.zeros_(self.mem_in.weight)

        def forward(self, M):
            return self.attn(self.query.expand(M.shape[0], -1, -1), M)[:, 0]

    class Update(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_in = nn.Linear(D, D); self.mot_in = nn.Linear(4, D); self.pick_in = nn.Linear(24, D)
            self.img_pos = nn.Parameter(torch.randn(1, N_IMG, D) * 0.02); self.type_emb = nn.Parameter(torch.randn(3, D) * 0.02)
            self.slot_emb = nn.Parameter(torch.randn(1, N_SLOTS, D) * 0.02)
            self.attn = Attn(); self.ln1 = nn.LayerNorm(D)
            self.ffn = nn.Sequential(nn.Linear(D, FFN), nn.ReLU(), nn.Linear(FFN, D)); self.ln2 = nn.LayerNorm(D)
            self.gate = nn.Linear(2 * D, 1, bias=False)
            p = torch.linspace(GATE_P_FIRST, GATE_P_LAST, N_SLOTS); self.gate_bias = nn.Parameter(torch.log(p / (1 - p)))

        def forward(self, M_old, img32, motion, pick):
            mot = torch.stack([motion[:, 0], motion[:, 1], torch.cos(motion[:, 2]), torch.sin(motion[:, 2])], -1)
            kv = torch.cat([self.img_in(img32) + self.img_pos + self.type_emb[0], (self.mot_in(mot) + self.type_emb[1])[:, None],
                            (self.pick_in(pick.reshape(pick.shape[0], -1)) + self.type_emb[2])[:, None]], 1)
            x = self.ln1(M_old + self.attn(M_old + self.slot_emb, kv))
            cand = self.ln2(x + self.ffn(x))
            z = self.gate(torch.cat([M_old, cand], -1))
            g = torch.sigmoid(z + self.gate_bias[None, :, None])
            return (1 - g) * M_old + g * cand, g[..., 0]

    class StartState(nn.Module):
        def __init__(self):
            super().__init__(); self.M0 = nn.Parameter(torch.zeros(N_SLOTS, D))

    return GenRead, ScorerRead, Update, StartState


def block_means(tokens):
    """(B, 512, D) tokens of the 16 x 32 grid -> (B, 32, D) 4 x 8 block means (s6_model.block_means)."""
    B = tokens.shape[0]
    return tokens.reshape(B, POOL_H, GRID_H // POOL_H, POOL_W, GRID_W // POOL_W, D).mean((2, 4)).reshape(B, N_IMG, D)


def motion_between(prev_pose: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """(dx, dy, dheading) of the car since the previous call, in the CURRENT car frame.

    Poses are (x, y, heading) in the rollout's fixed local frame. Equal to
    s6_model.motion_from_prev_future: R(-h_now) (p_now - p_prev), heading change wrapped.
    """
    dx, dy = float(pose[0] - prev_pose[0]), float(pose[1] - prev_pose[1])
    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
    dh = float(pose[2] - prev_pose[2]); dh = (dh + math.pi) % (2 * math.pi) - math.pi
    return np.array([c * dx + s * dy, -s * dx + c * dy, dh], dtype=np.float32)


class _Session:
    __slots__ = ("M", "t_us", "pose", "calls", "chain_len", "chain_resets", "starts", "norm_sum", "gate_sum",
                 "differ_scorer", "differ_final")

    def __init__(self):
        self.M = None            # the carried memory (None = start state)
        self.t_us = None; self.pose = None
        self.calls = 0; self.chain_len = 0; self.chain_resets = 0; self.starts = 0
        self.norm_sum = 0.0; self.gate_sum = 0.0; self.differ_scorer = 0; self.differ_final = 0

    def summary(self) -> dict:
        n = max(self.calls, 1)
        return {"calls": self.calls, "chain_resets": self.chain_resets, "memory_norm_mean": self.norm_sum / n,
                "gate_mean": self.gate_sum / n, "picks_differ_scorer": self.differ_scorer, "picks_differ_final": self.differ_final}


class MemoryDriveJEPA(MomentumDriveJEPA):
    """0.1.8b-p06's worker model with build 5's generator, scorer and memory blocks."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        torch = self._torch
        pin = K.memory_pin()
        self.pin = pin
        self.mode = pin["mode"]
        self.shadow = bool(K.MEMORY_SHADOW) and os.environ.get(K.MEMORY_SHADOW_ENV, "on").strip().lower() != "off"
        ck = torch.load(pin["path"], map_location="cpu")
        if int(ck.get("version", 1)) != K.MEMORY_VERSION:
            raise RealDriveJEPAError(f"memory checkpoint version {ck.get('version')} != {K.MEMORY_VERSION}")
        if ck["start"]["sha256"] != K.CHECKPOINT_SHA256:
            raise RealDriveJEPAError("memory checkpoint was trained from a different base checkpoint than the pinned 0.1.8b-p06")
        m = self._agent._pad_model
        named = dict(m.named_parameters())
        gen, sco = ck["generator"], ck["scorer"]
        if not all(n.startswith(GEN_PREFIXES) for n in gen) or set(sco) != {n for n in named if n.startswith("scorer.")}:
            raise RealDriveJEPAError("memory checkpoint: generator / scorer tensor names do not match the model")
        with torch.no_grad():
            for group in (gen, sco):
                for n, t in group.items():
                    if named[n].shape != t.shape:
                        raise RealDriveJEPAError(f"memory checkpoint: shape of {n}")
                    named[n].copy_(t.to(named[n].dtype))
        GenRead, ScorerRead, Update, StartState = _blocks()
        self.gen_read, self.sc_read, self.update, self.start = GenRead(), ScorerRead(), Update(), StartState()
        skipped = 0
        for name, mod in (("gen_read", self.gen_read), ("sc_read", self.sc_read), ("update", self.update), ("start", self.start)):
            mod.load_state_dict({k[len(name) + 1:]: v for k, v in ck["memory"].items() if k.startswith(name + ".")}, strict=True)
            mod.to(self._device).eval()
            for p in mod.parameters():
                p.requires_grad_(False)
        skipped = sum(1 for k in ck["memory"] if k.startswith("read_out."))
        other = [k for k in ck["memory"] if not k.startswith(("gen_read.", "sc_read.", "update.", "start.", "read_out."))]
        if other:
            raise RealDriveJEPAError(f"memory checkpoint: unknown tensors {other[:3]}")
        self.loaded = {"generator_tensors": len(gen), "scorer_tensors": len(sco),
                       "memory_tensors": len(ck["memory"]) - skipped, "read_out_tensors_skipped": skipped,
                       "trained_mode": ck.get("mode"), "step": ck.get("step"), "run": ck.get("run")}
        del ck
        self._agent.eval()
        self._sessions: dict[str, _Session] = {}
        self._ctx = None
        m.forward = self._model_forward          # nn.Module.__call__ looks forward up on the instance first
        print(
            f"drivejepa-memory checkpoint: verified {pin['path']} sha256 {pin['sha256']} bytes {pin['bytes']} "
            f"source {pin['source']} | mode {self.mode} (sidecar {pin['sidecar']}) version {K.MEMORY_VERSION} | "
            f"loaded {self.loaded} | read-out NOT loaded | shadow {'on' if self.shadow else 'off'} | "
            f"chain rule {K.MEMORY_CHAIN_MIN_US}-{K.MEMORY_CHAIN_MAX_US} us",
            file=sys.stderr, flush=True,
        )

    # ------------------------------------------------------------------
    def _m0(self):
        return self.start.M0.to(self._device).float()[None].expand(1, -1, -1)

    def _branch(self, m, ego_status, image_feature, M):
        """Generator + scorer for one memory. drive_jepa_model.py:61-73 and scorer.py:41-42 with the two reads."""
        bev = m.hist_encoding(ego_status)[:, None] + m.init_feature.weight[None]
        bev = bev + self.gen_read(bev, M)                                   # the generator read
        proposal_list = []
        for refine in m._trajectory_head:
            bev, proposals = refine(bev, image_feature)
            proposal_list.append(proposals)
        proposals = proposal_list[-1]
        summary = self.sc_read(M)                                           # the scorer read
        B, P, T = proposals.shape[:3]
        pf = bev.reshape(B, P, T, -1).amax(-2)
        mlp = m.scorer.pred_score.mlp
        h = mlp[0](pf) + self.sc_read.mem_in(summary)[:, None, :]           # rating head, first layer widened
        for layer in list(mlp)[1:]:
            h = layer(h)
        pred_logit = h.reshape(B, -1, m.scorer.score_num)
        return proposal_list, proposals, pred_logit

    def _model_forward(self, features: dict) -> dict:
        """Replaces DriveJEPAModel.forward on the instance (drive_jepa_model.py:49-96)."""
        torch = self._torch
        m = self._agent._pad_model
        ctx = self._ctx
        if ctx is None:
            raise RealDriveJEPAError("memory forward called without a memory request")
        features["lidar2img"] = features["lidar2img"][:, 1:2]
        ego_status = features["ego_status"][:, -1]
        cam_f_2 = features["camera_feature_2"]
        cam_f_1 = features["camera_feature_1"]
        cam_f_2 = m.transform(cam_f_2)
        cam_f_1 = m.transform(cam_f_1)
        camera_feature = torch.cat([cam_f_2[:, None], cam_f_1[:, None]], dim=1)
        batch_size = ego_status.shape[0]
        if batch_size != 1:
            raise RealDriveJEPAError("the memory adapter serves batch 1 only")
        image_feature = m._backbone(camera_feature, img_metas=features)
        have_prev = features.get("past_ego_simulated_states") is not None

        # ---- driving branch --------------------------------------------------------------
        M = ctx["M_drive"]
        proposal_list, proposals, pred_logit = self._branch(m, ego_status, image_feature, M)
        output = {"proposals": proposals, "proposal_list": proposal_list, "pred_logit": pred_logit,
                  "pred_agents_states": None, "pred_area_logit": None, "bev_semantic_map": None,
                  "agent_states": None, "agent_labels": None}
        pdm_score = torch.sigmoid(pred_logit)[:, :, -1]
        scorer_pick = torch.argmax(pdm_score, dim=1)                         # before the comfort re-score
        picked = proposals[torch.arange(batch_size), scorer_pick]
        if have_prev:
            pdm_score = m.calibrate_score(features, proposals, pdm_score)
        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]
        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        # ---- memory update, after the pick -------------------------------------------------
        t0 = time.perf_counter()
        tokens = image_feature[0][0].permute(1, 0, 2)                        # (num_cam, 512, B, D) -> (B, 512, D)
        bm = block_means(tokens)
        motion = torch.from_numpy(ctx["motion"]).to(self._device)[None]
        M_new, gate = self.update(M, bm, motion, picked)
        ctx["M_new_drive"] = M_new.float()
        diag = ctx["diag"]
        diag.update(read_norm=float(M.norm()), read_is_start=bool(torch.equal(M, self._m0())),
                    gate_mean=float(gate.mean()), gate_min=float(gate.min()), gate_max=float(gate.max()),
                    written_norm=float(M_new.norm()), change_abs=float((M_new - M).norm()),
                    pick_scorer=int(scorer_pick[0]), pick_final=int(token[0]))
        diag["ms_update"] = (time.perf_counter() - t0) * 1e3

        # ---- shadow: the other mode's memory on the same input --------------------------------
        if self.shadow:
            t0 = time.perf_counter()
            Ms = ctx["M_shadow"]
            _, props_s, logit_s = self._branch(m, ego_status, image_feature, Ms)
            score_s = torch.sigmoid(logit_s)[:, :, -1]
            pick_s = torch.argmax(score_s, dim=1)
            picked_s = props_s[torch.arange(batch_size), pick_s]
            if have_prev:                                                    # the class's own stage: not the timed wrapper
                score_s = type(m).calibrate_score(m, features, props_s, score_s)
            token_s = torch.argmax(score_s, dim=1)
            traj_s = props_s[torch.arange(batch_size), token_s]
            Ms_new, gate_s = self.update(Ms, bm, motion, picked_s)
            ctx["M_new_shadow"] = Ms_new.float()
            gap = float((trajectory[0, :, :2] - traj_s[0, :, :2]).norm(dim=-1).max())
            menu_gap = float((proposals[0, :, :, :2] - props_s[0, :, :, :2]).norm(dim=-1).max())
            diag.update(shadow_pick_scorer=int(pick_s[0]), shadow_pick_final=int(token_s[0]),
                        picks_differ_scorer=bool(int(pick_s[0]) != int(scorer_pick[0])),
                        picks_differ_final=bool(int(token_s[0]) != int(token[0])),
                        picked_plan_gap_m=gap, menu_gap_m=menu_gap, shadow_gate_mean=float(gate_s.mean()),
                        shadow_read_norm=float(Ms.norm()), ms_shadow=(time.perf_counter() - t0) * 1e3)
        return output

    # ------------------------------------------------------------------
    def _control(self, msg: dict) -> dict:
        op, uuid = msg.get("op"), str(msg.get("session_uuid"))
        if op not in ("start", "close"):
            raise RealDriveJEPAError(f"unknown memory control op {op!r}")
        old = self._sessions.pop(uuid, None)
        summary = None if old is None else old.summary()
        if op == "start":
            s = _Session(); s.starts = 1; self._sessions[uuid] = s
        print(f"drivejepa-memory session {op}: {uuid} memory set to the start state; previous {summary}",
              file=sys.stderr, flush=True)
        return {"memory_control": op, "session_uuid": uuid, "mode": self.mode, "summary": summary}

    def forward(self, features: dict) -> dict:
        if K.MEMORY_CONTROL_KEY in features:
            return self._control(features[K.MEMORY_CONTROL_KEY])
        req = features.get(K.MEMORY_FEATURE_KEY)
        if req is None:
            raise RealDriveJEPAError("request carries no 'memory' key: the planner behind this worker is not the memory adapter's")
        uuid = str(req["session_uuid"]); t_us = int(req["time_us"]); pose = np.asarray(req["ego_pose"], dtype=np.float64)
        s = self._sessions.get(uuid)
        if s is None:
            s = _Session(); s.starts = 1; self._sessions[uuid] = s
        dt = None if s.t_us is None else t_us - s.t_us
        chained = dt is not None and K.MEMORY_CHAIN_MIN_US <= dt <= K.MEMORY_CHAIN_MAX_US
        chain_reset = dt is not None and not chained
        if chained:
            M_carried = s.M if s.M is not None else self._m0()
            motion = motion_between(s.pose, pose)
        else:
            M_carried = self._m0()
            motion = np.zeros(3, dtype=np.float32)
        carry = self.mode == "carry"
        diag = {"mode": self.mode, "version": K.MEMORY_VERSION, "call": s.calls + 1, "dt_us": dt,
                "chain_reset": bool(chain_reset), "chain_len": (s.chain_len if chained else 0),
                "chain_resets": s.chain_resets + int(chain_reset), "motion": [float(v) for v in motion]}
        self._ctx = {"M_drive": M_carried if carry else self._m0(), "M_shadow": self._m0() if carry else M_carried,
                     "motion": motion, "diag": diag}
        try:
            plain = {k: v for k, v in features.items() if k != K.MEMORY_FEATURE_KEY}
            result = super().forward(plain)
            ctx = self._ctx
        finally:
            self._ctx = None
        # commit only after a successful forward
        if carry:
            s.M = ctx["M_new_drive"]
        else:
            s.M = ctx.get("M_new_shadow")        # the shadow's carried chain (None when the shadow is off)
        s.t_us = t_us; s.pose = pose; s.calls += 1
        s.chain_len = (s.chain_len + 1) if chained else 1
        s.chain_resets += int(chain_reset)
        s.norm_sum += diag["read_norm"]; s.gate_sum += diag["gate_mean"]
        s.differ_scorer += int(diag.get("picks_differ_scorer", False)); s.differ_final += int(diag.get("picks_differ_final", False))
        result["memory"] = diag
        return result

    __call__ = forward
