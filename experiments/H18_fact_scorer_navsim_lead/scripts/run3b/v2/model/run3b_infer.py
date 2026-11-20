"""Run 3B v2 at inference: load a trainer checkpoint and pick one of the 32 proposals. Shared by the Paddock adapter
(worktrees/gate-oct27 .../models/run3b_v2/policy.py) and the NAVSIM agent (run3b/v2/navtest/run3b_agent.py), so both run
the same code.

  load_run3b(ckpt)  the net as train3.py builds it (build_encoder + Run3Bv2(vit, reader_layers)), every trainable tensor
                    (LoRA A/B, agent decoder, planner, path reader) from ckpt["trainable"] ("._orig_mod." from
                    torch.compile stripped). Raises unless the checkpoint's tensors and the model's trainable parameters are
                    the same set with the same shapes. The frozen base ViT is WA-JEPA's released encoder, as in training.
  Picker.forward    features {clips (B, 4, 3, 2, 256, 512) normalised, b0 f0 l0 r0, frames t - 0.5 s, t; lidar2img (B, 4,
                    4, 4) Drive-JEPA's 768x432; ego_status (B, 4, 11)} -> decoder's agents (no teacher forcing) -> planner's
                    32 proposals -> path reader facts -> r3b/score.py score_torch (T4) with R = the expert's-metres head, ntg
                    = head's 4 s value < 0.5 m, P = the checkpoint's -> argmax. = eval_v2's "pick (expert metres head)".
  clips_from_agent_input   WA-JEPA's preprocessing (img[28:-28] -> 512x256, ImageNet norm) of a NAVSIM AgentInput's
                    cameras[-2], cameras[-1] (navsim drive_jepa_perception_based/bevformer/wajepa.py, the same function
                    run 3A's token builder calls).
"""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
R3B = Path(os.environ.get("RUN3B_R3B", "{runs}/run3b/r3b"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import model2 as M2  # noqa: E402  (puts the navsim_v2 devkit, run3a-adapted and r3b on sys.path)

DOMAINS = ("navsim", "lead")


def _score_module():
    name = "run3b_r3b_score"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(R3B / "score.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_run3b(ckpt_path, device="cuda"):
    """-> (net in eval mode on device, info dict)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    vit, msg = M2.build_encoder("cpu", checkpointing=False, compile_blocks=False)
    net = M2.Run3Bv2(vit, int(args.get("reader_layers", 2)))
    want = {n: p for n, p in net.named_parameters() if p.requires_grad}
    sd = {k.replace("._orig_mod.", "."): v for k, v in ck["trainable"].items()}
    missing, extra = sorted(set(want) - set(sd)), sorted(set(sd) - set(want))
    if missing or extra:
        raise RuntimeError(f"checkpoint does not match the model: missing {missing[:5]} ({len(missing)}), "
                           f"unexpected {extra[:5]} ({len(extra)})")
    with torch.no_grad():
        for k, v in sd.items():
            if want[k].shape != v.shape:
                raise RuntimeError(f"{k}: checkpoint {tuple(v.shape)} vs model {tuple(want[k].shape)}")
            want[k].copy_(v)
    net.eval().to(device)
    info = {"checkpoint": str(ckpt_path), "sha256": sha256(ckpt_path), "step": ck.get("step"), "tag": ck.get("tag"),
            "trainable_tensors": len(sd), "encoder_load": str(msg), "P": float(args.get("P", 0.108)),
            "pick": "argmax T4 fact score (r3b/score.py), expert metres = head, ntg = head 4 s < 0.5 m",
            "agents": "decoder's own (no teacher forcing)"}
    return net, info


class Picker:
    def __init__(self, net, P, domain="navsim"):
        if domain not in DOMAINS:
            raise ValueError(f"score domain {domain!r}; expected one of {DOMAINS}")
        self.net, self.P, self.domain = net, float(P), domain
        self.score_torch = _score_module().score_torch

    @torch.no_grad()
    def forward(self, features):
        net = self.net
        dev = next(net.parameters()).device
        clips = features["clips"].to(dev).float()
        B = clips.shape[0]
        l2i = features["lidar2img"].to(dev).float()
        ego = features["ego_status"].to(dev).float()
        tokens = net.encoder(clips.reshape(B * 4, *clips.shape[2:]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, agents = net.decode(tokens, l2i)
            pl = net.plan(tokens, l2i, ego, agents)
            proposals = pl["proposals"].float()
            logits, numbers = net.read(proposals, tokens, l2i, agents)
        em = pl["expert_metres"].float().reshape(B, 8)
        p = torch.sigmoid(logits.float())
        facts = {k: p[..., i] for i, k in enumerate(M2.YESNO)}
        for i, k in enumerate(M2.NUMBERS):
            facts[k] = numbers[..., i].float()
        facts["ttc"] = facts["ttc"].clamp(0.0, 4.0)
        facts["cutin_brake"] = facts["cutin_brake"].clamp(0.0, 8.0)
        ntg = em[:, -1] < 0.5
        nav = torch.full((B,), self.domain == "navsim", dtype=torch.bool, device=dev)
        score, a, b = self.score_torch(facts, em, ntg, nav, P=self.P)
        pick = score.argmax(1)
        trajectory = proposals[torch.arange(B, device=dev), pick]
        return {"trajectory": trajectory, "proposals": proposals, "pdm_score": score, "score_a": a, "score_b": b,
                "expert_metres": em, "ntg": ntg, "pick": pick}


def clips_from_agent_input(agent_input):
    from navsim.agents.drive_jepa_perception_based.bevformer.wajepa import clips_from_agent_input as f
    return f(agent_input)
