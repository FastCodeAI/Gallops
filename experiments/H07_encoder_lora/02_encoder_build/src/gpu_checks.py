"""GPU checks on the real agent: B.2 equality, T1, T2, T3 and Part E.

    ./run.sh src/gpu_checks.py        # writes out/gpu_checks.json

Reads only train/selection scenes (splits.json).  No checkpoint is saved.
"""

from __future__ import annotations

import copy
import json
import random
import time

import numpy as np
import torch

import common as C
import lora as L
import players as P
import splits as S
from pairs import PairDataset, build_pair

RUN = C.kit_runner_module()
ADP = C.kit_adapter_module()
ENC_PATH = "_pad_model._backbone.img_backbone"


def selection_keyframes(data, n=3):
    """First e2e keyframe (frame 40, RUN:236-238) of the first n selection scenes."""
    out = []
    for sid in data["selection"][:n]:
        out.append((sid, data["scene_paths"][sid], 40))
    return out


def baseline_features(agent, scene_path, kf):
    """compute_trajectory's feature path (abstract_agent.py:76-84) for one keyframe."""
    adapter = ADP.KITScenesSceneAdapter(scene_path)
    sample = adapter.build_sample(kf, num_future_frames=agent._trajectory_sampling.num_poses, command_mode="gt")
    feats = {}
    for b in agent.get_feature_builders():
        feats.update(b.compute_features(sample.agent_input))
    return {k: v.unsqueeze(0).cuda() for k, v in feats.items()}


def forward_full(agent, feats):
    with torch.no_grad():
        return agent.forward({k: v.clone() for k, v in feats.items()})


def freeze_all(module):
    for p in module.parameters():
        p.requires_grad_(False)


def main() -> None:
    out: dict = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    if not C.require_gpu(14000, "gpu_checks"):
        out["status"] = "BLOCKED-GPU"
        (C.OUT / "gpu_checks.json").write_text(json.dumps(out, indent=1))
        return
    out["free_mib_start"] = C.gpu_free_mib()
    t_start = time.time()
    data = S.load_splits()
    assert C.sha256_file(C.AGENT_CKPT) == C.AGENT_CKPT_SHA256

    agent = RUN.build_agent(C.AGENT_CKPT, "drive_jepa_perception_based_agent")  # RUN:108-116
    out["agent_build_s"] = round(time.time() - t_start, 1)
    freeze_all(agent)

    # ---- B.2 loader == the baseline feature path ------------------------------
    captured = {}
    hook = agent._pad_model._backbone.register_forward_pre_hook(lambda m, a, k=None: captured.__setitem__("x", a[0].detach().clone()))
    b2 = []
    kfs = selection_keyframes(data)
    base_out = {}
    for sid, path, kf in kfs:
        feats = baseline_features(agent, path, kf)
        base_out[(sid, kf)] = forward_full(agent, feats)
        ours = build_pair(path, kf)[None].cuda()
        b2.append({"scene": sid, "keyframe": kf, "bit_equal": L.bit_equal(ours, captured["x"]),
                   "max_abs_diff": float((ours - captured["x"]).abs().max())})
    hook.remove()
    out["B2"] = b2
    print("B2", b2)

    # ---- determinism control: the plain agent twice ---------------------------
    sid, path, kf = kfs[0]
    feats0 = baseline_features(agent, path, kf)
    r1, r2 = forward_full(agent, feats0), forward_full(agent, feats0)
    out["control_plain_twice_bit_equal"] = L.bit_equal(r1["proposals"], r2["proposals"])

    # ---- T1 encoder at zero LoRA ----------------------------------------------
    plain_enc = agent._pad_model._backbone.img_backbone
    learner, key = P.make_learner_and_key(plain_enc)
    learner.eval()
    pairs = torch.stack([build_pair(path, kf) for _, path, kf in kfs]).cuda()
    with torch.no_grad():
        ya, yb = P.encode(learner, pairs), P.encode(plain_enc, pairs)
    out["T1"] = {"bit_equal": L.bit_equal(ya, yb), "max_abs_diff": float((ya - yb).abs().max()),
                 "batch": list(pairs.shape), "dtype": str(ya.dtype)}
    print("T1", out["T1"])

    # ---- T2 zero LoRA inside the full agent -----------------------------------
    agent_lora = copy.deepcopy(agent)
    freeze_all(agent_lora)
    L.wrap(agent_lora.get_submodule(ENC_PATH))
    t2 = []
    for sid, path, kf in kfs:
        feats = baseline_features(agent, path, kf)
        a = base_out[(sid, kf)]
        b = forward_full(agent_lora, feats)
        t2.append({"scene": sid, "keyframe": kf,
                   "trajectory_bit_equal": L.bit_equal(a["trajectory"], b["trajectory"]),
                   "proposals_bit_equal": L.bit_equal(a["proposals"], b["proposals"]),
                   "pred_logit_bit_equal": L.bit_equal(a["pred_logit"], b["pred_logit"]),
                   "proposals_shape": list(b["proposals"].shape)})
    out["T2"] = t2
    print("T2", t2)

    # ---- T3 trainable set ------------------------------------------------------
    ok, info = L.check_t3_trainable(agent_lora, ENC_PATH, expected_wrapped=96)
    out["T3"] = {"ok": ok, **info,
                 "projector_requires_grad": any(p.requires_grad for p in agent_lora._pad_model._backbone.projector.parameters())}
    print("T3", out["T3"])
    del agent_lora
    torch.cuda.empty_cache()

    # ---- Part E: one training step -------------------------------------------
    e50 = torch.load(C.E50, map_location="cpu", mmap=True)
    guesser = P.build_guesser(e50).cuda()
    del e50
    gens = P.mask_generators()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    train_ds = PairDataset(data["train"][:40], data["scene_paths"])
    rng = random.Random(0)

    def batch(n):
        idx = rng.sample(range(len(train_ds)), n)
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_ds, idx), batch_size=n, num_workers=8)
        x, sids, ts = next(iter(loader))
        return x.cuda(), list(zip(sids, [int(t) for t in ts]))

    def grads_state():
        lora_A = [p.grad for n, p in learner.named_parameters() if n.endswith("lora_A")]
        lora_B = [p.grad for n, p in learner.named_parameters() if n.endswith("lora_B")]
        base = [p.grad for n, p in learner.named_parameters() if "lora_" not in n]
        g = [p.grad for p in guesser.parameters()]
        nz = lambda gs: sum(1 for t in gs if t is not None and float(t.abs().sum()) > 0)  # noqa: E731
        return {
            "guesser_tensors_nonzero_grad": nz(g), "guesser_tensors": len(g),
            "lora_A_nonzero_grad": nz(lora_A), "lora_B_nonzero_grad": nz(lora_B), "lora_layers": len(lora_A),
            "base_encoder_grads_not_None": sum(t is not None for t in base),
            "answer_key_grads_not_None": sum(p.grad is not None for p in key.parameters()),
            "agent_grads_not_None": sum(p.grad is not None for p in agent.parameters()),
        }

    learner.train(); guesser.train(); key.eval()
    xb, which = batch(2)
    out["E_batch"] = which
    masks = P.sample_masks(gens, 2, "cuda")

    # E.2 warm-up mode: LoRA held at zero
    L.set_lora_trainable(learner, False)
    for p in guesser.parameters():
        p.requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, per = P.game_loss(learner, key, guesser, xb, masks)
    loss.backward()
    out["E2_warmup"] = {"loss": float(loss), "per_mask": per, **grads_state()}
    print("E2", out["E2_warmup"])
    guesser.zero_grad(set_to_none=True)

    # E.3 main mode: LoRA + guesser
    L.set_lora_trainable(learner, True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, per = P.game_loss(learner, key, guesser, xb, masks)
    loss.backward()
    first = {"loss": float(loss), "per_mask": per, **grads_state()}
    key_B_before = [p.detach().clone() for n, p in key.named_parameters() if n.endswith("lora_B")]
    opt = torch.optim.AdamW([p for p in learner.parameters() if p.requires_grad] + list(guesser.parameters()), lr=1e-4)
    opt.step(); opt.zero_grad(set_to_none=True)
    m = 0.99925
    L.momentum_update(key, learner, m)
    learner_B = [p.detach() for n, p in learner.named_parameters() if n.endswith("lora_B")]
    key_B = [p.detach() for n, p in key.named_parameters() if n.endswith("lora_B")]
    rule_err = max(float((k - (m * b + (1 - m) * q)).abs().max()) for k, b, q in zip(key_B, key_B_before, learner_B))
    base_same = all(L.bit_equal(a, b) for (na, a), (nb, b) in zip(key.named_parameters(), learner.named_parameters()) if "lora_" not in na)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss2, per2 = P.game_loss(learner, key, guesser, xb, masks)
    loss2.backward()
    second = {"loss": float(loss2), **grads_state()}
    out["E3_main"] = {"step1_before_optimizer": first,
                      "momentum": {"m": m, "max_abs_err_vs_rule": rule_err, "base_identical": base_same,
                                   "key_B_nonzero_after": sum(int(float(t.abs().sum()) > 0) for t in key_B)},
                      "step2_after_one_update": second}
    print("E3", json.dumps(out["E3_main"], indent=1))
    opt.zero_grad(set_to_none=True)
    del opt

    # E.4 initial warm-up loss on 8 batches (fresh learner/key: LoRA zero)
    learner, key = P.make_learner_and_key(plain_enc)
    guesser = P.build_guesser(torch.load(C.E50, map_location="cpu", mmap=True)).cuda().eval()
    learner.eval()
    e4 = []
    for i in range(8):
        xb, which = batch(8)
        masks = P.sample_masks(gens, 8, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            loss, per = P.game_loss(learner, key, guesser, xb, masks)
        e4.append({"loss": float(loss), "per_mask": per,
                   "hidden_tokens": [int(pm.shape[1]) for _, pm in masks]})
        print("E4", i, e4[-1])
    out["E4_initial_warmup_loss"] = {"batches": e4, "mean": float(np.mean([r["loss"] for r in e4])),
                                     "e50_pretraining_final_loss_for_reference": 0.48458}
    out["free_mib_end"] = C.gpu_free_mib()
    out["max_memory_allocated_mib"] = torch.cuda.max_memory_allocated() // (1 << 20)
    out["wall_s"] = round(time.time() - t_start, 1)
    (C.OUT / "gpu_checks.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("done", out["wall_s"], "s")


if __name__ == "__main__":
    main()
