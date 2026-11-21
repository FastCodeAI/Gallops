# SPDX-License-Identifier: Apache-2.0
"""Worker half of approach 4 (motion history): MH/src/model_a4.py's A4Model, built around the LIVE 0.1.8b-p06 agent.

What runs (Gallops-sid/memory/4_motion_history/PREREG_APPROACH4.md section 1, BUILD2_MODEL.md):

    encoder     0.1.8b-p06, unchanged, live on the two camera frames (as p06's worker)
    generator   the checkpoint's 41 tensors, copied INTO the live agent's parameters by A4Model.load_trainable
    scorer      the checkpoint's 30 tensors, the same way
    new blocks  motion encoder (48 -> 256 -> GELU -> 256), no_history (256), link 1 (256 -> 256), link 2 (W_new, 1024 x 256)

Nothing in the upstream forward (drive_jepa_model.py:50-94), the comfort re-score (calibrate_score), momentum.py or the
batch fix is changed. The history enters through three hooks, at exactly the places model_a4 adds it:

    _pad_model, before its forward   z = A4Model.summary(features["ego_history"], training=False): arm A never hides,
                                     arm B always uses no_history (model_a4.py summary)
    _trajectory_head[0], first call  bev + link1(z)[:, None]: the input of the first refiner pass is
                                     hist_encoding(ego)[:, None] + init_feature.weight[None] (drive_jepa_model.py:61-62), so this
                                     is model_a4.starts, same order of additions. The 4 heads are ONE module called 4 times
                                     (model_a4 asserts it), so only the first call of each forward adds.
    scorer.pred_score.mlp[0], output mlp[0](pf) + link2(z)[:, None, :]: model_a4.rating, the two-piece widened layer.

At a step-0 checkpoint both links are exactly zero, so every addition is x + 0 and the run is p06 bit for bit (the wiring proof).
Each forward asserts 4 head calls and 1 mlp[0] call. **Runs only inside the Drive-JEPA conda env**, like real_model.py.
"""

from __future__ import annotations

import hashlib
import sys

from . import constants as K


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def install(model) -> dict:
    """Overlay the approach-4 checkpoint named by constants.a4_pin() on a loaded MomentumDriveJEPA (after the batch fix).
    Returns what was loaded, for the worker log."""
    import torch

    pin = K.a4_pin()
    sha = _sha256(pin["checkpoint"])
    if sha != pin["sha256"]:
        raise RuntimeError(f"approach-4 checkpoint {pin['checkpoint']} sha256 {sha} is not the config's {pin['sha256']}")
    for p in (K.A4_PAI_SRC, K.A4_MH_SRC):
        if p not in sys.path:
            sys.path.insert(0, p)
    import model_a4 as MA

    agent = model._agent
    rng = torch.get_rng_state()                       # A4Model.__init__ seeds torch; the worker's own stream is put back
    a4 = MA.A4Model(agent, pin["arm"], 0)
    torch.set_rng_state(rng)
    state = torch.load(pin["checkpoint"], map_location="cpu", weights_only=False)
    if state.get("arm") != pin["arm"]:
        raise RuntimeError(f"checkpoint arm {state.get('arm')!r} is not the config's {pin['arm']!r}")
    a4.load_trainable(state)
    a4.eval()
    for p in a4.parameters():
        p.requires_grad_(False)
    agent.eval()

    m = agent._pad_model
    head = m._trajectory_head[0]
    mlp0 = m.scorer.pred_score.mlp[0]
    st = {"z": None, "heads": 0, "mlp0": 0, "calls": 0, "hidden": 0}

    def pre_model(_mod, args):
        h = args[0][K.HISTORY_FEATURE_KEY]
        z, hide = a4.summary(h.to(torch.float32), training=False)
        st.update(z=z, heads=0, mlp0=0)
        st["hidden"] += int(hide.sum())

    def pre_head(_mod, args):
        st["heads"] += 1
        if st["heads"] == 1:
            return (args[0] + a4.link1(st["z"])[:, None],) + tuple(args[1:])
        return None

    def post_mlp0(_mod, _inp, out):
        st["mlp0"] += 1
        return out + a4.link2(st["z"])[:, None, :]

    def post_model(_mod, _args, _out):
        if st["heads"] != 4 or st["mlp0"] != 1:
            raise RuntimeError(f"approach-4 hooks: {st['heads']} head calls (4 expected), {st['mlp0']} mlp[0] calls (1 expected)")
        st["calls"] += 1
        st["z"] = None

    m.register_forward_pre_hook(pre_model)
    head.register_forward_pre_hook(pre_head)
    mlp0.register_forward_hook(post_mlp0)
    m.register_forward_hook(post_model)
    model._a4, model._a4_state = a4, st
    new = {n: float(p.detach().abs().max()) for n, p in a4.new_params}
    return {"checkpoint": pin["checkpoint"], "sha256": sha, "arm": pin["arm"], "config": pin["config"],
            "step": state.get("step"), "progress_passes": state.get("progress_passes"), "hide_prob_train": a4.hide_prob,
            "eval_hides": pin["arm"] == "B", "generator_tensors": len(a4.gen_params), "scorer_tensors": len(a4.score_params),
            "new_tensors": len(a4.new_params), "link1_max_abs": max(new["link1.weight"], new["link1.bias"]),
            "link2_max_abs": new["link2.weight"], "model_a4_sha256": _sha256(K.A4_MH_SRC + "/model_a4.py")}
