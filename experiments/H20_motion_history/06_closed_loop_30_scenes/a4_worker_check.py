"""Closed loop, before the wiring proof: the approach-4 worker overlay (adapters/drivejepa_a4/a4.py) checked offline in the worker's own env.

Built exactly as bridge._worker_main builds it: MomentumDriveJEPA on the pinned p06 checkpoint, CPU, batch fix installed and on; then
  1. p06 alone on one input (no overlay)
  2. a4.install with the step-0 snapshot (runs/arm_A/ckpt/p00.0.pt): outputs must be bit-identical to 1
  3. arm A's chosen snapshot p00.2 loaded into the same overlay: outputs must move; a different history must move them again
  4. arm B's chosen snapshot p00.6 with arm B's eval rule (always hides): a different history must NOT move them
Input: random camera frames (seeded), car state / camera geometry / history of one real training moment. Output MH/out/a4_worker_check.json.
Run with the worker python, PYTHONPATH=Alpasim/adapters, cwd Alpasim/adapters.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

AD = Path("{paddock_sid}/Alpasim/adapters")
MH = Path(__file__).resolve().parents[1]
TOK = Path("{approach4_store}/tokens")
CL = Path("{approach4_store}/cl/model")
sys.path.insert(0, str(AD))


def main():
    import pandas as pd
    import torch
    from drivejepa_a4 import constants as K, batch_fix, a4
    from drivejepa_a4.momentum import MomentumDriveJEPA
    ckpt, sha, nbytes, src = K.checkpoint_pin()
    model = MomentumDriveJEPA(checkpoint_path=ckpt, device="cpu")
    batch_fix.install(model._agent._pad_model); model._agent.eval(); batch_fix.set_fix(True, True, True)

    ix = pd.read_parquet(TOK / "index.parquet", columns=["sample_id", "clip", "ego_status"])
    clip = ix["clip"].iloc[0]
    side = np.load(sorted(Path("{staging}/ego_history").glob(f"{clip}*.npz"))[0])
    k = int(np.where(side["history"][:, :, 5].sum(1) == 8)[0][0])                 # the clip's first moment with all 8 history rows valid
    hist = side["history"][k].astype(np.float32)
    idx = ix[ix["sample_id"] == str(side["sample_id"][k])].iloc[0]
    cam = np.load(TOK / "camera.npz")
    j = list(cam["clip"]).index(idx["clip"])
    rng = np.random.default_rng(0)
    ego = np.asarray(idx["ego_status"], np.float32)[None, None].copy(); ego[..., 5:7] = 0
    f = {"camera_feature_1": rng.random((1, 3, 256, 512), dtype=np.float32), "camera_feature_2": rng.random((1, 3, 256, 512), dtype=np.float32),
         "ego_status": ego, "lidar2img": cam["lidar2img"][j][None].astype(np.float32), "img_shape": cam["img_shape"][j][None].astype(np.float32),
         K.HISTORY_FEATURE_KEY: hist[None]}
    other = dict(f); h2 = hist.copy(); h2[:, :5] = h2[::-1, :5]; other[K.HISTORY_FEATURE_KEY] = h2[None]     # a different (reversed) history
    keys = ("proposals", "pred_logit", "pdm_score", "trajectory")
    def run(feats):
        o = model.forward(feats)
        return {k: np.array(o[k]) for k in keys}
    same = lambda a, b: all(np.array_equal(a[k], b[k]) for k in keys)                                          # noqa: E731
    diff = lambda a, b: float(max(np.abs(a[k] - b[k]).max() for k in keys))                                    # noqa: E731

    res = {"input": {"sample_id": idx["sample_id"], "history_valid_rows": int(hist[:, 5].sum())}, "torch": torch.__version__}
    p06 = run(f)
    os.environ[K.A4_CONFIG_ENV] = str(CL / "step0A" / "a4_config.json")
    res["install_step0"] = a4.install(model)
    s0 = run(f)
    res["step0_vs_p06_identical"] = same(s0, p06); res["step0_vs_p06_maxdiff"] = diff(s0, p06)
    res["step0_calls"] = model._a4_state["calls"]

    ov = model._a4
    st = torch.load(json.load(open(CL / "armA" / "a4_config.json"))["checkpoint"], map_location="cpu", weights_only=False)
    ov.load_trainable(st)
    a = run(f); a2 = run(other)
    res["armA_vs_p06_maxdiff"] = diff(a, p06); res["armA_history_moves"] = diff(a, a2)
    res["armA_pick"] = int(np.argmax(a["pdm_score"][0])); res["p06_pick"] = int(np.argmax(p06["pdm_score"][0]))

    st = torch.load(json.load(open(CL / "armB" / "a4_config.json"))["checkpoint"], map_location="cpu", weights_only=False)
    ov.load_trainable(st); ov.arm = "B"
    b = run(f); b2 = run(other)
    res["armB_vs_p06_maxdiff"] = diff(b, p06); res["armB_history_moves"] = diff(b, b2); res["armB_hidden_total"] = model._a4_state["hidden"]
    res["pass"] = bool(res["step0_vs_p06_identical"] and res["armA_vs_p06_maxdiff"] > 0 and res["armA_history_moves"] > 0
                       and res["armB_history_moves"] == 0.0)
    (MH / "out" / "a4_worker_check.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "install_step0"}, default=str))
    print(json.dumps(res["install_step0"], default=str))


if __name__ == "__main__":
    main()
