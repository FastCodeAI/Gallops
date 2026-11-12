"""Make "Encoder, generator and scorer trained together" (4.5 passes) loadable by the regression gate.

    ./run_prep.sh export    # build_ours on CPU -> merged {"state_dict": {agent._pad_model.*}} on the NAS + key checks
    ./run_prep.sh load      # fresh process, no torch.load shim: construct and initialize() as Paddock's node does
    ./run_prep.sh compare   # 5 held-out moments: node-style loaded agent vs in-memory build_ours, CPU, eval mode

CPU only, Drive-JEPA env {p4}/naren/envs/drive-jepa. Each step writes regression/out/prep_<step>.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CKPT = Path("{runs}/drivejepa_kitscenes/generator_joint_kit/s5/ckpt/p04.5.pt")
RELEASED = Path("{weights}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt")
EXPORT_DIR = Path("{nas}/siddesh/checkpoints/drivejepa_kit_enc_gen_scorer_p045")
EXPORT = EXPORT_DIR / "merged_state_dict.ckpt"
DEVKIT = Path("{drivejepa_repo}/navsim_v2")
NEED_GB = 15.0


def wj(name: str, obj) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=str) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def avail_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def wait_ram(tag: str) -> float:
    t0 = time.time()
    while (a := avail_gb()) < NEED_GB:
        print(f"[{tag}] {a:.1f} GB available < {NEED_GB}; waiting", flush=True)
        time.sleep(60)
    print(f"[{tag}] {a:.1f} GB available", flush=True)
    return round(time.time() - t0, 1)


def group(k: str) -> str:
    k = k.removeprefix("agent._pad_model.")
    if k.startswith("_backbone.img_backbone."):
        return "encoder"
    if k.startswith("_backbone.projector."):
        return "projector"
    if k.startswith(("hist_encoding.", "init_feature.", "_trajectory_head.")):
        return "generator"
    if k.startswith("scorer.pred_score."):
        return "scorer pick head (pred_score)"
    if k.startswith("scorer."):
        return "other scorer heads"
    return "other"


# --------------------------------------------------------------------------- #
def build_ours_cpu():
    """final5.build_ours(4.5) unchanged, with the devkit's GPU move made a no-op (CPU only)."""
    import final as FN  # noqa: F401  (09_heldout_comparison; loads RUN)
    import final5 as F5

    FN.RUN.move_all_modules_to_gpu = lambda instance: None
    return F5.build_ours(4.5)


def export() -> None:
    import torch

    torch.set_num_threads(4)
    res = {"ram_wait_s": wait_ram("export")}
    agent = build_ours_cpu()
    sd = {"agent." + k: v.detach().cpu() for k, v in agent.state_dict().items()}
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = EXPORT.with_suffix(".tmp")
    torch.save({"state_dict": sd}, tmp)
    tmp.replace(EXPORT)
    res["export"] = {"path": str(EXPORT), "bytes": EXPORT.stat().st_size, "sha256": sha256(EXPORT), "keys": len(sd)}

    # checks against the released state_dict, read from the files on disk
    del agent
    new = torch.load(EXPORT, map_location="cpu", weights_only=True)
    res["export_top_level_keys"] = sorted(new)
    new = new["state_dict"]
    rel = torch.load(RELEASED, map_location="cpu", weights_only=False)["state_dict"]
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    c = {}
    c["released_keys"] = len(rel)
    c["key_set_identical"] = set(new) == set(rel)
    c["missing"] = sorted(set(rel) - set(new))[:10]
    c["extra"] = sorted(set(new) - set(rel))[:10]
    shared = sorted(set(new) & set(rel))
    c["shape_mismatch"] = [k for k in shared if new[k].shape != rel[k].shape]
    c["dtype_mismatch"] = [k for k in shared if new[k].dtype != rel[k].dtype]
    differ = [k for k in shared if not torch.equal(new[k], rel[k])]
    c["n_differ"] = len(differ)
    by: dict[str, list[int]] = {}
    for k in shared:
        g = by.setdefault(group(k), [0, 0])
        g[0] += 1
        g[1] += k in differ
    c["by_group_total_differ"] = by
    c["projector_equal_released"] = by.get("projector", [0, 0])[1] == 0
    c["other_scorer_heads_equal_released"] = by.get("other scorer heads", [0, 0])[1] == 0
    c["other_equal_released"] = by.get("other", [0, 0])[1] == 0
    c["max_abs_change_by_group"] = {
        g: max((float((new[k] - rel[k]).abs().max()) for k in differ if group(k) == g), default=0.0) for g in by
    }

    # the 4 generator head copies vs each other and vs the trained head.0 in p04.5.pt
    p = "agent._pad_model._trajectory_head."
    head0 = {k[len(p) + 2:]: v for k, v in new.items() if k.startswith(p + "0.")}
    copies_equal = all(torch.equal(new[f"{p}{i}.{n}"], t) for i in (1, 2, 3) for n, t in head0.items())
    trained = {n[len("_trajectory_head.0."):]: t for n, t in ck["generator"].items() if n.startswith("_trajectory_head.0.")}
    c["head_tensors_per_copy"] = len(head0)
    c["head_copies_identical"] = copies_equal
    c["trained_head0_tensors"] = len(trained)
    c["head0_equals_trained"] = all(torch.equal(head0[n], t) for n, t in trained.items())
    c["other_generator_equals_trained"] = all(
        torch.equal(new["agent._pad_model." + n], t) for n, t in ck["generator"].items() if not n.startswith("_trajectory_head.")
    )
    c["pred_score_equals_trained"] = all(torch.equal(new["agent._pad_model." + n], t) for n, t in ck["pred_score"].items())
    # encoder: merged weight == base + B@A (alpha = r = 16, scale 1), recomputed from released + p04.5 lora
    enc_ok, enc_n, enc_max = True, 0, 0.0
    pre = "agent._pad_model._backbone.img_backbone."
    for n in {n.rsplit(".lora_", 1)[0] for n in ck["lora"]}:
        a, b = ck["lora"][n + ".lora_A"], ck["lora"][n + ".lora_B"]
        want = rel[pre + n + ".weight"] + b @ a
        d = float((new[pre + n + ".weight"] - want).abs().max())
        enc_max = max(enc_max, d)
        enc_n += 1
        enc_ok &= d <= 1e-6
    c["encoder_layers_merged"] = enc_n
    c["encoder_merge_matches_base_plus_BA"] = enc_ok
    c["encoder_merge_max_abs_diff"] = enc_max
    c["lora_or_training_keys_present"] = [k for k in new if any(s in k for s in ("lora_", "key_lora", "guesser", ".base."))]
    c["non_tensor_values"] = [k for k, v in new.items() if not isinstance(v, torch.Tensor)]
    res["checks"] = c
    wj("prep_export.json", res)
    print(json.dumps(res, indent=1, default=str))


# --------------------------------------------------------------------------- #
def load() -> None:
    """policy.py:156-161 at Paddock 9585f81, in a process that never installs the torch.load shim."""
    os.environ.setdefault("DRIVE_JEPA_MSDA", "auto")
    os.environ["NAVSIM_EXP_ROOT"] = "{weights}"
    os.environ["NAVSIM_DEVKIT_ROOT"] = str(DEVKIT)
    sys.path.insert(0, str(DEVKIT))
    os.chdir(DEVKIT)  # model.yaml: cwd {repo}/navsim_v2
    import torch

    torch.set_num_threads(4)
    res = {"ram_wait_s": wait_ram("load"), "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    config = DriveJEPAConfig()
    agent = DriveJEPAAgent(config=config, lr=1e-4, checkpoint_path=str(EXPORT))
    agent.initialize()
    agent.eval()
    res["initialize"] = "ok (strict load_state_dict raised nothing)"
    new = torch.load(EXPORT, map_location="cpu", weights_only=True)["state_dict"]
    sd = agent.state_dict()
    res["loaded_equals_file"] = all(torch.equal(sd[k.replace("agent._pad_model", "_pad_model")], v) for k, v in new.items())
    res["loaded_keys"] = len(sd)
    wj("prep_load.json", res)
    print(json.dumps(res, indent=1))


# --------------------------------------------------------------------------- #
def compare() -> None:
    import pickle

    import numpy as np
    import torch

    torch.set_num_threads(4)
    res = {"ram_wait_s": wait_ram("compare")}
    import common as C
    import f_baseline as FB
    import final as FN

    ours = build_ours_cpu()
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    node = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=str(EXPORT))
    node.initialize()
    node.eval()

    held = pickle.load(open("{runs}/drivejepa_kitscenes/generator_joint_kit/s5/heldout223.pkl", "rb"))
    keys = sorted((m["scene_id"], m["keyframe_index"], m["split"]) for m in held["moments"])
    pick = [keys[i] for i in (0, 50, 100, 150, 200)]  # fixed rule: every 50th of the sorted 223
    gpu_dump = held["dumps"][5]
    n_future = ours._trajectory_sampling.num_poses
    rows, worst = [], {"pdm_score": 0.0, "pred_logit": 0.0, "proposals": 0.0, "trajectory": 0.0}

    def fwd(agent, feats):
        with torch.no_grad():
            o = agent.forward(dict(feats))
        return {"proposals": o["proposals"][0].numpy(), "pred_logit": o["pred_logit"][0].numpy(),
                "pdm_score": o["pdm_score"][0].numpy(), "chosen": int(torch.argmax(o["pdm_score"], dim=1)[0]),
                "trajectory": o["trajectory"][0].numpy()}

    for sid, kf, split in pick:
        ad = FN.ADP.KITScenesSceneAdapter(FB.ROOT / "data" / split / sid)
        smp = ad.build_sample(kf, num_future_frames=n_future, command_mode="gt")
        feats = {}
        for b in ours.get_feature_builders():
            feats.update(b.compute_features(smp.agent_input))
        feats = {k: v.unsqueeze(0) for k, v in feats.items()}
        a, b = fwd(ours, feats), fwd(node, feats)
        diff = {f: float(np.abs(a[f] - b[f]).max()) for f in worst}
        for f in worst:
            worst[f] = max(worst[f], diff[f])
        g = gpu_dump[(sid, kf)]
        rows.append({"scene_id": sid, "keyframe_index": kf, "split": split,
                     "pick_ours": a["chosen"], "pick_loaded": b["chosen"], "pick_same": a["chosen"] == b["chosen"],
                     "max_abs_diff": diff, "pick_stored_gpu_eval": int(g["chosen"]),
                     "cpu_vs_stored_gpu_pdm_score_max_abs": float(np.abs(a["pdm_score"] - g["pdm_score"]).max())})
        print(rows[-1], flush=True)
    res["moments"] = rows
    res["max_abs_diff_over_5"] = worst
    res["all_picks_same"] = all(r["pick_same"] for r in rows)
    res["within_1e-5"] = all(v <= 1e-5 for k, v in worst.items() if k in ("pdm_score", "trajectory"))
    res["note"] = "trajectory = the picked 8-pose plan; pdm_score = the 32 menu scores"
    del C
    wj("prep_compare.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "moments"}, indent=1))


if __name__ == "__main__":
    {"export": export, "load": load, "compare": compare}[sys.argv[1]]()
