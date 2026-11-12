"""Part F: the one evaluation on val + overlap_train_val, chosen checkpoint only.

    ./run.sh src/f_final.py

1. Chosen = best selection ADE (out/e_passes.json "best_pass").  Its LoRA is
   merged into the encoder weights (W + alpha/r * B A, lora.LoRALinear.merged_weight)
   and each LoRALinear is replaced by a plain nn.Linear; merged vs unmerged must
   agree on the 103 selection moments within 1e-4 m.
2. The s1 baseline wrapper loop (f_baseline.py:97-131 = RUN.main's loop with
   --splits val overlap_train_val --protocol e2e --command gt --eval-horizon 3)
   with the merged agent on all 223 moments.
3. Compared with kitscenes_e2e_final.pkl (read-only, hash-checked).
No other checkpoint is run on these scenes.
"""

from __future__ import annotations

import copy
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

import common as C
import evalsets as E
import f_baseline as FB  # s1: build_root, forward_dump, SPLITS, ROOT
import lora as L
from s2common import CKPT, DATA, OUT, wait_for_gpu, write_json

RUN = C.kit_runner_module()
ADP = C.kit_adapter_module()


def merge(agent_lora):
    merged = copy.deepcopy(agent_lora)
    enc = merged.get_submodule(E.ENC_PATH)
    for name, m in list(L.lora_modules(enc).items()):
        lin = nn.Linear(m.base.in_features, m.base.out_features, bias=m.base.bias is not None,
                        device=m.base.weight.device, dtype=m.base.weight.dtype)
        with torch.no_grad():
            lin.weight.copy_(m.merged_weight())
            if m.base.bias is not None:
                lin.bias.copy_(m.base.bias)
        lin.requires_grad_(False)
        L._set_child(enc, name, lin)
    assert not L.lora_modules(merged)
    return merged


def agg(rows):
    out = {"n": len(rows)}
    for k in ("ade", "fde@1s", "fde@2s", "fde@3s"):
        v = [r[k] for r in rows]
        out[k] = {"mean": float(np.mean(v)), "median": float(np.median(v))}
    return out


def main() -> None:
    hist = json.loads((OUT / "e_passes.json").read_text())
    best = hist["best_pass"]
    out = {"chosen_pass": best, "gpu_wait_s": wait_for_gpu(8192, "F")}
    ck = torch.load(CKPT / f"pass_{best:02d}" / "state.pt", map_location="cpu", weights_only=False)
    assert C.sha256_file(C.AGENT_CKPT) == C.AGENT_CKPT_SHA256
    agent = RUN.build_agent(C.AGENT_CKPT, "drive_jepa_perception_based_agent")
    for p in agent.parameters():
        p.requires_grad_(False)
    agent_lora = copy.deepcopy(agent)
    L.wrap(agent_lora.get_submodule(E.ENC_PATH))
    enc = agent_lora.get_submodule(E.ENC_PATH)
    params = dict(enc.named_parameters())
    with torch.no_grad():
        for n, t in ck["lora"].items():
            params[n].copy_(t.cuda())
    del agent
    merged = merge(agent_lora)
    merged.eval(); agent_lora.eval()
    plain_keys = set(agent_lora.state_dict()) - {k for k in agent_lora.state_dict() if ".lora_" in k}
    out["merged_state_keys_match_plain"] = {k.replace(".base.", ".") for k in plain_keys} == set(merged.state_dict())

    # F.1 merged vs unmerged on the 103 selection moments
    sel = E.load_selection_features()
    du, ru = E.selection_eval(agent_lora, sel, RUN)
    dm, rm = E.selection_eval(merged, sel, RUN)
    diffs = [float(np.abs(du[k]["trajectory"][:, :2].astype(np.float64) - dm[k]["trajectory"][:, :2]).max()) for k in du]
    out["F1_merge_check"] = {"moments": len(diffs), "max_abs_xy_diff_m": max(diffs),
                             "max_abs_pose_diff_incl_heading": max(float(np.abs(du[k]["trajectory"] - dm[k]["trajectory"]).max()) for k in du),
                             "pass": max(diffs) <= 1e-4, "picks_equal": sum(du[k]["chosen"] == dm[k]["chosen"] for k in du),
                             "selection_ade_unmerged": E.aggregate(ru)["ade"]["mean"],
                             "selection_ade_merged": E.aggregate(rm)["ade"]["mean"],
                             "selection_ade_recorded_in_pass": ck["row"]["selection"]["ade"]["mean"]}
    print(json.dumps(out, indent=1), flush=True)
    if not out["F1_merge_check"]["pass"]:
        write_json(OUT / "f_final.json", {**out, "status": "STOP: merged != unmerged"})
        return
    torch.save(merged.get_submodule(E.ENC_PATH).state_dict(), DATA / f"merged_encoder_pass_{best:02d}.pt")

    # F.2 the 223 baseline moments
    out["root"] = FB.build_root()
    assert C.sha256_file(C.BASELINE_PKL) == C.BASELINE_PKL_SHA256
    with open(C.BASELINE_PKL, "rb") as fh:
        base_rows = pickle.load(fh)
    base = {(r["scene_id"], int(r["keyframe_index"])): r for r in base_rows}
    n_future = merged._trajectory_sampling.num_poses
    targets = [(sp, d.name) for sp in FB.SPLITS for d in sorted((FB.ROOT / "data" / sp).iterdir()) if d.is_dir()]

    def prepare(t):
        split, sid = t
        adapter = ADP.KITScenesSceneAdapter(FB.ROOT / "data" / split / sid)  # RUN:232
        kfs = adapter.valid_keyframes(num_future_frames=10, stride=90, first_frame=40)  # RUN:236-238
        return split, sid, adapter.ground_offset, [
            (kf, adapter.build_sample(kf, num_future_frames=n_future, command_mode="gt")) for kf in kfs]  # RUN:250-254

    rows, dump = [], {}
    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        for split, sid, ground, samples in pool.map(prepare, targets):
            for kf, sample in samples:
                predicted = merged.compute_trajectory(sample.agent_input)  # RUN:256
                metrics = RUN.displacement_errors(predicted.poses, sample.future_poses, sample.future_interval_s, 3.0)
                d = FB.forward_dump(merged, sample.agent_input)
                assert L.bit_equal(torch.from_numpy(d["trajectory"]), torch.from_numpy(predicted.poses.astype(np.float32)))
                dump[(sid, kf)] = d
                b = base[(sid, kf)]
                assert np.array_equal(b["human_poses"], sample.future_poses), (sid, kf)
                rows.append({"scene_id": sid, "split": split, "keyframe_index": kf, "predicted_poses": predicted.poses,
                             "human_poses": sample.future_poses, "ground_offset": ground, **metrics,
                             "august_ade": b["ade"], "delta_ade": metrics["ade"] - b["ade"]})
            print(f"{len(rows):3d} moments {time.time() - t0:5.0f}s {sid} ({split})", flush=True)
    out["wall_s_223"] = round(time.time() - t0, 1)
    with open(OUT / "f_final_rows.pkl", "wb") as fh:
        pickle.dump(rows, fh)
    with open(OUT / "f_final_dump.pkl", "wb") as fh:
        pickle.dump(dump, fh)

    assert {(r["scene_id"], r["keyframe_index"]) for r in rows} == set(base) and len(rows) == len(base_rows)
    table = {}
    for name, keep in (("val", ["val"]), ("overlap_train_val", ["overlap_train_val"]), ("all", FB.SPLITS)):
        ours = [r for r in rows if r["split"] in keep]
        aug = [r for r in base_rows if r["split"] in keep]
        d = np.array([r["delta_ade"] for r in ours])
        table[name] = {"august": agg(aug), "lora": agg(ours),
                       "delta_ade_mean": float(d.mean()), "delta_ade_median": float(np.median(d)),
                       "better_by_more_than_1cm": int((d < -0.01).sum()), "worse_by_more_than_1cm": int((d > 0.01).sum()),
                       "within_1cm": int((np.abs(d) <= 0.01).sum())}
    out["F3"] = table
    out["known_software_noise"] = "up to 1.2 mm per waypoint (ADE 0.2 mm) on the one moment checked in s1 (00fc1b9b, keyframe 40)"
    write_json(OUT / "f_final.json", out)
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
