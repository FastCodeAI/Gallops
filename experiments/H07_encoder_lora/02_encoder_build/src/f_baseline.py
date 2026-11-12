"""Part F: reproduce the 2.364 m baseline and dump all 32 candidates.

    ./run.sh src/f_baseline.py

The only script that reads val / overlap_train_val, inference only.
1. out/kit_root/data/{val,overlap_train_val}/<id> -> symlinks into Naren's roots.
2. The sample loop is RUN.main's (run_kitscenes_inference.py:184-275) with the
   baseline arguments (--splits val overlap_train_val --protocol e2e --command gt
   --eval-horizon 3): same calls, same order.  The only addition sits beside
   RUN:256: the features are built again (abstract_agent.py:79-84) and
   agent.forward is run under no_grad to keep proposals / pred_logit / pdm_score.
3. Gate F1: every predicted pose bit-identical to kitscenes_e2e_final.pkl and the
   aggregate reproduced.  On the first difference the loop stops.
4. Gate F2: the same samples through the agent with zero LoRA -> dump identical.
"""

from __future__ import annotations

import copy
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import common as C
import drift as D
import lora as L
import splits as S

RUN = C.kit_runner_module()
ADP = C.kit_adapter_module()
from scene_lengths import baseline_scene_paths  # noqa: E402

ROOT = C.OUT / "kit_root"
SPLITS = ["val", "overlap_train_val"]
ENC_PATH = "_pad_model._backbone.img_backbone"


def build_root() -> dict:
    paths = baseline_scene_paths()
    n = 0
    for split in SPLITS:
        folder = ROOT / "data" / split
        folder.mkdir(parents=True, exist_ok=True)
        for sid, src in paths[split].items():
            link = folder / sid
            if not link.is_symlink():
                link.symlink_to(src, target_is_directory=True)
            assert link.resolve() == Path(src).resolve()
            n += 1
    return {"root": str(ROOT), "scenes": n, "per_split": {s: len(paths[s]) for s in SPLITS}}


def forward_dump(agent, agent_input) -> dict:
    feats = {}
    for b in agent.get_feature_builders():
        feats.update(b.compute_features(agent_input))
    feats = {k: v.unsqueeze(0).cuda() for k, v in feats.items()}
    with torch.no_grad():
        o = agent.forward(feats)
    chosen = int(torch.argmax(o["pdm_score"], dim=1)[0])
    return {
        "proposals": o["proposals"][0].cpu().numpy(),
        "pred_logit": o["pred_logit"][0].cpu().numpy(),
        "pdm_score": o["pdm_score"][0].cpu().numpy(),
        "chosen": chosen,
        "trajectory": o["trajectory"][0].cpu().numpy(),
    }


def main() -> None:
    out: dict = {}
    out["root"] = build_root()
    print(out["root"])
    assert C.sha256_file(C.BASELINE_PKL) == C.BASELINE_PKL_SHA256
    with open(C.BASELINE_PKL, "rb") as fh:
        base_rows = pickle.load(fh)
    base = {(r["scene_id"], int(r["keyframe_index"])): r for r in base_rows}

    if not C.require_gpu(9000, "part F"):
        out["status"] = "BLOCKED-GPU"
        (C.OUT / "f_baseline.json").write_text(json.dumps(out, indent=1))
        return
    assert C.sha256_file(C.AGENT_CKPT) == C.AGENT_CKPT_SHA256
    agent = RUN.build_agent(C.AGENT_CKPT, "drive_jepa_perception_based_agent")
    agent_lora = copy.deepcopy(agent)
    for p in agent_lora.parameters():
        p.requires_grad_(False)
    L.wrap(agent_lora.get_submodule(ENC_PATH))
    agent_lora.eval()
    num_future_frames = agent._trajectory_sampling.num_poses

    # RUN:184-201 target list (no streaming: every scene is under the merged root)
    targets = []
    for split in SPLITS:
        split_root = ROOT / "data" / split
        targets.extend((split, d.name) for d in sorted(split_root.iterdir()) if d.is_dir())

    def prepare(t):
        split, sid = t
        adapter = ADP.KITScenesSceneAdapter(ROOT / "data" / split / sid)  # RUN:232
        kfs = adapter.valid_keyframes(num_future_frames=10, stride=90, first_frame=40)  # RUN:236-238
        return split, sid, adapter.ground_offset, [
            (kf, adapter.build_sample(kf, num_future_frames=num_future_frames, command_mode="gt")) for kf in kfs
        ]  # RUN:250-254

    results, dump_f1, dump_f2 = [], {}, {}
    first_diff = None
    gpu_s = 0.0
    t_start = time.time()
    with ThreadPoolExecutor(8) as pool:
        for split, sid, ground, samples in pool.map(prepare, targets):
            for kf, sample in samples:
                t0 = time.perf_counter()
                predicted = agent.compute_trajectory(sample.agent_input)  # RUN:256
                elapsed = time.perf_counter() - t0
                d1 = forward_dump(agent, sample.agent_input)
                d2 = forward_dump(agent_lora, sample.agent_input)
                gpu_s += time.perf_counter() - t0
                metrics = RUN.displacement_errors(predicted.poses, sample.future_poses,
                                                  sample.future_interval_s, 3.0)  # RUN:259-261
                row = {"scene_id": sid, "split": split, "keyframe_index": kf, "predicted_poses": predicted.poses,
                       "human_poses": sample.future_poses, "ground_offset": ground, "inference_seconds": elapsed,
                       **metrics}
                results.append(row)
                dump_f1[(sid, kf)], dump_f2[(sid, kf)] = d1, d2
                assert L.bit_equal(torch.from_numpy(d1["trajectory"]), torch.from_numpy(predicted.poses.astype(np.float32))), \
                    "dump trajectory != compute_trajectory output"
                b = base.get((sid, kf))
                same = b is not None and np.asarray(b["predicted_poses"]).tobytes() == predicted.poses.tobytes() \
                    and np.asarray(b["predicted_poses"]).dtype == predicted.poses.dtype
                if not same:
                    first_diff = {
                        "index": len(results) - 1, "scene_id": sid, "split": split, "keyframe": kf,
                        "in_baseline": b is not None,
                        "max_abs_pose_diff": None if b is None else float(np.abs(np.asarray(b["predicted_poses"], dtype=np.float64) - predicted.poses).max()),
                        "human_poses_equal": None if b is None else bool(np.array_equal(b["human_poses"], sample.future_poses)),
                        "ours_ade": metrics["ade"], "baseline_ade": None if b is None else b["ade"],
                        "dtypes": [None if b is None else str(np.asarray(b["predicted_poses"]).dtype), str(predicted.poses.dtype)],
                    }
                    break
            print(f"{len(results):3d} samples  {time.time() - t_start:6.0f}s  {sid} ({split})", flush=True)
            if first_diff:
                break

    out["samples_run"] = len(results)
    out["gpu_seconds_forward"] = round(gpu_s, 1)
    out["wall_s"] = round(time.time() - t_start, 1)
    out["first_diff"] = first_diff
    with open(C.OUT / "f1_rows.pkl", "wb") as fh:
        pickle.dump(results, fh)
    with open(C.OUT / "dump_f1.pkl", "wb") as fh:
        pickle.dump(dump_f1, fh)
    with open(C.OUT / "dump_f2.pkl", "wb") as fh:
        pickle.dump(dump_f2, fh)

    def agg(rows):
        return {k: {"n": len(rows), "mean": float(np.mean([r[k] for r in rows])), "median": float(np.median([r[k] for r in rows]))}
                for k in ("ade", "fde@1s", "fde@2s", "fde@3s")}

    out["F1"] = {
        "all_bit_identical": first_diff is None and len(results) == len(base_rows)
        and {(r["scene_id"], r["keyframe_index"]) for r in results} == set(base),
        "ours": agg(results),
        "baseline_pkl": agg(base_rows),
        "per_split_ours": {s: agg([r for r in results if r["split"] == s]) for s in SPLITS if any(r["split"] == s for r in results)},
        "per_split_baseline": {s: agg([r for r in base_rows if r["split"] == s]) for s in SPLITS},
    }
    if first_diff is None:
        out["F2"] = {"dump_identical": all(
            all(L.bit_equal(torch.as_tensor(dump_f1[k][f]), torch.as_tensor(dump_f2[k][f]))
                for f in ("proposals", "pred_logit", "pdm_score", "trajectory")) and dump_f1[k]["chosen"] == dump_f2[k]["chosen"]
            for k in dump_f1)}
        out["F5_self"] = D.drift(dump_f1, dump_f1)["summary"]
        out["F5_f1_vs_f2"] = D.drift(dump_f1, dump_f2)["summary"]
        out["chosen_index_hist"] = {int(i): int(n) for i, n in zip(*np.unique([d["chosen"] for d in dump_f1.values()], return_counts=True))}
    (C.OUT / "f_baseline.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in out.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
