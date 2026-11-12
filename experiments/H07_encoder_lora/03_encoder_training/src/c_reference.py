"""Part C: the selection "before" numbers with zero LoRA.

    ./run.sh src/c_reference.py moments   # C.1: 103 selection moments (reads NAS, GPU ~3 GB)
    ./run.sh src/c_reference.py fixed     # C.2: fixed guesser-loss set (needs the frame cache)

C.1 is the s1 baseline wrapper (f_baseline.py:103-131) pointed at the selection
scenes: ADP.KITScenesSceneAdapter(scene) (RUN:232), valid_keyframes(10, 90, 40)
(RUN:236-238), build_sample(..., command_mode="gt") (RUN:250-254),
agent.compute_trajectory (RUN:256), RUN.displacement_errors(..., 3.0)
(RUN:259-261).  Beside it, per moment:
  * the agent features are saved (selection_features.pt) for every later pass;
  * evalsets.forward_dump on the saved features must equal compute_trajectory
    bit-for-bit (so later passes, which skip the NAS, run the same computation);
  * the same with the encoder wrapped by zero LoRA (s1 T2/F2 on 103 moments).
"""

from __future__ import annotations

import copy
import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import common as C
import drift as D
import evalsets as E
import lora as L
import splits as S
from s2common import OUT, SEL_FEATS, wait_for_gpu, write_json

RUN = C.kit_runner_module()
ADP = C.kit_adapter_module()


def freeze(m):
    for p in m.parameters():
        p.requires_grad_(False)


def moments() -> None:
    data = S.load_splits()
    out = {"gpu_wait_s": wait_for_gpu(8192, "C.1")}
    assert C.sha256_file(C.AGENT_CKPT) == C.AGENT_CKPT_SHA256
    agent = RUN.build_agent(C.AGENT_CKPT, "drive_jepa_perception_based_agent")  # RUN:108-116
    freeze(agent)
    agent_lora = copy.deepcopy(agent)
    L.wrap(agent_lora.get_submodule(E.ENC_PATH))
    agent_lora.eval()
    n_future = agent._trajectory_sampling.num_poses
    builders = agent.get_feature_builders()

    def prepare(sid):
        adapter = ADP.KITScenesSceneAdapter(Path(data["scene_paths"][sid]))
        kfs = adapter.valid_keyframes(num_future_frames=10, stride=90, first_frame=40)
        return sid, adapter.ground_offset, [(kf, adapter.build_sample(kf, num_future_frames=n_future, command_mode="gt"))
                                            for kf in kfs]

    entries, rows, dump0, dumpz = [], [], {}, {}
    checks = {"forward_dump_eq_compute_trajectory": 0, "zero_lora_dump_eq_plain": 0}
    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        for sid, ground, samples in pool.map(prepare, sorted(data["selection"])):
            for kf, sample in samples:
                predicted = agent.compute_trajectory(sample.agent_input)  # RUN:256
                metrics = RUN.displacement_errors(predicted.poses, sample.future_poses,
                                                  sample.future_interval_s, 3.0)  # RUN:259-261
                feats = {}
                for b in builders:
                    feats.update(b.compute_features(sample.agent_input))  # abstract_agent.py:79-80
                e = {"scene_id": sid, "keyframe": kf, "features": feats, "future_poses": sample.future_poses,
                     "future_interval_s": sample.future_interval_s, "ground_offset": ground,
                     "driving_command": sample.agent_input.ego_statuses[-1].driving_command,
                     "ego_speed": float(np.linalg.norm(sample.agent_input.ego_statuses[-1].ego_velocity))}
                d0 = E.forward_dump(agent, feats)
                dz = E.forward_dump(agent_lora, feats)
                checks["forward_dump_eq_compute_trajectory"] += int(
                    d0["trajectory"].dtype == predicted.poses.dtype and d0["trajectory"].tobytes() == predicted.poses.tobytes())
                checks["zero_lora_dump_eq_plain"] += int(all(
                    np.asarray(d0[f]).tobytes() == np.asarray(dz[f]).tobytes() for f in ("proposals", "pred_logit", "pdm_score", "trajectory"))
                    and d0["chosen"] == dz["chosen"])
                entries.append(e)
                dump0[(sid, kf)], dumpz[(sid, kf)] = d0, dz
                rows.append({"scene_id": sid, "keyframe_index": kf, "predicted_poses": predicted.poses,
                             "human_poses": sample.future_poses, "ground_offset": ground, **metrics})
            print(f"{len(rows):3d} moments {time.time() - t0:5.0f}s {sid}", flush=True)
    torch.save(entries, SEL_FEATS)
    with open(OUT / "c_selection_zero_dump.pkl", "wb") as fh:
        pickle.dump(dump0, fh)
    with open(OUT / "c_selection_zero_rows.pkl", "wb") as fh:
        pickle.dump(rows, fh)

    # the cached-feature path once more, as every pass will run it
    _, rows_cached = E.selection_eval(agent_lora, E.load_selection_features(), RUN)
    out.update({
        "moments": len(rows), "scenes": len({r["scene_id"] for r in rows}),
        "checks": {**checks, "of": len(rows),
                   "drift_zero_lora_vs_plain": D.drift(dump0, dumpz)["summary"]},
        "cached_path_ade_equal": [r["ade"] for r in rows] == [r["ade"] for r in rows_cached],
        "selection_before": E.aggregate(rows),
        "chosen_index_hist": {int(i): int(n) for i, n in zip(*np.unique([d["chosen"] for d in dump0.values()], return_counts=True))},
        "wall_s": round(time.time() - t0, 1),
        "selection_features_file": str(SEL_FEATS), "selection_features_MB": round(SEL_FEATS.stat().st_size / 1e6, 1),
    })
    write_json(OUT / "c_reference.json", {**(json.loads((OUT / "c_reference.json").read_text()) if (OUT / "c_reference.json").exists() else {}),
                                          "C1": out})
    print(json.dumps(out, indent=1, default=str))


def fixed() -> None:
    import players as P

    data = S.load_splits()
    out = {"gpu_wait_s": wait_for_gpu(8192, "C.2")}
    t0 = time.time()
    fs = E.FixedSet(data["selection"])
    out["load_s"] = round(time.time() - t0, 1)
    ckpt = torch.load(C.AGENT_CKPT, map_location="cpu", mmap=True)["state_dict"]
    plain = P.build_vit()
    plain.load_state_dict(P.agent_encoder_state(ckpt), strict=True)
    del ckpt
    learner, key = P.make_learner_and_key(plain.cuda())
    guesser = P.build_guesser(torch.load(C.E50, map_location="cpu", mmap=True)).cuda()
    t0 = time.time()
    loss = fs.loss(learner, key, guesser)
    loss_again = fs.loss(learner, key, guesser)
    out.update({
        "pairs": len(fs.items), "scenes": len({s for s, _ in fs.items}), "batches": len(fs.masks),
        "hidden_counts_first_batch": [int(p.shape[1]) for _, p in fs.masks[0]],
        "items_sha256": __import__("hashlib").sha256(json.dumps(fs.items).encode()).hexdigest(),
        "zero_lora_loss": loss, "repeat_loss": loss_again, "repeat_identical": loss == loss_again,
        "eval_s": round((time.time() - t0) / 2, 1),
    })
    write_json(OUT / "c_reference.json", {**(json.loads((OUT / "c_reference.json").read_text()) if (OUT / "c_reference.json").exists() else {}),
                                          "C2": out})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"moments": moments, "fixed": fixed}[sys.argv[1]]()
