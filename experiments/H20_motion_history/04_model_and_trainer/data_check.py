"""Build 2, step 2 checks (CPU only): (a) the loader's per-moment values against 0.1.8b's planner_dataset on 100 random training moments;
(b) one full pass of 0.1.8b's sampler (pass 1, seed 0) through the loader, timed; every manifest moment must be served.

(a) planner_dataset = pai_data.planner_dataset(agent, tr, "train") exactly as train_018b.train() builds it; the agent object only supplies
feature/target builders (no weights are read here, so it is constructed without initialize()). Compared bit for bit: ego_status (1, 11)
after zero_accel, trajectory, lidar2img, img_shape, has_label2, metric_cache_path, token. Output MH/out/data_check.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parents[1] / "out"


def main():
    import s6_env  # noqa: F401  (approach 3's navsim path shim; import first)
    import pai_common as C
    C.setup_env()
    import pai_train as T
    import pai_data as PD
    import a4_data as D
    tr, pk, _ = T.manifests()
    res = {}
    t0 = time.time(); ds = D.training_set(tr); res["build_s"] = round(time.time() - t0, 1)

    # (a) 100 moments vs planner_dataset
    from drive_jepa_agent_s10 import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path="", vocab_path=str(C.VOCAB), index_kind="ext",
                           init_checkpoint="", freeze_encoder=True, scorer_impl="s3", pool_workers=0)
    T._module_level_slim()
    ref = PD.planner_dataset(agent, tr, "train")
    pick = np.random.default_rng(0).choice(len(tr), 100, replace=False)
    eq = {k: 0 for k in ("ego_status", "trajectory", "lidar2img", "img_shape", "has_label2", "metric_cache_path", "token")}
    dtypes = {}
    for i in pick:
        f0, t0_ = ref[int(i)]
        f1, t1 = ds.item(int(i))
        for k in ("ego_status", "lidar2img", "img_shape"):
            eq[k] += int(f0[k].dtype == f1[k].dtype and f0[k].shape == f1[k].shape and torch.equal(f0[k], f1[k]))
            dtypes[k] = [str(f0[k].dtype), list(f0[k].shape), str(f1[k].dtype), list(f1[k].shape)]
        eq["trajectory"] += int(t0_["trajectory"].dtype == t1["trajectory"].dtype and torch.equal(t0_["trajectory"], t1["trajectory"]))
        dtypes["trajectory"] = [str(t0_["trajectory"].dtype), list(t0_["trajectory"].shape), str(t1["trajectory"].dtype)]
        eq["has_label2"] += int(bool(t0_["has_label2"]) == bool(t1["has_label2"]))
        eq["metric_cache_path"] += int(t0_["metric_cache_path"] == t1["metric_cache_path"])
        eq["token"] += int(t0_["token"] == t1["token"])
    res["vs_planner_dataset"] = {"moments": 100, "equal": eq, "dtypes_shapes": dtypes,
                                 "label2_in_sample": int(sum(bool(tr.has_label2.iloc[int(i)]) for i in pick))}
    print(json.dumps(res["vs_planner_dataset"]), flush=True)

    if sys.argv[1:] == ["equality-only"]:
        (OUT / "data_check_equality.json").write_text(json.dumps(res, indent=1))
        assert all(v == 100 for v in eq.values()), eq
        return
    # (b) one full pass of 0.1.8b's sampler
    bl = PD.batches_017(PD.sampler_017(tr, T.SEED, 1), T.BATCH)
    seen = np.zeros(len(tr), bool); t0 = time.time(); n = 0; waits = []
    it = D.Prefetch(ds, bl)
    for b, idx in zip(it, bl):
        f, t = b
        assert f["tokens"].shape == (len(idx), 512, 256) and f["ego_history"].shape == (len(idx), 8, 6)
        seen[idx] = True; n += 1; waits.append(it.last_wait_s)
        if n % 500 == 0:
            print(f"{n}/{len(bl)} batches {time.time() - t0:.0f} s", flush=True)
    dt = time.time() - t0
    res["full_pass"] = {"batches": len(bl), "draws": int(sum(len(b) for b in bl)), "moments_served": int(seen.sum()), "manifest": len(tr),
                        "wall_s": round(dt, 1), "s_per_batch": dt / len(bl), "wait_s_per_batch_mean": float(np.mean(waits))}
    assert seen.all()
    OUT.mkdir(exist_ok=True)
    (OUT / "data_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["full_pass"]))
    assert all(v == 100 for v in eq.values()), eq


if __name__ == "__main__":
    sys.exit(main())
