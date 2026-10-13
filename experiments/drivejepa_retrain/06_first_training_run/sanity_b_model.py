#!/usr/bin/env python
"""S4 offline sanity, step B (conda drive-jepa python, SHIPPED torch 2.1, CPU): the same three examples through (1) the adapter's model
wrapper adapters/drivejepa/real_model.RealDriveJEPA with the candidate checkpoint, fed the step-A features, and (2) the training-side
loader + agent. Asserts bit-identical inputs and the same chosen plan. Usage: sanity_b_model.py <candidate.ckpt>"""
import sys, json, pathlib, numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "05_small_overfit_check/train_patch")); import s35_common as C; C.setup_env()
import torch, pandas as pd
from torch.utils.data.dataloader import default_collate
import s3_common as C3
ck = sys.argv[1]; picks = pd.read_csv(HERE / "sanity/picks.csv"); man = pd.read_parquet(C.S3 / "manifest.parquet"); man = man[man.sample_id.isin(picks.sample_id)]
sys.path.insert(0, str(C.RETRAIN.parents[1] / "adapters")); from drivejepa.real_model import RealDriveJEPA
adapter_model = RealDriveJEPA(checkpoint_path=ck, device="cpu")
agent = C.build_agent(vocab="ext", init=False, pool_workers=0); sd = torch.load(ck, map_location="cpu")["state_dict"]; agent.load_state_dict({k.replace("agent._pad_model", "_pad_model"): v for k, v in sd.items()}, strict=True); agent.eval()
from dataset_s35 import CacheOnlyDataset
ds = CacheOnlyDataset(cache_path=str(C.NAS / "04_training_set/s1/cache"), feature_builders=agent.get_feature_builders(), target_builders=agent.get_target_builders(), zero_accel=True, has_label2=dict(zip(man.sample_id, man.has_label2)),
                      token_clips=dict(zip(man.sample_id, man.clip_uuid)), slim_root=str(C.NAS / "06_first_training_run/slim"), split_guard=lambda u: C3._assert_train(list(u)))
rows = []
for tok in picks.sample_id:
    z = np.load(HERE / f"sanity/features_{tok}.npz"); f_tr, t_tr = ds[ds.tokens.index(tok)]
    same = {k: bool(torch.equal(torch.from_numpy(z[k]), f_tr[k])) for k in ("camera_feature_1", "camera_feature_2", "lidar2img", "img_shape")}; same["ego_status (row the model reads)"] = bool(torch.equal(torch.from_numpy(z["ego_status"][-1:]), f_tr["ego_status"]))
    out_a = adapter_model.forward({k: z[k][None] for k in ("camera_feature_1", "camera_feature_2", "camera_feature", "lidar2img", "img_shape", "ego_status")})
    with torch.no_grad(): out_t = agent.forward({k: v.clone() for k, v in default_collate([(f_tr, t_tr)])[0].items()})
    pa, pt = out_a["proposals"][0], out_t["proposals"][0].numpy(); ca, ct = int(out_a["pdm_score"][0].argmax()), int(out_t["pdm_score"][0].argmax())
    rows.append(dict(token=tok, **same, inputs_bit_identical=all(same.values()), max_abs_diff_proposals_m=float(np.abs(pa - pt).max()), max_abs_diff_scores=float(np.abs(out_a["pdm_score"][0] - out_t["pdm_score"][0].numpy()).max()), adapter_pick=ca, training_side_pick=ct, same_pick=ca == ct,
                     chosen_plan_max_abs_diff_m=float(np.abs(out_a["trajectory"][0] - out_t["trajectory"][0].numpy()).max()), chosen_end_x_m=float(pa[ca, -1, 0]), human_end_x_m=float(z["label"][-1, 0])))
    print(json.dumps(rows[-1]), flush=True)
pd.DataFrame(rows).to_csv(HERE / "sanity/result.csv", index=False); ok = all(r["inputs_bit_identical"] and r["same_pick"] for r in rows); print("SANITY", "PASS" if ok else "FAIL", "| torch", torch.__version__, "| checkpoint", ck)
