#!/usr/bin/env python
"""S4 offline sanity, step A (alpasim venv, CPU): three examples through the ADAPTER's own preprocessing with the 0.1.6 patch applied to a
scratch copy (adapter_016/_import, never under adapters/): raw clip frame -> rectify -> planner._pipeline_a (mask inside the planner) ->
planner._model_ego_status (acceleration zeroed). Writes sanity/features_<token>.npz for step B.
Run: cd paddock-sid/Alpasim && CUDA_VISIBLE_DEVICES="" alpasim/.venv/bin/python analysis_3.0/Retrain-data/06_first_training_run/sanity_a_preprocess.py"""
import sys, pathlib, numpy as np, pandas as pd, cv2
HERE = pathlib.Path(__file__).resolve().parent; S3 = HERE.parent / "04_training_set"; sys.path.insert(0, str(S3))
import s3_common as C; import s1_common as C1; C1.NAS = C.S1ROOT; import s1_convert as S
sys.path.insert(0, str(HERE / "adapter_016/_import"))
from drivejepa016 import planner as P16, constants as K16
assert K16.ADAPTER_VERSION == "0.1.6" and K16.BONNET_MASK_ROWS == 37 and K16.ZERO_ACCELERATION and "adapter_016" in P16.__file__
man = pd.read_parquet(S3 / "manifest.parquet"); clips = pd.read_csv(S3 / "clips.csv").set_index("uuid"); OUT = HERE / "sanity"; OUT.mkdir(exist_ok=True)
pick = pd.concat([man[(man.type == "A") & man.has_label2 & (man.speed > 25)].sample(1, random_state=1), man[(man.type == "B_pass") & man.has_label2 & (man.speed > 3)].sample(1, random_state=2), man[(man.source == "raw") & (man.type == "ordinary") & (man.speed > 5)].sample(1, random_state=3)])
C._assert_train(pick.clip_uuid); S.P = P16; S.BONNET_MASK_ROWS = 0           # the converter's composition, but every planner function is 0.1.6's and the mask is the PLANNER's
for r in pick.itertuples():
    d, ts, ego, ft, R_N, t_N = S.load_clip(r.clip_uuid, int(clips.loc[r.clip_uuid, "chunk"])); t = int(round(r.anchor_t * 1e6)); near = lambda q: int(np.argmin(np.abs(ts - q))); jo, jn = near(t - S.STEP_US), near(t)
    maps = S.build_maps(ft, R_N); fr = S.decode(d / f"{r.clip_uuid}.{C.FRONT}.mp4", sorted({jo, jn})); rect = {j: cv2.cvtColor(S.rectify(f, maps), cv2.COLOR_BGR2RGB) for j, f in fr.items()}
    feats, label, command, route, lag, _ = S.sample(ego, t, None, rect[jo], rect[jn]); reported = feats["ego_status"].copy(); feats["ego_status"] = P16.DriveJEPAPlanner._model_ego_status(feats["ego_status"])
    np.savez(OUT / f"features_{r.sample_id}.npz", **feats, label=label, reported_ego_status=reported, clip=r.clip_uuid, type=r.type, has_label2=r.has_label2)
    print(r.sample_id, r.type, f"{r.speed:.1f} m/s | masked rows max", float(feats["camera_feature_1"][:, -37:].max()), "| accel fed", feats["ego_status"][-1, 5:7].tolist(), "reported", reported[-1, 5:7].round(3).tolist(), flush=True)
pick[["sample_id", "clip_uuid", "type", "source", "speed", "has_label2"]].to_csv(OUT / "picks.csv", index=False)
