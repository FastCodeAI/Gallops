#!/usr/bin/env python
"""S9 C.4, EXAM CLIPS, INSPECTION ONLY, NOT AN EVALUATION. Turns the model inputs saved by s9_trial_inputs.py into the 256 features per
plan (slot_9) with the pass-9 weights, CPU, same split forward as s6_dump.py / s9_c_dump.py; checks that the 32 plans equal the ones
the trial stored; writes c4_trial_features.npz (features, plans, scores; no camera frames) and deletes the NAS scratch folder with the
frames. Run with run_s9_dump.sh-style env (see run_s9_trial_features.sh)."""
import os, sys, glob, shutil, pathlib
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
HERE = pathlib.Path(__file__).resolve().parent; RETRAIN = HERE.parent; sys.path.insert(0, str(RETRAIN / "05_small_overfit_check/train_patch"))
import s35_common as C; C.setup_env()
import numpy as np, torch; torch.set_num_threads(4)
TMP = pathlib.Path("{runs}/drivejepa_retrain/11_scorer_head_and_pullout_grades/trial_tmp"); TRIAL = RETRAIN / "07_closed_loop_test" / "menus"
agent = C.build_agent(vocab="ext", freeze_encoder=True, init=True, scorer_impl="s3", pool_workers=0); agent.eval(); m = agent._pad_model
sd = torch.load(C.NAS / "06_first_training_run/run/ckpt/pass_09.pt", map_location="cpu"); assert sd["passno"] == 9; missing, unexpected = m.load_state_dict({k[len("_pad_model."):]: v for k, v in sd["trainable"].items()}, strict=False); assert not unexpected and all(k.startswith("_backbone") for k in missing)
R = {}; put = lambda k, v: R.setdefault(k, []).append(v); worst = 0.0; worst_s = 0.0
with torch.no_grad():
    for sc in ("d2715023", "b44a4a44"):
        for fn in sorted(glob.glob(str(TMP / sc / "replan_*.npz"))):
            z = np.load(fn, allow_pickle=True); f = {k: torch.from_numpy(np.ascontiguousarray(z[k])) for k in ("camera_feature_1", "camera_feature_2", "ego_status", "img_shape", "lidar2img")}
            f["lidar2img"] = f["lidar2img"][:, 1:2]; ego = f["ego_status"][:, -1]; cam = torch.cat([m.transform(f["camera_feature_2"])[:, None], m.transform(f["camera_feature_1"])[:, None]], dim=1); imf = m._backbone(cam, img_metas=f)
            bev = m.hist_encoding(ego)[:, None] + m.init_feature.weight[None]
            for refine in m._trajectory_head: bev, props = refine(bev, imf)
            slot = bev.reshape(1, 32, 8, -1).amax(-2); logit = m.scorer.pred_score.mlp[-1](m.scorer.pred_score.mlp[:-1](slot)); P = props[0].numpy(); k = int(z["idx"]); st = np.load(TRIAL / f"d016_{sc}" / f"replan_{k:03d}.npz", allow_pickle=True)
            dev = float(np.abs(P - st["proposals"]).max()); ds = float(np.abs(torch.sigmoid(logit)[0, :, -1].numpy() - st["S_plain"]).max()); worst = max(worst, dev); worst_s = max(worst_s, ds)
            put("scene", sc); put("idx", k); put("t_after_ho", float(z["t_after_ho"])); put("menu", P.astype(np.float32)); put("slot_9", slot[0].numpy().astype(np.float32)); put("logit_9", logit[0].numpy().astype(np.float32)); put("S_plain", st["S_plain"].astype(np.float32)); put("pdm_score", st["pdm_score"].astype(np.float32))
            put("chosen", int(st["chosen"])); put("ego_v", float(st["ego_v"])); put("plan_dev_vs_trial_m", dev); put("score_dev_vs_trial", ds)
print("replans", len(R["idx"]), "| worst plan difference vs the trial's stored plans %.2e m | worst raw-score difference %.2e" % (worst, worst_s)); assert worst < 1e-3 and worst_s < 1e-3, "recomputed plans do not match the trial's"
np.savez(HERE / "c4_trial_features.npz", **{k: np.stack(v) for k, v in R.items()}); shutil.rmtree(TMP, ignore_errors=True); print("features written; scratch folder with the frames deleted:", not TMP.exists())
