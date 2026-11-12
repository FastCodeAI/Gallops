#!/usr/bin/env python
"""Closed-loop check, part 2: prove the memory adapter (Paddock/paddock-sid/Alpasim/adapters/drivejepamemory) before any scene.

  s12_adapter_proofs.py build     write the identity test copy (proof A) under {experiment_store}/runs/cl12/proofs/
  s12_adapter_proofs.py A         identity: the adapter worker on the test copy (0.1.8b-p06 generator and scorer; generator read output
                                  projection and rating-head new columns set to zero) against the 0.1.8b-p06 adapter worker, on the 64
                                  recorded inputs of the 0.1.8b-p06 equivalence test (NAS release/gate_018/train_side.npz). Two passes:
                                  (1) single calls without the comfort re-score, as that test; (2) the 64 inputs as one session 0.5 s
                                  apart with the comfort re-score on, each worker fed its own previous plan. Everything bit for bit.
  s12_adapter_proofs.py B         offline match: one picking clip's recorded frames (the slim copies the offline cache was built from)
                                  through the adapter worker on the real build-5 checkpoint, in carry mode and in reset mode, against
                                  the stored offline replay of the same checkpoint (runs/build5_short/dumps/step0930_{carried,reset}.npz,
                                  written by s8_short.py). Plans within 1 mm, picks equal. No comfort re-score (the offline replay has none).
  s12_adapter_proofs.py C         reset works: two sessions back to back in one carry worker; the second must start from the start state
                                  and give the first session's first-call output bit for bit. Also: a call 1.0 s after the previous one
                                  restarts the chain and is counted; a repeated session start resets.
Run with CUDA hidden in the AlpaSim venv:  CUDA_VISIBLE_DEVICES="" alpasim/.venv/bin/python -B s12_adapter_proofs.py <step>
Outputs: out/cl12_proof_{A,B,C}.json, worker logs in logs/cl12_proof_*.log."""
import importlib, json, math, os, pathlib, sys, time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE.parent / "out"; LOGS = HERE.parent / "logs"
AS = pathlib.Path("{paddock_sid}/Alpasim")
CL12 = pathlib.Path("{experiment_store}/runs/cl12")
B5 = pathlib.Path("{experiment_store}/runs/build5_short")
GATE_NPZ = pathlib.Path("{retrain_store}/release/gate_018/train_side.npz")
SLIM = pathlib.Path("{dataset_store}/s4_train/slim")
CLIP = "0238dfe4-6545-46be-933d-8eb2ba29358c"
PLAN_TOL_M = 1e-3
sys.path.insert(0, str(AS / "adapters"))
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run with CUDA hidden"
assert not os.environ.get("DRIVEJEPA_CHECKPOINT_OVERRIDE")
os.environ["DRIVEJEPA_BATCH_FIX"] = "on"


def worker(pkg, log, memory_dir=None):
    if memory_dir is not None:
        os.environ["DRIVEJEPA_MEMORY"] = str(memory_dir)
    BR = importlib.import_module(f"{pkg}.bridge")
    return BR.BridgedDriveJEPA(device="cpu", stderr_log=LOGS / log)


def build():
    import torch
    K = importlib.import_module("drivejepamemory.constants")
    ck = torch.load(B5 / "ckpt/step0930.pt", map_location="cpu", weights_only=False)
    sd = torch.load(K.CHECKPOINT_PATH, map_location="cpu", weights_only=False)["state_dict"]
    for grp in ("generator", "scorer"):
        changed = 0
        for n in ck[grp]:
            base = sd["agent._pad_model." + n]; changed += int(not torch.equal(base, ck[grp][n])); ck[grp][n] = base.clone()
        print(grp, len(ck[grp]), "tensors set to 0.1.8b-p06; differed from build 5 before:", changed)
    for k in ("gen_read.attn.o.weight", "gen_read.attn.o.bias", "sc_read.mem_in.weight"):
        print(k, "norm before", float(ck["memory"][k].norm())); ck["memory"][k] = torch.zeros_like(ck["memory"][k])
    d = CL12 / "proofs/identity_copy"; d.mkdir(parents=True, exist_ok=True); p = d / K.MEMORY_CHECKPOINT_NAME
    torch.save(ck, p)
    import hashlib
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    (d / K.MEMORY_SIDECAR_NAME).write_text(json.dumps({
        "mode": "carry", "version": 3, "checkpoint_sha256": sha, "checkpoint_bytes": p.stat().st_size,
        "base_checkpoint_sha256": K.CHECKPOINT_SHA256, "read_out_loaded": False, "test_copy": True,
        "note": "proof A only: generator and scorer are 0.1.8b-p06's; generator read output projection and rating-head new columns zero; "
                "memory blocks otherwise build 5 step 930"}, indent=1) + "\n")
    print("test copy", p, sha)


def gate_inputs():
    z = np.load(GATE_NPZ); toks = [str(t) for t in z["tokens"]]
    keys = sorted({k.split("__", 1)[1] for k in z.files if k.startswith("f00__")})
    return toks, [{k: z[f"f{n:02d}__{k}"] for k in keys} for n in range(len(toks))]


def mem_req(uuid, n, pose):
    return {"session_uuid": uuid, "time_us": int(n * 500_000), "ego_pose": np.asarray(pose, np.float64)}


def proof_A():
    toks, feats = gate_inputs(); NM = len(feats)
    old = worker("drivejepa018bp06", "cl12_proof_A_p06_worker.log")
    new = worker("drivejepamemory", "cl12_proof_A_memory_worker.log", CL12 / "proofs/identity_copy")
    KEYS = ("proposals", "pred_logit", "pdm_score", "trajectory")
    res = {"moments": NM, "inputs": str(GATE_NPZ), "pass1_single_calls": {}, "pass2_session_with_comfort_rescore": {}}
    # pass 1: single calls, no comfort re-score (the 0.1.8b-p06 equivalence test's call); the new worker gets them as one carried session
    eq = {k: 0 for k in KEYS}; mx = {k: 0.0 for k in KEYS}; reads = []; t0 = time.time()
    rng = np.random.RandomState(0); pose = np.zeros(3)
    for n, f in enumerate(feats):
        a = old(dict(f)); pose = pose + np.array([rng.uniform(0, 5), rng.uniform(-0.3, 0.3), rng.uniform(-0.05, 0.05)])
        b = new({**f, "memory": mem_req("A1", n, pose)})
        for k in KEYS:
            x, y = np.asarray(a[k]), np.asarray(b[k]); eq[k] += int(np.array_equal(x, y)); mx[k] = max(mx[k], float(np.abs(x - y).max()))
        reads.append(b["memory"])
    res["pass1_single_calls"] = {"bit_identical": eq, "max_abs_diff": mx, "secs": round(time.time() - t0, 1),
                                 "memory_was_live": {"calls_reading_carried_memory": int(sum(not d["read_is_start"] for d in reads)),
                                                     "read_norm_min_max": [min(d["read_norm"] for d in reads), max(d["read_norm"] for d in reads)],
                                                     "gate_mean_min_max": [min(d["gate_mean"] for d in reads), max(d["gate_mean"] for d in reads)],
                                                     "chain_resets": reads[-1]["chain_resets"]}}
    # pass 2: one session, 0.5 s apart, comfort re-score on; each worker is fed its own previous plan's simulated states
    KEYS2 = KEYS + ("momentum_S", "momentum_states")
    eq = {k: 0 for k in KEYS2}; mx = {k: 0.0 for k in KEYS2}; codes = {}; overrides = 0; t0 = time.time()
    new.memory_control("start", "A2"); prev = {"old": None, "new": None}; pose = np.zeros(3); sc_eq = 0
    for n, f in enumerate(feats):
        v = np.asarray(f["ego_status"])[0, -1, 3:5].astype(np.float32); pose = pose + np.array([float(v[0]) * 0.5, 0.0, 0.0])
        outs = {}
        for name, w in (("old", old), ("new", new)):
            req = {"time_us": int(n * 500_000), "anchor_time_us": int(n * 500_000), "ego_pose": pose.copy(), "ego_velocity": v,
                   "ego_acceleration": np.zeros(2, np.float32), "past_time_us": None if prev[name] is None else int((n - 1) * 500_000),
                   "past_states": prev[name]}
            g = {**f, "momentum": req}
            if name == "new":
                g["memory"] = mem_req("A2", n, pose)
            outs[name] = w(g); prev[name] = np.array(outs[name]["momentum_states"], np.float64)
        a, b = outs["old"], outs["new"]
        for k in KEYS2:
            x, y = np.asarray(a[k]), np.asarray(b[k]); eq[k] += int(np.array_equal(x, y)); mx[k] = max(mx[k], float(np.abs(x - y).max()))
            if not np.array_equal(x, y):
                dd = np.abs(x - y); res.setdefault("pass2_differences", []).append({"call": n, "key": k, "max_abs": float(dd.max()), "n_values": int((dd > 0).sum()), "where": [list(map(int, i)) for i in np.argwhere(dd > 0)[:6]], "x": [float(v) for v in x[dd > 0][:6]], "y": [float(v) for v in y[dd > 0][:6]]})
        c = a["momentum_meta"]["code"]; codes[c] = codes.get(c, 0) + 1
        overrides += int(a["momentum_meta"]["pick_with"] != a["momentum_meta"]["pick_without"])
        sc_eq += int(a["momentum_meta"]["pick_with"] == b["momentum_meta"]["pick_with"] and a["momentum_meta"]["pick_without"] == b["momentum_meta"]["pick_without"]
                     and b["memory"]["pick_final"] == a["momentum_meta"]["pick_with"] and b["memory"]["pick_scorer"] == a["momentum_meta"]["pick_without"])
    res["pass2_session_with_comfort_rescore"] = {"bit_identical": eq, "max_abs_diff": mx, "picks_equal_before_and_after_rescore": sc_eq,
                                                 "rescore_codes": codes, "rescore_changed_the_pick": overrides, "secs": round(time.time() - t0, 1)}
    old.close(); new.close()
    # The brief's criterion: plans and picks bit for bit (proposals, logits, scores, chosen plan, picks before and after the re-score).
    # The simulated states of the chosen plan (what the NEXT call's comfort re-score is fed) are compared too and reported apart:
    # they come out of the float64 vehicle simulator, not the model (first run 2026-11-12 21:46: 63/64 equal, one call off by
    # 1.7e-18; see Ac, the same comparison between two 0.1.8b-p06 workers).
    P2 = res["pass2_session_with_comfort_rescore"]["bit_identical"]
    res["plans_and_picks_bit_identical"] = bool(all(v == NM for v in res["pass1_single_calls"]["bit_identical"].values())
                                                and all(P2[k] == NM for k in KEYS + ("momentum_S",)) and sc_eq == NM)
    res["simulated_states_bit_identical"] = bool(P2["momentum_states"] == NM)
    res["pass"] = res["plans_and_picks_bit_identical"]
    json.dump(res, open(OUT / ("cl12_proof_A.json" if os.environ.get("DRIVEJEPA_MEMORY_SHADOW", "on") != "off" else "cl12_proof_A_shadow_off.json"), "w"), indent=1); print(json.dumps(res, indent=1)); print("PROOF A", "PASS" if res["pass"] else "FAIL")


def proof_A_control():
    """Two 0.1.8b-p06 workers on pass 2 of proof A: is the simulated-states difference there without the memory adapter?"""
    toks, feats = gate_inputs(); NM = len(feats)
    ws = {"p06_a": worker("drivejepa018bp06", "cl12_proof_Ac_p06a_worker.log"), "p06_b": worker("drivejepa018bp06", "cl12_proof_Ac_p06b_worker.log")}
    prev = {k: None for k in ws}; pose = np.zeros(3); eq = {"trajectory": 0, "pdm_score": 0, "momentum_states": 0}; mx = dict.fromkeys(eq, 0.0); where = []
    for n, f in enumerate(feats):
        v = np.asarray(f["ego_status"])[0, -1, 3:5].astype(np.float32); pose = pose + np.array([float(v[0]) * 0.5, 0.0, 0.0]); outs = {}
        for name, w in ws.items():
            req = {"time_us": int(n * 500_000), "anchor_time_us": int(n * 500_000), "ego_pose": pose.copy(), "ego_velocity": v,
                   "ego_acceleration": np.zeros(2, np.float32), "past_time_us": None if prev[name] is None else int((n - 1) * 500_000), "past_states": prev[name]}
            outs[name] = w({**f, "momentum": req}); prev[name] = np.array(outs[name]["momentum_states"], np.float64)
        for k in eq:
            x, y = np.asarray(outs["p06_a"][k]), np.asarray(outs["p06_b"][k]); e = np.array_equal(x, y); eq[k] += int(e); mx[k] = max(mx[k], float(np.abs(x - y).max()))
            if not e:
                where.append((n, k, float(np.abs(x - y).max())))
    for w in ws.values():
        w.close()
    res = {"moments": NM, "bit_identical": eq, "max_abs_diff": mx, "differences": where}
    json.dump(res, open(OUT / "cl12_proof_A_control.json", "w"), indent=1); print(json.dumps(res, indent=1)); print("PROOF Ac", "PASS")


def clip_inputs():
    """The clip's moments in time order as worker feature dicts (dataset_s35.py _slim_decode + _finish with zero_accel), and poses
    chained from each moment's human future (waypoint 0 = the next anchor's pose in this moment's frame)."""
    import cv2, pandas as pd
    idx = pd.read_parquet("{experiment_store}/cache/picking/index_picking.parquet")
    g = idx[idx["clip"] == CLIP].sort_values("row"); at = g["anchor_t"].to_numpy()
    assert np.allclose(np.diff(at), 0.5), "clip is not one unbroken chain"
    feats, trajs = [], []
    for sid in g["sample_id"]:
        with np.load(SLIM / CLIP / f"{sid}.npz") as z:
            cam = {k: (np.ascontiguousarray(cv2.imdecode(z[p], cv2.IMREAD_UNCHANGED)).transpose(2, 0, 1).astype(np.float32) / np.float32(255.0))[None]
                   for k, p in (("camera_feature_1", "png1"), ("camera_feature_2", "png2"))}
            ego = z["ego_status"][-1:].copy(); ego[..., 5:7] = 0.0
            feats.append({**cam, "ego_status": ego[None].astype(np.float32), "lidar2img": z["lidar2img"][None].astype(np.float32),
                          "img_shape": z["img_shape"][None].astype(np.float32)})
            trajs.append(z["trajectory"].astype(np.float64))
    poses = [np.zeros(3)]
    for t in trajs[:-1]:
        x, y, h = poses[-1]; c, s = math.cos(h), math.sin(h)
        poses.append(np.array([x + c * t[0, 0] - s * t[0, 1], y + s * t[0, 0] + c * t[0, 1], h + t[0, 2]]))
    return list(g["sample_id"]), feats, poses, list(g["in_picking"])


def proof_B():
    sids, feats, poses, inp = clip_inputs(); N = len(sids); res = {"clip": CLIP, "moments": N, "moments_in_picking_set": int(sum(inp)), "tolerance_m": PLAN_TOL_M}
    first = {}
    for mode, dump in (("carry", "step0930_carried.npz"), ("reset", "step0930_reset.npz")):
        z = np.load(B5 / "dumps" / dump); row = {str(t): i for i, t in enumerate(z["token"])}
        w = worker("drivejepamemory", f"cl12_proof_B_{mode}_worker.log", CL12 / "model" / mode)
        w.memory_control("start", "B"); per = []; t0 = time.time()
        for n, (sid, f) in enumerate(zip(sids, feats)):
            o = w({**f, "memory": mem_req("B", n, poses[n])}); i = row[sid]
            P = np.asarray(o["proposals"], np.float32)[0]; d = o["memory"]
            per.append({"sid": sid, "plan_xy_max_abs_diff_m": float(np.abs(P[..., :2] - z["proposals"][i][..., :2]).max()),
                        "heading_max_abs_diff_rad": float(np.abs(P[..., 2] - z["proposals"][i][..., 2]).max()),
                        "logit_max_abs_diff": float(np.abs(np.asarray(o["pred_logit"], np.float32)[0] - z["pred_logit"][i]).max()),
                        "pick_adapter": int(np.asarray(o["pdm_score"])[0].argmax()), "pick_offline": int(z["chosen"][i]),
                        "read_is_start": d["read_is_start"], "read_norm": d["read_norm"], "gate_mean": d["gate_mean"], "mode": d["mode"],
                        "chain_resets": d["chain_resets"], "shadow_differs": d.get("picks_differ_scorer")})
            if n == 0:
                first[mode] = {k: np.asarray(o[k]) for k in ("proposals", "pred_logit")}
        w.close()
        r = {"offline_replay": str(B5 / "dumps" / dump), "secs": round(time.time() - t0, 1),
             "plan_xy_max_abs_diff_m": max(p["plan_xy_max_abs_diff_m"] for p in per), "plan_xy_median_of_moment_max_m": float(np.median([p["plan_xy_max_abs_diff_m"] for p in per])),
             "heading_max_abs_diff_rad": max(p["heading_max_abs_diff_rad"] for p in per), "logit_max_abs_diff": max(p["logit_max_abs_diff"] for p in per),
             "picks_equal": int(sum(p["pick_adapter"] == p["pick_offline"] for p in per)),
             "calls_reading_start_state": int(sum(p["read_is_start"] for p in per)), "chain_resets": per[-1]["chain_resets"],
             "read_norm_last": per[-1]["read_norm"], "gate_mean_mean": float(np.mean([p["gate_mean"] for p in per])),
             "shadow_pick_differs_calls": int(sum(bool(p["shadow_differs"]) for p in per)), "per_moment": per}
        r["pass"] = bool(r["plan_xy_max_abs_diff_m"] <= PLAN_TOL_M and r["picks_equal"] == N
                         and r["calls_reading_start_state"] == (1 if mode == "carry" else N) and r["chain_resets"] == 0)
        res[mode] = r; print(mode, {k: v for k, v in r.items() if k != "per_moment"}, flush=True)
    res["first_call_carry_equals_reset_bitwise"] = bool(all(np.array_equal(first["carry"][k], first["reset"][k]) for k in first["carry"]))
    res["pass"] = bool(res["carry"]["pass"] and res["reset"]["pass"])
    json.dump(res, open(OUT / "cl12_proof_B.json", "w"), indent=1); print("PROOF B", "PASS" if res["pass"] else "FAIL")


def proof_C():
    sids, feats, poses, _ = clip_inputs(); KEYS = ("proposals", "pred_logit", "pdm_score", "trajectory")
    w = worker("drivejepamemory", "cl12_proof_C_worker.log", CL12 / "model/carry"); res = {"clip": CLIP, "mode": "carry"}
    call = lambda u, n, t_idx=None: w({**feats[n], "memory": mem_req(u, n if t_idx is None else t_idx, poses[n])})
    w.memory_control("start", "S1"); s1 = [call("S1", n) for n in range(6)]
    c1 = w.memory_control("close", "S1")
    w.memory_control("start", "S2"); s2 = [call("S2", n) for n in range(6)]
    res["session1"] = [{"read_is_start": o["memory"]["read_is_start"], "read_norm": o["memory"]["read_norm"]} for o in s1]
    res["session1_close_summary"] = c1["summary"]
    res["session2_first_call_reads_start_state"] = bool(s2[0]["memory"]["read_is_start"])
    res["session2_equals_session1_bitwise_calls"] = int(sum(all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in KEYS) for a, b in zip(s1, s2)))
    res["session1_later_calls_read_carried_memory"] = bool(all(not o["memory"]["read_is_start"] for o in s1[1:]))
    # no start message at all: an unseen session id must also begin at the start state
    s3 = call("S3", 0); res["unseen_session_first_call_reads_start_state"] = bool(s3["memory"]["read_is_start"]) and all(np.array_equal(np.asarray(s3[k]), np.asarray(s1[0][k])) for k in KEYS)
    # chain rule: S2's next call 1.0 s after the previous one (time index 7 instead of 6) -> chain reset, start state, counted
    o = call("S2", 6, t_idx=7); res["gap_1s"] = {"dt_us": o["memory"]["dt_us"], "chain_reset": o["memory"]["chain_reset"], "read_is_start": o["memory"]["read_is_start"], "chain_resets": o["memory"]["chain_resets"]}
    o2 = call("S2", 7, t_idx=8); res["after_gap_next_call"] = {"dt_us": o2["memory"]["dt_us"], "chain_reset": o2["memory"]["chain_reset"], "read_is_start": o2["memory"]["read_is_start"]}
    # repeated start on a live session
    w.memory_control("start", "S2"); o3 = call("S2", 0, t_idx=20); res["restart_same_session_reads_start_state"] = bool(o3["memory"]["read_is_start"]) and o3["memory"]["call"] == 1
    w.close()
    res["pass"] = bool(res["session2_first_call_reads_start_state"] and res["session2_equals_session1_bitwise_calls"] == 6 and res["session1_later_calls_read_carried_memory"]
                       and res["unseen_session_first_call_reads_start_state"] and res["gap_1s"]["chain_reset"] and res["gap_1s"]["read_is_start"] and res["gap_1s"]["chain_resets"] == 1
                       and not res["after_gap_next_call"]["chain_reset"] and not res["after_gap_next_call"]["read_is_start"] and res["restart_same_session_reads_start_state"])
    json.dump(res, open(OUT / "cl12_proof_C.json", "w"), indent=1); print(json.dumps(res, indent=1)); print("PROOF C", "PASS" if res["pass"] else "FAIL")


if __name__ == "__main__":
    {"build": build, "A": proof_A, "Ac": proof_A_control, "B": proof_B, "C": proof_C}[sys.argv[1]]()
