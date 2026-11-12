"""Two closing checks for encoder LoRA on KIT.  No training.

    ./run.sh src/close.py random     # Check 1: random-change control, 103 selection moments
    ./run.sh src/close.py zero223    # Check 2: zero LoRA on val + overlap_train_val (223)

Check 1.  For each of the 96 LoRA layers, the trained pass-3 change is
dW = (alpha/r) * B @ A (lora.LoRALinear.forward / merged_weight, s1 lora.py).
A random LoRA draws A, B ~ N(0, 1) with seed 1..5 and scales B so that its
||(alpha/r) * B @ A||_F equals the pass-3 layer's.  Each is evaluated exactly as
the s2 per-pass check: agent with the encoder wrapped by LoRA (fp32, unmerged),
evalsets.selection_eval on the saved selection features (s2 train.py `evaluate`),
drift vs the s2 zero-LoRA dump (s1 drift.py), best-of-32 and plan speed as in
s2 g_extra.py.  Pass 3 itself is re-run in the same process as a reproduction
check against its recorded ADE.

Check 2.  The s2 final-evaluation loop (s2 f_final.py, = s1 f_baseline.py's
RUN.main loop with --splits val overlap_train_val --protocol e2e --command gt
--eval-horizon 3) with the plain agent, i.e. zero LoRA = the original encoder.
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
import f_baseline as FB
import g_extra as G
import lora as L
from s2common import CKPT, OUT as S2OUT, wait_for_gpu, write_json

RUN = C.kit_runner_module()
ADP = C.kit_adapter_module()
OUT = Path(__file__).resolve().parent.parent / "out"
SEEDS = [1, 2, 3, 4, 5]


def build_agent_lora():
    assert C.sha256_file(C.AGENT_CKPT) == C.AGENT_CKPT_SHA256
    agent = RUN.build_agent(C.AGENT_CKPT, "drive_jepa_perception_based_agent")
    for p in agent.parameters():
        p.requires_grad_(False)
    agent_lora = copy.deepcopy(agent)
    L.wrap(agent_lora.get_submodule(E.ENC_PATH))
    agent_lora.eval()
    return agent, agent_lora


def set_lora(enc, sd: dict) -> None:
    params = dict(enc.named_parameters())
    assert set(sd) == {n for n in params if ".lora_" in n}
    with torch.no_grad():
        for n, t in sd.items():
            params[n].copy_(t.to(params[n].device, params[n].dtype))


def layer_norms(sd: dict, scale: float) -> dict:
    """layer name -> ||scale * B @ A||_F (float64)."""
    names = sorted({n.rsplit(".", 1)[0] for n in sd})
    return {m: float(torch.linalg.matrix_norm(scale * sd[m + ".lora_B"].double() @ sd[m + ".lora_A"].double()))
            for m in names}


def random_lora(trained: dict, seed: int, scale: float) -> dict:
    target = layer_norms(trained, scale)
    g = torch.Generator().manual_seed(seed)
    out = {}
    for m in sorted(target):
        A0, B0 = trained[m + ".lora_A"], trained[m + ".lora_B"]
        A = torch.randn(A0.shape, generator=g, dtype=torch.float64)
        B = torch.randn(B0.shape, generator=g, dtype=torch.float64)
        B *= target[m] / float(torch.linalg.matrix_norm(scale * B @ A))
        out[m + ".lora_A"], out[m + ".lora_B"] = A.float(), B.float()
    return out


def summarise(dump, rows, zero, human, dt) -> dict:
    keys = sorted(dump)
    dr = D.drift(zero, dump)["summary"]
    ades = [r["ade"] for r in rows]
    best = [min(G.ade3(q, human[k], dt) for q in dump[k]["proposals"]) for k in keys]
    return {"ade_mean": float(np.mean(ades)), "ade_median": float(np.median(ades)),
            "fde3_mean": float(np.mean([r["fde@3s"] for r in rows])),
            "best_of_32_ade_mean": float(np.mean(best)),
            "mean_proposal_shift_m": dr["mean_proposal_shift_m"], "max_proposal_shift_m": dr["max_proposal_shift_m"],
            "mean_abs_grade_change": dr["mean_grade_change"], "picks_changed": dr["chosen_changed"],
            "plan_speed_mean_mps": float(np.mean([G.plan_stats(dump[k]["trajectory"])[0] for k in keys]))}


def random_check() -> None:
    out = {"gpu_wait_s": wait_for_gpu(8192, "check 1")}
    t0 = time.time()
    ck = torch.load(CKPT / "pass_03" / "state.pt", map_location="cpu", weights_only=False)
    trained = ck["lora"]
    scale = L.ALPHA / L.RANK
    _, agent_lora = build_agent_lora()
    enc = agent_lora.get_submodule(E.ENC_PATH)
    sel = E.load_selection_features()
    zero = pickle.load(open(S2OUT / "c_selection_zero_dump.pkl", "rb"))
    zrows = pickle.load(open(S2OUT / "c_selection_zero_rows.pkl", "rb"))
    human = {(r["scene_id"], r["keyframe_index"]): np.asarray(r["human_poses"]) for r in zrows}
    dt = 0.5
    assert all(abs(e["future_interval_s"] - dt) < 1e-9 for e in sel)

    table = {"zero": summarise(zero, zrows, zero, human, dt)}
    set_lora(enc, trained)
    dump, rows = E.selection_eval(agent_lora, sel, RUN)
    table["pass3"] = summarise(dump, rows, zero, human, dt)
    out["pass3_reproduces_recorded_ade"] = {"now": table["pass3"]["ade_mean"],
                                            "recorded": ck["row"]["selection"]["ade"]["mean"]}
    tn = layer_norms(trained, scale)
    out["trained_layer_norms"] = {"sum": sum(tn.values()), "min": min(tn.values()), "max": max(tn.values())}
    for s in SEEDS:
        rl = random_lora(trained, s, scale)
        rn = layer_norms(rl, scale)
        worst = max(abs(rn[m] - tn[m]) / tn[m] for m in tn)
        set_lora(enc, rl)
        dump, rows = E.selection_eval(agent_lora, sel, RUN)
        table[f"seed{s}"] = {**summarise(dump, rows, zero, human, dt), "norm_match_worst_rel_err": worst}
        with open(OUT / f"random_seed{s}_dump.pkl", "wb") as fh:
            pickle.dump(dump, fh)
        print(s, table[f"seed{s}"], flush=True)
    seeds = [table[f"seed{s}"] for s in SEEDS]
    table["random_mean"] = {k: float(np.mean([r[k] for r in seeds])) for k in table["pass3"]}
    table["random_range"] = {k: [float(min(r[k] for r in seeds)), float(max(r[k] for r in seeds))] for k in table["pass3"]}
    p3 = table["pass3"]["ade_mean"]
    beats_all = all(r["ade_mean"] - p3 > 0.1 for r in seeds)
    near_mean = abs(table["random_mean"]["ade_mean"] - p3) <= 0.1
    out["verdict"] = "TRAINED-MATTERS" if beats_all else ("DRIFT" if near_mean else "MIXED")
    out["verdict_inputs"] = {"pass3_ade": p3, "random_ades": [r["ade_mean"] for r in seeds],
                             "margin_to_best_random": min(r["ade_mean"] for r in seeds) - p3,
                             "random_mean_minus_pass3": table["random_mean"]["ade_mean"] - p3}
    out["table"] = table
    out["gpu_process_s"] = round(time.time() - t0, 1)
    write_json(OUT / "check1_random.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "table"}, indent=1))


def zero223() -> None:
    out = {"gpu_wait_s": wait_for_gpu(8192, "check 2")}
    t0 = time.time()
    agent, _ = build_agent_lora()
    out["root"] = FB.build_root()
    assert C.sha256_file(C.BASELINE_PKL) == C.BASELINE_PKL_SHA256
    aug = pickle.load(open(C.BASELINE_PKL, "rb"))
    base = {(r["scene_id"], int(r["keyframe_index"])): r for r in aug}
    n_future = agent._trajectory_sampling.num_poses
    targets = [(sp, d.name) for sp in FB.SPLITS for d in sorted((FB.ROOT / "data" / sp).iterdir()) if d.is_dir()]

    def prepare(t):
        split, sid = t
        adapter = ADP.KITScenesSceneAdapter(FB.ROOT / "data" / split / sid)  # RUN:232
        kfs = adapter.valid_keyframes(num_future_frames=10, stride=90, first_frame=40)  # RUN:236-238
        return split, sid, [(kf, adapter.build_sample(kf, num_future_frames=n_future, command_mode="gt")) for kf in kfs]

    rows, dump = [], {}
    with ThreadPoolExecutor(8) as pool:
        for split, sid, samples in pool.map(prepare, targets):
            for kf, sample in samples:
                predicted = agent.compute_trajectory(sample.agent_input)  # RUN:256
                m = RUN.displacement_errors(predicted.poses, sample.future_poses, sample.future_interval_s, 3.0)
                d = FB.forward_dump(agent, sample.agent_input)
                assert L.bit_equal(torch.from_numpy(d["trajectory"]), torch.from_numpy(predicted.poses.astype(np.float32)))
                assert np.array_equal(base[(sid, kf)]["human_poses"], sample.future_poses)
                dump[(sid, kf)] = d
                rows.append({"scene_id": sid, "split": split, "keyframe_index": kf, "predicted_poses": predicted.poses,
                             "human_poses": sample.future_poses, **m})
    out["wall_s"] = round(time.time() - t0, 1)
    with open(OUT / "zero223_rows.pkl", "wb") as fh:
        pickle.dump(rows, fh)
    with open(OUT / "zero223_dump.pkl", "wb") as fh:
        pickle.dump(dump, fh)
    assert {(r["scene_id"], r["keyframe_index"]) for r in rows} == set(base)

    # vs August
    diff = np.array([r["ade"] - base[(r["scene_id"], r["keyframe_index"])]["ade"] for r in rows])
    pose = np.array([np.abs(np.asarray(base[(r["scene_id"], r["keyframe_index"])]["predicted_poses"], dtype=np.float64)
                            - r["predicted_poses"]).max() for r in rows])
    out["vs_august"] = {"ade_diff_mean": float(diff.mean()), "ade_absdiff_mean": float(np.abs(diff).mean()),
                        "ade_absdiff_max": float(np.abs(diff).max()), "pose_absdiff_max_m": float(pose.max()),
                        "pose_absdiff_median_m": float(np.median(pose)),
                        "bit_identical_moments": int(sum(
                            np.asarray(base[(r["scene_id"], r["keyframe_index"])]["predicted_poses"]).tobytes()
                            == r["predicted_poses"].tobytes() for r in rows)),
                        "moments": len(rows)}

    # vs pass 3 (s2 Part F dump)
    p3d = pickle.load(open(S2OUT / "f_final_dump.pkl", "rb"))
    p3r = {(r["scene_id"], r["keyframe_index"]): r for r in pickle.load(open(S2OUT / "f_final_rows.pkl", "rb"))}
    human = {(r["scene_id"], r["keyframe_index"]): np.asarray(r["human_poses"]) for r in rows}
    split_of = {(r["scene_id"], r["keyframe_index"]): r["split"] for r in rows}
    zr = {(r["scene_id"], r["keyframe_index"]): r for r in rows}
    dt = 0.5
    table = {}
    for name, keep in (("val", {"val"}), ("overlap_train_val", {"overlap_train_val"}), ("all", set(FB.SPLITS))):
        ks = sorted(k for k in zr if split_of[k] in keep)
        res = {"n": len(ks)}
        for side, dmp, rr in (("zero", dump, zr), ("pass3", p3d, p3r)):
            ade = np.array([rr[k]["ade"] for k in ks])
            best = np.array([min(G.ade3(q, human[k], dt) for q in dmp[k]["proposals"]) for k in ks])
            res[side] = {"ade_mean": float(ade.mean()), "ade_median": float(np.median(ade)),
                         "fde3_mean": float(np.mean([rr[k]["fde@3s"] for k in ks])),
                         "best_of_32_ade_mean": float(best.mean()), "best_of_32_ade_median": float(np.median(best)),
                         "regret_mean": float((ade - best).mean()), "regret_median": float(np.median(ade - best)),
                         "plan_speed_mean_mps": float(np.mean([G.plan_stats(dmp[k]["trajectory"])[0] for k in ks]))}
        dz = D.drift({k: dump[k] for k in ks}, {k: p3d[k] for k in ks})["summary"]
        dade = np.array([p3r[k]["ade"] - zr[k]["ade"] for k in ks])
        res["pass3_vs_zero"] = {**dz, "better_by_more_than_10cm": int((dade < -0.1).sum()),
                                "worse_by_more_than_10cm": int((dade > 0.1).sum())}
        table[name] = res
    out["table"] = table
    write_json(OUT / "check2_zero223.json", out)
    print(json.dumps(out, indent=1, default=str)[:5000])


if __name__ == "__main__":
    {"random": random_check, "zero223": zero223}[sys.argv[1]]()
