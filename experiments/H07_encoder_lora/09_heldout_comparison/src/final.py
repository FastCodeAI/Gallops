"""Final comparison of four Drive-JEPA models on the 223 held-out KITScenes moments (GPU part).

    setsid nohup ./run.sh src/final.py > logs/final.log 2>&1 &

The four models (all with the released scorer and projector; batch 1, eval mode, no GridMask):
  1 released    release checkpoint (sha256-checked), nothing changed
  2 encoder     release + pass-3 encoder LoRA (experiments/drivejepa_kitscenes s2 pass_03/state.pt), merged
  3 generator   model 2 + generator from the phase-1 rerun (s2 w0 p09.5.pt)
  4 joint       release + LoRA and generator from phase 2 (s3 phase2 p01.0.pt), LoRA merged as
                in 08_generator_and_encoder_together evaluation (G0.build unmerged -> load -> FF.merge)

Order:
  G2  models 3 and 4 on the 103 baseline picking moments, through the path they were checked
      with (kitdata.assemble -> evalsets.forward_dump); dumps must be bit-identical to the saved
      s2 / s3 dumps and the 103-moment 1b mean equal to the recorded value.  Also run through the
      KIT wrapper path used below (adapter.build_sample -> f_baseline.forward_dump) for the record.
      If G2 fails, nothing held-out is read.
  223 one pass over val + overlap_train_val with all four models: the 04_encoder_checks / s2 f_final
      wrapper loop unchanged (adapter.build_sample, compute_trajectory, displacement_errors at
      3 s, forward_dump).  This is the only read of these moments.
  G1  models 1 and 2 must equal the 04_encoder_checks zero-LoRA rows and the s2 f_final rows per moment.
      If G1 fails, the results are kept on disk but no analysis is written.
Analysis (CPU, no model) is src/final_report.py.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import cache as K           # experiments/drivejepa_kitscenes s2
import common as C          # experiments/drivejepa_kitscenes s1
import evalsets as E        # experiments/drivejepa_kitscenes s2
import f_baseline as FB     # experiments/drivejepa_kitscenes s1 (wrapper loop pieces)
import f_final as FF        # experiments/drivejepa_kitscenes s2 (merge)
import g0_gpu as G0         # 05_generator_survey (build)
import kitdata as DT        # 06_generator_only_fast_rate
import lora as L            # experiments/drivejepa_kitscenes s1
import phase1 as P1         # 06_generator_only_fast_rate
import splits as S          # experiments/drivejepa_kitscenes s1
from s2common import OUT as ENC_S2OUT

RUN = C.kit_runner_module()
ADP = C.kit_adapter_module()
HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "out"
DATA = Path("{runs}/drivejepa_kitscenes/generator_joint_kit/s4")
GJK = HERE.parent
ENC_S3OUT = HERE.parent.parent.parent / "experiments/drivejepa_kitscenes" / "experiments/drivejepa_kitscenes" / "04_encoder_checks" / "out"
P95 = Path("{runs}/drivejepa_kitscenes/generator_joint_kit/s2/w0/ckpt/p09.5.pt")
P95_DUMP = Path("{runs}/drivejepa_kitscenes/generator_joint_kit/s2/w0/dumps/p09.5.pkl")
P10 = Path("{runs}/drivejepa_kitscenes/generator_joint_kit/s3/phase2/ckpt/p01.0.pt")
P10_DUMP = Path("{runs}/drivejepa_kitscenes/generator_joint_kit/s3/phase2/dumps/p01.0.pkl")
NAMES = {1: "Released model", 2: "Encoder only, tuned on KIT", 3: "Generator only, trained on KIT",
         4: "Generator and encoder trained together"}

log = P1.log


def wj(name: str, obj) -> None:
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    tmp.replace(OUT / name)


def bits(x) -> bytes:
    return np.asarray(x).tobytes()


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def load_gen(agent, sd: dict) -> None:
    """The 41 trained generator tensors (s3 Model.load_gen); the fixed grids stay as released."""
    m = agent._pad_model
    cur = {n: p for n, p in m.named_parameters()
           if n.startswith(P1.GEN_PREFIXES) and not n.endswith(P1.FIXED_SUFFIXES)}
    assert set(sd) == set(cur) and len(cur) == 41
    with torch.no_grad():
        for n, t in sd.items():
            cur[n].copy_(t.to(cur[n].device))


def load_lora(enc, sd: dict) -> None:
    params = dict(enc.named_parameters())
    assert set(sd) == {n for n in params if ".lora_" in n} and len(sd) == 192
    with torch.no_grad():
        for n, t in sd.items():
            params[n].copy_(t.to(params[n].device, params[n].dtype))


def build_models() -> tuple[dict, dict]:
    info = {}
    m1 = G0.build(pass3=False)                     # sha256 of the release asserted inside
    m2 = G0.build(pass3=True, merged=True)
    m3 = G0.build(pass3=True, merged=True)
    ck3 = torch.load(P95, map_location="cpu", weights_only=False)
    assert ck3["w"] == 0.0 and ck3["progress_passes"] == 9.5
    load_gen(m3, ck3["generator"])
    a4 = G0.build(pass3=True, merged=False)
    ck4 = torch.load(P10, map_location="cpu", weights_only=False)
    assert ck4["run"] == "phase2" and ck4["progress_passes"] == 1.0
    info["model4_checkpoint_keys"] = sorted(ck4)
    load_gen(a4, ck4["generator"])
    load_lora(a4.get_submodule(E.ENC_PATH), ck4["lora"])
    m4 = FF.merge(a4)
    del a4
    models = {1: m1, 2: m2, 3: m3, 4: m4}
    for m in models.values():
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        assert not L.lora_modules(m)
    # sanity: what differs between the models
    sd = {i: m._pad_model.state_dict() for i, m in models.items()}
    enc = lambda k: k.startswith("_backbone.img_backbone.")
    gen = lambda k: k.startswith(P1.GEN_PREFIXES)
    for i, j in ((1, 2), (2, 3), (3, 4), (1, 4)):
        assert set(sd[i]) == set(sd[j])
        diff = [k for k in sd[i] if not torch.equal(sd[i][k], sd[j][k])]
        info[f"tensors_differing_{i}_vs_{j}"] = {"encoder": sum(map(enc, diff)), "generator": sum(map(gen, diff)),
                                                "other": sum(not enc(k) and not gen(k) for k in diff)}
    return models, info


# --------------------------------------------------------------------------- #
# G2: models 3 and 4 on the 103 picking moments
# --------------------------------------------------------------------------- #


def gate_g2(models: dict) -> dict:
    meta = torch.load(DT.DATA / "scene_meta.pt", weights_only=False)
    mom = torch.load(DT.DATA / "moments.pt", weights_only=False)["selection"]
    g0 = pickle.load(open(GJK / "05_generator_survey" / "out" / "g_menu_per_moment.pkl", "rb"))["original"]
    cont = sorted(g0)
    assert len(cont) == 103
    by = {P1.key(m): m for m in mom}
    human = {k: by[k]["future_poses"] for k in cont}
    saved = {3: pickle.load(open(P95_DUMP, "rb")), 4: pickle.load(open(P10_DUMP, "rb"))}
    recorded = {
        3: [c for c in json.loads((GJK / "07_generator_only_gentle_rate" / "out" / "run_w0.json").read_text())["checks"]
            if c["pass"] == 9.5][0]["summary"]["CONT_m1b_mean"],
        4: [c for c in json.loads((GJK / "08_generator_and_encoder_together" / "out" / "run_phase2.json").read_text())["checks"]
            if c["pass"] == 1.0][0]["summary"]["CONT_m1b_mean"]}
    fc = K.FrameCache()
    out = {}
    for i in (3, 4):
        dump = {}
        for k in cont:
            d = E.forward_dump(models[i], DT.assemble(by[k], meta[k[0]], fc))
            dump[k] = {f: d[f] for f in ("proposals", "pdm_score", "chosen")}
        bit = sum(all(bits(dump[k][f]) == bits(saved[i][k][f]) for f in ("proposals", "pdm_score", "chosen"))
                  for k in cont)
        now = float(np.mean([P1.per_moment(dump, human, None)[k]["m1b"] for k in cont]))
        out[i] = {"bit_identical_to_saved_dump": f"{bit} / {len(cont)}", "m1b_103_now": now,
                  "m1b_103_recorded": recorded[i], "pass": bit == len(cont) and now == recorded[i]}

    # the KIT wrapper path on the same 103 (record only: shows the 223 path sees the same model)
    data = S.load_splits()
    n_future = models[1]._trajectory_sampling.num_poses

    def prepare(sid):
        ad = ADP.KITScenesSceneAdapter(Path(data["scene_paths"][sid]))
        kfs = ad.valid_keyframes(num_future_frames=10, stride=90, first_frame=40)
        return sid, [(kf, ad.build_sample(kf, num_future_frames=n_future, command_mode="gt")) for kf in kfs]

    eq = {3: 0, 4: 0}
    maxdiff = {3: 0.0, 4: 0.0}
    seen = 0
    with ThreadPoolExecutor(8) as pool:
        for sid, samples in pool.map(prepare, sorted({k[0] for k in cont})):
            for kf, smp in samples:
                k = (sid, kf)
                assert k in human and np.array_equal(human[k], smp.future_poses)
                seen += 1
                for i in (3, 4):
                    d = FB.forward_dump(models[i], smp.agent_input)
                    eq[i] += int(bits(d["proposals"]) == bits(saved[i][k]["proposals"]) and d["chosen"] == saved[i][k]["chosen"])
                    maxdiff[i] = max(maxdiff[i], float(np.abs(d["proposals"].astype(np.float64) - saved[i][k]["proposals"]).max()))
    assert seen == len(cont)
    out["wrapper_path_103"] = {i: {"bit_identical_menus_and_picks": f"{eq[i]} / {seen}", "max_abs_diff_m": maxdiff[i]}
                               for i in (3, 4)}
    out["pass"] = out[3]["pass"] and out[4]["pass"]
    return out


# --------------------------------------------------------------------------- #
# The 223 held-out moments (read once)
# --------------------------------------------------------------------------- #


def held_out(models: dict) -> dict:
    root = FB.ROOT / "data"
    targets = [(sp, d.name) for sp in FB.SPLITS for d in sorted((root / sp).iterdir()) if d.is_dir()]
    for sp in FB.SPLITS:      # read-only stand-in for FB.build_root's check (which may create links)
        for d in (root / sp).iterdir():
            assert d.is_symlink()
    n_scenes = {sp: sum(t[0] == sp for t in targets) for sp in FB.SPLITS}
    assert n_scenes == {"val": 117, "overlap_train_val": 23}, n_scenes
    data = S.load_splits()
    assert not {t[1] for t in targets} & (set(data["train"]) | set(data["selection"]))
    assert C.sha256_file(C.BASELINE_PKL) == C.BASELINE_PKL_SHA256
    base = {(r["scene_id"], int(r["keyframe_index"])): r for r in pickle.load(open(C.BASELINE_PKL, "rb"))}
    res = loop(models, [(sp, sid, root / sp / sid) for sp, sid in targets], base)
    assert {(m["scene_id"], m["keyframe_index"]) for m in res["moments"]} == set(base) and len(res["moments"]) == 223
    with open(DATA / "heldout223.pkl", "wb") as fh:
        pickle.dump({**res, "names": NAMES}, fh)
    return {"wall_s": res["wall_s"], "scenes": n_scenes, "moments": len(res["moments"])}


def loop(models: dict, targets: list, base: dict | None) -> dict:
    """The 04_encoder_checks zero223 / s2 f_final loop, unchanged except that it runs every model on each sample."""
    n_future = models[1]._trajectory_sampling.num_poses

    def prepare(t):
        split, sid, path = t
        adapter = ADP.KITScenesSceneAdapter(path)                                                 # RUN:232
        kfs = adapter.valid_keyframes(num_future_frames=10, stride=90, first_frame=40)            # RUN:236-238
        return split, sid, [(kf, adapter.build_sample(kf, num_future_frames=n_future, command_mode="gt")) for kf in kfs]

    rows = {i: [] for i in models}
    dumps = {i: {} for i in models}
    moments = []
    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        for split, sid, samples in pool.map(prepare, targets):
            for kf, sample in samples:
                assert abs(sample.future_interval_s - 0.5) < 1e-9
                if base is not None:
                    assert np.array_equal(base[(sid, kf)]["human_poses"], sample.future_poses)
                ego = DT.ego_feature(sample.agent_input)
                moments.append({"scene_id": sid, "split": split, "keyframe_index": kf,
                                "human_poses": sample.future_poses,
                                "speed_mps": float(np.linalg.norm(ego[-1, 3:5].numpy())),
                                "heading_3s_rad": float(np.asarray(sample.future_poses)[5, 2])})
                for i, agent in models.items():
                    predicted = agent.compute_trajectory(sample.agent_input)                       # RUN:256
                    m = RUN.displacement_errors(predicted.poses, sample.future_poses, sample.future_interval_s, 3.0)
                    d = FB.forward_dump(agent, sample.agent_input)
                    assert L.bit_equal(torch.from_numpy(d["trajectory"]), torch.from_numpy(predicted.poses.astype(np.float32)))
                    dumps[i][(sid, kf)] = d
                    rows[i].append({"scene_id": sid, "split": split, "keyframe_index": kf,
                                    "predicted_poses": predicted.poses, "human_poses": sample.future_poses, **m})
            log(f"{len(moments):3d} moments {time.time() - t0:5.0f}s {sid[:8]} ({split})")
    return {"moments": moments, "rows": rows, "dumps": dumps, "wall_s": round(time.time() - t0, 1)}


def gate_g1() -> dict:
    res = pickle.load(open(DATA / "heldout223.pkl", "rb"))
    ref = {1: (pickle.load(open(ENC_S3OUT / "zero223_rows.pkl", "rb")), pickle.load(open(ENC_S3OUT / "zero223_dump.pkl", "rb"))),
           2: (pickle.load(open(ENC_S2OUT / "f_final_rows.pkl", "rb")), pickle.load(open(ENC_S2OUT / "f_final_dump.pkl", "rb")))}
    rec = json.loads((ENC_S3OUT / "check2_zero223.json").read_text())["table"]["all"]
    rec_ade = {1: rec["zero"]["ade_mean"], 2: rec["pass3"]["ade_mean"]}
    out = {}
    for i in (1, 2):
        rr = {(r["scene_id"], r["keyframe_index"]): r for r in ref[i][0]}
        now = {(r["scene_id"], r["keyframe_index"]): r for r in res["rows"][i]}
        assert set(rr) == set(now)
        keys = ("ade", "fde@1s", "fde@2s", "fde@3s")
        per_eq = sum(all(now[k][x] == rr[k][x] for x in keys) and bits(now[k]["predicted_poses"]) == bits(rr[k]["predicted_poses"])
                     for k in rr)
        dump_eq = sum(all(bits(res["dumps"][i][k][f]) == bits(ref[i][1][k][f]) for f in ("proposals", "pdm_score", "trajectory"))
                      and res["dumps"][i][k]["chosen"] == ref[i][1][k]["chosen"] for k in rr)
        # averaged over sorted moment keys, as 04_encoder_checks close.py builds its table (summation order matters
        # in the last digit: the loop-order mean of model 1 is 2.3718799477361716 vs 2.371879947736172)
        ade = float(np.array([now[k]["ade"] for k in sorted(now)]).mean())
        out[i] = {"ade_mean_now": ade, "ade_mean_loop_order": float(np.mean([r["ade"] for r in res["rows"][i]])), "ade_mean_recorded": rec_ade[i], "per_moment_rows_equal": f"{per_eq} / {len(rr)}",
                  "dumps_bit_identical": f"{dump_eq} / {len(rr)}",
                  "pass": ade == rec_ade[i] and per_eq == len(rr) and dump_eq == len(rr)}
    out["pass"] = out[1]["pass"] and out[2]["pass"]
    return out


def smoke() -> None:
    """Wiring test on 2 picking scenes; reads nothing held-out.  Output to the s4 data folder."""
    models, info = build_models()
    log(json.dumps(info))
    data = S.load_splits()
    sids = sorted(data["selection"])[:2]
    res = loop(models, [("selection", s, Path(data["scene_paths"][s])) for s in sids], None)
    with open(DATA / "smoke_picking2.pkl", "wb") as fh:
        pickle.dump(res, fh)
    for i in models:
        log(f"model {i}: {len(res['rows'][i])} rows, ADE {np.mean([r['ade'] for r in res['rows'][i]]):.4f}, "
            f"keys {sorted(res['rows'][i][0])}")
    log(f"moments[0] {({k: v for k, v in res['moments'][0].items() if k != 'human_poses'})}")
    log("smoke done")


def regate_g1() -> None:
    """G1 again from the saved results (no model, no held-out read).  Used once: the first G1 averaged
    model 1's rows in loop order and missed the recorded mean in the last digit."""
    state = json.loads((OUT / "state.json").read_text())
    state["G1_first_attempt_loop_order_mean"] = state.pop("G1")
    state["G1"] = gate_g1()
    log(f"G1 {json.dumps(state['G1'])}")
    wj("state.json", state)
    log("done" if state["G1"]["pass"] else "STOP: G1 failed; no analysis")


def main() -> None:
    state = {"preflight": P1.preflight()}
    t0 = time.time()
    models, info = build_models()
    state["models"] = {**info, "names": NAMES}
    log(json.dumps(info))
    state["G2"] = gate_g2(models)
    log(f"G2 {json.dumps(state['G2'])}")
    wj("state.json", state)
    if not state["G2"]["pass"]:
        log("STOP: G2 failed; held-out moments not read")
        return
    state["heldout"] = held_out(models)
    del models
    torch.cuda.empty_cache()
    state["G1"] = gate_g1()
    log(f"G1 {json.dumps(state['G1'])}")
    state["wall_s"] = round(time.time() - t0, 1)
    wj("state.json", state)
    if not state["G1"]["pass"]:
        log("STOP: G1 failed; no analysis")
        return
    log("done")


if __name__ == "__main__":
    {"smoke": smoke, "all": main, "g1": regate_g1}[sys.argv[1] if len(sys.argv) > 1 else "all"]()
