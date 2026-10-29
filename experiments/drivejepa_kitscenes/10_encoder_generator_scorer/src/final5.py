"""Held-out evaluation of "Encoder, generator and scorer trained together" (GPU part; run once).

    setsid nohup ./run.sh src/final5.py > logs/final5.log 2>&1 &

Order:
  gate A  "Generator only, trained on KIT" rebuilt as in 09_heldout_comparison (release + pass-3 LoRA merged + phase-1
          rerun w0 p9.5 generator) on the 103 baseline picking moments: dump bit-identical to the saved
          p9.5 dump, 103-moment closest FDE (3 s) mean equal to 09_heldout_comparison's G2 record, and the KIT wrapper
          path bit-identical on the same 103.  If gate A fails, nothing held-out is read.
  gate B  the chosen checkpoint, loaded and merged, reproduces its own saved picking dump bit for bit
          on the 103 (proves the checkpoint is loaded as it was evaluated).
  223     one pass over val + overlap_train_val with 09_heldout_comparison.loop unchanged (adapter.build_sample,
          compute_trajectory, displacement_errors at 3 s, forward_dump).  Only read of these moments here.
Analysis (CPU, no model) is src/report.py.
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
import f_baseline as FB     # experiments/drivejepa_kitscenes s1
import f_final as FF        # experiments/drivejepa_kitscenes s2 (merge)
import final as FN          # 09_heldout_comparison (models, loop)
import g0_gpu as G0         # 05_generator_survey
import kitdata as DT        # 06_generator_only_fast_rate
import lora as L            # experiments/drivejepa_kitscenes s1
import phase1 as P1         # 06_generator_only_fast_rate
import splits as S          # experiments/drivejepa_kitscenes s1

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "out"
GJK = HERE.parent
DATA = Path("{runs}/drivejepa_kitscenes/generator_joint_kit/s5")
NAME = "Encoder, generator and scorer trained together"

log = P1.log


def wj(name: str, obj) -> None:
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    tmp.replace(OUT / name)


def bits(x) -> bytes:
    return np.asarray(x).tobytes()


def picking_103():
    meta = torch.load(DT.DATA / "scene_meta.pt", weights_only=False)
    mom = torch.load(DT.DATA / "moments.pt", weights_only=False)["selection"]
    g0 = pickle.load(open(GJK / "05_generator_survey" / "out" / "g_menu_per_moment.pkl", "rb"))["original"]
    cont = sorted(g0)
    assert len(cont) == 103
    by = {P1.key(m): m for m in mom}
    return meta, by, cont


def dump103(agent, meta, by, cont) -> dict:
    fc = K.FrameCache()
    out = {}
    for k in cont:
        d = E.forward_dump(agent, DT.assemble(by[k], meta[k[0]], fc))
        out[k] = {f: d[f] for f in ("proposals", "pdm_score", "chosen")}
    return out


def gate_a(meta, by, cont) -> dict:
    m3 = G0.build(pass3=True, merged=True)
    ck = torch.load(FN.P95, map_location="cpu", weights_only=False)
    assert ck["w"] == 0.0 and ck["progress_passes"] == 9.5
    FN.load_gen(m3, ck["generator"])
    m3.eval()
    saved = pickle.load(open(FN.P95_DUMP, "rb"))
    s4 = json.loads((GJK / "09_heldout_comparison" / "out" / "state.json").read_text())["G2"]
    human = {k: by[k]["future_poses"] for k in cont}
    dump = dump103(m3, meta, by, cont)
    bit = sum(all(bits(dump[k][f]) == bits(saved[k][f]) for f in ("proposals", "pdm_score", "chosen")) for k in cont)
    now = float(np.mean([P1.per_moment(dump, human, None)[k]["m1b"] for k in cont]))
    # the KIT wrapper path on the same 103 (09_heldout_comparison gate_g2, model 3 part)
    data = S.load_splits()
    n_future = m3._trajectory_sampling.num_poses

    def prepare(sid):
        ad = FN.ADP.KITScenesSceneAdapter(Path(data["scene_paths"][sid]))
        kfs = ad.valid_keyframes(num_future_frames=10, stride=90, first_frame=40)
        return sid, [(kf, ad.build_sample(kf, num_future_frames=n_future, command_mode="gt")) for kf in kfs]

    eq, seen = 0, 0
    with ThreadPoolExecutor(8) as pool:
        for sid, samples in pool.map(prepare, sorted({k[0] for k in cont})):
            for kf, smp in samples:
                d = FB.forward_dump(m3, smp.agent_input)
                eq += int(bits(d["proposals"]) == bits(saved[(sid, kf)]["proposals"]) and d["chosen"] == saved[(sid, kf)]["chosen"])
                seen += 1
    res = {"bit_identical_to_saved_p9.5_dump": f"{bit} / {len(cont)}",
           "closest_fde3_103_now": now, "closest_fde3_103_s4_record": s4["3"]["m1b_103_now"],
           "s4_record_pass": s4["3"]["pass"],
           "wrapper_path_bit_identical": f"{eq} / {seen}",
           "s4_wrapper_record": s4["wrapper_path_103"]["3"]["bit_identical_menus_and_picks"]}
    res["pass"] = (bit == len(cont) and now == s4["3"]["m1b_103_now"] and eq == seen == len(cont)
                   and res["s4_wrapper_record"] == f"{seen} / {seen}")
    del m3
    torch.cuda.empty_cache()
    return res


def build_ours(chosen: float):
    ck = torch.load(DATA / "ckpt" / f"p{chosen:04.1f}.pt", map_location="cpu", weights_only=False)
    assert ck["progress_passes"] == chosen
    a = G0.build(pass3=True, merged=False)
    FN.load_gen(a, ck["generator"])
    FN.load_lora(a.get_submodule(E.ENC_PATH), ck["lora"])
    m = a._pad_model
    cur = {n: p for n, p in m.named_parameters() if n.startswith("scorer.pred_score.")}
    assert set(cur) == set(ck["pred_score"]) and len(cur) == 10
    with torch.no_grad():
        for n, t in ck["pred_score"].items():
            cur[n].copy_(t.to(cur[n].device))
    agent = FF.merge(a)
    del a
    agent.eval()
    for p in agent.parameters():
        p.requires_grad_(False)
    assert not L.lora_modules(agent)
    return agent


def held_out(agent) -> dict:
    """09_heldout_comparison held_out, with this model only (FN.loop unchanged; its model key 1 is this run's model)."""
    root = FB.ROOT / "data"
    targets = [(sp, d.name) for sp in FB.SPLITS for d in sorted((root / sp).iterdir()) if d.is_dir()]
    for sp in FB.SPLITS:
        for d in (root / sp).iterdir():
            assert d.is_symlink()
    n_scenes = {sp: sum(t[0] == sp for t in targets) for sp in FB.SPLITS}
    assert n_scenes == {"val": 117, "overlap_train_val": 23}, n_scenes
    data = S.load_splits()
    assert not {t[1] for t in targets} & (set(data["train"]) | set(data["selection"]))
    assert C.sha256_file(C.BASELINE_PKL) == C.BASELINE_PKL_SHA256
    base = {(r["scene_id"], int(r["keyframe_index"])): r for r in pickle.load(open(C.BASELINE_PKL, "rb"))}
    res = FN.loop({1: agent}, [(sp, sid, root / sp / sid) for sp, sid in targets], base)
    assert {(m["scene_id"], m["keyframe_index"]) for m in res["moments"]} == set(base) and len(res["moments"]) == 223
    res = {"moments": res["moments"], "rows": {5: res["rows"][1]}, "dumps": {5: res["dumps"][1]},
           "wall_s": res["wall_s"], "names": {5: NAME}}
    with open(DATA / "heldout223.pkl", "wb") as fh:
        pickle.dump(res, fh)
    return {"wall_s": res["wall_s"], "scenes": n_scenes, "moments": len(res["moments"])}


def main() -> None:
    run = json.loads((OUT / "run.json").read_text())
    chosen = run["chosen"]
    assert chosen is not None
    state = {"preflight": P1.preflight(), "chosen_check": chosen}
    meta, by, cont = picking_103()
    state["gate_A_generator_only_unchanged"] = gate_a(meta, by, cont)
    log(f"gate A {json.dumps(state['gate_A_generator_only_unchanged'])}")
    wj("final_state.json", state)
    if not state["gate_A_generator_only_unchanged"]["pass"]:
        log("STOP: gate A failed; held-out moments not read")
        return
    ours = build_ours(chosen)
    saved = pickle.load(open(DATA / "dumps" / f"p{chosen:04.1f}.pkl", "rb"))
    d = dump103(ours, meta, by, cont)
    bit = sum(all(bits(d[k][f]) == bits(saved[k][f]) for f in ("proposals", "pdm_score", "chosen")) for k in cont)
    state["gate_B_checkpoint_reproduces_its_dump"] = {"bit_identical_103": f"{bit} / {len(cont)}", "pass": bit == len(cont)}
    log(f"gate B {state['gate_B_checkpoint_reproduces_its_dump']}")
    wj("final_state.json", state)
    if bit != len(cont):
        log("STOP: gate B failed; held-out moments not read")
        return
    state["heldout"] = held_out(ours)
    wj("final_state.json", state)
    log("done")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
