"""Build 2, step 4: the step-0 check (GPU, after the memory gate). On the 1,131 graded picking-set moments, fp32, batch 1, eval, no previous plan:
  p06 through its ORIGINAL path: pai_model.forward_dump on the strict-loaded agent (live images from the slim copy through the frozen encoder),
  vs the wrapper (model_a4.A4Model) at step 0 with the real history, tokens from approach 3's picking cache. Arms A and B.
Required: picks identical on all 1,131; rating logits (all 6 per plan, 32 plans) max abs diff exactly 0.0; pick_grade_mean 0.76849 (both).
Also recorded: cached tokens vs the live encoder's tokens on every moment (bit for bit; a difference is a FINDING, reported, not hidden),
the loader's ego / camera inputs vs the live features, the hide masks (A never, B always), and the grades vs 0.1.8b's stored p0.6 grades.
Link sanity: link 1 and link 2 weights set to N(0, 0.01^2) (biases stay zero, no_history stays zero) in both wrappers: A's logits must move,
B's must not (B reads only no_history = 0); B is also run with the histories shuffled across moments (must equal B unshuffled).
Output MH/out/step0_check.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# the grading pool spawns workers that re-import this module as __mp_main__: approach 3's grader patch must run at import time
# here (as s6_train.py does at its top), or s6_env.init_worker finds _ORIG unset in the child (TypeError: NoneType not callable)
import s6_env as _SE  # noqa: E402
import pai_model_b as _MB  # noqa: E402
_SE.patch_grader(_MB)

OUT = Path(__file__).resolve().parents[1] / "out"
P06_PICK_GRADE = 0.768489                                   # PREREG §4 (run.json checks[pass=0.6].safety.pick_grade_mean, 6 d.p.)
RUN_JSON = Path("{retrain_store}/018b/runs/018b_attempt2/run.json")
STORED_GRADES = Path("{retrain_store}/018b/runs/018b_attempt2/dumps/p00.6_grades.npy")


def main():
    import s6_env  # noqa: F401
    import pai_common as C
    C.setup_env()
    gate = C.memory_gate(True)
    import pai_train as T
    import pai_data as PD
    import pai_model as PM
    import pai_model_b as MB
    import a4_data as D
    import model_a4 as MA
    s6_env.patch_grader(MB)
    res = {"memory_gate": gate}
    tr, pk, _ = T.manifests()
    gi = np.where(pk.has_label2.to_numpy())[0]
    assert len(gi) == 1131
    agent, info = MA.load_agent()
    res["load"] = info
    A = MA.A4Model(agent, "A", T.SEED); B = MA.A4Model(agent, "B", T.SEED)
    res["load_from_p06"] = A.load_from_p06()
    A.eval_mode(); B.eval_mode()
    T._module_level_slim()
    live = PD.planner_dataset(agent, pk.assign(type="ordinary"), "val")
    cached = D.picking_set(pk)
    cap = {}
    agent._pad_model._backbone.register_forward_hook(lambda mod, inp, out: cap.__setitem__("feat", out[0]))
    rows = []; t0 = time.time(); t_wrap = 0.0
    mc, menus = [], []
    for n, i in enumerate(gi):
        f, t = live[int(i)]
        d = PM.forward_dump(agent, f)                                            # p06, original path
        live_tok = cap["feat"][0, :, 0, :].cpu()
        fc, tc = cached.item(int(i))
        assert tc["token"] == t["token"]
        r = {"token": t["token"], "tokens_equal_live": bool(torch.equal(fc["tokens"], live_tok)),
             "tokens_max_abs_diff": float((fc["tokens"] - live_tok).abs().max()),
             "ego_equal": bool(torch.equal(fc["ego_status"], f["ego_status"])),
             "camera_equal": bool(torch.equal(fc["lidar2img"], f["lidar2img"]) and torch.equal(fc["img_shape"], f["img_shape"])),
             "mc_equal": tc["metric_cache_path"] == t["metric_cache_path"]}
        ts = time.time()
        with torch.no_grad():
            for name, mdl in (("A", A), ("B", B)):
                o = mdl.forward_moment(fc["tokens"][None].cuda(), fc["ego_status"][None].cuda()[:, -1], fc["lidar2img"][None].cuda(),
                                       fc["img_shape"][None].cuda(), fc["ego_history"][None].cuda(), training=False, replay=False)
                r[f"{name}_pick"] = int(o["chosen"][0]); r[f"{name}_hide"] = bool(o["hide"][0])
                r[f"{name}_logit_maxdiff"] = float(np.abs(o["pred_logit"][0].cpu().numpy() - d["pred_logit"]).max())
                r[f"{name}_plans_equal"] = bool(np.array_equal(o["proposals"][0].cpu().numpy(), d["proposals"]))
        t_wrap += time.time() - ts
        r["p06_pick"] = d["chosen"]
        rows.append(r); mc.append(t["metric_cache_path"]); menus.append(d["proposals"])
        if (n + 1) % 200 == 0:
            print(f"{n + 1}/1131 {time.time() - t0:.0f} s", flush=True)
    res["forward_time_s"] = {"total": round(time.time() - t0, 1), "wrapper_two_arms": round(t_wrap, 1)}
    # grades (0.1.7's grader, the pool train_018b uses)
    tg = time.time()
    grader = MB.Grader(10)
    grades = np.stack(grader.grade_menus(mc, menus))
    res["grading_s"] = round(time.time() - tg, 1)
    fin = grades[..., -1]
    p06_pick = np.array([r["p06_pick"] for r in rows])
    out = {"moments": len(rows), "p06_pick_grade_mean": float(fin[np.arange(len(rows)), p06_pick].mean())}
    stored = np.load(STORED_GRADES) if STORED_GRADES.exists() else None
    out["grades_equal_stored_p06"] = None if stored is None else bool(stored.shape == grades.shape and np.array_equal(stored, grades))
    rj = json.loads(RUN_JSON.read_text())
    out["run_json_p06_pick_grade_mean"] = next(c["safety"]["pick_grade_mean"] for c in rj["checks"] if c["pass"] == 0.6)
    for name in ("A", "B"):
        pick = np.array([r[f"{name}_pick"] for r in rows])
        out[name] = {"picks_equal": int((pick == p06_pick).sum()), "plans_equal": int(sum(r[f"{name}_plans_equal"] for r in rows)),
                     "logit_max_abs_diff": max(r[f"{name}_logit_maxdiff"] for r in rows),
                     "pick_grade_mean": float(fin[np.arange(len(rows)), pick].mean()),
                     "hidden": int(sum(r[f"{name}_hide"] for r in rows))}
    out["cache_vs_live"] = {"tokens_equal": int(sum(r["tokens_equal_live"] for r in rows)), "tokens_max_abs_diff": max(r["tokens_max_abs_diff"] for r in rows),
                            "ego_equal": int(sum(r["ego_equal"] for r in rows)), "camera_equal": int(sum(r["camera_equal"] for r in rows)),
                            "metric_cache_path_equal": int(sum(r["mc_equal"] for r in rows))}
    res["step0"] = out
    print(json.dumps(out), flush=True)

    # link sanity on the first 100 graded moments
    g = torch.Generator(device="cpu").manual_seed(123)
    with torch.no_grad():
        for mdl in (A, B):
            mdl.link1.weight.copy_(torch.randn(mdl.link1.weight.shape, generator=g) * 0.01)
            mdl.link2.weight.copy_(torch.randn(mdl.link2.weight.shape, generator=g) * 0.01)
    sub = gi[:100]
    items = [cached.item(int(i)) for i in sub]
    hist = torch.stack([f["ego_history"] for f, _ in items])
    shuf = hist[torch.randperm(len(items), generator=g)]
    base = {r["token"]: r for r in rows}
    s = {"A_moved": 0, "A_logit_maxdiff": 0.0, "A_picks_changed": 0, "B_moved": 0, "B_logit_maxdiff": 0.0, "B_shuffled_history_maxdiff": 0.0,
         "A_shuffled_history_moved": 0, "moments": len(items)}
    with torch.no_grad():
        for k, (f, t) in enumerate(items):
            ref = PM.forward_dump(agent, live[int(sub[k])][0])["pred_logit"]
            args = (f["tokens"][None].cuda(), f["ego_status"][None].cuda()[:, -1], f["lidar2img"][None].cuda(), f["img_shape"][None].cuda())
            for name, mdl in (("A", A), ("B", B)):
                o = mdl.forward_moment(*args, hist[k][None].cuda(), training=False, replay=False)
                o2 = mdl.forward_moment(*args, shuf[k][None].cuda(), training=False, replay=False)
                dd = float(np.abs(o["pred_logit"][0].cpu().numpy() - ref).max())
                d2 = float((o["pred_logit"] - o2["pred_logit"]).abs().max())
                s[f"{name}_moved"] += int(dd > 0); s[f"{name}_logit_maxdiff"] = max(s[f"{name}_logit_maxdiff"], dd)
                if name == "A":
                    s["A_picks_changed"] += int(int(o["chosen"][0]) != base[t["token"]]["p06_pick"]); s["A_shuffled_history_moved"] += int(d2 > 0)
                else:
                    s["B_shuffled_history_maxdiff"] = max(s["B_shuffled_history_maxdiff"], d2)
    res["link_sanity"] = s
    print(json.dumps(s), flush=True)
    grader.close()
    res["pass"] = bool(all(out[a]["picks_equal"] == 1131 and out[a]["logit_max_abs_diff"] == 0.0 and round(out[a]["pick_grade_mean"], 5) == 0.76849
                           for a in ("A", "B")) and out["A"]["hidden"] == 0 and out["B"]["hidden"] == 1131
                      and s["A_moved"] > 0 and s["B_moved"] == 0 and s["B_shuffled_history_maxdiff"] == 0.0)
    OUT.mkdir(exist_ok=True)
    (OUT / "step0_check.json").write_text(json.dumps(res, indent=1, default=str))
    print("PASS" if res["pass"] else "FAIL", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
