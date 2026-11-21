"""Build step 3 tests T1-T5 (Sid's brief). Usage: run_model.sh <abs>/s6_tests.py {params|t1|t2t3|t4}; writes out/s6_<test>*.json.
T1 step-0 identity: carry and wiped, untrained, vs 0.1.8b's cached-token forward (BUILD2 exactness method: pai_model.forward_dump with
   the encoder swapped for the cached tokens, s5_cache_check.py) on all 2,591 moments of the 84 picking clips replayed in order; plans,
   logits, picks bit-identical; plus the check's numbers at step 0 (= 0.1.8b's baseline) and the motion-source consistency check.
T2 routing: carry mode, 5 warm-start steps at the real learning rates, then one backward per lesson alone on a fresh batch; table lesson x
   parameter group with "gradient non-zero".
T3 wiped: M all zeros at every read over 5 steps; after them the update block gets no gradient; carry: M non-zero after the first moment.
T4 smoke: 50 steps per mode, real learning rates, seed 0; losses every 10 steps, gate mean per slot, M norm, grading wait, step time, finite.
T5 timing: from T4's steps 10-50: s/step and its split (forward incl. the replays, grading wait, backward); projected pass / 3-pass wall
   with and without event over-sampling; the grading pool on the 2 pinned cores (taskset -c 0-1 nice 19, run_model.sh)."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
import s6_env as SE                       # FIRST: re-points s35_common at the local navsim tree (module docstring)
import pai_common as C
C.setup_env()
import pai_model as PM, pai_model_b as MB
SE.patch_grader(MB)
import s6_model as SM, s6_data as SD, s6_train as ST
import pai_train as T

OUT = ST.OUT; LOG_EVERY = 10; RUNS_T = SD.A3 / "runs" / "build3_tests"; SEED = 0
VERSION = int(os.environ.get("MEMORY_VERSION", "1")); SFX = "" if VERSION == 1 else f"_v{VERSION}"     # build 4: MEMORY_VERSION=2 re-runs T1-T4 on the version-2 model


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gpu_line():
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def head() -> dict:
    return {"started": time.strftime("%F %T"), "version": VERSION, "env_patch": SE.PATCHED, "gpu_before": gpu_line(), "loadavg": open("/proc/loadavg").read().split()[:3], "avail_ram_gb": round(C.avail_ram_gb(), 1),
            "affinity": sorted(os.sched_getaffinity(0)), "nice": os.nice(0), "seed": SEED}


STATE0 = {}


def build(agent, mode):
    """fresh memory blocks (seed 0) around the agent with its 0.1.8b generator / scorer weights restored (tests train them)."""
    torch.manual_seed(SEED)
    m = SM.MemoryModel(agent, mode, VERSION).cuda()
    if STATE0:
        m.load_trainable({**STATE0, "memory": m.trainable_state()["memory"]})
    else:
        STATE0.update({k: v for k, v in m.trainable_state().items() if k != "memory"})
    return m


GRADER = {}


def grader_once():
    if "g" not in GRADER:
        GRADER["g"] = MB._GRADER = MB.Grader(ST.POOL)
    return GRADER["g"]


def grad_table(model) -> dict:
    return {g: {"nonzero_tensors": int(sum(1 for p in ps if p.grad is not None and bool((p.grad != 0).any()))), "tensors": len(ps),
                "any": bool(any(p.grad is not None and bool((p.grad != 0).any()) for p in ps))} for g, ps in model.groups.items()}


def zero_grads(model):
    for p in model.parameters():
        p.grad = None


# ----------------------------------------------------------------------------------------------------------------------- #
def t_params(agent=None, info=None):
    if agent is None:
        agent, info = SM.load_agent()
    m = build(agent, "carry")
    R = {**head(), "agent": info, "new_params": m.new_param_count(), "target": 1_500_000}
    R["new_params"]["vs_target_pct"] = round(100 * (R["new_params"]["total_new"] / R["target"] - 1), 1)
    if VERSION == 2:
        M0 = m.start.M0.detach().cpu(); x = torch.nn.functional.normalize(M0, dim=-1); cmat = x @ x.T
        R["start_state"] = {"row_norms_min_max": [float(M0.norm(dim=-1).min()), float(M0.norm(dim=-1).max())], "max_abs_pairwise_cosine": float((cmat - torch.eye(16)).abs().max()),
                            "gate_bias": m.update.gate_bias.detach().cpu().tolist(), "gate_start_prob": torch.sigmoid(m.update.gate_bias.detach().cpu()).tolist(),
                            "gate_weight_std": float(m.update.gate.weight.std()), "gate_weight_abs_max": float(m.update.gate.weight.abs().max())}
    C.wj(OUT / f"s6_params{SFX}.json", R); log(json.dumps(R["new_params"]), json.dumps(R.get("start_state")))


def reference_dump(agent, store: SD.ClipStore) -> dict:
    """0.1.8b's cached-token forward: s5_cache_check.py's CachedEncoder swap + pai_model.forward_dump (batch 1, fp32), every moment of every clip."""
    m = agent._pad_model; bk = m._backbone

    class CachedEncoder(torch.nn.Module):
        def __init__(self): super().__init__(); self.feat = None; self.spatial = torch.tensor([[16, 32]], device="cuda"); self.lsi = torch.tensor([0], device="cuda")
        def forward(self, img, len_queue=None, **kwargs): return self.feat, self.spatial, self.lsi, kwargs
    swap = CachedEncoder().cuda().eval(); m._backbone = swap; dump = {}
    try:
        for ci, c in enumerate(store.clips):
            rec = store.load(c); prev = None
            for i, tok in enumerate(rec["sample_id"]):
                swap.feat = rec["tokens"][i].cuda()[None, :, None, :]
                f = {"lidar2img": rec["lidar2img"], "img_shape": rec["img_shape"], "ego_status": rec["ego"][i:i + 1],
                     "camera_feature_1": torch.zeros(3, 256, 512), "camera_feature_2": torch.zeros(3, 256, 512)}
                d = PM.forward_dump(agent, f)
                consecutive = i > 0 and abs(rec["anchor_t"][i] - rec["anchor_t"][i - 1] - SD.STEP_S) < 1e-6
                dump[tok] = {"proposals": d["proposals"], "pred_logit": d["pred_logit"], "chosen": d["chosen"], "human": rec["traj"][i].numpy(),
                             "mc": rec["metric_cache_path"][i], "lab": bool(rec["has_label2"][i]), "clip": c, "row": i, "prev": prev if consecutive else None,
                             "in_picking": bool(rec["in_picking"][i])}
                prev = tok
            if (ci + 1) % 20 == 0:
                log(f"reference {ci + 1}/{len(store.clips)} clips")
    finally:
        m._backbone = bk
    return dump


def bit_compare(a: dict, b: dict) -> dict:
    ks = sorted(a); assert set(ks) == set(b)
    out = {"moments": len(ks)}
    for f in ("proposals", "pred_logit", "chosen"):
        out[f"{f}_bit_identical"] = all(np.asarray(a[k][f], np.float32).tobytes() == np.asarray(b[k][f], np.float32).tobytes() for k in ks)
    out["picks_equal"] = int(sum(int(a[k]["chosen"]) == int(b[k]["chosen"]) for k in ks))
    out["max_abs_plan_diff_m"] = float(max(np.abs(np.asarray(a[k]["proposals"], np.float64) - np.asarray(b[k]["proposals"], np.float64)).max() for k in ks))
    out["max_abs_logit_diff"] = float(max(np.abs(np.asarray(a[k]["pred_logit"], np.float64) - np.asarray(b[k]["pred_logit"], np.float64)).max() for k in ks))
    out["pass"] = bool(out["proposals_bit_identical"] and out["pred_logit_bit_identical"] and out["chosen_bit_identical"])
    return out


def t1(agent=None, info=None):
    R = head()
    if agent is None:
        agent, info = SM.load_agent()
    R["agent"] = info
    store = SD.ClipStore("picking"); pk = ST.picking_frame()
    assert set(pk.sample_id) <= set(store.idx.sample_id) and int(store.idx.in_picking.sum()) == 2000 == len(pk)
    t0 = time.time(); ref = reference_dump(agent, store); R["reference"] = {"moments": len(ref), "wall_s": round(time.time() - t0, 1)}
    R["motion_source_check"] = ST.motion_source_check(ref)
    dumps = {}
    for mode in SM.MODES:
        model = build(agent, mode); t0 = time.time()
        d = ST.replay_slice(model, store, progress=log); dumps[mode] = d
        R[mode] = {"vs_018b_cached_forward": bit_compare(ref, d), "replay_wall_s": round(time.time() - t0, 1),
                   "M_read_max_abs_over_all_moments": float(max(x["M_read_max_abs"] for x in d.values()))}
        log(mode, json.dumps(R[mode]))
        del model; torch.cuda.empty_cache()
    R["pass"] = bool(R["carry"]["vs_018b_cached_forward"]["pass"] and R["wiped"]["vs_018b_cached_forward"]["pass"])
    if VERSION == 2:                            # bit-identical dumps have the same menus: 0.1.8b's step-0 measures (s6_t1.json, pilot step 0) apply
        R["M0_read_at_every_moment"] = {m: R[m]["M_read_max_abs_over_all_moments"] for m in SM.MODES}
        R["ended"] = time.strftime("%F %T"); C.wj(OUT / f"s6_t1{SFX}.json", R); log("T1", json.dumps({k: R[k] for k in ("pass", "M0_read_at_every_moment")})); return
    grader = grader_once()
    t0 = time.time(); t_rep = time.time(); m_ref, grades = ST.measures(ref, pk, grader); R["step0_018b"] = m_ref
    m_ref["time_s"] = {"replay": R["carry"]["replay_wall_s"], "measures": round(time.time() - t_rep, 1)}
    for mode in SM.MODES:                       # bit-identical dumps have the same menus: the reference grades apply; else re-grade
        if R[mode]["vs_018b_cached_forward"]["pass"]:
            R[f"step0_{mode}"] = {**m_ref, "note": "dump bit-identical to the reference: its measures and menu grades apply", "vs_018b": {"closest_of_32_ade3_delta_m": 0.0, "ade_within_tolerance": True, "follows_previous_rate_018b": m_ref["follows_previous"]["rate"]}}
        else:
            R[f"step0_{mode}"], _ = ST.measures(dumps[mode], pk, grader, m_ref)
    R["measures_wall_s"] = round(time.time() - t0, 1)
    R["step0_identical_measures"] = bool(R["step0_carry"]["pick_grade_mean_map"] == m_ref["pick_grade_mean_map"] == R["step0_wiped"]["pick_grade_mean_map"])
    R["pass"] = bool(R["carry"]["vs_018b_cached_forward"]["pass"] and R["wiped"]["vs_018b_cached_forward"]["pass"])
    RUNS_T.mkdir(parents=True, exist_ok=True)
    T.save_dump(RUNS_T / "step0_018b_picking_dump.npz", {k: v for k, v in ref.items()}); np.save(RUNS_T / "step0_018b_grades.npy", grades)
    R["ended"] = time.strftime("%F %T"); C.wj(OUT / "s6_t1.json", R)
    log("T1", json.dumps({k: R[k] for k in ("pass", "step0_identical_measures")}), json.dumps({k: R["step0_018b"][k] for k in ("pick_grade_mean_map", "hard_picks_above10", "go_rate_pull_away", "closest_of_32_ade3_all")}), R["step0_018b"]["follows_previous"])


# ----------------------------------------------------------------------------------------------------------------------- #
def warm_steps(model, store, grader, index_reader, opts, batches, n, tag) -> list:
    recs = []; pf = SD.Prefetcher(store, batches[:n])
    for step, items in enumerate(pf, 1):
        model.train(); q = ST.window_step(model, SD.window_tensors(items), grader, index_reader, opts); q["data_wait_s"] = round(pf.last_wait_s, 3); recs.append(q)
        log(f"{tag} step {step}/{n} data {q['data_wait_s']:.1f}s loss {q['loss']:.4f} gen {q['generator']:.4f} grade {q['grade']:.4f} human {q['human']:.4f} col {q['collision']:.4f} "
            f"drv {q['drivable']:.4f} map {q['map_rows']} wait {q['grade_wait_s']:.2f}s fwd {q['forward_s']:.2f}s bwd {q['backward_s']:.2f}s step {q['step_s']:.2f}s "
            f"Mnorm {q['M_norm_last']:.3f} gate {np.mean(q['gate_mean_per_slot']):.3f} Mread {q['M_read_max_abs']:.2e}")
    pf.close()
    return recs


def map_batch_index(store, batches, start):
    """first batch at index >= start whose two windows both hold map moments (has_label2), so every lesson has a graph."""
    for j in range(start, len(batches)):
        if all(bool(store.by_clip[c].has_label2.iloc[s:s + SD.WIN].any()) for c, s in batches[j]):
            return j
    raise RuntimeError("no map batch")


def t2map(agent=None, info=None):
    """T2 again on a batch WITH map moments (the chain's fresh batch had none: grade / collision / drivable were constant zeros there)."""
    R = head()
    if agent is None:
        agent, info = SM.load_agent()
    R["agent"] = info
    store = SD.ClipStore("training"); grader = grader_once(); index_reader = T.parallel_index_reader(agent)
    plan = SD.plan_pass(store, SEED, 1, True, SD.memory_moment_set()); batches = plan["batches"]
    model = build(agent, "carry"); opts = model.optimisers(); np.random.seed(SEED + 1)
    R["t2_warm"] = warm_steps(model, store, grader, index_reader, opts, batches, 5, "T2map warm")
    j = map_batch_index(store, batches, 5); items = next(iter(SD.Prefetcher(store, batches[j:j + 1]))); model.train()
    q = ST.window_step(model, SD.window_tensors(items), grader, index_reader, None, lessons_alone=True)
    R["t2_batch"] = {"index": j, "map_rows": q["map_rows"], "clips": [r["clip"] for r, _ in items]}
    alone = {}
    for name, loss in q["_losses"].items():
        zero_grads(model)
        if not torch.is_tensor(loss) or not loss.requires_grad:
            alone[name] = {"no_graph": True}; continue
        loss.backward(retain_graph=True); alone[name] = grad_table(model)
    zero_grads(model)
    expected = {"generator": {"generator", "generator_lookup", "update_block"}, "grade": {"rating_head", "scorer_summary", "update_block"},
                "human": {"rating_head", "scorer_summary", "update_block"}, "collision": {"collision_head"}, "drivable": {"drivable_head"}}
    if VERSION == 2:                            # the learned start state is read by the generator and (through the summary) by the rating head
        for k in ("generator", "grade", "human"):
            expected[k] = expected[k] | {"start_state"}
    table = {}; mism = []
    for name, tab in alone.items():
        if tab.get("no_graph"):
            table[name] = "no graph"; mism.append(f"{name}: no graph"); continue
        table[name] = {g: ("yes" if tab[g]["any"] else "no") for g in model.groups}
        got = {g for g in model.groups if tab[g]["any"]}
        if got != expected[name]:
            mism.append(f"{name}: got {sorted(got)} expected {sorted(expected[name])}")
    R["t2_routing"] = {"table": table, "expected": {k: sorted(v) for k, v in expected.items()}, "mismatches": mism, "pass": not mism, "detail": alone}
    log("T2map", json.dumps(R["t2_routing"]["table"]), "mismatches", mism)
    R["ended"] = time.strftime("%F %T"); C.wj(OUT / f"s6_t2_map{SFX}.json", R)


def t2t3(agent=None, info=None):
    R = head()
    if agent is None:
        agent, info = SM.load_agent()
    R["agent"] = info
    store = SD.ClipStore("training"); grader = grader_once(); index_reader = T.parallel_index_reader(agent)
    plan = SD.plan_pass(store, SEED, 1, True, SD.memory_moment_set()); batches = plan["batches"]
    R["pass_plan"] = {k: v for k, v in plan.items() if k != "batches"}
    # ---- T2: carry, 5 warm-start steps, then each lesson alone --------------------------------------------------------------------
    model = build(agent, "carry"); opts = model.optimisers(); np.random.seed(SEED + 1)
    R["t2_warm"] = warm_steps(model, store, grader, index_reader, opts, batches, 5, "T2 warm")
    R["t3_carry_M_after_first_moment_norm"] = [r["M_after_first_moment_norm"] for r in R["t2_warm"]]
    items = next(iter(SD.Prefetcher(store, batches[5:6]))); model.train()
    q = ST.window_step(model, SD.window_tensors(items), grader, index_reader, None, lessons_alone=True)
    R["t2_batch"] = {"map_rows": q["map_rows"], "clips": [r["clip"] for r, _ in items]}
    alone = {}
    for name, loss in q["_losses"].items():
        zero_grads(model)
        if not torch.is_tensor(loss) or not loss.requires_grad:
            alone[name] = {"no_graph": True}; continue
        loss.backward(retain_graph=True); alone[name] = grad_table(model)
    zero_grads(model)
    expected = {"generator": {"generator", "generator_lookup", "update_block"}, "grade": {"rating_head", "scorer_summary", "update_block"},
                "human": {"rating_head", "scorer_summary", "update_block"}, "collision": {"collision_head"}, "drivable": {"drivable_head"}}
    if VERSION == 2:                            # the learned start state is read by the generator and (through the summary) by the rating head
        for k in ("generator", "grade", "human"):
            expected[k] = expected[k] | {"start_state"}
    table = {}; mism = []
    for name, tab in alone.items():
        if tab.get("no_graph"):
            table[name] = "no graph (lesson is a constant 0 on this batch)"; mism.append(f"{name}: no map rows in the batch"); continue
        table[name] = {g: ("yes" if tab[g]["any"] else "no") for g in model.groups}
        got = {g for g in model.groups if tab[g]["any"]}
        if got != expected[name]:
            mism.append(f"{name}: got {sorted(got)} expected {sorted(expected[name])}")
    R["t2_routing"] = {"table": table, "expected": {k: sorted(v) for k, v in expected.items()}, "mismatches": mism, "pass": not mism, "detail": alone}
    log("T2", json.dumps(R["t2_routing"]["table"]), "mismatches", mism)
    del model, opts, q; torch.cuda.empty_cache()
    # ---- T3: wiped ----------------------------------------------------------------------------------------------------------------
    model = build(agent, "wiped"); opts = model.optimisers(); np.random.seed(SEED + 1)
    R["t3_warm"] = warm_steps(model, store, grader, index_reader, opts, batches, 5, "T3 warm")
    items = next(iter(SD.Prefetcher(store, batches[5:6]))); model.train()
    q = ST.window_step(model, SD.window_tensors(items), grader, index_reader, None, lessons_alone=True)
    zero_grads(model); sum(v for v in q["_losses"].values() if torch.is_tensor(v)).backward()
    upd = [p.grad for p in model.groups["update_block"]]
    dev = [r["M_read_max_dev_from_start"] for r in R["t3_warm"]] + [q["M_read_max_dev_from_start"]]     # version 1: from zeros; version 2: from M0
    R["t3_carry_M_after_first_moment_dev_from_start_norm"] = [r["M_after_first_moment_dev_from_start_norm"] for r in R["t2_warm"]]
    R["t3_wiped"] = {"M_read_max_abs_per_step": [r["M_read_max_abs"] for r in R["t3_warm"]] + [q["M_read_max_abs"]],
                     "M_read_max_dev_from_start_per_step": dev, "M_equals_start_at_every_read": bool(max(dev) == 0.0),
                     "M_zero_at_every_read": bool(max([r["M_read_max_abs"] for r in R["t3_warm"]] + [q["M_read_max_abs"]]) == 0.0),
                     "update_block_grad_after_5_steps": {"tensors": len(upd), "none": int(sum(g is None for g in upd)),
                                                         "all_zero_or_none": bool(all(g is None or not bool((g != 0).any()) for g in upd))},
                     "other_groups_grad": {g: bool(any(p.grad is not None and bool((p.grad != 0).any()) for p in ps)) for g, ps in model.groups.items()}}
    R["t3_wiped"]["pass"] = bool(R["t3_wiped"]["M_equals_start_at_every_read"] and R["t3_wiped"]["update_block_grad_after_5_steps"]["all_zero_or_none"]
                                 and all(x > 0 for x in R["t3_carry_M_after_first_moment_dev_from_start_norm"]))
    log("T3", json.dumps({k: v for k, v in R["t3_wiped"].items() if k != "other_groups_grad"}))
    zero_grads(model); R["ended"] = time.strftime("%F %T"); R["gpu_peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
    C.wj(OUT / f"s6_t2t3{SFX}.json", R)


# ----------------------------------------------------------------------------------------------------------------------- #
def t4(agent=None, info=None):
    R = head()
    if agent is None:
        agent, info = SM.load_agent()
    R["agent"] = info
    store = SD.ClipStore("training"); grader = grader_once(); index_reader = T.parallel_index_reader(agent)
    mem = SD.memory_moment_set()
    plan_on = SD.plan_pass(store, SEED, 1, True, mem); plan_off = SD.plan_pass(store, SEED, 1, False, None)
    R["pass_plan"] = {"oversample_on": {k: v for k, v in plan_on.items() if k != "batches"}, "oversample_off": {k: v for k, v in plan_off.items() if k != "batches"}}
    N = 50
    for mode in SM.MODES:
        torch.cuda.reset_peak_memory_stats(); model = build(agent, mode); opts = model.optimisers(); np.random.seed(SEED + 1)
        recs = warm_steps(model, store, grader, index_reader, opts, plan_on["batches"], N, f"T4 {mode}")
        every = [{"step": i + 1, **{k: round(r[k], 4) for k in ("loss",) + ST.LESSONS}, "gate_mean": round(float(np.mean(r["gate_mean_per_slot"])), 4),
                  "M_norm": round(r["M_norm_last"], 4), "grade_wait_s": r["grade_wait_s"], "step_s": r["step_s"]} for i, r in enumerate(recs) if (i + 1) % LOG_EVERY == 0]
        mean = lambda k, a, b: float(np.mean([r[k] for r in recs[a:b]]))
        first, last = recs[:10], recs[-10:]
        R[mode] = {"steps": N, "every_10": every, "finite_all": all(r["finite"] for r in recs), "map_rows_per_step": mean("map_rows", 0, N),
                   "losses_first10_vs_last10": {k: [float(np.mean([r[k] for r in first])), float(np.mean([r[k] for r in last]))] for k in ("loss",) + ST.LESSONS},
                   "gate_mean_per_slot_step1": recs[0]["gate_mean_per_slot"], "gate_mean_per_slot_step50": recs[-1]["gate_mean_per_slot"], "M_norm_step50": recs[-1]["M_norm_last"], "M_read_max_abs_max": max(r["M_read_max_abs"] for r in recs),
                   "timing_steps_10_50": {"step_s": mean("step_s", 9, N), "forward_s": mean("forward_s", 9, N), "grade_wait_s": mean("grade_wait_s", 9, N), "backward_s": mean("backward_s", 9, N)},
                   "gpu_peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30, "records": recs}
        R[mode]["losses_falling"] = {k: R[mode]["losses_first10_vs_last10"][k][1] < R[mode]["losses_first10_vs_last10"][k][0] for k in ("loss",) + ST.LESSONS}
        log(f"T4 {mode}", json.dumps({k: R[mode][k] for k in ("finite_all", "losses_first10_vs_last10", "M_norm_step50", "timing_steps_10_50", "gpu_peak_alloc_gb")}),
            "gate step1", np.round(recs[0]["gate_mean_per_slot"], 3).tolist(), "step50", np.round(recs[-1]["gate_mean_per_slot"], 3).tolist())
        del model, opts; torch.cuda.empty_cache()
    # ---- T5 projections (carry-mode timing; wiped is the same work) --------------------------------------------------------------
    tm = R["carry"]["timing_steps_10_50"]; s = tm["step_s"]
    j1 = json.load(open(OUT / "s6_t1.json")) if (OUT / "s6_t1.json").exists() else None     # the build-3 check timing (step-0 measures are not re-run in version 2)     # check = one in-order replay + the measures (grading)
    check_s = (min(j1["carry"]["replay_wall_s"], j1["wiped"]["replay_wall_s"]) + j1["step0_018b"]["time_s"]["measures"]) if j1 else None
    proj = {}
    for key, plan in (("oversample_on", plan_on), ("oversample_off", plan_off)):
        n = plan["steps"]; checks = n // ST.CHECK_EVERY + 1
        per_pass = n * s + (checks * check_s if check_s else 0)
        proj[key] = {"steps_per_pass": n, "checks_per_pass": checks, "train_h_per_pass": n * s / 3600, "check_h_per_pass": (checks * check_s / 3600) if check_s else None,
                     "wall_h_per_pass": per_pass / 3600, "wall_h_3_passes": 3 * per_pass / 3600 + (check_s / 3600 if check_s else 0)}
    share = tm["grade_wait_s"] / s
    # the grading pool runs on 2 cores; its work per step W is CPU-bound and splits over cores up to the 10 workers: wait(k) ~ max(0, W * 2 / k - overlap)
    W = tm["grade_wait_s"] + tm["forward_s"] * 0.5                                    # lower-bound estimate: the wait plus the part overlapped with the forward
    proj["grading"] = {"wait_share_of_step": share, "grading_dominates": share > 0.5, "wait_s_measured_2_cores": tm["grade_wait_s"],
                       "estimate_note": "CPU-bound pool: wait(k cores) ~ max(0, W * 2 / k - forward_s), W = wait + 0.5 x forward (what was overlapped); not run",
                       "estimated_wait_s_4_cores": max(0.0, W * 2 / 4 - tm["forward_s"]), "estimated_wait_s_8_cores": max(0.0, W * 2 / 8 - tm["forward_s"]),
                       "step_s_est_4_cores": tm["forward_s"] + tm["backward_s"] + max(0.0, W * 2 / 4 - tm["forward_s"]),
                       "step_s_est_8_cores": tm["forward_s"] + tm["backward_s"] + max(0.0, W * 2 / 8 - tm["forward_s"])}
    R["t5"] = {"timing_steps_10_50_carry": tm, "timing_steps_10_50_wiped": R["wiped"]["timing_steps_10_50"], "check_s_step0": check_s, "projection": proj,
               "pinned_cores": sorted(os.sched_getaffinity(0))}
    R["ended"] = time.strftime("%F %T"); C.wj(OUT / f"s6_t4{SFX}.json", R)
    log("T5", json.dumps(R["t5"]))


def run_all():
    """one process: the model and the grading pool are loaded once (the NAS reads at ~4 MB/s today, flag in the report). Build 4: MEMORY_VERSION=2 -> T1-T4 on the version-2 model."""
    t0 = time.time(); agent, info = SM.load_agent(); log(f"agent loaded in {time.time() - t0:.0f} s")
    for name, fn in (("params", t_params), ("t1", t1), ("t2map", t2map), ("t2t3", t2t3), ("t4", t4)):
        t0 = time.time(); fn(agent, info); log(f"== {name} done in {(time.time() - t0) / 60:.1f} min"); torch.cuda.empty_cache()
    if "g" in GRADER:
        GRADER["g"].close()


def run_rest():
    """after the 2026-11-09 session restart killed the chain in T4: routing on a map batch, then T4/T5, one process."""
    t0 = time.time(); agent, info = SM.load_agent(); log(f"agent loaded in {time.time() - t0:.0f} s")
    for name, fn in (("t2map", t2map), ("t4", t4)):
        t0 = time.time(); fn(agent, info); log(f"== {name} done in {(time.time() - t0) / 60:.1f} min"); torch.cuda.empty_cache()
    if "g" in GRADER:
        GRADER["g"].close()


if __name__ == "__main__":
    try:
        {"params": t_params, "t1": t1, "t2t3": t2t3, "t2map": t2map, "t4": t4, "all": run_all, "rest": run_rest}[sys.argv[1]]()
    finally:
        if "g" in GRADER:                      # 2026-11-10: a standalone test left its 10 grading workers orphaned (pilot step 0)
            GRADER["g"].close()
    sys.stdout.flush(); os._exit(0)
