"""Build step 5 (Sid, 2026-11-11): the short check of the recall lesson (s6_model.py VERSION 3, s6_train.recall_lesson, weight 2.0).
  s9_short.py before          the "before" columns: every rule measure with THIS code on build 4's step-930 checkpoint (version 2) and on the
                              pilot's carry checkpoint at step 930 (version 1), all 84 picking clips, carried and reset. Neither has a trained
                              read-out, so recall is the LINEAR PROBE's gap (the design note's reader). -> out/build5_before.json
  s9_short.py run [--resume]  carry, version 3, seed 0, the first 930 steps of the pilot's seeded pass-1 plan. Probes at steps 0, 310, 620 (fixed
                              20-clip subset) and 930 (all 84 clips), each a replay in order in carry mode AND with the memory reset to M0 before
                              every moment (same weights). At 930 also the full standard check in both modes.
                              RESUME TEST (on purpose): the first launch stops after step 310's probe; before stopping it runs steps 311-320
                              WITHOUT saving them (the "uninterrupted" losses), then exits. `--resume` restores weights, optimiser states and the
                              numpy / torch random states saved at step 310, re-runs 311-320 and compares the losses, then continues to 930.
Rule measures (in_picking moments after the 4 warm-up moments of a chain; recall rows need 2 s / 4 s of history in the chain):
  (a) slots alike: mean pairwise cosine of the memory read                         <= 0.90
  (b) recall at 2 s: recall lesson's loss, carried vs reset, (reset - carried) / reset   >= 0.05
  (c) recall at 4 s: same                                                           >= 0.06
  (d) absolute change per moment ||M_t - M_(t-1)|| / ||M0||                         >= 0.025
  (e) memory norm ||M read|| / ||M0||                                               >= 0.5      (version 1 has no M0: 64 = 16 rows of norm 16)
  (f) gate mean over the 16 slots >= 0.10; >= 8 slots >= 0.05; max / min >= 3
  (g) no NaN / inf      (h) pick grade on the 1,131 graded moments >= 0.74
Outputs: {experiment_store}/runs/build5_short/ (run.json, status.json, log.txt, resume.pt, ckpt/, dumps/), out/build5_short.json."""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import s6_env as SE
import pai_common as C
C.setup_env()
import pai_model_b as MB
SE.patch_grader(MB)
import s6_model as SM, s6_data as SD, s6_train as ST
import pai_train as T
import s7_pilot as P
import s8_short as S8

RUNS = SD.A3 / "runs" / "build5_short"; OUT = ST.OUT
SEED, VERSION, N_STEPS, PROBE_AT, N_SUBSET, LOG_EVERY, RESUME_AT, RESUME_N = 0, 3, 930, (310, 620, 930), 20, 100, 310, 10
RULE = {"a_cos_read_max": 0.90, "b_recall_2s_min": 0.05, "c_recall_4s_min": 0.06, "d_abs_change_min": 0.025, "e_norm_ratio_min": 0.5,
        "f_gate_mean_min": 0.10, "f_gate_slots_min": 8, "f_gate_slot_floor": 0.05, "f_gate_ratio_min": 3.0, "h_pick_grade_min": 0.74}
RESUME_RULE = {"first_step_max_abs_diff": 0.005, "ten_steps_max_abs_diff": 0.05}
B4 = SD.A3 / "runs" / "build4_short" / "ckpt" / "step0930.pt"; PILOT = SD.A3 / "runs" / "pilot_seed0" / "carry" / "ckpt" / "p00.2.pt"
RATIO_BATCH = 130                                                  # the map batch of T2 (24 map rows) of the seeded pass-1 plan
dev = "cuda"
log = S8.log


def m0_norm(model) -> float:
    return float(model.start.M0.detach().norm()) if model.version >= 2 else 64.0


def krr(X, Y, folds, lam_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """out/build5_measure.py's linear probe: leave-clips-out kernel ridge, best ridge strength of the grid; returns the test squared error."""
    X = X.to(dev).double(); Y = Y.to(dev).double(); best = None
    for lam in lam_grid:
        se = 0.0; s0 = 0.0; n = 0
        for te in folds:
            tr = ~te; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
            K = Xtr @ Xtr.T / X.shape[1]; ym = Y[tr].mean(0)
            A = torch.linalg.solve(K + lam * K.diagonal().mean() * torch.eye(K.shape[0], device=dev, dtype=torch.float64), Y[tr] - ym)
            Pr = (Xte @ Xtr.T / X.shape[1]) @ A + ym
            se += float(((Pr - Y[te]) ** 2).sum()); s0 += float(((ym - Y[te]) ** 2).sum()); n += Y[te].numel()
        r = {"lam": lam, "mse": se / n, "r2": 1 - se / s0}
        if best is None or r["mse"] < best["mse"]:
            best = r
    return best


def probe_gap(dc: dict, dr: dict, clips: list) -> dict:
    """linear probe on [current standardised block means ; memory written], carried vs reset, target = the block means 2 s / 4 s ago (z);
    5 folds of clips; the gap = (reset error - carried error) / reset error."""
    out = {}; toks = [t for t in dc if dc[t]["in_picking"]]; cidx = {c: i for i, c in enumerate(clips)}
    by = {}
    for t, d in dc.items():
        by[(d["clip"], d["row"])] = t
    for L in SM.RECALL_LAGS:
        sel = [t for t in toks if dc[t]["chain"] >= max(L, P.WARM_CHAIN)]
        if len(sel) < 50:
            out[f"{L // 2}s"] = None; continue
        past = [by[(dc[t]["clip"], dc[t]["row"] - L)] for t in sel]
        Y = torch.stack([dc[t]["_z"] for t in past]); Xn = torch.stack([dc[t]["_z"] for t in sel]); Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-6)
        fold = torch.tensor([cidx[dc[t]["clip"]] % 5 for t in sel]); folds = [(fold == f).to(dev) for f in range(5)]
        r = {"samples": len(sel), "now_only": krr(Xn, Y, folds)}
        for name, d in (("carried", dc), ("reset", dr)):
            Xm = torch.stack([d[t]["_M"] for t in sel]); Xm = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-6)
            r[name] = krr(torch.cat([Xn / math.sqrt(Xn.shape[1]), Xm / math.sqrt(Xm.shape[1])], 1), Y, folds)
        r["gap"] = (r["reset"]["mse"] - r["carried"]["mse"]) / r["reset"]["mse"]; out[f"{L // 2}s"] = r
    return out


def health(dump: dict, last_M: dict, ref_norm: float) -> dict:
    st = S8.memory_stats(dump, last_M)
    after = [d for d in dump.values() if d["in_picking"] and d["chain"] >= P.WARM_CHAIN]
    gate = np.array(st["gate_mean_per_slot"])
    st.update(ref_norm=ref_norm, abs_change_per_moment=float(np.mean([d["abs_change"] for d in after])), abs_change_over_ref=float(np.mean([d["abs_change"] for d in after]) / ref_norm),
              norm_over_ref=float(st["M_read_norm_mean"] / ref_norm), gate_mean=float(gate.mean()))
    for L in SM.RECALL_LAGS:
        k = f"recall_{L // 2}s"; v = [d[k] for d in after if k in d and d["chain"] >= L]
        st[k] = float(np.mean(v)) if v else None; st[k + "_n"] = len(v)
    if len(last_M) >= 3:
        X = torch.stack([last_M[c].flatten() for c in sorted(last_M)]); X = torch.nn.functional.normalize(X - X.mean(0), dim=-1); cm = X @ X.T; n = cm.shape[0]
        st["between_clip_cos_mean_removed"] = float((cm.sum() - cm.diagonal().sum()) / (n * (n - 1)))
    return st


def measure(model, store_pk, clips, run_dir, tag) -> dict:
    """carried and reset replays of the same weights -> rule measures + information (no grading here)."""
    ref = m0_norm(model); res = {"tag": tag, "clips": len(clips), "ref_norm_M0": ref}; dumps = {}
    for mode, name in (("carry", "carried"), ("wiped", "reset")):
        model.mode = mode; t0 = time.time(); dump, last_M = S8.replay_memory(model, store_pk, clips, progress=lambda s: log(run_dir, s), keep=True)
        res[name] = health(dump, last_M, ref); res[name]["replay_s"] = round(time.time() - t0, 1); dumps[name] = dump
    model.mode = "carry"
    res["probe"] = probe_gap(dumps["carried"], dumps["reset"], clips)
    c, r = res["carried"], res["reset"]
    res["recall_gap"] = {k: ((r[f"recall_{k}"] - c[f"recall_{k}"]) / r[f"recall_{k}"] if (c.get(f"recall_{k}") is not None and r.get(f"recall_{k}")) else None) for k in ("2s", "4s")}
    res["read_norms"] = S8.read_norms(model)
    if model.version >= 3:
        ro = model.read_out; res["read_norms"].update(read_out_output_weight_norm=float(ro.out.weight.norm()), read_out_attention_out_norm=float(ro.attn.o.weight.norm()))
    return res, dumps


def rule_eval(res: dict, pick_grade, recall_source: str = "trained read-out") -> dict:
    c = res["carried"]; g = np.array(c["gate_mean_per_slot"])
    gap = res["recall_gap"] if recall_source == "trained read-out" else {k: (res["probe"][k]["gap"] if res["probe"].get(k) else None) for k in ("2s", "4s")}
    val = {"a": c["cos_read_mean"], "b": gap["2s"], "c": gap["4s"], "d": c["abs_change_over_ref"], "e": c["norm_over_ref"],
           "f": {"gate_mean": float(g.mean()), "slots_ge_0.05": int((g >= RULE["f_gate_slot_floor"]).sum()), "max_over_min": float(g.max() / g.min()), "min": float(g.min()), "median": float(np.median(g)), "max": float(g.max())},
           "g_non_finite_moments": c["non_finite_moments"] + res["reset"]["non_finite_moments"], "h": pick_grade}
    ok = {"a_slots_alike": val["a"] is not None and val["a"] <= RULE["a_cos_read_max"], "b_recall_2s": val["b"] is not None and val["b"] >= RULE["b_recall_2s_min"],
          "c_recall_4s": val["c"] is not None and val["c"] >= RULE["c_recall_4s_min"], "d_abs_change": val["d"] >= RULE["d_abs_change_min"], "e_norm": val["e"] >= RULE["e_norm_ratio_min"],
          "f_gate": bool(val["f"]["gate_mean"] >= RULE["f_gate_mean_min"] and val["f"]["slots_ge_0.05"] >= RULE["f_gate_slots_min"] and val["f"]["max_over_min"] >= RULE["f_gate_ratio_min"]),
          "g_finite": val["g_non_finite_moments"] == 0, "h_pick_grade": pick_grade is not None and pick_grade >= RULE["h_pick_grade_min"]}
    return {"values": val, "conditions": ok, "failed": [k for k, v in ok.items() if not v], "pass": all(ok.values()), "rule": RULE, "recall_source": recall_source}


def before():
    gate = C.memory_gate(True)   # first thing: loading the agent can take many minutes on a busy machine and the load moves meanwhile
    agent, info = SM.load_agent(); store_pk = SD.ClipStore("picking"); R = {"started": time.strftime("%F %T"), "agent": info, "gate": gate}
    b4 = json.load(open(OUT / "build4_short.json")); pil = json.loads((PILOT.parent.parent / "run.json").read_text()); pc = next(c for c in pil["checks"] if c["step"] == 930)
    known = {"build4_step930": b4["step930"]["check"]["pick_grade_mean_map"], "pilot_step930": pc["pick_grade_mean_map"]}
    for name, path in (("build4_step930", B4), ("pilot_step930", PILOT)):
        model, meta = SM.MemoryModel.load_checkpoint(agent, path); assert model.mode == "carry"
        res, _ = measure(model, store_pk, store_pk.clips, None, name)
        R[name] = {"checkpoint": str(path), "version": model.version, "measures": res, "pick_grade_from_its_own_check": known[name],
                   "rule": rule_eval(res, known[name], recall_source="linear probe (no trained read-out in this checkpoint)")}
        log(None, name, json.dumps(R[name]["rule"]["values"]), "failed", R[name]["rule"]["failed"])
        del model; torch.cuda.empty_cache()
    R["ended"] = time.strftime("%F %T"); C.wj(OUT / "build5_before.json", R)


def rng_state():
    return {"numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def set_rng(s):
    np.random.set_state(s["numpy"]); torch.set_rng_state(s["torch"]); torch.cuda.set_rng_state_all(s["cuda"])


def run(resume: bool):
    run_dir = RUNS; (run_dir / "ckpt").mkdir(parents=True, exist_ok=True); (run_dir / "dumps").mkdir(exist_ok=True)
    gate = C.memory_gate(True)
    log(run_dir, f"start build5 short check: carry, version {VERSION}, seed {SEED}, {N_STEPS} steps, w_recall {ST.W_RECALL}, resume={resume} affinity={sorted(os.sched_getaffinity(0))} nice={os.nice(0)} gate={gate}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    agent, info = SM.load_agent(); torch.manual_seed(SEED); model = SM.MemoryModel(agent, "carry", VERSION).cuda()
    store, store_pk, pk = SD.ClipStore("training"), SD.ClipStore("picking"), ST.picking_frame()
    ev = P.event_sets(store_pk.idx); mem = SD.memory_moment_set(); subset = store_pk.clips[:N_SUBSET]
    grader = MB._GRADER = MB.Grader(ST.POOL); index_reader = T.parallel_index_reader(agent); opts = model.optimisers()
    RES = run_dir / "resume.pt"; RJ = run_dir / "run.json"
    try:
        plan = SD.plan_pass(store, SEED, 1, True, mem); assert plan["steps"] >= N_STEPS
        ratio_items = next(iter(SD.Prefetcher(store, plan["batches"][RATIO_BATCH:RATIO_BATCH + 1])))
        if resume:
            assert RES.exists() and RJ.exists(), "nothing to resume"
            z = torch.load(RES, map_location="cpu", weights_only=False); assert z["mode"] == "carry" and z["version"] == VERSION and z["seed"] == SEED
            model.load_trainable(z["weights"])
            for o, sd in zip(opts, z["opts"]):
                o.load_state_dict(sd)
            s0 = z["step"]; rec = json.loads(RJ.read_text()); assert rec.get("ended") is None, "run already ended"
            assert len(rec["losses"]["loss"]) == s0, (len(rec["losses"]["loss"]), s0)
            set_rng(z["rng"]); ST._ROUTING["checked"] = False
            rec.setdefault("resumes", []).append({"at": time.strftime("%F %T"), "step": s0, "gate": gate, "random_states_restored": True})
            log(run_dir, f"RESUMED at step {s0} (next step {s0 + 1}); weights, optimiser states and numpy / torch random states restored")
        else:
            assert not RES.exists(), f"{RES} exists: use --resume"
            rec = {"run": "build5_short", "mode": "carry", "version": VERSION, "seed": SEED, "steps_planned": N_STEPS, "probe_at": list(PROBE_AT), "subset_clips": subset,
                   "start": info, "memory_gate": gate, "affinity": sorted(os.sched_getaffinity(0)), "new_params": model.new_param_count(), "rule": RULE, "resume_rule": RESUME_RULE,
                   "target_stats": model.read_out.stats_info, "pass_plan": {k: v for k, v in plan.items() if k != "batches"},
                   "settings": {"w_human": ST.W_HUMAN, "w_pull_away": ST.W_PULL, "w_recall": ST.W_RECALL, "recall_lags_moments": list(SM.RECALL_LAGS), "oversample": True,
                                "batch_windows": SD.BATCH_WINDOWS, "window": SD.WIN, "warm": SD.WARM, "lr": {"generator": 3e-5, "scorer": 1e-4, "memory_blocks_M0_read_out": 1e-4}},
                   "losses": {k: [] for k in ("loss",) + ST.lessons_for(VERSION) + ("recall_raw", "recall_2s", "recall_4s", "recall_zero_reference")}, "steps": [], "probes": [], "started": time.strftime("%F %T")}
            s0 = 0

        def do_probe(step):
            t0 = time.time(); full = step == N_STEPS; clips = store_pk.clips if full else subset
            st, dumps = measure(model, store_pk, clips, run_dir, f"step {step} {'all 84' if full else f'subset {len(clips)}'} clips"); st["step"] = step
            model.eval(); st["gradient_norms"] = {k: v for k, v in ST.routing_grads(model, SD.window_tensors(ratio_items), grader, index_reader).items()}; st["gradient_norms"]["batch"] = RATIO_BATCH
            if step > 0:
                ck = run_dir / "ckpt" / f"step{step:04d}.pt"; model.save_checkpoint(ck, {"step": step, "seed": SEED, "run": "build5_short"}); st["ckpt"] = str(ck)
            pick = None
            if full:
                for name, mode in (("carried", "carry"), ("reset", "wiped")):
                    model.mode = mode; c, grades = P.pilot_measures(model, dumps[name], pk, grader, None, ev); c["gates"] = S8.gates(c)
                    st[name]["check"] = {k: v for k, v in c.items() if k != "pick_grade_per_moment"}
                    T.save_dump(run_dir / "dumps" / f"step{step:04d}_{name}.npz", dumps[name]); np.save(run_dir / "dumps" / f"step{step:04d}_{name}_grades.npy", grades)
                model.mode = "carry"; pick = st["carried"]["check"]["pick_grade_mean_map"]
            st["rule"] = rule_eval(st, pick); st["probe_wall_s"] = round(time.time() - t0, 1); rec["probes"].append(st); C.wj(RJ, rec)
            if step > 0:
                torch.save({"weights": model.trainable_state(), "opts": [o.state_dict() for o in opts], "step": step, "mode": "carry", "version": VERSION, "seed": SEED, "rng": rng_state()}, RES)
            v = st["rule"]["values"]; gr = st["gradient_norms"].get("ratio_recall_over_driving") or {}
            C.wj(run_dir / "status.json", {"updated": time.strftime("%F %T"), "step": step, "values": v, "failed": st["rule"]["failed"]})
            log(run_dir, f"PROBE step {step} ({st['tag']}): (a) cos {v['a']:.4f} (b) recall 2s gap {v['b']} [carried {st['carried']['recall_2s']} reset {st['reset']['recall_2s']}] (c) 4s gap {v['c']} "
                         f"[{st['carried']['recall_4s']} / {st['reset']['recall_4s']}] (d) change {v['d']:.4f} (e) norm {v['e']:.3f} (f) gate mean {v['f']['gate_mean']:.3f} min/med/max {v['f']['min']:.3f}/{v['f']['median']:.3f}/{v['f']['max']:.3f} "
                         f"slots {v['f']['slots_ge_0.05']} | probe gap 2s {(st['probe'].get('2s') or {}).get('gap')} 4s {(st['probe'].get('4s') or {}).get('gap')} | grad ratio update {gr.get('update_block')} gate {gr.get('gate')} M0 {gr.get('start_state')} "
                         f"| failed {st['rule']['failed']} ({st['probe_wall_s']:.0f} s)")
            if full:
                c, r = st["carried"]["check"], st["reset"]["check"]
                log(run_dir, f"CHECK step {step}: pick grade carried {c['pick_grade_mean_map']:.4f} reset {r['pick_grade_mean_map']:.4f} hard {c['hard_picks_above10']} go {c['go_rate_pull_away']} ade32 {c['closest_of_32_ade3_all']:.4f} "
                             f"follows {c['follows_previous']['rate']:.3f} gates {c['gates']['pass']} | rule {st['rule']['pass']} failed {st['rule']['failed']}")

        if not resume:
            do_probe(0); np.random.seed(SEED + 1)                                         # the vocabulary term's draws (as the pilot, pass 1)
        stop = None; pf = SD.Prefetcher(store, plan["batches"][s0:N_STEPS]); KEYS = list(rec["losses"])
        test_resumed = [] if (resume and s0 == RESUME_AT and "uninterrupted" in rec.get("resume_test", {})) else None
        for step in range(s0 + 1, N_STEPS + 1):
            items = next(pf); model.train()
            q = ST.window_step(model, SD.window_tensors(items), grader, index_reader, opts); q["data_wait_s"] = round(pf.last_wait_s, 3)
            for k in KEYS:
                rec["losses"][k].append(q[k])
            rec["steps"].append({k: q.get(k) for k in ("map_rows", "grade_wait_s", "forward_s", "backward_s", "step_s", "data_wait_s", "M_norm_last", "recall_forward_s")} | {"gate_mean": float(np.mean(q["gate_mean_per_slot"]))})
            if test_resumed is not None and len(test_resumed) < RESUME_N:
                test_resumed.append({k: q[k] for k in KEYS})
                if len(test_resumed) == RESUME_N:
                    un = rec["resume_test"]["uninterrupted"]; d = [abs(a["loss"] - b["loss"]) for a, b in zip(un, test_resumed)]
                    rec["resume_test"].update(resumed=test_resumed, abs_diff_loss=d, first_step_abs_diff=d[0], max_abs_diff=max(d), rule=RESUME_RULE,
                                              max_abs_diff_per_lesson={k: max(abs(a[k] - b[k]) for a, b in zip(un, test_resumed)) for k in KEYS},
                                              matched=bool(d[0] <= RESUME_RULE["first_step_max_abs_diff"] and max(d) <= RESUME_RULE["ten_steps_max_abs_diff"]))
                    log(run_dir, f"RESUME TEST steps {RESUME_AT + 1}-{RESUME_AT + RESUME_N}: |loss diff| first {d[0]:.5f} max {max(d):.5f} matched {rec['resume_test']['matched']}"); C.wj(RJ, rec)
            if not q["finite"]:
                stop = f"non-finite loss at step {step}"; log(run_dir, "STOP " + stop); break
            if step % LOG_EVERY == 0:
                H = rec["losses"]; k = -LOG_EVERY; S = rec["steps"][k:]
                log(run_dir, f"step {step}/{N_STEPS} " + " ".join(f"{kk} {np.mean(H[kk][k:]):.4f}" for kk in H)
                             + f" | map {np.mean([s['map_rows'] for s in S]):.1f} step {np.mean([s['step_s'] for s in S]):.2f}s wait {np.mean([s['grade_wait_s'] for s in S]):.2f}s "
                               f"gate {np.mean([s['gate_mean'] for s in S]):.3f} Mnorm {np.mean([s['M_norm_last'] for s in S]):.1f} ram {C.avail_ram_gb():.0f} GB")
                C.wj(RJ, rec)
            if C.avail_ram_gb() < 20:
                stop = "available RAM < 20 GB"; log(run_dir, "STOP " + stop); break
            if step in PROBE_AT:
                do_probe(step)
                if step == RESUME_AT and not resume:                                      # the resume test: 10 unsaved steps, then a planned exit
                    un = []
                    for j in range(RESUME_N):
                        items = next(pf); model.train(); q = ST.window_step(model, SD.window_tensors(items), grader, index_reader, opts); un.append({k: q[k] for k in KEYS})
                    rec["resume_test"] = {"at_step": RESUME_AT, "uninterrupted": un, "stopped_at": time.strftime("%F %T")}; C.wj(RJ, rec)
                    log(run_dir, f"PLANNED STOP after step {RESUME_AT} for the resume test (steps {RESUME_AT + 1}-{RESUME_AT + RESUME_N} run unsaved; losses recorded); relaunch with --resume")
                    pf.close(); return
        pf.close()
        rec["stopped"] = stop or f"reached {N_STEPS} steps"; rec["ended"] = time.strftime("%F %T"); C.wj(RJ, rec)
        last = rec["probes"][-1]; done = last["step"] == N_STEPS
        H = rec["losses"]; S = rec["steps"]
        final = {"run": str(run_dir), "stopped": rec["stopped"], "steps_done": len(H["loss"]), "resumes": rec.get("resumes", []), "started": rec["started"], "ended": rec["ended"],
                 "resume_test": {k: v for k, v in rec.get("resume_test", {}).items()}, "probes": rec["probes"], "step930": last if done else None, "rule": last.get("rule"),
                 "verdict": ("PASS" if last["rule"]["pass"] else "FAIL") if done else "INCOMPLETE",
                 "training": {"gate_mean_first_100": float(np.mean([s["gate_mean"] for s in S[:100]])), "gate_mean_last_100": float(np.mean([s["gate_mean"] for s in S[-100:]])),
                              "losses_first_100": {k: float(np.mean(v[:100])) for k, v in H.items()}, "losses_last_100": {k: float(np.mean(v[-100:])) for k, v in H.items()},
                              "recall_raw_every_100": [float(np.mean(H["recall_raw"][i:i + 100])) for i in range(0, len(H["recall_raw"]), 100)],
                              "recall_over_zero_reference_every_100": [float(np.mean(H["recall_raw"][i:i + 100]) / np.mean(H["recall_zero_reference"][i:i + 100])) for i in range(0, len(H["recall_raw"]), 100)],
                              "step_s_mean": float(np.mean([s["step_s"] for s in S])), "recall_forward_s_mean": float(np.mean([s["recall_forward_s"] for s in S]))}}
        C.wj(OUT / "build5_short.json", final)
        log(run_dir, f"ENDED: {rec['stopped']}; verdict {final['verdict']} failed {last['rule']['failed']}")
    finally:
        grader.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=("before", "run")); ap.add_argument("--resume", action="store_true"); a = ap.parse_args()
    try:
        before() if a.what == "before" else run(a.resume)
    finally:
        if MB._GRADER is not None and hasattr(MB._GRADER, "close"):
            MB._GRADER.close()
    sys.stdout.flush(); os._exit(0)
