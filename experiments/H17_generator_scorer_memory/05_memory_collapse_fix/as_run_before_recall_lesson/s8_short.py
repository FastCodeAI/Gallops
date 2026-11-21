"""Build step 4 (Sid, 2026-11-11): the short check of the memory fix (s6_model.py VERSION 2: learned start state M0, per-slot gate bias).
  s8_short.py before          the "before" column: (a) (b) (c) with THIS code on the pilot's carry checkpoint at step 930 (version 1, pass 0.217)
                              -> out/build4_before.json
  s8_short.py run [--resume]  carry mode, version 2, seed 0, the first 930 steps of the pilot's seeded pass-1 plan (same batches, same lessons,
                              learning rates, windows, event over-sampling, replay routing: s6_train.window_step unchanged). Probes at steps
                              0, 310, 620 on a fixed 20-clip subset of the picking clips and at step 930 on all 84 picking clips (replayed in
                              order from each clip's first moment); at 930 also the full standard check (pick grade, hard-brake picks,
                              pull-away go rate, closest-of-32 ADE (3 s), pick-follows-previous) and the same checkpoint in wiped mode
                              (memory reset to M0 before every moment). Outputs under {experiment_store}/runs/build4_short/
                              (run.json, status.json, log.txt, resume.pt, ckpt/step0310.pt ..., dumps/), the rule verdict in out/build4_short.json.
                              Resumable: --resume continues from resume.pt at the next step (RESUMED line in log.txt + run.json); never from
                              step 0 silently. Stops on a non-finite loss or available RAM < 20 GB.
Measures (per probe; in_picking moments after the 4 warm-up moments of a chain, as the pilot's slot diversity):
  (a) slot diversity = mean pairwise cosine between the 16 slots of the memory READ at the moment (and of the memory WRITTEN, information)
  (b) gate mean per slot (16 values)
  (c) memory change per moment = mean over the moments of ||M_t - M_(t-1)|| / ||M_(t-1)||, M_(t-1) the memory read, M_t the memory written
  (d) no NaN / inf in the memory written, the plans or the logits of any moment
  (e) pick grade on the graded in_picking moments (step 930 only)
  between clips (information): mean pairwise cosine, across clips, between the memory written at each clip's LAST in_picking moment
  read-projection norms: generator read output projection (weight, bias), rating head new columns, and (version 2) M0 and the gate biases.
PASS rule (reports/BUILD4_PREREG.md): (a) <= 0.90; (b) >= 8 of 16 slots with gate mean >= 0.05 AND max >= 3 x min; (c) >= 0.05; (d) none;
(e) >= 0.74."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import s6_env as SE
import pai_common as C
C.setup_env()
import pai_model_b as MB
SE.patch_grader(MB)
import s6_model as SM, s6_data as SD, s6_train as ST            # s6_train puts s2_train/src and s9_018b_train/src in front of s1_build/src
import pai_train as T
import s7_pilot as P

RUNS = SD.A3 / "runs" / "build4_short"; OUT = ST.OUT
SEED, VERSION, N_STEPS, PROBE_AT, N_SUBSET, LOG_EVERY = 0, 2, 930, (310, 620, 930), 20, 100
RULE = {"cos_read_max": 0.90, "gate_slots_min": 8, "gate_min": 0.05, "gate_ratio_min": 3.0, "change_min": 0.05, "pick_grade_min": 0.74}
ADE_REF, ADE_TOL, GO_MIN = 0.238, 0.02, 0.66                       # choice gates (Sid, build 4 decision 3)
PILOT_BEFORE = SD.A3 / "runs" / "pilot_seed0" / "carry" / "ckpt" / "p00.2.pt"     # the pilot's carry arm at step 930 (pass 0.217, version 1)


def log(run_dir, *a):
    s = time.strftime("%F %T") + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    if run_dir is not None:
        with open(run_dir / "log.txt", "a") as fh:
            fh.write(s + "\n")


@torch.no_grad()
def replay_memory(model, store, clips, progress=None) -> tuple[dict, dict]:
    """s7_pilot.replay_pilot plus, per moment, the relative memory change and a finite flag; and the memory written at each clip's last in_picking moment."""
    model.eval(); dump = {}; last_M = {}; dev = "cuda"
    for ci, c in enumerate(clips):
        rec = store.load(c); N = rec["tokens"].shape[0]
        tokens = rec["tokens"].to(dev); ego = rec["ego"].to(dev); traj = rec["traj"].to(dev)
        l2i = rec["lidar2img"][None].to(dev); shp = rec["img_shape"][None].to(dev)
        M = model.init_memory(1, dev); prev = None; chain = 0
        for i in range(N):
            if model.mode == "wiped":
                M = model.init_memory(1, dev)
            consecutive = i > 0 and abs(rec["anchor_t"][i] - rec["anchor_t"][i - 1] - SD.STEP_S) < 1e-6
            if not consecutive:
                M = model.init_memory(1, dev); prev = None; chain = 0
            motion = SM.motion_from_prev_future(traj[i - 1:i]) if consecutive else torch.zeros(1, 3, device=dev)
            o = model.forward_moment(tokens[i:i + 1], ego[i:i + 1], l2i, shp, M, motion, replay=False)
            Mn = o["M_new"].float(); tok = rec["sample_id"][i]; Mnorm = float(M.norm())
            dump[tok] = {"proposals": o["proposals"][0].float().cpu().numpy(), "pred_logit": o["pred_logit"][0].float().cpu().numpy(),
                         "chosen": int(o["chosen"][0]), "human": rec["traj"][i].numpy(), "mc": rec["metric_cache_path"][i], "lab": bool(rec["has_label2"][i]),
                         "clip": c, "row": i, "prev": prev, "in_picking": bool(rec["in_picking"][i]), "chain": chain,
                         "gate": o["gate"][0].float().cpu().numpy(), "cos_read": P.pairwise_cos(M[0]) if float(M.abs().max()) > 0 else None,
                         "cos_new": P.pairwise_cos(Mn[0]) if float(Mn.abs().max()) > 0 else None, "M_read_norm": Mnorm,
                         "change": float((Mn - M).norm() / Mnorm) if Mnorm > 0 else None,
                         "finite": bool(torch.isfinite(Mn).all() and torch.isfinite(o["pred_logit"]).all() and torch.isfinite(o["proposals"]).all())}
            if dump[tok]["in_picking"]:
                last_M[c] = Mn[0].cpu()
            M = model.step_memory(Mn); prev = tok; chain += 1
        if progress and (ci + 1) % 28 == 0:
            progress(f"replayed {ci + 1}/{len(clips)} clips")
    return dump, last_M


def memory_stats(dump: dict, last_M: dict) -> dict:
    after = [d for d in dump.values() if d["in_picking"] and d["chain"] >= P.WARM_CHAIN]
    cr = [d["cos_read"] for d in after if d["cos_read"] is not None]; cn = [d["cos_new"] for d in after if d["cos_new"] is not None]
    ch = [d["change"] for d in after if d["change"] is not None]
    gate = np.mean([d["gate"] for d in after], 0) if after else None
    out = {"clips": len({d["clip"] for d in dump.values()}), "moments": len(dump), "moments_after_warmup": len(after),
           "cos_read_mean": float(np.mean(cr)) if cr else None, "cos_new_mean": float(np.mean(cn)) if cn else None,
           "gate_mean_per_slot": gate.tolist() if gate is not None else None,
           "gate_min": float(gate.min()) if gate is not None else None, "gate_median": float(np.median(gate)) if gate is not None else None, "gate_max": float(gate.max()) if gate is not None else None,
           "slots_gate_ge_0.05": int((gate >= RULE["gate_min"]).sum()) if gate is not None else None,
           "gate_max_over_min": float(gate.max() / gate.min()) if gate is not None and gate.min() > 0 else None,
           "memory_change_per_moment": float(np.mean(ch)) if ch else None, "memory_change_median": float(np.median(ch)) if ch else None,
           "M_read_norm_mean": float(np.mean([d["M_read_norm"] for d in after])) if after else None,
           "finite_all": bool(all(d["finite"] for d in dump.values())), "non_finite_moments": int(sum(not d["finite"] for d in dump.values()))}
    if len(last_M) >= 2:
        X = torch.nn.functional.normalize(torch.stack([last_M[c].flatten() for c in sorted(last_M)]), dim=-1); cm = X @ X.T; n = cm.shape[0]
        out["between_clip_cos_mean"] = float((cm.sum() - cm.diagonal().sum()) / (n * (n - 1))); out["between_clip_clips"] = n
    return out


def read_norms(model) -> dict:
    o = model.gen_read.attn.o
    r = {"generator_read_out_proj_weight_norm": float(o.weight.norm()), "generator_read_out_proj_bias_norm": float(o.bias.norm()),
         "rating_head_new_columns_norm": float(model.sc_read.mem_in.weight.norm())}
    if model.version == 2:
        M0 = model.start.M0.detach().float(); x = torch.nn.functional.normalize(M0, dim=-1); cmat = x @ x.T
        r.update(M0_norm=float(M0.norm()), M0_row_norm_mean=float(M0.norm(dim=-1).mean()), M0_pairwise_cos_mean=float((cmat.sum() - cmat.diagonal().sum()) / 240),
                 gate_bias=model.update.gate_bias.detach().cpu().tolist(), gate_bias_sigmoid=torch.sigmoid(model.update.gate_bias.detach().cpu()).tolist(),
                 gate_weight_norm=float(model.update.gate.weight.norm()))
    return r


def gates(c: dict) -> dict:
    g = {"hard_picks_above10": c["hard_picks_above10"], "hard_ok": c["hard_picks_above10"] == 0, "go_rate_pull_away": c["go_rate_pull_away"],
         "go_ok": c["go_rate_pull_away"] is not None and c["go_rate_pull_away"] >= GO_MIN, "closest_of_32_ade3": c["closest_of_32_ade3_all"],
         "ade_ok": c["closest_of_32_ade3_all"] <= ADE_REF + ADE_TOL, "ade_ref_plus_tol": ADE_REF + ADE_TOL}
    g["pass"] = bool(g["hard_ok"] and g["go_ok"] and g["ade_ok"])
    return g


def rule_eval(st: dict, pick_grade: float | None) -> dict:
    a = st["cos_read_mean"] is not None and st["cos_read_mean"] <= RULE["cos_read_max"]
    b = st["gate_mean_per_slot"] is not None and st["slots_gate_ge_0.05"] >= RULE["gate_slots_min"] and st["gate_max_over_min"] is not None and st["gate_max_over_min"] >= RULE["gate_ratio_min"]
    c = st["memory_change_per_moment"] is not None and st["memory_change_per_moment"] >= RULE["change_min"]
    d = bool(st["finite_all"])
    e = pick_grade is not None and pick_grade >= RULE["pick_grade_min"]
    conds = {"a_slot_diversity": a, "b_gate_spread": b, "c_memory_change": c, "d_finite": d, "e_pick_grade": e}
    return {"conditions": conds, "failed": [k for k, v in conds.items() if not v], "pass": all(conds.values()), "rule": RULE}


def probe(model, store_pk, clips, run_dir, tag) -> tuple[dict, dict, dict]:
    t0 = time.time(); dump, last_M = replay_memory(model, store_pk, clips, progress=lambda s: log(run_dir, s))
    st = memory_stats(dump, last_M); st["read_norms"] = read_norms(model); st["replay_s"] = round(time.time() - t0, 1); st["tag"] = tag; st["mode"] = model.mode
    return st, dump, last_M


def before():
    """the pilot's carry checkpoint at step 930 (version 1) measured with the same code on all 84 picking clips."""
    agent, info = SM.load_agent(); model, meta = SM.MemoryModel.load_checkpoint(agent, PILOT_BEFORE); assert model.version == 1 and model.mode == "carry"
    store_pk = SD.ClipStore("picking")
    st, dump, _ = probe(model, store_pk, store_pk.clips, None, "before: pilot carry step 930")
    pil = json.loads((PILOT_BEFORE.parent.parent / "run.json").read_text()); pc = next(c for c in pil["checks"] if c["step"] == 930)
    R = {"started": time.strftime("%F %T"), "checkpoint": str(PILOT_BEFORE), "checkpoint_meta": {k: v for k, v in meta.items() if k != "check"}, "agent": info,
         "stats": st, "rule_abc_only": rule_eval(st, None),
         "pilot_check_same_numbers": {"pick_grade": pc["pick_grade_mean_map"], "cos_read_mean": pc["slots"]["cos_read_mean"], "gate_mean_per_slot": pc["slots"]["gate_mean_per_slot"],
                                      "M_read_norm_mean": pc["slots"]["M_read_norm_mean"], "memory_use": pc["memory_use"]},
         "agrees_with_pilot_check": {"cos_read": abs(st["cos_read_mean"] - pc["slots"]["cos_read_mean"]) < 1e-6,
                                     "gate": bool(np.allclose(st["gate_mean_per_slot"], pc["slots"]["gate_mean_per_slot"], atol=1e-6))},
         "ended": time.strftime("%F %T")}
    C.wj(OUT / "build4_before.json", R)
    print(json.dumps({k: v for k, v in st.items() if k != "gate_mean_per_slot"}, indent=1)); print("agrees", R["agrees_with_pilot_check"])


def save_resume(path, model, opts, step, extra):
    torch.save({"weights": model.trainable_state(), "opts": [o.state_dict() for o in opts], "step": step, "mode": model.mode, "version": model.version, "seed": SEED, **extra}, path)


def run(resume: bool):
    run_dir = RUNS; (run_dir / "ckpt").mkdir(parents=True, exist_ok=True); (run_dir / "dumps").mkdir(exist_ok=True)
    gate = C.memory_gate(True)
    log(run_dir, f"start build4 short check: carry, version {VERSION}, seed {SEED}, {N_STEPS} steps, resume={resume} affinity={sorted(os.sched_getaffinity(0))} nice={os.nice(0)} gate={gate} env_patch={SE.PATCHED}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    agent, info = SM.load_agent(); torch.manual_seed(SEED); model = SM.MemoryModel(agent, "carry", VERSION).cuda()
    store, store_pk, pk = SD.ClipStore("training"), SD.ClipStore("picking"), ST.picking_frame()
    ev = P.event_sets(store_pk.idx); mem = SD.memory_moment_set(); subset = store_pk.clips[:N_SUBSET]
    grader = MB._GRADER = MB.Grader(ST.POOL); index_reader = T.parallel_index_reader(agent); opts = model.optimisers()
    RES = run_dir / "resume.pt"; RJ = run_dir / "run.json"
    try:
        if resume:
            assert RES.exists() and RJ.exists(), "nothing to resume"
            z = torch.load(RES, map_location="cpu", weights_only=False); assert z["mode"] == "carry" and z["version"] == VERSION and z["seed"] == SEED
            model.load_trainable(z["weights"])
            for o, sd in zip(opts, z["opts"]):
                o.load_state_dict(sd)
            s0 = z["step"]; rec = json.loads(RJ.read_text()); assert rec.get("ended") is None, "run already ended"
            rec.setdefault("resumes", []).append({"at": time.strftime("%F %T"), "step": s0, "gate": gate})
            log(run_dir, f"RESUMED at step {s0} (next step {s0 + 1}); the vocabulary-term draws restart from the seed (as the pilot's resume)")
        else:
            assert not RES.exists(), f"{RES} exists: use --resume"
            rec = {"run": "build4_short", "mode": "carry", "version": VERSION, "seed": SEED, "steps_planned": N_STEPS, "probe_at": list(PROBE_AT), "subset_clips": subset,
                   "start": info, "memory_gate": gate, "affinity": sorted(os.sched_getaffinity(0)), "new_params": model.new_param_count(), "rule": RULE,
                   "settings": {"w_human": ST.W_HUMAN, "w_pull_away": ST.W_PULL, "oversample": True, "batch_windows": SD.BATCH_WINDOWS, "window": SD.WIN, "warm": SD.WARM,
                                "lr": {"generator": 3e-5, "scorer": 1e-4, "memory_blocks_and_M0": 1e-4}, "gates": {"hard": 0, "go_min": GO_MIN, "ade_max": ADE_REF + ADE_TOL}},
                   "losses": {k: [] for k in ("loss",) + ST.LESSONS}, "steps": [], "probes": [], "started": time.strftime("%F %T")}
            s0 = 0
        plan = SD.plan_pass(store, SEED, 1, True, mem); rec["pass_plan"] = {k: v for k, v in plan.items() if k != "batches"}
        assert plan["steps"] >= N_STEPS

        def do_probe(step):
            t0 = time.time(); full = step == N_STEPS; clips = store_pk.clips if full else subset
            st, dump, last_M = probe(model, store_pk, clips, run_dir, f"step {step} {'all 84' if full else f'subset {len(clips)}'} clips"); st["step"] = step
            if step > 0:
                ck = run_dir / "ckpt" / f"step{step:04d}.pt"; model.save_checkpoint(ck, {"step": step, "seed": SEED, "run": "build4_short"}); st["ckpt"] = str(ck)
            if full:
                c, grades = P.pilot_measures(model, dump, pk, grader, None, ev); c["gates"] = gates(c); st["check"] = {k: v for k, v in c.items() if k != "pick_grade_per_moment"}
                T.save_dump(run_dir / "dumps" / f"step{step:04d}.npz", dump); np.save(run_dir / "dumps" / f"step{step:04d}_grades.npy", grades)
                st["rule"] = rule_eval(st, c["pick_grade_mean_map"])
                model.mode = "wiped"                                                       # the same weights, memory reset to M0 before every moment
                stw, dumpw, _ = probe(model, store_pk, store_pk.clips, run_dir, f"step {step} wiped (reset to M0) all 84 clips")
                cw, gradesw = P.pilot_measures(model, dumpw, pk, grader, None, ev); cw["gates"] = gates(cw); stw["check"] = {k: v for k, v in cw.items() if k != "pick_grade_per_moment"}
                T.save_dump(run_dir / "dumps" / f"step{step:04d}_wiped.npz", dumpw); np.save(run_dir / "dumps" / f"step{step:04d}_wiped_grades.npy", gradesw)
                model.mode = "carry"; st["wiped_same_weights"] = stw
                log(run_dir, f"CHECK step {step}: pick grade carry {c['pick_grade_mean_map']:.4f} wiped(M0) {cw['pick_grade_mean_map']:.4f} hard {c['hard_picks_above10']} go {c['go_rate_pull_away']} "
                             f"ade32 {c['closest_of_32_ade3_all']:.4f} follows {c['follows_previous']['rate']:.3f} gates {c['gates']['pass']} | rule {st['rule']['pass']} failed {st['rule']['failed']}")
            st["probe_wall_s"] = round(time.time() - t0, 1); rec["probes"].append(st); C.wj(RJ, rec)
            if step > 0:
                save_resume(RES, model, opts, step, {})
            C.wj(run_dir / "status.json", {"updated": time.strftime("%F %T"), "step": step, "cos_read": st["cos_read_mean"], "gate_min_med_max": [st["gate_min"], st["gate_median"], st["gate_max"]],
                                           "change": st["memory_change_per_moment"], "finite": st["finite_all"], "between_clip_cos": st.get("between_clip_cos_mean")})
            log(run_dir, f"PROBE step {step} ({st['tag']}): cos_read {st['cos_read_mean']} cos_new {st['cos_new_mean']} gate min/med/max {st['gate_min']:.3f}/{st['gate_median']:.3f}/{st['gate_max']:.3f} "
                         f"slots>=0.05 {st['slots_gate_ge_0.05']} change {st['memory_change_per_moment']} Mnorm {st['M_read_norm_mean']:.1f} between-clip cos {st.get('between_clip_cos_mean')} "
                         f"finite {st['finite_all']} norms o {st['read_norms']['generator_read_out_proj_weight_norm']:.4f} newcols {st['read_norms']['rating_head_new_columns_norm']:.4f} ({st['probe_wall_s']:.0f} s)")

        if not resume:
            do_probe(0)
        stop = None
        np.random.seed(SEED + 1)                                                          # the vocabulary term's draws (as the pilot, pass 1)
        pf = SD.Prefetcher(store, plan["batches"][s0:N_STEPS])
        for step in range(s0 + 1, N_STEPS + 1):
            items = next(pf); model.train()
            q = ST.window_step(model, SD.window_tensors(items), grader, index_reader, opts); q["data_wait_s"] = round(pf.last_wait_s, 3)
            for k in rec["losses"]:
                rec["losses"][k].append(q[k])
            rec["steps"].append({k: q.get(k) for k in ("map_rows", "grade_wait_s", "forward_s", "backward_s", "step_s", "data_wait_s", "M_norm_last")} | {"gate_mean": float(np.mean(q["gate_mean_per_slot"]))})
            if not q["finite"]:
                stop = f"non-finite loss at step {step}"; log(run_dir, "STOP " + stop); break
            if step % LOG_EVERY == 0:
                H = rec["losses"]; k = -LOG_EVERY
                log(run_dir, f"step {step}/{N_STEPS} " + " ".join(f"{kk} {np.mean(H[kk][k:]):.4f}" for kk in H)
                             + f" | map {np.mean([s['map_rows'] for s in rec['steps'][k:]]):.1f} step {np.mean([s['step_s'] for s in rec['steps'][k:]]):.2f}s wait {np.mean([s['grade_wait_s'] for s in rec['steps'][k:]]):.2f}s "
                               f"data {np.mean([s['data_wait_s'] for s in rec['steps'][k:]]):.2f}s gate {np.mean([s['gate_mean'] for s in rec['steps'][k:]]):.3f} Mnorm {np.mean([s['M_norm_last'] for s in rec['steps'][k:]]):.1f} ram {C.avail_ram_gb():.0f} GB")
                C.wj(RJ, rec)
            if C.avail_ram_gb() < 20:
                stop = "available RAM < 20 GB"; log(run_dir, "STOP " + stop); break
            if step in PROBE_AT:
                do_probe(step)
        pf.close()
        rec["stopped"] = stop or f"reached {N_STEPS} steps"; rec["ended"] = time.strftime("%F %T"); C.wj(RJ, rec)
        last = rec["probes"][-1]
        final = {"run": str(run_dir), "stopped": rec["stopped"], "steps_done": len(rec["losses"]["loss"]), "resumes": len(rec.get("resumes", [])), "started": rec["started"], "ended": rec["ended"],
                 "probes": [{k: v for k, v in s.items() if k not in ("wiped_same_weights",)} for s in rec["probes"]],
                 "step930": last if last["step"] == N_STEPS else None, "rule": last.get("rule"), "verdict": ("PASS" if last.get("rule", {}).get("pass") else "FAIL") if last["step"] == N_STEPS else "INCOMPLETE"}
        C.wj(OUT / "build4_short.json", final)
        log(run_dir, f"ENDED: {rec['stopped']}; verdict {final['verdict']}" + (f" failed {last['rule']['failed']}" if last.get("rule") else ""))
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
