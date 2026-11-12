"""The pilot (Sid, 2026-11-10): memory carried vs memory wiped, seed 0, same batch order, trained in parallel, judged on the picking set.
One arm per process: s7_pilot.py <carry|wiped> [--resume]   (launcher run_model_pilot.sh pins the arm's two cores, nice 19)

Trainer = build 3 as built and tested (s6_train.window_step: 2 windows x 16 moments per step, attempt-2 lessons by import, the scorer reads
memory only through its own summary, grading pool of 10 on the arm's two cores). Rules: up to 2 passes (Sid's lean plan); a check every 930 steps and at
each pass end; stop after 6 checks without a better choosable pick; chosen = best pick grade among checks passing the safety gates
(hard-brake picks above 10 m/s = 0, pull-away go rate >= 0.66). Event over-sampling on. Seed 0.
Outputs under {runs}/generator_scorer_memory/runs/pilot_seed0/<arm>/: ckpt/p<pass>.pt (+ .json: mode flag, start sha, seed, step),
dumps/p<pass>.npz (the check's 2,591 plans / logits / picks) + p<pass>_grades.npy, run.json (every check), status.json, resume.pt
(weights + optimiser states + sampler position + counters), log.txt (a progress line every 100 steps). Resumable: --resume continues from
resume.pt at the next step of the same pass plan (seeded, deterministic); it never restarts from step 0 silently (RESUMED line + run.json).
An arm stops on any non-finite loss (recorded). Checks (s6_train.run_check + pilot extras, replay in order from each clip's first moment):
pick grade (map), hard-brake picks, pull-away go rate, closest-of-32 ADE (3 s), pick-follows-previous, scorer-pick ADE (3 s) on all
in_picking and on event moments, pick grade on graded event moments (out/memory_moments_picking.parquet), slot diversity (mean pairwise
cosine between the 16 slots of the memory READ at in_picking moments after the warm-up, and of the memory WRITTEN), gate mean per slot,
memory-use norms (generator read output projection; rating head's new columns)."""
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
import pai_train as T                                           # (s1's older pai_train has no parallel_index_reader / save_dump)

RUNS = SD.A3 / "runs" / "pilot_seed0"; OUT = ST.OUT
SEED, MAX_PASSES, CHECK_EVERY, NO_IMPROVE, GO_PULL_MIN, LOG_EVERY = 0, 2, 930, 6, 0.66, 100      # 2 passes: Sid's lean plan (amended brief, 2026-11-09 23:15)
WARM_CHAIN = SD.WARM                                               # slot diversity counted from the 5th moment of a clip's chain on


def log(run_dir, *a):
    s = time.strftime("%F %T") + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(run_dir / "log.txt", "a") as fh:
        fh.write(s + "\n")


def event_sets(idx: pd.DataFrame) -> dict:
    """sample ids of the in_picking event moments (any type) and of the graded ones, from out/memory_moments_picking.parquet."""
    mm = pd.read_parquet(OUT / "memory_moments_picking.parquet")
    ev = set(zip(mm.clip_id, np.round(mm.anchor_t.astype(float), 3)))
    key = list(zip(idx["clip"], np.round(idx.anchor_t.astype(float), 3)))
    is_ev = np.array([k in ev for k in key]); pk = idx.in_picking.to_numpy(); lab = idx.has_label2.to_numpy()
    return {"event": set(idx.sample_id[is_ev & pk]), "event_graded": set(idx.sample_id[is_ev & pk & lab])}


def pairwise_cos(M: torch.Tensor) -> float:
    """mean pairwise cosine similarity between the 16 slots of one (16, D) memory (zero slots give nan -> excluded by the caller)."""
    x = torch.nn.functional.normalize(M.float(), dim=-1); c = x @ x.T; n = c.shape[0]
    return float((c.sum() - c.diagonal().sum()) / (n * (n - 1)))


@torch.no_grad()
def replay_pilot(model, store, progress=None) -> dict:
    """s6_train.replay_slice plus, per moment: gate per slot, cosine between slots of the memory read and of the memory written, chain index."""
    model.eval(); dump = {}
    for ci, c in enumerate(store.clips):
        rec = store.load(c); N = rec["tokens"].shape[0]; dev = "cuda"
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
            tok = rec["sample_id"][i]
            cos_read = pairwise_cos(M[0]) if float(M.abs().max()) > 0 else None
            cos_new = pairwise_cos(o["M_new"][0]) if float(o["M_new"].abs().max()) > 0 else None
            dump[tok] = {"proposals": o["proposals"][0].float().cpu().numpy(), "pred_logit": o["pred_logit"][0].float().cpu().numpy(),
                         "chosen": int(o["chosen"][0]), "human": rec["traj"][i].numpy(), "mc": rec["metric_cache_path"][i], "lab": bool(rec["has_label2"][i]),
                         "clip": c, "row": i, "prev": prev, "in_picking": bool(rec["in_picking"][i]), "chain": chain,
                         "gate": o["gate"][0].float().cpu().numpy(), "cos_read": cos_read, "cos_new": cos_new, "M_read_norm": float(M.norm())}
            M = model.step_memory(o["M_new"]); prev = tok; chain += 1
        if progress and (ci + 1) % 28 == 0:
            progress(f"replayed {ci + 1}/{len(store.clips)} clips")
    return dump


def pilot_measures(model, dump: dict, pk: pd.DataFrame, grader, ref: dict | None, ev: dict) -> tuple[dict, np.ndarray]:
    """s6_train.measures + the pilot extras."""
    m, grades = ST.measures(dump, pk, grader, ref)
    toks = list(pk.sample_id); gi = np.where(pk.has_label2.to_numpy())[0]
    fin = grades[..., -1]; ch = np.array([dump[toks[i]]["chosen"] for i in gi]); pick_grade = fin[np.arange(len(gi)), ch]
    gtok = [toks[i] for i in gi]; evg = np.array([t in ev["event_graded"] for t in gtok])
    per = T.per_moment({k: dump[k] for k in toks})
    ev_all = [t for t in toks if t in ev["event"]]
    m["pick_grade_per_moment"] = dict(zip(gtok, pick_grade.tolist()))
    m["event"] = {"graded_event_moments": int(evg.sum()), "pick_grade_event": float(pick_grade[evg].mean()) if evg.any() else None,
                  "pick_grade_non_event": float(pick_grade[~evg].mean()) if (~evg).any() else None,
                  "event_moments": len(ev_all), "pick_ade3_event": float(np.mean([per[t]["m6_pick_ade"] for t in ev_all])) if ev_all else None}
    after = [d for t, d in dump.items() if d["in_picking"] and d["chain"] >= WARM_CHAIN]
    cr = [d["cos_read"] for d in after if d["cos_read"] is not None]; cn = [d["cos_new"] for d in after if d["cos_new"] is not None]
    m["slots"] = {"moments_after_warmup": len(after), "cos_read_mean": float(np.mean(cr)) if cr else None, "cos_new_mean": float(np.mean(cn)) if cn else None,
                  "gate_mean_per_slot": np.mean([d["gate"] for d in after], 0).tolist() if after else None,
                  "M_read_norm_mean": float(np.mean([d["M_read_norm"] for d in after])) if after else None}
    o = model.gen_read.attn.o; m["memory_use"] = {"generator_read_out_proj_weight_norm": float(o.weight.norm()), "generator_read_out_proj_bias_norm": float(o.bias.norm()),
                                                  "rating_head_new_columns_norm": float(model.sc_read.mem_in.weight.norm())}
    return m, grades


def run_check(model, store_pk, pk, grader, ref, ev, progress=None):
    t0 = time.time()
    if model.version >= 3:                      # build 5: the replay with the per-clip buffer of past block means (recall lesson per moment)
        import s8_short as S8
        dump, _ = S8.replay_memory(model, store_pk, store_pk.clips, progress)
    else:
        dump = replay_pilot(model, store_pk, progress)
    t1 = time.time()
    m, grades = pilot_measures(model, dump, pk, grader, ref, ev)
    if model.version >= 3:
        after = [d for d in dump.values() if d["in_picking"] and d["chain"] >= WARM_CHAIN]
        m["recall"] = {k: (float(np.mean([d[k] for d in after if k in d])) if any(k in d for d in after) else None) for k in ("recall_2s", "recall_4s")}
    m["time_s"] = {"replay": round(t1 - t0, 1), "measures": round(time.time() - t1, 1)}; m["mode"] = model.mode
    return m, dump, grades


def save_resume(path, model, opts, pos, counters):
    torch.save({"weights": model.trainable_state(), "opts": [o.state_dict() for o in opts], "pos": pos, "counters": counters, "mode": model.mode, "seed": SEED}, path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=SM.MODES); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--version", type=int, default=1, choices=SM.VERSIONS); a = ap.parse_args()       # 1 = the pilot as run; 3 = with the recall lesson (build 5)
    run_dir = RUNS / a.mode; (run_dir / "ckpt").mkdir(parents=True, exist_ok=True); (run_dir / "dumps").mkdir(exist_ok=True)
    gate = C.memory_gate(True)
    log(run_dir, f"start mode={a.mode} resume={a.resume} affinity={sorted(os.sched_getaffinity(0))} nice={os.nice(0)} gate={gate} env_patch={SE.PATCHED}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    agent, info = SM.load_agent(); torch.manual_seed(SEED); model = SM.MemoryModel(agent, a.mode, a.version).cuda()
    store, store_pk, pk = SD.ClipStore("training"), SD.ClipStore("picking"), ST.picking_frame()
    ev = event_sets(store_pk.idx); mem = SD.memory_moment_set()
    grader = MB._GRADER = MB.Grader(ST.POOL); index_reader = T.parallel_index_reader(agent); opts = model.optimisers()
    RES = run_dir / "resume.pt"; RJ = run_dir / "run.json"
    if a.resume:
        assert RES.exists() and RJ.exists(), "nothing to resume"
        z = torch.load(RES, map_location="cpu", weights_only=False); assert z["mode"] == a.mode and z["seed"] == SEED
        model.load_trainable(z["weights"])
        for o, sd in zip(opts, z["opts"]):
            o.load_state_dict(sd)
        (p0, s0) = z["pos"]; cnt = z["counters"]; rec = json.loads(RJ.read_text()); assert rec.get("ended") is None, "run already ended"
        gstep, best, since = cnt["gstep"], cnt["best"], cnt["since"]
        rec.setdefault("resumes", []).append({"at": time.strftime("%F %T"), "pos": [p0, s0], "gstep": gstep, "counters": cnt, "gate": gate})
        log(run_dir, f"RESUMED at pass {p0} step {s0} (global step {gstep}); best {best} since {since}")
    else:
        assert not RES.exists(), f"{RES} exists: use --resume"
        rec = {"arm": a.mode, "seed": SEED, "start": info, "memory_gate": gate, "affinity": sorted(os.sched_getaffinity(0)), "new_params": model.new_param_count(),
               "settings": {"w_human": ST.W_HUMAN, "w_pull_away": ST.W_PULL, "hard_gate": "hard-brake picks above 10 m/s = 0", "go_gate": GO_PULL_MIN, "check_every": CHECK_EVERY,
                            "max_passes": MAX_PASSES, "no_improve_stop": NO_IMPROVE, "oversample": True, "batch_windows": SD.BATCH_WINDOWS, "window": SD.WIN, "warm": SD.WARM},
               "event_subset": {"in_picking_event": len(ev["event"]), "in_picking_graded_event": len(ev["event_graded"])},
               "version": a.version, "losses": {k: [] for k in ("loss",) + ST.lessons_for(a.version)}, "steps": [], "checks": [], "passes": [], "started": time.strftime("%F %T")}
        p0, s0, gstep, best, since = 1, 0, 0, -1.0, 0
    ref = rec["checks"][0] if rec["checks"] else None

    def do_check(progress, pos):
        nonlocal best, since, ref
        t0 = time.time(); c, dump, grades = run_check(model, store_pk, pk, grader, ref, ev, progress=lambda s: log(run_dir, s)); c.update(step=gstep, progress=progress, pass_=pos[0])
        if ref is None:
            ref = c
        ck = run_dir / "ckpt" / f"p{progress:04.1f}.pt"; model.save_checkpoint(ck, {"step": gstep, "progress": progress, "seed": SEED, "arm": a.mode, "check": {k: v for k, v in c.items() if k != "pick_grade_per_moment"}})
        T.save_dump(run_dir / "dumps" / f"p{progress:04.1f}.npz", dump); np.save(run_dir / "dumps" / f"p{progress:04.1f}_grades.npy", grades)
        c["ckpt"] = str(ck); c["check_wall_s"] = round(time.time() - t0, 1)
        if progress > 0:
            if c["gate_ok"] and c["pick_grade_mean_map"] > best:
                best, since = c["pick_grade_mean_map"], 0
            else:
                since += 1
        c["counters"] = {"best": best, "since": since}
        rec["checks"].append(c); C.wj(RJ, rec)
        save_resume(RES, model, opts, pos, {"gstep": gstep, "best": best, "since": since})
        C.wj(run_dir / "status.json", {"updated": time.strftime("%F %T"), "arm": a.mode, "pass": progress, "step": gstep, "pick_grade": c["pick_grade_mean_map"],
                                       "event_pick_grade": c["event"]["pick_grade_event"], "hard": c["hard_picks_above10"], "go_pull": c["go_rate_pull_away"], "gate_ok": c["gate_ok"],
                                       "closest32_ade3": c["closest_of_32_ade3_all"], "follows_prev": c["follows_previous"]["rate"], "slots_cos_read": c["slots"]["cos_read_mean"],
                                       "best": best, "since": since, "check_wall_s": c["check_wall_s"]})
        log(run_dir, f"CHECK pass {progress:.3f} step {gstep}: pick grade {c['pick_grade_mean_map']:.4f} event {c['event']['pick_grade_event']} hard {c['hard_picks_above10']} "
                     f"go(pull) {c['go_rate_pull_away']} ade32 {c['closest_of_32_ade3_all']:.4f} follows {c['follows_previous']['rate']:.3f} cos_read {c['slots']['cos_read_mean']} "
                     f"gate_ok {c['gate_ok']} best {best:.4f} since {since} ({c['check_wall_s']:.0f} s)")
        if since >= NO_IMPROVE:
            return f"no check passing both gates with a better pick grade for {NO_IMPROVE} checks (pass {progress:.3f})"
        return None

    stop = None
    if not a.resume:
        stop = do_check(0.0, (1, 0))
    for p in range(p0, MAX_PASSES + 1):
        if stop:
            break
        plan = SD.plan_pass(store, SEED, p, True, mem); n = plan["steps"]
        if p == p0 and a.resume and s0 >= n:
            continue
        if not any(x.get("pass_") == p for x in rec["passes"]):
            rec["passes"].append({"pass_": p, **{k: v for k, v in plan.items() if k != "batches"}})
        first = s0 + 1 if (p == p0 and a.resume) else 1
        np.random.seed(SEED + p)                                                                  # the vocabulary term's draws
        for _ in range(first - 1):                                                                # replay the draws made before the resume point is not possible
            pass                                                                                   # (np.random.choice counts vary per batch): recorded as a flag in the report
        pf = SD.Prefetcher(store, plan["batches"][first - 1:]); t_pass = time.time()
        for step in range(first, n + 1):
            items = next(pf); model.train()
            q = ST.window_step(model, SD.window_tensors(items), grader, index_reader, opts); q["data_wait_s"] = round(pf.last_wait_s, 3); gstep += 1
            for k in rec["losses"]:
                rec["losses"][k].append(q[k])
            rec["steps"].append({k: q.get(k) for k in ("map_rows", "grade_wait_s", "forward_s", "backward_s", "step_s", "data_wait_s", "M_norm_last")})
            if not q["finite"]:
                stop = f"non-finite loss at pass {p} step {step} (global {gstep})"; log(run_dir, "STOP " + stop); break
            if step % LOG_EVERY == 0:
                H = rec["losses"]; k = -LOG_EVERY
                log(run_dir, f"pass {p} step {step}/{n} (global {gstep}) " + " ".join(f"{kk} {np.mean(H[kk][k:]):.4f}" for kk in H)
                             + f" | map {np.mean([s['map_rows'] for s in rec['steps'][k:]]):.1f} step {np.mean([s['step_s'] for s in rec['steps'][k:]]):.2f}s wait {np.mean([s['grade_wait_s'] for s in rec['steps'][k:]]):.2f}s "
                               f"data {np.mean([s['data_wait_s'] for s in rec['steps'][k:]]):.2f}s ram {C.avail_ram_gb():.0f} GB | best {best:.4f} since {since}")
                C.wj(RJ, rec)
            if C.avail_ram_gb() < 20:
                stop = "available RAM < 20 GB"; log(run_dir, "STOP " + stop); break
            if step % CHECK_EVERY == 0 or step == n:
                prog = round(p - 1 + step / n, 3) if step < n else float(p)
                stop = do_check(prog, (p, step))
                if stop:
                    log(run_dir, "STOP " + stop); break
        pf.close()
        if stop:
            break
    rec["stopped"] = stop or f"reached {MAX_PASSES} passes"
    good = [c for c in rec["checks"] if c["progress"] > 0 and c["gate_ok"]]
    ch = max(good, key=lambda c: (c["pick_grade_mean_map"], -c["progress"])) if good else None
    rec["chosen"] = {"progress": ch["progress"], "step": ch["step"], "pick_grade": ch["pick_grade_mean_map"], "ckpt": ch["ckpt"]} if ch else None
    rec["ended"] = time.strftime("%F %T"); C.wj(RJ, rec)
    log(run_dir, f"ENDED: {rec['stopped']}; chosen {rec['chosen']}")
    grader.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush(); os._exit(0)
