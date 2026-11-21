"""Approach 4 training script (build 2, step 5): 0.1.8b attempt 2's recipe (LORA Refiner/Generator-Joint-PAI-alpasim/s9_018b_train/src/
train_018b.py) with PREREG_APPROACH4.md Amendment 1, on model_a4.A4Model and the a4_data loader.

    run.sh train_a4.py --arm A --dry-run              # 3 steps on the GPU, then exit (build 2)
    run.sh train_a4.py --arm A --go-from-sid          # the run (refuses without the flag)

Kept from train_018b.py (imported where the code allows, copied where it is inside train()):
  * sample list and sampler: pai_train.manifests, pai_data.sampler_017 / batches_017 (seed 0 + pass), batch 32; global numpy reseeded
    SEED + pass at a pass start (the vocabulary term's draws, train_018b.py:1325)
  * lessons, one backward: generator (drive_jepa_agent_s10.generator_loss_s10, spacing weight 0) + grade + collision + drivable + human
    (pai_model_c.Model018c.scorer_lessons: human x0.5, pull-away x2, hard-brake rule; S35 grading of the generator's own 32 plans on
    pai_model_b.Grader, 10 processes); bf16 autocast for the forward as pai_model_b.Model018b.forward
  * optimisers: generator Adam 3e-5, the 30 scorer tensors Adam 1e-4 (pai_model.optimisers); new blocks Adam 1e-4 (model_a4.optimisers)
  * checks at step 0, every 930 steps and at a pass end: the 2,000 picking moments (fp32, batch 1, eval, no previous plan), the 1,131 map
    menus graded, train_018b.safety (pick grade, hard picks above 10 m/s, pull-away go rate ...), pai_train.summary, logit stats,
    per-lesson gradient shares on the rating head (4 fixed batches, seed 12345)
  * choice (PREREG §1): highest pick grade among trained checks with hard picks above 10 m/s <= step 0's AND pull-away go rate >= 0.66;
    ties to the earlier check; stops: hard gate failing 2 checks in a row, 6 checks without a better choosable pick, 3 passes, a
    non-finite loss, available RAM < 20 GB or the STOP file. The kill switch (PREREG §3) is applied later, to the chosen check.
Changed by Amendment 1 (and nothing else): no hide-and-guess (no pair stream, no guesser, no answer key, no LoRA, no encoder optimiser),
no GridMask (cached tokens), the encoder is frozen; the grading therefore has no hide-and-guess forward to overlap with and is collected
right after submit. Arms A and B differ only in --arm (the hiding probability).
Resume (TRAIN brief): a full save (weights, 3 Adam states, position, all RNGs incl. the hiding generator, run record, selection counters) at every check and every 30 min; --resume <file> restores it. Span fixed at 0.1.8b attempt 2's 1.2 passes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

import s6_env  # noqa: F401  (approach 3's navsim path shim: first)

# the grading pool spawns workers that re-import this module as __mp_main__: approach 3's grader patch must run at import time
# here (as s6_train.py does at its top), or s6_env.init_worker finds _ORIG unset in the child (TypeError: NoneType not callable)
import s6_env as _SE  # noqa: E402
import pai_model_b as _MB  # noqa: E402
_SE.patch_grader(_MB)

SEED, BATCH = 0, 32
MAX_PASSES, GATE_FAILS, NO_IMPROVE, CHECK_EVERY = 3, 2, 6, 930      # train_018b.py:63
GO_PULL_MIN = 0.66                                                  # train_018b.py:72
POOL = 10                                                           # train_018b.py:64
N_SHARE_BATCHES = 4
LESSONS = ("generator", "grade", "human", "collision", "drivable")


def step_forward(model, f, t):
    """one training forward + every lesson (no backward). Returns (loss, record, r, L)."""
    import torch
    import pai_model as PM
    ego = f["ego_status"][:, -1]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = model.forward_moment(f["tokens"], ego, f["lidar2img"], f["img_shape"], f["ego_history"], training=True, replay=True)
    pl = [q.float() for q in o["proposal_list"]]
    scores_index = model.agent.read_scores_index(t)
    gl, parts = model.agent.generator_loss_s10(pl, t["trajectory"], model.cfg, scores_index, 0.0)
    sl, d, tgt, n_wp = PM.scorer_loss(o["logit"], o["proposals"], t["trajectory"])
    r = {"score": sl, "target": tgt, "logit": o["logit"], "agents_state": o["agents_state"], "area_logit": o["area_logit"],
         "proposals": o["proposals"], "ego": ego, "gen": gl, "hide": o["hide"]}
    return r


def train_step(model, opts, batch, grader, backward: bool = True) -> dict:
    import torch
    import pai_train as T
    f, t = T.to_dev(batch)
    model.train_mode()
    r = step_forward(model, f, t)
    tg = time.time()
    g = grader.collect(grader.submit(t, r["proposals"]))
    wait = time.time() - tg
    L = model.scorer_lessons(r, t, g)
    loss = r["gen"] + L["grade"] + L["collision"] + L["drivable"] + L["human"]
    rec = {"loss": float(loss), "generator": float(r["gen"]), **{k: float(L[k]) for k in ("grade", "human", "collision", "drivable")},
           "map_rows": L["map_rows"], "zeroed_targets": L["zeroed_targets"], "grade_wait_s": wait, "pull_rows": L.get("pull_rows", 0),
           "pull_map_rows": L.get("pull_map_rows", 0), "hidden_rows": int(r["hide"].sum()), "finite": bool(torch.isfinite(loss))}
    if backward and rec["finite"]:
        loss.backward()
        enc = model.m._backbone
        rec["encoder_grads"] = sum(int(p.grad is not None) for p in enc.parameters())
        rec["encoder_requires_grad"] = sum(int(p.requires_grad) for p in enc.parameters())
        rec["grad_norm_new"] = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for _, p in model.new_params if p.grad is not None))) \
            if any(p.grad is not None for _, p in model.new_params) else 0.0
        for o in opts:
            o.step(); o.zero_grad(set_to_none=True)
    model.zero_grad_all()
    return rec


def fixed_batches(ds, n: int, seed: int = 12345):
    """pai_train.fixed_batches (:186-201) planner half: the same moments (torch.randperm(len(tr), Generator(12345))), no pair batches."""
    import torch
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=rng)[:n * BATCH].tolist()
    return [ds.batch(idx[s * BATCH:(s + 1) * BATCH]) for s in range(n)]


def lesson_shares(model, fbs, grader) -> dict:
    """train_018b.lesson_shares (:268-288) on the rating head, five lessons (no hide-and-guess, no LoRA); plus the new blocks."""
    import torch
    import pai_train as T
    rows = []
    for b in fbs:
        f, t = T.to_dev(b)
        model.train_mode()
        r = step_forward(model, f, t)
        L = model.scorer_lessons(r, t, grader.collect(grader.submit(t, r["proposals"])))
        losses = {"generator": r["gen"], **{k: L[k] for k in ("grade", "human", "collision", "drivable")}}
        n = {"head": {}, "new": {}}
        for k, v in losses.items():
            for grp, ps in (("head", model.heads["rating"]), ("new", [p for _, p in model.new_params])):
                if not v.requires_grad:
                    n[grp][k] = 0.0; continue
                gr = torch.autograd.grad(v, ps, retain_graph=True, allow_unused=True)
                n[grp][k] = float(torch.sqrt(sum((x.float() ** 2).sum() for x in gr if x is not None))) if any(x is not None for x in gr) else 0.0
        rows.append(n); model.zero_grad_all()
    sh = {}
    for grp in ("head", "new"):
        per = [{k: (v / sum(n[grp].values()) if sum(n[grp].values()) > 0 else float("nan")) for k, v in n[grp].items()} for n in rows]
        sh[grp] = {k: float(np.mean([p[k] for p in per])) for k in LESSONS}
    return {"share": sh, "norms_per_batch": rows}


def forward_dumps(model, ds_pk) -> dict:
    """the check forward: every picking moment, fp32, batch 1, eval, no previous plan (pai_model.forward_dump's settings)."""
    import torch
    model.eval_mode()
    out = {}
    with torch.no_grad():
        for i in range(len(ds_pk)):
            f, t = ds_pk.item(i)
            o = model.forward_moment(f["tokens"][None].cuda(), f["ego_status"][None].cuda()[:, -1], f["lidar2img"][None].cuda(),
                                     f["img_shape"][None].cuda(), f["ego_history"][None].cuda(), training=False, replay=False)
            out[t["token"]] = {"proposals": o["proposals"][0].cpu().numpy(), "pred_logit": o["pred_logit"][0].cpu().numpy(),
                               "pdm_score": o["pdm_score"][0].cpu().numpy(), "chosen": int(o["chosen"][0]),
                               "human": t["trajectory"].numpy(), "mc": t["metric_cache_path"], "lab": bool(t["has_label2"])}
    return out


def run_check(model, ds_pk, pk, groups, grader, fbs, shares: bool = True):
    import train_018b as TR
    import pai_train as T
    t0 = time.time()
    dump = forward_dumps(model, ds_pk)
    t1 = time.time()
    toks = list(pk.sample_id)
    gi = np.where(pk.has_label2.to_numpy())[0]
    assert all(dump[toks[i]]["lab"] and dump[toks[i]]["mc"] for i in gi)
    grades = np.stack(grader.grade_menus([dump[toks[i]]["mc"] for i in gi], [dump[toks[i]]["proposals"] for i in gi]))
    t2 = time.time()
    c = {"summary": T.summary(T.per_moment(dump), groups), "safety": TR.safety(dump, pk, grades), "logits": T.logit_stats(dump)}
    c["shares"] = lesson_shares(model, fbs, grader) if shares else None
    c["time_s"] = {"forward": round(t1 - t0, 1), "grading": round(t2 - t1, 1), "shares": round(time.time() - t2, 1)}
    return c, {"dump": dump, "grades": grades}


def save_ckpt(model, path: Path, meta: dict) -> None:
    """weights and their record written together (Gallops rule: the label is written by the function that saves the weights)."""
    import torch
    import s6_model as SM
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"model": "approach 4 (motion history)", "arm": model.arm, "hide_prob": model.hide_prob, "seed": model.seed,
           "start": {"checkpoint": str(SM.P06), "sha256": SM.P06_SHA}, "prereg_sha256_after_amendment_1":
           "f8f4b975edaf712fcc02342c151e21303337d1dfa0880f64385e0d9bbb0ba450", "saved": time.strftime("%F %T"), **meta}
    torch.save({**model.trainable_state(), **rec}, path)
    path.with_suffix(".json").write_text(json.dumps(rec, indent=1, default=str) + "\n")


def setup(arm: str):
    import torch
    import pai_common as C
    C.setup_env()
    import pai_train as T
    import pai_model_b as MB
    import a4_data as D
    import model_a4 as MA
    s6_env.patch_grader(MB)
    import det_grid_sample
    det_grid_sample.install()          # repeatable grid_sample backward (forward untouched); see det_grid_sample.py
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    tr, pk, _ = T.manifests()
    agent, info = MA.load_agent()
    model = MA.A4Model(agent, arm, SEED)
    info = {"load": info, "load_from_p06": model.load_from_p06()}
    agent.read_scores_index = T.parallel_index_reader(agent)
    ds = D.training_set(tr)
    return tr, pk, model, ds, info


def dry_run(arm: str) -> dict:
    """3 steps of pass 1 on the GPU: losses finite, no encoder gradient; s/step and peak memory."""
    import torch
    import pai_common as C
    import pai_data as PD
    import pai_model_b as MB
    import a4_data as D
    gate = C.memory_gate(True)
    tr, pk, model, ds, info = setup(arm)
    grader = MB._GRADER = MB.Grader(POOL)
    opts = model.optimisers()
    bl = PD.batches_017(PD.sampler_017(tr, SEED, 1), BATCH)
    np.random.seed(SEED + 1)
    it = D.Prefetch(ds, bl[:3])
    torch.cuda.reset_peak_memory_stats()
    before = {n: p.detach().clone() for n, p in model.new_params}
    recs = []
    for b in it:
        torch.cuda.synchronize(); t0 = time.time()
        q = train_step(model, opts, b, grader)
        torch.cuda.synchronize(); q["step_s"] = time.time() - t0; q["loader_wait_s"] = it.last_wait_s
        recs.append(q)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items()}), flush=True)
    moved = {n: bool(not torch.equal(before[n], p.detach())) for n, p in model.new_params}
    steps_pass = len(bl)
    rj = json.loads(Path("{retrain_store}/018b/runs/018b_attempt2/run.json").read_text())
    res = {"arm": arm, "memory_gate": gate, "info": info, "steps": recs, "all_finite": all(r["finite"] for r in recs),
           "encoder_grads": sum(r.get("encoder_grads", 0) for r in recs), "encoder_requires_grad": recs[-1].get("encoder_requires_grad"),
           "new_params_changed": moved, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
           "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
           "s_per_step_steps_2_3": float(np.mean([r["step_s"] for r in recs[1:]])), "s_per_step_first": recs[0]["step_s"],
           "grade_wait_s_steps_2_3": float(np.mean([r["grade_wait_s"] for r in recs[1:]])),
           "steps_per_pass": steps_pass, "max_steps_3_passes": 3 * steps_pass,
           "018b_attempt2_steps_run": len(rj["losses"]["loss"]), "018b_attempt2_checks": len(rj["checks"]),
           "018b_check_time_s_p06": next(c["time_s"] for c in rj["checks"] if c["pass"] == 0.6)}
    grader.close()
    return res


SPAN_STEPS = 5580          # 0.1.8b attempt 2's span: run.json checks at steps 0, 930, ..., 5580 = passes 0.0, 0.2, ..., 1.2 (TRAIN brief)
SAVE_EVERY_S = 1800        # a timed save every 30 min between check saves
STATUS = None              # set in train(): runs/STATUS.txt


def rng_state() -> dict:
    import torch
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(), "numpy": np.random.get_state(),
            "python": random.getstate()}


def set_rng_state(s: dict) -> None:
    import torch
    torch.set_rng_state(s["torch"]); torch.cuda.set_rng_state_all(s["cuda"]); np.random.set_state(s["numpy"]); random.setstate(s["python"])


def atomic_save(obj, path: Path) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp); tmp.replace(path)


def train(arm: str, out: Path, resume: Path | None = None, max_gsteps: int = SPAN_STEPS, checks: bool = True,
          exit_after: int | None = None) -> dict:
    """The run. Everything a later step depends on is saved together (weights, the three Adam states, position, every RNG including the
    hiding generator, the run record and the selection counters) at every check and every 30 min; --resume restores all of it.
    The sampler needs no state: its batch list is a pure function of (seed, pass) (pai_data.sampler_017), so the position is enough.
    No learning-rate scheduler exists in 0.1.8b's recipe (constant Adam rates), so there is none to save.
    Span: fixed at 0.1.8b attempt 2's 1.2 passes (TRAIN brief). 0.1.8b's stopping rules are evaluated at every check and recorded
    (prereg_stop_at) but do not end the run; selection only considers checks up to where they would have stopped it, as under 0.1.8b.
    exit_after / checks=False are for the resume test only."""
    import torch
    import pai_common as C
    import pai_data as PD
    import pai_train as T
    import pai_model_b as MB
    import a4_data as D
    import a4_config as CFG
    import train_018b  # noqa: F401  (its import sets C.LOGS to 0.1.8b's NAS folder; import it first so ours wins)
    global STATUS
    STATUS = CFG.RUNS / "STATUS.txt" if checks else None
    C.LOGS = out / "logs"
    C.open_log(f"train_a4_{arm}")
    gate = C.memory_gate(True)
    stop_file = out / "STOP"
    if resume is None:
        assert not stop_file.exists() and not (out / "run.json").exists() and not (out / "ckpt").exists(), f"{out} not empty: archive it first"
    tr, pk, model, ds, info = setup(arm)
    grader = MB._GRADER = MB.Grader(POOL)
    if checks:
        groups = T.groups_of(pk); ds_pk = D.picking_set(pk); fbs = fixed_batches(ds, N_SHARE_BATCHES)
    opts = model.optimisers()
    n_pass = len(PD.batches_017(PD.sampler_017(tr, SEED, 1), BATCH))

    if resume is not None:
        S = torch.load(resume, map_location="cpu", weights_only=False)
        rec, cnt, pos = S["rec"], S["counters"], S["pos"]
        assert rec["settings"]["arm"] == arm, (rec["settings"]["arm"], arm)
        model.load_trainable(S)
        for o, sd in zip(opts, S["optimisers"]):
            o.load_state_dict(sd)
        rec.setdefault("resumes", []).append({"at": time.strftime("%F %T"), "from": str(resume), "pos": pos, "memory_gate": gate})
        C.log(f"resumed from {resume} at pass {pos['pass']} step {pos['step']} (global {pos['gstep']})")
        set_rng_state(S["rng"])                                             # last, after everything that might draw
    else:
        rec = {"run": f"approach 4 arm {arm}", "settings": {"arm": arm, "hide_prob": model.hide_prob, "w_human": model.w_human,
               "w_pull_away": model.w_pull, "hard_rule": True, "hide_and_guess": False, "gridmask": False, "go_rate_gate_pull_away": GO_PULL_MIN,
               "span_steps": max_gsteps, "check_every": CHECK_EVERY, "lr": {"generator": 3e-5, "scorer": 1e-4, "new": 1e-4}},
               "start": {"memory_gate_start": gate, "started": time.strftime("%F %T")}, "start_info": info, "checks": [],
               "losses": {k: [] for k in ("loss",) + LESSONS}, "zeroed_targets": [], "grade_wait_s": [], "pull_rows": [],
               "pull_map_rows": [], "map_rows": [], "hidden_rows": [], "step_s": [], "prereg_stop_at": None, "prereg_stop_reason": None}
        cnt = {"best": None, "since": 0, "fails": 0, "ref_hard": None}
        pos = {"pass": 1, "step": 0, "gstep": 0}
    H = rec["losses"]
    gstep = pos["gstep"]

    def state(p, step):
        nxt = {"pass": p + 1, "step": 0} if step == n_pass else {"pass": p, "step": step}
        return {**model.trainable_state(), "optimisers": [o.state_dict() for o in opts], "pos": {**nxt, "gstep": gstep},
                "rng": rng_state(), "rec": rec, "counters": dict(cnt)}

    def status(p, step, last=None):
        if STATUS is None:
            return
        c = last or (rec["checks"][-1] if rec["checks"] else None)
        sf = c["safety"] if c else {}
        sps = float(np.mean(rec["step_s"][-100:])) if len(rec["step_s"]) > 1 else 2.5       # before any step: the resume test's ~2.5 s
        left = max_gsteps - gstep
        n_chk = sum(1 for s in range(CHECK_EVERY, max_gsteps + 1, CHECK_EVERY) if s > gstep)
        ct = [sum(x["time_s"].values()) for x in rec["checks"]]
        eta = time.time() + left * sps + n_chk * (float(np.mean(ct)) if ct else 500.0)
        line = (f"{time.strftime('%F %T')} arm {arm} | step {gstep}/{max_gsteps} | pass {gstep / n_pass:.2f} | {sps:.2f} s/step | "
                f"last check p{c['pass'] if c else '-'}: go_rate_pull_away {sf.get('go_rate_pull_away')} waits_correctly "
                f"{sf.get('waits_correctly')} pick_grade_mean {sf.get('pick_grade_mean')} hard_picks_above10 {sf.get('hard_picks_above10')} | "
                f"ETA arm {arm} {time.strftime('%F %H:%M', time.localtime(eta))}")
        try:
            STATUS.write_text(line + "\n")
        except OSError as e:                                                   # a status line must never stop a run
            C.log(f"STATUS.txt not written: {e}")

    def after_check(c, progress):
        sf = c["safety"]
        c["gate_hard_ok"] = sf["hard_picks_above10"] <= cnt["ref_hard"]
        c["gate_go_ok"] = sf["go_rate_pull_away"] is not None and sf["go_rate_pull_away"] >= GO_PULL_MIN
        c["gate_ok"] = c["gate_hard_ok"] and c["gate_go_ok"]
        cnt["fails"] = 0 if c["gate_hard_ok"] else cnt["fails"] + 1
        if c["gate_ok"] and sf["pick_grade_mean"] > cnt["best"]:
            cnt["best"], cnt["since"] = sf["pick_grade_mean"], 0
        else:
            cnt["since"] += 1
        why = None
        if cnt["fails"] >= GATE_FAILS:
            why = f"hard-brake gate failed at {GATE_FAILS} checks in a row (pass {progress:.1f})"
        elif cnt["since"] >= NO_IMPROVE:
            why = f"no check passing both gates with a better pick grade for {NO_IMPROVE} checks (pass {progress:.1f})"
        if why and rec["prereg_stop_at"] is None:
            rec["prereg_stop_at"], rec["prereg_stop_reason"] = progress, why
        c["train_recent"] = {k: float(np.mean(v[-50:])) if v else None for k, v in H.items()}
        C.log(f"pass {progress:.1f}: pick grade {sf['pick_grade_mean']:.4f} hard {sf['hard_picks_above10']}/{cnt['ref_hard']} "
              f"go(pulled away) {sf['go_rate_pull_away']} waits {sf['waits_correctly']} gates hard {c['gate_hard_ok']} go {c['gate_go_ok']} "
              f"best {cnt['best']:.4f} since {cnt['since']}" + (f" | 0.1.8b's rule would stop here: {why}" if why else ""))

    def check(progress, p, step):
        torch.cuda.reset_peak_memory_stats()
        c, extra = run_check(model, ds_pk, pk, groups, grader, fbs)
        c["gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30                # logging only (the 2nd arm-B crash was an OOM)
        C.log(f"check {progress:.1f} gpu peak reserved {c['gpu_peak_reserved_gb']:.2f} GB")
        T.save_dump(out / "dumps" / f"p{progress:04.1f}.npz", extra["dump"])
        np.save(out / "dumps" / f"p{progress:04.1f}_grades.npy", extra["grades"])
        c.update({"pass": progress, "step": gstep})
        rec["checks"].append(c)
        if progress == 0.0:
            cnt["ref_hard"] = c["safety"]["hard_picks_above10"]; cnt["best"] = c["safety"]["pick_grade_mean"]
            c["gate_hard_ok"], c["gate_go_ok"], c["gate_ok"] = True, c["safety"]["go_rate_pull_away"] >= GO_PULL_MIN, True
            rec["reference_step0"] = {"hard_picks_above10": cnt["ref_hard"], "pick_grade_mean": cnt["best"]}
        else:
            after_check(c, progress)
        # weights + record together (Gallops rule), and the full resume state in the same file
        path = out / "ckpt" / f"p{progress:04.1f}.pt"
        meta = {"progress_passes": progress, "step": gstep}
        atomic_save({**state(p, step), **ckpt_record(model, meta)}, path)
        path.with_suffix(".json").write_text(json.dumps(ckpt_record(model, meta), indent=1, default=str) + "\n")
        C.wj(out / "run.json", rec)
        status(p, step, c)
        return c

    (out / "dumps").mkdir(parents=True, exist_ok=True)
    if resume is None and checks:
        check(0.0, 1, 0)
    last_save = time.time()
    stop = None
    p, start = pos["pass"], pos["step"]
    while gstep < max_gsteps and p <= MAX_PASSES and stop is None:
        bl = PD.batches_017(PD.sampler_017(tr, SEED, p), BATCH)
        assert len(bl) == n_pass
        if start == 0:
            np.random.seed(SEED + p)                                          # train_018b.py:1325, at a pass start only
        it = D.Prefetch(ds, bl[start:])
        for step, b in enumerate(it, start=start + 1):
            ts = time.time()
            q = train_step(model, opts, b, grader)
            if not q["finite"]:
                stop = f"non-finite loss at pass {p} step {step}"; break
            for k in H:
                H[k].append(q[k])
            for k in ("zeroed_targets", "pull_rows", "pull_map_rows", "map_rows", "hidden_rows"):
                rec[k].append(q[k])
            rec["grade_wait_s"].append(round(q["grade_wait_s"], 3))
            gstep += 1
            rec["step_s"].append(round(time.time() - ts, 3))
            if step % 25 == 0:
                C.log(f"pass {p} step {step}/{len(bl)} (global {gstep}) " + " ".join(f"{k} {np.mean(H[k][-25:]):.4f}" for k in H)
                      + f" hidden {np.mean(rec['hidden_rows'][-25:]):.1f}/batch wait {np.mean(rec['grade_wait_s'][-25:]):.2f}s "
                      f"step {np.mean(rec['step_s'][-25:]):.2f}s loader {it.last_wait_s:.2f}s ram {C.avail_ram_gb():.0f} GB "
                      f"gpu peak {torch.cuda.max_memory_reserved() / 2**30:.1f} GB")
                status(p, step)
                if stop_file.exists():
                    stop = f"STOP file: {stop_file.read_text().strip()}"; break
                if C.avail_ram_gb() < 20:
                    stop = f"available RAM {C.avail_ram_gb():.1f} GB < 20"; break
            if checks and (step % CHECK_EVERY == 0 or step == len(bl)):
                prog = round(p - 1 + step / len(bl), 3) if step < len(bl) else float(p)
                check(prog, p, step); last_save = time.time()
            elif time.time() - last_save > SAVE_EVERY_S:
                atomic_save(state(p, step), out / "resume" / "last.pt"); last_save = time.time()
                C.log(f"timed save at global step {gstep}")
            if exit_after is not None and gstep == exit_after:
                atomic_save(state(p, step), out / "resume" / "last.pt")
                C.log(f"exit_after: saved at global step {gstep}")
                grader.close()
                return rec
            if gstep >= max_gsteps:
                break
        p, start = p + 1, 0
    rec["stopped"] = stop or f"span of {max_gsteps} steps completed"
    if checks:
        lim = rec["prereg_stop_at"]
        good = [c for c in rec["checks"] if c["pass"] > 0 and c["gate_ok"] and (lim is None or c["pass"] <= lim)]
        ch = max(good, key=lambda c: (c["safety"]["pick_grade_mean"], -c["pass"])) if good else None
        rec["chosen"] = ch["pass"] if ch else None
        rec["chosen_beats_step0_pick_grade"] = bool(ch and ch["safety"]["pick_grade_mean"] > rec["reference_step0"]["pick_grade_mean"])
    rec["ended"] = time.strftime("%F %T")
    if checks:
        C.wj(out / "run.json", rec)
        status(p, 0)
    else:
        atomic_save(state(p, 1), out / "final.pt")
    C.log(f"stopped: {rec['stopped']}; chosen {rec.get('chosen')}")
    grader.close()
    if stop and not stop.startswith("STOP file"):
        raise RuntimeError(stop)                                              # a crash for the queue (resumed once from the last save)
    return rec


def ckpt_record(model, meta: dict) -> dict:
    import s6_model as SM
    return {"model": "approach 4 (motion history)", "arm": model.arm, "hide_prob": model.hide_prob, "seed": model.seed,
            "start": {"checkpoint": str(SM.P06), "sha256": SM.P06_SHA}, "prereg_sha256_after_amendment_1":
            "f8f4b975edaf712fcc02342c151e21303337d1dfa0880f64385e0d9bbb0ba450", "saved": time.strftime("%F %T"), **meta}




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--go-from-sid", action="store_true")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--out", type=Path, help="run folder (default runs/arm_<arm>)")
    ap.add_argument("--test-steps", type=int, help="resume test: run this many global steps, no checks, save final.pt")
    ap.add_argument("--test-exit-after", type=int, help="resume test: save resume/last.pt after this global step and exit")
    a = ap.parse_args()
    rc = 0
    try:
        if a.dry_run:
            r = dry_run(a.arm)
            o = Path(__file__).resolve().parents[1] / "out" / f"dry_run_{a.arm}.json"
            o.write_text(json.dumps(r, indent=1, default=str))
            print(json.dumps({k: r[k] for k in ("arm", "all_finite", "encoder_grads", "s_per_step_steps_2_3", "peak_gpu_gb", "steps_per_pass")}))
        elif a.test_steps:
            import a4_config as CFG
            train(a.arm, a.out, a.resume, max_gsteps=a.test_steps, checks=False, exit_after=a.test_exit_after)
        else:
            if not a.go_from_sid:
                raise SystemExit("training needs --go-from-sid (Sid's go)")
            import a4_config as CFG
            train(a.arm, a.out or CFG.RUNS / f"arm_{a.arm}", a.resume)
    except BaseException:
        import traceback
        traceback.print_exc()
        rc = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
