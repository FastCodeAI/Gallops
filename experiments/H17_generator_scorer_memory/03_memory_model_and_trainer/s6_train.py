"""Build step 3: the in-order trainer and the in-order check for the memory model (s6_model.py) on the build-2 cache (s6_data.py).

One step = 2 windows x 16 moments (32 moments). Moment by moment in time order: read M -> generator -> scorer -> pick -> update M;
backprop through all 16 moments; M is zeros (detached) at the window start. Moments 0-3 warm up (no loss), 4-15 are scored; every lesson
is averaged over the scored moments (the 24 rows of a step), exactly the weighted means of 0.1.8b attempt 2.

Lessons, 0.1.8b attempt 2 unchanged, reused by import (settings as NAS/018b/runs/018b_attempt2/run.json: w_human 0.5, w_pull_away 2.0):
  generator      drive_jepa_agent_s10.generator_loss_s10 (:384-424): winner-takes-all L1 + 0.5 x vocabulary term, prev_weight 0.1
  grade, collision x1, drivable x2 (map moments), human closeness x0.5 (all moments), pull-away x2, hard-brake rule:
                 pai_model_c.Model018c.scorer_lessons (:44-87), with pai_model_b.hard_brake (:168-173) and the online pai_model_b.Grader
                 (:201-258) grading the generator's own 32 plans of every scored map moment, submitted moment by moment while the window
                 runs and collected after the forward (the hide-and-guess lesson of 0.1.8b is gone with the frozen encoder).
Replay routing (pai_model.py:194-201, s8 REPORT G2): the scorer lessons read the rating / collision / drivable heads on a replay bev
built from the same ego/query init and the same four proposal lists with the refiner's weights detached. HOW M ENTERS THE REPLAY: the
replay's generator read runs with its weights detached AND M detached, so through the replay bev the scorer lessons reach nothing
trainable; they reach M (and so the update block) only through the scorer's own summary, which is attached. Hence grade / human ->
rating head + summary + update block, never the generator or its look-up (as 0.1.8b's scorer lessons never reached the generator);
collision / drivable -> their heads only. The alternative (M attached through the detached look-up in the replay) would also let the
scorer lessons shape what the GENERATOR reads from memory; it is listed as OPEN in the report.
Optimisers: generator Adam 3e-5, scorer Adam 1e-4 (pai_model.py:233-235), memory blocks Adam 1e-4; bf16 autocast as 0.1.8b
(pai_model.py:210); seed recorded. Grading pool = 10 spawned processes on the launcher's 2 pinned cores (run_model.sh: taskset -c 0-1).
Check (every 930 steps, 0.1.8b's cadence, train_018b.py:65): the 84 picking clips REPLAYED IN ORDER from each clip's first moment in the
run's mode (fp32, batch 1, the check forward), measured on the 2,000 in_picking moments: pick grade (map moments, Grader.grade_menus
on the moment's menu as train_018b.run_check :192-216), hard-brake picks above 10 m/s, pull-away go rate (train_018b.go_split :137-152,
safety :155-189), closest-of-32 ADE (3 s) (pai_train.per_moment m1a :117-129), pick-follows-previous rate (defined in follows_previous).
Stop and choice rules (train_018b.py:63, 1301-1327; the no-improvement stop and the choice): 6 checks without a better choosable pick,
at most 3 passes; chosen = best pick grade among checks passing both safety gates (hard-brake picks: Sid's rule "must be 0"; pull-away go
rate >= 0.66). Nothing here is run as a training run in build 3: only the tests in s6_tests.py call these functions."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import s6_env as SE
import pai_common as C
import pai_model as PM, pai_model_b as MB
SE.patch_grader(MB)
G = Path("{working_notes}/LORA Refiner/Generator-Joint-PAI-alpasim")
for p in (G / "s9_018b_train/src", G / "s2_train/src"):         # in FRONT of s1_build/src (run_model.sh's PYTHONPATH has s2 behind s1,
    while str(p) in sys.path:                                       # and s1 has an older pai_train without save_dump): s2's pai_train is the
        sys.path.remove(str(p))                                     # one train_018b.py uses
    sys.path.insert(0, str(p))
import pai_model_c as MC, pai_train as T, train_018b as TB
import s6_model as SM, s6_data as SD

OUT = Path(os.environ.get("OUT", str(Path(__file__).resolve().parents[1] / "out"))); RUNS = SD.A3 / "runs"
CHECK_EVERY, MAX_PASSES, NO_IMPROVE, GO_PULL_MIN, POOL = 930, 3, 6, 0.66, 10          # train_018b.py:63-65, 71
HARD_GATE_RULE = "zero"                                                               # Sid (build 3 brief): hard-brake picks above 10 m/s must be 0
ADE_TOL = 0.02                                                                        # closest-of-32 ADE (3 s) within +0.02 m of 0.1.8b's
LESSONS = ("generator", "grade", "human", "collision", "drivable")
W_HUMAN, W_PULL = MC.W_HUMAN, MC.W_PULL                                               # 0.5, 2.0 (pai_model_c.py:24-25)
W_RECALL = 2.0                                                                        # build 5 (Sid): the recall lesson's weight, fixed
RECALL_ALLOWED = ("update_block", "start_state", "read_out")                          # the only groups the recall lesson may reach
_ROUTING = {"checked": False}


def lessons_for(version: int) -> tuple:
    return LESSONS + (("recall",) if version >= 3 else ())


def recall_lesson(model, outs, first: int = SD.WARM) -> dict:
    """build 5, the memory's own lesson (version 3). At every scored moment t whose target lies inside the window, the read-out predicts,
    from o["M_new"] (the memory written at t; NEVER a tensor that passed through the generator read or the scorer summary), the standardised
    32 block means of t - 4 (2 s ago) and t - 8 (4 s ago); smooth-L1 (beta 1), mean over the predictions (12 at 2 s + 8 at 4 s per window).
    Targets and the read-out's "now" input are block means of cached tokens (constants). On the first training call of a process the routing
    is asserted with autograd: no gradient to any group outside RECALL_ALLOWED."""
    ro = model.read_out; B = outs[0]["M_new"].shape[0]
    Z = [ro.standardise(o["block_means"]) for o in outs]; assert not any(z.requires_grad for z in Z)
    Ms, now, tgt, lag = [], [], [], []
    for t in range(first, len(outs)):
        for li, L in enumerate(SM.RECALL_LAGS):
            if t - L >= 0:
                Ms.append(outs[t]["M_new"].float()); now.append(Z[t]); tgt.append(Z[t - L]); lag.append(torch.full((B,), li, device=Z[t].device, dtype=torch.long))
    lag = torch.cat(lag); pred = ro(torch.cat(Ms), torch.cat(now), lag); tgt = torch.cat(tgt)
    per = torch.nn.functional.smooth_l1_loss(pred, tgt, reduction="none", beta=1.0).mean((1, 2))
    zero = float(torch.nn.functional.smooth_l1_loss(torch.zeros_like(tgt), tgt, beta=1.0))      # reference only: "predict the average" on the same targets (batches differ)
    out = {"loss": per.mean(), "lag_2s": per[lag == 0].mean(), "lag_4s": per[lag == 1].mean(), "zero_reference": zero, "n": int(per.numel()), "n_2s": int((lag == 0).sum()), "n_4s": int((lag == 1).sum())}
    if not _ROUTING["checked"] and out["loss"].requires_grad:
        banned = [p for g, ps in model.groups.items() if g not in RECALL_ALLOWED for p in ps if not any(p is q for q in ro.parameters())]
        gr = torch.autograd.grad(out["loss"], banned, retain_graph=True, allow_unused=True)
        assert all(x is None for x in gr), "the recall lesson reaches a parameter outside the read-out, the update block and M0"
        _ROUTING["checked"] = True; _ROUTING["banned_tensors_checked"] = len(banned)
    return out


class _Lessons:
    """what pai_model_c.Model018c.scorer_lessons reads from `self`: the two weights and the config (pai_model_c.py:47-48, 83)."""
    w_human, w_pull = W_HUMAN, W_PULL

    def __init__(self, cfg):
        self.cfg = cfg


def scorer_lessons(model, r, targets, graded):
    return MC.Model018c.scorer_lessons(_Lessons(model.cfg), r, targets, graded)


def merge_graded(jobs: list, rows_per_moment: int) -> dict | None:
    """Grader.collect of the per-moment jobs -> one graded dict in the stacked order (row = (t - WARM) * B + b), as scorer_lessons expects."""
    parts = [(k, MB.Grader.collect(j)) for k, j in jobs]
    parts = [(k, g) for k, g in parts if g is not None]
    if not parts:
        return None
    rows = [k * rows_per_moment + rr for k, g in parts for rr in g["rows"]]
    return {"rows": rows, **{key: torch.cat([g[key] for _, g in parts]) for key in ("target_scores", "gt_states", "gt_valid", "gt_ego_areas")}}


def motion_for_window(traj):
    """(B, 16, 8, 3) human futures -> (B, 16, 3) car motion since the previous moment in the current frame; zeros at the first moment."""
    m = torch.zeros(traj.shape[0], traj.shape[1], 3, device=traj.device)
    m[:, 1:] = torch.stack([SM.motion_from_prev_future(traj[:, t - 1]) for t in range(1, traj.shape[1])], 1)
    return m


def window_step(model, batch, grader, index_reader, opts=None, lessons_alone: bool = False) -> dict:
    """one step on `batch` (s6_data.window_tensors). opts None = no backward. Returns the record (and the loss tensors if lessons_alone)."""
    B, L = batch["ego"].shape[:2]; dev = batch["ego"].device
    motion = motion_for_window(batch["traj"])
    M = model.init_memory(B, dev); outs = []; jobs = []
    torch.cuda.synchronize(); t0 = time.time(); t_first_submit = None
    with torch.autocast("cuda", dtype=torch.bfloat16):                                                   # pai_model.py:210
        for t in range(L):
            if model.mode == "wiped":
                M = model.init_memory(B, dev)
            o = model.forward_moment(batch["tokens"][:, t], batch["ego"][:, t], batch["lidar2img"], batch["img_shape"], M, motion[:, t], replay=t >= SD.WARM)
            if t >= SD.WARM:
                tg = {"has_label2": batch["has_label2"][:, t], "metric_cache_path": [batch["metric_cache_path"][b][t] for b in range(B)]}
                job = grader.submit(tg, o["proposals"])
                if job is not None:
                    jobs.append((t - SD.WARM, job)); t_first_submit = t_first_submit or time.time()
            outs.append(o); M = model.step_memory(o["M_new"])
    torch.cuda.synchronize(); t1 = time.time()
    S = outs[SD.WARM:]                                                                                    # the scored moments
    cat = lambda key: torch.cat([o[key] for o in S])                                                       # row = (t - WARM) * B + b
    r = {"proposals": cat("proposals"), "proposal_list": [torch.cat([o["proposal_list"][i] for o in S]).float() for i in range(4)],
         "logit": cat("logit"), "agents_state": cat("agents_state"), "area_logit": cat("area_logit"),
         "ego": torch.cat([batch["ego"][:, t] for t in range(SD.WARM, L)])}
    target = torch.cat([batch["traj"][:, t] for t in range(SD.WARM, L)])
    targets = {"trajectory": target, "has_label2": torch.cat([batch["has_label2"][:, t] for t in range(SD.WARM, L)]),
               "metric_cache_path": [batch["metric_cache_path"][b][t] for t in range(SD.WARM, L) for b in range(B)],
               "token": [batch["sample_id"][b][t] for t in range(SD.WARM, L) for b in range(B)]}
    ti = time.time()
    if "scores_index" in batch:                                                   # read with the clip in the prefetch thread (s6_data.ClipStore._read)
        scores_index = [batch["scores_index"][b][t] for t in range(SD.WARM, L) for b in range(B)]
        for row, a in enumerate(scores_index):
            assert a.size == 0 or a.max() < len(model.agent.anchors), "anchor index beyond the vocabulary"       # read_scores_index :369
            assert (a.size > 0) == bool(targets["has_label2"][row]) or a.size == 0, row
    else:
        scores_index = index_reader(targets)
    t_index = time.time() - ti
    gl, parts = model.agent.generator_loss_s10(r["proposal_list"], target, model.cfg, scores_index, 0.0)   # drive_jepa_agent_s10.py:384-424
    sl, d, tgt, n_wp = PM.scorer_loss(r["logit"], r["proposals"], target)                                 # pai_model.py:252-259
    r.update(score=sl, d=d, target=tgt)
    tw = time.time(); graded = merge_graded(jobs, B); wait = time.time() - tw
    Ls = scorer_lessons(model, r, targets, graded)
    losses = {"generator": gl, "grade": Ls["grade"], "human": Ls["human"], "collision": Ls["collision"], "drivable": Ls["drivable"]}
    t_rc = time.time(); rc = None
    if model.version >= 3:
        rc = recall_lesson(model, outs); losses["recall"] = W_RECALL * rc["loss"]
        torch.cuda.synchronize()
    recall_s = time.time() - t_rc
    loss = sum(losses.values())
    rec = {"recall": 0.0, "recall_raw": float(rc["loss"]) if rc else None, "recall_2s": float(rc["lag_2s"]) if rc else None, "recall_4s": float(rc["lag_4s"]) if rc else None,
           "recall_n": rc["n"] if rc else 0, "recall_zero_reference": rc["zero_reference"] if rc else None, "recall_forward_s": round(recall_s, 4),
           "loss": float(loss), **{k: float(v) for k, v in losses.items()}, "map_rows": Ls["map_rows"], "zeroed_targets": Ls["zeroed_targets"],
           "pull_rows": Ls.get("pull_rows", 0), "grade_wait_s": round(wait, 3), "forward_s": round(t1 - t0, 3), "index_s": round(t_index, 3), "finite": bool(torch.isfinite(loss)),
           "gate_mean_per_slot": torch.stack([o["gate"].float().mean(0) for o in outs]).mean(0).detach().cpu().tolist(),
           "M_norm_last": float(outs[-1]["M_new"].float().norm(dim=(1, 2)).mean()),
           "M_read_max_abs": float(max(o["M_read"].abs().max() for o in outs)),
           "M_after_first_moment_norm": float(outs[0]["M_new"].float().norm(dim=(1, 2)).mean()),
           # build 4 (version 2): the memory read vs the START state (M0; zeros in version 1) and the memory written at the first moment vs it
           "M_read_max_dev_from_start": float(max((o["M_read"].float() - model.init_memory(B, dev).detach()).abs().max() for o in outs)),
           "M_after_first_moment_dev_from_start_norm": float((outs[0]["M_new"].float() - model.init_memory(B, dev).detach()).norm(dim=(1, 2)).mean()),
           "grade_rows_submitted": sum(len(j["rows"]) for _, j in jobs)}
    if lessons_alone:
        rec["_losses"] = losses; rec["_outs"] = outs
        return rec
    tb = time.time()
    if opts is not None and rec["finite"]:
        loss.backward()
        for o in opts:
            o.step(); o.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); rec["backward_s"] = round(time.time() - tb, 3); rec["step_s"] = round(time.time() - t0, 3)
    for p in model.parameters():
        p.grad = None
    return rec


def routing_grads(model, batch, grader, index_reader) -> dict:
    """gradient norms per lesson alone (and of generator + grade + human together) on the update block, its gate, M0 and the read-out, on one
    batch, without changing any weight (no optimiser step; gradients cleared). The caller sets train / eval mode."""
    q = window_step(model, batch, grader, index_reader, None, lessons_alone=True); L = q["_losses"]
    sets = {"update_block": model.groups["update_block"], "start_state": model.groups.get("start_state", []), "read_out": model.groups.get("read_out", []),
            "gate": [model.update.gate.weight] + ([model.update.gate_bias] if model.version >= 2 else [model.update.gate.bias])}
    norm = lambda ps: float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in ps if p.grad is not None) + 0.0)) if any(p.grad is not None for p in ps) else 0.0
    def clear():
        for p in model.parameters():
            p.grad = None
    out = {"map_rows": q["map_rows"], "values": {k: float(v) for k, v in L.items()}, "per_lesson": {}}; t_b = {}
    combos = {**{k: [k] for k in L}, "driving_generator_grade_human": ["generator", "grade", "human"]}
    for name, ks in combos.items():
        tot = sum(L[k] for k in ks if torch.is_tensor(L[k]) and L[k].requires_grad)
        clear()
        if not torch.is_tensor(tot) or not tot.requires_grad:
            out["per_lesson"][name] = None; continue
        torch.cuda.synchronize(); t0 = time.time(); tot.backward(retain_graph=True); torch.cuda.synchronize(); t_b[name] = round(time.time() - t0, 4)
        out["per_lesson"][name] = {g: norm(ps) for g, ps in sets.items()}
    clear(); out["backward_s"] = t_b
    if out["per_lesson"].get("recall") and out["per_lesson"].get("driving_generator_grade_human"):
        r, d = out["per_lesson"]["recall"], out["per_lesson"]["driving_generator_grade_human"]
        out["ratio_recall_over_driving"] = {g: (r[g] / d[g] if d[g] > 0 else None) for g in ("update_block", "gate", "start_state")}
    return out


# ----------------------------------------------------------------------------------------------------------------------- #
# The in-order check
# ----------------------------------------------------------------------------------------------------------------------- #


def picking_frame() -> pd.DataFrame:
    """the frozen picking set (2,000 moments; sha checked as pai_train.manifests does, :69-72)."""
    _, pk, _ = T.manifests()
    return pk


@torch.no_grad()
def replay_slice(model, store: SD.ClipStore, clips=None, progress=None) -> dict:
    """every moment of each clip in order from its first moment, fp32, batch 1, the run's mode. Returns {token: dump} for ALL moments
    (proposals, pred_logit, chosen, human, mc, lab, clip, row, prev token) so the check can also compute pick-follows-previous."""
    model.eval(); dump = {}
    for ci, c in enumerate(clips or store.clips):
        rec = store.load(c); N = rec["tokens"].shape[0]; dev = "cuda"
        tokens = rec["tokens"].to(dev); ego = rec["ego"].to(dev); traj = rec["traj"].to(dev)
        l2i = rec["lidar2img"][None].to(dev); shp = rec["img_shape"][None].to(dev)
        M = model.init_memory(1, dev); prev = None
        for i in range(N):
            if model.mode == "wiped":
                M = model.init_memory(1, dev)
            consecutive = i > 0 and abs(rec["anchor_t"][i] - rec["anchor_t"][i - 1] - SD.STEP_S) < 1e-6
            if not consecutive:
                M = model.init_memory(1, dev); prev = None                                                 # a missing anchor starts a new chain
            motion = SM.motion_from_prev_future(traj[i - 1:i]) if consecutive else torch.zeros(1, 3, device=dev)
            o = model.forward_moment(tokens[i:i + 1], ego[i:i + 1], l2i, shp, M, motion, replay=False)
            tok = rec["sample_id"][i]
            dump[tok] = {"proposals": o["proposals"][0].float().cpu().numpy(), "pred_logit": o["pred_logit"][0].float().cpu().numpy(),
                         "chosen": int(o["chosen"][0]), "human": rec["traj"][i].numpy(), "mc": rec["metric_cache_path"][i], "lab": bool(rec["has_label2"][i]),
                         "clip": c, "row": i, "prev": prev, "in_picking": bool(rec["in_picking"][i]), "M_read_max_abs": float(M.abs().max())}
            M = model.step_memory(o["M_new"]); prev = tok
        if progress and (ci + 1) % 20 == 0:
            progress(f"replayed {ci + 1}/{len(clips or store.clips)} clips, {len(dump)} moments")
    return dump


def follows_previous(dump: dict, toks: list) -> dict:
    """pick-follows-previous rate over the given moments that have a consecutive previous moment in the clip: the previous pick (8 x 3,
    previous frame) moved into the current frame with the previous moment's logged motion (s6_model.to_current_frame); its poses at
    1.0 ... 3.5 s after the previous moment = 0.5 ... 3.5 s after the current one (prev waypoints 1-7 vs current waypoints 0-6, the shared
    0.5-3.5 s window); ADE over those 7 poses (xy) < 0.5 m counts as "follows"."""
    ok, n, ades = 0, 0, []
    for tok in toks:
        d = dump[tok]; p = d["prev"]
        if p is None:
            continue
        q = dump[p]
        prev_pick = torch.from_numpy(np.asarray(q["proposals"], np.float64))[q["chosen"]][None]          # (1, 8, 3) in the previous frame
        moved = SM.to_current_frame(prev_pick, torch.from_numpy(np.asarray(q["human"], np.float64))[None])[0]   # -> current frame
        cur = np.asarray(d["proposals"], np.float64)[d["chosen"]]
        ade = float(np.linalg.norm(moved[1:8, :2].numpy() - cur[0:7, :2], axis=-1).mean())
        ades.append(ade); ok += int(ade < 0.5); n += 1
    return {"rate": ok / n if n else None, "n": n, "ade_mean_m": float(np.mean(ades)) if ades else None, "ade_median_m": float(np.median(ades)) if ades else None}


def motion_source_check(dump: dict) -> dict:
    """consistency of the motion source: the previous moment's human future moved into the current frame vs the current human future
    (prev waypoints 1-7 vs current 0-6); both come from the same ego log, so the mismatch should be ~0."""
    errs = []
    for tok, d in dump.items():
        if d["prev"] is None:
            continue
        q = dump[d["prev"]]
        ph = torch.from_numpy(np.asarray(q["human"], np.float64))[None]
        moved = SM.to_current_frame(ph, ph)[0]
        errs.append(float(np.linalg.norm(moved[1:8, :2].numpy() - np.asarray(d["human"], np.float64)[0:7, :2], axis=-1).max()))
    return {"pairs": len(errs), "max_mismatch_m": float(np.max(errs)) if errs else None, "p99_mismatch_m": float(np.percentile(errs, 99)) if errs else None,
            "median_mismatch_m": float(np.median(errs)) if errs else None}


SAT = {"ties": 0}


def safety_tolerant(dump, pk, grades):
    """train_018b.safety (:155-189) with ONE line changed: its assertion "pick is not the argmax of the last logit" becomes a count.
    The model's pick is argmax(sigmoid(last logit)) (drive_jepa_model.py:85, 90); when two logits are both so large that sigmoid rounds
    both to 1.0 in fp32, that argmax is the first saturated index while argmax of the raw logit is the larger one, and 0.1.8b's assertion
    fires (it did in the pilot's carry arm, pass 1.216, 2026-11-10 04:54). The pick used everywhere stays the model's actual pick; the
    number of such saturated ties is recorded (SAT["ties"]). Built from the imported source so nothing else differs."""
    import inspect
    src = inspect.getsource(TB.safety)
    old = 'assert (ch == lg.argmax(1)).all(), "pick is not the argmax of the last logit"'
    assert old in src, "train_018b.safety changed"
    src = src.replace(old, 'SAT["ties"] = int((ch != lg.argmax(1)).sum())').replace("def safety(", "def _safety_tolerant(")
    ns = dict(vars(TB)); ns["SAT"] = SAT; exec(src, ns)
    return ns["_safety_tolerant"](dump, pk, grades)


def measures(dump: dict, pk: pd.DataFrame, grader, ref: dict | None = None) -> dict:
    """the check's numbers on the 2,000 picking moments (train_018b.safety / go_split / pai_train.per_moment + follows_previous)."""
    toks = list(pk.sample_id); gi = np.where(pk.has_label2.to_numpy())[0]
    assert all(dump[toks[i]]["lab"] and dump[toks[i]]["mc"] for i in gi)
    t0 = time.time()
    grades = np.stack(grader.grade_menus([dump[toks[i]]["mc"] for i in gi], [dump[toks[i]]["proposals"] for i in gi]))
    tg = time.time() - t0
    SAT["ties"] = 0; sf = safety_tolerant(dump, pk, grades)
    lg_all = np.stack([np.asarray(dump[k]["pred_logit"], np.float32).reshape(32, -1)[:, -1] for k in toks])
    sat = {"saturated_ties": SAT["ties"], "max_last_logit": float(lg_all.max()), "moments_with_last_logit_above_16": int((lg_all.max(1) > 16).sum()),
           "pick_grade_raw_argmax_map": None}
    per = T.per_moment({k: dump[k] for k in toks}); groups = T.groups_of(pk); summ = T.summary(per, groups)
    out = {"pick_grade_mean_map": sf["pick_grade_mean"], "map_moments": sf["map_moments"], "hard_picks_above10": sf["hard_picks_above10"],
           "moments_above10": sf["moments_above10"], "go_rate_pull_away": sf["go_rate_pull_away"], "standstill_pull_away": sf["standstill_pull_away"],
           "closest_of_32_ade3_all": summ["ALL"]["m1a"], "pick_ade3_all": summ["ALL"]["m6_pick_ade"], "pick_ade3_slow15": summ["SLOW15"]["m6_pick_ade"],
           "follows_previous": follows_previous(dump, toks), "safety": sf, "summary": summ, "grading_s": round(tg, 1), "logit_saturation": sat}
    if sat["saturated_ties"]:                       # the pick grade the raw-logit argmax would have had (information)
        fin = grades[..., -1]; raw = lg_all[gi].argmax(1); sat["pick_grade_raw_argmax_map"] = float(fin[np.arange(len(gi)), raw].mean())
    if ref is not None:
        out["vs_018b"] = {"closest_of_32_ade3_delta_m": out["closest_of_32_ade3_all"] - ref["closest_of_32_ade3_all"],
                          "ade_within_tolerance": bool(out["closest_of_32_ade3_all"] <= ref["closest_of_32_ade3_all"] + ADE_TOL),
                          "follows_previous_rate_018b": ref["follows_previous"]["rate"]}
    out["gate_hard_ok"] = (out["hard_picks_above10"] == 0) if HARD_GATE_RULE == "zero" else (ref is None or out["hard_picks_above10"] <= ref["hard_picks_above10"])
    out["gate_go_ok"] = out["go_rate_pull_away"] is not None and out["go_rate_pull_away"] >= GO_PULL_MIN
    out["gate_ok"] = bool(out["gate_hard_ok"] and out["gate_go_ok"])
    return out, grades


def run_check(model, store_pk, pk, grader, ref=None, progress=None) -> tuple[dict, dict]:
    t0 = time.time(); dump = replay_slice(model, store_pk, progress=progress); t1 = time.time()
    m, grades = measures(dump, pk, grader, ref)
    m["time_s"] = {"replay": round(t1 - t0, 1), "measures": round(time.time() - t1, 1)}; m["mode"] = model.mode
    return m, {"dump": dump, "grades": grades}


# ----------------------------------------------------------------------------------------------------------------------- #
# The run (built; NOT run in build 3)
# ----------------------------------------------------------------------------------------------------------------------- #


def train(mode: str, run_name: str, seed: int = 0, oversample: bool = True, go_from_sid: bool = False) -> None:
    """passes of in-order windows with a check every CHECK_EVERY steps and at the end of each pass; 0.1.8b's stop and choice rules."""
    assert go_from_sid, "a training run needs Sid's go"
    C.setup_env(); torch.manual_seed(seed); np.random.seed(seed)
    gate = C.memory_gate(True)
    agent, info = SM.load_agent(); model = SM.MemoryModel(agent, mode).cuda()
    store, store_pk, pk = SD.ClipStore("training"), SD.ClipStore("picking"), picking_frame()
    grader = MB._GRADER = MB.Grader(POOL); index_reader = T.parallel_index_reader(agent); opts = model.optimisers()
    run_dir = RUNS / run_name; (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    rec = {"mode": mode, "seed": seed, "oversample": oversample, "start": info, "memory_gate": gate, "losses": {k: [] for k in ("loss",) + LESSONS},
           "steps": [], "checks": [], "settings": {"w_human": W_HUMAN, "w_pull_away": W_PULL, "hard_gate": HARD_GATE_RULE, "go_gate": GO_PULL_MIN}}
    mem = SD.memory_moment_set() if oversample else None
    c0, _ = run_check(model, store_pk, pk, grader); c0.update(step=0, progress=0.0); rec["checks"].append(c0); ref = c0
    best, since, gstep = -1.0, 0, 0
    model.save_checkpoint(run_dir / "ckpt" / "p00.0.pt", {"step": 0, "progress": 0.0, "seed": seed, "check": c0})

    def do_check(progress):
        nonlocal best, since
        c, _ = run_check(model, store_pk, pk, grader, ref); c.update(step=gstep, progress=progress); rec["checks"].append(c)
        model.save_checkpoint(run_dir / "ckpt" / f"p{progress:04.1f}.pt", {"step": gstep, "progress": progress, "seed": seed, "check": c})
        if c["gate_ok"] and c["pick_grade_mean_map"] > best:
            best, since = c["pick_grade_mean_map"], 0
        else:
            since += 1
        C.wj(run_dir / "run.json", rec)
        return f"no check passing both gates with a better pick grade for {NO_IMPROVE} checks (pass {progress:.1f})" if since >= NO_IMPROVE else None

    stop = None
    for p in range(1, MAX_PASSES + 1):
        plan = SD.plan_pass(store, seed, p, oversample, mem); rec.setdefault("passes", []).append({k: v for k, v in plan.items() if k != "batches"})
        np.random.seed(seed + p)                                                                           # the vocabulary term's draws (train_018b.py:1335)
        pf = SD.Prefetcher(store, plan["batches"])
        for step, items in enumerate(pf, 1):
            model.train(); q = window_step(model, SD.window_tensors(items), grader, index_reader, opts); gstep += 1
            for k in rec["losses"]:
                rec["losses"][k].append(q[k])
            rec["steps"].append({k: q[k] for k in ("map_rows", "grade_wait_s", "step_s", "M_norm_last")})
            if not q["finite"]:
                stop = f"non-finite loss at pass {p} step {step}"; break
            if step % CHECK_EVERY == 0 or step == plan["steps"]:
                stop = do_check(round(p - 1 + step / plan["steps"], 3) if step < plan["steps"] else float(p))
                if stop:
                    break
            if C.avail_ram_gb() < 20:
                stop = "available RAM < 20 GB"; break
        pf.close()
        if stop:
            break
    rec["stopped"] = stop or f"reached {MAX_PASSES} passes"
    good = [c for c in rec["checks"] if c["progress"] > 0 and c["gate_ok"]]
    ch = max(good, key=lambda c: (c["pick_grade_mean_map"], -c["progress"])) if good else None
    rec["chosen"] = ch["progress"] if ch else None; rec["ended"] = time.strftime("%F %T")
    C.wj(run_dir / "run.json", rec); grader.close()
