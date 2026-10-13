#!/usr/bin/env python
"""S4: the full training run. Extends the S3.5 smoke trainer (05_small_overfit_check/s35_train.py); all model / loader changes stay in
05_small_overfit_check/train_patch/. Upstream recipe kept: Adam from agent.get_optimizers, lr 1e-4, batch 32, 16-bit mixed precision, no gradient
clipping, GridMask in train mode. Sid's settings: all of 04_training_set/manifest.parquet; type-A or graded samples drawn twice per pass
(cap 2x); shipped checkpoint; encoder frozen; vocab_10240; acceleration zeroed; 10 passes; evaluate on the unseen set after every pass;
save every pass. Caps: 12 GB GPU (hard per-process limit), 10 h wall cumulative over resumes. Resumable per pass.
Unseen set: val/manifest_val.parquet until val_full/A2_DONE.json exists, then val_full from the next pass on; at the end every saved
pass is evaluated on val_full so all passes have numbers on the same set."""
import os, sys, json, time, pathlib, argparse
HERE = pathlib.Path(__file__).resolve().parent; SMOKE = HERE.parent / "05_small_overfit_check"; sys.path.insert(0, str(SMOKE / "train_patch"))
import s35_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--passes", type=int, default=10); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--device", default="cuda"); ap.add_argument("--workers", type=int, default=14)
ap.add_argument("--pool", type=int, default=10); ap.add_argument("--cap-gb", type=float, default=12.0); ap.add_argument("--cap-h", type=float, default=10.0); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="run")
ap.add_argument("--reserve-gb", type=float, default=8.0); ap.add_argument("--score-chunk", type=int, default=2); ap.add_argument("--redo-eval", default=""); ap.add_argument("--ckpt-run", default=""); ap.add_argument("--reeval-passes", default=""); ap.add_argument("--seed", type=int, default=20261008); ap.add_argument("--reeval-only", action="store_true")
a = ap.parse_args(); T0 = time.time(); OUT = HERE / a.out; OUT.mkdir(exist_ok=True); (OUT / "readouts").mkdir(exist_ok=True)
NASOUT = C.NAS / "06_first_training_run" / (a.ckpt_run or a.out); CK = NASOUT / "ckpt"; CK.mkdir(parents=True, exist_ok=True); SLIM = C.NAS / "06_first_training_run" / "slim"   # --ckpt-run: a side process evaluating another run's saved passes
C.setup_env()
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
import s3_common as C3
from dataset_s35 import CacheOnlyDataset
log_f = open(OUT / "train.log", "a")
STATE_F = OUT / "state.json"; state = json.load(open(STATE_F)) if STATE_F.exists() else dict(pass_done=0, elapsed_h=0.0, stopped=None, tripwires=[], eval_set="val", gpu_max_reserved_gb=0.0, launches=0)
state["launches"] += 1; EL0 = state["elapsed_h"]
def elapsed(): return EL0 + (time.time() - T0) / 3600
def log(*x):
    s = f"[{time.strftime('%m-%d %H:%M:%S')} +{elapsed():5.2f}h] " + " ".join(str(v) for v in x); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
def save_state(): state["elapsed_h"] = elapsed(); json.dump(state, open(STATE_F, "w"), indent=1)
dev = torch.device(a.device); cuda = dev.type == "cuda"
if cuda:
    tot = torch.cuda.get_device_properties(0).total_memory; torch.cuda.set_per_process_memory_fraction(min(1.0, a.cap_gb * 2**30 / tot))
    log("GPU", torch.cuda.get_device_name(0), f"total {tot/2**30:.1f} GiB, process cap {a.cap_gb} GiB; torch {torch.__version__}; launch {state['launches']}, resumed at pass {state['pass_done']}, {EL0:.2f} h used")
S3 = C.S3; VAL_SMALL = S3 / "val/manifest_val.parquet"; VAL_FULL = S3 / "val_full/manifest_val_full.parquet"; VAL_FULL_DONE = S3 / "val_full/A2_DONE.json"
tr = pd.read_parquet(S3 / "manifest.parquet")
if a.limit: tr = tr.groupby(["type", "has_label2"], group_keys=False).head(a.limit)
C3._assert_train(set(tr.clip_uuid)); C3._assert_train(set(tr[tr.source == "nurec"].clip_uuid), nurec=True); assert set(tr.split) == {"train"}
vocab = np.load(C.VOCAB["ext"]); assert vocab.shape == (10240, 40, 3)
agent = C.build_agent(vocab="ext", freeze_encoder=True, init=True, scorer_impl="s3", pool_workers=a.pool, score_chunk=a.score_chunk); assert agent.anchors.shape == (10240, 8, 3)
log("launch args:", vars(a)); log("vocabulary", C.VOCAB["ext"], vocab.shape, "| init checkpoint", C.CKPT, C.CKPT_SHA256[:12], "| encoder frozen | zero_accel on | scorer_impl s3")
cache = C.NAS / "04_training_set/s1/cache"
def make_ds(man, guard):
    return CacheOnlyDataset(cache_path=str(cache), feature_builders=agent.get_feature_builders(), target_builders=agent.get_target_builders(), zero_accel=True,
                            has_label2=dict(zip(man.sample_id, man.has_label2)), token_clips=dict(zip(man.sample_id, man.clip_uuid)), slim_root=str(SLIM), split_guard=lambda u: guard(list(u)))
ds_tr = make_ds(tr, C3._assert_train); assert ds_tr.tokens == list(tr.sample_id)
EVAL = {}                                                         # name -> (manifest, dataset)
def eval_set(name):
    if name not in EVAL:
        if name == "train_probe":   # fixed seeded sample of the TRAINING set (150 per type x graded group), evaluated like the unseen set: training error per type, same mode
            man = tr.groupby(["type", "has_label2"], group_keys=False).apply(lambda g: g.sample(min(len(g), 150), random_state=a.seed)).reset_index(drop=True)
            EVAL[name] = (man, make_ds(man, C3._assert_train)); log("training probe:", len(man), "samples |", man.groupby(["type", "has_label2"]).size().to_dict()); return EVAL[name]
        man = pd.read_parquet(VAL_SMALL if name == "val" else VAL_FULL)
        if a.limit: man = man.groupby(["type", "has_label2"], group_keys=False).head(a.limit)
        C3.assert_val(set(man.clip_uuid)); C3.assert_val(set(man[man.source == "nurec"].clip_uuid), nurec=True); assert set(man.split) == {"val"}
        assert not (set(man.group) & set(pd.read_parquet(S3 / "manifest.parquet", columns=["group"]).group)), "unseen chunk also in train"; assert not (set(man.clip_uuid) & set(tr.clip_uuid))
        EVAL[name] = (man, make_ds(man, C3.assert_val)); log(f"unseen set '{name}':", len(man), "samples,", man.clip_uuid.nunique(), "clips,", int(man.has_label2.sum()), "graded |", man.groupby("type").size().to_dict())
    return EVAL[name]
agent.to(dev); m = agent._pad_model
if cuda:   # shared card: other tenants fill it in bursts (first launch died at step 0 with 0.5 GiB free on the whole GPU). Take our working set once
    try:   # and keep it in torch's cache, so later steps do not depend on what the neighbours leave free. Stays inside the 12 GiB cap.
        _r = torch.empty(int(a.reserve_gb * 2**30), dtype=torch.uint8, device=dev); del _r; log(f"pre-reserved {torch.cuda.memory_reserved()/2**30:.1f} GiB of GPU memory (cap {a.cap_gb} GiB)")
    except torch.OutOfMemoryError: log("could not pre-reserve GPU memory (card is full right now); continuing without")
def guarded(fn, what):
    """Run fn(). An out-of-memory error while WE hold less than 90 % of our cap is the neighbours' doing, not our cap: wait and retry
    (up to 30 min). At or near our own cap it is tripwire T4."""
    for k in range(31):
        try: return fn()
        except torch.OutOfMemoryError as e:
            ours = torch.cuda.memory_reserved() / 2**30
            if ours >= 0.9 * a.cap_gb or k == 30: raise
            state["external_oom_waits"] = state.get("external_oom_waits", 0) + 1; log(f"  GPU full from other tenants during {what} (we hold {ours:.1f} GiB): waiting 60 s, retry {k+1}/30"); torch.cuda.empty_cache() if ours < 1 else None; time.sleep(60)
TRAINABLE = lambda: {k: v for k, v in agent.state_dict().items() if "_backbone" not in k}
opt = agent.get_optimizers(); scaler = torch.amp.GradScaler("cuda", enabled=cuda)
def ck_path(p): return CK / f"pass_{p:02d}.pt"
def load_weights(p): sd = torch.load(ck_path(p), map_location=dev); missing, unexpected = agent.load_state_dict(sd["trainable"], strict=False); assert not unexpected and all("_backbone" in k for k in missing); return sd
def to_dev(b):
    f, t = b; f = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in f.items()}; t = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()}; return f, t
LK = ["loss", "trajectory_loss", "final_score_loss", "sub_score_loss", "pred_ce_loss", "pred_l1_loss", "pred_area_loss", "min_loss", "inter_loss"]; SK = ["final_score_loss", "pred_ce_loss", "pred_l1_loss", "pred_area_loss"]
def fv(x): return float(x.detach().float().cpu()) if torch.is_tensor(x) else float(x)
def step_loss(batch):
    f, t = to_dev(batch)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cuda):
        pred = agent.forward(f)
    pred = {k: (v.float() if torch.is_tensor(v) else ([x.float() for x in v] if isinstance(v, list) else v)) for k, v in pred.items()}      # loss in fp32
    return pred, agent.compute_loss(f, t, pred), t
def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else np.nan
def batches(n, bs):
    b = [list(range(i, min(i + bs, n))) for i in range(0, n, bs)]
    if len(b) > 1 and len(b[-1]) < 4: tail = b.pop(); b[-1] = b[-1] + tail            # upstream zips the 4 refinement stages with the batch (agent :226): never fewer than 4 rows
    return b
@torch.no_grad()
def evaluate(name, p):
    """Eval mode (GridMask off). Losses are batch means, accumulated weighted by batch size; readouts are per sample."""
    man, ds = eval_set(name); TY = dict(zip(man.sample_id, man.type)); SP = dict(zip(man.sample_id, man.speed)); SRC = dict(zip(man.sample_id, man.source)); t0 = time.time()
    # batches are homogeneous in (type, graded) so that every loss term can be attributed to a group (losses are batch means)
    grp = (man.type + "|" + man.has_label2.map({True: "graded", False: "ungraded"})).to_numpy(); bl = []; bg = []
    for g in sorted(set(grp)):
        ii = np.where(grp == g)[0]
        for b in batches(len(ii), a.bs): bl.append(ii[b].tolist()); bg.append(g if len(b) >= 4 else "mixed")
    for k in [k for k, b in enumerate(bl) if len(b) < 4][::-1]:           # a group with fewer than 4 samples joins a neighbouring batch (agent :226 needs >= 4 rows)
        small = bl.pop(k); bg.pop(k); j = k - 1 if k > 0 else 0; bl[j] = bl[j] + small; bg[j] = "mixed"
    assert sorted(i for b in bl for i in b) == list(range(len(ds))), "evaluation batches must cover every example exactly once"
    agent.eval(); np.random.seed(1234); dl = DataLoader(ds, batch_sampler=bl, num_workers=a.workers, collate_fn=default_collate); rows = []; L = {k: 0.0 for k in LK}; n = nl = 0; Ls = {k: 0.0 for k in SK}; GR = []
    for bi, batch in enumerate(dl):
        pred, ld, t = guarded(lambda: step_loss(batch), f"eval '{name}' pass {p}"); B = len(t["token"]); n += B
        for k in LK: L[k] += fv(ld[k]) * B
        GR.append(dict(set=name, passno=p, group=bg[bi], n=B, **{k: fv(ld[k]) for k in LK}))
        props = pred["proposals"].cpu().numpy(); pdm = pred["pdm_score"].cpu().numpy(); tgt = t["trajectory"].cpu().numpy(); h2 = t["has_label2"].cpu().numpy().astype(bool); ref = None
        if h2.any():
            fs = agent.compute_score(t, pred["proposals"], test=False)[0]; ref = fs.cpu().numpy(); nl += int(h2.sum())
            for k in SK: Ls[k] += fv(ld[k]) * int(h2.sum())
        j = 0
        for i, tok in enumerate(t["token"]):
            ch = int(pdm[i].argmax()); st = np.diff(np.concatenate([np.zeros((32, 1, 2)), props[i][:, :, :2]], 1), axis=1); ends = np.linalg.norm(st[:, -1], axis=1) / 0.5; d4 = np.linalg.norm(props[i, :, -1, :2] - tgt[i, -1, :2], axis=1)
            r = dict(passno=p, set=name, token=tok, type=TY[tok], source=SRC[tok], speed=SP[tok], has_label2=bool(h2[i]), dist4=float(d4[ch]), ade=float(np.linalg.norm(props[i, ch, :, :2] - tgt[i, :, :2], axis=1).mean()), best_dist4=float(d4.min()),
                     fastest_end=float(ends.max()), chosen_end=float(ends[ch]), human_end=float(np.linalg.norm(tgt[i, -1, :2] - tgt[i, -2, :2]) / 0.5))
            if h2[i]:
                g = ref[j]; j += 1; r.update(top1=bool(g[ch] >= g.max() - 1e-9), top1_strict=bool(ch == int(g.argmax())), rank_corr=spearman(pdm[i], g), ref_chosen=float(g[ch]), ref_best=float(g.max()), ref_std=float(g.std()), dist4_of_ref_best=float(d4[int(g.argmax())]))
            rows.append(r)
    agent.train(); df = pd.DataFrame(rows); df.to_parquet(OUT / "readouts" / f"{name}_pass{p:02d}.parquet", index=False); pd.DataFrame(GR).to_parquet(OUT / "readouts" / f"{name}_pass{p:02d}_batchloss.parquet", index=False)
    out = dict(set=name, passno=p, n=n, n_label2=nl, **{k: L[k] / max(n, 1) for k in LK}, **{f"{k}_graded": Ls[k] / max(nl, 1) for k in SK}, ordinary_dist4=float(df[df.type == "ordinary"].dist4.mean()), eval_secs=time.time() - t0)
    ev = pd.read_csv(OUT / "evals.csv") if (OUT / "evals.csv").exists() else pd.DataFrame(); ev = pd.concat([ev[~((ev.set == name) & (ev.passno == p))] if len(ev) else ev, pd.DataFrame([out])], ignore_index=True); ev.to_csv(OUT / "evals.csv", index=False)
    log(f"eval '{name}' pass {p}: total {out['loss']:.3f} (generator {out['trajectory_loss']:.3f}, scorer on graded {out['final_score_loss_graded']:.3f}) | ordinary 4 s distance {out['ordinary_dist4']:.2f} m | {out['eval_secs']:.0f} s"); return out
def have(name, p):
    if not (OUT / "evals.csv").exists(): return None
    ev = pd.read_csv(OUT / "evals.csv"); q = ev[(ev.set == name) & (ev.passno == p)]; return q.iloc[0].to_dict() if len(q) else None
def eval_other_pass(name, p, current):
    """evaluate a SAVED pass without disturbing training: load its weights, evaluate, put the current weights back"""
    load_weights(p); out = evaluate(name, p); agent.load_state_dict(current, strict=False); return out

# ---------------------------------------------------------------- sampler (Sid: highway x2, graded x2, not stacking)
TYPE = tr.type.to_numpy(); H2 = tr.has_label2.to_numpy(); twice = np.where((TYPE == "A") | H2)[0]; draws = np.concatenate([np.arange(len(tr)), twice])
tab = pd.DataFrame({"type": TYPE, "graded": H2, "w": 1 + ((TYPE == "A") | H2)}); sh = tab.groupby("type").agg(samples=("w", "size"), draws=("w", "sum")); sh["raw_share"] = sh.samples / len(tab); sh["draw_share"] = sh.draws / sh.draws.sum()
log("sampler:", len(draws), "draws per pass from", len(tr), "samples (", len(twice), "drawn twice ) | graded share of draws", round(float(tab[tab.graded].w.sum() / tab.w.sum()), 4), "raw", round(float(H2.mean()), 4)); log("\n" + sh.round(4).to_string())
assert tab.w.max() == 2
# ---------------------------------------------------------------- pass 0 / resume
if state["pass_done"] == 0 and not ck_path(0).exists(): torch.save(dict(trainable=TRAINABLE(), passno=0), ck_path(0)); log("saved", ck_path(0))
if state["pass_done"] > 0:
    sd = load_weights(state["pass_done"]); opt.load_state_dict(sd["opt"]); scaler.load_state_dict(sd["scaler"]); log("resumed weights + optimiser from", ck_path(state["pass_done"]))
def stop(code, why): state["stopped"] = code; state["tripwires"].append(dict(code=code, why=why, at_pass=state["pass_done"])); save_state(); log("TRIPWIRE", code, why)
def check_tripwires():
    ev = pd.read_csv(OUT / "evals.csv"); cur = pd.read_csv(OUT / "curve.csv"); name = state["eval_set"]; e = ev[ev.set == name].sort_values("passno"); p = state["pass_done"]
    seq = e[e.passno.isin([p - 3, p - 2, p - 1, p])]
    if len(seq) == 4 and (np.diff(seq.loss.to_numpy()) > 0).all():
        trl = cur[cur.passno.isin([p - 2, p - 1, p])].train_loss.to_numpy(); prev = cur[cur.passno == p - 3].train_loss.to_numpy()
        if len(prev) and len(trl) == 3 and (np.diff(np.r_[prev, trl]) < 0).all(): return stop("T1", f"unseen total error rose 3 passes in a row on '{name}' ({seq.loss.round(3).tolist()}) while training error fell")
    b = e[e.passno == 0]; c = e[e.passno == p]
    if len(b) and len(c) and c.ordinary_dist4.iloc[0] > 1.10 * b.ordinary_dist4.iloc[0]: return stop("T2", f"ordinary 4 s distance on '{name}' {c.ordinary_dist4.iloc[0]:.3f} m > 1.10 x pass 0 ({b.ordinary_dist4.iloc[0]:.3f} m)")
if a.redo_eval:   # recompute every saved pass on the named sets (used once, after the evaluation-batching fix of 04:20)
    cur_w = TRAINABLE()
    for name in a.redo_eval.split(","):
        for p in range(0, state["pass_done"] + 1): eval_other_pass(name, p, cur_w)
if not a.reeval_only:
    if have(state["eval_set"], 0) is None: evaluate(state["eval_set"], 0)
    if have("train_probe", 0) is None: evaluate("train_probe", 0)
    agent.train(); pass_secs = []
    for p in range(state["pass_done"] + 1, a.passes + 1):
        if state["stopped"]: break
        est = (np.mean(pass_secs) / 3600) if pass_secs else 0.0
        if elapsed() + est > a.cap_h: stop("T4", f"wall cap {a.cap_h} h: {elapsed():.2f} h used, next pass needs ~{est:.2f} h"); break
        gen = torch.Generator().manual_seed(a.seed + p); perm = draws[torch.randperm(len(draws), generator=gen).numpy()]
        dl = DataLoader(Subset(ds_tr, perm.tolist()), batch_sampler=batches(len(perm), a.bs), num_workers=a.workers, collate_fn=default_collate, prefetch_factor=4, persistent_workers=False)
        L = {k: 0.0 for k in LK}; n = nl = steps = 0; seen = {}; t_ep = time.time(); bad = None; t_log = time.time()
        try:
            for batch in dl:
                def one():
                    opt.zero_grad(set_to_none=True); pred, ld, t = step_loss(batch)
                    if np.isfinite(fv(ld["loss"])): scaler.scale(ld["loss"]).backward()
                    return pred, ld, t
                pred, ld, t = guarded(one, f"pass {p} step {steps}"); lv = fv(ld["loss"])
                if not np.isfinite(lv): bad = f"non-finite loss at pass {p} step {steps}"; break
                scaler.step(opt); scaler.update()
                B = len(t["token"]); n += B; nl += int(batch[1]["has_label2"].sum()); steps += 1
                for k in LK: L[k] += fv(ld[k]) * B
                if time.time() - t_log > 600:
                    t_log = time.time(); r = (time.time() - t_ep) / steps; log(f"  pass {p} step {steps}/{len(dl)}: running loss {L['loss']/n:.3f} | {r:.2f} s/step | pass ETA {(len(dl)-steps)*r/60:.0f} min | GPU {torch.cuda.max_memory_reserved()/2**30:.1f} GiB"); save_state()
        except torch.OutOfMemoryError as e:
            stop("T4", f"GPU memory cap {a.cap_gb} GiB hit at pass {p} step {steps}: {str(e)[:120]}"); break
        if bad: stop("T3", bad); break
        tr_secs = time.time() - t_ep; pass_secs.append(tr_secs); pt = pd.Series(TYPE[perm]).value_counts(); state["gpu_max_reserved_gb"] = max(state["gpu_max_reserved_gb"], torch.cuda.max_memory_reserved() / 2**30 if cuda else 0)
        torch.save(dict(trainable=TRAINABLE(), opt=opt.state_dict(), scaler=scaler.state_dict(), passno=p), ck_path(p)); state["pass_done"] = p; save_state()
        if state["eval_set"] == "val" and VAL_FULL_DONE.exists() and VAL_FULL.exists():
            state["eval_set"] = "val_full"; log("val_full is ready: switching per-pass evaluation to it from this pass; evaluating pass 0 on it first (tripwire baseline)")
            if have("val_full", 0) is None: eval_other_pass("val_full", 0, TRAINABLE())
        e = evaluate(state["eval_set"], p); evaluate("train_probe", p)
        row = dict(passno=p, **{f"train_{k}": L[k] / n for k in LK}, train_secs=tr_secs, steps=steps, sec_per_step=tr_secs / steps, draws=n, graded_draws=nl, graded_share_seen=nl / n, **{f"seen_share_{k}": float(v) / len(perm) for k, v in pt.items()},
                   eval_set=state["eval_set"], unseen_loss=e["loss"], gpu_reserved_gb=torch.cuda.max_memory_reserved() / 2**30 if cuda else 0, gpu_alloc_gb=torch.cuda.max_memory_allocated() / 2**30 if cuda else 0, elapsed_h=elapsed())
        cur = pd.read_csv(OUT / "curve.csv") if (OUT / "curve.csv").exists() else pd.DataFrame(); pd.concat([cur, pd.DataFrame([row])], ignore_index=True).to_csv(OUT / "curve.csv", index=False)
        log(f"PASS {p} done: training error {row['train_loss']:.3f} (generator {row['train_trajectory_loss']:.3f}, scorer {row['train_final_score_loss']:.3f}) | unseen '{state['eval_set']}' {e['loss']:.3f} | {row['sec_per_step']:.2f} s/step, {tr_secs/60:.0f} min + eval {e['eval_secs']/60:.1f} min | seen {pt.div(len(perm)).round(3).to_dict()} graded {nl/n:.3f} | GPU {row['gpu_reserved_gb']:.1f} GiB | {elapsed():.2f} h")
        save_state(); check_tripwires()
# ---------------------------------------------------------------- every saved pass on val_full
if VAL_FULL_DONE.exists() and VAL_FULL.exists():
    cur_w = TRAINABLE(); todo = [p for p in ([int(x) for x in a.reeval_passes.split(",")] if a.reeval_passes else range(0, state["pass_done"] + 1)) if ck_path(p).exists() and have("val_full", p) is None]; log("re-evaluation on val_full: passes", todo)
    for p in todo:
        if elapsed() > a.cap_h and not a.reeval_only: log(f"wall cap {a.cap_h} h reached during re-evaluation: passes {[q for q in todo if q >= p]} not evaluated on val_full (run --reeval-only)"); state["reeval_pending"] = [q for q in todo if q >= p]; break
        eval_other_pass("val_full", p, cur_w)
    else: state["reeval_pending"] = []
else: log("val_full not ready at the end of training: no re-evaluation"); state["reeval_pending"] = "val_full missing"
save_state(); log("DONE", json.dumps({k: state[k] for k in ("pass_done", "stopped", "eval_set", "gpu_max_reserved_gb", "elapsed_h")}))
