#!/usr/bin/env python
"""S10 task 4: pass-9 menus + true grades on the UNSEEN graded set, at a chosen batch size (the car runs batch 1), CPU, float32, eval mode.

A copy of the pass-9 path of 08_scorer_diagnosis/s6_dump.py (that file is not edited and cannot be imported: it runs at import). Same dataset
class, same manifest order (clip, time), same split forward (encoder, then history encoding + 4 refinement stages + scorer), same grader call
(s3_scorer.score on the metric cache), same load / memory gate. Differences: one checkpoint (pass 9) instead of three, batch size is a flag,
the batch corrections of train_patch/batch_fix.py are a flag (default off; a no-op at batch 1), output goes to NAS/12_second_training_run/<out>.
Per example (keys as S6 so its readers work): token, human (8,3), menu_9 (32,8,3), slot_9 (32,256) f16, feat_9 (32,1024) f16, logit_9 (32,6),
grade_9 (32,9) columns s3_scorer.TERMS (last = total), exempt_9, ddc_standard_9, maxspeed_9, states_main, and slice / speed / clip from the manifest.
Resumable by shard (finished shard files are skipped). DONE.json with counts is written when every shard exists.

  s10_dump_b1.py --bs 1 --out base_b1 [--fix] [--threads 6 --pool 2 --loaders 1]      full run
  s10_dump_b1.py --check REPORT_DIR                                                   the three checks of the task (128 rows), writes check_a2.json"""
import os, sys, time, json, pathlib, argparse, hashlib
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run with CUDA_VISIBLE_DEVICES=\"\""
HERE = pathlib.Path(__file__).resolve().parent; RETRAIN = HERE.parent; S6 = RETRAIN / "08_scorer_diagnosis"
sys.path.insert(0, str(RETRAIN / "05_small_overfit_check/train_patch")); sys.path.insert(0, str(HERE / "train_patch"))
import s35_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--bs", type=int, default=1); ap.add_argument("--fix", action="store_true"); ap.add_argument("--out", default="base_b1"); ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--pool", type=int, default=2); ap.add_argument("--loaders", type=int, default=1); ap.add_argument("--shard", type=int, default=256); ap.add_argument("--check", default=""); ap.add_argument("--check-n", type=int, default=64)
a = ap.parse_args(); MAIN = 9
NAS = pathlib.Path("{runs}/drivejepa_retrain"); OUT = NAS / "12_second_training_run" / a.out; S6DUMP = NAS / "08_scorer_diagnosis/dump/unseen"; CKPT = NAS / "06_first_training_run/run/ckpt/pass_09.pt"
if not a.check: OUT.mkdir(parents=True, exist_ok=True)
LOG = (pathlib.Path(a.check) if a.check else OUT) / "load_log.csv"
C.setup_env()
import numpy as np, pandas as pd, torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
assert not torch.cuda.is_available(), "CPU only"
torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
import s3_common as C3
from dataset_s35 import CacheOnlyDataset


# ---------------------------------------------------------------- load / memory gate (s6_dump.py:41-100)
def meminfo(k):
    for l in open("/proc/meminfo"):
        if l.startswith(k + ":"): return int(l.split()[1]) / 1048576
def state():
    l1, l5, l15 = open("/proc/loadavg").read().split()[:3]; return float(l1), float(l5), float(l15), meminfo("SwapFree"), meminfo("MemAvailable")
_last = [0.0]
def log_load(note, force=False):
    if force or time.time() - _last[0] >= 120:
        _last[0] = time.time(); l1, l5, l15, sf, ma = state()
        with open(LOG, "a") as f: f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S+05:30')},{l1},{l5},{l15},{sf:.1f},{ma:.1f},{note}\n")
POOL = [None]; PAUSED = dict(load_s=0.0, mem_s=0.0, n_load=0, n_mem=0)
def pool():
    # spawn, not fork: a forked worker would share (and be charged for) the parent's model memory
    if POOL[0] is None: POOL[0] = ProcessPoolExecutor(max_workers=a.pool, mp_context=mp.get_context("spawn"))
    return POOL[0]
def my_rss_gb():
    par = {}; rss = {}
    for d in os.listdir("/proc"):
        if not d.isdigit(): continue
        try:
            for l in open(f"/proc/{d}/status"):
                if l.startswith("PPid:"): par[int(d)] = int(l.split()[1])
                elif l.startswith("VmRSS:"): rss[int(d)] = int(l.split()[1]) / 1048576
        except OSError: pass
    mine = {os.getpid()}; grew = True
    while grew:
        grew = False
        for c, pp in par.items():
            if pp in mine and c not in mine: mine.add(c); grew = True
    pss = 0.0
    for q in mine:
        try:
            for l in open(f"/proc/{q}/smaps_rollup"):
                if l.startswith("Pss:"): pss += int(l.split()[1]) / 1048576; break
        except OSError: pss += rss.get(q, 0.0)
    return pss, sum(rss.get(q, 0.0) for q in mine)
RSS_CAP_GB, START_AVAIL_GB, STOP_AVAIL_GB = 12.0, 40.0, 20.0; PEAK = [0.0, 0.0]
def gate():
    """Blocks while the machine is over the limits; exits if our own memory passes the cap."""
    l1, l5, l15, sf, ma = state(); r, r_rss = my_rss_gb(); PEAK[0] = max(PEAK[0], r); PEAK[1] = max(PEAK[1], r_rss)
    if r > RSS_CAP_GB:
        if POOL[0] is not None: POOL[0].shutdown(wait=False, cancel_futures=True)
        log_load(f"ABORT own memory {r:.1f} GB > cap {RSS_CAP_GB}", force=True); raise SystemExit(f"own resident memory {r:.1f} GB exceeds the {RSS_CAP_GB} GB cap")
    if ma < STOP_AVAIL_GB:
        if POOL[0] is not None: POOL[0].shutdown(wait=True); POOL[0] = None
        PAUSED["n_mem"] += 1; t = time.time(); log_load(f"STOPPED available RAM {ma:.1f} GB < {STOP_AVAIL_GB}", force=True)
        while True:
            time.sleep(120); l1, l5, l15, sf, ma = state(); log_load("stopped (memory)")
            if ma > START_AVAIL_GB and l5 < 32.0: break
        PAUSED["mem_s"] += time.time() - t; log_load("restarted", force=True)
    elif l5 >= 32.0:
        PAUSED["n_load"] += 1; t = time.time(); log_load(f"PAUSED 5-min load {l5} >= 32", force=True)
        while state()[1] >= 32.0 or state()[4] < STOP_AVAIL_GB: time.sleep(120); log_load("paused (load)")
        PAUSED["load_s"] += time.time() - t; log_load("resumed", force=True)
    log_load(f"running own {r:.1f} GB (summed RSS {r_rss:.1f})")


# ---------------------------------------------------------------- grading worker (s6_dump.py:104-111, one menu)
def grade(task):
    from navsim.agents.drive_jepa_perception_based.score_module import compute_navsim_score as CNS
    import s3_scorer as SS
    mc = CNS.load_metric_cache(task["mc"]); sc, ex = SS.score(mc, task["menu"])
    return dict(grade=sc.astype(np.float32), exempt=ex["exempt"], ddc_standard=ex["ddc_standard"].astype(np.float32), maxspeed=ex["max_speed"].astype(np.float32),
                states_main=SS.scorer_for(mc.ego_state.car_footprint.vehicle_parameters)._states[1:].astype(np.float64))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    man = pd.read_parquet(S6 / "set_unseen.parquet").sort_values(["clip_uuid", "anchor_t"]).reset_index(drop=True); assert man.has_label2.all() and len(man) == 11842    # s6_dump.py:115
    guard = C3.assert_val; guard(set(man.clip_uuid)); guard(set(man.clip_uuid), nurec=True)                                                                           # :117, refuses exam / training clips
    agent = C.build_agent(vocab="ext", freeze_encoder=True, init=True, scorer_impl="s3", pool_workers=0); agent.eval(); m = agent._pad_model
    class ReadOnlyDataset(CacheOnlyDataset):
        def _slim_write(self, *x, **k): raise RuntimeError("read-only on 06_first_training_run/slim: this example has no slim copy")
    ds = ReadOnlyDataset(cache_path=str(C.NAS / "04_training_set/s1/cache"), feature_builders=agent.get_feature_builders(), target_builders=agent.get_target_builders(), zero_accel=True,
                         has_label2=dict(zip(man.sample_id, man.has_label2)), token_clips=dict(zip(man.sample_id, man.clip_uuid)), slim_root=str(C.NAS / "06_first_training_run/slim"), split_guard=lambda u: guard(list(u)))
    idx = {t: i for i, t in enumerate(ds.tokens)}; order = [idx[t] for t in man.sample_id]
    sd = torch.load(CKPT, map_location="cpu"); assert sd["passno"] == MAIN
    w = {k[len("_pad_model."):]: v for k, v in sd["trainable"].items()}; missing, unexpected = m.load_state_dict(w, strict=False); assert not unexpected and all(k.startswith("_backbone") for k in missing), (missing[:3], unexpected[:3])
    m.eval(); assert not m.training and m.init_feature.weight.dtype == torch.float32

    @torch.no_grad()
    def trunk(t, ego, image_feature):                                                                    # s6_dump.py:132-137
        bev = t.hist_encoding(ego)[:, None] + t.init_feature.weight[None]
        for refine in t._trajectory_head: bev, props = refine(bev, image_feature)
        B = len(props); slot = bev.reshape(B, props.shape[1], props.shape[2], -1).amax(-2)
        h = t.scorer.pred_score.mlp[:-1](slot); logit = t.scorer.pred_score.mlp[-1](h)
        return props, slot, h, logit

    @torch.no_grad()
    def fwd(f):                                                                                          # s6_dump.py:150-152
        f["lidar2img"] = f["lidar2img"][:, 1:2]; ego = f["ego_status"][:, -1]
        cam = torch.cat([m.transform(f["camera_feature_2"])[:, None], m.transform(f["camera_feature_1"])[:, None]], dim=1)
        te = time.time(); imf = m._backbone(cam, img_metas=f); te = time.time() - te; tt = time.time(); o = trunk(m, ego, imf); return o, te, time.time() - tt

    def loader(rows, bs): return DataLoader(ds, batch_sampler=[rows[i:i + bs] for i in range(0, len(rows), bs)], num_workers=a.loaders, collate_fn=default_collate)

    # ================================================================ checks (task 4, A.2)
    if a.check:
        import batch_fix as BF
        REP = pathlib.Path(a.check); n = a.check_n; nsh = -(-len(man) // a.shard)
        hw20 = ((man["slice"] == "highway") & (man.speed >= 20)).to_numpy(); ordy = (man["slice"] == "ordinary").to_numpy()
        sh_h = max(range(nsh), key=lambda s: hw20[s * a.shard:s * a.shard + n].sum()); sh_o = max(range(nsh), key=lambda s: ordy[s * a.shard:s * a.shard + n].sum())
        pos = list(range(sh_h * a.shard, sh_h * a.shard + n)) + list(range(sh_o * a.shard, sh_o * a.shard + n)); rows = [order[p] for p in pos]; sl = man["slice"].to_numpy()[pos]; spd = man.speed.to_numpy()[pos]
        st = {}
        for s in (sh_h, sh_o):
            z = np.load(S6DUMP / f"shard_{s:04d}.npz"); assert list(z["token"][:n]) == list(man.sample_id[s * a.shard:s * a.shard + n])
            for k in ("menu_9", "logit_9", "grade_9", "slot_9"): st.setdefault(k, []).append(z[k][:n])
        st = {k: np.concatenate(v) for k, v in st.items()}
        def run(bs, fix, do_grade):
            if fix: BF.set_fix(1, 1, 1)
            R = dict(menu=[], logit=[], slot=[], grade=[]); t0 = time.time()
            for f, tg in loader(rows, bs):
                (props, slot, h, logit), _, _ = fwd(f); B = len(tg["token"])
                R["menu"].append(props.numpy().astype(np.float32)); R["logit"].append(logit.numpy().astype(np.float32)); R["slot"].append(slot.numpy().astype(np.float32))
                if do_grade: R["grade"] += [g["grade"] for g in pool().map(grade, [dict(mc=tg["metric_cache_path"][i], menu=props[i].float().numpy()) for i in range(B)])]
            if fix: BF.set_fix()
            R = {k: (np.concatenate(v) if k != "grade" else np.stack(v)) for k, v in R.items() if len(v)}; R["secs"] = time.time() - t0; return R
        A = run(8, False, True); B1 = run(1, False, True)
        found = BF.install(m); m.eval(); B1i = run(1, False, False); C8 = run(8, True, False)
        grp = {"highway shard rows": np.arange(2 * n) < n, "ordinary shard rows": np.arange(2 * n) >= n, "highway, car >= 20 m/s": (sl == "highway") & (spd >= 20), "highway": sl == "highway", "close-pass": sl == "close-pass", "ordinary": sl == "ordinary", "all": np.ones(2 * n, bool)}
        rowmax = lambda x, y: np.abs(x - y).reshape(len(x), -1).max(1)
        res = dict(shards=dict(highway=int(sh_h), ordinary=int(sh_o)), rows_per_shard=n, slice_counts={k: int(v.sum()) for k, v in grp.items()}, threads=a.threads, pool=a.pool, batch_fix_installed_on={k: len(v) for k, v in found.items()},
                   a_batch8_nofix_vs_S6_stored=dict(menu_max_abs_m=float(np.abs(A["menu"] - st["menu_9"]).max()), menu_xy_max_abs_m=float(np.abs(A["menu"][..., :2] - st["menu_9"][..., :2]).max()), logit_max_abs=float(np.abs(A["logit"] - st["logit_9"]).max()),
                                                    grade_total_max_abs=float(np.abs(A["grade"][..., -1] - st["grade_9"][..., -1]).max()), grade_all_terms_max_abs=float(np.abs(A["grade"] - st["grade_9"]).max()),
                                                    slot_max_abs_vs_float16_store=float(np.abs(A["slot"] - st["slot_9"].astype(np.float32)).max()), secs=A["secs"]),
                   batch1_after_install_vs_before=dict(menu_max_abs=float(np.abs(B1i["menu"] - B1["menu"]).max()), logit_max_abs=float(np.abs(B1i["logit"] - B1["logit"]).max())),
                   b_batch1_nofix_vs_a={}, c_batch8_allfix_vs_batch1={}, batch1_secs_per_example_incl_grading=B1["secs"] / (2 * n))
        dxy = rowmax(B1["menu"][..., :2], A["menu"][..., :2]); dg = np.abs(B1["grade"][..., -1] - A["grade"][..., -1]); gA = A["grade"][..., -1].max(1); gB = B1["grade"][..., -1].max(1); cxy = rowmax(C8["menu"][..., :2], B1["menu"][..., :2])
        for k, v in grp.items():
            if not v.any(): continue
            res["b_batch1_nofix_vs_a"][k] = dict(rows=int(v.sum()), plan_xy_diff_max_m=float(dxy[v].max()), plan_xy_diff_median_over_rows_m=float(np.median(dxy[v])), grade_total_abs_diff_mean_over_plans=float(dg[v].mean()), grade_total_abs_diff_max=float(dg[v].max()),
                                                 plans_whose_grade_changes_share=float((dg[v] > 1e-6).mean()), best_grade_mean_batch8=float(gA[v].mean()), best_grade_mean_batch1=float(gB[v].mean()),
                                                 share_with_plan_ge_095_batch8=float((gA[v] >= 0.95).mean()), share_with_plan_ge_095_batch1=float((gB[v] >= 0.95).mean()), mean_grade_all_plans_batch8=float(A["grade"][v][..., -1].mean()), mean_grade_all_plans_batch1=float(B1["grade"][v][..., -1].mean()),
                                                 logit_max_abs_diff=float(rowmax(B1["logit"], A["logit"])[v].max()))
            res["c_batch8_allfix_vs_batch1"][k] = dict(rows=int(v.sum()), plan_xy_diff_max_m=float(cxy[v].max()), plan_xy_diff_median_over_rows_m=float(np.median(cxy[v])), plan_all_dims_max=float(rowmax(C8["menu"], B1["menu"])[v].max()), logit_max_abs_diff=float(rowmax(C8["logit"], B1["logit"])[v].max()))
        json.dump(res, open(REP / "check_a2.json", "w"), indent=1); print(json.dumps(res, indent=1), flush=True)
        if POOL[0] is not None: POOL[0].shutdown()
        sys.exit(0)

    # ================================================================ full run
    if a.fix:
        import batch_fix as BF
        print("batch_fix installed on:", {k: len(v) for k, v in BF.install(m).items()}, flush=True); m.eval(); BF.set_fix(1, 1, 1)
    msl = man["slice"].to_numpy(); mspd = man.speed.to_numpy(float); mclip = man.clip_uuid.to_numpy()
    shards = [list(range(i, min(i + a.shard, len(order)))) for i in range(0, len(order), a.shard)]; t0 = time.time(); n_done = 0; tim = dict(encoder=0.0, trunk=0.0, grade=0.0, load=0.0); said100 = False
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] start: bs {a.bs}, fix {a.fix}, threads {a.threads}, pool {a.pool}, loaders {a.loaders}, {len(shards)} shards, {sum((OUT / f'shard_{si:04d}.npz').exists() for si in range(len(shards)))} already present, out {OUT}", flush=True)
    for si, pos in enumerate(shards):
        fn = OUT / f"shard_{si:04d}.npz"
        if fn.exists(): continue
        gate(); R = {}; put = lambda k, v: R.setdefault(k, []).append(v); tl = time.time(); sh = [order[p] for p in pos]
        for f, tg in loader(sh, a.bs):
            tim["load"] += time.time() - tl; gate(); B = len(tg["token"])
            (props, slot, h, logit), te, tt = fwd(f); tim["encoder"] += te; tim["trunk"] += tt
            tgd = time.time(); G = list(pool().map(grade, [dict(mc=tg["metric_cache_path"][i], menu=props[i].float().numpy()) for i in range(B)])); tim["grade"] += time.time() - tgd
            for i in range(B):
                put("token", tg["token"][i]); put("human", tg["trajectory"][i].numpy().astype(np.float32)); put("states_main", G[i]["states_main"])
                put("menu_9", props[i].numpy().astype(np.float32)); put("slot_9", slot[i].numpy().astype(np.float16)); put("feat_9", h[i].numpy().astype(np.float16)); put("logit_9", logit[i].numpy().astype(np.float32))
                for k in ("grade", "exempt", "ddc_standard", "maxspeed"): put(f"{k}_9", G[i][k])
            n_done += B; tl = time.time()
            if not said100 and n_done >= 100:
                said100 = True; el = time.time() - t0; print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] TIMED {n_done} examples in {el:.1f} s = {el / n_done:.3f} s/example wall | {json.dumps({k: round(v / n_done, 3) for k, v in tim.items()})}", flush=True)
        assert R["token"] == [ds.tokens[j] for j in sh] == list(man.sample_id[pos]), "shard order"
        Z = {k: np.stack(v) for k, v in R.items()}; Z["slice"] = msl[pos].astype(str); Z["speed"] = mspd[pos]; Z["clip"] = mclip[pos].astype(str)
        tmp = OUT / f"shard_{si:04d}.tmp.npz"; np.savez(tmp, **Z); os.replace(tmp, fn)
        el = time.time() - t0; print(f"[{time.strftime('%H:%M:%S')}] shard {si + 1}/{len(shards)}: {n_done} examples this launch, {el / max(n_done, 1):.2f} s/example wall | {json.dumps({k: round(v / max(n_done, 1), 3) for k, v in tim.items()})} | own memory peak {PEAK[0]:.1f} GB | paused {PAUSED}", flush=True)
    gate()
    if POOL[0] is not None: POOL[0].shutdown()
    el = time.time() - t0; ch = os.times(); cpu_s = ch.user + ch.system + ch.children_user + ch.children_system
    # completion: every shard present, tokens = manifest in order
    tok = np.concatenate([np.load(OUT / f"shard_{si:04d}.npz")["token"] for si in range(len(shards))]); assert list(tok) == list(man.sample_id), "dump tokens != manifest"
    info = dict(done_at=time.strftime("%Y-%m-%dT%H:%M:%S+05:30"), examples=int(len(tok)), shards=len(shards), clips=int(man.clip_uuid.nunique()), by_slice={k: int(v) for k, v in man["slice"].value_counts().items()}, highway_ge20=int(((man["slice"] == "highway") & (man.speed >= 20)).sum()),
                bs=a.bs, batch_fix=bool(a.fix), checkpoint=str(CKPT), checkpoint_sha256=sha256(CKPT), manifest=str(S6 / "set_unseen.parquet"), manifest_sha256=sha256(S6 / "set_unseen.parquet"), script_sha256=sha256(__file__), dtype="float32", eval_mode=not m.training,
                examples_this_launch=n_done, wall_s_this_launch=el, cpu_s_this_launch=cpu_s, s_per_example_wall=el / max(n_done, 1), per_example_s={k: v / max(n_done, 1) for k, v in tim.items()}, threads=a.threads, pool=a.pool, loaders=a.loaders, paused=PAUSED,
                peak_own_pss_gb=PEAK[0], torch=torch.__version__, cuda=torch.cuda.is_available())
    json.dump(info, open(OUT / "DONE.json", "w"), indent=1); print("DONE", json.dumps(info), flush=True)
