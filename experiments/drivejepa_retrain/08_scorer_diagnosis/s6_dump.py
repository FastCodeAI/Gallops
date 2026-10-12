#!/usr/bin/env python
"""S6: CPU forward passes + true grades of the menu, per example, for three checkpoints (shipped = pass 0, pass 1, pass 9).

Our own wrapper; nothing vendored or under s3_*/06_first_training_run is edited. The encoder is frozen and identical in all three checkpoints
(06_first_training_run/candidate.json), so it runs once per example; each checkpoint's trunk (generator + scorer) then runs on the same image feature.
Per example and checkpoint p this stores:
  menu_p      (32, 8, 3)   the checkpoint's own 32 plans
  logit_p     (32, 6)      its scorer outputs for its own plans (sigmoid of the last = the score the pick is taken on)
  feat_p      (32, 1024)   the scorer's internal features just before its final layer
  slot_p      (32, 256)    the per-plan feature the scorer network reads
  grade_p     (32, 9)      EPDMS calculator on those 32 plans (s3_scorer.score, D2 + D3), columns s3_scorer.TERMS
  exempt_p    (32,)        D3 exemption applied
and, OFF-DESIGN (head swap): swap_q (32, 6) = score head of checkpoint q in {0, 1} applied to the pass-9 trunk's features.
For pass 9 it also keeps the simulated states of the 32 plans (needed offline for the momentum-aware selection).
Model input per adapter 0.1.6: acceleration zeroed by the loader (zero_accel=True); the bonnet mask (37 rows) is in the stored inputs.

Parallel safety (gate agreed with Sid 2026-10-07, replaces the free-swap gate): CPU only; pauses while the 5-minute load is >= 32;
releases its grading workers and waits when available RAM < 20 GB; (re)starts only at available RAM > 40 GB and load < 32; exits if its
own memory (PSS summed over all its processes) passes 12 GB. Appends load, swap and available RAM to load_log.csv every 2 minutes.
Usage (see run_s6_dump.sh): s6_dump.py --set unseen|seen [--limit N] [--threads T] [--pool K] [--loaders L]"""
import os, sys, time, json, copy, pathlib, argparse
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run with CUDA_VISIBLE_DEVICES=\"\""
HERE = pathlib.Path(__file__).resolve().parent; RETRAIN = HERE.parent; sys.path.insert(0, str(RETRAIN / "05_small_overfit_check/train_patch"))
import s35_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True, choices=["unseen", "seen"]); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--pool", type=int, default=4); ap.add_argument("--loaders", type=int, default=2); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--shard", type=int, default=256); ap.add_argument("--tag", default="")
a = ap.parse_args(); PASSES = [0, 1, 9]; MAIN = 9
NAS = pathlib.Path("{runs}/drivejepa_retrain/08_scorer_diagnosis"); OUT = NAS / ("dump" + a.tag) / a.set; OUT.mkdir(parents=True, exist_ok=True); LOG = HERE / "load_log.csv"
C.setup_env()
import numpy as np, pandas as pd, torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
assert not torch.cuda.is_available(), "S6 is CPU only"
torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
import s3_common as C3
from dataset_s35 import CacheOnlyDataset


# ---------------------------------------------------------------- load / memory gate
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
    # spawn, not fork: a forked worker would share (and be charged for) the parent's model memory; a spawned one holds only the calculator
    if POOL[0] is None: POOL[0] = ProcessPoolExecutor(max_workers=a.pool, mp_context=mp.get_context("spawn"))
    return POOL[0]
def my_rss_gb():
    """Memory held by this process and every descendant (loaders, grading workers), GB. Returns (pss, rss). PSS (proportional set size,
    /proc/<pid>/smaps_rollup) splits each shared page between the processes sharing it, so the sum is the physical memory we really
    occupy; summed RSS counts a page forked from the parent once per child (the 10-example run of 01:28 read 22.1 GB that way while the
    parent peaked at 7.4 GB). The cap is checked on PSS; RSS is logged next to it."""
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
        log_load(f"running {a.set}: ABORT own memory {r:.1f} GB > cap {RSS_CAP_GB}", force=True); raise SystemExit(f"own resident memory {r:.1f} GB exceeds the {RSS_CAP_GB} GB cap")
    if ma < STOP_AVAIL_GB:
        if POOL[0] is not None: POOL[0].shutdown(wait=True); POOL[0] = None
        PAUSED["n_mem"] += 1; t = time.time(); log_load(f"running {a.set}: STOPPED available RAM {ma:.1f} GB < {STOP_AVAIL_GB}", force=True)
        while True:
            time.sleep(120); l1, l5, l15, sf, ma = state(); log_load(f"running {a.set}: stopped (memory)")
            if ma > START_AVAIL_GB and l5 < 32.0: break
        PAUSED["mem_s"] += time.time() - t; log_load(f"running {a.set}: restarted", force=True)
    elif l5 >= 32.0:
        PAUSED["n_load"] += 1; t = time.time(); log_load(f"running {a.set}: PAUSED 5-min load {l5} >= 32", force=True)
        while state()[1] >= 32.0 or state()[4] < STOP_AVAIL_GB: time.sleep(120); log_load(f"running {a.set}: paused (load)")
        PAUSED["load_s"] += time.time() - t; log_load(f"running {a.set}: resumed", force=True)
    log_load(f"running {a.set} own {r:.1f} GB (summed RSS {r_rss:.1f})")


# ---------------------------------------------------------------- grading worker (same call as drive_jepa_agent_s35.get_sub_score_s35, "s3" branch, :52)
def grade(task):
    from navsim.agents.drive_jepa_perception_based.score_module import compute_navsim_score as CNS
    import s3_scorer as SS
    mc = CNS.load_metric_cache(task["mc"]); out = {}
    for p, poses in task["menus"].items():
        sc, ex = SS.score(mc, poses); out[f"grade_{p}"] = sc.astype(np.float32); out[f"exempt_{p}"] = ex["exempt"]; out[f"ddc_standard_{p}"] = ex["ddc_standard"].astype(np.float32); out[f"maxspeed_{p}"] = ex["max_speed"].astype(np.float32)
        if p == task["main"]: out["states_main"] = SS.scorer_for(mc.ego_state.car_footprint.vehicle_parameters)._states[1:].astype(np.float64)
    return out


if __name__ == "__main__":
    man = pd.read_parquet(HERE / f"set_{a.set}.parquet").sort_values(["clip_uuid", "anchor_t"]).reset_index(drop=True); assert man.has_label2.all()
    if a.limit: man = man.groupby("type", group_keys=False).head(-(-a.limit // man.type.nunique())).head(a.limit).reset_index(drop=True)
    guard = C3.assert_val if a.set == "unseen" else C3._assert_train; guard(set(man.clip_uuid)); guard(set(man.clip_uuid), nurec=True)
    agent = C.build_agent(vocab="ext", freeze_encoder=True, init=True, scorer_impl="s3", pool_workers=0); agent.eval(); m = agent._pad_model
    class ReadOnlyDataset(CacheOnlyDataset):
        def _slim_write(self, *x, **k): raise RuntimeError("S6 is read-only on 06_first_training_run/slim: this example has no slim copy")
    ds = ReadOnlyDataset(cache_path=str(C.NAS / "04_training_set/s1/cache"), feature_builders=agent.get_feature_builders(), target_builders=agent.get_target_builders(), zero_accel=True,
                         has_label2=dict(zip(man.sample_id, man.has_label2)), token_clips=dict(zip(man.sample_id, man.clip_uuid)), slim_root=str(C.NAS / "06_first_training_run/slim"), split_guard=lambda u: guard(list(u)))
    idx = {t: i for i, t in enumerate(ds.tokens)}; order = [idx[t] for t in man.sample_id]
    # one small trunk (everything except the encoder) per checkpoint
    bb = m._backbone; m._backbone = None; TR = {}
    for p in PASSES:
        t = copy.deepcopy(m); sd = torch.load(C.NAS / f"06_first_training_run/run/ckpt/pass_{p:02d}.pt", map_location="cpu"); assert sd["passno"] == p
        w = {k[len("_pad_model."):]: v for k, v in sd["trainable"].items()}; missing, unexpected = t.load_state_dict(w, strict=False); assert not unexpected and not missing, (missing[:3], unexpected[:3]); TR[p] = t.eval()
    m._backbone = bb; m.load_state_dict({**{k: v for k, v in m.state_dict().items() if k.startswith("_backbone")}, **TR[MAIN].state_dict()}, strict=True)   # agent == pass 9 whole, for the parity check

    @torch.no_grad()
    def trunk(t, ego, image_feature):
        bev = t.hist_encoding(ego)[:, None] + t.init_feature.weight[None]                               # drive_jepa_model.py:61-62
        for refine in t._trajectory_head: bev, props = refine(bev, image_feature)                        # :65-69
        B = len(props); slot = bev.reshape(B, props.shape[1], props.shape[2], -1).amax(-2)                # scorer.py:41
        h = t.scorer.pred_score.mlp[:-1](slot); logit = t.scorer.pred_score.mlp[-1](h)                    # scorer.py:42, split before the final layer
        return props, slot, h, logit

    shards = [order[i:i + a.shard] for i in range(0, len(order), a.shard)]; t0 = time.time(); n_done = 0; tim = dict(encoder=0.0, trunks=0.0, grade=0.0, load=0.0); parity = None
    for si, sh in enumerate(shards):
        fn = OUT / f"shard_{si:04d}.npz"
        if fn.exists(): continue
        gate(); R = {}; put = lambda k, v: R.setdefault(k, []).append(v); tl = time.time()
        dl = DataLoader(ds, batch_sampler=[sh[i:i + a.bs] for i in range(0, len(sh), a.bs)], num_workers=a.loaders, collate_fn=default_collate)
        for f, tg in dl:
            tim["load"] += time.time() - tl; gate(); B = len(tg["token"])
            with torch.no_grad():
                if parity is None:                                                                         # once: our split forward == the model's own forward, pass 9
                    ref = agent.forward({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
                te = time.time(); f["lidar2img"] = f["lidar2img"][:, 1:2]; ego = f["ego_status"][:, -1]                                        # drive_jepa_model.py:50-51
                cam = torch.cat([m.transform(f["camera_feature_2"])[:, None], m.transform(f["camera_feature_1"])[:, None]], dim=1)            # :53-57
                imf = m._backbone(cam, img_metas=f); tim["encoder"] += time.time() - te; tt = time.time(); O = {p: trunk(TR[p], ego, imf) for p in PASSES}   # :60
                slot9 = O[MAIN][1]; SW = {q: TR[q].scorer.pred_score(slot9.clone()) for q in PASSES if q != MAIN}; tim["trunks"] += time.time() - tt
                if parity is None:
                    parity = dict(menu_max_abs=float((O[MAIN][0] - ref["proposals"]).abs().max()), logit_max_abs=float((O[MAIN][3] - ref["pred_logit"]).abs().max()), same_pick=bool((torch.sigmoid(O[MAIN][3])[:, :, -1].argmax(1) == ref["pdm_score"].argmax(1)).all()))
                    print("parity of the split forward vs the model's own forward (pass 9):", parity, flush=True); assert parity["menu_max_abs"] < 1e-4 and parity["logit_max_abs"] < 1e-3 and parity["same_pick"], parity
            tgd = time.time(); tasks = [dict(mc=tg["metric_cache_path"][i], main=MAIN, menus={p: O[p][0][i].float().numpy() for p in PASSES}) for i in range(B)]; G = list(pool().map(grade, tasks)); tim["grade"] += time.time() - tgd
            for i in range(B):
                put("token", tg["token"][i]); put("human", tg["trajectory"][i].numpy().astype(np.float32)); put("states_main", G[i]["states_main"])
                for p in PASSES:
                    put(f"menu_{p}", O[p][0][i].numpy().astype(np.float32)); put(f"slot_{p}", O[p][1][i].numpy().astype(np.float16)); put(f"feat_{p}", O[p][2][i].numpy().astype(np.float16)); put(f"logit_{p}", O[p][3][i].numpy().astype(np.float32))
                    for k in ("grade", "exempt", "ddc_standard", "maxspeed"): put(f"{k}_{p}", G[i][f"{k}_{p}"])
                for q in SW: put(f"swap_{q}", SW[q][i].numpy().astype(np.float32))
            n_done += B; tl = time.time()
        assert R["token"] == [ds.tokens[j] for j in sh], "shard order"
        tmp = OUT / f"shard_{si:04d}.tmp.npz"; np.savez(tmp, **{k: np.stack(v) for k, v in R.items()}); os.replace(tmp, fn)
        el = time.time() - t0; print(f"[{time.strftime('%H:%M:%S')}] {a.set} shard {si + 1}/{len(shards)}: {n_done} examples this launch, {el / max(n_done, 1):.2f} s/example wall | {json.dumps({k: round(v / max(n_done, 1), 3) for k, v in tim.items()})} | own memory peak {PEAK[0]:.1f} GB | paused {PAUSED}", flush=True)
    gate()
    if POOL[0] is not None: POOL[0].shutdown()
    el = time.time() - t0; ch = os.times(); cpu_s = ch.user + ch.system + ch.children_user + ch.children_system
    info = dict(set=a.set, examples_this_launch=n_done, wall_s=el, cpu_s=cpu_s, cpu_h=cpu_s / 3600, s_per_example_wall=el / max(n_done, 1), cpu_s_per_example=cpu_s / max(n_done, 1), per_example_s={k: v / max(n_done, 1) for k, v in tim.items()},
                threads=a.threads, pool=a.pool, loaders=a.loaders, paused=PAUSED, peak_own_pss_gb=PEAK[0], peak_own_summed_rss_gb=PEAK[1], parity=parity, torch=torch.__version__, cuda=torch.cuda.is_available())
    with open(HERE / f"dump_{a.set}{a.tag}_launches.jsonl", "a") as fh: fh.write(json.dumps(info) + "\n")
    print("DONE", json.dumps(info), flush=True)
