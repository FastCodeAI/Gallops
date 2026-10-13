#!/usr/bin/env python
"""S10 task 7: menus of ONE generator checkpoint + the MAP-FREE checks of S9 on the 283 raw pull-out clips (9,053 examples), plus clearances.

Scene building and the checks are S9's own functions, imported from 11_scorer_head_and_pullout_grades/s9_c_dump.py (not edited; its module level parses flags, so it
is imported under a temporary argv): load_raw_scene, grade (strict and lenient collision, time buffer, comfort), NoMap. The load / memory
gate and sha256 come from s10_dump_b1.py the same way (its log goes to this run's folder; S9's own log file is never written).
Stored per example: everything s9_c_dump.py stored (same keys; "_9" is the reader's key, NOT the pass) PLUS
  clear_m (32,)        smallest distance in metres between the car's footprint polygon along the plan and any recorded box, over the 41
                       simulated states (0 .. 4 s at 10 Hz) that the checks use: footprint k against the boxes of frame k (the scorer's own
                       _ego_polygons and _observation[k] after S9's grade()); 0 = touching or overlapping; inf = no box in any frame
  clear_t_idx (32,)    the frame of that smallest distance (-1 if no box)
  clear_after0_m (32,) the same over frames 1 .. 40 only (frame 0 is the car where it is, identical for every plan)
  human_clear_m, human_clear_t_idx, human_clear_after0_m   the same for the human path (the calculator's reference row)
  boxes_total          boxes summed over the 41 frames
  ckpt_sha256
  s10_dump_pullout.py --ckpt CKPT --out dump_gen/pullout --bs 1 --fix [--threads 4 --pool 2 --loaders 1]     full run, resumable by shard, DONE.json
  s10_dump_pullout.py --ckpt PASS9 --bs 8 --check REPORT_DIR --check-n 32                                    first rows of shard 0 against S9's stored arrays"""
import os, sys, time, json, pathlib, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", default=""); ap.add_argument("--bs", type=int, default=1); ap.add_argument("--fix", action="store_true")
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--pool", type=int, default=2); ap.add_argument("--loaders", type=int, default=1); ap.add_argument("--shard", type=int, default=256)
ap.add_argument("--check", default=""); ap.add_argument("--check-n", type=int, default=32)
a = ap.parse_args(); assert a.check or a.out
HERE = pathlib.Path(__file__).resolve().parent; RETRAIN = HERE.parent; sys.path.insert(0, str(RETRAIN / "11_scorer_head_and_pullout_grades"))
_argv = sys.argv; _common = ["--threads", str(a.threads), "--pool", str(a.pool), "--loaders", str(a.loaders), "--shard", str(a.shard)]
sys.argv = [_argv[0]] + _common + (["--check", a.check] if a.check else ["--out", a.out])
import s10_dump_b1 as B                                                                       # gate(), sha256(); LOG = <out or check dir>/load_log.csv
sys.argv = [_argv[0]] + _common + ["--bs", str(a.bs)]
_sni = B.torch.set_num_interop_threads; B.torch.set_num_interop_threads = lambda n: None      # already set to 1 by s10_dump_b1; a second call raises
import s9_c_dump as S9                                                                        # load_raw_scene(), grade(), NoMap
sys.argv = _argv; B.torch.set_num_interop_threads = _sni
C = B.C; C3 = B.C3; np = B.np; pd = B.pd; torch = B.torch; OUT = B.OUT
S9DUMP = B.NAS / "11_scorer_head_and_pullout_grades/pullout/dump"; CKPT = pathlib.Path(a.ckpt)


def grade_clear(task):
    """S9's grade() unchanged, then the clearances read from the scorer object that grade() just used (s3_scorer keeps one per vehicle box)."""
    import shapely, s2_grade as G, s3_scorer as SS
    out = S9.grade(task); S = S9.load_raw_scene(task["clip"], task["chunk"]); i = int(round(task["t_us"] / 1e6 / G.K.DT))
    sc = SS.scorer_for(G.ego_state_at(S, i, S["vp"]).car_footprint.vehicle_parameters); P = sc._ego_polygons; N, T = P.shape; assert N == len(task["menu"]) + 1 and T == 41, P.shape
    assert np.allclose(np.hypot(sc._states[1:, :, 3], sc._states[1:, :, 4]).max(1), out["max_sim_speed"], atol=1e-4), "scorer object is not the one grade() used"
    D = np.full((N, T), np.inf); nb = 0
    for k in range(T):
        g = np.asarray(sc._observation[k]._geometries, dtype=object); nb += len(g)
        if len(g): D[:, k] = shapely.distance(P[:, k][:, None], g[None, :]).min(1)
    def pack(Dm, off):
        j = Dm.argmin(1); v = Dm[np.arange(N), j]; return v.astype(np.float32), np.where(np.isfinite(v), j + off, -1).astype(np.int16)
    v, j = pack(D, 0); v1, _ = pack(D[:, 1:], 1)
    out.update(clear_m=v[1:], clear_t_idx=j[1:], clear_after0_m=v1[1:], human_clear_m=v[0], human_clear_t_idx=j[0], human_clear_after0_m=v1[0], boxes_total=np.int32(nb))
    return out


if __name__ == "__main__":
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    man = pd.read_parquet(RETRAIN / "04_training_set/manifest.parquet"); man = man[man.type == "B_raw"].sort_values(["clip_uuid", "anchor_t"]).reset_index(drop=True); assert not man.has_label2.any() and (man.source == "raw").all()   # s9_c_dump.py main, unchanged
    clips = pd.read_csv(RETRAIN / "04_training_set/clips.csv").set_index("uuid"); lab = pd.read_parquet(RETRAIN / "04_training_set/labels_index.parquet").set_index("sample_id"); man["chunk"] = clips.loc[man.clip_uuid, "chunk"].to_numpy(); man["t_us"] = lab.loc[man.sample_id, "t_us"].to_numpy()
    C3._assert_train(set(man.clip_uuid)); assert len(man) == 9053 and man.clip_uuid.nunique() == 283; print(f"pull-out set: {len(man)} examples, {man.clip_uuid.nunique()} clips, {man.chunk.nunique()} chunks", flush=True)
    agent = C.build_agent(vocab="ext", freeze_encoder=True, init=True, scorer_impl="s3", pool_workers=0); agent.eval(); m = agent._pad_model
    ds = B.CacheOnlyDataset(cache_path=str(C.NAS / "04_training_set/s1/cache"), feature_builders=agent.get_feature_builders(), target_builders=agent.get_target_builders(), zero_accel=True, has_label2=dict(zip(man.sample_id, man.has_label2)),
                            token_clips=dict(zip(man.sample_id, man.clip_uuid)), slim_root=None, split_guard=lambda u: C3._assert_train(list(u)))
    idx = {t: i for i, t in enumerate(ds.tokens)}; order = [idx[t] for t in man.sample_id]; info = man.set_index("sample_id")
    import s10_dump_gen_ckpt as LK
    ck_note = LK.load_ckpt(m, CKPT, torch); ck_sha = B.sha256(CKPT); m.eval(); assert not m.training; print(f"checkpoint {CKPT} sha256 {ck_sha}: {ck_note}", flush=True)
    if a.fix:
        import batch_fix as BF
        print("batch_fix installed on:", {k: len(v) for k, v in BF.install(m).items()}, flush=True); m.eval(); BF.set_fix(1, 1, 1)

    @torch.no_grad()
    def trunk(t, ego, image_feature):                                                               # s9_c_dump.py trunk(), unchanged
        bev = t.hist_encoding(ego)[:, None] + t.init_feature.weight[None]
        for refine in t._trajectory_head: bev, props = refine(bev, image_feature)
        Bn = len(props); slot = bev.reshape(Bn, props.shape[1], props.shape[2], -1).amax(-2); logit = t.scorer.pred_score.mlp[-1](t.scorer.pred_score.mlp[:-1](slot)); return props, slot, logit
    pool = ProcessPoolExecutor(max_workers=a.pool, mp_context=mp.get_context("spawn")); tim = dict(encoder=0.0, trunk=0.0, grade=0.0, load=0.0)

    def run_rows(rows, bs, R, gated):
        """forward + checks for dataset rows `rows`, appended to R in order; returns the number done"""
        put = lambda k, v: R.setdefault(k, []).append(v); tl = time.time(); n = 0
        for f, tg in B.DataLoader(ds, batch_sampler=[rows[i:i + bs] for i in range(0, len(rows), bs)], num_workers=a.loaders, collate_fn=B.default_collate):
            tim["load"] += time.time() - tl
            if gated: B.gate()
            with torch.no_grad():
                te = time.time(); f["lidar2img"] = f["lidar2img"][:, 1:2]; ego = f["ego_status"][:, -1]; cam = torch.cat([m.transform(f["camera_feature_2"])[:, None], m.transform(f["camera_feature_1"])[:, None]], dim=1)
                imf = m._backbone(cam, img_metas=f); tim["encoder"] += time.time() - te; tt = time.time(); props, slot, logit = trunk(m, ego, imf); tim["trunk"] += time.time() - tt
            tgd = time.time(); toks = list(tg["token"]); tasks = [dict(clip=info.loc[t, "clip_uuid"], chunk=int(info.loc[t, "chunk"]), t_us=int(info.loc[t, "t_us"]), menu=props[i].float().numpy()) for i, t in enumerate(toks)]; G_ = list(pool.map(grade_clear, tasks)); tim["grade"] += time.time() - tgd
            for i, t in enumerate(toks):
                put("token", t); put("clip", info.loc[t, "clip_uuid"]); put("anchor_t", float(info.loc[t, "anchor_t"])); put("human", tg["trajectory"][i].numpy().astype(np.float32)); put("ego_status", ego[i].numpy().astype(np.float32))
                put("menu_9", props[i].numpy().astype(np.float32)); put("slot_9", slot[i].numpy().astype(np.float16)); put("logit_9", logit[i].numpy().astype(np.float32))
                for k, v in G_[i].items(): put(k, v)
            n += len(toks); tl = time.time()
            global n_done, said100
            if gated:
                n_done += len(toks)
                if not said100 and n_done >= 100:
                    said100 = True; el = time.time() - t0; print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] TIMED {n_done} examples in {el:.1f} s = {el / n_done:.3f} s/example wall | {json.dumps({k: round(v / n_done, 3) for k, v in tim.items()})}", flush=True)
        return n

    if a.check:
        n = a.check_n; z = np.load(S9DUMP / "shard_0000.npz"); assert list(z["token"][:n]) == list(man.sample_id[:n]); R = {}; t0 = time.time(); run_rows(order[:n], a.bs, R, False); Z = {k: np.stack(v) for k, v in R.items()}
        new = [k for k in Z if k not in z.files]; missing = [k for k in z.files if k not in Z]
        diff = {k: (float(np.abs(Z[k].astype(np.float64) - z[k][:n].astype(np.float64)).max()) if Z[k].dtype.kind in "fiub" else int((Z[k] != z[k][:n]).sum())) for k in Z if k in z.files}
        fin = np.isfinite(Z["clear_m"]); hit = Z["clear_m"] <= 0
        res = dict(rows=n, bs=a.bs, fix=bool(a.fix), checkpoint=str(CKPT), checkpoint_sha256=ck_sha, stored=str(S9DUMP / "shard_0000.npz"), secs=time.time() - t0, threads=a.threads, pool=a.pool, keys_new=new, stored_keys_missing=missing, max_abs_diff_vs_stored=diff,
                   clearance=dict(plans=int(Z["clear_m"].size), finite=int(fin.sum()), zero=int(hit.sum()), median_m=float(np.median(Z["clear_m"][fin])) if fin.any() else None, min_m=float(Z["clear_m"].min()),
                                  human_median_m=float(np.median(Z["human_clear_m"][np.isfinite(Z["human_clear_m"])])) if np.isfinite(Z["human_clear_m"]).any() else None, human_zero=int((Z["human_clear_m"] <= 0).sum()),
                                  # consistency with S9's strict collision check: a plan that fails it (nc_strict < 1) touched a box, so its clearance must be 0
                                  plans_failing_strict_collision=int((Z["nc_strict"] < 1).sum()), of_those_clear_is_zero=int(((Z["nc_strict"] < 1) & hit).sum()), plans_clear_zero_but_passing_strict=int((hit & (Z["nc_strict"] >= 1)).sum())))
        json.dump(res, open(pathlib.Path(a.check) / "check_gen_pullout.json", "w"), indent=1); print(json.dumps(res, indent=1), flush=True); pool.shutdown(); sys.exit(0)

    shards = [order[i:i + a.shard] for i in range(0, len(order), a.shard)]; t0 = time.time(); n_done = 0; said100 = False
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] start: pull-out, bs {a.bs}, fix {a.fix}, threads {a.threads}, pool {a.pool}, loaders {a.loaders}, {len(shards)} shards, {sum((OUT / f'shard_{si:04d}.npz').exists() for si in range(len(shards)))} already present, out {OUT}", flush=True)
    for si, sh in enumerate(shards):
        fn = OUT / f"shard_{si:04d}.npz"
        if fn.exists(): continue
        B.gate(); R = {}; run_rows(sh, a.bs, R, True); assert R["token"] == [ds.tokens[j] for j in sh], "shard order"
        Z = {k: np.stack(v) for k, v in R.items()}; Z["ckpt_sha256"] = np.array(ck_sha); tmp = OUT / f"shard_{si:04d}.tmp.npz"; np.savez(tmp, **Z); os.replace(tmp, fn); el = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] shard {si + 1}/{len(shards)}: {n_done} examples this launch, {el / max(n_done, 1):.2f} s/example wall | {json.dumps({k: round(v / max(n_done, 1), 3) for k, v in tim.items()})} | own memory peak {B.PEAK[0]:.1f} GB | paused {B.PAUSED}", flush=True)
    B.gate(); pool.shutdown(); el = time.time() - t0; ch = os.times(); cpu_s = ch.user + ch.system + ch.children_user + ch.children_system
    tok = []; shas = set()
    for si in range(len(shards)):
        z = np.load(OUT / f"shard_{si:04d}.npz"); tok.append(z["token"]); shas.add(str(z["ckpt_sha256"]))
    tok = np.concatenate(tok); assert list(tok) == list(man.sample_id), "dump tokens != manifest"; assert shas == {ck_sha}, f"shards were written from different checkpoints: {shas}"
    res = dict(done_at=time.strftime("%Y-%m-%dT%H:%M:%S+05:30"), set="pull-out (B_raw)", examples=int(len(tok)), shards=len(shards), clips=int(man.clip_uuid.nunique()), bs=a.bs, batch_fix=bool(a.fix), checkpoint=str(CKPT), checkpoint_sha256=ck_sha, checkpoint_note=ck_note,
               script_sha256=B.sha256(__file__), s9_script_sha256=B.sha256(S9.__file__), dtype="float32", eval_mode=not m.training, examples_this_launch=n_done, wall_s_this_launch=el, cpu_s_this_launch=cpu_s, s_per_example_wall=el / max(n_done, 1),
               per_example_s={k: v / max(n_done, 1) for k, v in tim.items()}, threads=a.threads, pool=a.pool, loaders=a.loaders, paused=B.PAUSED, peak_own_pss_gb=B.PEAK[0], torch=torch.__version__, cuda=torch.cuda.is_available())
    json.dump(res, open(OUT / "DONE.json", "w"), indent=1); print("DONE", json.dumps(res), flush=True)
