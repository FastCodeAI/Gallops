#!/usr/bin/env python
"""S10 task 7: menus + true grades of ONE generator checkpoint on the UNSEEN or the SEEN graded set of S6, CPU, float32, eval mode.

The main loop of s10_dump_b1.py (hashed, not edited) with three things made flags: the set (--set unseen|seen, manifest
08_scorer_diagnosis/set_<set>.parquet, clip guard as s6_dump.py:117), the checkpoint (--ckpt: a pass file with 'trainable', or a merged file
with 'state_dict' in upstream's layout, whose encoder tensors must equal the loaded shipped encoder bit for bit) and the output folder.
The load / memory gate, the grading worker and sha256 are imported from s10_dump_b1.py (its module level parses flags, so it is imported
under a temporary argv; grading workers are spawned and re-import this file the same way).
Array keys are those of S6 / s10_dump_b1.py (menu_9, logit_9, grade_9, ... : "_9" is the reader's key, NOT the pass; the checkpoint is
named in DONE.json and in every shard as ckpt_sha256) so s10_base_tables.py reads the folder unchanged.

  s10_dump_gen.py --set unseen --ckpt CKPT --out dump_gen/unseen --bs 1 --fix [--threads 4 --pool 2 --loaders 1]      full run, resumable by shard
  s10_dump_gen.py --set seen --ckpt PASS9 --bs 8 --check REPORT_DIR --check-n 32                                       first rows of shard 0 against S6's stored arrays"""
import os, sys, time, json, pathlib, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True, choices=["unseen", "seen"]); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", default=""); ap.add_argument("--bs", type=int, default=1)
ap.add_argument("--fix", action="store_true"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--pool", type=int, default=2); ap.add_argument("--loaders", type=int, default=1); ap.add_argument("--shard", type=int, default=256)
ap.add_argument("--check", default=""); ap.add_argument("--check-n", type=int, default=32)
a = ap.parse_args(); assert a.check or a.out
_argv = sys.argv; sys.argv = [_argv[0], "--threads", str(a.threads), "--pool", str(a.pool), "--loaders", str(a.loaders), "--shard", str(a.shard)] + (["--check", a.check] if a.check else ["--out", a.out])
import s10_dump_b1 as B                                                                       # gate(), pool(), grade(), sha256(); LOG = <out or check dir>/load_log.csv
sys.argv = _argv
C = B.C; C3 = B.C3; np = B.np; pd = B.pd; torch = B.torch; S6 = B.S6; OUT = B.OUT
S6DUMP = B.NAS / "08_scorer_diagnosis/dump" / a.set; CKPT = pathlib.Path(a.ckpt)


if __name__ == "__main__":
    man = pd.read_parquet(S6 / f"set_{a.set}.parquet").sort_values(["clip_uuid", "anchor_t"]).reset_index(drop=True); assert man.has_label2.all() and len(man) == 11842
    guard = C3.assert_val if a.set == "unseen" else C3._assert_train; guard(set(man.clip_uuid)); guard(set(man.clip_uuid), nurec=True)                               # s6_dump.py:117
    agent = C.build_agent(vocab="ext", freeze_encoder=True, init=True, scorer_impl="s3", pool_workers=0); agent.eval(); m = agent._pad_model
    class ReadOnlyDataset(B.CacheOnlyDataset):
        def _slim_write(self, *x, **k): raise RuntimeError("read-only on 06_first_training_run/slim: this example has no slim copy")
    ds = ReadOnlyDataset(cache_path=str(C.NAS / "04_training_set/s1/cache"), feature_builders=agent.get_feature_builders(), target_builders=agent.get_target_builders(), zero_accel=True,
                         has_label2=dict(zip(man.sample_id, man.has_label2)), token_clips=dict(zip(man.sample_id, man.clip_uuid)), slim_root=str(C.NAS / "06_first_training_run/slim"), split_guard=lambda u: guard(list(u)))
    idx = {t: i for i, t in enumerate(ds.tokens)}; order = [idx[t] for t in man.sample_id]
    import s10_dump_gen_ckpt as LK
    ck_note = LK.load_ckpt(m, CKPT, torch); ck_sha = B.sha256(CKPT); m.eval(); assert not m.training and m.init_feature.weight.dtype == torch.float32
    print(f"checkpoint {CKPT} sha256 {ck_sha}: {ck_note}", flush=True)

    @torch.no_grad()
    def trunk(t, ego, image_feature):                                                                    # s10_dump_b1.py trunk(), unchanged
        bev = t.hist_encoding(ego)[:, None] + t.init_feature.weight[None]
        for refine in t._trajectory_head: bev, props = refine(bev, image_feature)
        Bn = len(props); slot = bev.reshape(Bn, props.shape[1], props.shape[2], -1).amax(-2)
        h = t.scorer.pred_score.mlp[:-1](slot); logit = t.scorer.pred_score.mlp[-1](h)
        return props, slot, h, logit

    @torch.no_grad()
    def fwd(f):
        f["lidar2img"] = f["lidar2img"][:, 1:2]; ego = f["ego_status"][:, -1]
        cam = torch.cat([m.transform(f["camera_feature_2"])[:, None], m.transform(f["camera_feature_1"])[:, None]], dim=1)
        te = time.time(); imf = m._backbone(cam, img_metas=f); te = time.time() - te; tt = time.time(); o = trunk(m, ego, imf); return o, te, time.time() - tt

    def loader(rows, bs): return B.DataLoader(ds, batch_sampler=[rows[i:i + bs] for i in range(0, len(rows), bs)], num_workers=a.loaders, collate_fn=B.default_collate)

    if a.fix:
        import batch_fix as BF
        print("batch_fix installed on:", {k: len(v) for k, v in BF.install(m).items()}, flush=True); m.eval(); BF.set_fix(1, 1, 1)

    # ================================================================ check: first rows of shard 0 against S6's stored arrays
    if a.check:
        n = a.check_n; z = np.load(S6DUMP / "shard_0000.npz"); assert list(z["token"][:n]) == list(man.sample_id[:n]); R = {}; t0 = time.time()
        for f, tg in loader(order[:n], a.bs):
            (props, slot, h, logit), _, _ = fwd(f); G = list(B.pool().map(B.grade, [dict(mc=tg["metric_cache_path"][i], menu=props[i].float().numpy()) for i in range(len(tg["token"]))]))
            for i in range(len(G)):
                R.setdefault("menu_9", []).append(props[i].numpy().astype(np.float32)); R.setdefault("logit_9", []).append(logit[i].numpy().astype(np.float32)); R.setdefault("slot_9", []).append(slot[i].numpy().astype(np.float16)); R.setdefault("feat_9", []).append(h[i].numpy().astype(np.float16))
                R.setdefault("human", []).append(tg["trajectory"][i].numpy().astype(np.float32)); R.setdefault("states_main", []).append(G[i]["states_main"])
                for k in ("grade", "exempt", "ddc_standard", "maxspeed"): R.setdefault(f"{k}_9", []).append(G[i][k])
        res = dict(set=a.set, rows=n, bs=a.bs, fix=bool(a.fix), checkpoint=str(CKPT), checkpoint_sha256=ck_sha, stored=str(S6DUMP / "shard_0000.npz"), secs=time.time() - t0, threads=a.threads, pool=a.pool,
                   max_abs_diff_vs_stored={k: float(np.abs(np.stack(v).astype(np.float64) - z[k][:n].astype(np.float64)).max()) for k, v in R.items()})
        json.dump(res, open(pathlib.Path(a.check) / f"check_gen_{a.set}.json", "w"), indent=1); print(json.dumps(res, indent=1), flush=True); B.POOL[0].shutdown(); sys.exit(0)

    # ================================================================ full run (s10_dump_b1.py's loop)
    msl = man["slice"].to_numpy(); mspd = man.speed.to_numpy(float); mclip = man.clip_uuid.to_numpy()
    shards = [list(range(i, min(i + a.shard, len(order)))) for i in range(0, len(order), a.shard)]; t0 = time.time(); n_done = 0; tim = dict(encoder=0.0, trunk=0.0, grade=0.0, load=0.0); said100 = False
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] start: set {a.set}, bs {a.bs}, fix {a.fix}, threads {a.threads}, pool {a.pool}, loaders {a.loaders}, {len(shards)} shards, {sum((OUT / f'shard_{si:04d}.npz').exists() for si in range(len(shards)))} already present, out {OUT}", flush=True)
    for si, pos in enumerate(shards):
        fn = OUT / f"shard_{si:04d}.npz"
        if fn.exists(): continue
        B.gate(); R = {}; put = lambda k, v: R.setdefault(k, []).append(v); tl = time.time(); sh = [order[p] for p in pos]
        for f, tg in loader(sh, a.bs):
            tim["load"] += time.time() - tl; B.gate(); Bn = len(tg["token"])
            (props, slot, h, logit), te, tt = fwd(f); tim["encoder"] += te; tim["trunk"] += tt
            tgd = time.time(); G = list(B.pool().map(B.grade, [dict(mc=tg["metric_cache_path"][i], menu=props[i].float().numpy()) for i in range(Bn)])); tim["grade"] += time.time() - tgd
            for i in range(Bn):
                put("token", tg["token"][i]); put("human", tg["trajectory"][i].numpy().astype(np.float32)); put("states_main", G[i]["states_main"])
                put("menu_9", props[i].numpy().astype(np.float32)); put("slot_9", slot[i].numpy().astype(np.float16)); put("feat_9", h[i].numpy().astype(np.float16)); put("logit_9", logit[i].numpy().astype(np.float32))
                for k in ("grade", "exempt", "ddc_standard", "maxspeed"): put(f"{k}_9", G[i][k])
            n_done += Bn; tl = time.time()
            if not said100 and n_done >= 100:
                said100 = True; el = time.time() - t0; print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] TIMED {n_done} examples in {el:.1f} s = {el / n_done:.3f} s/example wall | {json.dumps({k: round(v / n_done, 3) for k, v in tim.items()})}", flush=True)
        assert R["token"] == [ds.tokens[j] for j in sh] == list(man.sample_id[pos]), "shard order"
        Z = {k: np.stack(v) for k, v in R.items()}; Z["slice"] = msl[pos].astype(str); Z["speed"] = mspd[pos]; Z["clip"] = mclip[pos].astype(str); Z["ckpt_sha256"] = np.array(ck_sha)
        tmp = OUT / f"shard_{si:04d}.tmp.npz"; np.savez(tmp, **Z); os.replace(tmp, fn)
        el = time.time() - t0; print(f"[{time.strftime('%H:%M:%S')}] shard {si + 1}/{len(shards)}: {n_done} examples this launch, {el / max(n_done, 1):.2f} s/example wall | {json.dumps({k: round(v / max(n_done, 1), 3) for k, v in tim.items()})} | own memory peak {B.PEAK[0]:.1f} GB | paused {B.PAUSED}", flush=True)
    B.gate()
    if B.POOL[0] is not None: B.POOL[0].shutdown()
    el = time.time() - t0; ch = os.times(); cpu_s = ch.user + ch.system + ch.children_user + ch.children_system
    tok = []; shas = set()
    for si in range(len(shards)):
        z = np.load(OUT / f"shard_{si:04d}.npz"); tok.append(z["token"]); shas.add(str(z["ckpt_sha256"]))
    tok = np.concatenate(tok); assert list(tok) == list(man.sample_id), "dump tokens != manifest"; assert shas == {ck_sha}, f"shards were written from different checkpoints: {shas}"
    info = dict(done_at=time.strftime("%Y-%m-%dT%H:%M:%S+05:30"), set=a.set, examples=int(len(tok)), shards=len(shards), clips=int(man.clip_uuid.nunique()), by_slice={k: int(v) for k, v in man["slice"].value_counts().items()}, highway_ge20=int(((man["slice"] == "highway") & (man.speed >= 20)).sum()),
                bs=a.bs, batch_fix=bool(a.fix), checkpoint=str(CKPT), checkpoint_sha256=ck_sha, checkpoint_note=ck_note, manifest=str(S6 / f"set_{a.set}.parquet"), manifest_sha256=B.sha256(S6 / f"set_{a.set}.parquet"), script_sha256=B.sha256(__file__), dtype="float32", eval_mode=not m.training,
                examples_this_launch=n_done, wall_s_this_launch=el, cpu_s_this_launch=cpu_s, s_per_example_wall=el / max(n_done, 1), per_example_s={k: v / max(n_done, 1) for k, v in tim.items()}, threads=a.threads, pool=a.pool, loaders=a.loaders, paused=B.PAUSED,
                peak_own_pss_gb=B.PEAK[0], torch=torch.__version__, cuda=torch.cuda.is_available())
    json.dump(info, open(OUT / "DONE.json", "w"), indent=1); print("DONE", json.dumps(info), flush=True)
