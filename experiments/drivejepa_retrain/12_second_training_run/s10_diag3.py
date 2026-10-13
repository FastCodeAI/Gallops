#!/usr/bin/env python
"""S10 task 3: CPU-only diagnostic of the batch dependence (no training, no GPU, float32, eval mode, jitter off).
Setup = the header of s10_gen_train.py (executed verbatim up to the gate-A section, output dir moved to NAS/12_second_training_run/diag3).
The frozen image encoder is run once per example (true forward, batch 16) and its output kept; every batching experiment then runs
the rest of the model (history encoding, 4 refinement stages, scorer) on those features. Q1 instead runs the encoder inside each
batching it tests. Corrections come from train_patch/batch_fix.py. Stages: enc_ho q1 runs enc_probe q6 (default: all, in that order)."""
import os, sys, pathlib, time
assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "CPU only"
HERE = pathlib.Path(__file__).resolve().parent
STAGES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["enc_ho", "q1", "runs", "enc_probe", "q6"]; SMOKE = "--smoke" in sys.argv
src = open(HERE / "s10_gen_train.py").read().split("# ================================================================ gate A")[0]
assert 'C.NAS / "12_second_training_run" / "gen" / a.run' in src; src = src.replace('C.NAS / "12_second_training_run" / "gen" / a.run', 'C.NAS / "12_second_training_run" / "diag3"').replace("__file__", "str(HERE / 's10_gen_train.py')")
sys.argv = ["s10_gen_train.py", "--mode", "readout", "--device", "cpu", "--workers", "12"]
exec(src)
import batch_fix as BF
import torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 24))); OUT = NASOUT; assert str(OUT).endswith("12_second_training_run/diag3") and not cuda
FOUND = BF.install(m); log("batch_fix installed on:", FOUND); agent.eval()
P0 = C.NAS / "06_first_training_run/run/ckpt/pass_00.pt"; sd0 = torch.load(P0, map_location="cpu"); assert sd0["passno"] == 0; SD = {0: sd0["trainable"], 9: REF9}
def use(p):
    mi, un = agent.load_state_dict(SD[p], strict=False); assert not un and all("_backbone" in k for k in mi); agent.eval()
# pass 0 against the shipped checkpoint file
if "enc_ho" in STAGES:
    ck = torch.load(C.CKPT, map_location="cpu"); ck = ck.get("state_dict", ck); eq = ne = miss = 0; nel = []
    for k, v in SD[0].items():
        c = [kk for kk in ck if kk.endswith(k.replace("_pad_model.", "", 1)) or kk.endswith(k)]
        if not c: miss += 1; continue
        if ck[c[0]].shape == v.shape and torch.equal(ck[c[0]], v): eq += 1
        else: ne += 1; nel.append(k)
    json.dump(dict(pass_00_file=str(P0), pass_00_sha256=sha256(P0), shipped_ckpt=str(C.CKPT), shipped_sha256=sha256(C.CKPT), shipped_sha256_expected=C.CKPT_SHA256, tensors_in_pass_00=len(SD[0]), equal_to_shipped=eq, different=ne, different_keys=nel[:20], not_found_in_shipped=miss),
              open(OUT / "pass0_vs_shipped.json", "w"), indent=1); log("pass 0 vs shipped:", eq, "equal,", ne, "different,", miss, "not found", nel[:6]); del ck

# ---------------------------------------------------------------- encoder: true forward once per example, kept
_bb = m._backbone.forward
def bb_cached(camera_feature, img_metas=None, **kw):
    if img_metas is not None and "_feat" in img_metas:
        x = img_metas["_feat"]; return x.permute(1, 0, 2)[None], torch.as_tensor([[16, 32]], dtype=torch.long), torch.zeros(1, dtype=torch.long), {"img_metas": img_metas}
    return _bb(camera_feature, img_metas=img_metas, **kw)
m._backbone.forward = bb_cached
def lite(it): return ({k: (v.clone() if torch.is_tensor(v) else v) for k, v in it[0].items() if not k.startswith("camera_feature")}, {k: (v.clone() if torch.is_tensor(v) else v) for k, v in it[1].items()})   # clone: do not keep the loader's shared-memory handles
@torch.no_grad()
def enc_true(items):
    """the encoder exactly as drive_jepa_model.py:53-60 calls it, on this batch -> (bs, 512, 256)"""
    f, _ = default_collate(items); f["lidar2img"] = f["lidar2img"][:, 1:2]
    cam = torch.cat([m.transform(f["camera_feature_2"])[:, None], m.transform(f["camera_feature_1"])[:, None]], dim=1)
    ff, ss, li, _ = _bb(cam, img_metas=f); assert ss.tolist() == [[16, 32]] and li.tolist() == [0] and ff.shape[0] == 1; return ff[0].permute(1, 0, 2).contiguous()
def load_items(ds, idx): return list(DataLoader(ds, batch_size=None, sampler=list(idx), num_workers=12))
def build_bank(ds, idx, path, bs=16):
    if path.exists(): b = torch.load(path); assert b["idx"] == list(idx); log("bank loaded", path, tuple(b["feat"].shape)); return b
    t0 = time.time(); feat = torch.empty(len(idx), 512, 256); items = []; buf = []; n = 0
    for it in DataLoader(ds, batch_size=None, sampler=list(idx), num_workers=12):
        buf.append(it)
        if len(buf) == bs or n + len(buf) == len(idx):
            feat[n:n + len(buf)] = enc_true(buf); items += [lite(x) for x in buf]; n += len(buf); buf = []
            if (n // bs) % 20 == 0: log(f"  encoder {n}/{len(idx)}  {(time.time()-t0)/n:.3f} s/example")
    b = dict(idx=list(idx), feat=feat, items=items, sec_per_example=(time.time() - t0) / len(idx)); torch.save(b, path); log("bank saved", path, f"{b['sec_per_example']:.3f} s/example"); return b
@torch.no_grad()
def head(items, feat, fix=(0, 0, 0), rec=False):
    """history encoding + 4 refinement stages + scorer on given encoder features; items are lite (no camera tensors)"""
    f, t = default_collate(items); n = len(items); f["camera_feature_1"] = torch.zeros(n, 3, 2, 2); f["camera_feature_2"] = torch.zeros(n, 3, 2, 2); f["_feat"] = feat
    BF.set_fix(*fix); BF.STATE["rec"] = [] if rec else None; pred, t = forward(agent, (f, t)); r = BF.STATE["rec"]; BF.STATE["rec"] = None; BF.set_fix(); return pred, t, r
def pack(pred): return torch.stack(pred["proposal_list"], 1), pred["pred_logit"], pred["pdm_score"]      # (bs,4,32,8,3), (bs,32,S), (bs,32)
FIXES = {"nofix": (0, 0, 0), "fix1": (1, 0, 0), "fix1+fix2": (1, 1, 0), "fix1+fix3": (1, 0, 1), "fix1+fix2+fix3": (1, 1, 1)}
dsh = make_ds(ho); assert dsh.tokens == list(ho.sample_id); NH = len(ho)
HW = np.where(((ho.slice == "highway") & (ho.speed >= 20)).to_numpy())[0]; ORD = np.where((ho.slice == "ordinary").to_numpy())[0]; assert len(HW) == 575
rng = np.random.default_rng(a.seed)
bank = build_bank(dsh, range(NH if not SMOKE else 96), OUT / ("enc_ho.pt" if not SMOKE else "enc_ho_smoke.pt")) if any(s in STAGES for s in ("enc_ho", "q1", "runs")) else None
if bank is not None and "enc_ho" in STAGES:
    f0, t0_ = bank["items"][0]; log("feature keys:", {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in f0.items()}, "| target keys:", {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in t0_.items()})

# ================================================================ Q1: is the cause whole? (pass 9, true encoder forward inside every batching)
if "q1" in STAGES:
    use(9); t0 = time.time()
    if SMOKE: rows = list(range(0, 16)); B = 8; others_pool = np.arange(16, 96)
    else: rows = HW[0:32].tolist() + HW[288:320].tolist() + ORD[0:32].tolist() + ORD[1400:1432].tolist(); B = 32; others_pool = np.setdiff1d(np.arange(NH), rows)
    n = len(rows); kind = ["highway >= 20 m/s" if ho.slice.iloc[r] == "highway" else "ordinary" for r in rows]
    r1 = np.random.default_rng(a.seed + 3); pr = r1.permutation(n); oth = r1.permutation(others_pool); npb = B // 4; cb = []
    for k in range(n // npb):                                   # (c): each batch = B/4 of the rows + 3B/4 other held-out rows, positions shuffled
        mem = [rows[j] for j in pr[k * npb:(k + 1) * npb]] + oth[k * (B - npb):(k + 1) * (B - npb)].tolist(); cb.append([mem[j] for j in r1.permutation(B)])
    need = sorted(set(rows) | {x for b_ in cb for x in b_}); real = dict(zip(need, load_items(dsh, need))); log(f"q1: {n} rows, {len(need)} examples loaded with cameras ({time.time()-t0:.0f} s)")
    BATCH = {"a_batch1": [[r] for r in rows], "b_batch32_clip_order": [rows[i:i + B] for i in range(0, n, B)], "c_batch32_shuffled_with_others": cb, "d_batch32_reversed": [rows[i:i + B][::-1] for i in range(0, n, B)]}
    RES = {}; ENC = {}
    for bn, bl_ in BATCH.items():
        for b_ in bl_:
            fe = enc_true([real[i] for i in b_]); its = [lite(real[i]) for i in b_]
            for fn, fx in FIXES.items():
                pl, lg, sc = pack(head(its, fe, fx)[0])
                for j, i in enumerate(b_):
                    if i in rows: RES[(bn, fn, i)] = (pl[j].clone(), lg[j].clone(), sc[j].clone())
            for j, i in enumerate(b_):
                if i in rows: ENC[(bn, i)] = fe[j].clone()
        log(f"q1 batching {bn} done ({time.time()-t0:.0f} s)")
    out = []; per_row = []
    for fn in FIXES:
        for bn in list(BATCH)[1:]:
            dp = np.array([float((RES[(bn, fn, i)][0][..., :2] - RES[("a_batch1", fn, i)][0][..., :2]).abs().max()) for i in rows]); dh = np.array([float((RES[(bn, fn, i)][0][..., 2] - RES[("a_batch1", fn, i)][0][..., 2]).abs().max()) for i in rows])
            dl = np.array([float((RES[(bn, fn, i)][1] - RES[("a_batch1", fn, i)][1]).abs().max()) for i in rows]); pk = np.array([int(RES[(bn, fn, i)][2].argmax() != RES[("a_batch1", fn, i)][2].argmax()) for i in rows])
            st = np.array([[float((RES[(bn, fn, i)][0][s, ..., :2] - RES[("a_batch1", fn, i)][0][s, ..., :2]).abs().max()) for s in range(4)] for i in rows])
            for j, i in enumerate(rows): per_row.append(dict(fix=fn, batching=bn, row=i, token=ho.sample_id.iloc[i], kind=kind[j], speed=float(ho.speed.iloc[i]), plan_xy_max_abs_diff_m=dp[j], heading_max_abs_diff_rad=dh[j], logit_max_abs_diff=dl[j], pick_changed=pk[j], **{f"stage{s+1}_xy_diff_m": st[j, s] for s in range(4)}))
            for kd in ["all", "highway >= 20 m/s", "ordinary"]:
                k_ = np.array([kd == "all" or x == kd for x in kind])
                if not k_.any(): continue
                out.append(dict(fix=fn, batching=bn + " vs a_batch1", rows=kd, n=int(k_.sum()), plan_xy_diff_max_m=dp[k_].max(), plan_xy_diff_median_over_rows_m=np.median(dp[k_]), heading_diff_max_rad=dh[k_].max(), logit_diff_max=dl[k_].max(), logit_diff_median_over_rows=np.median(dl[k_]), picks_changed=int(pk[k_].sum()),
                                **{f"stage{s+1}_xy_diff_max_m": st[k_, s].max() for s in range(4)}))
    b1 = {fn: dict(plans_max_abs_diff_vs_nofix=max(float((RES[("a_batch1", fn, i)][0] - RES[("a_batch1", "nofix", i)][0]).abs().max()) for i in rows), logits_max_abs_diff_vs_nofix=max(float((RES[("a_batch1", fn, i)][1] - RES[("a_batch1", "nofix", i)][1]).abs().max()) for i in rows)) for fn in FIXES}
    encd = {bn: dict(max_abs=max(float((ENC[(bn, i)] - ENC[("a_batch1", i)]).abs().max()) for i in rows), scale=max(float(ENC[("a_batch1", i)].abs().max()) for i in rows)) for bn in list(BATCH)[1:]}
    # the kept encoder features (batch 16) against the encoder run at batch 1, and the plans they give
    bi = {x: k for k, x in enumerate(bank["idx"])}; kept = dict(encoder_max_abs=max(float((bank["feat"][bi[i]] - ENC[("a_batch1", i)]).abs().max()) for i in rows))
    kept["plans_xy_max_abs_m_batch1_kept_vs_true"] = max(float((pack(head([bank["items"][bi[i]]], bank["feat"][bi[i]][None])[0])[0][0][..., :2] - RES[("a_batch1", "nofix", i)][0][..., :2]).abs().max()) for i in rows)
    pd.DataFrame(out).to_csv(OUT / "q1_summary.csv", index=False); pd.DataFrame(per_row).to_csv(OUT / "q1_per_row.csv", index=False)
    # trace under both fixes: row 0 inside a batch of 8 against alone, every leaf module outside the encoder
    def first_tensor(o):
        if torch.is_tensor(o): return o
        if isinstance(o, (list, tuple)):
            for x in o:
                r_ = first_tensor(x)
                if r_ is not None: return r_
        return None
    def trun(ii, fx):
        store = {}; mods = [(n_, mod) for n_, mod in m.named_modules() if n_ and not n_.startswith("_backbone") and not any(True for _ in mod.children())]
        hs = [mod.register_forward_hook(lambda mod, inp, out_, n_=n_: store.setdefault(n_, []).append((first_tensor(inp), first_tensor(out_)))) for n_, mod in mods]
        pred = head([lite(real[i]) for i in ii], torch.stack([ENC[("a_batch1", i)] for i in ii]), fx)[0]
        for h in hs: h.remove()
        return store, pred["proposal_list"]
    def row0(x, y):
        if x is None or y is None or x.dim() != y.dim(): return None
        dd = [k for k in range(x.dim()) if x.shape[k] != y.shape[k]]
        if len(dd) != 1 or y.shape[dd[0]] * 8 != x.shape[dd[0]]: return None
        return x.narrow(dd[0], TR * y.shape[dd[0]], y.shape[dd[0]])
    TR = 5; tr_rows = []; i8 = [rows[0]] + rows[n // 2:n // 2 + 3] + rows[5:9]; tsum = {}
    for fn, fx in FIXES.items():
        SB, PB = trun(i8, fx); SA, PA = trun(i8[TR:TR + 1], fx); nb = 0; first = None; first_in = None
        for n_ in SA:
            for c, ((ia, oa), (ib, ob)) in enumerate(zip(SA[n_], SB.get(n_, []))):
                ri = row0(ib, ia); ro = row0(ob, oa); d_ = dict(fix=fn, module=n_, call=c, in_diff=None if ri is None else float((ri.float() - ia.float()).abs().max()), out_diff=None if ro is None else float((ro.float() - oa.float()).abs().max()), out_scale=None if oa is None else float(oa.float().abs().max())); tr_rows.append(d_)
                if d_["out_diff"] is not None and d_["out_diff"] > 1e-3: nb += 1; first = first or f"{n_} call {c} (in_diff {d_['in_diff']}, out_diff {d_['out_diff']:.4g})"; first_in = first_in or (f"{n_} call {c} (in_diff {d_['in_diff']:.4g})" if (d_["in_diff"] or 0) > 1e-3 else None)
        tsum[fn] = dict(leaf_module_calls_compared=sum(1 for r_ in tr_rows if r_["fix"] == fn and r_["out_diff"] is not None), calls_with_row0_output_diff_above_1e_3=nb, first_such=first, first_call_with_input_diff_above_1e_3=first_in, traced_row=TR, largest_out_diff=max((r_["out_diff"] for r_ in tr_rows if r_["fix"] == fn and r_["out_diff"] is not None)), stage_plans_traced_row_diff=[float((x[TR:TR + 1] - y).abs().max()) for x, y in zip(PB, PA)])
    pd.DataFrame(tr_rows).to_csv(OUT / "q1_trace_row5_batch8_vs_alone.csv", index=False)
    # how often does a row's camera mask differ from row 0's in the clip-order batches (the line fix2 changes)
    cm = []
    for b_ in BATCH["b_batch32_clip_order"]:
        _, _, r = head([lite(real[i]) for i in b_], torch.stack([ENC[("b_batch32_clip_order", i)] for i in b_]), (0, 0, 0), rec=True)
        for d_ in r:
            if "cam_mask" in d_: mk = d_["cam_mask"][0]; cm.append(float((mk != mk[:1]).float().mean()))          # (bs, 256) against row 0
    json.dump(dict(rows=n, batch=B, batch1_fix_vs_nofix=b1, encoder_output_vs_batch1=encd, kept_encoder_features_vs_batch1=kept, trace=tsum, camera_mask_share_of_tokens_differing_from_row0_per_call=dict(mean=float(np.mean(cm)), max=float(np.max(cm)), calls=len(cm)), secs=time.time() - t0), open(OUT / "q1_extra.json", "w"), indent=1)
    log("q1 summary:\n" + pd.DataFrame(out).round(6).to_string()); log("q1 extra:", json.dumps(json.load(open(OUT / "q1_extra.json")), indent=1)); del real, RES, ENC

# ================================================================ runs: SEEN held-out, every example, per pass x batching x fix (Q2, Q3, Q6b, Q7)
def oor_stats(r, B):
    """per example: (4 stages, 8 waypoints) shares over the 32 plans. own = the example's own reference points; used = the points the block actually used for it"""
    refs = [d_ for d_ in r if "ref_in" in d_]; locs = [d_ for d_ in r if "loc" in d_]; assert len(refs) == 4 and len(locs) == 4
    o = {k: np.zeros((B, 4, 8), np.float32) for k in ("own_out", "own_x_hi", "own_x_lo", "own_y_out", "used_out", "zero_mass", "loc_zero")}
    for s in range(4):
        ri = refs[s]["ref_in"][:B, :, 0].reshape(B, 32, 8, 2); ru = refs[s]["ref_used"][:, :, 0].reshape(B, 2, 32, 8, 2)
        out = lambda x: ((x < 0) | (x > 1)).any(-1).float()
        o["own_out"][:, s] = out(ri).mean(1); o["own_x_hi"][:, s] = (ri[..., 0] > 1).float().mean(1); o["own_x_lo"][:, s] = (ri[..., 0] < 0).float().mean(1); o["own_y_out"][:, s] = ((ri[..., 1] < 0) | (ri[..., 1] > 1)).float().mean(1)
        o["used_out"][:, s] = out(ru).mean((1, 2))
        loc = locs[s]["loc"]; w = locs[s]["w"]; H, W = locs[s]["shapes"][0].tolist(); assert (H, W) == (32, 8)          # (2B,256,heads,1,pts,2): x -> W (8 waypoints), y -> H (32 plans)
        px = loc[..., 0] * W - 0.5; py = loc[..., 1] * H - 0.5; z = ((px <= -1) | (px >= W) | (py <= -1) | (py >= H)).float()   # grid_sample align_corners=False, zeros padding: all four neighbours outside
        nh = w.shape[2]; zm = (z * w).sum((-1, -2)).mean(-1); lz = z.mean((-1, -2, -3))                                # attention weights sum to 1 over the points per head per queue row
        o["zero_mass"][:, s] = zm.reshape(B, 2, 32, 8).mean((1, 2)); o["loc_zero"][:, s] = lz.reshape(B, 2, 32, 8).mean((1, 2))
    return o
def run_set(p, batching, fx_name, idx_sel=None):
    path = OUT / f"hrun_p{p}_{batching}_{fx_name.replace('+', '_')}.npz"
    if path.exists() and not SMOKE: return path
    use(p); t0 = time.time(); N = len(bank["idx"]); base = np.arange(N)
    if batching == "one": bl_ = [[i] for i in (base if idx_sel is None else idx_sel)]
    elif batching == "clip": bl_ = batches(N, a.bs)
    else: o_ = np.random.default_rng(a.seed).permutation(N); bl_ = [o_[b_].tolist() for b_ in batches(N, a.bs)]      # = s10_gen_train.py:125
    PL = np.full((N, 4, 32, 8, 3), np.nan, np.float32); SC = np.full((N, 32), np.nan, np.float32); LG = None; O = None; pos = np.full((N, 2), -1, np.int32)
    for bi_, b_ in enumerate(bl_):
        pred, t, r = head([bank["items"][i] for i in b_], bank["feat"][b_], FIXES[fx_name], rec=True); pl, lg, sc = pack(pred); o = oor_stats(r, len(b_))
        if LG is None: LG = np.full((N,) + tuple(lg.shape[1:]), np.nan, np.float32); O = {k: np.full((N, 4, 8), np.nan, np.float32) for k in o}
        PL[b_] = pl.numpy(); SC[b_] = sc.numpy(); LG[b_] = lg.numpy(); pos[b_, 0] = bi_; pos[b_, 1] = np.arange(len(b_))
        for k in o: O[k][b_] = o[k]
    np.savez(path, plans=PL, scores=SC, logits=LG, batch_pos=pos, **O); log(f"run pass {p} {batching} {fx_name}: {len(bl_)} batches, {time.time()-t0:.0f} s -> {path.name}"); return path
if "runs" in STAGES:
    tg = torch.stack([it[1]["trajectory"] for it in bank["items"]]).numpy(); np.save(OUT / "ho_targets.npy", tg); ho.iloc[bank["idx"]][["sample_id", "clip_uuid", "type", "slice", "speed", "has_label2"]].to_parquet(OUT / "ho_meta.parquet", index=False)
    for p, bt, fx in [(9, "one", "nofix"), (9, "shuffle", "nofix"), (9, "clip", "nofix"), (9, "shuffle", "fix1+fix2"), (9, "clip", "fix1+fix2"), (9, "shuffle", "fix1+fix2+fix3"), (9, "clip", "fix1+fix2+fix3"), (0, "one", "nofix"), (0, "shuffle", "nofix"), (0, "clip", "nofix"), (0, "shuffle", "fix1+fix2+fix3"), (0, "clip", "fix1+fix2+fix3")]:
        run_set(p, bt, fx)

# ================================================================ Q6: the training probe with the corrections on (pass 9, shuffled batches of 32)
if "enc_probe" in STAGES or "q6" in STAGES:
    spe = bank["sec_per_example"] if bank is not None else torch.load(OUT / "enc_ho.pt")["sec_per_example"] if (OUT / "enc_ho.pt").exists() else 0.3
    NB = 4 if SMOKE else int(os.environ.get("PROBE_BATCHES", min(300, int(40 * 60 / (32 * spe)))))
    perm = sampler(1); bl = batches(len(perm), a.bs)[:NB]; draws = perm[:NB * a.bs]; uniq = sorted(set(draws.tolist())); dst = make_ds(tr); assert dst.tokens == list(tr.sample_id)
    log(f"probe: {NB} batches, {len(draws)} draws, {len(uniq)} distinct examples (encoder {spe:.3f} s/example)")
    pb = build_bank(dst, uniq, OUT / (f"enc_probe_{NB}.pt" if not SMOKE else "enc_probe_smoke.pt")); row = {x: k for k, x in enumerate(uniq)}
if "q6" in STAGES:
    use(9); TY = dict(zip(tr.sample_id, tr.type)); SP = dict(zip(tr.sample_id, tr.speed)); res = {}
    for fn in ("nofix", "fix1+fix2", "fix1+fix2+fix3"):
        np.random.seed(1234); R = []; E = []; t0 = time.time()
        for bi_, b_ in enumerate(bl):
            ii = [row[int(perm[j])] for j in b_]; pred, t, _ = head([pb["items"][i] for i in ii], pb["feat"][ii], FIXES[fn])
            _, pr = agent.generator_loss_s10(pred["proposal_list"], t["trajectory"], cfg, agent.read_scores_index(t), 0.0)
            R.append({k: fv(pr[k]) for k in ("human", "catalogue", "spacing", "human_last", "catalogue_last", "spacing_last")})
            for i, tok in enumerate(t["token"]): E.append(dict(batch=bi_, token=tok, type=TY[tok], speed=SP[tok], graded=bool(t["has_label2"][i]), human=float(pr["human_rows"][i]), catalogue=float(pr["cat_rows"][i]), spacing=float(pr["spacing_rows"][i])))
        R = pd.DataFrame(R); E = pd.DataFrame(E); E["slice"] = E.type.map(SLICE); E.to_parquet(OUT / f"probe_per_example_{fn.replace('+', '_')}.parquet", index=False); R.to_csv(OUT / f"probe_per_batch_{fn.replace('+', '_')}.csv", index=False); by = {}
        for sl in ["highway", "close-pass", "ordinary", "pull-out (ungraded)", "all", "highway, car >= 20 m/s"]:
            k = slice_mask(E, sl); d = E[k]
            if len(d): by[sl] = dict(draws=int(len(d)), share_of_draws=float(len(d) / len(E)), human_part=float(d.human.mean()), catalogue_part=float(d.catalogue.mean()), spacing_sq_mean_m2=float(d.spacing.mean()), spacing_sq_median_m2=float(d.spacing.median()), spacing_sq_max_m2=float(d.spacing.max()))
        res[fn] = dict(batches=int(len(R)), draws=int(len(E)), C0=float(R.catalogue_last.mean()), H0=float(R.human_last.mean()), S0=float(R.spacing_last.mean()), S0_median_per_batch=float(R.spacing_last.median()), S0_max_per_batch=float(R.spacing_last.max()),
                       stage_combined=dict(C=float(R.catalogue.mean()), H=float(R.human.mean()), S=float(R.spacing.mean())), by_slice=by, secs=time.time() - t0); log(f"probe {fn}:", json.dumps({k: v for k, v in res[fn].items() if k != "by_slice"})); log("\n" + pd.DataFrame(by).T.round(4).to_string())
    out = dict(definition="as spacing_norm.json (last refinement stage, per batch, mean over the probe batches), re-measured on CPU in float32 with the encoder output of each example computed once at batch 16",
               probe=f"first {NB} batches of the pass-1 sampler (seed {a.seed}+1) on the reduced training manifest, batch {a.bs}, eval mode, no grad, pass 9, numpy seed 1234", pass_09_sha256=P9_SHA,
               fixed_fix1_fix2=res["fix1+fix2"], fixed_fix1_fix2_fix3=res["fix1+fix2+fix3"], nofix_same_batches_cpu_float32=res["nofix"], nofix_task2_gpu_16bit_300_batches=json.load(open(HERE / "spacing_norm.json")))
    if not SMOKE: json.dump(out, open(HERE / "spacing_norm_fixed.json", "w"), indent=1)
log("DONE stages", STAGES)
