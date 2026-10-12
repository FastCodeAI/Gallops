#!/usr/bin/env python
"""S10 task 6: SEEN held-out readout of one checkpoint, forward only, once in FLOAT32 (autocast off, TF32 off) and once under 16-bit
autocast (as the trainer's own readout), on the same batches (batch 32 in clip order, batch fix on). Nothing graded, UNSEEN never touched.
The model, data, patches and helper functions are those of s10_gen_train.py: its text up to the gate-A marker is executed unchanged
(the hashed trainer is not edited). Usage (through run_s10.sh's environment):
  s10_pilot_readout.py --mode readout --run t6_readout/<arm> --init <pass_xx.pt> [--ref-range wide] --workers 3 --reserve-gb 0
Writes NAS/12_second_training_run/gen/<run>/: t6_per_example.parquet, t6_menu_f32.npz, t6_readout.json."""
import sys, pathlib
HERE_ = pathlib.Path(__file__).resolve().parent; SRC = HERE_ / "s10_gen_train.py"; MARK = "# ================================================================ gate A"
head = SRC.read_text().split(MARK)[0]; assert "def readout(" in head
G = dict(__name__="s10_gen_train_head", __file__=str(SRC)); exec(compile(head, str(SRC), "exec"), G)
a = G["a"]; agent = G["agent"]; ho = G["ho"]; torch = G["torch"]; np = G["np"]; pd = G["pd"]; log = G["log"]; NASOUT = G["NASOUT"]; RR = G["RR"]
import json, time
assert a.mode == "readout" and a.batch_fix == "on" and G["cuda"]
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False   # float32 leg = true float32 (the car: CPU float32)
SEED_BOOT, NB = 20261007, 1000                                                              # s7_common / s10_base_tables.py

def fwd(batch, amp):
    f, t = G["to_dev"](batch)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp): pred = agent.forward(f)
    return pred["proposals"].float(), torch.stack([x.float() for x in pred["proposal_list"]], 1), t

@torch.no_grad()
def main():
    t0 = time.time(); ds = G["make_ds"](ho); assert ds.tokens == list(ho.sample_id); agent.eval(); rows = []; MENU = {}
    dl = G["DataLoader"](ds, batch_sampler=G["batches"](len(ds), a.bs), num_workers=a.workers, collate_fn=G["default_collate"])
    for batch in dl:
        out = {}
        for nm, amp in (("f32", False), ("amp", True)):
            P, PL, t = G["guarded"](lambda: fwd(batch, amp), "t6 readout"); tgt = t["trajectory"]; assert torch.equal(P, PL[:, -1])
            hum = torch.linalg.norm(P - tgt[:, None], dim=-1, ord=1).mean(-1).amin(1).cpu().numpy(); spc = agent.spacing_rows(P).cpu().numpy()
            Pn = P.cpu().numpy().astype(np.float64); tn = tgt.cpu().numpy().astype(np.float64)
            d4 = np.linalg.norm(Pn[:, :, -1, :2] - tn[:, None, -1, :2], axis=-1).min(1); sp = G["step_speeds"](Pn[..., :2]); sw = sp.max(-1) - sp.min(-1)
            out[nm] = dict(P=P, PL=PL, hum=hum, spc=spc, d4=d4, sw=sw, finite=torch.isfinite(PL).all(dim=(1, 2, 3, 4)).cpu().numpy())
        dm = (out["f32"]["P"][..., :2] - out["amp"]["P"][..., :2]).abs().amax(dim=(1, 2, 3)).cpu().numpy(); dst = (out["f32"]["PL"][..., :2] - out["amp"]["PL"][..., :2]).abs().amax(dim=(1, 2, 3, 4)).cpu().numpy()
        for i, tok in enumerate(t["token"]):
            r = dict(token=tok, diff_menu_xy_max_m=float(dm[i]), diff_4stages_xy_max_m=float(dst[i]))
            for nm in ("f32", "amp"):
                o = out[nm]; r.update({f"human_term_{nm}": float(o["hum"][i]), f"best_dist4_{nm}": float(o["d4"][i]), f"spacing_sq_{nm}": float(o["spc"][i]), f"swing_{nm}": o["sw"][i].astype(np.float32).tolist(), f"finite_{nm}": bool(o["finite"][i])})
            rows.append(r); MENU[tok] = out["f32"]["P"][i].cpu().numpy()
    df = pd.DataFrame(rows).set_index("token").loc[list(ho.sample_id)].reset_index(); df = pd.concat([ho[["clip_uuid", "type", "slice", "speed", "has_label2"]], df], axis=1)
    df.to_parquet(NASOUT / "t6_per_example.parquet", index=False); np.savez_compressed(NASOUT / "t6_menu_f32.npz", token=np.array(list(ho.sample_id)), menu=np.stack([MENU[k] for k in ho.sample_id]))
    CLIP = df.clip_uuid.to_numpy()
    def boot(ii, stat):
        u, inv = np.unique(CLIP[ii], return_inverse=True); idx = [ii[inv == k] for k in range(len(u))]; rng = np.random.default_rng(SEED_BOOT)
        o = [stat(np.concatenate([idx[k] for k in rng.integers(0, len(u), len(u))])) for _ in range(NB)]; lo, hi = np.percentile(o, [2.5, 97.5]); return [float(stat(ii)), float(lo), float(hi)]
    res = dict(set="SEEN held-out (145 clips)", init=str(a.init), batch_size=a.bs, batching="clip order", batch_fix=a.batch_fix, ref_range=a.ref_range, ref_x_hi_m=RR.STATE["x_hi"], graded=False, bootstrap=f"clip level, {NB} resamples, seed {SEED_BOOT}, 95 %",
               tf32_in_float32_leg=False, all_finite=dict(f32=bool(df.finite_f32.all()), amp=bool(df.finite_amp.all())), secs=time.time() - t0, precision={})
    for nm in ("f32", "amp"):
        SW = np.stack(df[f"swing_{nm}"].to_numpy()); H = df[f"human_term_{nm}"].to_numpy(); D4 = df[f"best_dist4_{nm}"].to_numpy(); S = df[f"spacing_sq_{nm}"].to_numpy(); o = {}
        for sl in G["SLICES"]:
            k = G["slice_mask"](df, sl); ii = np.where(k)[0]; hb = boot(ii, lambda j: H[j].mean()); sb = boot(ii, lambda j: np.median(SW[j]))
            o[sl] = dict(examples=int(k.sum()), clips=int(df.clip_uuid[k].nunique()), human_path_term=hb[0], human_path_term_lo95=hb[1], human_path_term_hi95=hb[2], best_of_32_dist_4s_mean_m=float(D4[k].mean()), best_of_32_dist_4s_median_m=float(np.median(D4[k])),
                         share_with_plan_within_1p5m_4s=float((D4[k] <= 1.5).mean()), swing_median_all_32_plans_mps=sb[0], swing_median_lo95=sb[1], swing_median_hi95=sb[2], share_plans_swing_above_5mps=float((SW[k] > 5).mean()),
                         spacing_sq_mean_m2=float(S[k].mean()), spacing_sq_median_m2=float(np.median(S[k])))
        res["precision"][nm] = o
    k = G["slice_mask"](df, "highway, car >= 20 m/s"); ko = G["slice_mask"](df, "ordinary")
    res["f32_vs_amp_plan_diff_m"] = {s: dict(rows=int(kk.sum()), menu_xy_max=float(df.diff_menu_xy_max_m[kk].max()), menu_xy_median_over_rows=float(df.diff_menu_xy_max_m[kk].median()), four_stages_xy_max=float(df.diff_4stages_xy_max_m[kk].max()), four_stages_xy_median_over_rows=float(df.diff_4stages_xy_max_m[kk].median()))
                                     for s, kk in (("highway, car >= 20 m/s", k), ("ordinary", ko))}
    res["frozen_check"] = G["frozen_check"](); res["gpu_max_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    json.dump(res, open(NASOUT / "t6_readout.json", "w"), indent=1)
    for nm in ("f32", "amp"): log(f"t6 readout {nm}:\n" + pd.DataFrame(res["precision"][nm]).T.round(3).to_string())
    log("f32 vs amp:", json.dumps(res["f32_vs_amp_plan_diff_m"])); log("DONE t6 readout", f"{res['secs']:.0f} s")
main()
