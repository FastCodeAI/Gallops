"""Build 5 design note: the ONE measurement (Sid's brief 2026-11-11, parts 2 and 3). Forward only, no training, 20 picking clips
(the build-4 fixed subset = first 20 clips of the picking index). Lives in out/ (no edits to src/). Writes out/build5_measure.json.
Models replayed in order from each clip's first moment, carry and reset-before-every-moment:
  A  version 2 untrained (seed 0, as s8_short.run builds it)            -> part 2
  B  the pilot's carry checkpoint at step 930 (version 1, p00.2.pt)      -> part 3 "collapsed" column
  C  build 4's carry checkpoint at step 930 (version 2, step0930.pt)     -> part 3 "build 4" column
Targets = the 4 x 8 block means of the cached token grid (s6_model.block_means = pooled[:, 1:], verified here on one clip), also a coarse
2 x 4 pooling and the global mean; standardised per (position, dimension) with the 20 clips' statistics (z)."""
import json, os, sys, time, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn
import s6_env as SE
import pai_common as C
C.setup_env()
import s6_model as SM, s6_data as SD, s6_train as ST
OUT = Path("{working_folder}/out")
PILOT = SD.A3 / "runs/pilot_seed0/carry/ckpt/p00.2.pt"; B4 = SD.A3 / "runs/build4_short/ckpt/step0930.pt"
LAGS = (2, 4, 8); N_CLIPS = 20; D = 256; dev = "cuda"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def pcos(M):
    x = torch.nn.functional.normalize(M.float(), dim=-1); c = x @ x.T; n = c.shape[0]
    return float((c.sum() - c.diagonal().sum()) / (n * (n - 1)))


@torch.no_grad()
def replay(model, store, clips):
    model.eval(); rows = []
    for ci, c in enumerate(clips):
        rec = store.load(c); N = rec["tokens"].shape[0]
        tokens = rec["tokens"].to(dev); ego = rec["ego"].to(dev); traj = rec["traj"].to(dev)
        l2i = rec["lidar2img"][None].to(dev); shp = rec["img_shape"][None].to(dev)
        M = model.init_memory(1, dev); chain = 0
        for i in range(N):
            if model.mode == "wiped":
                M = model.init_memory(1, dev)
            cons = i > 0 and abs(rec["anchor_t"][i] - rec["anchor_t"][i - 1] - SD.STEP_S) < 1e-6
            if not cons:
                M = model.init_memory(1, dev); chain = 0
            motion = SM.motion_from_prev_future(traj[i - 1:i]) if cons else torch.zeros(1, 3, device=dev)
            o = model.forward_moment(tokens[i:i + 1], ego[i:i + 1], l2i, shp, M, motion, replay=False)
            Mn = o["M_new"].float()
            rows.append({"ci": ci, "row": i, "chain": chain, "pk": bool(rec["in_picking"][i]), "M": Mn[0].flatten().cpu(), "Mread_norm": float(M.norm()),
                         "Mnew_norm": float(Mn.norm()), "abs_change": float((Mn - M).norm()), "gate": o["gate"][0].float().cpu(),
                         "cos": pcos(M[0]) if float(M.abs().max()) > 0 else float("nan")})
            M = model.step_memory(Mn); chain += 1
    return rows


def health(rows):
    a = [r for r in rows if r["pk"] and r["chain"] >= 4]
    g = torch.stack([r["gate"] for r in a]).mean(0).numpy(); mn = np.mean([r["Mread_norm"] for r in a]); ac = np.mean([r["abs_change"] for r in a])
    return {"moments": len(a), "cos_read_mean": float(np.nanmean([r["cos"] for r in a])), "gate_per_slot": g.tolist(), "gate_min": float(g.min()), "gate_median": float(np.median(g)),
            "gate_max": float(g.max()), "gate_mean": float(g.mean()), "slots_gate_ge_0.05": int((g >= 0.05).sum()), "gate_max_over_min": float(g.max() / g.min()),
            "M_read_norm_mean": float(mn), "abs_change_per_moment": float(ac), "abs_change_median": float(np.median([r["abs_change"] for r in a])),
            "rel_change_per_moment": float(np.mean([r["abs_change"] / r["Mread_norm"] for r in a if r["Mread_norm"] > 0])),
            "norm_over_64": float(mn / 64.0), "abs_change_over_64": float(ac / 64.0)}


def krr(X, Y, folds, lam_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """leave-clips-out kernel ridge (linear kernel on standardised features); R2 = 1 - MSE / MSE(train-mean prediction), over all target dims.
    The ridge strength is the best of the grid on the test folds (same grid for every input): an upper-ish linear probe, not a trained read-out."""
    X = X.to(dev).double(); Y = Y.to(dev).double(); best = None
    for lam in lam_grid:
        se = 0.0; s0 = 0.0; n = 0
        for te in folds:
            tr = ~te; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
            K = Xtr @ Xtr.T / X.shape[1]; ym = Y[tr].mean(0)
            A = torch.linalg.solve(K + lam * K.diagonal().mean() * torch.eye(K.shape[0], device=dev, dtype=torch.float64), Y[tr] - ym)
            P = (Xte @ Xtr.T / X.shape[1]) @ A + ym
            se += float(((P - Y[te]) ** 2).sum()); s0 += float(((ym - Y[te]) ** 2).sum()); n += Y[te].numel()
        r = {"lam": lam, "mse": se / n, "mse_train_mean": s0 / n, "r2": 1 - se / s0}
        if best is None or r["r2"] > best["r2"]:
            best = r
    return best


class ReadOut(nn.Module):
    """the proposed read-out (design note part 1): one question per block position and lag (learned position + lag embedding + a projection of
    the CURRENT block mean, a cached constant), one attention layer into M, LayerNorm, Linear. 403,968 parameters."""

    def __init__(self, n_pos=32, lags=2):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, n_pos, D) * 0.02); self.lag = nn.Parameter(torch.randn(lags, D) * 0.02); self.now = nn.Linear(D, D)
        self.attn = SM.Attn(); self.ln = nn.LayerNorm(D); self.out = nn.Linear(D, D)

    def forward(self, M, z_now, lag_idx):
        q = self.pos + self.lag[lag_idx][:, None] + self.now(z_now)
        return self.out(self.ln(q + self.attn(q, M)))


def main():
    R = {"started": time.strftime("%F %T"), "affinity": sorted(os.sched_getaffinity(0)), "nice": os.nice(0), "gate": C.memory_gate(True)}
    torch.manual_seed(0); np.random.seed(0)
    agent, info = SM.load_agent(); R["agent"] = info; state0 = None
    store = SD.ClipStore("picking"); clips = store.clips[:N_CLIPS]; R["clips"] = clips
    # ---- targets ------------------------------------------------------------------------------------------------------------------
    bm, meta = [], []
    for ci, c in enumerate(clips):
        rec = store.load(c); b = SM.block_means(rec["tokens"].to(dev)).cpu(); bm.append(b)
        if ci == 0:
            raw = torch.load(store.dir / store.by_clip[c].file.iloc[0], map_location="cpu", weights_only=False)
            R["pooled_check"] = {"clip": c, "pooled_shape": list(raw["pooled"].shape), "max_abs_diff_block_means_vs_pooled_1_to_33": float((raw["pooled"][:, 1:].float() - b).abs().max()),
                                 "max_abs_diff_global_mean_vs_pooled_0": float((raw["pooled"][:, 0].float() - rec["tokens"].mean(1)).abs().max())}
    BM = torch.cat(bm).to(dev)                                                    # (n, 32, 256), the replay order
    n = BM.shape[0]; R["moments"] = n
    mu, sd = BM.mean(0, keepdim=True), BM.std(0, keepdim=True) + 1e-6; Z = (BM - mu) / sd
    coarse = BM.reshape(n, 4, 8, D).reshape(n, 2, 2, 4, 2, D).mean((2, 4)).reshape(n, 8, D); Zc = (coarse - coarse.mean(0, keepdim=True)) / (coarse.std(0, keepdim=True) + 1e-6)
    glob = BM.mean(1, keepdim=True); Zg = (glob - glob.mean(0, keepdim=True)) / (glob.std(0, keepdim=True) + 1e-6)
    TG = {"block_means_32": Z, "coarse_8": Zc, "global_1": Zg}
    R["raw_stats"] = {"block_mean_abs_mean": float(BM.abs().mean()), "per_dim_std_mean": float(sd.mean()), "per_dim_mean_abs_mean": float(mu.abs().mean()),
                      "share_of_raw_power_in_the_mean": float((mu ** 2).mean() / (BM ** 2).mean())}
    # ---- the three models, carry and reset -------------------------------------------------------------------------------------------
    def fresh():
        nonlocal state0
        torch.manual_seed(0); m = SM.MemoryModel(agent, "carry", 2).cuda()
        if state0 is None:
            state0 = {k: v for k, v in m.trainable_state().items() if k != "memory"}
        return m
    A = fresh(); R["A_new_params"] = A.new_param_count()
    runs = {}; base = None
    for name, build in (("A_v2_untrained", lambda: A), ("B_pilot_step930", lambda: SM.MemoryModel.load_checkpoint(agent, PILOT)[0]), ("C_build4_step930", lambda: SM.MemoryModel.load_checkpoint(agent, B4)[0])):
        m = build(); assert m.mode == "carry"; R.setdefault("versions", {})[name] = m.version
        for mode in ("carry", "wiped"):
            m.mode = mode; t0 = time.time(); rows = replay(m, store, clips); runs[(name, mode)] = rows
            assert len(rows) == n; log(name, mode, f"replayed {n} moments in {time.time() - t0:.0f} s")
        m.mode = "carry"; base = base or runs[(name, "carry")]
        R.setdefault("health", {})[name] = {"carry": health(runs[(name, "carry")]), "reset": health(runs[(name, "wiped")])}
        log(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in R["health"][name]["carry"].items() if k != "gate_per_slot"}))
    ci = torch.tensor([r["ci"] for r in base]); chain = torch.tensor([r["chain"] for r in base])
    Mx = {k: torch.stack([r["M"] for r in v]) for k, v in runs.items()}
    fold_of = ci % 5
    # ---- part 2: the lesson's size --------------------------------------------------------------------------------------------------
    sl1 = torch.nn.functional.smooth_l1_loss; cosd = lambda a, b: float(1 - torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())
    torch.manual_seed(0); ro = ReadOut().cuda(); R["readout_params"] = int(sum(p.numel() for p in ro.parameters()))
    R["lesson"] = {}; R["probe"] = {}
    for L in LAGS:
        idx = torch.nonzero(chain >= L).flatten(); past = idx - L; assert bool((ci[idx] == ci[past]).all())
        folds = [(fold_of[idx] == f).to(dev) for f in range(5)]
        R["lesson"][f"lag_{L}"] = {"samples": int(len(idx)), "seconds_back": L * 0.5}
        for tname, T in TG.items():
            tgt, now = T[past.to(dev)], T[idx.to(dev)]; zero = torch.zeros_like(tgt); e = {}
            e["zero_prediction"] = {"mse": float(((tgt) ** 2).mean()), "smooth_l1": float(sl1(zero, tgt)), "cosine_distance": None}
            e["copy_current"] = {"mse": float(((now - tgt) ** 2).mean()), "smooth_l1": float(sl1(now, tgt)), "cosine_distance": cosd(now, tgt)}
            if tname == "block_means_32":
                raw_t, raw_n = BM[past.to(dev)], BM[idx.to(dev)]
                e["raw_space"] = {"zero_mse": float((raw_t ** 2).mean()), "mean_mse": float(((raw_t - mu) ** 2).mean()), "copy_current_mse": float(((raw_n - raw_t) ** 2).mean()),
                                  "copy_current_cosine_distance": cosd(raw_n, raw_t), "copy_current_smooth_l1": float(sl1(raw_n, raw_t))}
                Mc = Mx[("A_v2_untrained", "carry")][idx].to(dev).reshape(-1, 16, D); li = torch.full((len(idx),), 0 if L <= 4 else 1, device=dev, dtype=torch.long)
                with torch.no_grad():
                    p = ro(Mc, now, li)
                e["untrained_readout_default_init"] = {"mse": float(((p - tgt) ** 2).mean()), "smooth_l1": float(sl1(p, tgt)), "cosine_distance": cosd(p, tgt), "output_rms": float((p ** 2).mean().sqrt())}
                e["untrained_readout_plus_current_skip"] = {"mse": float(((p + now - tgt) ** 2).mean()), "smooth_l1": float(sl1(p + now, tgt))}
            R["lesson"][f"lag_{L}"][tname] = e
            # linear probes (leave-clips-out): what is linearly recoverable about the target from now / from the memory
            Y = tgt.reshape(len(idx), -1); Xnow = Z[idx.to(dev)].reshape(len(idx), -1); pr = {"now_only": krr(Xnow, Y, folds)}
            for name in ("A_v2_untrained", "B_pilot_step930", "C_build4_step930"):
                for mode in ("carry", "wiped"):
                    Xm = Mx[(name, mode)][idx]
                    Xs = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-6)
                    pr[f"{name}_{'carry' if mode == 'carry' else 'reset'}_memory_only"] = krr(Xm, Y, folds)
                    Xn = (Xnow - Xnow.mean(0)) / (Xnow.std(0) + 1e-6)
                    pr[f"{name}_{'carry' if mode == 'carry' else 'reset'}_now_plus_memory"] = krr(torch.cat([Xn.cpu() / math.sqrt(Xn.shape[1]), Xs / math.sqrt(Xs.shape[1])], 1), Y, folds)
            R["probe"].setdefault(f"lag_{L}", {})[tname] = pr
            log(f"lag {L} {tname}: zero {e['zero_prediction']['mse']:.3f} copy {e['copy_current']['mse']:.3f} | R2 now {pr['now_only']['r2']:.3f} "
                + " ".join(f"{k.split('_')[0]}{'c' if 'carry' in k else 'r'}{'+' if 'plus' in k else ''} {v['r2']:.3f}" for k, v in pr.items() if k != "now_only"))
    # ---- similarity of the targets across time and clips (risk: between-clip similarity) --------------------------------------------
    last = torch.tensor([max(i for i, r in enumerate(base) if r["ci"] == c) for c in range(N_CLIPS)])
    def bc(X):
        x = torch.nn.functional.normalize(X.reshape(X.shape[0], -1), dim=-1); cm = x @ x.T; k = cm.shape[0]
        return float((cm.sum() - cm.diagonal().sum()) / (k * (k - 1)))
    R["between_clip_cosine"] = {"raw_block_means": bc(BM[last.to(dev)]), "standardised_block_means": bc(Z[last.to(dev)]), "raw_global_mean": bc(glob[last.to(dev)]),
                                **{f"memory_{k[0]}_{'carry' if k[1] == 'carry' else 'reset'}": bc(v[last]) for k, v in Mx.items()},
                                **{f"memory_minus_clip_mean_{k[0]}_carry": bc(v[last] - v[last].mean(0)) for k, v in Mx.items() if k[1] == "carry"}}
    tok = Z.reshape(-1, D); ev = torch.linalg.eigvalsh(torch.cov(tok.T.double())).flip(0).clamp(min=0); cum = (ev.cumsum(0) / ev.sum()).cpu().numpy()
    R["token_pca"] = {"tokens": int(tok.shape[0]), "dims_for_50pct": int(np.searchsorted(cum, 0.5) + 1), "dims_for_90pct": int(np.searchsorted(cum, 0.9) + 1), "dims_for_99pct": int(np.searchsorted(cum, 0.99) + 1)}
    # ---- cost of the read-out: forward + backward for one step's 40 predictions (2 windows x (12 at 2 s + 8 at 4 s)) ---------------------
    Mb = torch.randn(40, 16, D, device=dev, requires_grad=True); zb = torch.randn(40, 32, D, device=dev); lb = torch.randint(0, 2, (40,), device=dev); tb = torch.randn(40, 32, D, device=dev)
    for _ in range(5):
        sl1(ro(Mb, zb, lb) + zb, tb).backward()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(100):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = sl1(ro(Mb, zb, lb) + zb, tb)
        l.backward()
    torch.cuda.synchronize(); R["readout_cost"] = {"forward_backward_s_per_step": (time.time() - t0) / 100, "predictions_per_step": 40, "note": "GPU shared with other users' jobs"}
    R["ended"] = time.strftime("%F %T"); C.wj(OUT / "build5_measure.json", R); log("done", json.dumps(R["readout_cost"]), R["token_pca"], R["between_clip_cosine"])


if __name__ == "__main__":
    main(); sys.stdout.flush(); os._exit(0)
