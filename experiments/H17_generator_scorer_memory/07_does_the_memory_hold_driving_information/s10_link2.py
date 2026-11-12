"""Link-2 test (Sid, 2026-11-11): does the trained memory hold information that matters for driving? No training. Two steps here:

  s10_link2.py sample    the primary sample: 300 training clips NOT used in the first 930 steps of the seeded pass-1 plan (the plan the pilot,
                         build 4 and build 5 all ran: s6_data.plan_pass(store, 0, 1, True, mem), s9_short.py:166), stratified by clip type
                         (ordinary raw / reconstructed / pull-out), proportional to the unused pool, fixed seed -> out/link2_sample.{json,csv}
  s10_link2.py replay    forward replays (GPU, batch 1, no autocast, as s8_short.replay_memory) of every sample clip and of the 84 picking
                         clips, in order from the clip's first moment, for each checkpoint, with the SAME weights in two modes:
                           carried: the memory written at t is read at t + 1
                           wiped:   the memory is reset to the start state (M0; zeros in version 1) before EVERY moment
                         Kept per moment: the memory written after the update at t in both modes (16 x 256), the scorer's summary of the memory
                         READ at t in carry mode (256: what the rating head reads at t) and of the memory written at t (information).
                         -> {runs}/generator_scorer_memory/runs/link2/{feats_<slice>.npz, mem_<checkpoint>_<slice>.npz, replay.json}
The probes are in s10_probe.py (CPU)."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent; OUT = HERE.parent / "out"
A3 = Path("{runs}/generator_scorer_memory"); RUN = A3 / "runs" / "link2"
SEED_PLAN, PASS, N_STEPS = 0, 1, 930
SAMPLE_SEED, N_SAMPLE, WARM = 20261111, 300, 4
CKPTS = {"build5": A3 / "runs/build5_short/ckpt/step0930.pt", "build4": A3 / "runs/build4_short/ckpt/step0930.pt",
         "pilot": A3 / "runs/pilot_seed0/carry/ckpt/p00.2.pt"}


def stratum(typ: str, source: str) -> str:
    """clip type as b4b_moments.py CT: pull-out = raw pull-out clips (type B_raw); reconstructed = NuRec clips (source nurec: high-speed A,
    close-pass B_pass, NuRec ordinary); ordinary = the raw ordinary clips."""
    return "pull-out" if typ == "B_raw" else ("reconstructed" if source == "nurec" else "ordinary")


def sample():
    import s6_data as SD
    store = SD.ClipStore("training"); mem = SD.memory_moment_set()
    plan = SD.plan_pass(store, SEED_PLAN, PASS, True, mem); assert plan["steps"] >= N_STEPS
    used = sorted({c for b in plan["batches"][:N_STEPS] for c, _ in b})
    first = store.idx.groupby("clip", sort=False).first()
    n_mom = store.idx.groupby("clip", sort=False).size()
    pool = pd.DataFrame({"clip": first.index, "type": first["type"].values, "source": first["source"].values, "moments": n_mom.loc[first.index].values})
    pool["stratum"] = [stratum(t, s) for t, s in zip(pool["type"], pool["source"])]
    pool["used_first_930"] = pool["clip"].isin(used)
    free = pool[(~pool.used_first_930) & (pool.moments > WARM)].sort_values("clip").reset_index(drop=True)
    counts = free.stratum.value_counts().sort_index(); raw = counts / counts.sum() * N_SAMPLE
    alloc = np.floor(raw).astype(int); rest = N_SAMPLE - int(alloc.sum())
    for k in (raw - alloc).sort_values(ascending=False).index[:rest]:
        alloc[k] += 1
    rng = np.random.RandomState(SAMPLE_SEED); pick = []
    for k in counts.index:
        g = free[free.stratum == k]; pick += list(g.iloc[np.sort(rng.choice(len(g), int(alloc[k]), replace=False))]["clip"])
    sel = free[free["clip"].isin(pick)].reset_index(drop=True); assert len(sel) == N_SAMPLE and not sel["clip"].isin(used).any()
    ev = pd.read_parquet(OUT / "memory_moments_train.parquet"); evc = set(ev.clip_id)
    sel["has_event_moments"] = sel["clip"].isin(evc)
    sel.to_csv(OUT / "link2_sample.csv", index=False)
    info = {"made": time.strftime("%F %T"), "plan": {"seed": SEED_PLAN, "pass": PASS, "first_steps": N_STEPS, "function": "s6_data.plan_pass(store, 0, 1, True, mem)"},
            "training_clips": int(len(pool)), "clips_used_in_first_930_steps": len(used), "unused_pool_clips": int(len(free)),
            "unused_pool_by_stratum": {k: int(v) for k, v in counts.items()}, "sample_seed": SAMPLE_SEED, "sample_by_stratum": {k: int(v) for k, v in alloc.items()},
            "sample_by_type": {k: int(v) for k, v in sel["type"].value_counts().items()}, "sample_moments": int(sel.moments.sum()),
            "sample_clips_with_event_moments": int(sel.has_event_moments.sum()), "clips": list(sel["clip"])}
    (OUT / "link2_sample.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "clips"}, indent=1))


@torch.no_grad()
def replay_clip(model, rec, dev="cuda"):
    """one clip, both modes, batch 1 (the loop of s8_short.replay_memory: restart at a missing anchor, logged motion from the previous
    moment's human future). Returns per moment: chain (moments since the chain start), M written carried / wiped, summaries."""
    import s6_model as SM, s6_data as SD
    N = rec["tokens"].shape[0]; tokens = rec["tokens"].to(dev); ego = rec["ego"].to(dev); traj = rec["traj"].to(dev)
    l2i = rec["lidar2img"][None].to(dev); shp = rec["img_shape"][None].to(dev)
    Mc = model.init_memory(1, dev); chain = 0; out = {k: [] for k in ("chain", "C", "W", "Cs_read", "Cs_next", "bm", "chosen_c", "chosen_w")}
    for i in range(N):
        consecutive = i > 0 and abs(rec["anchor_t"][i] - rec["anchor_t"][i - 1] - SD.STEP_S) < 1e-6
        if not consecutive:
            Mc = model.init_memory(1, dev); chain = 0
        motion = SM.motion_from_prev_future(traj[i - 1:i]) if consecutive else torch.zeros(1, 3, device=dev)
        oc = model.forward_moment(tokens[i:i + 1], ego[i:i + 1], l2i, shp, Mc, motion, replay=False)
        ow = model.forward_moment(tokens[i:i + 1], ego[i:i + 1], l2i, shp, model.init_memory(1, dev), motion, replay=False)
        Mn = oc["M_new"].float()
        out["chain"].append(chain); out["C"].append(Mn[0].flatten().cpu()); out["W"].append(ow["M_new"][0].float().flatten().cpu())
        out["Cs_read"].append(oc["summary"][0].float().cpu()); out["Cs_next"].append(model.sc_read(Mn)[0].float().cpu())
        out["bm"].append(oc["block_means"][0].float().flatten().cpu()); out["chosen_c"].append(int(oc["chosen"][0])); out["chosen_w"].append(int(ow["chosen"][0]))
        assert torch.isfinite(Mn).all() and torch.isfinite(ow["M_new"]).all()
        Mc = Mn; chain += 1
    return {k: (np.asarray(v) if k in ("chain", "chosen_c", "chosen_w") else torch.stack(v).numpy()) for k, v in out.items()}


def replay():
    import pai_common as C
    gate = C.memory_gate(True)                    # first thing: loading the agent can take many minutes on a busy machine
    import s6_env as SE
    C.setup_env()
    import pai_model_b as MB
    SE.patch_grader(MB)
    import s6_model as SM, s6_data as SD
    RUN.mkdir(parents=True, exist_ok=True)
    info_path = RUN / "replay.json"; info = json.loads(info_path.read_text()) if info_path.exists() else {}
    agent, ainfo = SM.load_agent(); info["agent"] = ainfo; info.setdefault("gates", []).append({"at": time.strftime("%F %T"), **gate})
    sel = json.loads((OUT / "link2_sample.json").read_text())["clips"]
    slices = {"primary": (SD.ClipStore("training", keep_clips=4), sel), "picking": (SD.ClipStore("picking", keep_clips=4), None)}
    slices["picking"] = (slices["picking"][0], slices["picking"][0].clips)
    for sl, (store, clips) in slices.items():
        for name, path in CKPTS.items():
            dst = RUN / f"mem_{name}_{sl}.npz"
            if dst.exists():
                print("have", dst.name, flush=True); continue
            model, meta = SM.MemoryModel.load_checkpoint(agent, path); model.eval(); t0 = time.time()
            acc = {k: [] for k in ("C", "W", "Cs_read", "Cs_next", "chain", "bm", "chosen_c", "chosen_w")}; F = {k: [] for k in ("clip", "row", "anchor_t", "ego", "traj", "in_picking", "sample_id")}
            for ci, c in enumerate(clips):
                rec = store.load(c); r = replay_clip(model, rec)
                for k in acc:
                    acc[k].append(r[k])
                n = len(r["chain"]); F["clip"] += [c] * n; F["row"] += list(range(n)); F["anchor_t"].append(rec["anchor_t"]); F["ego"].append(rec["ego"].numpy())
                F["traj"].append(rec["traj"].numpy()); F["in_picking"].append(np.asarray(rec["in_picking"], bool)); F["sample_id"] += list(rec["sample_id"])
                if (ci + 1) % 25 == 0:
                    print(time.strftime("%F %T"), sl, name, f"{ci + 1}/{len(clips)} clips {time.time() - t0:.0f} s ram {C.avail_ram_gb():.0f} GB", flush=True)
                    C.memory_gate(False)
            cat = {k: np.concatenate(v) for k, v in acc.items()}
            fpath = RUN / f"feats_{sl}.npz"
            feats = dict(clip=np.array(F["clip"]), row=np.array(F["row"]), sample_id=np.array(F["sample_id"]), anchor_t=np.concatenate(F["anchor_t"]), ego=np.concatenate(F["ego"]).astype(np.float32),
                         traj=np.concatenate(F["traj"]).astype(np.float64), in_picking=np.concatenate(F["in_picking"]), chain=cat["chain"], bm=cat["bm"].astype(np.float32))
            if fpath.exists():                    # the features do not depend on the checkpoint: check, do not rewrite
                old = np.load(fpath); assert (old["clip"] == feats["clip"]).all() and (old["chain"] == feats["chain"]).all() and np.abs(old["bm"] - feats["bm"]).max() == 0
            else:
                np.savez(fpath, **feats)
            np.savez(dst, C=cat["C"].astype(np.float32), W=cat["W"].astype(np.float32), Cs_read=cat["Cs_read"].astype(np.float32), Cs_next=cat["Cs_next"].astype(np.float32),
                     chosen_c=cat["chosen_c"], chosen_w=cat["chosen_w"])
            info.setdefault("replays", {})[f"{name}_{sl}"] = {"checkpoint": str(path), "version": model.version, "mode_in_checkpoint": meta.get("mode"), "clips": len(clips),
                                                               "moments": int(len(cat["chain"])), "moments_after_warm_up": int((cat["chain"] >= WARM).sum()), "wall_s": round(time.time() - t0, 1),
                                                               "pick_differs_carried_vs_wiped": float((cat["chosen_c"] != cat["chosen_w"]).mean()), "ended": time.strftime("%F %T")}
            info_path.write_text(json.dumps(info, indent=1) + "\n"); print(time.strftime("%F %T"), "saved", dst.name, info["replays"][f"{name}_{sl}"], flush=True)
            del model; torch.cuda.empty_cache()
    print("REPLAYS DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["sample", "replay"]); a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    {"sample": sample, "replay": replay}[a.cmd]()
