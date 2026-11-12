"""Build step 2, part 1: the full fp32 token cache of one slice with 0.1.8b-p06 (the one GPU job; run once per slice).

Cached per moment, exactly as the build-1 sample (s4_cache_sample.py:3-8, 59-60): feat_flatten[0, :, b, :] of ImgEncoder.forward
(simple_image_encoder.py:56-66), (512, 256) fp32 = the projector's tokens of the two frames fused; plus the pooled copy
probe.pool_map (gallops memory/probe.py:66-72), 33 x 256 fp32. Beside them, since the slim file is open anyway: the newest ego-status
row (11 values, the converter's accelerations kept: zero_accel=False here; the encoder never reads ego_status) and the (8, 3) human
future. No measure is computed on anything.

Model loading and the capture hook are s4_cache_sample.py:30-38, 48-50 verbatim. The encoder is called as drive_jepa_model.py:53-59
does (Normalize both frames, stack [older, newer], one backbone call), fp32, eval mode, GridMask asserted off. Before any file is
written the first loader batch is encoded at the encoder batch (ENC_BS) and again one moment at a time, and the two must be bit-identical
(the exactness check compares against a batch-1 full forward); a second pass must reproduce the first (determinism). Either failing -> STOP
(exit 3), nothing cached. First launch, ENC_BS=8 (Sid's batch 8): the batch-8 tokens differ from batch 1 by up to 0.021 (deterministic),
so the job stopped (out/s5_cache_training_STOP.json, logs/s5_cache_training_batch8_STOP.log) and was relaunched with ENC_BS=1, the only
setting under which the cache can reproduce the check forward bit for bit; the loader still delivers batches of 8.

Files: {runs}/generator_scorer_memory/cache/<slice>/<clip>.pt = {"sample_id", "anchor_t", "tokens" (N,512,256), "pooled" (N,33,256),
"ego_status" (N,11), "trajectory" (N,8,3), "has_label2"} in time order; index_<slice>.parquet (sample_id, clip, anchor_t, file, row,
sha256, ...); pooled_<slice>.npy (N, 8448) = every moment's pooled copy in index order (one array per slice).
Resumable: a clip whose file exists and whose sha256 equals the one in index_partial.jsonl is skipped.
Loader: 8 spawned workers, nice 19, all cores (Sid's one-off exception for this job); the main process is not pinned either.
Usage: run_model_cache.sh <abs path>/s5_cache_full.py <training|picking|graded_heldout|raw_heldout>"""
import os, sys, json, time, hashlib, io, subprocess
from pathlib import Path
import numpy as np, torch, pandas as pd
import pai_common as C
C.setup_env()
import pai_model as PM, pai_data as D, pai_slim
sys.path.insert(0, "{gallops}/src"); from gallops.memory.probe import pool_map
OUT=Path(os.environ["OUT"]); A3=Path("{runs}/generator_scorer_memory"); CACHE=A3/"cache"
P06=Path("{runs}/generator_joint_pai/018b/release/drive_jepa_018b_p06.ckpt"); P06_SHA="3d566eb211b5ce0f556680196a46fc0a74da29091ae62ac588910a48d62b8e66"
BS=8; WORKERS=8; PROGRESS_CLIPS=500; ENC_BS=int(os.environ.get("ENC_BS","1"))   # loader batch 8; encoder batch from ENC_BS (see docstring)

def slice_manifest(which):
    """Rows (sample_id, clip_uuid, anchor_t, has_label2, type, source, group, [in_picking]) sorted by clip then time, and the data roots."""
    if which=="training":
        man=pd.read_parquet(C.MANIFEST); cfg=dict(cache=C.CACHE, slim=C.SLIM, split="train")
    elif which=="picking":   # the 84 picking clips as WHOLE clips: every val_full moment of those clips, the picking moments flagged
        vf=pd.read_parquet(C.VAL_FULL); pk=pd.read_parquet("{runs}/generator_joint_pai/picking/picking_set.parquet")
        assert set(pk.sample_id)<=set(vf.sample_id) and pk.clip_uuid.nunique()==84
        man=vf[vf.clip_uuid.isin(set(pk.clip_uuid))].copy(); man["in_picking"]=man.sample_id.isin(set(pk.sample_id)); cfg=dict(cache=C.CACHE, slim=C.SLIM, split="val")
    elif which=="graded_heldout":
        man=pd.read_parquet(A3/"slices/manifest_graded_heldout.parquet"); cfg=dict(cache=C.CACHE, slim=C.SLIM, split="val")
    elif which=="raw_heldout":   # converted by b3_convert.py into A3/raw_heldout/cache/<clip>/<sid>/pad_*.gz; no slim copies exist
        man=pd.read_parquet(A3/"slices/manifest_raw_heldout.parquet"); cfg=dict(cache=A3/"raw_heldout/cache", slim=None, split="val")
    else: raise SystemExit(f"unknown slice {which}")
    if "in_picking" not in man: man["in_picking"]=False
    man=man.sort_values(["clip_uuid","anchor_t"]).reset_index(drop=True); assert man.sample_id.is_unique
    return man, cfg

def collate(b): return b
def worker_init(_):
    os.nice(19)
    try: os.sched_setaffinity(0, range(os.cpu_count()))
    except Exception: pass

def main():
    which=sys.argv[1]; DST=CACHE/which; DST.mkdir(parents=True, exist_ok=True)
    R={"slice": which, "started": time.strftime("%F %T"), "gpu_before": subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total,utilization.gpu","--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
       "loadavg_before": open("/proc/loadavg").read().split()[:3]}
    t=time.time(); assert C.sha256_file(P06)==P06_SHA, "p06 sha"; R["p06_sha_verified_s"]=round(time.time()-t,1)
    # --- agent: s4_cache_sample.py:30-38 verbatim ------------------------------------------------------------------------------------
    from drive_jepa_agent_s10 import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    import batch_fix as BF
    agent=DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path="", vocab_path=str(C.VOCAB), index_kind="ext", init_checkpoint=str(P06), freeze_encoder=True, scorer_impl="s3", pool_workers=0)
    agent.initialize(); found=BF.install(agent._pad_model); BF.set_fix(True, True, True); agent=agent.cuda().eval()
    m=agent._pad_model; bk=m._backbone; assert not bk.training and not bk.grid_mask.training
    sd=torch.load(P06, map_location="cpu", weights_only=False)["state_dict"]; cur=agent.state_dict()
    assert all(torch.equal(cur[k.replace("agent._pad_model","_pad_model")].cpu(), v) for k,v in sd.items()) and len(sd)==495; del sd
    R["agent"]={"init_checkpoint": str(P06), "tensors": 495, "loaded_equal": "495/495", "batch_fix": {k: len(v) for k,v in found.items()}}
    # --- the slice and what is already cached --------------------------------------------------------------------------------------
    man, cfg=slice_manifest(which); n_tot=len(man); clips=list(dict.fromkeys(man.clip_uuid)); expected=man.groupby("clip_uuid").size().to_dict()
    row0={}; off=0
    for c in clips: row0[c]=off; off+=expected[c]
    partial=DST/"index_partial.jsonl"; done={}
    if partial.exists():
        for line in open(partial):
            j=json.loads(line); p=DST/j["file"]
            if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==j["sha256"]: done[j["clip"]]=j
    R["resume"]={"clips_already_cached_and_verified": len(done)}
    todo_clips=[c for c in clips if c not in done]; todo=man[man.clip_uuid.isin(set(todo_clips))]
    print(f"[{which}] {n_tot} moments / {len(clips)} clips; cached+verified {len(done)}; to do {len(todo)} moments / {len(todo_clips)} clips", flush=True)
    def mm(name, shape):
        p=DST/f"{name}_{which}.npy"
        return np.load(p, mmap_mode="r+") if p.exists() else np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=shape)
    pooled_mm=mm("pooled", (n_tot, 33*256))
    # --- loader (spawned workers) and the encoder call --------------------------------------------------------------------------------
    ds=pai_slim.ReadOnlySlim(cache_path=str(cfg["cache"]), feature_builders=agent.get_feature_builders(), target_builders=agent.get_target_builders(), zero_accel=False,
                             has_label2=dict(zip(man.sample_id, man.has_label2)), tokens=list(todo.sample_id), token_clips=dict(zip(man.sample_id, man.clip_uuid)),
                             slim_root=(str(cfg["slim"]) if cfg["slim"] else None), split_guard=D.make_guard(cfg["split"]))
    dl=torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=False, num_workers=WORKERS, multiprocessing_context="spawn", collate_fn=collate, worker_init_fn=worker_init, prefetch_factor=4) if len(ds) else []
    @torch.no_grad()
    def encode(fl):
        cf1=torch.stack([f["camera_feature_1"] for f in fl]).cuda(); cf2=torch.stack([f["camera_feature_2"] for f in fl]).cuda()
        cam=torch.cat([m.transform(cf2)[:,None], m.transform(cf1)[:,None]], 1)        # drive_jepa_model.py:53-57
        o=bk(cam); ft=o[0]; assert ft.shape==(1,512,len(fl),256) and tuple(o[1].tolist()[0])==(16,32)
        return ft[0].permute(1,0,2).contiguous()                                        # (B, 512, 256) = feat_flatten[0, :, b, :]
    # --- the loop --------------------------------------------------------------------------------------------------------------------
    pend={}; written=0; n_mom=0; gpu_s=0.0; bytes_written=sum(j["bytes"] for j in done.values()); t_start=time.time(); checked=False
    at_of=dict(zip(man.sample_id, man.anchor_t)); clip_of=dict(zip(man.sample_id, man.clip_uuid))
    def flush(clip):
        g=pend.pop(clip); order=np.argsort([x["anchor_t"] for x in g]); g=[g[i] for i in order]
        sids=[x["sid"] for x in g]; exp=man[man.clip_uuid==clip]; assert sids==list(exp.sample_id), clip
        toks=torch.stack([x["tok"] for x in g]); pooled=torch.from_numpy(pool_map(toks.numpy()).astype(np.float32)).reshape(len(g),33,256)
        ego=torch.stack([x["ego"] for x in g]); traj=torch.stack([x["traj"] for x in g])
        rec={"sample_id": sids, "anchor_t": torch.tensor([x["anchor_t"] for x in g], dtype=torch.float64), "tokens": toks, "pooled": pooled, "ego_status": ego, "trajectory": traj,
             "has_label2": torch.tensor(list(exp.has_label2))}
        buf=io.BytesIO(); torch.save(rec, buf); b=buf.getvalue(); sha=hashlib.sha256(b).hexdigest(); p=DST/f"{clip}.pt"; tmp=DST/f".{clip}.pt.tmp"
        tmp.write_bytes(b); os.replace(tmp, p)
        r0=row0[clip]; pooled_mm[r0:r0+len(g)]=pooled.reshape(len(g),-1).numpy()
        j={"clip": clip, "file": p.name, "sha256": sha, "n": len(g), "bytes": len(b), "row0": r0}
        with open(partial, "a") as fh: fh.write(json.dumps(j)+"\n")
        done[clip]=j; return len(b)
    for batch in dl:
        fl=[f for f,_ in batch]; tl=[t for _,t in batch]
        def encode_batched(fl): return encode(fl) if ENC_BS>=len(fl) else torch.cat([encode(fl[i:i+ENC_BS]) for i in range(0,len(fl),ENC_BS)])
        torch.cuda.synchronize(); t0=time.time(); X=encode_batched(fl); torch.cuda.synchronize(); gpu_s+=time.time()-t0
        if not checked:   # batch invariance + determinism, before anything is written
            X1=torch.cat([encode([f]) for f in fl]); X8b=encode_batched(fl)
            R["batch_check"]={"encoder_batch": ENC_BS, "loader_batch": len(fl), "batch8_equals_batch1_bitwise": bool(torch.equal(X, X1)), "max_abs_diff_batch8_vs_batch1": float((X-X1).abs().max()), "batch8_deterministic": bool(torch.equal(X, X8b))}
            print("batch check", R["batch_check"], flush=True); checked=True
            if not (R["batch_check"]["batch8_equals_batch1_bitwise"] and R["batch_check"]["batch8_deterministic"]):
                R["stopped"]="batch-8 encoder output is not bit-identical to batch 1 (or not deterministic); nothing cached"; R["ended"]=time.strftime("%F %T")
                json.dump(R, open(OUT/f"s5_cache_{which}_STOP.json","w"), indent=1, default=float); print(json.dumps(R, indent=1, default=float), flush=True); os._exit(3)
        Xc=X.cpu()
        for i,(f,tg) in enumerate(zip(fl,tl)):
            sid=tg["token"]; clip=clip_of[sid]
            pend.setdefault(clip, []).append({"sid": sid, "anchor_t": float(at_of[sid]), "tok": Xc[i], "ego": f["ego_status"][-1].clone(), "traj": tg["trajectory"].clone()})
            n_mom+=1
            if len(pend[clip])==expected[clip]:
                bytes_written+=flush(clip); written+=1
                if written%PROGRESS_CLIPS==0:
                    el=time.time()-t_start; print(f"{time.strftime('%T')} {written}/{len(todo_clips)} clips, {n_mom} moments, {el/60:.1f} min, {n_mom/el:.1f} moments/s, GPU {gpu_s:.0f} s, {bytes_written/1e9:.1f} GB, avail {int(os.popen('free -g').read().split()[12])} GB", flush=True)
    assert not pend, f"incomplete clips: {list(pend)[:3]}"
    # --- index -----------------------------------------------------------------------------------------------------------------------
    rows=[]
    for c in clips:
        j=done[c]; sub=man[man.clip_uuid==c]
        for r,(sid,at,hl,ty,so,gr,ip) in enumerate(zip(sub.sample_id, sub.anchor_t, sub.has_label2, sub.type, sub.source, sub.group, sub.in_picking)):
            rows.append(dict(sample_id=sid, clip=c, anchor_t=float(at), file=j["file"], row=r, sha256=j["sha256"], has_label2=bool(hl), type=ty, source=so, group=gr, in_picking=bool(ip), pooled_row=j["row0"]+r))
    idx=pd.DataFrame(rows); assert len(idx)==n_tot and idx.sample_id.is_unique and set(idx.sample_id)==set(man.sample_id) and (idx.pooled_row.values==np.arange(n_tot)).all()
    ip=DST/f"index_{which}.parquet"; idx.to_parquet(ip, index=False); pooled_mm.flush()
    R["counts"]={"manifest_moments": n_tot, "cached_moments": int(len(idx)), "missing": sorted(set(man.sample_id)-set(idx.sample_id)), "clips": len(clips), "files": len(done)}
    R["disk"]={"clip_files_bytes": int(sum(j["bytes"] for j in done.values())), "pooled_npy_bytes": int((DST/f"pooled_{which}.npy").stat().st_size), "index_bytes": int(ip.stat().st_size), "index_sha256": C.sha256_file(ip)}
    R["timing"]={"wall_s": round(time.time()-t_start,1), "encoder_gpu_s_this_run": round(gpu_s,1), "moments_encoded_this_run": n_mom, "gpu_s_per_moment": (gpu_s/n_mom if n_mom else None), "moments_per_s_wall": (n_mom/(time.time()-t_start) if n_mom else None)}
    R["loader"]={"workers": WORKERS, "context": "spawn", "batch": BS, "encoder_batch": ENC_BS, "nice": 19, "affinity": "all cores (one-off exception)", "zero_accel": False, "slim_root": str(cfg["slim"]), "cache_path": str(cfg["cache"]), "split_guard": cfg["split"]}
    R["gpu_after"]=subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total,utilization.gpu","--format=csv,noheader"], capture_output=True, text=True).stdout.strip(); R["gpu_peak_alloc_gb"]=torch.cuda.max_memory_allocated()/1e9; R["ended"]=time.strftime("%F %T")
    json.dump(R, open(OUT/f"s5_cache_{which}.json","w"), indent=1, default=float); print(json.dumps(R, indent=1, default=float), flush=True); os._exit(0)

if __name__=="__main__":
    main()
