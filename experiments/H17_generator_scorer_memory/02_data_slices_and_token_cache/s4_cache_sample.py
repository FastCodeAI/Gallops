"""Build step 4: token-cache SAMPLE on ~1,000 picking-set moments (whole clips, time order) with 0.1.8b-p06, the one GPU job.

What is cached: the projector's 256-d tokens, 512 per moment (the 16 x 32 grid of the two frames fused as one 2-frame clip,
simple_image_encoder.py:33, 47-49), i.e. element [0] of ImgEncoder.forward's return (feat_flatten, (num_cam=1, 512, B, 256),
simple_image_encoder.py:64-66) -- exactly what pai_model.py's img_hook captures (pai_model.py:152-154) and what the refiner's
deformable cross-attention reads. GridMask is off: the agent is in eval mode (grid_mask.py:23 draws only in training; the check
forward asserts GridMask inactive, train_018b.py:132). Beside each moment a pooled copy, probe.pool_map (gallops memory/probe.py:51,
66-72): global mean + 4 x 8 block means = 33 x 256.

(a) normal forward = pai_model.forward_dump (batch 1, fp32, no previous plan, the check forward); (b) the same forward with the
encoder replaced by a module that returns the cached tokens (fp32 cache, and bf16 cache up-cast to fp32). Compared: pick,
plans, rating logit; graded moments re-graded with the S35 grader (pai_model_b.Grader.grade_menus) for both.
Timing: encoder-only forward at batch 1 fp32 / batch 8 fp32 / batch 8 bf16-autocast; a training-style step (generator forward in
train mode under bf16 autocast + frozen-generator replay + scorer in training mode + backward into generator and scorer) on 32
cached moments, 30 steps after 3 warm-up steps.
Writes: {experiment_store}/cache_sample/<clip>.pt (per clip: tokens fp32 + bf16 + pooled), out/s4_cache_sample.json."""
import os, sys, json, time, hashlib, subprocess
from pathlib import Path
import numpy as np, torch
import pai_common as C
C.setup_env()
import pai_model as PM, pai_data as D, pai_slim, pai_model_b as MB
sys.path.insert(0, "{gallops}/src"); from gallops.memory.probe import pool_map
OUT=Path(os.environ["OUT"]); DST=Path("{experiment_store}/cache_sample"); DST.mkdir(parents=True, exist_ok=True)
P06=Path("{retrain_store}/018b/release/drive_jepa_018b_p06.ckpt"); P06_SHA="3d566eb211b5ce0f556680196a46fc0a74da29091ae62ac588910a48d62b8e66"
def main():
    R={"started": time.strftime("%F %T"), "gpu_before": subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total,utilization.gpu","--format=csv,noheader"], capture_output=True, text=True).stdout.strip()}
    t=time.time(); assert C.sha256_file(P06)==P06_SHA, "p06 sha"; R["p06_sha_verified_s"]=round(time.time()-t,1)
    # --- agent: 0.1.8b-p06 strict-loaded the way 0.1.7's file is (drive_jepa_agent_s10.py:135-137), batch fix on, no LoRA ----------------
    from drive_jepa_agent_s10 import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    import batch_fix as BF
    agent=DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path="", vocab_path=str(C.VOCAB), index_kind="ext", init_checkpoint=str(P06), freeze_encoder=True, scorer_impl="s3", pool_workers=0)
    agent.initialize(); found=BF.install(agent._pad_model); BF.set_fix(True, True, True); agent=agent.cuda().eval()
    m=agent._pad_model; bk=m._backbone; assert not bk.training and not bk.grid_mask.training
    sd=torch.load(P06, map_location="cpu", weights_only=False)["state_dict"]; cur=agent.state_dict()
    assert all(torch.equal(cur[k.replace("agent._pad_model","_pad_model")].cpu(), v) for k,v in sd.items()) and len(sd)==495; del sd
    R["agent"]={"init_checkpoint": str(P06), "tensors": 495, "loaded_equal": "495/495", "batch_fix": {k: len(v) for k,v in found.items()}}
    # --- data: the sample list, through the lane's read-only slim dataset ---------------------------------------------------------------
    import pandas as pd
    S=pd.read_parquet(OUT/"s4_sample_moments.parquet"); toks=list(S.sample_id)
    ds=pai_slim.ReadOnlySlim(cache_path=str(C.CACHE), feature_builders=agent.get_feature_builders(), target_builders=agent.get_target_builders(), zero_accel=True,
                             has_label2=dict(zip(S.sample_id, S.has_label2)), tokens=toks, token_clips=dict(zip(S.sample_id, S.clip_uuid)), slim_root=str(C.SLIM), split_guard=D.make_guard("val"))
    assert ds.tokens==toks
    def items():
        for i in range(len(ds)): yield ds[i]
    # --- hooks: capture the encoder output; a swap module returns cached tokens instead -------------------------------------------------
    cap={}
    def img_hook(mod, inp, out): cap["out"]=out
    bk.register_forward_hook(img_hook)
    class CachedEncoder(torch.nn.Module):
        """Stands in for ImgEncoder: returns (feat_flatten, spatial_shape, level_start_index, kwargs) from a cache (num_cam, 512, B, 256)."""
        def __init__(self, ref): super().__init__(); self.feat=None; self.spatial=ref[1]; self.lsi=ref[2]
        def forward(self, img, len_queue=None, **kwargs): return self.feat, self.spatial, self.lsi, kwargs
    # --- pass (a): normal forward, cache the tokens -----------------------------------------------------------------------------------
    A={}; toks_fp32={}; feats_cpu={}; human={}; mc={}; t_fwd=[]; t_load=time.time()
    for f,tg in items():
        tok=tg["token"]; torch.cuda.synchronize(); t0=time.time(); d=PM.forward_dump(agent, f); torch.cuda.synchronize(); t_fwd.append(time.time()-t0)
        o=cap.pop("out"); ft=o[0]; assert ft.shape==(1,512,1,256) and tuple(o[1].tolist()[0])==(16,32), (ft.shape, o[1])
        toks_fp32[tok]=ft[0,:,0,:].detach().float().cpu(); A[tok]=d; feats_cpu[tok]={k:v for k,v in f.items() if k in ("lidar2img","img_shape","ego_status")}
        human[tok]=tg["trajectory"].numpy(); mc[tok]=tg["metric_cache_path"]
    d2=PM.forward_dump(agent, f); assert np.array_equal(d2["proposals"], d["proposals"]) and np.array_equal(d2["pred_logit"], d["pred_logit"]), "pass (a) not deterministic"
    ref=(None, o[1], o[2]); R["pass_a"]={"moments": len(A), "wall_s": round(time.time()-t_load,1), "forward_s_per_moment_batch1_fp32_incl_gen_scorer": float(np.mean(t_fwd[5:]))}
    # --- write the cache (fp32 + bf16 + pooled) per clip ---------------------------------------------------------------------------------
    bytes_written=0
    for clip, g in S.groupby("clip_uuid", sort=False):
        rec={}
        for tok in g.sample_id:
            x=toks_fp32[tok]; rec[tok]={"fp32": x.clone(), "bf16": x.to(torch.bfloat16), "pooled_fp32": torch.from_numpy(pool_map(x.numpy()).astype(np.float32)), "anchor_t": float(g.set_index("sample_id").loc[tok,"anchor_t"])}
        p=DST/f"{clip}.pt"; torch.save(rec, p); bytes_written+=p.stat().st_size
    R["cache_files"]={"clips": int(S.clip_uuid.nunique()), "bytes": bytes_written, "per_moment_fp32_bytes": 512*256*4, "per_moment_bf16_bytes": 512*256*2, "per_moment_pooled_fp32_bytes": 33*256*4}
    # --- pass (b): the forward from the cache, fp32 and bf16 ------------------------------------------------------------------------------
    swap=CachedEncoder(ref).cuda().eval(); m._backbone=swap
    def from_cache(tok, prec):
        x=toks_fp32[tok]; x=(x.to(torch.bfloat16).float() if prec=="bf16" else x)
        swap.feat=x.cuda()[None,:,None,:]           # (num_cam=1, 512, B=1, 256)
        f=dict(feats_cpu[tok]); f["camera_feature_1"]=torch.zeros(3,256,512); f["camera_feature_2"]=torch.zeros(3,256,512)
        return PM.forward_dump(agent, f)
    def compare(prec):
        pick=[]; dmax=[]; dmean=[]; lmax=[]; Bp={}
        for tok in toks:
            a=A[tok]; b=from_cache(tok, prec); Bp[tok]=b
            pick.append(a["chosen"]==b["chosen"]); dd=np.linalg.norm(a["proposals"][...,:2]-b["proposals"][...,:2], axis=-1); dmax.append(dd.max()); dmean.append(dd.mean())
            lmax.append(np.abs(np.asarray(a["pred_logit"]).reshape(32,-1)[:,-1]-np.asarray(b["pred_logit"]).reshape(32,-1)[:,-1]).max())
        return {"pick_agreement_pct": 100*float(np.mean(pick)), "picks_changed": int(len(pick)-sum(pick)), "plan_diff_max_m": float(np.max(dmax)), "plan_diff_mean_m": float(np.mean(dmean)), "rating_logit_diff_max": float(np.max(lmax)), "rating_logit_diff_mean_of_max": float(np.mean(lmax))}, Bp
    res={}; Bp={}
    for prec in ("fp32","bf16"):
        torch.cuda.synchronize(); t0=time.time(); res[prec], Bp[prec]=compare(prec); res[prec]["wall_s"]=round(time.time()-t0,1)
    m._backbone=bk
    # --- pick grade on the graded moments, (a) vs (b) -----------------------------------------------------------------------------------
    gi=[tok for tok in toks if bool(S.set_index("sample_id").loc[tok,"has_label2"])]; assert all(mc[t] for t in gi)
    grader=MB.Grader(workers=2)
    def pick_grade(dump):
        G=np.stack(grader.grade_menus([mc[t] for t in gi], [dump[t]["proposals"] for t in gi])); ch=np.array([dump[t]["chosen"] for t in gi])
        fin=G[...,-1]; return fin[np.arange(len(gi)), ch], G
    t0=time.time(); ga, Ga=pick_grade(A); grades={"graded_moments": len(gi), "pick_grade_mean_a": float(ga.mean()), "grading_wall_s": round(time.time()-t0,1)}
    for prec in ("fp32","bf16"):
        gb, Gb=pick_grade(Bp[prec]); grades[prec]={"pick_grade_mean_b": float(gb.mean()), "picked_plan_grade_changed": int((ga!=gb).sum()), "max_abs_grade_change_of_pick": float(np.abs(ga-gb).max()), "menu_grade_max_abs_change": float(np.abs(Ga-Gb).max())}
    grader.close(); R["exactness"]=res; R["grades"]=grades
    # --- timing: encoder-only ---------------------------------------------------------------------------------------------------------
    def enc_time(bs, autocast, n=30):
        x=torch.zeros(bs,2,3,256,512, device="cuda"); ts=[]
        with torch.no_grad():
            for i in range(n+3):
                torch.cuda.synchronize(); t0=time.time()
                with (torch.autocast("cuda", dtype=torch.bfloat16) if autocast else torch.autocast("cuda", enabled=False)): bk(x)
                torch.cuda.synchronize(); ts.append(time.time()-t0)
        return float(np.mean(ts[3:])/bs)
    R["timing_encoder_s_per_moment"]={"batch1_fp32": enc_time(1, False), "batch8_fp32": enc_time(8, False), "batch8_bf16": enc_time(8, True), "note": "zeros input; the real cache also reads PNGs from the NAS (pass_a wall time includes that)"}
    # --- timing: training step from the cache (generator + replay + scorer, backward), 32 moments --------------------------------------
    m._backbone=swap; model_params=[p for n,p in m.named_parameters() if n.startswith(PM.GEN_PREFIXES) and not n.endswith(PM.FIXED_SUFFIXES)]+[p for n,p in m.named_parameters() if n.startswith("scorer.")]
    for p in m.parameters(): p.requires_grad_(False)
    for p in model_params: p.requires_grad_(True)
    opt=torch.optim.Adam(model_params, lr=0.0)      # lr 0: weights never move
    heads=m._trajectory_head; bev_refiner=heads[0].Bev_refiner
    def step(batch_toks):
        feats={"camera_feature_1": torch.zeros(32,3,256,512,device="cuda"), "camera_feature_2": torch.zeros(32,3,256,512,device="cuda")}
        for k in ("lidar2img","img_shape","ego_status"): feats[k]=torch.stack([feats_cpu[t][k] for t in batch_toks]).cuda()
        feats["past_ego_simulated_states"]=None; feats["metric_cache"]=None; feats["observation_interval"]=None
        swap.feat=torch.stack([toks_fp32[t] for t in batch_toks],1)[None].cuda()   # (1, 512, 32, 256)
        target=torch.from_numpy(np.stack([human[t] for t in batch_toks])).cuda()
        m.eval(); [mod.train() for mod in (m.hist_encoding, m.init_feature, m._trajectory_head)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out=m(feats); ego=feats["ego_status"][:,-1]
            det={n: p.detach() for n,p in bev_refiner.named_parameters()}
            with torch.no_grad(): bev=m.hist_encoding(ego)[:,None]+m.init_feature.weight[None]
            for prop in out["proposal_list"]: bev=torch.func.functional_call(bev_refiner, det, (prop.detach(), bev, (swap.feat, ref[1], ref[2], {"img_metas": feats})))
            m.scorer.train(); sc=m.scorer(out["proposals"].detach(), bev); m.scorer.eval()
        pl=[q.float() for q in out["proposal_list"]]
        gl=sum(torch.abs(q-target[:,None]).mean() for q in pl)/len(pl)               # stand-in for the lessons: same tensors, a cheap loss
        sl=PM.scorer_loss(sc[0], out["proposals"], target)[0]+sum(s.float().mean() for s in sc[1:3])
        (gl+sl).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    ts=[]
    for i in range(33):
        bt=toks[(i*32)%(len(toks)-32):(i*32)%(len(toks)-32)+32]; torch.cuda.synchronize(); t0=time.time(); step(bt); torch.cuda.synchronize(); ts.append(time.time()-t0)
    R["timing_step_from_cache"]={"steps_timed": 30, "warmup": 3, "s_per_step_32_moments": float(np.mean(ts[3:])), "s_per_step_std": float(np.std(ts[3:])), "note": "generator forward (train mode, bf16 autocast) + frozen-generator replay + scorer (training mode) + backward into generator and scorer; no grading pool, no data loading (tokens in GPU memory)", "gpu_peak_alloc_gb": torch.cuda.max_memory_allocated()/1e9}
    m._backbone=bk
    # --- projections ----------------------------------------------------------------------------------------------------------------
    N={"training": 126981, "picking": 2000, "graded_heldout": 10516, "raw_heldout": int(os.environ.get("N_RAW_HELDOUT", "9600"))}
    spm={"batch1_fp32_full_forward": R["pass_a"]["forward_s_per_moment_batch1_fp32_incl_gen_scorer"], "pass_a_wall_incl_loading": R["pass_a"]["wall_s"]/len(A), **R["timing_encoder_s_per_moment"]}
    spm.pop("note"); proj={}
    for k,n in N.items(): proj[k]={"moments": n, "gpu_hours": {kk: n*v/3600 for kk,v in spm.items()}, "disk_gb": {"fp32": n*512*256*4/1e9, "bf16": n*512*256*2/1e9, "pooled_fp32": n*33*256*4/1e9}}
    tot=sum(N.values()); proj["total"]={"moments": tot, "gpu_hours": {kk: tot*v/3600 for kk,v in spm.items()}, "disk_gb": {"fp32": tot*512*256*4/1e9, "bf16": tot*512*256*2/1e9, "pooled_fp32": tot*33*256*4/1e9}}
    R["projection"]=proj; R["gpu_after"]=subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total,utilization.gpu","--format=csv,noheader"], capture_output=True, text=True).stdout.strip(); R["ended"]=time.strftime("%F %T")
    json.dump(R, open(OUT/"s4_cache_sample.json","w"), indent=1, default=float); print(json.dumps(R, indent=1, default=float)); sys.stdout.flush(); os._exit(0)

if __name__ == "__main__":
    main()
