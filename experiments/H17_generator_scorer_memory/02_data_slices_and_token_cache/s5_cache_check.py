"""Build step 2, part 1, after the cache: exactness on a sample from EACH slice, 200 moments as whole clips (the first clips of the
index until >= 200 moments). (a) the full forward = pai_model.forward_dump (batch 1, fp32, the check forward, as s4_cache_sample.py:58);
(b) the same forward with the encoder replaced by the cached tokens (s4_cache_sample.py:51-54, 73-78). Also: the tokens the batch-1
forward produces are compared bit for bit with the cached tokens (encoder batch 1).
training, picking: pick agreement %, max plan difference (m, xy over 32 x 8 poses), max logit difference (all logits). Must be 100 / 0 / 0.
graded_heldout, raw_heldout: only "identical: yes/no" for plans and logits; no grade, no ADE, no pick quality is computed or stored.
Usage: run_model_cache.sh <abs path>/s5_cache_check.py   (writes out/s5_cache_check.json)"""
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, pandas as pd
import pai_common as C
C.setup_env()
import pai_model as PM, pai_data as D, pai_slim
from s5_cache_full import slice_manifest, collate, worker_init, P06, P06_SHA, CACHE
OUT=Path(os.environ["OUT"]); N_SAMPLE=200; SLICES=("training","picking","graded_heldout","raw_heldout")

def main():
    R={"started": time.strftime("%F %T")}; assert C.sha256_file(P06)==P06_SHA, "p06 sha"
    from drive_jepa_agent_s10 import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    import batch_fix as BF
    agent=DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path="", vocab_path=str(C.VOCAB), index_kind="ext", init_checkpoint=str(P06), freeze_encoder=True, scorer_impl="s3", pool_workers=0)
    agent.initialize(); BF.install(agent._pad_model); BF.set_fix(True, True, True); agent=agent.cuda().eval()
    m=agent._pad_model; bk=m._backbone; assert not bk.training and not bk.grid_mask.training
    cap={}
    def img_hook(mod, inp, out): cap["out"]=out
    bk.register_forward_hook(img_hook)
    class CachedEncoder(torch.nn.Module):
        def __init__(self, ref): super().__init__(); self.feat=None; self.spatial=ref[1]; self.lsi=ref[2]
        def forward(self, img, len_queue=None, **kwargs): return self.feat, self.spatial, self.lsi, kwargs
    for which in SLICES:
        t0=time.time(); DST=CACHE/which; idx=pd.read_parquet(DST/f"index_{which}.parquet"); man, cfg=slice_manifest(which)
        clips=[]; n=0
        for c in dict.fromkeys(idx["clip"]):
            clips.append(c); n+=int((idx["clip"]==c).sum())
            if n>=N_SAMPLE: break
        sub=idx[idx["clip"].isin(clips)].sort_values(["clip","row"]); toks=list(sub.sample_id)
        cached={}
        for c in clips:
            rec=torch.load(DST/f"{c}.pt", map_location="cpu", weights_only=False)
            for r,sid in enumerate(rec["sample_id"]): cached[sid]=rec["tokens"][r]
        ds=pai_slim.ReadOnlySlim(cache_path=str(cfg["cache"]), feature_builders=agent.get_feature_builders(), target_builders=agent.get_target_builders(), zero_accel=True,
                                 has_label2=dict(zip(man.sample_id, man.has_label2)), tokens=toks, token_clips=dict(zip(man.sample_id, man.clip_uuid)),
                                 slim_root=(str(cfg["slim"]) if cfg["slim"] else None), split_guard=D.make_guard(cfg["split"]))
        dl=torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=8, multiprocessing_context="spawn", collate_fn=collate, worker_init_fn=worker_init, prefetch_factor=4)
        A={}; feats={}; tok_equal=0; tok_maxdiff=0.0; ref=None
        for batch in dl:
            f,tg=batch[0]; sid=tg["token"]; m._backbone=bk; d=PM.forward_dump(agent, f); o=cap.pop("out"); ft=o[0][0,:,0,:].detach().float().cpu(); ref=(None,o[1],o[2])
            tok_equal+=int(torch.equal(ft, cached[sid])); tok_maxdiff=max(tok_maxdiff, float((ft-cached[sid]).abs().max())); A[sid]=d
            feats[sid]={k:v for k,v in f.items() if k in ("lidar2img","img_shape","ego_status")}
        swap=CachedEncoder(ref).cuda().eval(); m._backbone=swap
        pick=[]; dplan=[]; dlog=[]; same_plan=[]; same_logit=[]
        for sid in toks:
            swap.feat=cached[sid].cuda()[None,:,None,:]; f=dict(feats[sid]); f["camera_feature_1"]=torch.zeros(3,256,512); f["camera_feature_2"]=torch.zeros(3,256,512)
            b=PM.forward_dump(agent, f); a=A[sid]
            pick.append(a["chosen"]==b["chosen"]); dplan.append(float(np.linalg.norm(a["proposals"][...,:2]-b["proposals"][...,:2], axis=-1).max())); dlog.append(float(np.abs(np.asarray(a["pred_logit"])-np.asarray(b["pred_logit"])).max()))
            same_plan.append(np.array_equal(a["proposals"], b["proposals"])); same_logit.append(np.array_equal(a["pred_logit"], b["pred_logit"]))
        m._backbone=bk
        r={"clips": len(clips), "moments": len(toks), "cached_tokens_equal_batch1_tokens": f"{tok_equal}/{len(toks)}", "cached_tokens_max_abs_diff": tok_maxdiff, "wall_s": round(time.time()-t0,1)}
        if which in ("training","picking"):
            r.update({"pick_agreement_pct": 100*float(np.mean(pick)), "picks_changed": int(len(pick)-sum(pick)), "plan_diff_max_m": float(max(dplan)), "logit_diff_max": float(max(dlog)), "plans_identical": all(same_plan), "logits_identical": all(same_logit)})
            r["pass"]=bool(r["pick_agreement_pct"]==100.0 and r["plan_diff_max_m"]==0.0 and r["logit_diff_max"]==0.0)
        else:   # held-out: identical yes/no only
            r.update({"plans_identical": "yes" if all(same_plan) else "no", "logits_identical": "yes" if all(same_logit) else "no"})
            r["pass"]=bool(all(same_plan) and all(same_logit))
        R[which]=r; print(which, json.dumps(r), flush=True)
    R["all_pass"]=all(R[w]["pass"] for w in SLICES); R["ended"]=time.strftime("%F %T")
    json.dump(R, open(OUT/"s5_cache_check.json","w"), indent=1, default=float); print(json.dumps(R, indent=1, default=float), flush=True); os._exit(0 if R["all_pass"] else 4)

if __name__=="__main__":
    main()
