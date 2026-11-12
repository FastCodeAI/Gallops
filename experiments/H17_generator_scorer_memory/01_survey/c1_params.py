import torch, collections, json, time, os, sys
P="{runs}/generator_joint_pai/018b/release/drive_jepa_018b_p06.ckpt"
t0=time.time(); sd=torch.load(P, map_location="cpu", weights_only=False, mmap=True)
sd=sd.get("state_dict", sd); print("loaded", len(sd), "tensors in %.0f s"%(time.time()-t0)); 
sd={k.replace("agent._pad_model.","",1):v for k,v in sd.items()}
groups=collections.OrderedDict([("encoder (_backbone.img_backbone)", lambda k:k.startswith("_backbone.img_backbone")), ("encoder projector", lambda k:k.startswith("_backbone.projector")),
 ("generator: hist_encoding", lambda k:k.startswith("hist_encoding")), ("generator: init_feature (queries)", lambda k:k.startswith("init_feature")),
 ("generator: refiner traj_decoder MLP", lambda k:k.startswith("_trajectory_head.0.traj_decoder")), ("generator: refiner Bev_refiner (TSA+SCA+FFN+pos)", lambda k:k.startswith("_trajectory_head.0.Bev_refiner")),
 ("generator: refiner copies 1-3 (shared weights, duplicate keys)", lambda k:k.startswith("_trajectory_head.") and not k.startswith("_trajectory_head.0.")),
 ("scorer: pred_score (rating head)", lambda k:k.startswith("scorer.pred_score")), ("scorer: pred_col_agent (collision head)", lambda k:k.startswith("scorer.pred_col_agent")), ("scorer: pred_area (drivable-area head)", lambda k:k.startswith("scorer.pred_area")), ("other", lambda k:True)])
tot=collections.OrderedDict((g,[0,0]) for g in groups)
for k,v in sd.items():
    for g,f in groups.items():
        if f(k): tot[g][0]+=v.numel(); tot[g][1]+=1; break
for g,(n,c) in tot.items(): print(f"{g:60s} {n:>12,d} params  {c:4d} tensors")
print("total", sum(n for n,_ in tot.values()), "unique (refiner once):", sum(n for g,(n,_) in tot.items() if "copies" not in g))
keys=["hist_encoding.weight","init_feature.weight","_trajectory_head.0.traj_decoder.mlp.0.weight","_trajectory_head.0.Bev_refiner.positional_encoding.row_embed.weight","_trajectory_head.0.Bev_refiner.bev_decoder.layers.0.attentions.0.sampling_offsets.weight","_trajectory_head.0.Bev_refiner.bev_decoder.layers.0.attentions.0.value_proj.weight","_trajectory_head.0.Bev_refiner.bev_decoder.layers.0.attentions.1.deformable_attention.sampling_offsets.weight","scorer.pred_score.mlp.6.weight","scorer.pred_col_agent.mlp.6.weight","scorer.pred_area.mlp.6.weight","_backbone.projector.weight"]
for k in keys: print(k, tuple(sd[k].shape) if k in sd else "MISSING")
print([k for k in sd if "Bev_refiner" in k and ".0." in k][:40])
