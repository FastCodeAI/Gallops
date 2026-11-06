"""Is the bf16 gap between merged and unmerged tokens rounding? Compares it with bf16 vs fp32 on the same encoder, on merge_pass.py's batch. Reads the saved merge; writes a new file only."""
import copy, json
import wiring as W
L, P, M = W.setup()
import numpy as np, torch
import data as D
OUT = W.RUNS / "merged/p1_05nov_pass_03"
ck = torch.load(W.RUNS / "runs/p1_05nov/save_0014_pass_03/state.pt", map_location="cpu", weights_only=False)
agent = W.build_agent("cuda")
plain = agent.get_submodule(W.ENC_PATH)
merged = copy.deepcopy(plain).eval()
merged.load_state_dict(torch.load(OUT / "merged_encoder_pass_03.pt", map_location="cuda"), strict=True)
unm = copy.deepcopy(plain).eval()
L.wrap(unm, 16, 16)
pr = dict(unm.named_parameters())
with torch.no_grad():
    for n, t in ck["lora"].items():
        pr[n].copy_(t.cuda())
fc = D.FrameCache(); sp = json.load(open(W.ROOT / "splits/clips.json")); it = D.pair_items(sp["heldout"], fc.frames, 1, 1)
x = torch.stack([D.pair_from_arrays(fc.frame(c, i - 1), fc.frame(c, i)) for c, i in (it[k] for k in np.linspace(0, len(it) - 1, 32).astype(int))]).cuda()
@torch.no_grad()
def tok(e, ac):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ac):
        return torch.cat([P.encode(e, x[b:b + 8]).float() for b in range(0, 32, 8)])
def d(a, b):
    e = (a - b).abs(); return dict(largest=round(float(e.max()), 4), mean=round(float(e.mean()), 5), p9999=round(float(torch.quantile(e.flatten()[::37], .9999)), 4))
mf, mb, uf, ub, pf, pb = tok(merged, False), tok(merged, True), tok(unm, False), tok(unm, True), tok(plain, False), tok(plain, True)
r = dict(released_bf16_vs_fp32=d(pb, pf), merged_bf16_vs_fp32=d(mb, mf), unmerged_bf16_vs_fp32=d(ub, uf), merged_vs_unmerged_bf16=d(mb, ub), merged_vs_unmerged_fp32=d(mf, uf), largest_token_value=round(float(mf.abs().max()), 2), cosine_merged_released=round(float(torch.nn.functional.cosine_similarity(mf, pf, dim=-1).mean()), 4), cosine_bf16_merged_unmerged=round(float(torch.nn.functional.cosine_similarity(mb, ub, dim=-1).mean()), 5))
json.dump(r, open(OUT / "merge_check_bf16.json", "x"), indent=1); print(json.dumps(r, indent=1))
