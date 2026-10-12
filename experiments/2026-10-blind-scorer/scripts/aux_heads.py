"""Run the released checkpoint's training-only auxiliary heads on cache v2 tokens (natives + swerves).

pred_col_agent: MLP(proposal_feature 256 -> 2*40*9), reshaped (8 waypoints, 5 sub-steps, 2 slots, 9);
  slot 0 = at-fault collision agent, slot 1 = TTC agent; last of 9 = validity logit (agent present).
pred_area: MLP(per-waypoint token 256 -> 5*2): per 0.1 s sub-step, [non-drivable, oncoming] logits.
Output, in probe_input order: (T, 64, 4) = max over 40 steps of sigmoid of [col slot0, col slot1, off-road, oncoming].
"""
import numpy as np, torch
from pathlib import Path
CK = "{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"
C = Path("{dj_cache}/v2")
sd = torch.load(CK, map_location="cpu", weights_only=False)["state_dict"]
nn = torch.nn
def mlp(prefix, out):
    m = nn.Sequential(nn.Linear(256, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Linear(1024, out))
    m.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}); return m.eval()
col = mlp("agent._pad_model.scorer.pred_col_agent.mlp.", 720)
area = mlp("agent._pad_model.scorer.pred_area.mlp.", 10)
P = np.load(C / "probe_input.npz")
out, idx = [], 0
torch.set_num_threads(2)
with torch.no_grad():
    for src in sorted(p for p in C.iterdir() if p.is_dir() and not p.name.startswith("__")):
        for f in sorted(src.glob("*.npz")):
            if f.name.endswith(".labels.npz") or ".tmp" in f.name or not f.with_name(f.name.replace(".npz", ".labels.npz")).exists():
                continue
            z = np.load(f)
            tok = np.concatenate([z["native_tokens"], z["vocab_tokens"][:, :32]], 1).astype(np.float32)  # (T,64,8,256)
            T = len(tok)
            assert np.allclose(np.concatenate([z["natives"], z["vocab"][:, :32]], 1), P["plans"][idx:idx + T]), f"order mismatch at {f}"
            t = torch.from_numpy(tok)
            c = col(t.amax(2)).reshape(T, 64, 8, 5, 2, 9)[..., 8]            # validity logits (T,64,8,5,2)
            pc = torch.sigmoid(c).reshape(T, 64, 40, 2).amax(2)             # (T,64,2)
            a = area(t).reshape(T, 64, 8, 5, 2)
            pa = torch.sigmoid(a).reshape(T, 64, 40, 2).amax(2)
            out.append(torch.cat([pc, pa], -1).numpy().astype(np.float32)); idx += T
res = np.concatenate(out); assert len(res) == len(P["plans"])
np.save(C / "aux_heads.npy", res); print("saved", res.shape)
