"""Stage run3a-checks: Run 3A checks 1-3 (paddock-ops/run3a/CLAUDE.md, "Checks") on the agent look
(drive_jepa_run3a/agent_look.py). 1 relative transform: synthetic agent, dx, dy at t = 1, 2, 3 s within 1e-4
of an independent numpy computation 2 selection: every waypoint keeps exactly the k nearest at its own time;
slots past the cut are never kept 3 empty list: all slots "none" -> >= 0.99 of the attention on "nothing"
Synthetic inputs only: no data, a CPU is enough. Exits non-zero when a check fails.

Ported from paddock-ops run3a/build/checks123.py; changed: this docstring and the exit code.

    PYTHONPATH=<Run 3A tree>/navsim_v2 python -m gallops.drivejepa_wajepa.run3a_checks"""

import numpy as np
import torch
from navsim.agents.drive_jepa_run3a.agent_look import AgentLook, relative_code, select_agents

torch.manual_seed(0)
ok_all = True

# ---- check 1: one curved ego path (8 waypoints, 0.5 s apart), one agent with constant velocity
T = 8
t = 0.5 * np.arange(1, T + 1)
psi = 0.15 * t  # turning left
xs = np.cumsum(5.0 * np.cos(psi) * 0.5)
ys = np.cumsum(5.0 * np.sin(psi) * 0.5)
poses = np.stack([xs, ys, psi], -1)[None, None]  # (1, 1, 8, 3)
c0, v = np.array([12.0, -3.5]), np.array([4.0, 1.5])
L, W, probs = 4.6, 1.9, np.array([0.9, 0.05, 0.03])
P = torch.tensor(poses, dtype=torch.float64)
code = relative_code(
    P,
    torch.tensor(c0[None, None]),
    torch.tensor(v[None, None]),
    torch.tensor([[L]]),
    torch.tensor([[W]]),
    torch.tensor(probs[None, None]),
    torch.zeros(1, 1, T, 1, dtype=torch.long),
)[0, 0, :, 0].numpy()
worst = 0.0
for ts in (1.0, 2.0, 3.0):
    j = int(round(ts / 0.5)) - 1
    rel = c0 + v * ts - poses[0, 0, j, :2]
    h = poses[0, 0, j, 2]
    dx, dy = np.cos(h) * rel[0] + np.sin(h) * rel[1], -np.sin(h) * rel[0] + np.cos(h) * rel[1]
    e = max(abs(code[j, 0] * 32 - dx), abs(code[j, 1] * 32 - dy))
    worst = max(worst, e)
    print(f"  t={ts:.0f} s: dx {dx:8.4f} vs {code[j, 0] * 32:8.4f}, dy {dy:8.4f} vs {code[j, 1] * 32:8.4f}")
c1 = worst < 1e-4
print(f"CHECK1 relative transform: worst |error| {worst:.2e} m ->", "PASS" if c1 else "FAIL")
ok_all &= c1

# ---- check 2: random batch, 300 slots, brute force
B, Pn, Q, K, cut = 3, 32, 300, 16, 0.5
poses = torch.randn(B, Pn, T, 3) * torch.tensor([15.0, 6.0, 0.3])
centre = torch.rand(B, Q, 2) * 64 - 32
vel = torch.randn(B, Q, 2) * 4
p_none = torch.rand(B, Q)
p_none[1] = 0.9  # sample 1: nearly all past the cut
p_none[1, :5] = 0.1  # ... except 5 slots -> fewer than k, rest masked
idx, valid = select_agents(poses, centre, vel, p_none, K, cut)
bad = 0
past_cut_kept = 0
for b in range(B):
    keep = (p_none[b] < cut).nonzero().flatten()
    for p in range(Pn):
        for j in range(T):
            at = centre[b, keep] + vel[b, keep] * 0.5 * (j + 1)
            d = ((at - poses[b, p, j, :2]) ** 2).sum(-1)
            want = set(keep[torch.argsort(d)[:K]].tolist())
            got = set(idx[b, p, j][valid[b, p, j]].tolist())
            bad += want != got
            past_cut_kept += int((p_none[b, idx[b, p, j][valid[b, p, j]]] >= cut).sum())
n = B * Pn * T
c2 = bad == 0 and past_cut_kept == 0
print(
    f"CHECK2 selection: {n - bad}/{n} waypoints keep exactly the k nearest at their own time; "
    f"slots past the cut kept: "
    f"{past_cut_kept}; sample with 5 live slots keeps {int(valid[1].sum(-1).max())} ->",
    "PASS" if c2 else "FAIL",
)
ok_all &= c2

# ---- check 3: all slots none
look = AgentLook(256, 8, K, cut).eval()
x = torch.randn(B, Pn * T, 256)
feat = torch.randn(B, Q, 256)
agents = {
    "feat": feat,
    "centre": centre,
    "vel": vel,
    "length": torch.full((B, Q), 4.5),
    "width": torch.full((B, Q), 1.9),
    "probs": torch.zeros(B, Q, 3),
    "p_none": torch.ones(B, Q),
}
with torch.no_grad():
    _, att = look(x, poses, agents, return_attn=True)
null_all_none = float(att["w"][..., -1].min())
# also with the cut off (every slot kept) and P(none) = 0.999: the log(1 - P(none) + 1e-4) term alone
look_nocut = AgentLook(256, 8, K, 1.01).eval()
look_nocut.load_state_dict(look.state_dict())
agents["p_none"] = torch.full((B, Q), 0.999)
with torch.no_grad():
    _, att2 = look_nocut(x, poses, agents, return_attn=True)
null_soft = float(att2["w"][..., -1].mean())
c3 = null_all_none >= 0.99
print(
    f"CHECK3 empty list: min attention on nothing {null_all_none:.4f} (all slots past the cut); "
    f"for information, cut off and P(none) = 0.999: mean {null_soft:.4f} ->",
    "PASS" if c3 else "FAIL",
)
ok_all &= c3
print("CHECKS_1_3", "PASS" if ok_all else "FAIL")
raise SystemExit(0 if ok_all else 1)
