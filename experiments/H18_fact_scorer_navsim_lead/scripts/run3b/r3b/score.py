"""run 3B score, version T4 of scorer_stats_v2 (5 %, 0.5 m; gamma 0.9), computed from facts (never a head).

    score = sum_k gamma^(k-1) (alive_k progress_k quality_k - P new_violation_k)
alive_k        1 - hit_k                          (hit latched)
progress_k     C_k - C_(k-1), C_k = R_k if m_k >= R_k - tol_k else m_k, / R_8;  tol_k = max(5 % R_k, 0.5 m); m = metres so far,
               R = the expert's metres so far (true in training; the scene head at test); R_8 floored at 0.5 m.
               Nothing to gain (expert under 0.5 m): 1/8 every step.
quality_k      lin(closeness, 1 at >= 2 s, 0 at 0) x lin(cut-in braking, 1 at <= 2, 0 at 6) x lin(|lane offset|, 1 at <= 0.5,
               0 at 2 m) x (1 - 0.5 wrong_way_k) x [NAVSIM: (1 - 0.5 comfort_k)] x max(0, 1 - max(0, m_k - R_k - tol_k) / R_8)
new_violation  first step of hit, off road, red (== 1), stop sign, left route (each counted once)

numpy (true facts, 0/1) and torch (predicted: probabilities and numbers) versions. With probabilities, "first step" is the
increase of the running max (cummax_k - cummax_(k-1)) and alive = 1 - cummax(p_hit); on 0/1 facts both equal the label rule.
check: score_np on the packed rows reproduces the pack's a - P b (scorer_stats_v2 score_v2.ab) exactly."""
import numpy as np
import torch

GAMMA, TOL, FLOOR, P_START = 0.9, (0.05, 0.5), 0.5, 0.108
MAPS = dict(close=(2.0, 0.0), lane=(0.5, 2.0), cutin=(2.0, 6.0), ww=0.5, comfort=0.5)  # scorer_stats_v1 partc.MAPS
PEN = ("hit", "offroad", "red", "stop", "left_route")


def _lin_np(x, one_at, zero_at):
    return np.clip((x - zero_at) / (one_at - zero_at), 0.0, 1.0)


def _first_np(a):
    a = a.astype(bool)
    f = np.zeros(a.shape)
    has = a.any(-1)
    idx = np.where(has)
    f[idx + (a.argmax(-1)[has],)] = 1.0
    return f


def parts_np(f, R, ntg, navsim):
    """f: dict of (n, 8) arrays (true facts; red as int8 1/-1/0, metres per step); R (8,) expert metres so far.
    -> a (n,), b (n,) with score = a - P b."""
    n = len(f["hit"])
    m = np.cumsum(f["metres"].astype(np.float64), -1)
    R = np.asarray(R, np.float64)
    Rd = max(R[-1], FLOOR)
    tl = np.maximum(TOL[0] * R, TOL[1])[None]
    C = np.where(m >= R[None] - tl, R[None], m) / Rd
    qmul = np.maximum(0.0, 1.0 - np.maximum(0.0, m - R[None] - tl) / Rd)
    prog = np.diff(np.c_[np.zeros(n), C], axis=1)
    if ntg:
        prog = np.full((n, 8), 1.0 / 8)
    q = (_lin_np(f["ttc"].astype(np.float64), *MAPS["close"]) * _lin_np(np.nan_to_num(f["cutin_brake"].astype(np.float64), nan=0.0), *MAPS["cutin"])
         * _lin_np(np.abs(f["lane_off"].astype(np.float64)), *MAPS["lane"]) * np.where(f["wrong_way"], MAPS["ww"], 1.0))
    if navsim:
        q = q * np.where(f["comfort_step"], MAPS["comfort"], 1.0)
    g = GAMMA ** np.arange(8)
    alive = ~f["hit"].astype(bool)
    a = (alive * prog * q * qmul * g).sum(1)
    ev = sum(_first_np(f[k] == 1 if k == "red" else f[k]) for k in PEN)
    b = (ev * g).sum(1)
    return a, b


def _lin(x, one_at, zero_at):
    return ((x - zero_at) / (one_at - zero_at)).clamp(0.0, 1.0)


def _first(p):
    c = torch.cummax(p, -1).values
    return c - torch.cat([torch.zeros_like(c[..., :1]), c[..., :-1]], -1)


def score_torch(pr, R, ntg, navsim, P=P_START):
    """pr: dict of (..., n, 8) tensors: probabilities hit offroad red stop left_route wrong_way comfort; numbers metres_sofar,
    ttc (s), cutin_brake, lane_off. R (..., 8) expert metres so far; ntg (...,) bool; navsim (...,) bool.
    -> score (..., n), and (a, b) with score = a - P b."""
    m = pr["metres_sofar"]
    R = R[..., None, :]
    Rd = R[..., -1:].clamp(min=FLOOR)
    tl = torch.maximum(TOL[0] * R, torch.full_like(R, TOL[1]))
    C = torch.where(m >= R - tl, R.expand_as(m), m) / Rd
    qmul = (1.0 - (m - R - tl).clamp(min=0.0) / Rd).clamp(min=0.0)
    prog = C - torch.cat([torch.zeros_like(C[..., :1]), C[..., :-1]], -1)
    prog = torch.where(ntg[..., None, None], torch.full_like(prog, 1.0 / 8), prog)
    q = (_lin(pr["ttc"], *MAPS["close"]) * _lin(pr["cutin_brake"], *MAPS["cutin"]) * _lin(pr["lane_off"].abs(), *MAPS["lane"])
         * (1.0 - (1.0 - MAPS["ww"]) * pr["wrong_way"]))
    q = q * torch.where(navsim[..., None, None], 1.0 - (1.0 - MAPS["comfort"]) * pr["comfort"], torch.ones_like(q))
    g = GAMMA ** torch.arange(8, device=m.device, dtype=m.dtype)
    alive = 1.0 - torch.cummax(pr["hit"], -1).values
    a = (alive * prog * q * qmul * g).sum(-1)
    b = (sum(_first(pr[k]) for k in PEN) * g).sum(-1)
    return a - P * b, a, b


def check(n=300):
    """score_np on packed rows == pack a, b; score_torch on the same 0/1 facts == the same."""
    import pickle
    import zlib
    out = {}
    for ds in ("navsim", "lead"):
        meta = pickle.load(open(f"{data.paddock_data}/run3b/pack/{ds}_meta.pkl", "rb"))["meta"]
        rng = np.random.default_rng(0)
        worst_np, worst_t = 0.0, 0.0
        with open(f"{data.paddock_data}/run3b/pack/{ds}.bin", "rb") as fh:
            for i in rng.choice(len(meta), n, replace=False):
                m = meta[i]
                fh.seek(m["offset"])
                o = pickle.loads(zlib.decompress(fh.read(m["nbytes"])))
                a, b = parts_np(o, o["expert_metres"], o["nothing_to_gain"], ds == "navsim")
                worst_np = max(worst_np, float(np.abs(a - o["a"]).max()), float(np.abs(b - o["b"]).max()))
                t = lambda x: torch.as_tensor(np.asarray(x, np.float64))  # noqa: E731
                pr = dict(hit=t(o["hit"]), offroad=t(o["offroad"]), red=t(o["red"] == 1), stop=t(o["stop"]), left_route=t(o["left_route"]),
                          wrong_way=t(o["wrong_way"]), comfort=t(o["comfort_step"]), metres_sofar=t(np.cumsum(o["metres"].astype(np.float64), -1)),
                          ttc=t(o["ttc"]), cutin_brake=t(o["cutin_brake"]), lane_off=t(o["lane_off"]))
                _, at, bt = score_torch(pr, t(o["expert_metres"]), torch.tensor(o["nothing_to_gain"]), torch.tensor(ds == "navsim"))
                worst_t = max(worst_t, float((at - t(o["a"])).abs().max()), float((bt - t(o["b"])).abs().max()))
        out[ds] = dict(samples=n, max_abs_diff_numpy=worst_np, max_abs_diff_torch=worst_t)
    print("SCORE_CHECK", out, flush=True)
    return out


if __name__ == "__main__":
    check()
