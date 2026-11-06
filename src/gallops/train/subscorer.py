"""The sub-score scorer over cached features: what it reads, how it is built, how it trains.

One definition, used by the leave-one-scene-out folds (`scorer_folds --variant probe-geo`) and by
`scorer_train`, which writes weights paddock's `drivejepa_subscore` adapter loads. That adapter is
not on paddock's `main`: it is copied in `experiments/GC07_subscore_scorer/paddock/`, from paddock's
branch `model/drivejepa-subscore` (027bc9f). The repos do not import each other, so `layout()` is
the contract: the adapter's `build_scorer` must produce exactly these state_dict keys and shapes,
and `tests/test_subscorer.py` pins them.

Per tick the scorer sees 64 candidate plans -- Drive-JEPA's 32 and the 32 swerves `scorer_cache`
injects through a second pass of the refiner's last stage -- and 32 straight probe paths reaching
32-128 m, run through the same pass but never driven:

* `own`   per plan, the max over its 8 last-stage refiner tokens (64 x 256)
* `probe` the 32 probes' 8 waypoint tokens each, probe-major (256 x 256), read by attention
* `geo`   per plan, its (x, y) / 20 m, its 8 headings and the ego speed / 10 m/s (64 x 25)

and answers six questions per plan (`CHANNELS`): no collision, on road, no TTC violation, in lane,
progress, and the final score built from them. The MLP starts from the released scorer's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TOKEN = 256
NATIVE = 32
SWERVES = 32
PLANS = NATIVE + SWERVES
PROBE_PATHS = 32
WAYPOINTS = 8
PROBE_TOKENS = PROBE_PATHS * WAYPOINTS
GEO = 2 * WAYPOINTS + WAYPOINTS + 1
HIDDEN = 1024
HEADS = 4
CHANNELS = ("no_collision", "on_road", "no_ttc", "in_lane", "progress", "final")
#: Sources whose ticks are real nuScenes logs, not rendered scenes.
REAL_SOURCES = ("nusc", "nusc_holdout")
#: Per-plan label arrays `scorer_labels` writes and the targets are built from.
LABEL_KEYS = ("hit5_at", "off_road_at", "ttc_at", "max_deviation", "travel", "human_travel")
LANE_LINE_M = 1.75


def layout() -> dict[str, tuple[int, ...]]:
    """state_dict key -> shape of the module `build` makes. Needs no torch."""
    n_in = TOKEN + TOKEN + GEO
    return {
        "mlp.0.weight": (HIDDEN, n_in),
        "mlp.0.bias": (HIDDEN,),
        "mlp.1.weight": (HIDDEN,),
        "mlp.1.bias": (HIDDEN,),
        "mlp.3.weight": (HIDDEN, HIDDEN),
        "mlp.3.bias": (HIDDEN,),
        "mlp.4.weight": (HIDDEN,),
        "mlp.4.bias": (HIDDEN,),
        "mlp.6.weight": (len(CHANNELS), HIDDEN),
        "mlp.6.bias": (len(CHANNELS),),
        "pos": (PROBE_TOKENS, TOKEN),
        "q.weight": (TOKEN, TOKEN + GEO),
        "q.bias": (TOKEN,),
        "attn.in_proj_weight": (3 * TOKEN, TOKEN),
        "attn.in_proj_bias": (3 * TOKEN,),
        "attn.out_proj.weight": (TOKEN, TOKEN),
        "attn.out_proj.bias": (TOKEN,),
    }


def geometry(plans: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """(T, P, 25) float16: plan (x, y) / 20 m, its headings, and the ego speed / 10 m/s."""
    t, p = plans.shape[:2]
    xy = plans[..., :2].reshape(t, p, 2 * WAYPOINTS) / 20.0
    speed_col = np.repeat(np.asarray(speed, dtype=np.float32)[:, None, None] / 10.0, p, axis=1)
    return np.concatenate([xy, plans[..., 2], speed_col], axis=-1).astype(np.float16)


def targets(d: dict) -> np.ndarray:
    """(T, P, 6) in [0, 1]: the six answers each plan is trained to give."""
    nc = np.isnan(d["hit5_at"]).astype(np.float32)
    dac = np.isnan(d["off_road_at"]).astype(np.float32)
    ttc = np.isnan(d["ttc_at"]).astype(np.float32)
    lk = (d["max_deviation"] <= LANE_LINE_M).astype(np.float32)
    ep = np.clip(d["travel"] / np.maximum(d["human_travel"][:, None], 1e-3), 0, 1).astype(np.float32)
    final = nc * dac * (5 * ep + 5 * ttc + 2 * lk) / 12
    return np.stack([nc, dac, ttc, lk, ep, final], axis=-1)


def channel_weights(source: np.ndarray, real: str) -> np.ndarray:
    """(T, 1, 6) loss weight per answer. `real` says what real nuScenes ticks may teach.

    "all": every answer. "mask": on road, in lane and progress only -- a real log's car ahead is
    almost always a lead car, which teaches that a short braking plan behind it is safe, the
    opposite of the frontal lesson. "weight": every answer at a fifth of a rendered tick's weight.
    """
    w = np.ones((len(source), 1, len(CHANNELS)), dtype=np.float32)
    is_real = np.isin(source, REAL_SOURCES)
    if real == "mask":
        w[is_real, :, 0] = 0.0
        w[is_real, :, 2] = 0.0
        w[is_real, :, 5] = 0.0
    elif real == "weight":
        w[is_real] = 0.2
    elif real != "all":
        raise ValueError(f"real={real!r}; expected all, mask or weight")
    return w


def labelled_files(cache: Path) -> list[Path]:
    """Cache files whose labels exist, in a fixed order: what `reduce` reads."""
    out = []
    for src in sorted(p for p in cache.iterdir() if p.is_dir() and not p.name.startswith("__")):
        for f in sorted(src.glob("*.npz")):
            if f.name.endswith(".labels.npz") or ".tmp" in f.name:
                continue
            if f.with_name(f.name.replace(".npz", ".labels.npz")).exists():
                out.append(f)
    return out


def reduce(cache: Path) -> dict:
    """Every labelled cached tick as arrays, kept in `<cache>/probe_input.npz`.

    Reading a few hundred cache files from the NAS takes minutes; the reduced arrays are what the
    folds and the trainer read. A cache file without its labels is skipped, so a cache still being
    built reduces to what is finished. The kept file is reused only while it covers exactly the
    labelled episodes on disk, and nothing under it is newer: a reduction made mid-build would
    otherwise be read forever after, and one made before a relabel would train on the old labels
    while the preflight, which reads `.labels.npz`, passed.
    """
    reduced = cache / "probe_input.npz"
    files = labelled_files(cache)
    wanted = {f"{f.parent.name}/{f.stem}" for f in files}
    if reduced.exists():
        kept = np.load(reduced)
        inputs = [p for f in files for p in (f, f.with_name(f.name.replace(".npz", ".labels.npz")))]
        newest = max((p.stat().st_mtime_ns for p in inputs), default=0)
        if newest < reduced.stat().st_mtime_ns and set(np.unique(kept["episode"]).tolist()) == wanted:
            return dict(kept)
    acc: dict[str, list] = {
        k: []
        for k in ("own", "probe", "plans", "pdm", "usable", "in_frame", "speed", "timestamp")
        + ("scene", "shape", "episode")
        + LABEL_KEYS
    }
    for f in files:
        z, lab = np.load(f), np.load(f.with_name(f.name.replace(".npz", ".labels.npz")))
        own = np.concatenate([z["native_tokens"], z["vocab_tokens"][:, :SWERVES]], 1).max(axis=2)
        n = len(own)
        acc["own"].append(own)
        acc["probe"].append(z["vocab_tokens"][:, SWERVES:].reshape(n, PROBE_TOKENS, -1))
        acc["plans"].append(np.concatenate([z["natives"], z["vocab"][:, :SWERVES]], 1).astype(np.float32))
        acc["pdm"].append(z["pdm_score"])
        for k in ("usable", "in_frame", "speed", "timestamp"):
            acc[k].append(z[k])
        for k in LABEL_KEYS:
            acc[k].append(lab[k] if k == "human_travel" else lab[k][:, :PLANS])
        shape, scene = f.stem.split("__")[0].split("-")
        acc["scene"].append(np.full(n, scene))
        acc["shape"].append(np.full(n, shape))
        acc["episode"].append(np.full(n, f"{f.parent.name}/{f.stem}"))
    if not acc["own"]:
        raise FileNotFoundError(f"no labelled cache files under {cache}")
    out = {k: np.concatenate(v) for k, v in acc.items()}
    tmp = reduced.with_name(reduced.name.replace(".npz", ".tmp.npz"))
    np.savez(tmp, **out)
    tmp.rename(reduced)
    return out


def source_of(d: dict) -> np.ndarray:
    return np.array([e.split("/")[0] for e in d["episode"]])


def build(torch, released: dict | None = None):
    """The scorer module. `released` (the vendor MLP's state) seeds its MLP: the released
    weights read `own`, and the columns for the probe context and geometry start at zero."""
    nn = torch.nn

    class SubScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(TOKEN + TOKEN + GEO, HIDDEN),
                nn.LayerNorm(HIDDEN),
                nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN),
                nn.LayerNorm(HIDDEN),
                nn.ReLU(),
                nn.Linear(HIDDEN, len(CHANNELS)),
            )
            self.pos = nn.Parameter(torch.zeros(PROBE_TOKENS, TOKEN))
            self.q = nn.Linear(TOKEN + GEO, TOKEN)
            self.attn = nn.MultiheadAttention(TOKEN, HEADS, batch_first=True)
            if released is not None:
                with torch.no_grad():
                    self.mlp[0].weight.zero_()
                    self.mlp[0].weight[:, :TOKEN] = released["0.weight"]
                    self.mlp[0].bias.copy_(released["0.bias"])
                    for i in (1, 3, 4, 6):
                        self.mlp[i].weight.copy_(released[f"{i}.weight"])
                        self.mlp[i].bias.copy_(released[f"{i}.bias"])

        def forward(self, own, probe, geo):
            ctx, _ = self.attn(self.q(torch.cat([own, geo], -1)), probe + self.pos, probe + self.pos)
            return self.mlp(torch.cat([own, ctx, geo], -1))

    return SubScorer()


def fit(model, arrays, train_idx, *, epochs: int, batch: int, lr: float, seed: int, device: str, torch):
    """BCE on the six answers, weighted per answer. `arrays` = (own, probe, geo, targets, weights),
    host tensors; each batch moves to the GPU on its own (the GPU is shared and often near full)."""
    own, probe, geo, y, w = arrays
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    model.train()
    for _ in range(epochs):
        order = rng.permutation(train_idx)
        for s in range(0, len(order), batch):
            b = torch.as_tensor(order[s : s + batch])
            out = model(own[b].to(device).float(), probe[b].to(device).float(), geo[b].to(device).float())
            wb = w[b].to(device)
            per = bce(out, y[b].to(device), reduction="none") * wb
            loss = per.sum() / (wb.sum() * out.shape[1]).clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict(model, arrays, idx, *, device: str, torch, batch: int = 128) -> np.ndarray:
    own, probe, geo = arrays[:3]
    outs = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            b = torch.as_tensor(idx[s : s + batch])
            outs.append(
                model(own[b].to(device).float(), probe[b].to(device).float(), geo[b].to(device).float())
            )
    return torch.cat(outs).cpu().numpy() if outs else np.zeros((0, PLANS, len(CHANNELS)), np.float32)


def formula(s: np.ndarray) -> np.ndarray:
    """The pick score from the six sigmoids: nc^4 * on_road * (5 progress + 5 no_ttc + 2 in_lane) / 12."""
    return s[..., 0] ** 4 * s[..., 1] * (5 * s[..., 4] + 5 * s[..., 2] + 2 * s[..., 3]) / 12


def choose(s: np.ndarray, released_pick: np.ndarray, rule: str, margin: float = 0.2) -> np.ndarray:
    """Which plan drives, per tick, from sigmoids `s` (T, P, 6) and the released pick.

    "absolute": the formula's best when the released pick's no-collision answer is below 0.5.
    "relative": the formula's best when its no-collision answer beats the released pick's by `margin`.
    """
    rows = np.arange(len(s))
    best = formula(s).argmax(1)
    nc_rel = s[rows, released_pick, 0]
    if rule == "absolute":
        switch = nc_rel < 0.5
    elif rule == "relative":
        switch = s[rows, best, 0] - nc_rel > margin
    else:
        raise ValueError(f"rule={rule!r}; expected absolute or relative")
    return np.where(switch, best, released_pick)


def sight_from_cache(d: dict, min_share: float = 0.5) -> tuple[dict, list[str]]:
    """Do the cached features depend on the image? Read off the GT drives, no model needed.

    In a GT-shadow drive the car follows the human's logged path, so at one timestamp every run and
    scenario of a scene has the same pose, speed and history; only the inserted car differs. Where
    that car is in view, the plans' features must differ between runs. Camera-blind they were
    bit-identical across all of them -- the test that found the camera-frame bug.
    """
    gt = np.where(source_of(d) == "gt")[0]
    groups: dict[tuple, list[int]] = {}
    for i in gt:
        groups.setdefault((str(d["scene"][i]), int(d["timestamp"][i])), []).append(int(i))
    shared = [g for g in groups.values() if len(g) > 1]
    in_view = [g for g in shared if d["in_frame"][g].any()]
    if not in_view:
        return {"shared_pose_groups": len(shared)}, [
            "sight: no GT-drive ticks share a pose with the hazard in view, so nothing checks that the "
            "cached features depend on the image; include a GT source"
        ]
    differs = [
        float(
            np.abs(d["own"][g, :NATIVE].astype(np.float32) - d["own"][g[0], :NATIVE].astype(np.float32)).max()
        )
        > 1e-3
        for g in in_view
    ]
    share = float(np.mean(differs))
    report = {
        "shared_pose_groups": len(shared),
        "in_view_groups": len(in_view),
        "features_differ": round(share, 3),
    }
    if share < min_share:
        return report, [
            f"sight: features differ between runs at the same pose on only {share:.0%} of in-view "
            f"GT groups (need {min_share:.0%}) -- the image is not reaching the cached features"
        ]
    return report, []
