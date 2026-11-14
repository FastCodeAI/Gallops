"""H15 B-2: what each trained read attends to on navtest -- a diagnostic, which chooses nothing.

    <[python] drivejepa> experiments/H15_perception_sidecar/scripts/b2_attention.py \
        --runs b2-1-O-s0,b2-1-D-s0,b2-1-HD-s0,b2-1-H6-s0,b2-1-H3-s0 [--rows 2048]

The adapter's read (`train_b2.SlotRead`) gives each of the refiner's 256 queries (32 proposals x 8 poses)
attention over the arm's tokens and a learned "no agent" token, on every pass. Here each run is restored as
its dumps restore it and run on a fixed, seeded sample of navtest rows; the read's weights (the mean of its
four heads) are kept and summed:
- the share on the null token, by pass;
- per token, the attention averaged over the queries, binned by the token's confidence and its distance from
  the ego, as mass, against the mass the bin would get if attention were uniform over the tokens it may read
  and the null (a ratio of 1.0 means "as if uniform"); for H6 and H3, the same for the slots the decoded list
  keeps (confidence >= 0.5);
- near the query: the share of a query's attention on tokens within NEAR_M of its own pose (the token at t,
  the pose at its time), against that share under uniform attention.
Agent arms (O, D, HD) read only the agents they keep; the hidden arms (H6, H3) read all 128 slots.

Writes {runs}/drivejepa_perception/part-b/b2/attention.json, one entry per run, added to what is there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("GALLOPS_CONFIG", str(REPO / "gallops.toml"))

import numpy as np  # noqa: E402

CONF_BINS = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0001)
DIST_BINS = (0.0, 10.0, 20.0, 30.0, 50.0, 1e9)
NEAR_M = 5.0
SAMPLE_SEED = 0


class Capture:
    """Stands in for a SlotRead's forward: computes the read's attention weights exactly as the read does
    (`train_b2.SlotRead.forward`), keeps their mean over heads, then runs the read itself."""

    def __init__(self, read, sink: list):
        self.read, self.original, self.sink = read, read.forward, sink
        read.forward = self

    def __call__(self, query, tokens, bias, keep):
        import torch

        read = self.read
        b, k, _ = query.shape
        tok = torch.cat([tokens, read.null.expand(b, 1, -1)], dim=1)
        kp = torch.cat([keep, torch.ones(b, 1, dtype=torch.bool, device=keep.device)], dim=1)
        bs = torch.cat([bias, torch.zeros(b, k, 1, device=bias.device, dtype=bias.dtype)], dim=-1)
        h, dh = read.heads, read.d // read.heads
        q = read.q(query).view(b, k, h, dh).transpose(1, 2)
        kk = read.k(tok).view(b, -1, h, dh).transpose(1, 2)
        logits = q @ kk.transpose(-1, -2) / dh**0.5 + bs[:, None] * read.bias_scale[None, :, None, None]
        logits = logits.masked_fill(~kp[:, None, None, :], float("-inf"))
        self.sink.append(logits.softmax(dim=-1).mean(1))
        return self.original(query, tokens, bias, keep)


class Spy:
    """Stands in for an adapter in the pass hook: keeps the proposals each pass hands it, then runs it."""

    def __init__(self, inner, sink: list):
        self.inner, self.sink = inner, sink

    def __call__(self, proposals, queries):
        self.sink.append(proposals)
        return self.inner(proposals, queries)


def summarise(weights, ctx: dict, proposals, arm: str) -> dict:
    """One pass of one chunk: weights (b, k, T + 1) over the tokens and the null; proposals (b, 32, 8, 3)."""
    w = weights.detach().float().cpu().numpy()
    b, k, t1 = w.shape
    tokens = w[..., :-1]
    agents = arm in ("O", "D", "HD")
    keep = ctx["mask"].cpu().numpy() if agents else np.ones((b, t1 - 1), bool)
    conf = ctx["conf"].cpu().numpy()
    centre = (ctx["xyz"] if agents else ctx["centre"]).cpu().numpy()[..., :2]
    per_token = tokens.mean(1)  # (b, T): each token's attention, averaged over the queries
    uniform = keep / (keep.sum(-1, keepdims=True) + 1.0)  # each token's share if attention were uniform
    dist = np.linalg.norm(centre, axis=-1)
    poses = proposals.reshape(b, -1, 3)[..., :2].detach().float().cpu().numpy()  # (b, k, 2)
    near = np.linalg.norm(poses[:, :, None, :] - centre[:, None, :, :], axis=-1) < NEAR_M
    near &= keep[:, None, :]
    out = {
        "rows": int(b),
        "null": float(w[..., -1].mean()),
        "near_share": float((tokens * near).sum(-1).mean()),
        "near_uniform": float((near.sum(-1) / (keep.sum(-1)[:, None] + 1.0)).mean()),
        "kept_per_row": float(keep.sum(-1).mean()),
    }
    for name, values, edges in (("conf", conf, CONF_BINS), ("dist", dist, DIST_BINS)):
        mass, count, base = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):  # noqa: B905 -- Python 3.9
            m = keep & (values >= lo) & (values < hi)
            mass.append(float((per_token * m).sum()))
            count.append(int(m.sum()))
            base.append(float((uniform * m).sum()))
        out[f"{name}_mass"], out[f"{name}_count"], out[f"{name}_uniform"] = mass, count, base
    if not agents:
        listed = conf >= 0.5
        out["listed_mass"] = float((per_token * listed).sum())
        out["listed_count"] = int(listed.sum())
        out["listed_uniform"] = float((uniform * listed).sum())
    return out


def merge(parts: list) -> dict:
    """One pass's chunks into one record: the shares averaged by rows, the masses and counts summed."""
    rows = sum(p["rows"] for p in parts)
    out = {"rows": rows}
    for key in ("null", "near_share", "near_uniform", "kept_per_row"):
        out[key] = sum(p[key] * p["rows"] for p in parts) / rows
    for key, first in parts[0].items():
        if key.endswith(("_mass", "_count", "_uniform")) and key != "near_uniform":
            if isinstance(first, list):
                out[key] = [sum(p[key][i] for p in parts) for i in range(len(first))]
            else:
                out[key] = sum(p[key] for p in parts)
    return out


def main() -> None:
    import torch

    from gallops.config import SiteConfig
    from gallops.drivejepa_perception import env, train_b2, train_refiner

    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="comma-separated run names under part-b/b2/runs, or paths")
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--out", default="", help="default: part-b/b2/attention.json")
    a = p.parse_args()
    cfg = SiteConfig.load()
    out_root = Path(cfg.resolve("{runs}/drivejepa_perception"))
    root = out_root / "part-b/b2/runs"
    out = Path(a.out) if a.out else out_root / "part-b/b2/attention.json"
    record = json.loads(out.read_text()) if out.exists() else {}
    entered = env.enter(cfg)
    device = torch.device("cuda")
    sources = {}
    for given in a.runs.split(","):
        run = Path(given) if "/" in given else root / given
        meta, model, hook = train_b2.restore(entered, run, device)
        arm = meta["args"]["arm"]
        if arm == "C":
            continue
        if arm not in sources:
            sources[arm] = train_b2.Source(out_root, "navtest", arm)
        src = sources[arm]
        rows = np.sort(np.random.default_rng(SAMPLE_SEED).choice(len(src.tokens), a.rows, replace=False))
        weights, proposals = [], []
        adapters = [hook.adapter] + ([hook.pass4] if hook.pass4 is not None else [])
        for adapter in adapters:
            Capture(adapter.read, weights)
        hook.adapter = Spy(hook.adapter, proposals)
        if hook.pass4 is not None:
            hook.pass4 = Spy(hook.pass4, proposals)
        per_pass = {k: [] for k in range(1, 5)}
        with torch.no_grad():
            for i in range(0, len(rows), train_b2.BATCH):
                chunk = rows[i : i + train_b2.BATCH]
                weights.clear()
                proposals.clear()
                ctx = src.context(chunk, device)
                for adapter in adapters:
                    adapter.context = ctx
                hook.k = 0
                train_refiner.forward(model, src.data, chunk, device)
                for k, (w, props) in enumerate(zip(weights, proposals), start=1):  # noqa: B905 -- Python 3.9
                    per_pass[k].append(summarise(w, ctx, props, arm))
        record[run.name] = {
            "arm": arm,
            "recipe": meta["args"]["recipe"],
            "stage": meta["stage"],
            "sample_rows": int(len(rows)),
            "passes": {str(k): merge(v) for k, v in per_pass.items() if v},
            "conf_bins": list(CONF_BINS),
            "dist_bins": list(DIST_BINS),
            "near_m": NEAR_M,
        }
        print(run.name, json.dumps({k: round(v["null"], 3) for k, v in record[run.name]["passes"].items()}))
        del model, hook
        torch.cuda.empty_cache()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
