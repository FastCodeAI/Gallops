"""What the frozen scorer picks, run by run: where the aux runs' gain in the picked plan comes from.

    <drive-jepa python> -m gallops.drivejepa_aux.diagnose --cache <cache> --epoch 10 <run dir> ... -o <md>

The heads leave the 32 proposals about as good as planning alone does (best-of-32 ADE: the report),
yet move the plan the frozen scorer picks. That has two readings, and they predict different
things: the scorer reads better features and picks better, or it has drifted onto a degenerate rule
-- "the slowest proposal" lowers L2 and collisions on a log where the release overshoots, and would
crawl in closed loop. Per run, on the held-out rows, at a fixed epoch (the release first):

- along-track error at 3 s, the picked plan's forward distance minus the human's: mean, median, and
  apart for the keyframes where the human is stopped (ego speed under 0.5 m/s) and moving;
- the mean lateral error at 3 s, and the pick's regret: its ADE over 3 s minus the best of 32's;
- how often the pick is the slowest of the 32 (by forward distance at 3 s) or the fastest, its mean
  rank by that distance (0 the slowest, 31 the fastest), how often it is the one nearest the human,
  how many distinct indices are picked and the share of the most common, and the scorer's top score.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np

from gallops.config import SiteConfig
from gallops.drivejepa_aux import cache, split

#: A keyframe whose ego speed is under this counts as the human standing still.
STOPPED_MPS = 0.5
#: Waypoint 5 is at 3 s.
AT_3S = 5


def picks(
    proposals: np.ndarray, pick: np.ndarray, top: np.ndarray, human: np.ndarray, speed: np.ndarray
) -> dict:
    """The table's row for one model: `proposals` (N, 32, 8, 3), `pick` (N,), the scorer's top score (N,),
    the human's plan (N, 8, 3) and the ego speed (N,)."""
    rows = np.arange(len(pick))
    plan = proposals[rows, pick]
    along = plan[:, AT_3S, 0] - human[:, AT_3S, 0]
    progress = proposals[:, :, AT_3S, 0]
    rank = (progress < progress[rows, pick][:, None]).sum(1)
    ade = np.linalg.norm(proposals[..., : AT_3S + 1, :2] - human[:, None, : AT_3S + 1, :2], axis=-1).mean(-1)
    stopped = speed < STOPPED_MPS
    counts = np.bincount(pick, minlength=proposals.shape[1]) / len(pick)
    return {
        "along_3s_mean_m": float(along.mean()),
        "along_3s_median_m": float(np.median(along)),
        "along_3s_stopped_m": float(along[stopped].mean()),
        "along_3s_moving_m": float(along[~stopped].mean()),
        "stopped_rows": int(stopped.sum()),
        "lateral_3s_m": float(np.abs(plan[:, AT_3S, 1] - human[:, AT_3S, 1]).mean()),
        "regret_ade_3s_m": float((ade[rows, pick] - ade.min(1)).mean()),
        "slowest": float((rank == 0).mean()),
        "fastest": float((rank == proposals.shape[1] - 1).mean()),
        "mean_rank": float(rank.mean()),
        "nearest_human": float((pick == ade.argmin(1)).mean()),
        "distinct_picks": int((counts > 0).sum()),
        "most_common_pick": float(counts.max()),
        "top_score": float(top.mean()),
    }


def table(results: dict) -> str:
    header = (
        "| model | along-track error at 3 s: mean / median (m) | human stopped / moving (m) | lateral (m) "
        "| regret over best of 32 (m) | picks the slowest | mean speed rank /31 "
        "| picks the nearest the human "
        "| distinct picks | most common | top score |"
    )
    out = [header, "|" + "---|" * 11]
    for name, r in results.items():
        out.append(
            f"| {name} | {r['along_3s_mean_m']:+.2f} / {r['along_3s_median_m']:+.2f} | "
            f"{r['along_3s_stopped_m']:+.2f} / {r['along_3s_moving_m']:+.2f} | {r['lateral_3s_m']:.2f} | "
            f"{r['regret_ade_3s_m']:.2f} | {r['slowest']:.1%} | {r['mean_rank']:.1f} | "
            f"{r['nearest_human']:.1%} | "
            f"{r['distinct_picks']} | {r['most_common_pick']:.1%} | {r['top_score']:.3f} |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="run directories ({runs}/drivejepa_aux/runs/<name>)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("-o", "--out", type=Path, help="write the markdown table here (default: print it)")
    args = ap.parse_args()
    runs = [r.resolve() for r in args.runs]
    out = args.out.resolve() if args.out else None

    from gallops.drivejepa_aux import step

    release = step.enter(SiteConfig.load())
    import torch

    data = cache.load(args.cache.resolve())
    _, held = split.partition(data["labels"]["scene"], split.load()["held_out"])
    held = held[data["inputs"]["built"][held]]
    human, speed = data["inputs"]["target"][held], np.linalg.norm(data["inputs"]["ego"][held, 3:5], axis=1)
    spatial = torch.as_tensor(data["inputs"]["spatial"], device=args.device)
    level = torch.as_tensor(data["inputs"]["level"], device=args.device)
    with contextlib.redirect_stdout(io.StringIO()):
        _, model, scene_heads, _ = step.build_model(release, args.device, aux=True, scene=True)
    base = step.trainable_state(model, scene_heads)

    def measure() -> dict:
        proposals, pick, top = [], [], []
        with torch.no_grad():
            for start in range(0, len(held), 128):
                fed = cache.batch(data, held[start : start + 128], args.device, with_labels=False)
                result = step.cached_forward(model, None, fed["inputs"], spatial, level, score=True)["out"]
                proposals.append(result["proposals"].float().cpu().numpy())
                pick.append(result["pdm_score"].argmax(1).cpu().numpy())
                top.append(result["pdm_score"].max(1).values.cpu().numpy())
        return picks(np.concatenate(proposals), np.concatenate(pick), np.concatenate(top), human, speed)

    results = {"release": measure()}
    for run in runs:
        step.load_trainable(model, None, base)
        state = torch.load(run / f"epoch_{args.epoch:02d}.pt", map_location="cpu")
        # the scorer reads what the refiner makes, never the heads
        step.load_trainable(model, None, {"model": state["model"]})
        results[run.name] = measure()
    text = f"Every model at epoch {args.epoch} on {len(held)} held-out keyframes.\n\n" + table(results)
    if out:
        out.write_text(text)
        print(f"wrote {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
