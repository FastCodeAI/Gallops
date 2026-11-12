"""Train H15's memory picker on the saved outputs and the strict grades; preflight first, always.

**Target.** BCE of the nudged final logit (`model.MemoryPicker`) against the strict PDM final grade of
each of the 32 paths (`grade.py`'s `scores[..., -1]`), at graded moments of the training logs: the 978
vendor train logs minus H15's 100 selection logs.

**Windows.** Each graded training moment is one example: its window is the moment and up to
min(warm-up, L) earlier moments of its chain, with warm-up drawn uniformly from 1..16 per example per
epoch (H15, "Fixed before training"). In reset mode the window's earlier moments are ignored by
construction. The order of examples and the warm-up draws come only from the seed and the epoch, so
arms (a) carry and (b) reset, and every L, see identical data in identical order; the initial weights
come only from the seed too (same parameter shapes in both modes).

**Selection.** After each epoch the selection logs' loss is computed with each moment's full window
(min(L, available) earlier moments). L and the epoch count are chosen by the rule in PREDICTIONS.md.
navtest is never read.

**Preflight** (`preflight`, run before the first step of every `final` run, no way to skip it):
committed tree; the test suite passes under this interpreter; 20 training moments printed for a look;
an untrained model reproduces the released picks on 200 val moments exactly; perturbing one path's
scorer input changes that path's x_i and no other; carry equals reset on chain-start moments.

    python -m gallops.memory.train preflight
    python -m gallops.memory.train select  [--epochs 8]          # (a) at L = 1, 2, 4 and (b), seed 0
    python -m gallops.memory.train final   --length L --epochs E  # (a) and (b), seeds 0, 1, 2; checkpoints

**The sidecar** (`sidecar_notes`) records what the weights are: (b) is written as length 0, since its
memory is wiped every moment whatever `--length` says; the train logs that gave a graded moment, by
count and by the sha256 of their sorted names; and the sha256 of every input file the run read
(`input_digests`), so the weights stay tied to their labels if the grades are ever rebuilt.

**One fit is one run.** `final` refuses, before anything trains, a run name whose directory already
holds a checkpoint (`refuse_existing_runs`): two fits under one name would read as one sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from gallops.config import REPO_ROOT
from gallops.memory.model import batch_windows

MODEL_NAME = "drivejepa_memory"
LENGTHS = (1, 2, 4)
WARMUP_MAX = 16
SEEDS = (0, 1, 2)
BATCH = 256
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 8
REPO = REPO_ROOT
#: The selection logs, relative to the repository.
SELECTION_LOGS = "experiments/H15_prediction_memory/selection_logs.txt"
#: Every file `Data` reads, relative to {data.navtrain.cache}: what the weights were trained on.
INPUTS = (
    "ticks/moments.npz",
    "align/prev.npy",
    "align/match.npy",
    "grades/row.npy",
    "grades/done.npy",
    "grades/scores.npy",
    "release/scorer_in.npy",
    "release/logits.npy",
    "release/ego.npy",
)


# --- data ------------------------------------------------------------------------------------------


class Data:
    """Every moment's inputs on the training device, plus targets, chains, matches and the splits."""

    def __init__(self, cfg, device: str):
        import torch

        from gallops.memory import ticks

        cache = Path(cfg.resolve("{data.navtrain.cache}"))
        self.table = ticks.load(cache / "ticks")
        n = len(self.table["log"])
        align = cache / "align"
        self.prev = np.load(align / "prev.npy")
        self.match = np.load(align / "match.npy").astype(np.int64)
        assert len(self.prev) == n and self.match.shape == (n, 32)
        grades = cache / "grades"
        row, done = np.load(grades / "row.npy"), np.load(grades / "done.npy")
        final = np.asarray(np.load(grades / "scores.npy", mmap_mode="r")[:, :, -1], np.float32)
        self.target = np.full((n, 32), np.nan, np.float32)
        self.target[row[done]] = final[done]
        self.graded = ~np.isnan(self.target[:, 0])
        release = cache / "release"
        self.logits_np = np.asarray(np.load(release / "logits.npy", mmap_mode="r"), np.float32)
        t0 = time.time()
        embed = np.asarray(np.load(release / "scorer_in.npy", mmap_mode="r"), np.float32)
        self.device = device
        self.embed = torch.from_numpy(embed).to(device)
        self.logits = torch.from_numpy(self.logits_np).to(device)
        self.ego = torch.from_numpy(np.asarray(np.load(release / "ego.npy"), np.float32)).to(device)
        self.target_t = torch.from_numpy(np.nan_to_num(self.target)).to(device)
        self.load_s = round(time.time() - t0, 1)
        selection = set((REPO / SELECTION_LOGS).read_text().split())
        split, logs = self.table["split"], self.table["log"]
        in_selection = np.isin(logs, list(selection))
        self.rows = {
            "train": np.flatnonzero(self.graded & (split == "train") & ~in_selection),
            "selection": np.flatnonzero(self.graded & in_selection),
            "val": np.flatnonzero(self.graded & (split == "val")),
        }
        self.train_logs = int(len(set(logs[(split == "train") & ~in_selection].tolist())))
        #: the train logs that gave at least one graded moment: the ones the weights actually saw
        self.train_logs_used = sorted(set(logs[self.rows["train"]].tolist()))

    def batch(self, targets, earlier):
        import torch

        rows, match, valid = batch_windows(self.prev, self.match, targets, earlier)
        r = torch.from_numpy(rows).to(self.device)
        return (
            self.embed[r],
            self.logits[r],
            self.ego[r],
            torch.from_numpy(match).to(self.device),
            torch.from_numpy(valid).to(self.device),
            self.target_t[torch.from_numpy(np.asarray(targets, np.int64)).to(self.device)],
        )


def pick_device(min_free_gb: float = 12.0) -> str:
    """cuda when a GPU has at least `min_free_gb` free (Sid D27), else cpu."""
    import torch

    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free / 2**30 >= min_free_gb:
            return "cuda"
    return "cpu"


def set_determinism(seed: int) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def epoch_plan(n: int, seed: int, epoch: int) -> tuple[np.ndarray, np.ndarray]:
    """The order of the n examples and their warm-ups: from seed and epoch only, never the arm."""
    rng = np.random.default_rng([seed, epoch])
    return rng.permutation(n), rng.integers(1, WARMUP_MAX + 1, size=n)


def loss_of(model, data: Data, rows: np.ndarray, mode: str, length: int, batch: int = 1024) -> float:
    """Mean BCE over every path of `rows`, each with its full window (min(L, available) earlier)."""
    import torch
    import torch.nn.functional as F

    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for k in range(0, len(rows), batch):
            part = rows[k : k + batch]
            embed, logits, ego, match, valid, target = data.batch(part, length if mode == "carry" else 0)
            score, _ = model(embed, logits, ego, match, valid, mode)
            total += float(F.binary_cross_entropy_with_logits(score, target, reduction="sum"))
            count += target.numel()
    model.train()
    return total / count


def scores_of(model, data: Data, rows: np.ndarray, mode: str, length: int, batch: int = 1024) -> np.ndarray:
    """(len(rows), 32) nudged scores, each with its full window."""
    import torch

    out = []
    model.eval()
    with torch.no_grad():
        for k in range(0, len(rows), batch):
            part = rows[k : k + batch]
            embed, logits, ego, match, valid, _ = data.batch(part, length if mode == "carry" else 0)
            out.append(model(embed, logits, ego, match, valid, mode)[0].cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros((0, 32), np.float32)


def train_one(data: Data, mode: str, length: int, seed: int, epochs: int, log=print) -> dict:
    """One run; returns per-epoch losses and the final state dict (CPU tensors)."""
    import torch
    import torch.nn.functional as F

    from gallops.memory.model import MemoryPicker

    set_determinism(seed)
    model = MemoryPicker().to(data.device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_rows = data.rows["train"]
    history, steps, t0 = [], 0, time.time()
    for epoch in range(epochs):
        order, warm = epoch_plan(len(train_rows), seed, epoch)
        earlier = np.minimum(warm, length)
        total = 0.0
        for k in range(0, len(order), BATCH):
            idx = order[k : k + BATCH]
            embed, logits, ego, match, valid, target = data.batch(train_rows[idx], earlier[idx])
            score, _ = model(embed, logits, ego, match, valid, mode)
            loss = F.binary_cross_entropy_with_logits(score, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
            total += float(loss) * len(idx)
        sel = loss_of(model, data, data.rows["selection"], mode, length)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / len(order),
                "selection_loss": sel,
                "steps": steps,
                "seconds": round(time.time() - t0, 1),
            }
        )
        log(
            f"    {mode} L={length} seed={seed} epoch {epoch + 1}: "
            f"train {total / len(order):.6f} selection {sel:.6f}"
        )
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {"history": history, "state": state, "steps": steps, "model": model}


# --- preflight -------------------------------------------------------------------------------------


def preflight(data: Data, out: Path) -> dict:
    """Every check must pass before a final run trains; the report lands beside the checkpoints."""
    import torch

    from gallops.memory.model import MemoryPicker

    checks, failures = {}, []

    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "tests"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    checks["committed"] = not dirty
    if dirty:
        failures.append(f"uncommitted changes under src/ or tests/: {dirty.splitlines()[:5]}")

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", str(REPO / "tests"), "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    checks["tests_this_interpreter"] = {
        "python": sys.version.split()[0],
        "passed": tests.returncode == 0,
        "summary": tests.stdout.strip().splitlines()[-1] if tests.stdout.strip() else tests.stderr[-300:],
    }
    if tests.returncode != 0:
        failures.append(f"tests fail under {sys.executable}")

    rng = np.random.default_rng(0)
    look = np.sort(rng.choice(data.rows["train"], size=20, replace=False))
    t = data.table
    sample = []
    for r in look:
        grade = data.target[r]
        pick = int(data.logits_np[r][:, -1].argmax())
        sample.append(
            {
                "row": int(r),
                "log": str(t["log"][r]),
                "chain_pos": int(t["chain_pos"][r]),
                "prev_row": int(data.prev[r]),
                "prev_chain_pos": int(t["chain_pos"][data.prev[r]]) if data.prev[r] >= 0 else None,
                "paths_matched": int((data.match[r] >= 0).sum()),
                "released_pick": pick,
                "pick_grade": round(float(grade[pick]), 4),
                "best_grade": round(float(grade.max()), 4),
                "grades_min_mean": [round(float(grade.min()), 4), round(float(grade.mean()), 4)],
            }
        )
    checks["twenty_moments"] = sample
    ok_sample = all(
        (s["prev_row"] < 0 or s["prev_chain_pos"] == s["chain_pos"] - 1)
        and 0 <= s["grades_min_mean"][0] <= s["best_grade"] <= 1
        for s in sample
    )
    checks["twenty_moments_consistent"] = ok_sample
    if not ok_sample:
        failures.append("the 20-moment sample has a chain or grade inconsistency")

    set_determinism(0)
    model = MemoryPicker().to(data.device)
    val = np.sort(np.random.default_rng(1).choice(data.rows["val"], size=200, replace=False))
    released = data.logits_np[val][:, :, -1]
    same = {}
    for mode in ("carry", "reset"):
        s = scores_of(model, data, val, mode, 4)
        same[mode] = {
            "picks_equal": int((s.argmax(1) == released.argmax(1)).sum()),
            "scores_equal_exactly": bool(np.array_equal(s, released)),
        }
    checks["untrained_reproduces_released_on_200_val"] = same
    if not all(v["picks_equal"] == 200 and v["scores_equal_exactly"] for v in same.values()):
        failures.append(f"an untrained model does not reproduce the released picks: {same}")

    r0 = data.rows["train"][:1]
    embed, logits, ego, match, valid, _ = data.batch(r0, 0)
    with torch.no_grad():
        x = model.step_input(embed[:, -1], logits[:, -1], ego[:, -1])
        bumped = embed.clone()
        bumped[:, -1, 3] += 1.0
        x2 = model.step_input(bumped[:, -1], logits[:, -1], ego[:, -1])
    changed = (x2 - x).abs().amax(-1)[0]
    checks["sight_scorer_input_reaches_x"] = {
        "path_3_changed": bool(changed[3] > 0),
        "other_paths_unchanged": bool((changed[torch.arange(32) != 3] == 0).all()),
    }
    if not all(checks["sight_scorer_input_reaches_x"].values()):
        failures.append("perturbing path 3's scorer input does not change exactly x_3")

    starts = np.flatnonzero(data.prev < 0)[:256]
    with torch.no_grad():
        model.out[-1].weight.normal_(0, 0.1)  # a non-zero nudge, or carry and reset agree trivially
        model.out[-1].bias.normal_(0, 0.1)
    a = scores_of(model, data, starts, "carry", 4)
    b = scores_of(model, data, starts, "reset", 4)
    checks["carry_equals_reset_on_chain_starts"] = {
        "moments": int(len(starts)),
        "equal": bool(np.array_equal(a, b)),
    }
    if not np.array_equal(a, b):
        failures.append("carry and reset differ on chain-start moments")

    checks["data"] = {
        "device": data.device,
        "train_moments": int(len(data.rows["train"])),
        "train_logs": data.train_logs,
        "selection_moments": int(len(data.rows["selection"])),
        "val_moments": int(len(data.rows["val"])),
        "inputs_load_s": data.load_s,
    }
    if len(data.rows["val"]) != 15320 or len(data.rows["selection"]) != 7007:
        failures.append(f"graded counts are not the pre-registered ones: {checks['data']}")
    result = {"passed": not failures, "failures": failures, "checks": checks}
    out.mkdir(parents=True, exist_ok=True)
    (out / "preflight.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


# --- commands --------------------------------------------------------------------------------------


def input_digests(cache: Path) -> dict:
    """sha256 of every file the trainer reads, and of the selection list, read in 16 MB blocks."""
    import hashlib

    def sha(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                digest.update(block)
        return digest.hexdigest()

    out = {name: sha(Path(cache) / name) for name in INPUTS}
    out["selection_logs.txt"] = sha(REPO / SELECTION_LOGS)
    return out


def sidecar_notes(
    mode: str,
    length: int,
    seed: int,
    epochs: int,
    history: list,
    preflight_passed: bool,
    train_logs_used: list,
    train_logs: int,
    inputs: dict,
) -> dict:
    """The `notes` of a memory checkpoint's sidecar. Pure, so `make check` tests it without torch."""
    import hashlib

    return {
        "experiment": "H15",
        "mode": mode,
        # (b) wipes its memory at every moment: whatever --length was, its weights carry none.
        "length": length if mode == "carry" else 0,
        "seed": seed,
        "epochs": epochs,
        "history": history,
        "preflight_passed": preflight_passed,
        "train_logs_used": len(train_logs_used),
        "train_logs_in_split": train_logs,
        "train_logs_used_sha256": hashlib.sha256("\n".join(sorted(train_logs_used)).encode()).hexdigest(),
        "inputs_sha256": inputs,
        "frozen": "released Drive-JEPA; this file holds the memory picker only",
    }


def refuse_existing_runs(cfg, runs: list[str]) -> None:
    """Stop before training if any run's directory already holds a checkpoint: one fit is one run."""
    taken = [run for run in runs if any(cfg.run_dir(MODEL_NAME, run).glob("step_*"))]
    if taken:
        raise SystemExit(
            f"these runs hold a checkpoint already: {', '.join(taken)}; one fit is one run, "
            "so a new fit needs new run names"
        )


def run_name(mode: str, length: int, seed: int) -> str:
    return f"h15-r3-{'a' if mode == 'carry' else 'b'}-L{length if mode == 'carry' else 0}-s{seed}"


def save_checkpoint(
    cfg,
    run: str,
    result: dict,
    mode: str,
    length: int,
    seed: int,
    epochs: int,
    pre: dict,
    train_logs_used: list,
    train_logs: int,
    inputs: dict,
):
    import torch

    from gallops import checkpoint as ck
    from gallops.drivejepa_b2d.provenance import git_sha

    step = int(result["steps"])
    directory = cfg.checkpoint_dir(MODEL_NAME, run, step)
    with ck.staged(directory) as tmp:
        weights = tmp / "memory.pt"
        torch.save(result["state"], weights)
        params = sum(v.numel() for v in result["state"].values()) / 1e6
        ck.write(
            tmp,
            ck.Checkpoint(
                model=MODEL_NAME,
                step=step,
                train_git_sha=git_sha(),
                train_run=run,
                trained_on=(
                    ck.TrainedOn(
                        "navsim",
                        "navtrain: vendor train logs minus the 100 H15 selection logs, "
                        f"the {len(train_logs_used)} of {train_logs} with a graded moment",
                        fraction=round(len(train_logs_used) / train_logs, 4),
                    ),
                ),
                selection_split="navtrain: the 100 H15 selection logs",
                params_m=round(params, 4),
                precision="fp32",
                weights_sha256=ck.sha256_file(weights),
                notes=sidecar_notes(
                    mode,
                    length,
                    seed,
                    epochs,
                    result["history"],
                    pre["passed"],
                    train_logs_used,
                    train_logs,
                    inputs,
                ),
            ),
        )
    return directory


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=("preflight", "select", "final"))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--length", type=int, choices=LENGTHS)
    ap.add_argument("--out", type=Path, help="default {data.navtrain.cache}/h15")
    args = ap.parse_args()

    import torch

    from gallops.config import SiteConfig

    torch.set_num_threads(2)
    cfg = SiteConfig.load()
    out = (args.out or Path(cfg.resolve("{data.navtrain.cache}")) / "h15").resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.part == "final" and args.length is not None:
        # before the inputs load: a refused run should cost seconds, not the load
        refuse_existing_runs(cfg, [run_name(m, args.length, s) for s in SEEDS for m in ("carry", "reset")])
    device = pick_device()
    print(f"device {device}", flush=True)
    data = Data(cfg, device)
    print(f"inputs loaded in {data.load_s} s; train {len(data.rows['train'])} moments", flush=True)
    pre = preflight(data, out)
    print(json.dumps({k: v for k, v in pre["checks"].items() if k != "twenty_moments"}, indent=1), flush=True)
    if not pre["passed"]:
        raise SystemExit(f"preflight failed: {pre['failures']}")
    if args.part == "preflight":
        return
    zero = {name: None for name in ("selection",)}
    from gallops.memory.model import MemoryPicker

    set_determinism(0)
    zero["selection"] = loss_of(MemoryPicker().to(device), data, data.rows["selection"], "reset", 0)
    if args.part == "select":
        runs = [("carry", length) for length in LENGTHS] + [("reset", 0)]
        results = {"device": device, "released_selection_loss": zero["selection"], "runs": []}
        for mode, length in runs:
            r = train_one(data, mode, length, 0, args.epochs, log=lambda s: print(s, flush=True))
            results["runs"].append({"mode": mode, "length": length, "seed": 0, "history": r["history"]})
            (out / "selection.json").write_text(json.dumps(results, indent=1) + "\n")
        return
    if args.length is None:
        raise SystemExit("final needs --length (chosen on the selection logs)")
    inputs = input_digests(Path(cfg.resolve("{data.navtrain.cache}")))
    results = {"device": device, "length": args.length, "epochs": args.epochs, "runs": []}
    for seed in SEEDS:
        for mode in ("carry", "reset"):
            run = run_name(mode, args.length, seed)
            r = train_one(data, mode, args.length, seed, args.epochs, log=lambda s: print(s, flush=True))
            directory = save_checkpoint(
                cfg,
                run,
                r,
                mode,
                args.length,
                seed,
                args.epochs,
                pre,
                data.train_logs_used,
                data.train_logs,
                inputs,
            )
            results["runs"].append(
                {
                    "run": run,
                    "mode": mode,
                    "seed": seed,
                    "checkpoint": str(directory),
                    "history": r["history"],
                }
            )
            (out / "final.json").write_text(json.dumps(results, indent=1) + "\n")
            print(f"  saved {directory}", flush=True)


if __name__ == "__main__":
    main()
