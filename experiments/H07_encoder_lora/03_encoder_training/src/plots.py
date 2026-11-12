"""Plots for the report: warm-up curve, main loss, selection ADE, drift.

    ./run.sh src/plots.py     # writes out/plots/*.png
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from s2common import LOGS, OUT  # noqa: E402


def smooth(v, k=50):
    v = np.asarray(v, dtype=float)
    if len(v) < k:
        return v
    return np.convolve(v, np.ones(k) / k, mode="valid")


def read_jsonl(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def main() -> None:
    folder = OUT / "plots"
    folder.mkdir(exist_ok=True)

    wu = json.loads((OUT / "d_warmup.json").read_text())
    ws = read_jsonl(LOGS / "warmup_steps.jsonl")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].plot(smooth([r["loss"] for r in ws]), color="#bbb", label="train batch loss (50-step mean)")
    ax[0].plot([c["step"] for c in wu["curve"]], [c["fixed_loss"] for c in wu["curve"]], "o-", color="#2a5d9f",
               label="fixed selection set (2,000 pairs)")
    ax[0].set_xlabel("warm-up step"); ax[0].set_ylabel("L1 loss"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"Warm-up (guesser only): {wu['stop_reason']}", fontsize=8)
    ax[1].plot(smooth([r["gn_guesser"] for r in ws]), color="#2a5d9f")
    ax[1].set_xlabel("warm-up step"); ax[1].set_ylabel("guesser grad norm (50-step mean)")
    fig.tight_layout(); fig.savefig(folder / "d_warmup.png", dpi=110); plt.close(fig)

    hist = json.loads((OUT / "e_passes.json").read_text())
    ref = json.loads((OUT / "c_reference.json").read_text())
    zero_ade = ref["C1"]["selection_before"]["ade"]["mean"]
    ms = read_jsonl(LOGS / "main_steps.jsonl")
    P = hist["passes"]
    x = [p["pass"] for p in P]
    steps = [p["global_step"] for p in P]

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.4))
    ax[0].plot(smooth([r["loss"] for r in ms]), color="#bbb", label="train batch loss (50-step mean)")
    ax[0].plot(steps, [p["fixed_loss"] for p in P], "o-", color="#2a5d9f", label="fixed selection set")
    ax[0].axhline(wu["curve"][-1]["fixed_loss"], ls=":", color="#2a5d9f", label="end of warm-up")
    ax[0].set_xlabel("main step"); ax[0].set_ylabel("L1 loss"); ax[0].legend(fontsize=8)
    ax[1].plot(smooth([r["gn_lora"] for r in ms]), label="LoRA")
    ax[1].plot(smooth([r["gn_guesser"] for r in ms]), label="guesser")
    ax[1].set_yscale("log"); ax[1].set_xlabel("main step"); ax[1].set_ylabel("grad norm (50-step mean)"); ax[1].legend(fontsize=8)
    ax[2].plot(x, [p["lora_B_norm"] for p in P], "o-")
    ax[2].set_xlabel("pass"); ax[2].set_ylabel("||LoRA B|| (all layers)")
    fig.tight_layout(); fig.savefig(folder / "e_loss.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.4))
    ax[0].plot(x, [p["selection"]["ade"]["mean"] for p in P], "o-", label="LoRA")
    ax[0].axhline(zero_ade, ls="--", color="k", label=f"zero LoRA {zero_ade:.4f}")
    ax[0].axvline(hist["best_pass"], ls=":", color="g", label=f"chosen pass {hist['best_pass']}")
    ax[0].set_xlabel("pass"); ax[0].set_ylabel("selection ADE mean (m), 103 moments"); ax[0].legend(fontsize=8)
    ax[1].plot(x, [p["drift_vs_zero"]["mean_proposal_shift_m"] for p in P], "o-", label="mean")
    ax[1].plot(x, [p["drift_vs_zero"]["max_proposal_shift_m"] for p in P], "s--", label="max")
    ax[1].set_yscale("log"); ax[1].set_xlabel("pass"); ax[1].set_ylabel("proposal shift vs zero LoRA (m)"); ax[1].legend(fontsize=8)
    ax[2].plot(x, [p["drift_vs_zero"]["mean_grade_change"] for p in P], "o-")
    ax[2].set_xlabel("pass"); ax[2].set_ylabel("mean |grade change|")
    ax[3].plot(x, [100 * p["drift_vs_zero"]["chosen_changed"] / p["drift_vs_zero"]["samples"] for p in P], "o-")
    ax[3].set_xlabel("pass"); ax[3].set_ylabel("moments whose pick changed (%)")
    fig.tight_layout(); fig.savefig(folder / "e_selection_drift.png", dpi=110); plt.close(fig)
    print("plots written to", folder)


if __name__ == "__main__":
    main()
