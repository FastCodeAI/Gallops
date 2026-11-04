# H15 — A perception side-car on frozen Drive-JEPA: 3D boxes and each agent's 1 s path, tested as an input to the scorer, to the refiner, and against LoRA on the ViT

**Owner:** Shreedhar · **Started:** 2026-11-03 · **Status:** running (setup, S0)

The facts — runs, data, checkpoints, results, gate verdicts — live in
[experiment.toml](experiment.toml) and are rendered into the repository README. This page is the
account: why, what was done, what it showed, and what was decided.

- **The plan**, with every step, rule and cost estimate: [docs/drivejepa-perception-sidecar.md](../../docs/drivejepa-perception-sidecar.md).
- **The predictions**, committed before anything trained: [PREDICTIONS.md](PREDICTIONS.md).

## Why

Drive-JEPA's planner reads image tokens, but it has no explicit account of the other agents. So
nothing in it can be asked *which* car it reacted to.

H15 tests a small network on the frozen ViT's tokens that names the agents, with 3D boxes and
each one's next second. It then asks, as separate questions, whether handing those agents to the
scorer or to the refiner improves the plan, and how much of either has to change to use them. It
also asks whether LoRA on the ViT does better than the frozen tap.

![H15 at a glance](figures/overview.png)

![The architecture, with the key sizes](figures/architecture.png)

## What was done

| Date | Step | What happened |
|---|---|---|
| 3 Nov | plan | Approved and committed: `docs/drivejepa-perception-sidecar.md` |
| 3 Nov | S0.1 | The site's `gallops.toml` points `navsim` at the real NAVSIM root (trainval: 1,310 logs, all 8 cameras, *counted*) and names the metric cache |
| 3 Nov | S0.2 | `Drive-JEPA-ext` recreated (details below) |
| 3 Nov | S0.5 | Download started: Drive-JEPA's `train_metric_cache_v2_pruned.tar` (33.5 GB) and `anchors_scores_index_v2.tar` (0.4 GB). Source: the Hugging Face dataset repo `LinhanWang/Drive-JEPA` at `9d435e12`, checked against its LFS sha256s |
| 3 Nov | S0.9 | H15 registered; predictions written |
| 3 Nov | S0.10 | The plan's diagrams rendered from Graphviz sources in [figures/](figures/) |
| 3 Nov | S0.2 | The release's forward on navtrain samples through Drive-JEPA-ext: finite, and bit-identical over two passes |
| 3 Nov | S0.3 | Splits committed: [splits.json](../../recipes/drivejepa_perception/splits.json). Select is H04's 51 held-out logs; report is a seeded draw of 70; train is the other 1,071 of navtrain's 1,192 logs |
| 3 Nov | S0.4 | The labels pass ran from `6efd8e6` on all 103,288 navtrain samples in 2 min (numbers below) |
| 3 Nov | S0.5 | Downloads verified against their LFS sha256s, then extracted into NAVSIM's `exp/` |
| 3 Nov | S0.5 | `pdm_labels` written and smoke-run (details below) |
| 3 Nov | S0.6 | The explainer: [docs/navsim-metric-cache.md](../../docs/navsim-metric-cache.md), with worked examples and figures |
| 3 Nov | S0 | The token pass rebuilt twice to make it affordable, each change held to the vendor's exact output (details below) |

**`pdm_labels` (S0.5) in detail:**
- **Measured:** a metric cache exists for 103,288 of 103,288 navtrain samples. "Pruned" removed
  three detection fields the scorer does not read.
- **A bug the smoke run caught:** the vendor's `after_score_proposals` already drops the PDM-Closed
  reference, so a second `[1:]` would have shifted every label by one candidate. The code now
  asserts one row per plan.

**The token pass (S0), and its two changes:**
- **First run:** 6.4 rows/s, an ETA of 4.4 h.
- **Inputs:** profiled, the vendor's feature builder spent most of a sample normalising four
  full-size camera images into a tensor the model never reads.
  - `fast_bev_feature` computes the geometry the model does read, held exactly equal to the
    vendor's on 12 rows.
  - The cache it builds is bit-identical to the vendor-path cache.
- **Still GPU-bound** at 6.2 rows/s: a batch-1 forward took 158 ms on the shared GPU, half of it
  the refiner's small kernels.
  - The pass now runs batched with `batch_fix`, held to batch 1 on 32 rows before it starts.
  - Measured on the smoke run: every pick equal, candidates within 3.8e-5 m, tokens within one fp16
    step.
- **Result:** 23.7 rows/s, about 70 min for the 103,288 rows.

**The labels pass (S0.4), *measured*:**
- **Agents within 60 m:** 7,104,100, of which 1,812,167 vehicles, 1,533,271 pedestrians, 16,245
  cyclists and 3,742,417 static.
- **Path joins:** 96.6% at t + 0.5 s and 93.4% at t + 1.0 s.
- **`check`, on 60 random windows:**
  - the ego future is 1.9e-6 off NAVSIM's `get_future_trajectory`;
  - path points are 3.7e-6 m off nuPlan's and NAVSIM's transforms;
  - re-runs equal the stored labels;
  - 27 of 8,208 joins (0.33%) change box size by more than 0.5 m.

**S0.2 in detail.**
- `Drive-JEPA-ext` is `e21f474` plus Paddock's patches 0001–0003, as commits `c92b06c`, `7f29654`
  and `afab5fc`.
- *Measured* on an H200 NVL (sm_90):
  - the compiled deformable-attention kernel is refused under `auto`;
  - the PyTorch path agrees between CUDA and CPU to 1.8e-7;
  - forced, the compiled kernel prints "no kernel image is available", does not raise, and
    returns output 0.989 off a reference whose largest value is 0.989.

## What it showed

Nothing yet: no run has trained.

![Each hypothesis, its tests and its readout](figures/hypotheses.png)

![How the results decide what is claimed](figures/decisions.png)

## Decision

None yet.
