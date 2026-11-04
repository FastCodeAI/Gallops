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
| 3 Nov | A, B | Code written and tested ahead of the data, all under `gallops.drivejepa_perception` (list below) |
| 3 Nov | — | The test suite passes under Drive-JEPA's 3.9 again, so 3.9 trainers' preflights can pass (`e959b06`) |
| 4 Nov | S0 | Token pass done (63 min); PDM labels for all 103,288 rows, re-scored exactly; the release read on select and report (S0.8) |
| 4 Nov | A0, A1 | The three taps probed; the ViT's own output chosen, and its cache pass started; the blind baseline trained |
| 4 Nov | B1 | The first C runs (loss `imitation`, `84fd73e`) lost to the release in every mode. The loss is now chosen on the select logs, C arm only, before any arm reads an agent (below) |

**Code written ahead of the data:**
- **Side-car and phase A:** the side-car, its targets and metrics, and the phase-A trainer.
- **B1 and B2:** the scorer modes and the B1 trainer; the refiner trainer (B2).
- **Shared by phase B:** the verdict rule (`b_report`) and the explanation test (B1.6).
- **Checks:** every trainer's preflight.

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
- **Done (4 Nov), from `a2c5255`:** 103,288 rows and 14,457 A0 tap rows in 63 min (56 GB), against
  the release whose sha256 is `740335c8`.
  - The startup check, batch 16 against batch 1 on 32 rows: every pick equal; candidates within
    4.6e-5 m, logits within 1.8e-5, tokens within 0.004.
  - The fast inputs equalled the vendor's on 12 rows.

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

So far, phase A has started. Structured results go into `experiment.toml` when B1 gives H15 its
headline; until then, the numbers live here.

### S0.5 · the PDM labels (4 Nov, from `b46230a`)

- **All 103,288 navtrain samples scored**, in 10 min on 48 workers; no metric cache missing.
- **Re-scoring:** `check-pdm` re-scored 50 random rows and found them exactly equal.
- **Mean final scores:** 0.798 for the release's 32 candidates, 0.967 for the best of them, and
  0.906 for the human's plan.
- **Collisions:** 7.51% of candidates have an at-fault collision, exactly the share whose NC is
  below 1.

### S0.8 · the release's own picks (in-sample for it)

| Split | Pick EPDMS | Best of 32 | Gap | L2 at 1 / 2 / 3 s (m) |
|---|---|---|---|---|
| select (4,457) | 0.920 | 0.961 | 0.041 | 0.68 / 1.48 / 2.53 |
| report (4,636) | 0.914 | 0.956 | 0.042 | 0.70 / 1.50 / 2.56 |

**The release's pick, term by term (report logs):**

| NC | DAC | TLC | DDC | EP | TTC | LK | HC |
|---|---|---|---|---|---|---|---|
| 0.996 | 0.994 | 0.979 | 0.989 | 0.919 | 0.996 | 0.860 | 0.975 |

- **P6** predicted a gap of at least 5 points. It is **4.2**, less than predicted.
- **What this means for B (*argued*, written before any B result):**
  - The terms agents can inform, NC and TTC, are already 0.996 on the release's pick in-sample.
    The pick loses most in lane keeping, progress and traffic lights.
  - So on NAVSIM in-sample, an agent-aware scorer has a few tenths of a point to win from
    collisions. P7's +0.3 to +1.0 is optimistic on this evidence.

### A0 · which frozen tap carries 3D

One-layer probes, 6 epochs on 10,000 train rows, read on the select logs, front view:

| Tap | Recall at 2 m, ≤ 30 m | Precision | F1 | Recall 0–20 / 20–35 / 35–50 m |
|---|---|---|---|---|
| post-projector, 256-d | 0.225 | 0.299 | 0.257 | 0.351 / 0.172 / 0.120 |
| **ViT output, 1024-d** | **0.313** | **0.367** | **0.338** | 0.459 / 0.251 / 0.140 |
| block 12, 1024-d | 0.242 | 0.303 | 0.269 | 0.369 / 0.188 / 0.095 |

- **P1 is supported at its lower edge.** The best tap reaches 0.313, against a predicted range of
  0.35–0.60 and a support threshold of 0.30.
- **The ViT output beats the projector's** by 0.088 recall, against a predicted ≥ 0.03.
- **All three were still rising** at epoch 6.
- **Decided:** A1 reads the ViT's own output, cached for every row by a second pass that is held
  to the full forward's tap.

### A1 · the no-camera baseline (blind arm, report logs)

- Trained identically on zeroed tokens, it detects nothing: recall 0 over 24,216 targets.
- The lead-gap reference is the train-mean prior, with an error of 8.53 m (P5's bar).

### B1 · the first loss pulled the scorer toward the human (a deviation, before any agent arm)

The plan's B1 loss was BCE on the PDM labels plus a listwise CE toward the human,
`softmax(−ADE/τ)`, weight 0.5, τ = 1 m. The first control runs (arm C, no agents; `84fd73e`)
trained with it in three modes, three seeds each (frozen at `84fd73e`; lora and full at `3e666e1`,
whose code is the same). *Measured*, select logs, each seed's kept epoch:

| Mode | Pick EPDMS, seeds 0 / 1 / 2 | L2 at 1 s (m) | EP of the pick |
|---|---|---|---|
| release, untrained | 0.9201 | 0.681 | 0.937 |
| frozen | 0.8947 / 0.8947 / 0.8955 | 0.150 / 0.149 / 0.157 | 0.879 / 0.882 / 0.879 |
| lora | 0.8943 / 0.8932 / 0.8944 | 0.167 / 0.157 / 0.167 | 0.877 / 0.875 / 0.879 |
| full | 0.8930 / 0.8941 / 0.8941 | 0.155 / 0.162 / 0.153 | 0.875 / 0.877 / 0.878 |

The fourth mode, `replace`, did not run: its preflight refused uncommitted code.

- **Every control lost 2.5–2.7 points to the release.** The kept epoch was 1 or 2 in every run.
- **Its L2 to the human fell from 0.68 m to 0.15 m at 1 s.** The scorer had learnt to pick the
  candidate nearest the human. Its progress fell with it, because the human's own plan scores
  0.906 against the release pick's 0.920 (S0.5, S0.8).
- *Argued:* an F-against-C comparison under that loss would ask whether agents help an imitation
  picker, which is not B1's question.

**What changed, and what did not:**
- **Changed:** the loss's listwise term is now chosen, by `train_scorer --choose-loss`, among three:
  - `imitation`: the first runs';
  - `bce`: none;
  - `best`: toward the PDM-best candidates, the choice the readout scores.
- **The rule, fixed before the choice ran:** seed 0 of every mode trains with each loss; the
  highest mean over the four modes of the select pick EPDMS at the kept epoch wins.
- **What the choice reads:** select logs, arm C only. Report logs and agents are not read.
- **Unchanged:** the arms, the modes, the seeds, the decision rule and every prediction. The chosen
  loss is used in every mode and arm.
- The imitation runs are kept, apart, as `{runs}/drivejepa_perception/b1-imitation/`.

**The choice (`b1-choose-loss`, `977bb56`), *measured*:** select logs, arm C, seed 0, each run's
kept epoch. The release's own pick scores 0.9201.

| Loss | frozen | lora | full | replace | Mean | NC / TTC of the pick, frozen | L2 at 1 s, frozen |
|---|---|---|---|---|---|---|---|
| imitation | 0.8947 | 0.8943 | 0.8930 | 0.8974 | 0.8948 | 0.998 / 0.996 | 0.15 m |
| **bce** | **0.9202** | **0.9200** | **0.9202** | **0.9192** | **0.9199** | 0.996 / 0.994 | 0.86 m |
| best | 0.8635 | 0.8669 | 0.8633 | 0.8637 | 0.8644 | 0.972 / 0.967 | 1.35 m |

- **Chosen: `bce`.** Every mode's control then sits within 0.001 of the release.
- **Ranking toward the PDM-best candidate made the control worse than imitation did.** Its pick
  moved away from the human (1.35 m at 1 s) and into collisions: NC 0.972 and TTC 0.967, against
  0.995 and 0.994 for the release.
  - *Argued:* the best candidate is often the fastest safe one. Without agents the control cannot
    tell safe from unsafe, so it learns "faster" and pays in collisions.
  - That is the gap the F arm is meant to fill. It is also why `best` is not a fair control:
    under it, F could beat C by repairing damage the loss did, not by adding a reason.
- **The replacement scorer** (B1.4: a new network, trained from scratch on the control's own
  inputs) reaches 0.9192, within 0.001 of the release, in two epochs.

![Each hypothesis, its tests and its readout](figures/hypotheses.png)

![How the results decide what is claimed](figures/decisions.png)

## Decision

None yet.
