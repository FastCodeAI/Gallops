# H18 — Fact scorer: H04's architecture trained from scratch on NAVSIM + LEAD

**Owner:** Navvrat · **Started:** 18 Nov 2026 (spec), trained 19–20 Nov · **Status:** running — gated by hand, FAIL
against released Drive-JEPA on NAVSIM; Bench2Drive's gate pairs running; no paddock ref carries its adapter
(hypothesis)

The facts are in [experiment.toml](experiment.toml), and what is off the repository is in [POINTERS.md](POINTERS.md).
This page is the account. The spec the run followed is [specs/run3b_v2_spec.md](specs/run3b_v2_spec.md), Navvrat's of
19 Nov, verbatim.

## Why

The sub-score agent-look model ([H04](../H04_subscore_scorer_navsim/README.md), called Run 3A while it ran) picks with
12 heads trained on NAVSIM's sub-scores, so it can only learn from NAVSIM. A scorer that predicts what happens along each
path, one waypoint at a time, can be labelled on CARLA too: the fact labels exist for NAVSIM's pool and for 98,641 LEAD
frames. This run trains H04's architecture end to end on both datasets with that scorer picking. The spec sets no pass
marks: numbers go next to H04's where they exist, and Navvrat decides.

## What was done

The fact-scorer model (`H18.fact-scorer`, called Run 3B v2 while it ran):

- **Encoder:** WA-JEPA ViT-L as in H04, base frozen, with a fresh LoRA (rank 32 on Q and V of all 24 blocks, B = 0, so
  at step 0 it is H04's encoder). Four cameras, two frames (t − 0.5 s and t), 512 × 256.
- **Agent decoder and planner:** H04's architecture (300 slots × 3 layers; 32 proposals × 8 waypoints, 4 rounds,
  footprint look and agent look), trained from scratch. Nothing is loaded from H04 or from Phase 1.
- **Scorer:** a path reader over each path's 8 waypoints (footprint look, agent look, self-attention) with 7 yes/no facts
  and 4 number facts per waypoint, and a scene head for the expert's metres. The score is computed from the facts
  (T4, γ 0.9, P 0.108), never a head; the pick is the proposal with the highest score, with the expert's metres from the
  head at test time.
- **Data:** 98,831 NAVSIM training samples and 116,069 LEAD samples, 32 of each per batch. Facts are labelled live on
  the GPU for every path the planner proposes; 0 label errors over the run.
- **Training:** 23,859 steps (7.1 passes), 19 Nov 03:15 to 20 Nov 09:41, the length a 48 h budget allowed at the
  measured step time. Teacher forcing on the agent look for the first pass, then down to 0.2 at 60 % of the run.
  Details in the run's notes in [experiment.toml](experiment.toml).
- **Checkpoint:** the last step, the end of the schedule. None was selected: the held-out reports came after every
  pass, and pass 6 had the highest held-out pick (0.876 against the last step's 0.866, n 1,008 each).

The code as it ran is in [scripts/](scripts/) (training, live labels, evaluation, launchers) and [paddock/](paddock/)
(the paddock adapter, in each of the two local worktrees the gate ran from). Machine roots are placeholders;
[AS_RUN_SHA256.txt](AS_RUN_SHA256.txt) holds each file's sha256 as it ran. The H04 code it imports from the Drive-JEPA
tree is [H04's overlay](../../recipes/drivejepa_wajepa/overlay/run3a.json), hashed in place in AS_RUN_SHA256.txt; of
those files only `drive_jepa_config.py` differs from the overlay, by the pick weights H04 locked on 25 Oct and a
machine-path default, neither of which this model reads.

### How it is fed in evaluation

Every evaluation feeds it as H04 is fed: the four cameras rescaled to 1920 × 1080, the t − 0.5 s and t frames through
WA-JEPA's crop and resize, H04's lidar2img and 11-value ego status. One file, `scripts/run3b/v2/model/run3b_infer.py`,
loads the checkpoint (it refuses a missing or extra tensor) and makes the pick, behind both the paddock adapter and the
NAVSIM agent. On the first 64 of the 1,008 NAVSIM held-out samples it gives the trainer's eval path the same proposals
(largest difference 0.0) and the same pick on 63 of 64; the other is a near-tie (top two 0.0007 apart, against bf16
score noise up to 0.0057).

## What it showed

**navtest**, all 12,146 scenes, NAVSIM's `run_pdm_score.py`, the same evaluation as H01's and H04's rows (no
momentum-aware selection), 0 failed scenes:

| Run | EPDMS, human filter | CSV score | NC | DAC | EP | TTC | LK | EC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Released Drive-JEPA | 87.78 | 84.49 | .985 | .986 | .891 | .979 | .976 | .478 |
| The sub-score agent-look model, locked pick (H04) | 87.30 | 84.09 | .986 | .984 | .900 | .981 | .977 | .406 |
| **The fact-scorer model** | **86.35** | **82.68** | .988 | .966 | .823 | .983 | .977 | .655 |
| The 4-camera planner (H01) | 85.27 | 81.98 | .980 | .972 | .879 | .973 | .967 | .401 |

Paired per scene, the fact-scorer model minus each run, 95 % intervals by bootstrap over the 136 logs: H04 −0.95
[−1.79, −0.19]; released −1.43 [−2.18, −0.66]; H01 +1.08 [+0.18, +2.09]. Its collision and time-to-collision terms are
the best of the four and its extended comfort the highest by far; ego progress (.823) and drivable area (.966) are the
lowest.

**Held-out NAVSIM** (1,008 samples, the trainer's report): pick EPDMS 0.866 against H04's 0.922 on the same samples
(H04's stored pick-tune dump); best of the 32 proposals 0.959 against 0.961. Agent recall at 10–20 m 0.579 against
H04's 0.621 on the same samples.

**The regression gate**, run by hand on 20 Nov from the worktrees and site config H04's gate ran under, against
released Drive-JEPA (the queue's stored runs), the subsets every gated record here has, in the same order:

| Benchmark | Metric | Released | The fact-scorer model | t | Units | Verdict | Against H04's gate runs |
|---|---|---:|---:|---:|---:|---|---|
| KITScenes | ADE (3 s), m | 2.5375 | 2.8596 | −1.46 | 117 | NEUTRAL | 2.159 → 2.914, t −4.51, REGRESSION |
| nuScenes | L2, 1-2-3 s, m | 2.4314 | 2.1173 | 2.37 | 143 | GAIN | 2.151 → 2.117, t 0.31, NEUTRAL |
| NAVSIM | EPDMS | 0.8572 | 0.8067 | −6.05 | 126 | REGRESSION | not paired: 83 shared tokens |
| neuroNCAP, quick (3 pairs) | NCAP | 3.1794 | 4.0229 | 0.98 | 3 | NEUTRAL | 4.107 → 4.023, NEUTRAL |
| Bench2Drive, crash check (route 3717) | DS | | | | 1 | running | |
| AlpaSim, crash check (1 clip) | scene score | 0.0000 | 0.0000 | | 1 | TOO FEW UNITS | |
| neuroNCAP, closed loop (15 pairs) | NCAP | 3.6792 | 3.9265 | 1.44 | 15 | NEUTRAL | 3.691 → 3.926, t 1.24, NEUTRAL |
| Bench2Drive, closed loop (7 routes) | DS | | | | 7 | running | H04: 60.39 |
| AlpaSim, closed loop (10 scenes) | scene score | 0.4162 | 0.1317 | −1.69 | 10 | NEUTRAL | 0.439 → 0.132, t −2.29, REGRESSION |

Overall **FAIL**, from the NAVSIM pair; on NAVSIM, ego progress (t −8.05) and drivable area (t −3.33) are the terms
beyond t crit. On AlpaSim the gate first printed CONFIG DIFFERS on the driver (this model's four-camera render against
Drive-JEPA's front camera), as it did for H01 and H04; its pairs are the same runs re-paired with `bridge_args`,
`driver_config` and `driver_profile` allowed. On AlpaSim its progress is 0.174 against released 0.473 (t −2.75).
Bench2Drive runs the gate's quick route and 7 routes against the queue's released run, scored without the comfort
fact (the spec masks comfort on all of CARLA); its two rows fill in when it ends. Not run: Bench2Drive on LEAD's
camera rig (the spec's evaluation; not built) and the light-swap test (named by the spec, not defined).

What is measured and what is not: every number above is a single run of one checkpoint; nothing separates the effect of
the fact scorer from training on LEAD, from the fresh LoRA or from starting the planner from scratch, since all four
changed at once.

## Decision

Not decided. The gate says FAIL on NAVSIM against released Drive-JEPA; the record stays `running` until the team takes
it, and no row in the decision log closes it yet.
