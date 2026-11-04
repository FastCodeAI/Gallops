# H15 — predictions, written before the first run

**Commit this file before anything trains.** Its commit date is the evidence that the predictions
came first; a prediction written after the result is not one.

- **Written:** 2026-11-03.
  - Drafted by Claude from the estimates in the plan (`docs/drivejepa-perception-sidecar.md`), at
    the request of the owner, Shreedhar, who approved committing them as H15's pre-registration.
  - A row changed after the run it decides is a new prediction, recorded as one.
- **Hypothesis:**
  - **H-A:** a side-car of about 4 M parameters on the frozen ViT's tokens beats no-camera and
    kinematic baselines at detecting agents and at their 1 s paths.
  - **H-B1, H-B2:** its decoded agents improve the scorer's pick (H-B1) and the refiner's
    proposals (H-B2) over the identical adaptation without them.
  - **H-B3:** LoRA on the ViT beats the frozen tap.

## What will be measured

**Splits.**
- Every row reads the **report** logs: a fresh seeded draw of about 70 navtrain logs.
- Choices (epoch, mode, threshold) are made on the **select** logs: H04's 51 held-out navtrain logs.
- The release trained on all of navtrain, so any comparison against R on these logs is
  **in-sample for R** and flagged.
- "2 SE" is twice the paired standard error, clustered by log.

| # | Benchmark / subset | Metric | Baseline | Predicted | Counts as support if | Counts as refuted if |
|---|---|---|---|---|---|---|
| P1 | A0 · report logs, front view, ≤ 30 m, 10k-sample probe | recall at 2 m (all classes), best of 3 taps | no-camera prior (0) | 0.45 (0.35–0.60); ViT-out 1024-d beats post-projector 256-d by ≥ 0.03 | the best tap ≥ 0.30 | every tap < 0.20 |
| P2 | A1 · report logs, front view, ≤ 30 m | recall / precision at 2 m | no-camera prior | recall 0.50 (0.40–0.60), precision ≥ 0.60 | recall ≥ 0.40 and precision ≥ 0.50, on all 3 seeds | recall < 0.30 on any seed |
| P3 | A1 · moving vehicles within 20 m | recall at 2 m | — | ≥ 0.75 | ≥ 0.65 | < 0.50 |
| P4 | A1 · matched moving agents | path FDE at 1 s, m | constant velocity from the predicted velocity | ≥ 10% lower than that baseline | lower beyond 2 SE | not lower |
| P5 | A1 · nearest in-lane vehicle ahead | lead-gap MAE, m | no-camera prior (train mean) | ≤ 3 m | ≤ half the prior's error | ≥ the prior's error |
| P6 | S0.8 · report logs | the release's pick EPDMS, and best-of-32 EPDMS | — | best-of-32 exceeds the pick by ≥ 5 points (in-sample for R) | — (a measurement, read before B) | — |
| P7 | B1 · report logs | the pick's EPDMS, F − C | arm C, same mode | at least one mode positive, +0.3 to +1.0 points; most of it in the NC and TTC terms | some mode beats C beyond C's seed range and 2 SE on every seed | no mode does |
| P8 | B1 · the pre-registered subsets below | the pick's EPDMS, F − C | arm C | ≥ 2× the all-sample F − C in each subset | larger than all-sample beyond 2 SE in ≥ 2 of 3 subsets | not larger in any |
| P9 | B1 modes | share of the best mode's F − C that B1.1 (frozen scorer + adapter) reaches | best mode | ≥ 50% | ≥ 50% | < 25% |
| P10 | B1.5 · best B1 mode | (S − C)/(F − C) and O − F | arms C, F | S − C ≤ 20% of F − C; O − F > 0 | both hold | S − C ≥ 50% of F − C |
| P11 | B1.6 · best F, seed 0 | pick change rate when the named agent is deleted ÷ the same for a random detected agent; grounding; NC-head AUC | random-agent deletion; the release's collision head | ratio ≥ 3; grounding ≥ 85% within 2 m; AUC ≥ 0.75 | all three | ratio < 1.5 |
| P12 | B2 · report logs | best-of-32 EPDMS, F − C (best mode) | arm C | +0.2 to +0.5 points | beyond C's seed range and 2 SE on every seed | not positive |
| P13 | B3.1 | A1's recall at 2 m, LoRA branch − frozen tap | A1's side-car | ≥ +0.05 | ≥ +0.05 beyond seed range | ≤ 0 |
| P14 | B3.2 · report logs, and nuScenes | pick EPDMS vs the best B1/B2 model; nuScenes L2 at 3 s vs R | best B1/B2; R | not better than the best B1/B2 by more than 2 SE; worse than R on nuScenes | — | better than the best B1/B2 beyond 2 SE and no nuScenes regression |
| P15 | OOD · nuScenes, H14's 102 held-out train scenes | side-car zero-shot recall at 2 m (vehicles, ≤ 30 m); best B1 model's picked L2 at 3 s and collision rate | R (out of sample for R) | recall ≥ 0.30; L2 not worse than R beyond 2 SE; collision rate lower | all three | L2 worse than R beyond 2 SE |

**Pre-registered subsets** (P8):
1. a vehicle ahead in the ego lane within 30 m;
2. the human slowing (speed drops by more than 1 m/s over 2 s) or stopped (under 0.5 m/s);
3. any agent within 10 m of some candidate within the first 2 s.

## Controls

- **C:** the same scorer or refiner adaptation, the same loss and the same data, with no agents.
- **S:** agents from another sample, in another log: does any extra input help?
- **O:** ground-truth agents: what would better perception be worth?
- **Baselines for H-A:** a no-camera prior, plus stationary and constant-velocity paths.
- **For explanations:** deleting a random detected agent, and the release's own trained
  collision-agent head.
- Every run is also read at a fixed epoch as well as the selected one.

## Decision rules

- **If H-A is refuted,** no B claim is made. Fix the labels or the loss, or go to B3.
- **If B1 or B2 is supported:**
  - the smallest mode within 2 SE of the best one is the candidate;
  - P10 must hold before any claim about perception;
  - P11 must hold before any claim about explanations.
- **If B1 and B2 are refuted under every mode:** record it. The side-car's agents do not help this
  planner on NAVSIM, and the explanation route falls back to the release's own collision head.
- **If inconclusive:** NEUTRAL at this n. More seeds only if the seed range, not the SE, is what
  blocks the verdict.

## Selection

| Split | Used for |
|---|---|
| select: H04's 51 held-out navtrain logs | epochs, modes, thresholds |
| report: a fresh seeded draw of about 70 navtrain logs | every number above |
| navtest, nuScenes `val`, KITScenes, neuroNCAP | the gate, later. Nothing here selects on them |
