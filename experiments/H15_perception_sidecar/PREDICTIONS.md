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
| P1 | A0 · **select** logs, front view, ≤ 30 m, 10k-sample probe (A0 chooses the tap, and its taps exist only for the select logs and 10k train samples; amended 3 Nov, before any run) | recall at 2 m (all classes), best of 3 taps | no-camera prior (0) | 0.45 (0.35–0.60); ViT-out 1024-d beats post-projector 256-d by ≥ 0.03 | the best tap ≥ 0.30 | every tap < 0.20 |
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

## The rerun, Part A (step 4, written 4 Nov 2026)

**Committed before step 5's first training step.** Copied from section 7 of the rerun's plan
(`docs/drivejepa-perception-sidecar-A.md`; every § below is the plan's).
- That section was committed in d1bacdc, before step 3's pipeline check, which read only the train and
  select logs.
- Q10 and Q11 were added on 4 Nov after the second review, with step 3's epoch-1 select values known
  (causal recall 0.70 under the strict rule, nearest-1 recall 0.56), as the plan says.

**The owner's choices** (plan section 11; 4 Nov, "use recommended settings"):
- 11.1: these criteria, as written. Nothing below was edited after that.
- 11.2: the report logs are the test set; navtest is not added.
- 11.3: no ego-motion input until after step 6.
- 11.4: every proposed metric. None was marked as recommended; all are readouts, not choices.
- 11.6: the step-5 runs as listed: 5a, then 5b.
- 11.7: the chosen causal rule adds a 4 s look-ahead at constant velocity and a 2 m corridor margin
  (`importance.CAUSAL_RULES["chosen"]`). Causal recall is read under it and under the strict rule.
- 11.8: pedestrian groups chain within 1.0 m, and no group is wider than 3 m.
- 11.10: traffic side is measured first, at steps 6 and 7.
- 11.12: step 7 reads nuScenes as it is and warped to NAVSIM's camera.
- 11.9 (step 5c, weighting what matters) and 11.11 (head changes) had no marked recommendation. They
  are not run in this round and stay open for the review of step 5.

### Gates: each must hold to go on

| # | Gate |
|---|---|
| G1 | every run's preflight passes, and its report is in `run.json` |
| G2 | each variant passes the overfit check before its first run |
| G3 | no run has a non-finite loss. Skipped (overflow) steps stay under 1% of steps. Clipped steps are reported |
| G4 | the owner has seen a run's curves (notebook 03) before its report numbers exist |
| G5 | v1 seed 0 (12 epochs) reproduces the first pass's A1 seed 0: select F1 at epoch 11 within ±0.01 of the 0.669 that run recorded there (its reread, quoted by B3.1's preflight in the README; its per-epoch files were deleted) |

### Predictions

**Scope.** Report logs, front view, within 30 m, unless stated; the chosen variant, three seeds.
**First-pass reference values** (A1, v1, report logs): recall 0.630–0.632, precision 0.736–0.742,
moving vehicles 0.935–0.936, lead-gap error 2.14–2.36 m against the prior's 8.53 m.

| # | Metric | Baseline | Predicted | Support | Refuted |
|---|---|---|---|---|---|
| Q1 | recall and precision at 2 m | the blind control | recall 0.60–0.70, precision 0.72–0.85 | recall ≥ 0.55 and precision ≥ 0.65 on every seed | recall < 0.40 on any seed |
| Q2 | recall of moving vehicles within 20 m | the blind control | ≥ 0.90 | ≥ 0.85 on every seed | < 0.70 on any seed |
| Q3 | per class: vehicle and pedestrian recall | — | vehicles ≥ 0.70, pedestrians 0.35–0.55 within 30 m | reported, not decided: the class mix differs between splits (§3.4) | — |
| Q4 | 1 s path FDE of matched moving agents | constant velocity from the side-car's own velocity | lower, by 3–8% | lower beyond 2 SE (clustered by log) on every seed | not lower on any seed |
| Q5 | lead-vehicle gap error | the train-mean prior | ≤ 2.5 m | ≤ half the prior's error on every seed | ≥ the prior's error |
| Q6 | sight: recall with another log's tokens | own tokens | ≤ 0.06 | ≤ 0.10 on every seed | > 0.20 on any seed |
| Q7 | variant (select, seed 0) | v1 | v2 beats v1 by 0.01–0.04 F1. v3 is within 0.01 of v2 | — (a choice by §6.5, recorded) | — |
| Q8 | schedule (select, v1 seed 0) | 12 epochs | 24 beats 12 by 0.005–0.02 F1 | — (a choice by §6.5, recorded) | — |
| Q9 | nuScenes, zero-shot: vehicle recall at 2 m within 30 m | — | 0.40–0.55 | ≥ 0.30 | < 0.20 |
| Q10 | causal recall (§3.8's rule as chosen at §11.7) | the blind control | 0.85–0.95 | ≥ 0.80 on every seed | < 0.70 on any seed |
| Q11 | recall of each sample's nearest target | — | 0.80–0.90 | reported, not decided | — |

**Q6's companion.** With the tokens zeroed, the first pass found 27 detections a sample on one
seed. That input never occurs in real data, so it is recorded as a diagnostic, not a gate.

**Q10 and Q11 fall under the scope line: within 30 m.** Step 5's per-epoch select records read causal
and nearest-k recall over all ranges, to 50 m, as notebook 04 §4e defined them. That is a diagnostic,
not these rows; step 6 reports both.

### How the choices at step 5 are made (plan §6.5, on select)

| Choice | Rule |
|---|---|
| Kept epoch | best F1 at 2 m within 30 m |
| Variant | the smallest variant within 0.01 F1 of the best, at its kept epoch, seed 0 |
| Schedule | 24 epochs if v1-24 beats v1-12 by ≥ 0.01 F1, else 12 |

### Decision rules

- **G5 fails:** stop. Find what differs between the passes before any other run.
- **Q1 refuted:** H-A is refuted for this design. Step 8 records it, and no B question opens.
- **Q1 to Q6 supported:** H-A holds on NAVSIM. Step 8 decides, with the owner, whether one B question
  opens, and which one.
  - Q9 says whether the side-car can be trusted out of domain.
- **Q9 refuted:** the side-car is NAVSIM-only. That limits any B question to NAVSIM and to the gate's
  in-domain benchmarks.
