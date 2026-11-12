<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H17 · A memory read by both the generator and the scorer, trained with them, picks better paths

**Siddesh** · offline · ⏸ parked (negative) · 7 Nov – 12 Nov 2026 · hypothesis

[← all experiments](../../README.md#hypothesis-experiments) · [the account](README.md) · [experiment.toml](experiment.toml)

**Builds on:** [H05](../H05_ssm_memory/CARD.md), [H15](../H15_prediction_memory/CARD.md)

## Question

H05 found that the camera's past, read through a small probe, adds little; H15 carried only the planner's own outputs and could not be told from the same layer wiped. Both left the released model frozen. Does a memory help when the two parts that decide the drive, the generator that proposes 32 paths and the scorer that picks one, are trained to use it?

## Hypothesis

A memory of 16 slots, written at every moment from the frozen encoder's output and read by both the generator and the scorer, trained together with them from Drive-JEPA 0.1.8b on Physical AI AV, makes the scorer pick better paths than the same model with the memory wiped at every moment. Expected: a higher mean grade of the picked path on the picking set, and at least +0.01 on moments within 5 s after a cut-in, a crossing or a hard brake. Refuted if the carried memory does not beat the wiped one.

## Predictions

⚠️ Not pre-registered: the rule for each run was written before it started (file times and run logs agree) but in a folder outside git, so no commit date shows it

## What we learned

Camera memory has a low ceiling on this data. With a healthy memory (slots distinct, gates open, earlier views recalled), carrying it gave a pick grade of 0.7313 against 0.7355 with the same weights and the memory reset every moment, on 1,131 graded moments: no gain. The first training run's gap in favour of the carried memory (0.7752 against 0.7639, 95 % interval -0.008 to +0.032) came from a collapsed memory whose 16 slots were 99.96 % alike, so it is not the memory working. A direct test found the trained memory adds nothing for predicting the driver beyond the current view and the car's own motion history. Given the true past camera view directly, a linear reader gains 1 to 2 % on speed and nothing on braking or steering. The car's own motion history is the stronger signal, and is a separate, later experiment. Limits: one seed, one data set, open loop only; the 12-scene closed-loop check of carried against reset had not finished when this was recorded.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H17.r1**<br>token cache: the frozen encoder's output at every moment of four slices | 3d566eb211b5 | —; frozen: all of Drive-JEPA 0.1.8b: encoder, generator and scorer | physicalai_av `training slice of the earlier retrain` 126,981 moments (train) — 4,015 clips<br>physicalai_av `picking set` 2,591 moments (selection) — 84 clips; 1,131 of the moments are graded<br>physicalai_av `graded held-out` 10,516 moments (held-out) — 355 clips; cached and never opened<br>physicalai_av `fresh raw held-out` 9,566 moments (held-out) — 300 clips; cached and never opened | — | — |
| **H17.r2**<br>first training run: memory carried against memory wiped, two passes, one seed | 3d566eb211b5 | generator, scorer, memory blocks (new, 1.67 M parameters); frozen: encoder (its output is read from the cache) | physicalai_av `training slice of the earlier retrain` 126,981 moments (train)<br>physicalai_av `picking set, graded moments` 1,131 moments (selection) | — | `{weights}/drivejepa_physicalai_memory/h17-r2-carried/step_000005218/`<br>sha256 `70b37896e5b1369a093195cfa4c46b62749200fdb29b6712e0f24d5ba604a5cd` |
| **H17.r3**<br>the memory-collapse fix: a learned start state and a gate per slot, 930 steps | 3d566eb211b5 | generator, scorer, memory blocks (version 2, 1.67 M parameters); frozen: encoder (its output is read from the cache) | — | — | `{weights}/drivejepa_physicalai_memory/h17-r3-collapse-fix/step_000000930/`<br>sha256 `645157874f555f922160cde67040e1f8f067954d0adf95ce2839dd6d3755d9fa` |
| **H17.r4**<br>the recall lesson: the memory is also taught to recall the view 2 s and 4 s ago, 930 steps | 3d566eb211b5 | generator, scorer, memory blocks (version 3, 1.67 M parameters), a training-only read-out (0.40 M parameters); frozen: encoder (its output is read from the cache) | — | — | `{weights}/drivejepa_physicalai_memory/h17-r4-recall-lesson/step_000000930/`<br>sha256 `84359783ef502e1ccc3a2dcff013bf45d215f9fb11e046cc8110c64581434e21` |

- **H17.r1:** 88.6 GB, fp32, one file per clip, under {experiment_store}/cache; exact against a full forward pass at encoder batch 1 on 200 moments per slice. The starting model is identified by its sha256 only (3d566eb2...8b8e66, POINTERS.md): its own training record is not in this repository. The code was in a folder outside git when it ran, so there is no train_git_sha; AS_RUN.sha256 holds each file's hash as it ran.
- **H17.r2:** Two arms from the same start, same batch order: the memory carried from moment to moment, and the same layer wiped at every moment. The chosen check of each arm is the best pick grade among its ten checks on the picking set, which is also the set reported on. The checkpoint pinned here is the carried arm's; the wiped arm's is in POINTERS.md. The memory collapsed in this run (see the first check above).
- **H17.r3:** Memory carried, the first 930 steps of the same batch order. The slots became distinct (mean pairwise cosine 0.44) and the gates spread, but the run failed its own health rule on one condition: the memory changed by 0.042 per moment against a bar of 0.05.
- **H17.r4:** Memory carried, the first 930 steps of the same batch order. Every memory-health condition written beforehand holds: slots distinct, gates open, and the carried memory recalls the view 2 s and 4 s back better than a memory reset each moment (6.0 % and 10.5 % lower recall error). The run failed its rule on one condition only: the pick grade, 0.7313 against a bar of 0.74.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H17.r2 | offline mean grade of the picked path, first training run: memory carried vs wiped 0.7639 → 0.7752 (n=1,131 moments)<br>Memory collapsed in this run (slots 99.96 % alike), so the gap is not the memory working. Gap +0.0113, 95 % interval -0.0076 to +0.0315 (46 clips resampled, 2,000 draws): it crosses zero. The starting model scores 0.7685 on the same moments. On the 34 moments after an event the carried arm is lower, 0.9574 against 0.9674, so the rule written beforehand said stop. One seed. | the same model trained with its memory wiped at every moment | ●●○○ moderate · open-loop, selected-on-report | experiments/H17_generator_scorer_memory/04_carried_vs_wiped_training_run/tables/pilot_final.json: comparison.overall |
| H17.r3 | offline mean grade of the picked path after the collapse fix: memory carried vs reset every moment 0.7520 → 0.7496 (n=1,131 moments)<br>Exact values 0.74957 and 0.75203. Step 930 was fixed before the run. One seed. | the same weights, memory reset to its start state at every moment | ●●○○ moderate · open-loop | experiments/H17_generator_scorer_memory/05_memory_collapse_fix/tables/build4_short.json: step930 |
| H17.r4 ★ | offline mean grade of the picked path: memory carried vs reset every moment 0.7355 → **0.7313** (n=1,131 moments)<br>Picking set, healthy memory. Exact values 0.731322 carried and 0.735510 reset: carried minus reset is -0.004189. Not selected on this set: step 930 was fixed before the run, and both numbers are the same checkpoint. No interval was computed. One seed. The starting model scores 0.7685 on the same moments, so both are below where training began. | the same weights, memory reset to its start state at every moment | ●●○○ moderate · open-loop | experiments/H17_generator_scorer_memory/06_recall_lesson/tables/build5_short.json: step930.carried.check.pick_grade_mean_map, step930.reset.check.pick_grade_mean_map |

## Checks that decided something

- **Was the memory in the first training run working?** (10 Nov): No. At the carried arm's chosen check the 16 slots were 99.96 % alike (mean pairwise cosine 0.9996) and the gates were almost shut (mean 0.006 per slot). → fix the memory before reading anything from the gap between the arms (H17.r3, H17.r4)
- **Does the trained memory hold information that matters for driving?** (12 Nov): No, by the rule written beforehand. Adding the carried memory predicts the driver's next 3 s no better than adding the wiped memory on any of six targets; both controls pass. On top of the current view and the car's own history it adds -2.8 % to +0.6 %. → ask whether the true past camera view helps at all
- **Does the true past camera view help predict the driver?** (12 Nov): Unclear, by the rule written beforehand. Views from 2 s and 4 s ago add at most 1.2 to 1.3 % on speed at +3 s (bar 2 %) and 1.7 % on event moments (bar 5 %); the linear reader's gains on speed exclude zero; nothing on braking or steering. → park the camera memory; the car's own motion history is a separate experiment
- **12 scenes in closed loop, memory carried against memory reset** (12 Nov): Not finished when this was recorded: the rule was written down and the adapter proofs had started, but no scene had been driven.
