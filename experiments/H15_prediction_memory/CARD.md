<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H15 · Carrying the planner's output embeddings across moments makes the picker steadier and earlier

**Naren, Siddesh** · NAVSIM, neuroNCAP, Bench2Drive, AlpaSim, KITScenes, nuScenes · 📝 planned · 5 Nov 2026 · hypothesis

[← all experiments](../../README.md#hypothesis-experiments) · [the account](README.md) · [experiment.toml](experiment.toml)

**Builds on:** [H05](../H05_ssm_memory/CARD.md)

## Question

Released Drive-JEPA decides every moment from scratch: its 32 proposals and their scores are recomputed from one moment (now and 0.5 s ago) with nothing kept. On saved data (40 val logs, released model, momentum off, predicted scores rather than true grades) back-to-back picks are more than 1 m apart on 45% of moments while a path 0.34 m from the old pick is still on the menu, and the top two predicted scores differ by 0.003. Does a small memory of what the planner and picker concluded at earlier moments settle those near-ties the right way, and brake no later on the side crashes?

## Hypothesis

A selective state-space memory per path, fed that path's planner output embedding (the scorer input, 256 numbers), its six released score logits and the ego status, and carried to the closest proposal of the next moment (ego-motion aligned; no match beyond 2 m starts empty), nudges the released final score; the camera never reaches it and everything released stays frozen, so the 32 proposals are the released ones and a zero nudge reproduces released picks. Arms, same size, data order and training (3 seeds each): (a) memory carried, length 1, 2 or 4 moments chosen on the selection set, warm-up 1-16; (b) the same layer wiped every moment; (c) released. Momentum is off for all three arms offline and on for all three in the gate. Expected: (a) picks better-graded paths than (b) on the 214 val logs, with fewer flips and better two-frame comfort, nothing regresses against released on the gate, and on neuroNCAP's side pairs (a) scores above (b) and brakes no later. Refuted at the go/no-go unless the 3-seed mean of (a) - (b) in true PDM grade of the pick is above 0 with a CI excluding 0 (flips and comfort are secondary, reported with CIs); after the gate unless (a) beats (b), nothing regresses against released and braking starts no later. The effect size is stated in PREDICTIONS.md once the val logs are graded, before training.

## Predictions

[PREDICTIONS.md](PREDICTIONS.md), written before the run.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H15.r1**<br>true grades of the 32 released proposals at every graded navtrain moment (built and timed; not run) | H05.r1 | —; frozen: all of Drive-JEPA: encoder, generator and scorer | — | — | — |

- **H15.r1:** The vendor's training-time grader (compute_navsim_score: simulate, score_proposals, after_score_proposals) on H05.r1's cached proposals with the train metric cache, without the anchors load; states keep x, y relative to a float64 origin. Timed 5 Nov on 200 random train moments, cores 0-3, 4 workers: 0.065 s per moment wall (worker p50 0.22 s), 361 KB of metric cache read per moment, all values finite; projected 17 min for the 15,320 val moments and 1.7 h for all 91,977. Navtest check (200 picks of navsim/drivejepa/20261023T202240-de2f00 against its CSV): collisions, drivable area, direction, time to collision and history comfort agree exactly on 200/200; progress within 1e-3 on 199; lane keeping differs on 33 and traffic light on 3, always ours 0 against the CSV's 1, 30 of them explained by the evaluator's human filter (the human also fails), which the training grader does not apply.
