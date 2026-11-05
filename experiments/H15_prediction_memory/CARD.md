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
| **H15.r1**<br>true grades of the 32 released proposals at every graded navtrain moment | H05.r1 | —; frozen: all of Drive-JEPA: encoder, generator and scorer | — | — | — |
| **H15.r2**<br>matching each path to the previous moment's closest path (built; statistics on the val logs) | H05.r1 | —; frozen: all of Drive-JEPA: encoder, generator and scorer | — | — | — |
| **H15.r3**<br>the memory picker: selection of length and epochs, then (a) carried and (b) wiped, 3 seeds, evaluated with (c) released on the val logs | released | memory picker (new: per-path input MLP, keep gate, nudge MLP with a zero-started last layer); frozen: all of Drive-JEPA: encoder, generator and scorer | — | — | — |

- **H15.r1:** The vendor's training-time grader (compute_navsim_score: simulate, score_proposals, after_score_proposals) on H05.r1's cached proposals with the train metric cache, without the anchors load: the strict grade H15 trains on (Sid D22). Also stored per path: lane keeping and traffic light the benchmark's way (lane keeping checked every 0.1 s, not 0.5 s; either set to 1 where the human driver's own path fails it), and the human path's own scores. States keep x, y relative to a float64 origin. Navtest check (200 picks of navsim/drivejepa/20261023T202240-de2f00 against its CSV): the five gates-and-comfort sub-scores and the forgiving lane keeping and traffic light agree exactly on 200/200, progress within 1e-4 on 199; strict lane keeping (33) and traffic light (3) differ only ours 0 against the CSV's 1, all explained: 30 by the human filter, 6 by the coarse 0.5 s lane check, and the progress case by the human filter (the human fails driving direction). Tolerance (Sid D23) set on those same 200 tokens. Batches: val (214 logs), selection (100), rest (878).
- **H15.r2:** The previous moment's waypoints 2-8 moved into the current ego frame (logged poses), compared with the current waypoints 1-7 by mean distance; each new path takes its closest old path, or none beyond 2 m. Reproduces Naren's slot-stability numbers on the same 40 val logs exactly (3,927 pairs; closest 0.397 m, same slot 38.47%, pick unchanged 46.88%). On all 214 val logs (18,798 pairs), from the new path's side: closest 0.344 m median, same slot 29.1%, no old path within 2 m for 0.60% of paths, the pick 0.18%; not concentrated in turns. Chain starts are 11.2% of val moments but 6 of the 15,320 graded ones.
- **H15.r3:** Strict final grade as the BCE target (Sid D22), graded moments of the 878 train logs that are not selection logs; windows along chains, warm-up 1-16; 8 epochs; length and epoch count chosen on the 100 selection logs by the rule in PREDICTIONS.md; momentum off for every arm offline (Sid D10).
