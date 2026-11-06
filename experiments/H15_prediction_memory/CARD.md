<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H15 · Carrying the planner's output embeddings across moments makes the picker steadier and earlier

**Naren, Siddesh** · NAVSIM, neuroNCAP, Bench2Drive, AlpaSim, KITScenes, nuScenes · ⏳ running · 5 Nov 2026 · hypothesis

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
| **H15.r3**<br>the memory picker: selection of length and epochs, then (a) carried and (b) wiped, 3 seeds, evaluated with (c) released on the val logs | released | memory picker (new: per-path input MLP, keep gate, nudge MLP with a zero-started last layer); frozen: all of Drive-JEPA: encoder, generator and scorer | navsim `navtrain: the 878 train logs that are not H15 selection logs` 69,650 samples (train) — graded moments; windows reach back along chains into ungraded moments<br>navsim `navtrain: the 100 H15 selection logs` 7,007 samples (selection) — chose the memory length and the epoch count<br>navsim `navtrain: the 214 vendor val logs` 15,320 samples (eval) — reported on; graded moments in 179 of the 214 logs; the bootstrap resamples all 214 | commit `cdacc3871df01d87547cb707391c4b62b05c3e57` | — |
| **H15.r3-a-s0**<br>(a) memory carried, length 4, seed 0: the checkpoint H15.r3 evaluates | released | memory picker, 0.333 M parameters (0.3333 M); frozen: all of Drive-JEPA: encoder, generator and scorer | — | commit `cdacc3871df01d87547cb707391c4b62b05c3e57` | `{weights}/drivejepa_memory/h15-r3-a-L4-s0/step_000001638/`<br>sha256 `0679ea453b9714dbc524d4d8435619572fac784ca006b9935adffbec6a5456b4` |
| **H15.r3-a-s1**<br>(a) memory carried, length 4, seed 1: the checkpoint H15.r3 evaluates | released | memory picker, 0.333 M parameters (0.3333 M); frozen: all of Drive-JEPA: encoder, generator and scorer | — | commit `cdacc3871df01d87547cb707391c4b62b05c3e57` | `{weights}/drivejepa_memory/h15-r3-a-L4-s1/step_000001638/`<br>sha256 `c23859b7798b95c68afeb926eadeba668b44c5a38867f38c7f61f35b54dbfdd7` |
| **H15.r3-a-s2**<br>(a) memory carried, length 4, seed 2: the checkpoint H15.r3 evaluates | released | memory picker, 0.333 M parameters (0.3333 M); frozen: all of Drive-JEPA: encoder, generator and scorer | — | commit `cdacc3871df01d87547cb707391c4b62b05c3e57` | `{weights}/drivejepa_memory/h15-r3-a-L4-s2/step_000001638/`<br>sha256 `df6298b2f15e95ea6b3ac60d2957de6533b7763f1fcae1df110c18b226c0e409` |
| **H15.r3-b-s0**<br>(b) the same layer, wiped every moment, seed 0: the checkpoint H15.r3 evaluates | released | memory picker, 0.333 M parameters (0.3333 M); frozen: all of Drive-JEPA: encoder, generator and scorer | — | commit `cdacc3871df01d87547cb707391c4b62b05c3e57` | `{weights}/drivejepa_memory/h15-r3-b-L0-s0/step_000001638/`<br>sha256 `dfd944c1c3f1ceabae1cfdfc2bf63c55c19872b887ad2ffd3909da2c29ed8bca` |
| **H15.r3-b-s1**<br>(b) the same layer, wiped every moment, seed 1: the checkpoint H15.r3 evaluates | released | memory picker, 0.333 M parameters (0.3333 M); frozen: all of Drive-JEPA: encoder, generator and scorer | — | commit `cdacc3871df01d87547cb707391c4b62b05c3e57` | `{weights}/drivejepa_memory/h15-r3-b-L0-s1/step_000001638/`<br>sha256 `7db7b315efef84aa418e44dd69a2fb5141374ff8d5f20425bf574e0aa7b110ca` |
| **H15.r3-b-s2**<br>(b) the same layer, wiped every moment, seed 2: the checkpoint H15.r3 evaluates | released | memory picker, 0.333 M parameters (0.3333 M); frozen: all of Drive-JEPA: encoder, generator and scorer | — | commit `cdacc3871df01d87547cb707391c4b62b05c3e57` | `{weights}/drivejepa_memory/h15-r3-b-L0-s2/step_000001638/`<br>sha256 `deb69754a231eb352fc5924686d8ab220a980633a117996d15749bc3b0152004` |

- **H15.r1:** The vendor's training-time grader (compute_navsim_score: simulate, score_proposals, after_score_proposals) on H05.r1's cached proposals with the train metric cache, without the anchors load: the strict grade H15 trains on (Sid D22). Also stored per path: lane keeping and traffic light the benchmark's way (lane keeping checked every 0.1 s, not 0.5 s; either set to 1 where the human driver's own path fails it), and the human path's own scores. States keep x, y relative to a float64 origin. Navtest check (200 picks of navsim/drivejepa/20261023T202240-de2f00 against its CSV): the five gates-and-comfort sub-scores and the forgiving lane keeping and traffic light agree exactly on 200/200, progress within 1e-4 on 199; strict lane keeping (33) and traffic light (3) differ only ours 0 against the CSV's 1, all explained: 30 by the human filter, 6 by the coarse 0.5 s lane check, and the progress case by the human filter (the human fails driving direction). Tolerance (Sid D23) set on those same 200 tokens. Batches: val (214 logs), selection (100), rest (878).
- **H15.r2:** The previous moment's waypoints 2-8 moved into the current ego frame (logged poses), compared with the current waypoints 1-7 by mean distance; each new path takes its closest old path, or none beyond 2 m. Reproduces Naren's slot-stability numbers on the same 40 val logs exactly (3,927 pairs; closest 0.397 m, same slot 38.47%, pick unchanged 46.88%). On all 214 val logs (18,798 pairs), from the new path's side: closest 0.344 m median, same slot 29.1%, no old path within 2 m for 0.60% of paths, the pick 0.18%; not concentrated in turns. Chain starts are 11.2% of val moments but 6 of the 15,320 graded ones.
- **H15.r3:** Strict final grade as the BCE target (Sid D22), graded moments of the 878 train logs that are not selection logs; windows along chains, warm-up 1-16; 8 epochs; length and epoch count chosen on the 100 selection logs by the rule in PREDICTIONS.md; momentum off for every arm offline (Sid D10). Selection (seed 0): L = 4 and 6 epochs by the rule; the three lengths' best selection losses differ by 5e-6 (0.288801 / 0.288804 / 0.288806), so the length is a coin toss; released 0.30156. Preflight passed (untrained = released on 200/200 val moments, carry = reset on 256 chain starts). Trained on GPU with cores 0-1 (Sid D27), 2.7 min for the six final runs. Checkpoints are runs H15.r3-a-s0 ... H15.r3-b-s2. Evaluated on the 214 val logs (15,320 graded moments in 179 of them), momentum off, 3-seed mean, log-clustered paired bootstrap, 2,000 draws. Verdict: P1 supported but not useful; (a) level with released on grade, -3.0 pts on two-frame comfort; matching rule: go to gate.
- **H15.r3-a-s0:** Trained by H15.r3 (its code, data and results); 6 epochs of 273 optimizer steps, batch 256, AdamW lr 3e-4. Sidecar notes: mode carry, length 4, seed 0, preflight_passed.
- **H15.r3-a-s1:** Trained by H15.r3 (its code, data and results); 6 epochs of 273 optimizer steps, batch 256, AdamW lr 3e-4. Sidecar notes: mode carry, length 4, seed 1, preflight_passed.
- **H15.r3-a-s2:** Trained by H15.r3 (its code, data and results); 6 epochs of 273 optimizer steps, batch 256, AdamW lr 3e-4. Sidecar notes: mode carry, length 4, seed 2, preflight_passed.
- **H15.r3-b-s0:** Trained by H15.r3 (its code, data and results); 6 epochs of 273 optimizer steps, batch 256, AdamW lr 3e-4. Sidecar notes: mode reset, length 0, seed 0, preflight_passed.
- **H15.r3-b-s1:** Trained by H15.r3 (its code, data and results); 6 epochs of 273 optimizer steps, batch 256, AdamW lr 3e-4. Sidecar notes: mode reset, length 0, seed 1, preflight_passed.
- **H15.r3-b-s2:** Trained by H15.r3 (its code, data and results); 6 epochs of 273 optimizer steps, batch 256, AdamW lr 3e-4. Sidecar notes: mode reset, length 0, seed 2, preflight_passed.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H15.r3 ★ | offline P1: mean strict PDM grade of the pick, (a) carried L=4 0.86715 → **0.86827** (n=15,320 samples)<br>(a) - (b) +0.00112, 95% CI [+0.00020, +0.00212], excludes 0 but below the +0.005 usefulness bar: supported but not useful (predicted +0.01). Per seed +0.00078 / +0.00020 / +0.00237. Released (c) 0.86880: (a) - (c) -0.00052 [-0.00298, +0.00206] | (b) wiped, same layer | ●●●○ strong · open-loop | {data}/memory_cache/h15/evaluate.json (navtrain cache); write-up MEMORY_H15_STEP3.md (Siddesh, 5 Nov) |
| H15.r3 | offline P2a: flip rate (back-to-back picks more than 1 m apart), (a) carried 0.4788 → 0.4590 (n=18,798 pairs)<br>(a) - (b) -0.0198 [-0.0234, -0.0162]; released (c) 0.4571, (a) - (c) +0.0019 [-0.0063, +0.0096] | (b) wiped, same layer | ●●●○ strong · open-loop | {data}/memory_cache/h15/evaluate.json (navtrain cache); write-up MEMORY_H15_STEP3.md (Siddesh, 5 Nov) |
| H15.r3 | offline P2b: two-frame extended comfort pass rate, (a) carried 0.3759 → 0.3962 (n=12,768 pairs)<br>(a) - (b) +0.0204 [+0.0162, +0.0250]; released (c) 0.4261: (a) - (c) -0.0299 [-0.0415, -0.0196] and (b) - (c) -0.0503 [-0.0635, -0.0391], both trained arms less comfortable than released | (b) wiped, same layer | ●●●○ strong · open-loop | {data}/memory_cache/h15/evaluate.json (navtrain cache); write-up MEMORY_H15_STEP3.md (Siddesh, 5 Nov) |
| H15.r3 | offline R1: picks a braking path where a best path brakes and released does not, (a) carried 0.1465 → 0.1264 (n=546 samples)<br>read-out, no prediction; (a) - (b) -0.0202 [-0.0355, -0.0047]; released 0 by the subset's definition. 'better' is not pinned: braking here is only good where the braking path is the best one, which these moments are | (b) wiped, same layer | ●●○○ moderate · open-loop | {data}/memory_cache/h15/evaluate.json (navtrain cache); write-up MEMORY_H15_STEP3.md (Siddesh, 5 Nov) |
| H15.r3 | offline pick is a best path, (a) carried 0.3947 → 0.3905 (n=15,320 samples)<br>(a) - (b) -0.0042 [-0.0061, -0.0024]; released 0.3582, (a) - (c) +0.0323 [+0.0275, +0.0373]; mean grade does not rise with it | (b) wiped, same layer | ●●●○ strong · open-loop | {data}/memory_cache/h15/evaluate.json (navtrain cache); write-up MEMORY_H15_STEP3.md (Siddesh, 5 Nov) |

## Regression gate

**H15.r3-a-s0** — 6 Nov 2026, quick tier, against released Drive-JEPA, the queue baselines {queue}/baselines/drivejepa.toml (job 4b7a and earlier; not re-run): ⏸ **HOLD**

| Benchmark | Metric | Baseline | Candidate | t | MDE | Units | Verdict |
|---|---|---|---|---|---|---|---|
| KITScenes | ADE (m) | 2.537 | 2.292 | 2.78 | 0.25 | 117 | 🟢 GAIN |
| nuScenes | L2 avg (m) | 2.431 | 2.336 | 1.59 | 0.169 | 143 | ⚪ NEUTRAL |
| NAVSIM | EPDMS | 0.857 | 0.858 | 0.13 | 0.011 | 126 | ⚪ NEUTRAL |
| neuroNCAP | NCAP score (quick: 3 pairs, cannot ship) | 3.179 | 2.533 | -0.71 | 4.866 | 3 | ⚪ NEUTRAL |
| Bench2Drive | driving score (route 3717, crash check) | 37.670 | 28.141 |  |  | 1 | ▫️ TOO FEW UNITS |
| AlpaSim | scene score (1acf2f5d, crash check) | 0.929 | 0.967 |  |  | 1 | ▫️ TOO FEW UNITS |

By hand from a throwaway paddock worktree (Sid D28-D30), momentum on, one lane. neuroNCAP quick first failed to start its renderer (port 8900 held by another user's process) and was re-run on port 8950 (Sid). The Bench2Drive and AlpaSim crash checks ran cleanly (one unit: no verdict). Beside the scores, NAVSIM's no-at-fault-collisions term is a REGRESSION, 0.993 to 0.988 (t -2.19); per-term pairs never decide the gate.

**H15.r3-a-s0** — 6 Nov 2026, closed tier, against released Drive-JEPA, the queue baselines {queue}/baselines/drivejepa.toml (job 4b7a and earlier; not re-run): ⏸ **HOLD**

| Benchmark | Metric | Baseline | Candidate | t | MDE | Units | Verdict |
|---|---|---|---|---|---|---|---|
| neuroNCAP | NCAP score (15 pairs) | 3.679 | 3.404 | -1.34 | 0.621 | 15 | ⚪ NEUTRAL |
| Bench2Drive | driving score (7 routes, cannot ship) | 59.036 | 68.512 | 0.89 | 35.664 | 7 | ⚪ NEUTRAL |
| AlpaSim | scene score (10 gate scenes) | 0.434 | 0.394 | -1.19 | 0.107 | 10 | ⚪ NEUTRAL |

Three lanes. AlpaSim on alpasim-base 9ee8dfb8 (rebuilt 3 Nov), checked first: released on 1acf2f5d gave 0.9294890766651916, the baseline's exact value (Naren's rule). P4 side pairs (braking start from ego poses): side-0103 score 0.00, did not brake, impact 8.71 m/s (released 2.44, 1.5 s, 3.17 m/s); side-0110 3.04, 0.6 s, 2.28 m/s (released 5.0, 0.2 s, none). AlpaSim at-fault collisions 3 against released 2 (new: d1a25a99).

**H15.r3-b-s0** — 6 Nov 2026, quick tier, against released Drive-JEPA, the queue baselines {queue}/baselines/drivejepa.toml (job 4b7a and earlier; not re-run): ⏸ **HOLD**

| Benchmark | Metric | Baseline | Candidate | t | MDE | Units | Verdict |
|---|---|---|---|---|---|---|---|
| KITScenes | ADE (m) | 2.537 | 2.261 | 3.17 | 0.246 | 117 | 🟢 GAIN |
| nuScenes | L2 avg (m) | 2.431 | 2.288 | 2.53 | 0.16 | 143 | 🟢 GAIN |
| NAVSIM | EPDMS | 0.857 | 0.856 | -0.32 | 0.01 | 126 | ⚪ NEUTRAL |
| neuroNCAP | NCAP score (quick: 3 pairs, cannot ship) | 3.179 | 4.213 | 1.32 | 4.186 | 3 | ⚪ NEUTRAL |
| Bench2Drive | driving score (route 3717, crash check) | 37.670 | 16.400 |  |  | 1 | ▫️ TOO FEW UNITS |
| AlpaSim | scene score (1acf2f5d, crash check) | 0.929 | 1.000 |  |  | 1 | ▫️ TOO FEW UNITS |

As (a), same commands and site. No REGRESSION, terms included.

**H15.r3-b-s0** — 6 Nov 2026, closed tier, against released Drive-JEPA, the queue baselines {queue}/baselines/drivejepa.toml (job 4b7a and earlier; not re-run): ⏸ **HOLD**

| Benchmark | Metric | Baseline | Candidate | t | MDE | Units | Verdict |
|---|---|---|---|---|---|---|---|
| neuroNCAP | NCAP score (15 pairs) | 3.679 | 3.531 | -0.38 | 1.17 | 15 | ⚪ NEUTRAL |
| Bench2Drive | driving score (7 routes, cannot ship) | 59.036 | 45.964 | -1.23 | 35.714 | 7 | ⚪ NEUTRAL |
| AlpaSim | scene score (10 gate scenes) | 0.434 | 0.508 | 0.68 | 0.342 | 10 | ⚪ NEUTRAL |

AlpaSim first INCOMPLETE: 2 of 10 scenes (e121e37d, d2715023) failed before driving (bridge not accepting connections within 60 s); completed on 6 Nov 23:15-23:30 IST by resuming the same run in place (paddock run --resume, the 8 finished scenes reused, Sid), then the gate re-checked the tier (gate_closed_b_recheck). P4 side pairs: side-0103 5.00, braking 0.9 s, no impact; side-0110 0.00, did not brake, impact 9.98 m/s. AlpaSim at-fault collisions: d1a25a99, b44a4a44 (released: b44a4a44, 8023dce5).
