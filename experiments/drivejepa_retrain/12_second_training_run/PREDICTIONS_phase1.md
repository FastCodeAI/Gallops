# S10 phase 1 (generator), version 2: predictions and rules, fixed before the pilots
Written by the research lead after tasks 2 to 4 and Sid's decision, before any S10 training step. An earlier draft was never hashed and no pilot ran under it.
Why version 2: plans depend on batch composition (three upstream lines). The car runs batch 1. All measurements below are batch-1 equivalent (batch fix on). Baseline = pass 9 at batch 1 on UNSEEN graded, from the dump `NAS/12_second_training_run/base_b1` launched in task 4 before any training; its tables are produced by `S10/s10_base_tables.py` and are not yet known to me.
Swing = fastest minus slowest of the 8 step speeds of one plan. Above 20 m/s = highway instants with car speed at least 20 m/s. Human-path error = training distance term to the human path, best of 32. Intervals: clip bootstrap, 95 percent.
Targets on UNSEEN, thresholds as in the goal file, measured once after the final checkpoint is fixed:
- T-G1 highway instants with a menu plan graded at least 0.95: at least 80 percent.
- T-G2 median swing, all 32 plans, above 20 m/s: under 5 m/s.
- T-G3 slots passing comfort on more than 25 percent of instants above 20 m/s: at least 24 of 32.
- W1 good-plan rate on ordinary and on close-pass not more than 2 points under the batch-1 pass-9 value.
- W2 human-path error on highway, close-pass and ordinary at most 1.05 x the batch-1 pass-9 value.
A tripwire failure on UNSEEN stops the run.
Pilots: P0 lambda 0, P1 0.1, P2 1, P3 10 with the batch fix; P4 lambda 0 with the batch fix and the forward reference range widened to the 99th percentile of the 4 s human forward distance. 1 pass each from pass 9, lr 1e-4, S4 data minus the 145 SEEN held-out clips, encoder and scorer frozen. w = lambda x C0 / S0 measured at pass 9 with the fix on. No capping; a non-finite run is disqualified; no other arm without an OK from Sid.
Choice on SEEN-held-out only. Reference: pass 9 at batch 1 (human-path error highway 1.311, close-pass 0.442, ordinary 0.367; median swing above 20 m/s 16.7). An arm qualifies if finite and its human-path error is at most 1.05 x the reference on each of the three slices. Choose the lowest median swing above 20 m/s. Ties within 0.5 m/s: prefer no range change over P4, then the smaller lambda. None qualifies: stop.
Then 2 passes from the chosen pilot at lr 2.5e-5, same settings. Final = among pilot, +1, +2 the lowest held-out swing that still qualifies. UNSEEN opens for the new model only after that choice is on disk.
Predictions: P0 alone brings held-out swing above 20 m/s from 16.7 to under 8 (60 percent sure). The best arm is under 5 (50). Every finite arm improves highway human-path error below 1.311 (80). P3 goes non-finite or fails the human-path rule (40). P4 has lower highway human-path error than P0 (60). On UNSEEN: T-G2 met (55), T-G3 met (45), T-G1 not met, expected 65 to 78 percent; W1 holds (75), W2 holds (75).

## Measured before the pilots

Appended by the task-5 worker on 2026-10-09 after gates Ga, Gb, Gc and before any pilot was started. Nothing above this heading was changed.

### spacing_norm_run.json (pass 9, batch fix on, standard range, 300 batches of the pass-1 sampler, GPU, 16-bit forward and float32 loss, eval mode)

C0 = 0.139491, H0 = 0.535095, S0 = 44.6967 m^2 (per batch: median 22.98, 99th percentile 197.0, largest 274.6, smallest 0.235).
w = lambda x C0 / S0: lambda 0.1 -> 0.000312084, lambda 1.0 -> 0.00312084, lambda 10.0 -> 0.0312084. P0 and P4 use w = 0.

| slice of the draws | draws | human part | catalogue part | spacing^2 mean, m^2 | spacing^2 median, m^2 |
|---|---|---|---|---|---|
| highway | 743 | 1.5934 | 0.8903 | 573.618 | 30.085 |
| close-pass | 261 | 0.4420 | 0.3513 | 0.281 | 0.214 |
| ordinary | 8001 | 0.4496 | 0.0732 | 0.323 | 0.230 |
| pull-out (ungraded) | 595 | 0.4043 | 0.0000 | 0.391 | 0.341 |
| all | 9600 | 0.5351 | 0.1395 | 44.697 | 0.255 |
| highway, car >= 20 m/s | 459 | 2.1870 | 1.1198 | 927.532 | 579.940 |

### ref_range.json

X99 = 114.1800 m = 99th percentile of the human path's forward distance at 4 s over the 122581 training examples (3870 clips). P4 maps forward -32 m .. 114.18 m onto 0 .. 1 (the upstream value times 0.43782); lateral axis unchanged. Percentiles of the forward distance, m: p0 -15.1, p1 -0.0, p50 30.3, p90 56.2, p95 62.0, p99 114.2, p99.9 145.9, p100 160.6. Share above 32 m: 47.1 percent.

### Gb: pass 9 on SEEN held-out, batch 32 in clip order, batch fix on, GPU 16-bit

| range | slice | examples / clips | human-path term | best of 32 at 4 s, mean / median, m | within 1.5 m | median swing, m/s | plans > 5 m/s | spacing^2 mean / median, m^2 |
|---|---|---|---|---|---|---|---|---|
| standard (-32..32 m) | highway | 1047 / 33 | 1.311 | 0.977 / 0.719 | 81.7 % | 6.69 | 62.1 % | 401.30 / 9.55 |
| standard (-32..32 m) | close-pass | 547 / 19 | 0.442 | 0.529 / 0.417 | 96.2 % | 2.37 | 10.5 % | 0.31 / 0.24 |
| standard (-32..32 m) | ordinary | 2806 / 93 | 0.367 | 0.460 / 0.297 | 95.5 % | 2.30 | 8.3 % | 0.32 / 0.23 |
| standard (-32..32 m) | all | 4400 / 145 | 0.601 | 0.592 / 0.387 | 92.3 % | 2.77 | 21.4 % | 95.73 / 0.29 |
| standard (-32..32 m) | highway, car >= 20 m/s | 575 / 27 | 1.878 | 1.172 / 0.851 | 74.6 % | 16.69 | 90.8 % | 728.95 / 262.13 |
| wide (-32..114.2 m), weights untouched | highway | 1047 / 33 | 1.103 | 0.789 / 0.624 | 88.5 % | 5.23 | 54.2 % | 5.19 / 4.59 |
| wide (-32..114.2 m), weights untouched | close-pass | 547 / 19 | 1.211 | 1.034 / 0.595 | 81.0 % | 6.97 | 71.7 % | 10.70 / 9.85 |
| wide (-32..114.2 m), weights untouched | ordinary | 2806 / 93 | 0.949 | 0.761 / 0.421 | 87.2 % | 6.45 | 64.1 % | 9.93 / 8.67 |
| wide (-32..114.2 m), weights untouched | all | 4400 / 145 | 1.018 | 0.802 / 0.498 | 86.8 % | 6.15 | 62.7 % | 8.90 / 6.48 |
| wide (-32..114.2 m), weights untouched | highway, car >= 20 m/s | 575 / 27 | 1.157 | 0.804 / 0.663 | 88.5 % | 4.96 | 49.2 % | 4.84 / 4.75 |

### Ga: batch 32 shuffled against batch 1, batch fix on, GPU, pass 9, 64 highway rows above 20 m/s and 64 ordinary

float32: largest plan difference 4.86 cm (highway; 1 of 64 highway rows above 2 cm, median over rows 0.86 mm; ordinary largest 0.64 mm). The expectation was under 2 cm.
16-bit autocast: largest 17.25 m (highway median over rows 1.23 m, ordinary largest 0.28 m); batch 1 in 16-bit against batch 1 in float32 differs by up to 9.65 m (highway median 1.24 m).
Fix on against fix off at batch 1: 0.0 (float32 and 16-bit, both ranges).

### Gc: 200-step smokes (not pilots; run names smoke_fix_l10, smoke_fix_l10_b, smoke_fixwide_l0)

Loss finite in all three. Lambda 10: w x spacing was 2.7 percent of the loss at step 1, 4.0 / 5.6 percent at step 50 (two runs), 2.1 / 3.1 percent at step 200; in the first 50 steps GradScaler skipped 3 updates for a non-finite 16-bit gradient and lowered its factor from 65536 to 8192, none after. Scorer and encoder tensors bit-identical after, the 4 refiner entries one object.
