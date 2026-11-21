# Link-2 test — pre-registration

Written 2026-11-12 00:49 IST (file time stamp), **before any probe was run on real data**. At this time: the sample is chosen, the forward replays are
finished, the probe code has been run only on synthetic data (`out/link2_selftest.json`). No probe output exists
(`{experiment_store}/runs/link2/probe/` does not exist). The target values of the sample have not been looked at; their
statistics (`out/link2_stats.json`) are computed by the same job chain immediately before the probes and cannot change anything below.

Code fixed at this point (sha256): `src/s10_probe.py` 80bea062…ccefd0, `src/s10_link2.py` d5059ebc…a9cbb,
`out/link2_sample.json` c14c85a1…7350b.

## Question

Carried memory barely changes the picks. Is that because the memory holds nothing driving needs (link 2 broken), or because it holds
useful information that the generator and the scorer do not use (link 2 fine)? Test: from features at moment t, predict what the human
driver does next; does adding the CARRIED memory predict better than adding the WIPED memory?

No training of the model, no held-out slice, no closed loop. Forward replays and small probes only.

## Model and checkpoints

- Starting model sha256 3d566eb2…2b8e66, verified at load, 495/495 tensors equal (`runs/link2/replay.json`, key `agent`).
- Main: build 5, `runs/build5_short/ckpt/step0930.pt` (model version 3).
- Information only: build 4 `runs/build4_short/ckpt/step0930.pt` (version 2); pilot `runs/pilot_seed0/carry/ckpt/p00.2.pt` (version 1).

## Samples

- **Primary**: 300 training clips not used in the first 930 steps of the seeded pass plan
  (`s6_data.plan_pass(store, 0, 1, True, mem)`, the call the runs made at `src/s9_short.py:166`). Selection: `src/s10_link2.py:33-61`.
  1,611 clips are used by those 930 steps; the unused pool is 2,403 clips (ordinary 1,773, reconstructed 468, pull-out 162);
  the sample is proportional by type with seed 20261111: **221 ordinary, 59 reconstructed, 20 pull-out**. Clip list:
  `out/link2_sample.json` (key `clips`) and `out/link2_sample.csv`. 9,454 moments replayed, **8,191 after the warm-up**
  (`runs/link2/replay.json`). 27 of the 300 clips hold event moments.
- **Secondary**: the 84 picking clips (2,591 moments replayed, 2,220 after the warm-up; only those flagged as picking moments are used).
- Replay (`src/s10_link2.py:64-85`): every clip in order from its first moment, batch 1, same weights, two modes — carried (memory
  written at t is read at t+1) and wiped (memory reset to the start state before every moment). A missing anchor restarts the chain.
- Moments used: those with at least 4 earlier moments in their chain (`src/s10_probe.py:170`).
- Event moments: rows of `out/memory_moments_train.parquet` / `out/memory_moments_picking.parquet` (within 5 s after an event), matched
  on clip and anchor time (`src/s10_probe.py:173-175`).

## Targets (human future, 8 waypoints at 0.5 s; waypoint j is the pose at (j+1)·0.5 s in the ego frame)

All in `src/s10_probe.py:40-49`.

| | target | formula | line |
|---|---|---|---|
| T1–T3 | speed at +1, +2, +3 s | ‖xy[2k] − xy[2k−2]‖ / 1.0 s for k = 1, 2, 3 (centred difference of the poses at k+0.5 s and k−0.5 s) | :43 |
| T4 | braking onset within 3 s (0/1) | segment speeds s_i = ‖displacement of 0.5 s segment i‖ / 0.5 for the 6 segments up to 3.0 s (first segment from the origin); deceleration d_i = (s_i − s_{i+1}) / 0.5; T4 = 1 if any of the 5 d_i ≥ 2.0 m/s² | :44-46, :48 |
| T5 | lateral offset at +3 s | y of waypoint index 5 | :49 |
| T6 | heading change at +3 s | heading of waypoint index 5, wrapped to (−π, π] | :47 |

T4 note: "deceleration ≥ 2 m/s² for ≥ 0.5 s" is read as a mean deceleration of at least 2 m/s² across one 0.5 s span between
consecutive segment centres. Moments where the driver is already braking at t count as positive (not only fresh onsets).

## Feature sets

| set | content | line |
|---|---|---|
| P | current ego row (11 values, accelerations zeroed) + current 32 block means (32 × 256 = 8,192) | :186, :208 |
| PH | P + ego history over the past 4 s: for each of t−0.5 … t−4 s the pose-derived speed and yaw rate, the logged speed, and an "exists" flag (32 values). Lags before the chain start are zero with flag 0. | :52-64 |
| PH+W | PH + wiped memory after the update at t (16 × 256) | :190 |
| PH+C | PH + carried memory after the update at t (16 × 256) | :190 |
| PH+Cs | PH + the scorer's summary of the memory it reads at t in carry mode (256; what the rating head reads) | :190 |
| PH+Cshuf | PH + carried memory of a moment of another clip (negative control) | :67-76, :191-192 |

## Probe (one fixed setup, `src/s10_probe.py:120-165`)

- Ego row, ego history and the intercept enter unpenalised (so the ego-history baseline is as strong as a linear fit allows).
- Block means and memory are standardised with the training fold's mean and sd, each block divided by √(its size), one ridge strength.
- T1–T3, T5, T6: ridge (exact kernel form). T4: L2-penalised logistic regression on the unpenalised block + the top 600 principal
  directions of the same kernel.
- Outer 5 folds by clip (seed 20261111). Ridge strength chosen inside each training fold by 5-fold cross-validation over clips,
  per target: grid 10^−3 … 10^3 in half-decades (ridge, exact leave-clips-out formula); 10^−2 … 10^3 in decades (logistic, refits).
- Error: mean squared error (T1–T3, T5, T6, recall control); log-loss (T4). AUC for T4 is reported too (information).
- **Gain of A over B = 1 − error(A) / error(B)** on the out-of-fold predictions (`:240-248`).
- 95 % interval: percentiles of a clip-clustered bootstrap, 2,000 draws, seed 20261112. Six-target (Bonferroni) intervals are reported
  as information.
- All moments and event moments reported separately.
- Self-test on synthetic data (`out/link2_selftest.json`): kernel solution equals the plain ridge to 1.8e-13; the leave-clips-out
  formula equals refits to 4.6e-14; a planted memory signal is found (gain 0.94 [0.92, 0.96]) and vanishes when shuffled
  (−0.001 [−0.022, 0.016]).

## Controls (both must pass, or the result is not trusted)

- **Positive.** Same probe, target = the standardised 32 block means of the moment 2 s earlier (the recall lesson's 2 s target;
  `src/s10_probe.py:177-181`). **Passes if** PH+C beats PH+W by ≥ 3 % with the 95 % interval above 0, primary sample, all moments.
  The figure is reported beside the read-out's 6.0 % (`reports/BUILD5_RECALL_LESSON.md`). The two are not the same measurement: the
  probe is linear and also sees the current block means and the ego history.
- **Negative.** Carried memories shuffled across clips (seed 20261113), each moment keeps its own P and H. **Passes if**, on every
  target T1–T6, the gain of PH+Cshuf over PH+W is below 1 % or its 95 % interval includes 0, primary sample, all moments.

## Rule (primary sample, all moments, build 5 checkpoint)

- **LINK 2 HOLDS** if PH+C beats PH+W by ≥ 2 % relative error on at least one of T1–T6, with the 95 % interval excluding 0, and both
  controls pass.
- **LINK 2 BROKEN** if no target shows a gain with the interval excluding 0, and both controls pass.
- **UNCLEAR** otherwise, including if a control fails.

Information, not rule: event moments, the picking clips, PH vs P, PH+W vs PH, PH+C vs PH, PH+Cs vs PH+W, the build 4 and pilot
checkpoints, AUC, Bonferroni intervals.

## Known limits, stated before the result

1. "Not used in the first 930 steps" rests on the deterministic plan function and its seed; the runs did not log clip names per step.
2. The starting generator and scorer were trained on the whole training slice. Only the memory blocks have not seen these 300 clips.
3. Only 27 primary clips hold event moments (event windows were drawn twice in training, so event clips were more often used).
   Event-moment intervals will be wide.
4. Moments 4–7 of a chain have a partial ego history (zeros + flag). The memory could not have seen those lags either.
5. One linear probe. Information that is only readable non-linearly would be missed; a BROKEN result means "not linearly readable
   beyond ego history and the current view".
6. Six targets are tested; the rule uses unadjusted 95 % intervals as written in the brief. Bonferroni intervals are given beside them.
7. The T4 logistic probe uses 600 directions, and its inner folds reuse the outer training fold's directions.
8. If T4 has fewer than 2 % or more than 98 % positives, its result is reported and flagged as weak.
