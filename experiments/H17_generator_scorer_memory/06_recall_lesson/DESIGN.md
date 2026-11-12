# Build 5 design note: a lesson of the memory's own (recall of earlier image summaries)

Sid's brief of 2026-11-11. Read-only: no code changed in `src/`, no training. One forward-only measurement
(`out/build5_measure.py`, launched by `out/build5_measure_launch.sh`; `out/build5_measure.json`, `logs/build5_measure.log`,
`logs/build5_chain.log`): 20 picking clips (the build-4 fixed subset, 640 moments), cores 0–1, nice 19, 09:51:35–10:04:46 IST (13 min),
after the machine gate (5-min load 29.6, available RAM 65.9 GB). No held-out slice opened. This note proposes; Sid decides.

## Recommended design in one paragraph

After the update at moment t, a read-out of 403,968 parameters asks the memory 32 questions per target (one per block position: learned
position + learned "how long ago" embedding + a projection of the CURRENT block mean, which is a cached constant), with one attention layer
into the 16 slots, and predicts the 32 block means (4 × 8, 256-d each) of the moment 2 s ago (t − 4) and of the moment 4 s ago (t − 8),
standardised per position and dimension with training-slice statistics. Loss: smooth-L1, output layer zero-initialised so the lesson starts
at the "predict the average" value (0.41). It is applied at every scored moment whose target lies inside the window (12 targets at 2 s + 8 at
4 s per window, 40 per step), in both arms, with a fixed weight of 2.0 (lighter alternative 0.5). Its gradient reaches the read-out, the
update block (gate included) and M0 only; no new detach is needed because nothing upstream of the memory depends on the generator or the
scorer (the pick fed to the update is already detached, `src/s6_model.py:271`). The next 930-step check replaces condition (c) by a recall
gap (carry vs the same weights reset before every moment), an absolute change per moment with a floor on the memory norm, and a gate floor;
each threshold is one the pilot's collapsed memory fails.

## What the measurement found (context for every choice below)

All values from `out/build5_measure.json`; targets standardised per (position, dimension) over the 640 moments ("z"); "2 s" = t − 4,
"4 s" = t − 8, "1 s" = t − 2 (measured for the argument only). Samples: 600 / 560 / 480 moments with 1 / 2 / 4 s of history in their chain.

**The cache holds the targets exactly.** `pooled[:, 1:]` of a clip file equals `s6_model.block_means(tokens)` to 9.5e-7 and `pooled[:, 0]`
the global mean to 3.1e-6 (`pooled_check`, clip `0238dfe4`): the 32 block means are the update block's own image input
(`src/s6_model.py:281`), so the target is "what the memory was shown".

**How predictable is the past from the present** (mean squared error in z; 1.0 ≈ predicting the average):

| target | how far back | predict the average ("zero") | copy the current summary | best linear guess from the current summary (R²) |
|---|---|---|---|---|
| 32 block means | 1 s | 0.988 | 0.439 | 0.41 |
| 32 block means | 2 s | 0.979 | 0.755 | 0.25 |
| 32 block means | 4 s | 0.964 | **1.114** | 0.12 |
| coarse 2 × 4 (8 vectors) | 2 s / 4 s | 0.976 / 0.960 | 0.669 / 1.058 | 0.31 / 0.14 |
| global mean (1 vector) | 2 s / 4 s | 0.971 / 0.950 | 0.605 / 0.997 | 0.37 / 0.17 |

(`lesson.lag_*`, `probe.lag_*.*.now_only`; the linear guess is a leave-clips-out ridge fit, 5 folds of 4 clips.) At 1 s the present
explains 41 % of the past, at 2 s 25 %, at 4 s 12 %; at 4 s copying the present is WORSE than predicting the average.

**What a memory adds on top of the present** (same linear probe on [current summary; memory], carried vs reset before every moment; gap =
reduction of the error in z; 32 block means):

| memory | 2 s: carried / reset / gap | 4 s: carried / reset / gap |
|---|---|---|
| version 2 untrained (gates 0.9 … 0.1) | 0.725 / 0.782 / **+0.057 (7.3 %)** | 0.838 / 0.895 / **+0.057 (6.4 %)** |
| pilot carry, step 930 (collapsed) | 0.785 / 0.782 / −0.003 (−0.4 %) | 0.864 / 0.904 / +0.040 (4.4 %) |
| build 4 carry, step 930 | 0.775 / 0.778 / +0.003 (0.4 %) | 0.877 / 0.898 / +0.022 (2.4 %) |

(`probe.lag_4|lag_8.block_means_32.*_now_plus_memory`.) Three readings. (1) A reset memory adds nothing to the present (0.782 vs 0.776 from
the present alone at 2 s): one update step does not leak a usable copy of the past. (2) The untrained version-2 memory already recalls
(its spread of gates is a bank of running averages); **930 steps of 0.1.8b's lessons removed almost all of it** (7.3 % → 0.4 % at 2 s):
this is Sid's diagnosis measured. (3) The pilot's collapsed memory keeps a 4.4 % gap at 4 s: with a gate of 0.016 it is one very slow
running average, which holds old content; so a recall rule at 4 s alone would not separate it, the 2 s gap does.

## PART 1: the lesson

**Targets: the 32 block means of t − 4 (2 s ago) and t − 8 (4 s ago).**
- Which moments. The two-frame input shows t and t − 0.5 s. At 1 s the present still explains 41 % and copying it costs 0.44; a 1 s target
  would mostly teach "repeat the present". At 2 s and 4 s the present explains 25 % and 12 %, so most of the target is reachable only
  through the memory. 4 s is the furthest target that fits a 16-moment window with a useful count (8 moments per window have it); 6 s would
  leave 4.
- Which part of the pooled copy. The global mean is the most predictable from the present (R² 0.37 at 2 s) and discards where things were;
  what driving needs from the past is layout (a lead that braked, a crossing actor now occluded), which lives in the block positions. The 32
  block means are also exactly what the update block is given, so the lesson asks the memory to keep what it saw. Cost: 32 × 256 values per
  target; a token's variance sits in few directions (90 of 256 dimensions hold 90 %, 12 hold 50 %: `token_pca`), so one target is ≈ 2,900
  effective values and two are ≈ 5,800 against a memory of 4,096: **perfect recall is not possible by capacity**, the lesson asks for the
  best compression. The coarse 2 × 4 pooling (8 vectors, ≈ 1,400 effective values for both targets) fits comfortably and is the alternative
  (OPEN 1).
- Standardised targets (z), statistics from the training slice's pooled array (`A3/cache/training/pooled_training.npy`, one CPU pass).
  Raw block means carry 49.5 % of their power in a constant mean (`raw_stats.share_of_raw_power_in_the_mean`); unstandardised, a bias would
  earn half the loss without any memory.

**Read-out: 403,968 parameters, reads M through learned questions.** M is 16 unordered slots with no spatial layout, so a direct linear map
from M would have to be 4,096 → 8,192 per target (67 M values). Instead, as the scorer's summary does with one question
(`src/s6_model.py` `ScorerRead`), the read-out forms 32 questions per target: position embedding (32 × 256 = 8,192) + lag embedding
(2 × 256 = 512) + Linear(256 → 256) of the current block mean at that position (65,792); one attention layer into M (`s6_model.Attn`,
263,168); LayerNorm (512); Linear(256 → 256) output, zero-initialised (65,792). Built and timed in the measurement script (`ReadOut`,
`readout_params`: 403,968; +24 % on the 1,669,648 new parameters of version 2).

**Distance: smooth-L1 (β = 1) on the standardised targets.** Cosine ignores magnitude and is already small between raw neighbours (copying the
present has raw cosine distance 0.17 at 2 s) so it would reward direction only; L2 on tokens with heavy tails lets a few dimensions set the
gradient; smooth-L1 is quadratic near the target and linear in the tails, the same family as the generator's L1 lesson. Standardised so every
dimension counts equally. Measured values in part 2.

**Which moments.** Every scored moment (window moments 4–15) whose target is inside the same window: 2 s targets at moments 4–15 (12), 4 s
targets at moments 8–15 (8): **20 targets per window, 40 per step**, each 32 × 256 values. Warm-up moments 0–3 have no 2 s target inside
the window (their past is before M0), so they get none; the memory is still trained through them because the chain from M0 is attached
(`src/s6_train.py` `window_step`: `M = model.init_memory(...)`, `M = model.step_memory(o["M_new"])`).

**Routing: no new detach point is needed; one existing detach carries it.** The lesson reads `o["M_new"]` (`src/s6_model.py:281-282`).
M_new = update(M_old, block_means(tokens), motion, picked): tokens are cached constants, motion is logged data, `picked` is detached
(`src/s6_model.py:271`), and M_old is the previous M_new or M0. So M_new is a function of the update block's parameters (gate weight and
biases included), M0, and constants; the generator, the scorer, the generator read and the scorer summary only READ M and never write into
it, so no gradient path from the lesson to them exists. The read-out's own inputs are M_new and the current block means (constants). The
build-5 routing test adds one row and one column: lesson "recall" → update block, start state M0, read-out: yes; every other group: no;
and the read-out column is "no" for the five existing lessons. To be stated in code as an assertion, not relied on silently: the lesson is
computed on `o["M_new"]`, never on a tensor that passed through `gen_read` or `sc_read`.

**The failure mode "predict 2 s ago from now".** Three guards. (1) The read-out is GIVEN the present (the projection of the current block
mean in each question), so whatever is predictable from now is learned in the read-out itself and earns the memory nothing; the memory is
paid only for what the present does not contain. (2) The health rule scores recall as carried minus reset with the same weights: the reset
memory contains exactly the one-step leak of the present, so the leak cancels. (3) Measured: a reset memory adds nothing to the present
(0.782 vs 0.776 at 2 s, 0.895 vs 0.897 at 4 s), so the leak is not there to exploit even in principle with a linear reader. No skip
connection from the present to the output: at 4 s copying the present (1.114) is worse than predicting the average (0.964).

## PART 2: the weight

**Measured size of the lesson at step 0** (smooth-L1 on standardised 32 block means; version-2 model untrained, carry;
`lesson.lag_4|lag_8.block_means_32`):

| prediction | 2 s ago | 4 s ago |
|---|---|---|
| (i) zero = predict the average | **0.409** | **0.405** |
| (ii) memory through an untrained read-out, default-initialised output | 0.516 | 0.514 |
| (iii) copy the current summary (no-memory baseline) | 0.308 | 0.424 |
| perfect recall | 0 | 0 |

In squared error (z): zero 0.979 / 0.964; untrained read-out 1.300 / 1.292; copy 0.755 / 1.114; best linear guess from the present 0.776 /
0.897; present + untrained carried memory 0.725 / 0.838. With the output layer zero-initialised (recommended) the lesson starts at (i),
**0.407** averaged over the 12 + 8 targets, not at (ii). The realistic floor is far above 0: the present alone removes about 20 % (2 s) and
7 % (4 s) of the squared error, and the untrained memory a further 7 % and 6 %. So the part of the lesson only the memory can earn is small
in absolute terms, a few hundredths of the 0.41.

**Recommended weight 2.0, fixed.** It adds 0.81 at step 0 to 0.1.8b's ≈ 2.3 (`runs/build4_short/run.json`: first 100 steps 2.369), i.e.
26 % of the new total 3.1, the second largest term after the human-closeness lesson (1.64) and above the generator lesson (0.59). The
weight does not touch the generator, the scorer or their reads (routing above); it only sets how the lesson's gradient on the update block
and M0 compares with the three driving lessons that reach them (generator, grade, human; `out/s6_t2_map_v2.json`), which are the ones that
closed the gates in build 4 (training-window gate mean 0.45 → 0.25). The memory-only part of the lesson being a few hundredths, a weight
below 1 would leave it smaller than the driving lessons' pull on the same parameters.
**Lighter alternative 0.5:** adds 0.20 (8 % of the total, about the size of the grade lesson 0.10–0.15); least disturbance to the driving
balance, but the gate would have to stay open for a reward of ≈ 0.01–0.02.
The ratio of gradient norms on the update block (recall vs driving) was not measured here (it needs a backward pass, outside this note's one
measurement); it is the first number build 5's routing test should print, and the weight should be re-read against it (OPEN 3).

**Fixed, not decaying**, for the 930-step check and the first full run: the driving lessons push the gates shut continuously, not only at
the start (build 4: slot 1's gate 0.90 → 0.79 → 0.76 → 0.55 at steps 0 / 310 / 620 / 930), so a decaying counter-weight would hand the gate
back to them. Decay becomes a question only if the pick grade suffers at the chosen checkpoint.

## PART 3: the health rule for the next 930-step check

Values on the same 20 clips with the same code (`health.*.carry`; in_picking moments after the warm-up, 393 moments); 84-clip values where
they exist (`out/build4_before.json`, `out/build4_short.json`). ‖M0‖ = 64 for version 2 (16 rows of norm 16); the pilot's version 1 has
no M0, so 64 (16 candidate rows of norm 16) is the reference for it too.

| measure | proposed threshold | pilot carry step 930 (collapsed) | build 4 carry step 930 | version 2 untrained | rule or information |
|---|---|---|---|---|---|
| slots alike: mean pairwise cosine of the memory read | ≤ 0.90 | **1.000** (0.99998 on 84 clips) fails | 0.454 (0.442 on 84) | 0.613 | rule |
| recall at 2 s: lesson loss, carried lower than reset (same weights) | ≥ 5 % lower | **−0.4 %** fails | 0.4 % fails | 7.3 % | rule |
| recall at 4 s: same | ≥ 6 % lower | **4.4 %** fails | 2.4 % fails | 6.4 % | rule |
| absolute change per moment ‖M_t − M_(t−1)‖ / ‖M0‖ | ≥ 0.025 | **0.012** fails | 0.037 | 0.068 | rule |
| memory norm ‖M‖ / ‖M0‖ (floor for the line above) | ≥ 0.5 | **0.30** fails | 0.93 | 0.99 | rule |
| gate at the check: mean over the 16 slots | ≥ 0.10 | **0.016** fails | 0.164 | 0.497 | rule |
| gate spread: slots with mean ≥ 0.05; max / min | ≥ 8; ≥ 3 | **0; 1.0** fails | 13; 26.7 | 16; 9.2 | rule |
| NaN / inf | none | none | none | none | rule |
| pick grade on the 1,131 graded moments | ≥ 0.74 (sanity) | 0.7526 | 0.7496 | 0.7685 | rule |
| gate over the training windows, last 100 steps; and last / first 100 | — | not recorded in the pilot's `run.json` | 0.247; 0.55 | — | information |
| relative change per moment (the old condition (c)) | — | 0.054 (0.057 on 84) | 0.040 (0.042 on 84) | 0.069 | information (it passes the collapsed memory) |
| between-clip cosine of the memory, after removing the mean over clips | — | −0.03 | −0.02 | −0.05 | information |
| carried vs reset pick grade | — | — | 0.7496 vs 0.7520 | — | information |

Every rule threshold is failed by the collapsed pilot memory. Build 4's step-930 memory passes the shape rules (slots, change, norm, gate)
and fails both recall rules, which is the point of build 5.

Caveats to the recall rows. For the existing checkpoints there is no trained read-out, so their values come from the linear probe above
(squared error in z, ridge, leave-clips-out, 20 clips); in build 5 the rule is evaluated with the trained read-out's own loss (smooth-L1)
on all 84 picking clips, carried vs reset. The relative thresholds (5 %, 6 %) are meant to carry over, but the probe and the trained
read-out are different readers, and the pilot's 4.4 % at 4 s is close to the 6 % line; the 2 s row is the one with a wide margin (−0.4 %
vs 5 %). Recommended: both rows are rule, and the check also runs the linear probe so the two readers can be compared (OPEN 4).

## PART 4: cost and risks

**Extra time per step: about 0.01–0.02 s.** The targets cost nothing to fetch: they are `block_means(tokens)` of moments already in the
window (computed for the update anyway, `src/s6_model.py:281`). The read-out's forward + backward for one step's 40 predictions took
0.0031 s on the shared GPU (`readout_cost`); the lesson's backward runs through the update chain that the driving lessons already
back-propagate through (in the wiped arm it adds the update block's backward, 16 small attention layers). Against build 4's 4.59 s per step
(`runs/build4_short/run.json`) that is under 0.5 %; against build 3's uncontended 1.65 s about 1 %. One-off: the standardisation statistics
(one CPU pass over the training pooled array). The check gains one extra replay in reset mode (already done at step 930 in build 4, ≈ 13
min with its grading; without grading ≈ 3 min).

**Risks.**
1. **The memory stores a compressed past that the generator and the scorer still ignore** (biggest). The lesson rewards holding the past,
   not using it; build 4 already shows the carried state worth −0.0024 pick grade. Measure: carried vs reset pick grade at every check
   (information at 930 steps, a rule candidate for the full run), and the read-projection norms. Mitigation if it persists: nothing in
   this note; it would mean the driving lessons have no use for a 2–4 s image past, which is a finding about the approach, not a bug.
2. **The lesson competes with the driving lessons** on the update block and M0. Measures: the three choice gates and the pick grade at
   step 930 against build 4's (0.7496; hard-brake picks 0; go 83.0 %; ADE (3 s) 0.246 m), and the gradient-norm ratio in the routing test.
   Mitigation: the lighter weight 0.5.
3. **Between-clip similarity stays high.** Measured: the raw block means are alike across clips (cosine 0.45) only through their constant
   mean; standardised they are not (−0.02), and the memories' similarity (0.74 untrained, 0.90 build 4, 0.95 pilot) also vanishes once the
   mean over clips is removed (−0.05, −0.02, −0.03) (`between_clip_cosine`). So flag 4 of build 4 is a shared offset, not missing content;
   standardised targets do not reward the offset. Measure: report the mean-removed cosine; no rule.
4. **The wiped arm stops being a pure control of the update block.** In wiped mode the update block gets no gradient today (T3); with the
   lesson it trains on one-step-from-M0 recall, which is the intended baseline, but the test "update block receives no gradient" must be
   rewritten (it becomes: only the recall lesson reaches it).
5. **Recall of 2 s and 4 s may be learned as two fixed running averages** (the pilot's single slow average already scores 4.4 % at 4 s).
   That is a legitimate memory, but a weak one. Measure: the 2 s rule, and the per-slot gate spread.

**What in the current code makes the plan hard.**
- `forward_moment` returns `M_new` but not the block means; the lesson needs them per moment (recompute from `batch["tokens"]`, or return
  them). Small.
- `window_step` sums five lessons into one loss and the record / the routing test iterate over `LESSONS`; a sixth lesson touches
  `s6_train.py`, `s6_tests.py` (T2 table, T3 wording, T4 record) and `s7_pilot.py` / `s8_short.py` (the check must compute the lesson in
  carry and in reset mode: the replay functions run under `no_grad` and do not keep the chain's past block means; they need a per-clip
  buffer).
- The read-out is new state: a version 3 with its own entry in `trainable_state` / `load_trainable` and `new_param_count`, so that version
  1 and 2 checkpoints still load.
- The standardisation statistics must be saved with the checkpoint (or the read-out is meaningless at load).
- Build 4's OPEN 2 is still open: Adam at 1e-4 cannot move the gate biases or M0 measurably in 930 steps (bias sigmoid 0.900 → 0.899), so
  under the new lesson the gate would again move through its weight only. This note does not change the learning rates (OPEN 5).
- Step-0 identity (T1) is unaffected: the read-out does not feed the generator or the scorer.

## OPEN for Sid

1. **Target detail.** Recommend: the 32 block means at 2 s and 4 s (what the update block sees; keeps layout). Alternative: the coarse
   2 × 4 pooling (8 vectors), whose two targets fit the memory's capacity (≈ 1,400 effective values vs 4,096) and which the present
   predicts a little better (R² 0.31 / 0.14); cheaper to satisfy, less layout.
2. **Targets in time.** Recommend: 2 s and 4 s (20 targets per window). Alternative: add 1 s (16 more targets per window; mostly
   predictable from the present, R² 0.41) or replace 4 s by 3 s (10 moments per window instead of 8).
3. **Weight.** Recommend: 2.0 fixed, re-read against the gradient-norm ratio on the update block that build 5's routing test prints before
   the 930-step check. Alternative: 0.5 fixed.
4. **Recall rule.** Recommend: carried lower than reset by ≥ 5 % at 2 s AND ≥ 6 % at 4 s, on the trained read-out's loss, 84 clips; linear
   probe reported beside it. Alternative: the 2 s row only as rule (the row with a wide margin over the collapsed memory), 4 s as information.
5. **Learning rate of the gate and M0 (build 4's OPEN 2).** Recommend: decide it together with this lesson, and change one thing at a time:
   first the lesson at the current rates, then, if the gate still closes through its weight, 1e-3 for gate and M0. Alternative: both at once
   (faster, but a pass or a fail cannot be attributed).
6. **Where the lesson's read-out lives at inference.** Recommend: training-only, dropped from the shipped model (it feeds nothing). Alternative:
   keep it as a diagnostic output in the closed-loop adapter.
7. **Both arms.** As decided, both get the lesson; note that the wiped arm's update block then trains (risk 4). Recommend: accept and rewrite T3.
   Alternative: in the wiped arm, compute the lesson without back-propagating into the update block (keeps T3 as is, but the arms then
   differ in more than the reset).

## Files

- `out/build5_measure.py` (the measurement, read-out definition included), `out/build5_measure_launch.sh`, `out/build5_measure.json`
- `logs/build5_measure.log`, `logs/build5_chain.log`
- Read: `reports/BUILD4_MEMORY_FIX.md`, `reports/PILOT_SEED0.md`, `reports/BUILD2_CACHE.md`, `src/s6_model.py`, `src/s6_train.py`, `src/s8_short.py`
