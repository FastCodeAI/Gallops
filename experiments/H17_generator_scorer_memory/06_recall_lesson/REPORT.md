# Build 5: the recall lesson (the memory's own lesson)

2026-11-11. Verdict of the 930-step short check by the pre-registered rule: **FAIL, on one condition only — (h) pick grade 0.7313 < 0.74.** The seven memory conditions all hold, including both recall conditions. Pre-registration: `reports/BUILD5_PREREG.md` (written 20:46 IST, run started 20:48 IST, `logs/build5_chain.log`).

Not done, as instructed: no full pilot, no held-out slice opened, no closed loop, no docker, no pushes.

## 1. The change

Model version 3 = version 2 (learned start state, spread gates) + a recall lesson. Versions 1 and 2 checkpoints still load (both were loaded for the "before" columns, `out/build5_before.json`).

- **Targets.** The 32 block means (4×8 blocks, 256 values each) of the encoder output 2 s ago (4 moments) and 4 s ago (8 moments), standardised per (position, value) with statistics of the training slice (126,981 moments; `out/recall_target_stats.pt`, `logs/build5_stats.log`), saved with the checkpoint.
- **Read-out** (`src/s6_model.py`, `ReadOut`). 32 questions per target (position embedding + lag embedding + a linear map of the *current* block mean), one attention layer into the memory, LayerNorm, a zero-initialised linear output. Training-only; dropped from the exported model; the checkpoint records this.
- **Lesson** (`src/s6_train.py`, `recall_lesson`). Smooth-L1 (β = 1) on the standardised targets, every scored moment whose target is inside the window (40 per step), weight 2.0 fixed. Computed on the memory just written; an assertion in code checks that its gradient reaches only the read-out, the update block (gates included) and the start state.
- Both arms get the lesson. Learning rates unchanged.

**New parameters** (`out/s6_params_v3.json`):

| block | parameters |
|---|---|
| generator read | 263,168 |
| scorer read summary | 263,424 |
| rating head, new columns | 262,144 |
| update block (incl. 16 gate biases) | 876,816 |
| start state | 4,096 |
| read-out (training-only) | 403,968 |
| **total** | **2,073,616** |
| exported (without the read-out) | 1,669,648 |

## 2. Tests (version 3; `logs/build5_tests.log`, 11:09–11:20 IST)

| test | result | evidence |
|---|---|---|
| T1 step-0 identity, carry and wiped, 2,591 moments | **pass**, max difference 0.0 in both | `out/s6_t1_v3.json` |
| T2 routing table on a map batch (24 map rows) | **pass**: recall reaches update block, start state, read-out only; no other lesson reaches the read-out; 89 banned tensors checked | `out/s6_t2_map_v3.json` |
| T2 gradient-norm ratio on the update block, recall (weight 2.0) vs generator + grade + human | **0.015** after 5 steps (0.247 vs 16.90); gate 0.041 | same file |
| T3 wiped: memory read = start state exactly | **pass**, deviation 0.0 over 6 steps | `out/s6_t2t3_v3.json` |
| T3 wiped: update-block gradient comes from recall only | **pass**: without recall 27/27 update tensors receive none; recall alone 27/27 non-zero; start state reached; no other group | same file |
| T3 carry: memory differs from the start state | **pass**: ‖M − M0‖ 20.7–34.2 | same file |
| T4 smoke, 50 steps per mode | **pass**, no NaN. Recall (unweighted) step 1 0.335 (2 s 0.331, 4 s 0.342), step 50 0.264 in both modes. Carry gates step 1 0.898…0.097, step 50 0.896…0.089. Ratio after 50 steps: 0.059 (gate 0.007) | `out/s6_t4_v3.json` |
| T5 timing | 2.04 s per step in carry; recall forward 0.0054 s = 0.26 % of the step; recall backward alone 0.022 s | same file |

Weight 2.0 was re-read against these ratios and left unchanged, as instructed. See flag 1 for how the ratio moved during the run.

## 3. The short check

Carry, version 3, seed 0, first 930 steps of the same pass plan, starting model 0.1.8b pass 0.6 (sha256 verified in every job, 495/495 tensors). Files: `{experiment_store}/runs/build5_short/` (`run.json`, `log.txt`, `ckpt/step0310|0620|0930.pt`, `dumps/`), summary `out/build5_short.json`. Mean step 3.77 s; machine shared throughout.

### 3.1 Rule at step 930 (84 picking clips, 1,705 moments after warm-up) with the "before" columns

| | measure | pilot step 930 | build 4 step 930 | **build 5 step 930** | threshold | build 5 |
|---|---|---|---|---|---|---|
| (a) | slots alike (cosine) | 1.000 | 0.442 | **0.355** | ≤ 0.90 | holds |
| (b) | recall 2 s, carried vs reset | 4.9 % *(probe)* | 7.2 % *(probe)* | **6.0 %** (0.2995 vs 0.3188) | ≥ 5 % | holds |
| (c) | recall 4 s, carried vs reset | 11.1 % *(probe)* | 15.4 % *(probe)* | **10.5 %** (0.3217 vs 0.3595) | ≥ 6 % | holds |
| (d) | absolute change per moment / ‖M0‖ | 0.012 | 0.039 | **0.050** | ≥ 0.025 | holds |
| (e) | memory norm / ‖M0‖ | 0.29 | 0.93 | **0.965** | ≥ 0.5 | holds |
| (f) | gate mean (min / median / max) | 0.017 (0.017 / 0.017 / 0.017) | 0.167 (0.022 / 0.126 / 0.553) | **0.216 (0.030 / 0.202 / 0.592)** | ≥ 0.10 | holds |
| | slots ≥ 0.05; max / min | 0; 1.0 | 13; 25.7 | **12; 20.0** | ≥ 8; ≥ 3 | holds |
| (g) | non-finite moments | 0 | 0 | **0** | 0 | holds |
| (h) | pick grade, 1,131 graded moments | 0.7526 | 0.7496 | **0.7313** | ≥ 0.74 | **fails** |
| | like-for-like: linear-probe recall gap 2 s / 4 s | 4.9 % / 11.1 % | 7.2 % / 15.4 % | **13.7 % / 24.6 %** | information | |

"Before" columns: `out/build5_before.json`, same code. Those checkpoints have no trained read-out, so their (b), (c) are the linear probe; build 5's are its own trained read-out. The last row is the only like-for-like recall comparison: by the probe, what can be recalled from the carried memory roughly doubled against build 4.

### 3.2 Along the run (steps 0, 310, 620 on the first 20 picking clips; 930 on all 84)

| step | slots alike | recall 2 s / 4 s (read-out) | probe 2 s / 4 s | change / ‖M0‖ | norm / ‖M0‖ | gate mean (slots ≥ 0.05) | recall ÷ driving gradient, update block (gate) | generator read projection norm | rating head new columns norm |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.613 | 0 % / 0 % | 6.7 % / 6.7 % | 0.068 | 0.985 | 0.497 (16) | not measured | 0 | 0 |
| 310 | 0.503 | 1.3 % / 3.0 % | 6.9 % / 10.4 % | 0.052 | 0.980 | 0.317 (16) | 0.68 (0.35) | 0.210 | 0.763 |
| 620 | 0.403 | 2.8 % / 5.6 % | 4.9 % / 9.0 % | 0.044 | 0.969 | 0.209 (14) | 1.21 (0.37) | 0.245 | 0.925 |
| 930 | 0.355 | 6.0 % / 10.5 % | 13.7 % / 24.6 % | 0.050 | 0.965 | 0.216 (12) | 1.26 (0.98) | 0.264 | 1.050 |

Source: `run.json` → `probes`. The 930 row is on 84 clips, the others on 20, so the last step is not a clean trend point.

### 3.3 Information measures (not part of the rule)

- **Pick grade, same weights:** carried 0.7313, reset to the start state every moment 0.7355 (build 4: 0.7496 / 0.7520; pilot step 930: carry 0.7526, its wiped arm 0.7563).
- **Choice gates** against build 4: hard picks above 10 m/s 0 (0); go rate at pull-away 75.5 % (83.0 %); closest-of-32 ADE (3 s) 0.265 m (0.246 m). The ADE gate (≤ 0.258 m) **fails**; in reset mode it is 0.264 m, also failing.
- **Gate mean over training windows:** first 100 steps 0.475, last 100 steps 0.242.
- **Between-clip cosine of the memory, mean over clips removed:** −0.006 carried (build 4: 0.044; pilot: 0.007). Raw between-clip cosine 0.755 (build 4 0.848, pilot 0.950).
- **Recall lesson's training value** (unweighted, mean per 100 steps): 0.389, 0.354, 0.324, 0.315, 0.306, 0.301, 0.293, 0.283, 0.288, 0.276; as a share of the all-zero-answer reference: 0.93 → 0.67.
- **Event moments** (34 graded): pick grade 0.953 carried, 0.949 reset.

### 3.4 Resume test

Stopped on purpose after the step-310 probe; steps 311–320 run unsaved first. Resumed job restored weights, optimisers and random-number states. Loss differences over the ten steps: first 0.00000, maximum 0.019 (criterion 0.005 / 0.05). **Matched.** (`out/build5_short.json` → `resume_test`.)

## 4. What the result says, and what it does not

- **Verified:** the memory now passes every memory-health condition, and the recall lesson does what it was built for: the carried memory recalls the scene 2 s and 4 s back better than a memory reset each moment, by the lesson's own measure and (more strongly) by the independent probe.
- **Verified:** the pick grade at step 930 is 0.7313, below the 0.74 line and below build 4 and the pilot at the same step.
- **Not established:** that the recall lesson *caused* the pick-grade drop. Resetting the memory every moment gives 0.7355 with the same weights, so the drop is not the carried memory misleading the scorer at read time; it sits in the weights. The pilot's own pick grade moved between 0.741 and 0.775 across its checks with no change of method (`runs/pilot_seed0/carry/run.json`), so one checkpoint at 0.731 is lower than anything seen but only ~0.01 outside that band. There is no build-5 run without the recall lesson to compare against.
- **Argued, not verified:** by step 620 the recall gradient on the update block exceeded the three driving lessons' combined (flag 1), which would be a mechanism for the memory being shaped for recall at the expense of driving. The closest-of-32 ADE (3 s) and go-rate moving the wrong way fit that, but the generator and scorer weights also differ run to run.

## 5. Flags

1. **The gradient balance reversed during the run.** At weight 2.0 the recall gradient on the update block was 1.5 % of the driving lessons' after 5 steps and 5.9 % after 50 (tests), but 68 % at step 310, 121 % at 620 and 126 % at 930 (gate: 98 % at 930). The driving gradient on the update block shrank (16.9 → 0.29) while recall's grew. The run-time figures are one batch (batch 130) in evaluation mode; the test figures are in training mode.
2. **Build 4 satisfies all eight conditions of this rule when recall is read with the probe.** The rule's recall conditions on build 5 use a different instrument (the trained read-out), stated in the pre-registration before launch.
3. **Probe disagreement with the design note, unresolved.** `reports/BUILD5_DESIGN_NOTE.md` gave build 4 a probe gap of 0.4 % / 2.4 % (20 clips, plain ridge); here it is 7.2 % / 15.4 % (84 clips, kernel ridge, regularisation chosen per fit). On 20 clips the build-5 probe gap also moved non-monotonically (6.9 → 4.9 → 13.7 % on 84), so the probe is noisy at small clip counts.
4. **Choice gates:** the closest-of-32 ADE (3 s) gate fails (0.265 m vs 0.258 m) and the go rate fell to 75.5 %. These were information-only in this pre-registration.
5. **Carried memory does not help the pick:** carried is 0.004 below reset, as in build 4 (0.002 below) and the pilot at step 930.
6. **Parameter count** 2,073,616 in training, 1,669,648 exported — above the 1.5 M target noted earlier.
7. **Recall at step 1 was 0.335, not the ≈ 0.41 expected.** 0.41 is the all-zero-answer value on picking clips; step 1 of training is one batch from the training slice. An all-zero reference per step was added to the log after the tests because batch-to-batch movement (0.37–0.43) is larger than early learning.
8. **In wiped mode the gates open under the recall lesson alone** (T4: step 50 gates 0.901…0.119 vs carry 0.896…0.089); at step 930 the reset replay's gate mean is 0.40 against 0.22 carried.
9. **Three launches of the "before" job, ~9 h lost.** (i) Refused by the job's own gate after an 18-minute model load on a busy machine; the gate check now runs first and the chains retry. (ii) Crashed on a bug of mine: an earlier edit to the replay code dropped its no-gradient marker; fixed in `src/s8_short.py`. It sat unnoticed for about five hours. (iii) Ran. The tests were not affected (they do not use the replay).
10. **Target statistics array was stored flat** (126,981 × 8,448); reshaped and checked against a clip file (max difference 0.0).
11. **Pre-registration edited once after launch**: a wrong input count in an explanatory note (corrected to 12,288), marked in the file; no rule content changed.
12. **Gradient ratio at step 0 not measured** (the read-out's output is zero-initialised, so recall has no gradient on the update block at step 0).
13. **Resume now restores random-number states**, which the pilot's resume did not; the first resumed step is bit-identical, later steps differ by up to 0.019 in loss.

## 6. OPEN — for Sid

1. **What to do about the failed pick-grade condition.**
   *Recommend:* one control run of the same version-3 code with the recall weight at 0 (930 steps, same seed, same check). It separates "the recall lesson costs pick grade" from run-to-run movement, which the current evidence cannot do. ~1.5 h.
   *Alternative:* accept the mechanism in flag 1 without the control and go straight to a lower weight (item 2).
2. **The recall weight.** It was fixed at 2.0 on a ratio measured in the first 50 steps; by step 620 recall dominates the update block.
   *Recommend:* if the control shows a cost, rerun at 0.5 (ratio at 930 would be ~0.3 by scaling).
   *Alternative:* keep 2.0 but stop the recall gradient at the gates only, leaving the write content trained by it.
3. **Whether (h) 0.74 is the right line for a 930-step check.** The pilot's carry arm itself dipped to 0.7408 and 0.7449 at later checks.
   *Recommend:* keep 0.74 but judge it on the mean of the 620 and 930 checkpoints (grading at 620 costs ~15 min).
   *Alternative:* leave as is.
4. **Carried memory still does not improve the pick** (flag 5), in three builds running.
   *Recommend:* treat as the next design question once the memory's health is settled — the memory now holds recallable content, but nothing teaches the scorer or generator to use it.
   *Alternative:* test on event moments only, where memory should matter (34 graded moments here: too few to read).
5. **Probe instrument** (flag 3).
   *Recommend:* fix one probe (84 clips, kernel ridge) for all future notes and re-state the design note's figures with it.
   *Alternative:* drop the probe and rely on the trained read-out only.
6. **Promote the choice gates into the rule** (flag 4) for the next pre-registration.
   *Recommend:* yes, the ADE (3 s) and go-rate gates as conditions.
   *Alternative:* keep them as information.
