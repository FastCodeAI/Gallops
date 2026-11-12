# Build step 4: the memory fix (learned start state + gate spread), T1–T4 again, and the short check

Sid's brief of 2026-11-11. Model `src/s6_model.py` version 2 (the pilot's checkpoints are version 1 and still load and evaluate unchanged);
tests `src/s6_tests.py` with `MEMORY_VERSION=2`; short check `src/s8_short.py`; chain `src/build4_chain.sh` (`logs/build4_chain.log`).
Pre-registration: `reports/BUILD4_PREREG.md` (written 2026-11-11 02:43 IST, before the launch at 2026-11-11 04:25:41). No full pilot, no held-out
slice opened, no closed loop, no docker, no pushes.

## Summary

| Item | Result | Evidence |
|---|---|---|
| New parameters | 1,669,648 (was 1,665,537): generator read 263,168, scorer read summary 263,424, rating head new columns 262,144, update block 876,816 (incl. 16 gate biases; the shared gate bias is gone), start state M0 4,096 | `out/s6_params_v2.json` |
| T1 step-0 identity | carry **PASS**, wiped **PASS**: plans, logits and picks bit-identical to 0.1.8b's cached-token forward on all 2,591 moments of the 84 picking clips, with M0 (max abs 4.58) read at every moment | `out/s6_t1_v2.json` |
| T2 routing | **PASS** on a batch with 24 map rows; M0 reached by: generator, grade, human | `out/s6_t2_map_v2.json` |
| T3 wiped | **PASS**: M read = M0 exactly at every read (max deviation 0.0e+00), update block gradient after 5 steps: 27 of 27 tensors None, the rest all-zero; carry M differs from M0 after the first moment (norm of the difference 26.11, 26.85, 28.15, 34.21, 20.36) | `out/s6_t2t3_v2.json` |
| T4 smoke (50 steps per mode) | NaN / inf: **no**; gate mean per slot at step 1 / 50 below | `out/s6_t4_v2.json` |
| Short check (carry, version 2, 930 steps) | **FAIL** by the pre-registered rule (failed: c_memory_change) | `out/build4_short.json`, `runs/build4_short/run.json` |

### Before / after at step 930 (84 picking clips, in_picking moments after the warm-up)

| Measure | Before: pilot carry step 930 (version 1) | After: build 4 carry step 930 (version 2) | Rule | Holds |
|---|---|---|---|---|
| (a) slot diversity, mean pairwise cosine of the memory read | 0.99998 | 0.4417 | ≤ 0.90 | yes |
| (b) gate mean per slot: min / median / max | 0.017 / 0.017 / 0.017 | 0.022 / 0.126 / 0.553 | ≥ 8 slots ≥ 0.05 and max ≥ 3 × min | yes (13 slots ≥ 0.05, max/min 25.7) |
| (c) memory change per moment ‖M_t − M_(t−1)‖ / ‖M_(t−1)‖ | 0.0570 | 0.0425 | ≥ 0.05 | NO |
| (d) NaN / inf | none | none | none | yes |
| (e) pick grade (1,131 graded moments) | 0.7526 (pilot's check) | 0.7496 | ≥ 0.74 | yes |
| memory read norm (mean) | 18.5 | 59.4 | — | — |
| between-clip cosine of the memory at each clip's last in_picking moment | 0.9503 | 0.8478 | information | — |
| same weights in wiped mode (reset to M0 every moment): pick grade | — | 0.7520 | information | — |

"Before" numbers computed by `src/s8_short.py before` on `runs/pilot_seed0/carry/ckpt/p00.2.pt` (`out/build4_before.json`); they agree with
the pilot's own check record for the cosine and the gate: cos_read yes, gate yes.

## The change (Part 1)

`src/s6_model.py` takes `version` (1 = the pilot's model, 2 = build 4); a checkpoint records it (`version` key; the pilot's checkpoints have
none and load as version 1, verified by the "before" column, which loaded `runs/pilot_seed0/carry/ckpt/p00.2.pt` and reproduced the pilot's own
check record exactly, `out/build4_before.json: agrees_with_pilot_check`).

1. **Learned start state** (`StartState.M0`, 16 × 256, new parameter group `start_state`). `init_memory` returns M0 (attached, so it trains)
   at the start of every window, clip or scene; in wiped mode `step_memory` returns M0 again before every moment. Initialisation: 16
   orthonormal rows (`torch.nn.init.orthogonal_`, seed 0) scaled by sqrt(256) = 16. Why that scale: the written candidate rows are LayerNorm
   outputs (unit variance per element, row norm ≈ 16; the pilot's untrained memory had norm 63 over 16 slots ≈ 16 per slot,
   `runs/pilot_seed0/carry/run.json: checks[0].slots.M_read_norm_mean`), so M0 rows and written rows have the same magnitude and the gate's
   Linear([old; candidate]) sees inputs on the pilot's scale. As built: row norms 16.000 ± 0.000002, largest |pairwise cosine| 5e-7
   (`out/s6_params_v2.json: start_state`). The slot identity on the update query (`slot_emb`) is kept.
2. **Gate start spread** (`Update(version=2)`). The gate is `sigmoid(w · [old; candidate] + b_slot)` with `w` (512) initialised N(0, 0.001²)
   (as built: std 0.00098, largest |w| 0.0029) and a per-slot bias `b_slot` (16) initialised to logit(p), p evenly spaced from 0.9 (slot 1)
   to 0.1 (slot 16): start probabilities 0.900, 0.847, 0.793, 0.740, 0.687, 0.633, 0.580, 0.527, 0.473, 0.420, 0.367, 0.313, 0.260, 0.207,
   0.153, 0.100 (`out/s6_params_v2.json: start_state.gate_start_prob`). The shared bias of the version-1 Linear is dropped (the per-slot bias
   replaces it). Both learnable, in the memory optimiser (Adam 1e-4) as before.

**Parameters per block** (`out/s6_params_v2.json: new_params`): generator read 263,168; scorer read summary 263,424; rating head new columns
262,144; update block 876,816 (was 876,801: −1 shared gate bias +16 per-slot biases); start state 4,096. Total new 1,669,648 (was 1,665,537;
+4,111; 11.3 % above the 1.5 M target). Memory state values unchanged (4,096).

Everything else is the pilot's trainer as built (`src/s6_train.py` unchanged apart from two extra measures in the step record:
`M_read_max_dev_from_start`, `M_after_first_moment_dev_from_start_norm`; `src/s6_tests.py` reads `MEMORY_VERSION`).

## T1–T4 on the version-2 model (Part 2)

All four run in one process on 2026-11-11 03:24–04:05 IST (`logs/build4_tests.log`, `logs/build4_chain.log`), p06 sha verified at load
(`out/s6_*_v2.json: agent.sha256`), cores 0–1, nice 19.

**T1 step-0 identity** (`out/s6_t1_v2.json`): **PASS** carry and wiped. The reference is 0.1.8b's cached-token forward recomputed on all 2,591
moments of the 84 picking clips (569 s); the untrained version-2 model replayed in order in each mode: plans, logits and picks bit-identical
(max |Δ| 0.0, picks equal 2,591 / 2,591) in both modes, while M0 was read at every moment (max |M read| 4.58 carry, 3.56 wiped). The reads
are invisible at step 0 because the generator read's output projection and the rating head's new columns are zero. The step-0 measures are
therefore the pilot's (`runs/pilot_seed0/carry/run.json: checks[0]`: pick grade 0.7685, hard 0, go 84.0 %, ADE 0.2384 m) and were not
re-graded. Motion-source check unchanged (2,494 pairs, median mismatch 0.0007 m).

**T2 routing** (`out/s6_t2_map_v2.json`, batch 130 of the seeded pass-1 plan, 24 map rows, carry, after 5 warm-start steps): **PASS**, no
mismatches. Each lesson alone, "gradient non-zero" per group:

| lesson | generator | generator read | rating head | scorer summary | collision head | drivable head | update block | **start state M0** |
|---|---|---|---|---|---|---|---|---|
| generator | yes | yes | no | no | no | no | yes | **yes** |
| grade | no | no | yes | yes | no | no | yes | **yes** |
| human | no | no | yes | yes | no | no | yes | **yes** |
| collision | no | no | no | no | yes | no | no | **no** |
| drivable | no | no | no | no | no | yes | no | **no** |

The seven build-3 columns are exactly build 3's table (`out/s6_t2_map.json`); M0 is reached by the generator lesson (through the generator
read) and by the grade and human lessons (through the scorer's summary, M being detached in the replay's generator read), not by collision or
drivable. The T2 on the chain's batch without map rows (`out/s6_t2t3_v2.json: t2_routing`) shows the same "no graph" for grade / collision /
drivable as in build 3.

**T3 wiped** (`out/s6_t2t3_v2.json: t3_wiped`): **PASS**. Wiped: max |M read − M0| = 0.0 at all 6 steps (5 warm-start steps + the lesson
step); the update block's 27 tensors all have gradient None after 5 steps, while generator, generator read, rating head, scorer summary and
**M0** do get gradients (M0 trains in wiped mode: max |M0| moves 3.56015 → 3.56022 over the 5 steps). Carry: M differs from M0 after the first
moment, ‖M_new − M0‖ = 26.1, 26.9, 28.1, 34.2, 20.4 at the 5 warm steps (norm of M_new 60.6, 58.7, 58.6, 57.4, 62.0).

**T4 smoke** (`out/s6_t4_v2.json`, 50 steps per mode, real learning rates, seed 0): NaN / inf **none** in either mode. Losses first 10 vs
last 10 steps: carry 2.415 → 2.312 (generator 0.665 → 0.583, grade 0.119 → 0.073, human 1.624 → 1.655, collision 0.0074 → 0.0006, drivable
0.0003 → 0.0001); wiped 2.377 → 2.322. Gate mean per slot (carry):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step 1 | 0.898 | 0.845 | 0.786 | 0.737 | 0.684 | 0.626 | 0.577 | 0.516 | 0.471 | 0.415 | 0.360 | 0.311 | 0.262 | 0.201 | 0.148 | 0.097 |
| step 50 | 0.898 | 0.841 | 0.785 | 0.706 | 0.675 | 0.620 | 0.571 | 0.510 | 0.437 | 0.408 | 0.346 | 0.287 | 0.258 | 0.196 | 0.124 | 0.078 |

Wiped: step 1 = step 50 to three decimals (0.899 … 0.098): its update block gets no gradient (T3), so its gate does not move. The spread is
there at step 1 and still at step 50 in carry. Step time 5.5 s (carry) / 5.0 s (wiped) against 1.65 s in build 3: the GPU was shared at
30–100 % utilisation with other users' jobs (`logs/build4_chain.log`); GPU peak 3.4 GB.

## The short check (Parts 3 and 4)

**Launch** (`logs/build4_chain.log`): 04:24:07 IST, after the machine gate (5-min load 24.5, available RAM 45.4 GB; GPU 79.7 GB used by
others at 100 %; NAS 69–77 MB/s on three 18 MB clip reads), cores 0–1, nice 19, detached. Carry, version 2, seed 0, the first 930 steps of the
pilot's seeded pass-1 plan (4,288 steps per pass, 898 duplicated event windows: `run.json: pass_plan`, identical to the pilot's
`passes[0]`). 930 steps 04:25:41–05:44 (4.59 s per step: grading wait 0.24 s, data wait 0.01 s, 3.9 map rows per step), no resume, no
non-finite loss, ended 06:09:28 (`runs/build4_short/run.json`). Losses first 100 vs last 100 steps: 2.369 → 2.340 (generator 0.592 → 0.561,
grade 0.100 → 0.099, human 1.642 → 1.631); the pilot's carry arm over its first 930 steps: 2.374 → 2.335 (`runs/pilot_seed0/carry/run.json`).

**Probes** (`run.json: probes`; steps 0, 310, 620 on the fixed 20-clip subset = the first 20 clips of the picking index, 393 in_picking
moments after the warm-up; step 930 on all 84 clips, 1,705 moments):

| step | clips | (a) cosine read / written | (b) gate min / median / max | slots ≥ 0.05 | max / min | (c) change per moment (mean / median) | memory norm | between-clip cosine | read norms: generator read o / new columns | gate weight norm |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 20 | 0.613 / 0.634 | 0.098 / 0.495 / 0.899 | 16 | 9.2 | 0.069 / 0.062 | 63.0 | 0.744 | 0.000 / 0.000 | 0.022 |
| 310 | 20 | 0.596 / 0.612 | 0.048 / 0.297 / 0.787 | 15 | 16.3 | 0.051 / 0.041 | 61.3 | 0.901 | 0.219 / 0.659 | 0.098 |
| 620 | 20 | 0.579 / 0.598 | 0.039 / 0.240 / 0.761 | 15 | 19.6 | 0.044 / 0.037 | 60.8 | 0.643 | 0.258 / 0.795 | 0.118 |
| **930** | **84** | **0.442 / 0.458** | **0.022 / 0.126 / 0.553** | **13** | **25.7** | **0.042 / 0.036** | 59.4 | 0.848 | 0.274 / 1.059 | 0.166 |

Gate mean per slot at step 930 (slots 1–16): 0.553, 0.387, 0.304, 0.255, 0.217, 0.177, 0.141, 0.137, 0.116, 0.080, 0.072, 0.053, 0.082,
0.047, 0.029, 0.022. The gate biases themselves barely moved (sigmoid 0.900 → 0.899 on slot 1, 0.100 → 0.099 on slot 16): the gates closed
through the gate weight (norm 0.022 → 0.166), i.e. the input-dependent term. M0 did not measurably move either (norm 64.0, pairwise cosine
0.0000 at every probe). The gate mean over the training windows fell from 0.45 (first 100 steps) to 0.25 (last 100) (`run.json: steps`).

**Standard check at step 930** (`run.json: probes[3].check`, `dumps/step0930.npz`): pick grade **0.7496** on the 1,131 graded moments
(pilot's carry arm at step 930: 0.7526; 0.1.8b: 0.7685); hard-brake picks above 10 m/s 0; pull-away go rate 83.0 % (pilot 80.2 %);
closest-of-32 ADE (3 s) 0.2462 m (≤ 0.258 m; pilot 0.2476 m); pick ADE (3 s) 1.166 m; pick-follows-previous 28.9 % (pilot 28.1 %, 0.1.8b
29.2 %); event pick grade 0.9585 on the 34 graded event moments; saturated ties 0, largest last logit 5.7. All three choice gates pass.

**Same weights in wiped mode** (memory reset to M0 before every moment; `probes[3].wiped_same_weights`, `dumps/step0930_wiped.npz`): pick
grade **0.7520** (carry 0.7496: the carried state is worth −0.0024 at step 930, inside the noise of a single check); hard 0, go 84.0 %, ADE
0.2505 m, follows 28.3 %; event 0.9614. Its slot cosine is 0.0000 (it reads M0's orthogonal rows), its one update from M0 changes the memory
by 34 % (0.344) and its between-clip cosine is 0.973.

**Verdict by the pre-registered rule: FAIL on (c)** (`out/build4_short.json: rule`). (a) 0.442 ≤ 0.90 holds; (b) 13 of 16 slots ≥ 0.05 and
max / min 25.7 ≥ 3 holds; (c) 0.042 < 0.05 **fails**; (d) no NaN / inf holds; (e) 0.7496 ≥ 0.74 holds. Per the brief, no further tuning.

## Flags

1. **Condition (c) also holds for the collapsed pilot memory, and fails for the fixed one.** With the same code, the pilot's carry step-930 checkpoint gives a memory change per moment of **0.057** (`out/build4_before.json`), above the 0.05 threshold, while build 4 gives 0.042. The measure is RELATIVE to ‖M_(t−1)‖: the pilot's memory had shrunk to norm 18.5 (gate 0.017 × candidates of norm ≈ 64 gives an absolute step ≈ 1.05 per moment, 5.7 % of 18.5), the build-4 memory keeps norm 59.4 (absolute step ≈ 2.5 per moment, 4.2 % of 59.4). In absolute terms the fixed memory changes 2.4 × more per moment than the collapsed one. The verdict is FAIL as pre-registered; the threshold's meaning is OPEN 1.
2. **The gates are closing during training, through the weight, not the bias.** Gate mean over the training windows 0.45 → 0.25 in 930 steps (`run.json: steps`); at the probes slot 1 fell 0.90 → 0.55 and slot 16 0.10 → 0.02; the biases are unchanged to 3 decimals (Adam 1e-4 can move a bias by at most ≈ 0.09 logit in 930 steps) while the gate weight norm grew 0.022 → 0.166. The spread survives (max / min 25.7, 13 slots ≥ 0.05) but the trend is the pilot's: the lessons push the gate shut. M0 likewise did not measurably move (norm 64.0, cosine 0.0000 at every probe): 930 Adam steps at 1e-4 move an element by at most ≈ 0.09 against elements of size ≈ 1.
3. **The carried state is worth nothing at step 930 on this measure.** The same weights with the memory reset to M0 before every moment grade 0.7520 vs 0.7496 carried (−0.0024). Both are below the pilot's version-1 carry arm at the same step (0.7526) and 0.1.8b (0.7685); the pilot's chosen checkpoint came later (pass 1.216, 0.7752), so step 930 says nothing about where this run would end. Single check, no bootstrap (not pre-registered).
4. **Between-clip cosine is high (0.848 at step 930; 0.950 for the pilot's step 930; 0.973 for the same weights in wiped mode).** The memory written at a clip's last in_picking moment points mostly the same way on every clip: the clip-specific part of the memory is a small fraction of its norm, dominated by a shared component (the update block's candidates are LayerNorm outputs that share a mean direction). The slot cosine (a) is the within-memory measure the rule uses; this between-clip measure was asked for as information and is not in the rule.
5. **Machine contention.** Steps took 4.59 s (short check) and 5.0–5.5 s (T4) against 1.65 s in build 3, with the GPU at 30–100 % utilisation from other users' jobs and the 5-min load 24–30 at launch (`logs/build4_chain.log`); T1's reference forward took 569 s against 148 s in build 3. The chain waited 40 min for the machine gate (02:44–03:24, load 46 → 30). Timings in this report are not comparable with build 3's T5.
6. **T1 did not re-grade the step-0 measures.** Both untrained dumps are bit-identical to the recomputed 0.1.8b reference, so the pilot's step-0 check is the step-0 measure; T1 in build 4 took 23 min (reference + two replays) and skipped the 7-min grading.
7. **The T2 table on the chain's own batch has no map rows (as in build 3).** Batch 5 of the seeded plan holds no map moment, so grade / collision / drivable are constant zeros there (`out/s6_t2t3_v2.json`); the routing verdict is from the map batch (batch 130, 24 map rows), as in build 3.
8. **Resume path not exercised.** The run completed without a crash; `--resume` (weights, optimiser states, step) is built and asserts like the pilot's, but was not tested in this run (the pilot's was).
9. **No orphaned processes after the chain** (`ps` at 06:10 IST: no spawn workers, no test or run pythons); the grading pool is closed in `finally` in both scripts.

## OPEN for Sid

1. **The rule's condition (c) is relative to the memory norm and does not separate collapsed from alive (flag 1).** Recommend: keep the FAIL as recorded (the rule was pre-registered), and for any next check re-register (c) as an ABSOLUTE change per moment, or as the relative change with the memory norm also bounded (e.g. ‖M‖ ≥ 0.5 × ‖M0‖), with the threshold chosen from the pilot's "before" value (0.057 relative, ≈ 1.05 absolute) so that the pilot's memory fails it. Alternative: keep (c) relative and raise the bar to 0.10, which the fixed model also fails today (0.042) — then the fix as built is insufficient and the gate trend (flag 2) must be addressed first.
2. **The gates close through the weight while the biases are frozen by the learning rate (flag 2).** Recommend: before any longer run, give the gate (weight and biases) and M0 a learning rate of 1e-3 (10 × the memory blocks') so the biases can actually move, and re-run T4 + a 930-step check under the same rule; cost ≈ 2.5 h at today's step time. Alternative: keep the learning rates and add a gate floor (e.g. g = 0.05 + 0.95 · sigmoid) that makes a fully closed gate impossible; a design change rather than a tuning change.
3. **The carried memory does not help the pick at step 930 (flag 3).** Recommend: accept that a 930-step check cannot show a pick-grade benefit (the pilot's gain, if any, came at pass 1.2) and read only the memory-health measures from it, as the rule does. Alternative: extend this run to the pilot's chosen point (pass 1.216, ≈ 4,300 more steps, ≈ 5.5 h at today's step time) and compare carry vs reset-to-M0 there.
4. **Between-clip similarity (flag 4).** Recommend: report it, add no rule; if a rule is wanted later, use the cosine between the clip-specific parts (M − mean over clips) rather than between the raw memories. Alternative: ignore (not in the brief's rule).
5. **Resume path (flag 8).** Recommend: exercise it once on purpose in the next run (stop at a probe, resume, compare the next 10 losses with an uninterrupted run); 15 min. Alternative: rely on the pilot's exercised resume (same mechanism).

## Files

- `src/s6_model.py` (version 2 behind `version`; `StartState`, `Update(version)`, `init_memory`, checkpoint `version` key), `src/s6_train.py` (two extra measures in the step record), `src/s6_tests.py` (`MEMORY_VERSION`), `src/s8_short.py`, `src/build4_chain.sh`, `src/s8_fill_report.py`
- `out/s6_params_v2.json`, `out/s6_t1_v2.json`, `out/s6_t2_map_v2.json`, `out/s6_t2t3_v2.json`, `out/s6_t4_v2.json`, `out/build4_before.json`, `out/build4_short.json`
- `logs/build4_chain.log`, `logs/build4_tests.log`, `logs/build4_before.log`, `logs/build4_short.log`
- `{experiment_store}/runs/build4_short/{run.json,status.json,log.txt,resume.pt,ckpt/step0310.pt,step0620.pt,step0930.pt (+.json),dumps/step0930.npz,step0930_grades.npy,step0930_wiped.npz,step0930_wiped_grades.npy}`
- `reports/BUILD4_PREREG.md`
