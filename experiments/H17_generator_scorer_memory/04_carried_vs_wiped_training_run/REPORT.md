# Drive-JEPA memory, approach 3: the pilot, memory carried vs memory wiped (seed 0)

shakaal, 2026-11-09 22:30 – 2026-11-10 23:41 IST. Two arms trained in parallel from 0.1.8b pass 0.6 on the cached tokens (build 2) with the build-3 model and trainer, judged on the picking set only. No held-out slice opened, no closed loop, no docker, no pushes. Writes only in this memory folder and `{runs}/generator_scorer_memory/runs/pilot_seed0/` (`R` below). Pre-registration: `reports/PILOT_PREREG.md` (23:09 IST, amended 23:22 IST for Sid's amended brief, before any training step).

Abbreviations as in `BUILD3_MODEL_TRAINER.md`; `R` = `{runs}/generator_scorer_memory/runs/pilot_seed0/`. Plain names: encoder, generator, scorer, rating head, memory. Every number below is read from a file named in the same row.

**Starting model:** `NAS/018b/release/drive_jepa_018b_p06.ckpt`, sha256 `3d566eb2…8e66` verified inside each arm before loading (`R/<arm>/run.json: start.sha256`), 495 / 495 tensors re-compared (`s6_model.load_agent`).

## Summary

| | carry | wiped | 0.1.8b (step 0) | files |
|---|---|---|---|---|
| chosen check | pass 1.216, step 5,218 | pass 1.216, step 5,218 | — | `R/<arm>/run.json: chosen` |
| stop reason, passes, wall | reached 2 passes; 2.00 passes; 23.5 h | reached 2 passes; 2.00 passes; 8.2 h | — | `run.json: stopped, started, ended` |
| overall pick grade (1,131 graded in_picking moments) | **0.7752** | **0.7639** | 0.7685 | `run.json: checks[chosen].pick_grade_mean_map` |
| event pick grade (34 graded event moments, too small to read) | **0.9574** | **0.9674** | 0.9629 | `checks[chosen].event.pick_grade_event` |
| safety gates (hard-brake picks = 0; pull-away go ≥ 0.66; closest-of-32 ADE (3 s) ≤ 0.258 m) | 0 / 74.5 % / 0.253 m → pass | 0 / 69.8 % / 0.249 m → pass | 0 / 84.0 % / 0.238 m | `checks[chosen]` |
| carry with memory zeroed at check time | pick grade **0.7662** (drop +0.0090; relies on memory: **yes**) | — | — | `out/pilot_final.json: comparison.carry_memory_zeroed` |
| slot diversity at the chosen check (mean pairwise cosine between the 16 slots, memory read / written) | 0.9996 / 0.9996 | — (memory zero) / 0.9979 | 0.9995 (untrained read) | `checks[chosen].slots` |
| **verdict by the pre-registered rule** | **STOP** | | | `out/pilot_final.json: comparison` |

## Step 0: clean-up and gates

**Idle processes.** `BUILD3_MODEL_TRAINER.md` flag 12 named 10 idle `drive-jepa` python processes "started 2026-11-07"; `ps -o lstart` shows they started **2026-11-08 00:24–00:25** (the flag's date was wrong by a day): 9 processes = the 8 spawned loader workers + the resource tracker of the build-2 cache job's first (batch-8) attempt, which exited with `os._exit(3)` without shutting its workers down. A second group of 11 had appeared at 2026-11-09 22:18–22:19: the 10 grading-pool workers + tracker of build 3's standalone routing test (`s6_tests.py t2map`), which exited without closing its pool (≈ 1 GB RSS each). All 20: user `siddesh`, parent pid 1 (orphaned), no children, 0 % CPU, every one a `multiprocessing.spawn` child, none attached to any live session or run. **Killed all 20 with SIGTERM at 22:55 IST** (`logs/pilot_step0.log`); available RAM rose 62 → 67 GB. The leak is closed in `s6_tests.py` (the pool is closed in a `finally`). A further 22 orphaned grading workers left by the two arms killed by hand at 23:20 (below) were killed the same way (`logs/pilot_chain.log`).

**Gates at launch** (`logs/pilot_chain.log`, `R/<arm>/run.json: memory_gate, start`): GPU 76.0 GB used of 97.9 GB by other users' jobs (room for 2 × 3.5 GB; build-3 smoke peak 3.4 GB); NAS read 84 MB/s on an 18 MB test read (> 30 MB/s); 1.5 TB free on the NAS volume (`df`); 5-min load 14.5, 61 GB available at the start that took (an earlier launch at 23:20 was refused by the memory gate, 5-min load 32.9, flag 2).

## Step 1: event moments on the picking clips

`src/b4c_events.py` with a new loader mode `picking` and the rules unchanged (`logs/b4c_picking.log`, `out/b4b_picking.json`, `out/b4b_picking_clips.csv`, `out/b4b_picking_events.csv`): the 84 picking clips = their val_full rows (2,591 moments, 0.47 clip-hours). Tracks: the **46 reconstructed clips all have pool clipgt tracks** (`load_nurec`); of the **38 raw clips, 4 have census track files and 34 have none** (`out/pilot_picking_clip_sources.csv`; these 34 get ego-only events, hard braking, as the held-out mode does). Events found: 1 hard-brake (3 m/s²), 3 cut-ins, 2 crossings, 0 out-of-view returns; 3,361 tracks. `src/s3_moments.py picking` (rule unchanged: a moment within (t_event, t_event + 5 s], one row per moment and type) → `out/memory_moments_picking.parquet`: 49 rows, **40 distinct event moments of 2,591 (1.5 %)** (`out/s3_moments_picking.json`).

Among the 2,000 in_picking moments: **34 event moments, all 34 graded** (`out/pilot_event_subset.json`; 25 cut-in, 12 crossing, 3 hard-brake rows; 5 clips; `out/pilot_event_moments_in_picking.csv`). **The event subset is 34 moments from 5 clips: too small to read (< 50)**, and 0.1.8b already grades it at 0.9629 at step 0 (little headroom).

## Step 2: pre-registration

`reports/PILOT_PREREG.md`, written 23:09 IST and amended 23:22 IST for the amended brief (2 passes, the CLOSE rule, what GO leads to), both before any training step (launch 1 at 23:09 died on an import, launch 2 at 23:10 was stopped by hand inside its step-0 check when the amended brief arrived, launch 3 at 23:20 was refused by the memory gate, launch 4 at 23:51 is the run; `logs/pilot_chain.log`). The rule as written: GO iff carry overall ≥ wiped overall AND carry − wiped on the graded event moments ≥ +0.01 AND both arms pass the safety gates; else STOP; CLOSE flag if |overall gap| < 0.005, or event gap in [0, +0.02], or the carry model's pick grade with its memory zeroed within 0.002 of its own.

## Step 3: the runs

`src/s7_pilot.py` (one arm per process) over the build-3 trainer (`s6_train.window_step`): 2 windows × 16 moments per step, attempt-2 lessons by import, scorer reads memory only through its own summary, grading pool of 10 on the arm's two cores; up to 2 passes of 4,288 steps (event over-sampling on; `run.json: passes`); a check every 930 steps and at each pass end (the 84 picking clips replayed in order, fp32, batch 1); stop after 6 checks without a better choosable pick; chosen = best pick grade among checks passing both safety gates. Each check saves `R/<arm>/ckpt/p<pass>.pt` (+ `.json`: mode flag, start sha, seed, step), `dumps/p<pass>.npz` + `_grades.npy`, `run.json`, `status.json`, `resume.pt` (weights, optimiser states, sampler position, counters; `--resume` continues at the next step of the same seeded pass plan and records a RESUMED line, never a silent restart). Progress line every 100 steps in `R/<arm>/log.txt`. Launcher `src/run_model_pilot.sh`: carry `taskset -c 0-1`, wiped `taskset -c 2-3`, nice 19 (Sid's one-off 4-core exception).

| | carry | wiped |
|---|---|---|
| started / ended (IST) | 11-09 23:52 / 11-10 23:25 (23.5 h) | 11-09 23:52 / 11-10 08:05 (8.2 h) |
| steps trained, passes | 9,485 steps, 2.00 passes | 8,585 steps, 2.00 passes |
| checks (trained) | 10 | 10 |
| stop reason | reached 2 passes | reached 2 passes |
| resumes | 1 | 0 |
| step time (mean, s) / grading wait / data wait | 2.63 / 0.42 / 0.02 | 2.67 / 0.39 / 0.01 |
| non-finite losses | none | none |

### Learning curves (every check; `run.json: checks`)

carry:

| pass | step | pick grade | event grade | hard | go (pull-away) | closest-32 ADE (3 s) | pick ADE (3 s) all / event | follows prev | slots cos read / written | gate mean | ‖gen read o‖ | ‖new cols‖ | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 0 | 0.7685 | 0.9629 | 0 | 84.0 % | 0.238 | 1.108 / 1.050 | 29.2 % | 0.9995 / 0.9996 | 0.466 | 0.000 | 0.000 | pass |
| 0.217 | 930 | 0.7526 | 0.9662 | 0 | 80.2 % | 0.248 | 1.163 / 1.435 | 28.1 % | 1.0000 / 1.0000 | 0.017 | 0.221 | 0.713 | pass |
| 0.434 | 1,860 | 0.7500 | 0.9329 | 0 | 71.7 % | 0.247 | 1.154 / 1.448 | 29.0 % | 0.9998 / 0.9998 | 0.014 | 0.329 | 1.001 | pass |
| 0.651 | 2,790 | 0.7656 | 0.9135 | 0 | 51.9 % | 0.252 | 1.272 / 1.725 | 27.6 % | 1.0000 / 1.0000 | 0.008 | 0.384 | 1.232 | FAIL |
| 0.868 | 3,720 | 0.7611 | 0.9646 | 0 | 77.4 % | 0.249 | 1.317 / 2.052 | 27.5 % | 0.9998 / 0.9999 | 0.005 | 0.461 | 1.638 | pass |
| 1.000 | 4,288 | 0.7699 | 0.9413 | 0 | 75.5 % | 0.250 | 1.131 / 1.766 | 29.9 % | 0.9998 / 0.9998 | 0.006 | 0.504 | 1.779 | pass |
| 1.216 | 5,218 | 0.7752 | 0.9574 | 0 | 74.5 % | 0.253 | 1.207 / 1.764 | 29.2 % | 0.9996 / 0.9996 | 0.006 | 0.574 | 2.128 | pass |
| 1.433 | 6,148 | 0.7408 | 0.9485 | 0 | 82.1 % | 0.262 | 1.154 / 1.530 | 27.2 % | 0.9995 / 0.9995 | 0.004 | 0.671 | 2.570 | pass |
| 1.649 | 7,078 | 0.7616 | 0.9370 | 1 | 60.4 % | 0.268 | 1.382 / 1.557 | 29.1 % | 0.9996 / 0.9996 | 0.005 | 0.740 | 2.760 | FAIL |
| 1.866 | 8,008 | 0.7622 | 0.9099 | 0 | 88.7 % | 0.261 | 1.045 / 1.201 | 28.9 % | 0.9992 / 0.9993 | 0.004 | 0.789 | 2.966 | pass |
| 2.000 | 8,585 | 0.7449 | 0.9174 | 0 | 56.6 % | 0.262 | 1.196 / 1.442 | 28.4 % | 0.9996 / 0.9996 | 0.004 | 0.853 | 3.009 | FAIL |

wiped:

| pass | step | pick grade | event grade | hard | go (pull-away) | closest-32 ADE (3 s) | pick ADE (3 s) all / event | follows prev | slots cos read / written | gate mean | ‖gen read o‖ | ‖new cols‖ | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 0 | 0.7685 | 0.9629 | 0 | 84.0 % | 0.238 | 1.108 / 1.050 | 29.2 % | — / 0.9980 | 0.532 | 0.000 | 0.000 | pass |
| 0.217 | 930 | 0.7563 | 0.8784 | 2 | 70.8 % | 0.249 | 1.144 / 1.252 | 30.5 % | — / 0.9981 | 0.532 | 0.144 | 0.947 | FAIL |
| 0.434 | 1,860 | 0.7481 | 0.9432 | 0 | 63.2 % | 0.239 | 1.156 / 1.330 | 29.4 % | — / 0.9981 | 0.532 | 0.154 | 1.377 | FAIL |
| 0.651 | 2,790 | 0.7481 | 0.8998 | 1 | 54.7 % | 0.278 | 1.298 / 1.045 | 26.9 % | — / 0.9980 | 0.531 | 0.175 | 1.544 | FAIL |
| 0.868 | 3,720 | 0.7481 | 0.9721 | 0 | 75.5 % | 0.248 | 1.176 / 1.500 | 27.2 % | — / 0.9981 | 0.532 | 0.227 | 1.702 | pass |
| 1.000 | 4,288 | 0.7595 | 0.9632 | 0 | 69.8 % | 0.239 | 1.147 / 1.148 | 29.2 % | — / 0.9979 | 0.532 | 0.221 | 2.024 | pass |
| 1.216 | 5,218 | 0.7639 | 0.9674 | 0 | 69.8 % | 0.249 | 1.177 / 1.172 | 29.8 % | — / 0.9979 | 0.532 | 0.248 | 2.037 | pass |
| 1.433 | 6,148 | 0.7476 | 0.9383 | 0 | 79.2 % | 0.252 | 1.097 / 1.348 | 28.8 % | — / 0.9981 | 0.533 | 0.280 | 2.187 | pass |
| 1.649 | 7,078 | 0.7615 | 0.9679 | 0 | 57.5 % | 0.251 | 1.213 / 1.173 | 27.9 % | — / 0.9981 | 0.532 | 0.324 | 2.520 | FAIL |
| 1.866 | 8,008 | 0.7506 | 0.8548 | 0 | 82.1 % | 0.262 | 1.145 / 1.402 | 29.0 % | — / 0.9980 | 0.532 | 0.422 | 2.975 | pass |
| 2.000 | 8,585 | 0.7368 | 0.9374 | 0 | 65.1 % | 0.256 | 1.268 / 1.441 | 26.7 % | — / 0.9979 | 0.532 | 0.420 | 3.177 | FAIL |

Curves: both arms dip below 0.1.8b in pass 1 and peak at pass 1.216; carry's slot cosine stays at 0.9992–1.0000 at every check (one slot in effect), its gate mean per slot falls from 0.5 (untrained) to 0.006 at the chosen check; memory-use norms rise monotonically in both arms. Carry checks with a failed gate: 3 of 10; wiped: 5 of 10.

## Step 5: the comparison (measuring only)

### Pre-registered measures at the chosen checkpoints

| measure | carry | wiped | gap (carry − wiped) | 0.1.8b step 0 |
|---|---|---|---|---|
| overall pick grade, 1,131 graded moments | 0.7752 | 0.7639 | **+0.0113** | 0.7685 |
| event pick grade, 34 graded event moments (too small to read) | 0.9574 | 0.9674 | **-0.0100** | 0.9629 |
| hard-brake picks above 10 m/s (= 0) | 0 | 0 | | 0 |
| pull-away go rate (≥ 0.66) | 74.5 % | 69.8 % | | 84.0 % |
| closest-of-32 ADE (3 s) (≤ 0.238 + 0.02 m) | 0.253 m | 0.249 m | | 0.238 m |
| safety gates | **0 / 74.5 % / 0.253 m → pass** | **0 / 69.8 % / 0.249 m → pass** | | pass |

**Rule:** carry overall ≥ wiped: yes; event gap ≥ +0.01: no; both arms pass the gates: yes → **STOP**. CLOSE flags: |overall gap| < 0.005: no; event gap in [0, +0.02]: no; carry with memory zeroed within 0.002 of itself: no → **not close**. **Verdict: STOP.**

**Information, not the rule — clip-clustered paired bootstrap (clips resampled with replacement, 2,000 draws, seed 0; `out/pilot_final.json: comparison.bootstrap`):** overall gap +0.0113 [95 % -0.0076, +0.0315] (1131 moments, 46 clips); event gap -0.0100 [95 % -0.0225, -0.0023] (34 moments, 5 clips); carry − carry-with-memory-zeroed +0.0090 [95 % -0.0088, +0.0350] (1131 moments, 46 clips).

### The carry arm with its memory zeroed at check time

The chosen carry checkpoint (`{runs}/generator_scorer_memory/runs/pilot_seed0/carry/ckpt/p01.2.pt`) replayed with `mode = wiped` (M zeroed before every moment, same weights; `s7_pilot_final.py`): pick grade **0.7662** against its own 0.7752 (drop +0.0090); event grade 0.9488; gates 0 / 39.6 % / 0.250 m → FAIL; pick-follows-previous 30.5 % vs 29.2 %. **The pick grade drops by more than 0.002 when the memory is zeroed: the carry model relies on its memory.** (`R/carry/dumps/chosen_memory_zeroed.npz`).

### Other diagnostics at the chosen checks

| | carry | wiped | 0.1.8b |
|---|---|---|---|
| scorer-pick ADE (3 s), all 2,000 / event moments | 1.207 / 1.764 m | 1.177 / 1.172 m | 1.108 / 1.050 m |
| pick-follows-previous rate | 29.2 % | 29.8 % | 29.2 % |
| slot cosine, memory read / written (after warm-up) | 0.9996 / 0.9996 | — / 0.9979 | 0.9995 / 0.9996 |
| gate mean (over slots) | 0.006 | 0.532 | — |
| ‖generator read output projection‖ / ‖rating head new columns‖ | 0.574 / 2.128 | 0.248 / 2.037 | 0 / 0 |

## Flags

1. **Carry arm crashed once and was resumed.** At 04:54 IST its pass-1.216 check hit 0.1.8b's check assertion "pick is not the argmax of the last logit" (`logs/pilot_carry_part1_crashed_p1.2_check.log`): the model's pick is argmax(sigmoid(last logit)) (`drive_jepa_model.py:85, 90`) and two logits so large that sigmoid rounds both to 1.0 in fp32 make the two argmaxes differ. The check now counts such saturated ties instead of aborting (`s6_train.safety_tolerant`, the imported function with that one line changed; verified identical on a clean dump) and records the largest last logit. Carry was resumed at 19:19 IST from `resume.pt` (pass 1.0, step 4,288; `run.json: resumes`), so the ≈ 930 pass-2 steps before the crash were trained again from the pass-1.0 weights, with the same batch order (seeded plan) but different numerics (atomic adds in the deformable attention, BUILD3 flag 6; the vocabulary-term draws also restart). In the resumed trajectory **no check showed a saturated tie** (largest last logit 7.1; `run.json: checks[*].logit_saturation`); the saturated check's dump of the crashed trajectory was not saved. Wall time of the carry arm includes the 14.4 h gap between the crash and the resume (04:54–19:19).
2. **Carry's 16 slots never differentiate and its gate is almost closed at the chosen check.** Mean pairwise cosine between the slots of the memory read at in_picking moments after the warm-up: 0.9996 (written: 0.9996); untrained it was 0.9995. The gate mean per slot at the chosen check is 0.006 (all 16 within 0.006–0.006), i.e. M_new ≈ 0.994 · M_old: after the first moments the memory barely changes (mean ‖M read‖ 14.5). The wiped arm's gate sits at 0.532 (its memory is zeroed before every moment, so its update is never read). Memory-use norms grow steadily in both arms (generator read output projection 0.574 carry vs 0.248 wiped; rating-head new columns 2.13 vs 2.04), so both arms use the memory path, the wiped arm as a constant input. The zero-memory diagnostic (step 5) is the direct test; OPEN 2.
3. **Event subset: 34 graded moments from 5 clips, and 0.1.8b already grades it at 0.963 at step 0.** Too small to read (pre-registered caveat) and near the ceiling: the event gap (-0.0100) moves by ±0.03 between neighbouring checks in both arms (`run.json: checks[*].event.pick_grade_event`). The 34 raw picking clips have no tracks (`out/pilot_picking_clip_sources.csv`), so only ego-only events could have been found on them; none were.
4. **Safety gates were the binding constraint, not the pick grade.** Of the trained checks, carry failed the choice gates (hard-brake picks = 0 and pull-away go ≥ 0.66) at 3 of 10 and wiped at 5 of 10; the closest-of-32 ADE (3 s) exceeded the pre-registered +0.02 m tolerance at 4 carry and 2 wiped checks (`run.json: checks`). The arm's choice rule (as written by Sid) uses the two gates; the pre-registered comparison adds the ADE tolerance; both chosen checks pass all three. The pull-away go rate swings between 0.52 and 0.89 from check to check (106 standstill moments), the same go/stay swing as 0.1.8b attempt 2 (`generator-joint-pai-s8` report).
5. **Pick grades went DOWN before they went up.** Both arms fell below 0.1.8b's 0.7685 for most of pass 1 (carry 0.7526 → 0.7500 → 0.7656 → 0.7611 → 0.7699; wiped 0.7563 → 0.7481 ×3 → 0.7595) and peaked at pass 1.216 (carry 0.7752, wiped 0.7639), then fell again by pass 2.0 (0.7449 / 0.7368 with failed gates). The chosen checks are single peaks on noisy curves; the clip-clustered bootstrap intervals in step 5 give the width.
6. **Wall time and throughput.** Steps averaged 2.63 s (carry) / 2.67 s (wiped) against 1.65 s alone in build 3: the two arms shared the GPU with other users' jobs at 90–100 % utilisation all night (`logs/pilot_chain.log`), grading wait 0.42 s, data wait 0.023 s. A pass took ≈ 4 h with its 5 checks (≈ 10 min each). Wiped: 8.2 h for 2 passes; carry: 23.5 h wall including the crash gap.
7. **Four launches before the run** (`logs/pilot_chain.log`): 23:09 import-order crash (the older `pai_train` copy imported first, fixed by importing the trainer module first); 23:10 a 3-pass launch stopped by hand inside its step-0 check when Sid's amended 2-pass brief arrived (its 22 orphaned grading workers killed); 23:20 refused by the memory gate (5-min load 32.9, inflated by the kill/relaunch churn); 23:51 the run. Nothing from the stopped launch was kept.
8. **BUILD3 flag 12's date was wrong**: the idle processes started 2026-11-08 00:24–00:25 (the batch-8 cache attempt's loader workers), not 2026-11-07; a second, larger group (10 grading workers, ≈ 1 GB each) came from build 3's standalone routing test on 2026-11-09 22:18. All 20 were mine, orphaned (parent pid 1), idle; killed (step 0). The leak is closed in `s6_tests.py`.
9. **Carry with memory zeroed, as measured:** pick grade 0.7662 vs the carry model's own 0.7752 (drop +0.0090); event 0.9488; gates FAIL (hard 0, go 39.6 %, ADE 0.250 m). The drop exceeds 0.002: the carry model relies on its memory.
10. **Status-script ETA was noisy** (it used the last 100-step line's step time); the chain log and `run.json` carry the real times.

## OPEN for Sid

1. **What next, given the verdict.** Recommend: STOP as pre-registered and review: the event gap (-0.0100 on 34 moments) fails the +0.01 rule while the overall gap is +0.0113; the event subset cannot carry a rule (flag 3). Alternative: re-register with the overall gap and the zero-memory test as the rule and the event subset as information, then decide on seeds 1 and 2; or run seeds 1 and 2 now under the current rule.
2. **The memory is single-slot and nearly frozen in carry (flag 2).** Recommend: before any further seed, change the update so that writing is forced early (e.g. gate bias initialised open, g ≈ 0.9, and a per-slot embedding also on the keys/values of the reads), and re-run the build-3 tests; cost ≈ 1 h of tests. Alternative: keep the design as is and let seeds 1 and 2 decide; the zero-memory test (+0.0090) says how much the carried state matters today.
3. **Choice gates vs comparison gates.** The arm's choice used hard-brake = 0 and pull-away go ≥ 0.66 (Sid's rule); the pre-registered safety gates add the ADE (3 s) tolerance. Recommend: add the ADE tolerance to the choice rule in any further run so a chosen check cannot fail the comparison. Alternative: keep them apart (as now) and report both.
4. **Resumed trajectory.** The carry arm's pass-2 segment was retrained after the crash (flag 1), so its curve is one trajectory with a restart, not the uninterrupted run. Recommend: accept (same batch order; the chosen check is in the resumed segment and is fully measured). Alternative: re-run carry from scratch (≈ 8 h) for a clean single trajectory.
5. **Logit saturation in the check.** Recommend: keep the tolerant check (counts ties, keeps the model's actual pick, records the largest last logit at every check). Alternative: restore the assertion and treat a tie as a failed check.
6. **Event subset.** Recommend: for any further pilot, define the event subset on the graded held-out slice (355 reconstructed clips with clipgt tracks, 3.55 % event moments ≈ 370 moments) rather than the picking set; it is read once at the end either way. Alternative: enlarge the picking set's event share by adding reconstructed val clips with events to the picking clips (a new frozen set).

## Files

- `src/s7_pilot.py` (the arms), `src/run_model_pilot.sh`, `src/pilot_status.sh`, `src/s7_pilot_final.py` (step 5), `src/b4c_events.py` + `src/s3_moments.py` (mode `picking`)
- `out/pilot_event_subset.json`, `out/pilot_event_moments_in_picking.csv`, `out/pilot_picking_clip_sources.csv`, `out/memory_moments_picking.parquet`, `out/s3_moments_picking.json`, `out/b4b_picking*.{json,csv}`, `out/pilot_final.json`
- `logs/pilot_chain.log` (every launch, stop and return code), `logs/pilot_step0.log`, `logs/pilot_{carry,wiped}.log`, earlier attempts `logs/pilot_*_attempt{1,2,3}_*.log`, `logs/b4c_picking.log`, `logs/s3_moments_picking.log`
- `R/{carry,wiped}/{run.json, status.json, log.txt, resume.pt, ckpt/p*.pt(+.json), dumps/p*.npz, dumps/p*_grades.npy}`, `R/carry/dumps/chosen_memory_zeroed.npz`
