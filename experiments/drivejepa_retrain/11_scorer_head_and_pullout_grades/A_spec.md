# S9 part A — spec for the generator-only retrain (GPU). NOT RUN.

Written 2026-10-09. Nothing in this file has been executed on the GPU. The one measurement in it (section 3) is a CPU
read of S6's stored UNSEEN menus (`s9_a_spacing_now.py`, `a_spacing_now.csv`).
Paths are relative to `paddock-sid/Alpasim/analysis_3.0/Retrain-data/` unless they start with `/`.

## 1. What it is

A short continuation of S4 from pass 9 in which only the generator learns, with one added loss term that penalises uneven
spacing of the 8 waypoints on all 32 plans. The scorer and the encoder do not change.

Why: S7 H1 and S8 L1. On 43.5 % of UNSEEN highway instants no menu plan grades ≥ 0.95; the limiting check is comfort on
62 % of them; the best plan's 8 step speeds spread over a median 18 m/s (catalogue 1 m/s); the conversion to the graded
path is the same code as for catalogue curves, so the unevenness is in the raw waypoints; it is a fixed shape per menu slot
above about 20 m/s; each waypoint is decoded on its own and the loss has no spacing term.

## 2. Settings

| item | value | source / note |
|---|---|---|
| Start | pass 9: `{runs}/drivejepa_retrain/06_first_training_run/run/ckpt/pass_09.pt` (trained tensors; sha256 `b3130f1f…28b5a`), i.e. the candidate `drive_jepa_s4_pass09.ckpt` (sha256 `8a71c3c0…acdd2`) | S6 REPORT 2.1 |
| Trained | generator only: `hist_encoding`, `init_feature`, `_trajectory_head` (S4 counted 2.38 M parameters for the generator) | `AGENT/drive_jepa_model.py:32-36` |
| Frozen | encoder (304.1 M) as in S4; **scorer** (`_pad_model.scorer.*`, 4.70 M): `requires_grad = False`, left out of the optimiser, kept in eval mode | new for A |
| Scorer losses | not computed (total-grade loss, collision-corner and area losses dropped from the sum): with the scorer frozen they would only push the generator's features toward the old scorer | `05_small_overfit_check/train_patch/drive_jepa_agent_s35.py:415-425` |
| Generator loss | unchanged distance terms (human path and up to 4 catalogue curves ≥ 0.95, best of 32, per refinement stage, earlier stages × 0.1) **plus the spacing term** | `drive_jepa_agent_s35.py:306-350` |
| Data, sampling, inputs | as S4: `04_training_set/manifest.parquet` (126,981 examples), 148,796 draws per pass, highway or graded twice, acceleration zeroed, bonnet mask 37 rows, catalogue `vocab_10240.npy` | `06_first_training_run/settings_manifest.json` |
| Optimiser | Adam, **learning rate 2.5e-5** (a quarter of S4's 1e-4), batch 32, 16-bit mixed precision with the loss in 32-bit, no clipping, GridMask jitter; fresh optimiser state (the stored one covers the scorer too) | S4 REPORT "Schedule" |
| Length | **1-pass pilot × 3 weights**, then **2 passes** at the chosen weight | brief |
| Seed | 20261008, sampler seed + pass | S4 |
| Patch form | a new training-side file next to `drive_jepa_agent_s35.py` (nothing vendored edited), diff listed in the run's report | S3.5 convention |

Note for whoever implements it: the upstream "push plans apart" term has weight 0 (`AGENT/drive_jepa_config.py:50`,
`inter_weight = 0`), so today nothing in the loss acts on the 31 plans that are not the closest one. S8 REPORT 2.2 step 3
listed that term without saying it is switched off; corrected here. The spacing term below is therefore the first loss
that touches all 32 plans.

## 3. The spacing term

For one plan with waypoints p(1)…p(8) in the car's frame and p(0) = the car's position (0, 0):

- d(k) = |p(k) − p(k−1)|, k = 1…8 (distance covered in each half second);
- s(k) = d(k+1) − 2 d(k) + d(k−1), k = 2…7 (second difference: zero for constant speed and for constant acceleration);
- **term = mean over the batch, the 32 plans and the 6 values of |s(k)|**, in metres, computed on the output of every
  refinement stage and combined like the distance term (earlier stages × 0.1, `prev_weight`).

It uses positions only (not heading), all 32 plans (not only the closest), and needs no label. It does not pull plans
toward any speed: a plan that brakes or accelerates steadily costs nothing.

Size of the term today, pass 9, UNSEEN menus (CPU measurement, `a_spacing_now.csv`):

| | all 32 plans, mean | median | 90th pct | plans failing comfort | plans passing comfort | generator's distance term, mean |
|---|---|---|---|---|---|---|
| all instants | 2.47 m | 0.39 m | 4.04 m | 8.78 m | 0.40 m | 0.74 |
| highway | 7.79 m | 1.71 m | 28.2 m | 12.7 m | 0.71 m | 1.60 |
| highway, car ≥ 20 m/s | 13.3 m | 6.23 m | 37.1 m | 15.0 m | 1.31 m | 2.31 |
| highway, car < 20 m/s | 0.79 m | 0.57 m | 1.58 m | 1.51 m | 0.58 m | 0.70 |
| close-pass | 0.40 m | 0.32 m | 0.70 m | 0.76 m | 0.35 m | 0.48 |
| ordinary | 0.39 m | 0.31 m | 0.72 m | 0.77 m | 0.35 m | 0.38 |
| human path | 0.046 m | | 0.086 m | | | |
| catalogue curves | 0.027 m | 0.020 m | 0.056 m | | | |

**Weights for the pilot: 0.03, 0.1, 0.3.** At the start these make the spacing term about 10 %, 33 % and 100 % of the
distance term over all instants (0.07, 0.25, 0.74 against 0.74), and on fast highway instants 0.4, 1.3 and 4.0 against 2.3.
The term is heavy-tailed (a few plans with one waypoint tens of metres out); if the first 200 steps of a pilot show the
total loss dominated by it, the fallback is the same term with each |s(k)| capped at 10 m, declared before the pilot.

## 4. Pilot and choice of weight

Three runs of 1 pass each (4,650 steps) from pass 9, identical except for the weight, plus the readout of pass 9 itself as
the zero-weight reference (no training). After each: the UNSEEN readout of section 6 on `val_full`.
Choice rule, fixed before the pilot: the largest weight that (a) does not trip T2 and (b) keeps "a plan within 1.5 m of the
human on the menu" (S4's 93 %; pass 9: 89.1 % on graded UNSEEN) within 2 points of pass 9. If none qualifies, stop and report.

## 5. Tripwires (S4's, `06_first_training_run/s4_train.py:143-181`)

- **T1** unseen total error rises 3 passes in a row while training error falls. With 2 passes it cannot fire; replaced for A
  by: unseen generator distance error after pass 2 worse than after pass 1 AND worse than pass 9 → stop, keep pass 1.
- **T2** ordinary 4 s distance on unseen clips more than 1.10 × the starting point's (pass 9: 5.51 m).
- **T3** any non-finite loss.
- **T4** GPU memory cap 12 GiB; wall cap to be set by Sid (S4 took 1.2 h per pass with grading in the loop; A grades
  nothing during training, so a pass should be shorter).
- New for A, **T5**: scorer tensors must be bit-identical to pass 9 after every pass (hash check). Any difference stops the run.

## 6. Readout after each pass (UNSEEN, `val_full`, graded examples; same code as S6/S7/S8 so numbers line up)

Generator, by slice:
- share of instants with a menu plan graded ≥ 0.95 and ≥ 0.9;
- for the best menu plan: comfort pass rate, spread of the 8 step speeds, deceleration asked in the first second;
- share of all 32 plans passing comfort; the spacing term itself; S8's per-slot shape table above 20 m/s;
- V1 (fastest of 32 ≥ 0.9 × human speed), a plan within 1.5 m of the human on the menu, ordinary 4 s distance (V4);
- share of plans failing each gate (the frozen scorer now sees different plans).

Frozen scorer on the new menu: regret, picks graded 0, true-best, pick ≥ 0.9, rank agreement. The scorer's features come
from the generator trunk (S6 2.3), so freezing the scorer's weights does not freeze what it reads; this row says whether
the old scorer still works on the new features.

## 7. Success numbers

**The brief says "success numbers as in the table above", and no table reached me with the brief. The numbers below are my
proposal, taken from the S6–S8 measurements they refer to; they need Sid's table or sign-off before the run.**

| id | statement (UNSEEN, graded, pass 9 value in brackets) | proposed line |
|---|---|---|
| A1 | highway instants with a menu plan graded ≥ 0.95 (56.5 %) | ≥ 70 % |
| A2 | highway, car ≥ 20 m/s: spread of the best plan's 8 step speeds, median (14.4 m/s; 18.3 on no-good-plan instants; catalogue 1 m/s) | ≤ 6 m/s |
| A3 | highway: share of all 32 plans passing comfort at car ≥ 20 m/s (no slot above 25 % today) | ≥ 50 % |
| A4 | ordinary: 4 s distance not worse than pass 9 by more than 5 % (5.51 m) | holds |
| A5 | a plan within 1.5 m of the human on the menu (89.1 %) | ≥ 87 % |
| A6 | frozen scorer on the new menu: regret not worse than 0.115 + 0.01; picks graded 0 not above 8.1 % + 1 | holds |
| A7 | no tripwire | holds |

## 8. What A does not do

- It does not touch the scorer, so the pull-out problem (S7 H2, S8 L3, S9 C) and the gate-failing picks (S8 L2, S9 B) are
  out of its reach.
- It does not add a spacing term at inference or resample plans in the adapter.
- No closed-loop trial; that is a separate step after the readout.
