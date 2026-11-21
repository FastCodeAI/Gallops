# Baselines for the approach-4 pre-registration

shakaal, 2026-11-17. Read-only: no builds, no runs, no simulator. The only computation was reading existing JSON/CSV files with plain `python3` (`json`/`csv` only, a few seconds each). Confidence words: **certain** (read from the run outputs or code), **likely**, **unknown**.

Path abbreviations as in `READONLY_CARSTATE_HISTORY.md`, plus:
- `G` = `Gallops/Gallops-sid/LORA Refiner/Generator-Joint-PAI-alpasim/`
- `A3` = `Gallops/Gallops-sid/memory/3_generator_scorer_memory_PAI/`
- `NASG` = `{retrain_store}/`

**Scoring.** Every closed-loop number below is `rollouts[0].score` of the run's `aggregate/results-summary.json`, read by the lane's own code (`G/s5_diagnosis/src/stepb.py` `score`). That is the **windowed** score. The **full-drive** re-score from earlier today (`A3/reports/FULLDRIVE_RESCORE.md`) is given beside it, because it changes the picture on these 12 scenes (§1b). The pre-registration must name one.

---

## 1. Closed loop (the gate)

### 1a. The 12 diagnosis scenes and why each was chosen

**The rule, verified against the data:** the 12 are **exactly the 12 of the 30 AlpaSim scenes where 0.1.8 and 0.1.7 scored differently**. On the other 18 the difference is 0.0000. Those 18 are the approach-3 CL18 set. Evidence:
- the per-scene "0.1.8 − 0.1.7" column, `G/s4_closed_loop/REPORT.md:159-190`;
- the kind labels, `G/s5_diagnosis/REPORT.md:168-181`;
- "The 12 scenes were chosen for losses and gains", `:220`;
- the list, `G/s10_018b_closed_loop/tables/scenes_12.txt`.

The selection rule is not written down anywhere as a formula; I reconstructed it from the 30-scene table.

| scene | set | why chosen (0.1.8 − 0.1.7) | what happens there (s5 diagnosis) |
|---|---|---|---|
| e3ed554f | EXT20 | big loss (−1.000) | scorer picks unsafe on a thinning menu, at-fault front |
| 8023dce5 | EXT20 | big loss (−1.000) | slow truck ahead, at-fault front; every 0.1.8 variant crashes here |
| 5f064b02 | EXT20 | big loss (−0.997) | standstill: 0.1.8 never pulls away |
| e121e37d | EXT20 | other loss (−0.257) | scorer hedging (slow / off-route) |
| 048b974e | P10 | other loss (−0.069) | scorer hedging |
| cdd9448c | EXT20 | other loss (−0.062) | scorer hedging |
| d2715023 | EXT20 | other loss (−0.032) | never-move scene, a lateral pull-out |
| bd07fb91 | EXT20 | gain (+1.000) | 0.1.7 at-fault, 0.1.8 passes |
| bae463d3 | EXT20 | gain (+0.689) | route: 0.1.8 keeps the human route |
| d1a25a99 | EXT20 | gain (+0.505) | mixed; off-road / collision |
| b1b170d0 | EXT20 | gain (+0.139) | fast handover (16.7 m/s) |
| 04749bb9 | P10 | gain (+0.044) | fast handover (23.7 m/s) |

**Caveat for the pre-registration:** these scenes were selected because two *other* models disagreed on them. They are the scenes most sensitive to the pick, so the 12-scene mean is not a 30-scene estimate (`s5_diagnosis/REPORT.md:220`). **Confidence:** certain for the membership, likely for the selection rule as stated.

### 1b. 0.1.8b-p06 on the 12, native driver

Runs: `NASG/018b/closed_loop/runs/drivejepa_018bp06_<scene>/` (2026-10-31, `G/s10_018b_closed_loop/REPORT.md:79-94`, `tables/stepD.csv`). Full-drive scores from `A3/out/fulldrive_rescore.csv` (`arm` = 018b).

| scene | p06 windowed | p06 full drive | exit (windowed) | 0.1.7 windowed | 0.1.7 full |
|---|---|---|---|---|---|
| e3ed554f | 1.000 | 1.000 | pass | 1.000 | 1.000 |
| 8023dce5 | **0.000** | 0.000 | at-fault front +6.90 s | 1.000 | 1.000 |
| 5f064b02 | 1.000 | 1.000 | pass | 1.000 | 1.000 |
| e121e37d | 1.000 | 1.000 | pass | 1.000 | 1.000 |
| 048b974e | 1.000 | 1.000 | pass | 1.000 | 1.000 |
| cdd9448c | 1.000 | 1.000 | pass | 1.000 | 1.000 |
| d2715023 | 1.000 | 1.000 | pass (moves) | 0.041 | 0.041 |
| bd07fb91 | 0.000 | 0.000 | at-fault front +6.95 s | 0.000 | 0.000 |
| bae463d3 | 0.380 | 0.380 | pass | 0.311 | 0.314 |
| d1a25a99 | 0.000 | 0.000 | off-road +6.95 s | 0.000 | 0.000 |
| b1b170d0 | 0.723 | **1.000** | pass | 0.782 | 1.000 |
| 04749bb9 | **0.000** | 0.000 | at-fault front +12.40 s | 0.956 | **0.000** |
| **mean (12)** | **0.5919** | **0.6150** | | **0.6742** | **0.6130** |

- On the windowed score, p06 trails 0.1.7 by 0.082 on these 12 (s10's bar was ≥ 0.6742, and p06 failed it).
- **On the full-drive score the two are level: 0.6150 vs 0.6130.** The full-drive score fails 0.1.7's 04749bb9 run (at-fault collision after the old window ended) and raises b1b170d0 for both models.

**Confidence:** certain.

### 1c. Run-to-run band: same checkpoint, same scenes

| repeat | scenes | result | source |
|---|---|---|---|
| p06 re-run on a rebuilt simulator image (`alpasim-base:0.134.0` id `9c5909da…`, renderer `nre-ga:26.04` digest `97f43e71…`), 2026-11-12/13 | e3ed554f, 8023dce5, d2715023, 04749bb9 | scores 1.0, 0.0, 1.0, 0.0, **equal to the originals**; **all 9,004 per-tick metric rows per scene bit-identical** | `A3/reports/CL12_MEMORY.md:84-90`; `A3/out/cl12_imagecheck.json`; runs `{experiment_store_h17}/runs/cl12/imagecheck_018bp06/` |
| any other repeat of p06 on any of the 12 | — | **none found** | listed every `drivejepa_018bp06_*` folder on the NAS: `NASG/018b/closed_loop/runs/` (12 scenes, once each), `approach3/runs/cl12/imagecheck_018bp06/` (the 4 above), `approach3/runs/cl18/018bp06/` (the other 18 scenes, not these) |

**Per-scene differences: 0.000 on 4 of 4. Mean difference 0.000.** The other 8 scenes were never repeated.

**What this means for a band.**
- The simulator plus driver is **deterministic**: the same checkpoint, inputs and image give bit-identical runs. A plain repeat therefore measures nothing; its band is 0 by construction. The memory notes record the same for 0.1.7 image re-checks (2026-10-27 and 2026-11-03), but I did not re-read those files for this report.
- The real uncertainty is **sensitivity**: a near-tie in the scorer flips a scene between about 0 and about 1. CL12 says it directly: "one scene flipping moves the 12-scene mean by about 0.08" (`A3/reports/CL12_MEMORY_PREREG.md:50`).
- The Paddock bridge vs the native driver on the same checkpoint (0.1.4b, 30 scenes) moved per-scene scores by up to 0.55, from near-ties (memory note on DRIVEJEPA_014B close-out; `paddock-sid/DRIVEJEPA_PADDOCK_ROUTE_CLOSE.md`, not re-read). That is the only measured "same weights, slightly different inputs" spread, and it comes from a different route and checkpoint.
- **Cost of one full repeat of p06 on the 12** (for completeness, although a repeat is expected to reproduce exactly): 5–7 min per scene measured from the run logs (imagecheck runs 5.3–6.8 min, s10 runs 5.0–5.4 min; the first imagecheck run took 46 min because it rebuilt the image). That is **≈ 60–85 min of run time for 12 scenes**, plus machine-gate waits, which reached 5 h in CL12 (`CL12_MEMORY.md:169`).
- **A band that means something** needs perturbed repeats, for example input noise or another seed. Whether AlpaSim's `random_seed` changes anything on these scenes is **unknown**.

**Confidence:** certain for the repeats, likely for the cost.

### 1d. Approach-3 build 5, carried vs reset: finished

The CL12 run used build 5 at step 930 (`CL12_MEMORY.md:3, 52-57`) and finished on 2026-11-13. Windowed scores from `A3/out/cl12_table.csv` / `CL12_MEMORY.md:96-110`; full drive from `A3/out/fulldrive_rescore.csv`.

| scene | carry | reset | carry − reset | carry full | reset full |
|---|---|---|---|---|---|
| e3ed554f | 1.000 | 1.000 | 0 | 1.000 | 1.000 |
| 8023dce5 | 0.000 | 0.000 | 0 | 0.000 | 0.000 |
| 5f064b02 | **1.000** | **0.172** | +0.828 | 1.000 | 0.000 |
| e121e37d | 1.000 | 1.000 | 0 | 1.000 | 1.000 |
| 048b974e | 0.948 | 1.000 | −0.052 | 0.948 | 1.000 |
| cdd9448c | 1.000 | 1.000 | 0 | 1.000 | 1.000 |
| d2715023 | 1.000 | 1.000 | 0 | 1.000 | 1.000 |
| bd07fb91 | 0.000 | 0.000 | 0 | 0.000 | 0.000 |
| bae463d3 | **1.000** | **0.040** | +0.960 | 1.000 | 0.040 |
| d1a25a99 | 0.000 | 0.000 | 0 | 0.000 | 0.000 |
| b1b170d0 | 1.000 | 1.000 | 0 | 1.000 | 1.000 |
| 04749bb9 | **0.707** | **0.000** | +0.707 | **0.000** (off-road) | 0.000 |
| **mean** | **0.7213** | **0.5177** | **+0.204** | **0.6623** | **0.5033** |

Read narrowly: the gain sits in 3 scene flips (`CL12_MEMORY.md:9-15`). Offline, the same weights pick slightly better reset than carried (0.7355 vs 0.7313).

On the 30 scenes with the full-drive score, carry is 0.6302 vs 0.1.8b 0.7022, "CLOSE" by its pre-registered rule (`A3/reports/FULLDRIVE_RESCORE.md:5-12`). **Confidence:** certain.

### 1e. The "collapse" floor: lowest score for a car that stalled or never started

| case | model | scene | score | exit | run |
|---|---|---|---|---|---|
| **lowest** | 0.1.8-p9 (graft) | d2715023 | **0.0018** | "pass" (never moves) | `NASG/graft/closed_loop/runs/drivejepa_018p9_d2715023/` (`G/s6_graft/REPORT.md:148`; `G/s10…/tables/stepD.csv` `s018p9`) |
| next | 0.1.8 | 5f064b02 | 0.004 | stall at a standstill | `G/s5_diagnosis/REPORT.md:172` |
| | 0.1.8 | d2715023 | 0.009 | pass (never moves) | `stepD.csv` `s018` |
| | 0.1.8b-p02 | d2715023 | 0.033 | pass | `stepD.csv` `s018bp02` |
| | 0.1.7 | d2715023 | 0.041 | pass (never moves) | `stepD.csv` `s017` |
| | 0.1.8-rs | d2715023 | 0.051 | pass (never moves) | `s5_diagnosis/REPORT.md:181` |

- A car that never moves is **not** failed by the scorer: its exit is "pass" and the score is tiny but non-zero (progress-based).
- A stall therefore costs a scene **0.95–1.00**, about the same as an at-fault crash (0.000).
- "Collapse" floor for the pre-registration: **per-scene ≤ 0.05 with no collision = stalled**, grounded in six real runs at 0.0018–0.051.

**Confidence:** certain for the values, likely for "≤ 0.05" as the threshold.

---

## 2. Pull-away go rate (kill switch)

**A definition exists and was a selection gate for 0.1.8b.** Code: `G/s9_018b_train/src/train_018b.py:137-152` (`go_split`), with gate `GO_PULL_MIN = 0.66` (`:72`, "Sid 2026-10-31"). On the map moments of the picking set (2.3 below):
- **standstill:** ego speed < 1 m/s **and** at least one safe go plan on the menu ("safe" = not colliding and not off drivable area; "go plan" = the plan's end point ≥ 1 m from its start) (`:141-143`);
- **pulled away:** standstill and the human's position at +4 s ≥ 1 m from the start (`:144`);
- **go rate:** the share of pulled-away moments where the scorer's pick is a go plan (`:146, 150`).

The companion measure is "waits correctly": standstill moments where the human stayed (< 1 m by +4 s) and the pick also stays (`:145, 151`).

It differs from the brief's proposal: +4 s and an end-distance of 1 m, instead of +2 s and a pick-speed threshold. I recommend keeping the existing one, because it already has a 0.1.8b value and history across checkpoints.

| measure | n | step 0 (= 0.1.7 weights) | **0.1.8b-p06** | spread across the 7 checks of the same run (p0.0–p1.2) | source |
|---|---|---|---|---|---|
| go rate where the human pulled away | **106** | 0.764 (81/106) | **0.840 (89/106)** | 0.566 – 0.840 | `NASG/018b/runs/018b_attempt2/run.json` `checks[].safety.go_rate_pull_away`; `G/s9_018b_train/REPORT.md:4` |
| waits correctly (human stayed) | 58 | 0.138 | **0.310 (18/58)** | 0.138 – 0.769 | same, `waits_correctly` |
| pull-away moments where best safe go = best stay (exact grade tie) | 106 | 66 | 66 | 65 – 68 | same, `pull_away_go_stay_grade_ties` |
| old all-standstill go rate (reference only) | 164 | 0.799 | 0.787 | 0.488 – 0.799 | `s9_018b_train/REPORT.md:78-84` |

Spread:
- Binomial SE at p06: √(0.84·0.16/106) ≈ **0.036**.
- Checkpoint-to-checkpoint swing within one run: **0.27**. The grader ties go and stay on about 2/3 of these moments (66/106), so the measure swings easily.
- Approach-3 build 5 at step 930: 0.755 (`A3/reports/BUILD5_RECALL_LESSON.md:79`).

**The sample (106) is fixed and on disk. No new model runs are needed for any already-trained checkpoint.** **Confidence:** certain.

---

## 3. Diagnostics

### 3.1 Pick grade 0.7685: source, split, definition

| item | value | evidence |
|---|---|---|
| value | **0.76849** (p0.6, step 2,790) | `run.json` `checks[pass=0.6].safety.pick_grade_mean`; `s9_018b_train/REPORT.md:4, 81` |
| definition | mean 0.1.7 grade (0.1.7's offline simulator-style score of a plan) of the scorer's own pick (argmax of the last logit; no previous plan, so no comfort re-score; acceleration zeroed; fp32, batch 1) on the **1,131 graded ("map") moments** | `s9_018b_train/REPORT.md:72-74`; `G/s7_018b_recon/REPORT.md:239`; `G/s1_build/REPORT.md:134` |
| split | the **picking set**: 2,000 moments / 84 clips / 46 whole chunks of NVIDIA **val_full**, 0 chunks shared with train or the held-out slice; sha `6d748db2…f8c1` | `G/s1_build/REPORT.md:28, 80-98` (`NASG/picking/picking_set.parquet`) |
| reference points | step 0 (0.1.7 weights) 0.7694; p06 0.7685; best of the run p1.0 0.7746 (failed the go gate); best plan on the menu 0.9043 (headroom 0.136) | `s9_018b_train/REPORT.md:78-84` |
| spread | 0.7576 – 0.7746 across the 7 checks of one run (range 0.017) | same |
| approach-3 values on the same 1,131 | build 5 carry 0.7313 / reset 0.7355; build 4 0.7496 / 0.7520; pilot 0.7526 / 0.7563 | `A3/reports/BUILD5_RECALL_LESSON.md:60, 78` |

**Caveat:** p06 was *chosen* on this same set, so 0.7685 is optimistic for p06 as a baseline (selection bias). val_full is also the split S4's pass 9 was chosen on (`G/s0_recon/REPORT.md:297`). Using it as a kill switch is fine; reporting on it as a headline is not.

**Confidence:** certain.

### 3.2 Hard-brake picks: existing definition and sample

| item | value | evidence |
|---|---|---|
| definition | a pick whose first step brakes harder than −6 m/s² while the car is above 10 m/s. First step = (\|P[0, :2]\| / 0.5 − v) / 0.5, where v is the picking-set speed | `train_018b.py:155-158` (`safety`); `G/s8_018b_build/src/pai_model_b.py:169-170`; s7 `pick_analyze.py:153-155` |
| sample | **745** picking-set moments above 10 m/s (map or not) | `G/s8_018b_build/REPORT.md:230, 316` |
| step 0 (0.1.7 weights) | **43 / 745** | `s9_018b_train/REPORT.md:78` |
| **0.1.8b-p06** | **0 / 745** (hard plans on the menus: 1,593) | `run.json` `checks[pass=0.6].safety.hard_picks_above10`; `REPORT.md:81` |
| spread | 0 on every trained check (p0.2–p1.2) | `REPORT.md:79-84` |
| closed-loop counterpart | driven plans demanding > 6 m/s²: 0.1.8 0 / 408, 0.1.8-rs 5 / 408 on the 12 | `G/s5_diagnosis/REPORT.md:192` |

Why it matters: a demanded deceleration above about 6 m/s² makes AlpaSim's controller swerve (memory note on the MPC study; `s5_diagnosis/REPORT.md:188-195`). **Confidence:** certain.

---

## 4. Summary tables

**Gate (closed loop, 12 scenes, native driver):**

| baseline | windowed mean | full-drive mean | at-fault crashes | band |
|---|---|---|---|---|
| 0.1.8b-p06 | 0.5919 | 0.6150 | 8023dce5, bd07fb91, 04749bb9 | repeats bit-identical (4/4); one scene flip ≈ 0.083 |
| 0.1.7 | 0.6742 | 0.6130 | bd07fb91 (windowed); + 04749bb9 under full drive | — |
| build 5 carry / reset | 0.7213 / 0.5177 | 0.6623 / 0.5033 | carry 8023dce5, bd07fb91; reset + 04749bb9 | one rollout each |
| collapse floor | stall = 0.0018 – 0.051 per scene | | | six real runs |

**Kill switch and diagnostics (picking set, 1,131 graded moments of 2,000 / val_full):**

| measure | p06 | step 0 | spread (one run, 7 checks) | n |
|---|---|---|---|---|
| go rate, human pulled away | 0.840 | 0.764 | 0.566 – 0.840 (SE ≈ 0.036) | 106 |
| waits correctly | 0.310 | 0.138 | 0.138 – 0.769 | 58 |
| pick grade | 0.7685 | 0.7694 | 0.7576 – 0.7746 | 1,131 |
| hard-brake picks > 10 m/s | 0 | 43 | 0 – 0 (trained) | 745 |

---

## Numbers still missing

1. **A run-to-run band with meaning.** Plain repeats are deterministic (0.000 on 4/4). No perturbed-input repeat of p06 exists, and whether AlpaSim's `random_seed` changes anything is unknown. 8 of the 12 scenes were never repeated at all.
2. **Which score the gate uses**, windowed or full drive. They rank p06 against 0.1.7 differently on these 12 (−0.082 vs +0.002).
3. **The written 12-scene selection rule.** It is reconstructed here ("0.1.8 ≠ 0.1.7 on the 30") but not recorded as such in s5.
4. **The brief's proposed 2 s pull-away variant** (stopped at t, human moving by +2 s, pick speed at +2 s above a threshold) was not computed. The existing +4 s / 1 m definition covers the same idea and has values. The per-check dumps on the NAS (`NASG/018b/runs/018b_attempt2/`) could give the variant without a model run, if wanted.
5. **Kill-switch values on an unselected split.** The picking set is where p06 was chosen; no held-out graded set has p06 values (the 300-clip held-out slice is raw and ungraded, `APPROACH3_RECON.md`).
6. **0.1.7 on the rebuilt image.** CL12 could not re-check it (its runner's path check failed after the 2026-11-09 move, `CL12_MEMORY_PREREG.md:74`). Its 12-scene numbers are from the older image.
