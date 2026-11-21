Written: 2026-11-17T05:01:49+0530
Gallops commit: d84fe74223c24c0bd9fb7bb27f9239adbfe10c1e (branch train/drivejepa-physicalai-retrain)

# Pre-registration: approach 4 (motion history)

Abbreviations as in `reports/READONLY_CARSTATE_HISTORY.md` and `reports/BASELINES_FOR_PREREG.md`: `MH` = `Gallops/Gallops-sid/memory/4_motion_history/`, `NASG` = `{retrain_store}/`.

## 1. Design (fixed; any change requires a new pre-registration)

- **Start:** the 0.1.8b-p06 weights (`NASG/018b/release/drive_jepa_018b_p06.ckpt`, sha256 `3d566eb211b5ce0f556680196a46fc0a74da29091ae62ac588910a48d62b8e66`), encoder frozen, cached tokens reused.
- **History:**
  - 8 past rows at t−0.5 … t−4.0 s, oldest first, each `[x, y, heading, speed, turn_rate, valid]`, built from poses only.
  - Built by `MH/src/ego_history.py`, sha256 `01a96fdb77a033e9c36715018b619a98b1978cc368df0ae712dd7627c2a472fe`.
  - Side file: the one built on 2026-11-17, 4,015 files (one per training clip, 126,981 moments). It is at `{staging}/ego_history/`; the NAS target `{nas_store}/memory/4_motion_history/ego_history/` was not writable at build time (`MH/reports/BUILD1_HISTORY_DATA.md`). A copy is the same side file only if every file's bytes are identical.
- **Motion encoder:** history (8 × 6) → 256. It enters through two zero-initialised links:
  - added to the generator's 256 plan starting points;
  - concatenated to each plan's 256 feature in the rating head (input 256 → 512; 6 outputs per plan unchanged).
  
  Step 0 equals p06.
- **Hiding:** in training, the 256 summary is replaced by a learned "no history" vector with probability 0.5 per moment (arm A) or always (arm B). In the car, arm A always sees history; arm B always uses the "no history" vector, as in its training.
- **Trained:** the motion encoder, generator and scorer. Lessons are 0.1.8b's, unchanged.
- **Data:** all moments train, including those with partial history. The valid flag is the only signal of missing rows.
- **Arms A and B** use identical code, data, schedule, seed and checkpoint selection rule. They differ only in the hiding probability.
- **Checkpoint selection rule** (both arms): 0.1.8b's, unchanged (`LORA Refiner/Generator-Joint-PAI-alpasim/s9_018b_train/src/train_018b.py`, recorded in its `REPORT.md`):
  - checks every 930 steps;
  - the chosen check is the one with the highest pick grade on the 1,131 graded picking-set moments, among checks with hard-brake picks above 10 m/s ≤ step 0's and pull-away go rate ≥ 0.66;
  - same stopping rules (at most 3 passes, the 6-check no-improvement stop, the 2-in-a-row hard-gate stop).
  
  Closed-loop results are never an input.

## 2. The gate (the only pass/fail)

- **Set:** all 30 AlpaSim scenes (`Paddock/paddock-sid/Alpasim/scene_sets/`, the 30-scene set of the p06 baseline runs below).
- **Driver:** native (adapters route).
- **Score:** the windowed score, `rollouts[0].score` of each run's `aggregate/results-summary.json`, mean over 30.

**Image precondition.**
- Before either arm drives, p06 is re-run on these 8 scenes on the rebuilt image: **e3ed554f, 8023dce5, d2715023, 04749bb9, 791d68ce, 026d6a39, 02e075b9, 89f2eeb8**.
- The last 4 come from the 18-scene set. They are its lowest, its two median and its highest p06 scores, taken as positions 1, 9, 10 and 18 after sorting the 18 by p06 score ascending, ties broken by scene id. Ties at 0.0 and 1.0 mean 3 of the 4 score 1.000 (scores in the table below).
- **Pass** = the per-tick metric rows are bit-identical to the recorded runs on every scene.
- **If any scene differs:** p06 is re-run on all 30 on that image, and that mean replaces the baseline below. The margin is unchanged.
- Both arms run only on an image that passed. Its id and renderer digest are recorded with the results.

**Baseline:** p06 windowed mean over 30 = **0.7527** (12: 0.5919; 18: 0.8599). Checkpoint `3d566eb2…8e66`. The runs are on two image builds, as recorded in `reports/BASELINES_FOR_PREREG.md`:
- the 12 on 2026-10-31, on the image lost on 2026-11-03, id not recorded;
- the 18 on 2026-11-14, on `alpasim-base:0.134.0` = `sha256:9c5909dafcd5574dca1c783e6abc29dfbdfcc9fc2860725dd80030475b8f8881`;
- renderer `nvcr.io/nvidia/nre/nre-ga:26.04`.

**PASS iff mean(A) ≥ mean(p06) + 0.0667 AND mean(A) > mean(B).** 0.0667 = two scene flips on 30 scenes. Anything else is FAIL. With the baseline above, the line is mean(A) ≥ 0.8194.

Both arms are driven on all 30. The 12 diagnosis scenes run first as a preview; the preview is recorded, never judged.

**Per-scene baseline (p06, windowed `rollouts[0].score`, checkpoint sha256 `3d566eb2…8e66` in every run log):**

| # | scene | set | p06 score | run folder |
|---|---|---|---|---|
| 1 | e3ed554f | 12 | 1.000000 | `NASG/018b/closed_loop/runs/drivejepa_018bp06_e3ed554f` |
| 2 | 8023dce5 | 12 | 0.000000 | `…/drivejepa_018bp06_8023dce5` |
| 3 | 5f064b02 | 12 | 1.000000 | `…/drivejepa_018bp06_5f064b02` |
| 4 | e121e37d | 12 | 1.000000 | `…/drivejepa_018bp06_e121e37d` |
| 5 | 048b974e | 12 | 1.000000 | `…/drivejepa_018bp06_048b974e` |
| 6 | cdd9448c | 12 | 1.000000 | `…/drivejepa_018bp06_cdd9448c` |
| 7 | d2715023 | 12 | 1.000000 | `…/drivejepa_018bp06_d2715023` |
| 8 | bd07fb91 | 12 | 0.000000 | `…/drivejepa_018bp06_bd07fb91` |
| 9 | bae463d3 | 12 | 0.379891 | `…/drivejepa_018bp06_bae463d3` |
| 10 | d1a25a99 | 12 | 0.000000 | `…/drivejepa_018bp06_d1a25a99` |
| 11 | b1b170d0 | 12 | 0.723395 | `…/drivejepa_018bp06_b1b170d0` |
| 12 | 04749bb9 | 12 | 0.000000 | `…/drivejepa_018bp06_04749bb9` |
| 13 | 01d503d4 | 18 | 1.000000 | `{experiment_store_h17}/runs/cl18/018bp06/runs/drivejepa_018bp06_01d503d4` |
| 14 | 023b7fcc | 18 | 1.000000 | `…/drivejepa_018bp06_023b7fcc` |
| 15 | 0245ff75 | 18 | 1.000000 | `…/drivejepa_018bp06_0245ff75` |
| 16 | 026d6a39 | 18 | 1.000000 | `…/drivejepa_018bp06_026d6a39` |
| 17 | 02e075b9 | 18 | 1.000000 | `…/drivejepa_018bp06_02e075b9` |
| 18 | 02eadd92 | 18 | 1.000000 | `…/drivejepa_018bp06_02eadd92` |
| 19 | 032b6f21 | 18 | 1.000000 | `…/drivejepa_018bp06_032b6f21` |
| 20 | 04394343 | 18 | 1.000000 | `…/drivejepa_018bp06_04394343` |
| 21 | 16b83346 | 18 | 1.000000 | `…/drivejepa_018bp06_16b83346` |
| 22 | 1acf2f5d | 18 | 0.834999 | `…/drivejepa_018bp06_1acf2f5d` |
| 23 | 2e1ce314 | 18 | 1.000000 | `…/drivejepa_018bp06_2e1ce314` |
| 24 | 6725df2d | 18 | 1.000000 | `…/drivejepa_018bp06_6725df2d` |
| 25 | 71d7d03b | 18 | 1.000000 | `…/drivejepa_018bp06_71d7d03b` |
| 26 | 791d68ce | 18 | 0.000000 | `…/drivejepa_018bp06_791d68ce` |
| 27 | 89f2eeb8 | 18 | 1.000000 | `…/drivejepa_018bp06_89f2eeb8` |
| 28 | 98dd56b2 | 18 | 0.791024 | `…/drivejepa_018bp06_98dd56b2` |
| 29 | acd22430 | 18 | 0.000000 | `…/drivejepa_018bp06_acd22430` |
| 30 | b44a4a44 | 18 | 0.851624 | `…/drivejepa_018bp06_b44a4a44` |
| | **mean** | 12: 0.5919 · 18: 0.8599 · **30: 0.7527** | | |

## 3. Kill switch (stops a run before simulation; not a verdict)

If `go_rate_pull_away` on the fixed 106 moments is **< 0.66** for an arm, that arm does not go to the simulator; it is diagnosed instead.

p06 reference: **0.8396** (89/106), from `NASG/018b/runs/018b_attempt2/run.json`, `checks[pass=0.6].safety.go_rate_pull_away`.

## 4. Diagnostics (recorded for both arms, no thresholds)

| measure | n | p06 |
|---|---|---|
| `pick_grade_mean` | 1,131 | 0.7685 (0.768489) |
| `hard_picks_above10` | 745 | 0 |
| `waits_correctly` | 58 | 0.3103 (18/58) |

Source: `NASG/018b/runs/018b_attempt2/run.json`, `checks[pass=0.6].safety`. The picking set is where p06 was selected; these numbers are not headlines.

## 5. Not allowed after this file is written

- Changing the scene set, the score, the margin, the hiding rate or the checkpoint selection rule.
- Choosing a checkpoint by closed-loop results.
- Re-running a scene and keeping the better result.

## Amendment 1, 2026-11-17T07:10:26+05:30

Training runs on the frozen encoder's cached tokens without GridMask, as approach 3 did; hide-and-guess is dropped because the encoder is frozen. Arm B measures the effect of this training path on its own. Gate, kill switch, margin and scene set unchanged.
