# S6 — Scorer diagnosis (checks a, b, c)

**STATUS 2026-10-08 22:45 IST: checks a, b and c are complete on UNSEEN and SEEN (11,842 examples each). One item is open: the
16-bit precision rerun on the GPU, deferred until the GPU is free (flag 14).** The SEEN tables and part b were delayed from 09:32 to
22:08 because the session that was to start them had ended (flag 19).

Paths below are relative to `paddock-sid/Alpasim/` unless they start with `/`. `AGENT` =
`adapters/drivejepa/src/navsim_v2/navsim/agents/drive_jepa_perception_based`, `RD` = `analysis_3.0/Retrain-data`.

## 1. Summary

Numbers are pass 9 on its own menu, 11,842 graded examples from 401 unseen
clips, 95 % clip-bootstrap intervals in brackets (table 4.1).

- **The scorer is better than a random pick at avoiding bad plans, and little better at finding the best one.** Mean regret 0.115
  (0.105–0.125) against 0.210 for a random pick. Pick = true best on 38.3 % (35.6–41.3) against 33.2 % for a random pick.
- **Picking the plan closest to the human beats the scorer on every grade ruler**: true best 50.9 %, graded ≥ 0.9 on 52.7 %, regret
  0.079. That baseline needs the human path, so it is a yardstick, not a deployable rule.
- **Ruler: S4's 15 % understates the pick, but the lenient ruler does not rescue it.** Pick within 1.5 m of the human 15.6 %; pick
  graded ≥ 0.9 49.5 %; gap 33.9 points (30.4–36.9). On close-pass the pick is graded ≥ 0.9 on 33.5 % (26.1–41.1), and only 50 % of
  close-pass menus contain a plan graded ≥ 0.9 at all.
- **The menu caps the result**: a plan graded ≥ 0.9 exists on 66.8 % of examples (58 % highway, 50 % close-pass, 75 % ordinary).
- **Passes 1 → 9: the strict rate fell (44.7 → 38.3 %) while the lenient rate rose (44.1 → 49.5 %)**; regret is the same (0.108,
  0.115). Rank agreement fell (median 0.52 → 0.39).
- **Highway is where the scorer fails outright**: pass 9 picks the true best on 11.9 % against 15.8 % for a random pick, and rank
  agreement is 0.06.
- **The top two scores are nearly tied on 84 % of examples, but the two plans usually grade the same**: they differ by more than 0.1
  in true grade on 11.5 % of those (21.8 % on highway). So the coin toss at the top costs little on most examples. P-a5 fails on
  its second half.
- **Menu slots are not interchangeable** (flag 17): some slots are fast plans that are the true best about half the time but grade
  poorly on average; the scorer mostly picks slow, safe slots. This is from a quarter of the shards and is descriptive only.
- **Seen vs unseen clips: the scorer is somewhat better on clips it trained on, not transformed.** SEEN against UNSEEN: regret 0.078
  (0.072–0.084) against 0.115 (0.105–0.125); pick graded ≥ 0.9 53.3 % against 49.5 %; true best 40.8 % against 38.3 %; rank agreement
  median 0.48 against 0.39. Highway is as bad on SEEN (true best 12.0 %) as on UNSEEN (11.9 %): that failure is not about unseen streets.
- **Part b: the features hold more than the scorer uses.** A plain linear probe on the 256 numbers the scorer reads separates
  passing from failing plans on UNSEEN at 0.83–0.89 for collision, drivable area, driving direction and time-to-collision, 0.95 for
  comfort and 0.85 for progress. The scorer's trained total output manages 0.67 on comfort and 0.56 on progress (table 4.7).
  Lane keeping is the one check that is hard to read at all: probe 0.73, scorer 0.63.
- **The map-dependent checks are readable from the camera features**, on unseen streets too (probe 0.88–0.89 for drivable area and
  driving direction). "The model never sees the map, so it cannot know" is not supported.
- **The per-check outputs are dead weight after retraining**: in pass 9 only time-to-collision's own output separates anything
  (0.75); the shipped model's comfort output did (0.80) and pass 9's does not (0.47). They carry zero training weight.
- **The generator leaves good plans off the menu, most of all on highway** (table 4.9): a catalogue curve graded ≥ 0.95 exists on
  93.6 % of highway examples; a menu plan graded ≥ 0.95 on 56.5 %. Over all UNSEEN examples it is 83.7 % against 65.6 %.
- The brief's premise that the scorer is "trained on catalogue curves" is wrong: it is trained on, and only ever scores, the
  generator's 32 menu plans, with true EPDMS calculator grades (2.2).
- Fresh 32-bit CPU runs reproduce S4's rates within 0.5 points, but pick a different plan than S4's 16-bit run on about half of the
  examples (2.5).

## 2. Part 0

### 2.1 Checkpoints (sha256)

| file | sha256 | note |
|---|---|---|
| `{runs}/drivejepa_retrain/06_first_training_run/candidate/drive_jepa_s4_pass09.ckpt` | `8a71c3c0d84a9758de46c7b46eaf18556a62ac4837feea7271040e7a0caacdd2` | matches the expected value |
| `{runs}/drivejepa_retrain/06_first_training_run/run/ckpt/pass_01.pt` | `f68b975d6a8291638511b466faab97bcdfac213166eecec748e4f27529e0c0bb` | trained tensors + optimiser, 85 MB |
| `{runs}/drivejepa_retrain/06_first_training_run/run/ckpt/pass_09.pt` | `b3130f1f2155f5a9a5c06191831dfba0e0c07a9841689b39d386c12061328a5a` | source of the candidate's trained tensors |
| `{runs}/alpasim_bulk/navsim-exp/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt` | `740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05` | shipped; matches `RD/05_small_overfit_check/train_patch/s35_common.py:15` |

### 2.2 The two confirmations Sid asked for

**(1) The grades the scorer trains on come from the EPDMS calculator run on the 32 menu plans during training. Confirmed. There is no
nearest-catalogue lookup anywhere in the scorer's loss.**

- `RD/05_small_overfit_check/train_patch/drive_jepa_agent_s35.py:357` takes `proposals = pred["proposals"]` (the 32 menu plans of each example).
- `:368-370` calls `self.compute_score(targets, proposals.detach(), test=False)` on the graded rows.
- `compute_score` (`:179-216`) sends each example's 32 plans (`"poses": poses`, `:194`, from `proposals.float().cpu().numpy()`, `:199`)
  to `get_sub_score_s35` (`:27-59`), which loads that example's metric cache (`:32`) and runs `s3_scorer.score(metric_cache, poses)`
  (`:52`), i.e. the EPDMS calculator with departures D2 and D3 (`RD/04_training_set/s3_scorer.py:28-43`).
- The result `target_scores` (32 plans × 9 columns, `:204`) is the scorer's training target: `:389-397` passes it to `score_loss`,
  where `final_score_loss = BCE(pred_logit[..., -1], target_scores[..., -1])` (`:248`).
- The catalogue enters only the generator's loss: `scores_index` (the stored list of good catalogue curves) is used by
  `trajectory_loss_anchors` (`:306-354`, called at `:377`) as extra targets for the plans. It never reaches `score_loss`.
- The vendored original does the same (`AGENT/drive_jepa_agent.py:276-278` compute_score on the proposals, `:289-298` score_loss).

**(2) How S4's 93 % and 15 % got menu grades: they did not use grades at all. They are distances to the human path.**

- `RD/06_first_training_run/s4_train.py:116` computes, per example, `d4` = distance at 4 s between each of the 32 plans and the human path;
  `:117` stores `dist4` (the scorer's pick) and `best_dist4` (the closest of the 32).
- `RD/06_first_training_run/s4_report.py:44` reports "pick within 1.5 m" = `dist4 <= 1.5` and "closest of 32 within 1.5 m" = `best_dist4 <= 1.5` on
  close-pass examples; `:50` sets V3 from the first.
- Where S4 did use menu grades (V2, "pick = true best", rank agreement), they are true calculator grades of the 32 plans:
  `s4_train.py:112` `agent.compute_score(t, pred["proposals"], test=False)[0]`, used at `:120`. No nearest-catalogue grade is used.

**(3) Loss form: neither of the two. It is a cross-entropy on the logit with the continuous grade itself as the target. There is no
threshold, and it is not squared or absolute error.**

- The loss object is `nn.BCEWithLogitsLoss()` with default settings (`RD/05_small_overfit_check/train_patch/drive_jepa_agent_s35.py:99`; vendored
  `AGENT/drive_jepa_agent.py:41`).
- It is applied as `self.bce_logit_loss(pred_logit[..., -1], target_scores[..., -1])` (`drive_jepa_agent_s35.py:248`; vendored `:163`).
  `target_scores[..., -1]` is the calculator's total grade, a number between 0 and 1, passed in as it is: no `>=`, no rounding, no
  threshold anywhere between `compute_score` (`:204-205`) and the loss (`:389-397`, `:248`).
- So each plan's target is "soft": the loss is lowest when sigmoid(output) equals the grade. It behaves like a regression onto the
  grade, measured with cross-entropy instead of squared error, averaged over graded examples × 32 plans with equal weight per plan.
- The only threshold in the pipeline is 0.95, and it belongs to the generator: catalogue curves graded ≥ 0.95 form the list of extra
  generator targets (`RD/04_training_set/s3_grade.py:64-68`). It does not touch the scorer loss.
- The pick at inference is the plan with the highest sigmoid(output) (`AGENT/drive_jepa_model.py:85, 90`), so only the ordering of the
  32 outputs matters for the pick, while the loss rewards matching each grade's value.

**Note on the brief.** The brief's premise "how [the scorer] is trained on catalogue curves" was wrong: the scorer is trained on the
generator's menu plans with true grades. As a result P-a1 and P-c1 were dropped and P-c1' added (section 3), and the catalogue parts
(a.1, the two catalogue comparison sets of c.2) are not measurable and are not run (Sid, 2026-10-07).

### 2.3 Scorer mechanics

| question | answer | where |
|---|---|---|
| Form of the scorer loss | Cross-entropy on a logit with the calculator's grade in [0, 1] as a soft target; no threshold, not squared error (detail in 2.2 (3)). | `drive_jepa_agent_s35.py:99, 245-248` |
| How many outputs | 6 per plan, not 9. They are compared with the last 6 calculator columns: driving direction, progress, time-to-collision, lane keeping, comfort, total. No output for collision, drivable area, traffic light. | `AGENT/score_module/scorer.py:13, 15, 42`; `drive_jepa_agent_s35.py:245-246`; column order `AGENT/score_module/compute_navsim_score.py:90-92` |
| Which outputs are trained | Only the total (weight 1). The five per-check outputs have weight 0. | `AGENT/drive_jepa_config.py:51-52`; `drive_jepa_agent_s35.py:417-418` |
| Other scorer-side losses | Predict the corners of the actor a plan collides with (weight 1 yes/no + 0.1 distance) and which area the car is in (weight 2). Graded rows only. | `drive_jepa_config.py:53-55`; `drive_jepa_agent_s35.py:218-243, 419-421` |
| Weight against the generator loss | Plain sum: 1 × generator + 1 × total-grade loss + the three above. | `drive_jepa_agent_s35.py:415-425`; `drive_jepa_config.py:49` |
| What the scorer scores at inference | The 32 menu plans, one feature per plan: the trunk's 32 × 8 slot features, max over the 8 poses, through a 3-layer network. Pick = highest sigmoid of the last output. | `scorer.py:41-42`; `AGENT/drive_jepa_model.py:71-73, 85, 90`; network `AGENT/bevformer/transformer_decoder.py:65-78` |
| Fixed slots or any path? | Fixed slots. The features start from 256 learned slot embeddings (32 plans × 8 poses) plus the ego-status encoding, and are refined 4 times by looking at the image where the current plan lies. The scorer takes no path as input, so it cannot score a catalogue curve or any outside path. Menu plans are **not** mapped to catalogue curves. | `drive_jepa_model.py:33, 61-68`; `AGENT/traj_refiner.py:23-29`; `AGENT/bevformer/bev_refiner.py:107-140` |
| Scorer and generator are not separate networks | The scorer's input features are produced by the generator trunk of the same checkpoint. "Scorer" in the tables means a whole checkpoint on its own menu (primary) or its score head moved onto the pass-9 trunk (secondary, off-design). | `drive_jepa_model.py:65-73` |
| Catalogue path | Vendored code opens `./data/8192.npy` (`AGENT/drive_jepa_agent.py:49`, `scripts/misc/calc_anchors_scores.py:15`). The patched loader serves `RD/04_training_set/vocab_10240.npy` (`s35_common.py:16`, checked `s4_train.py:38-39`; sha256 `4b0bc3c9…39e785` per `RD/06_first_training_run/settings_manifest.json`). | as cited |
| Stored grades | Catalogue curves: **total only**, 16-bit, one file per graded example (`anchors_scores_final.npy`), with gaps for curves the pre-filter skipped; plus the index of curves graded ≥ 0.95 (at most 256). All 9 columns are kept only for the human path and the stay-put plan (`pad_label2.gz`). Menu-plan grades are not stored: training and S4's evaluation computed them on the fly. All under `{runs}/drivejepa_retrain/04_training_set/s1/cache/<clip>/<example>/`. | `RD/04_training_set/s3_grade.py:60-71` |
| Checks computable without a map | `s3_scorer.py` does not say; there is no map-free mode in it. From the calculator code: only comfort uses no map. Collision and time-to-collision use actors, with the map deciding fault (multiple-lanes / off-road flags). Drivable area, driving direction, progress (measured along the mapped centre line), lane keeping and traffic light need the map. | `AGENT/score_module/train_pdm_scorer.py:404-449 (collision, map at :436-438), :484-489, :497-511, :548-561, :566-668 (map at :648-652, :668), :686-715, :723-749, :769` |
| There are 8 checks plus the total, not 9 checks | Columns: NC, DAC, TLC, DDC, EP, TTC, LK, HC, score. | `s3_scorer.py:16` |
| Plan format, generator vs calculator | Generator: 32 plans × 8 poses (x, y, heading), 0.5 s apart to 4 s, in the car's frame at the example's moment. Calculator wrapper takes exactly that (`(N, 8, 3)`). Inside, the vendored code interpolates each plan to 40 poses at 0.1 s and runs it through its vehicle simulator. No resampling on our side; the catalogue (40 poses) is cut to the same 8 by `[:, 4::5]`. | `drive_jepa_config.py:27, 36, 62`; `s3_scorer.py:29, 31`; `compute_navsim_score.py:18, 41-49`; `navsim/evaluate/pdm_score.py:71-99`; `drive_jepa_agent_s35.py:111` |
| Model input (0.1.6) | Acceleration zeroed by the loader (`dataset_s35.py:258`, `zero_accel=True` at `s4_train.py:43`). The bonnet mask (37 rows) is already baked into the stored inputs by the converter (`RD/02_build_samples/s1_convert.py:29, 128`; asserted `RD/04_training_set/s3_convert.py:11, 30`). | as cited |
| Momentum-aware selection (0.1.4b) | Lives in `adapters/drivejepa014b/momentum.py:216-221` (pick with and without), which feeds the previous plan to the vendored re-score `drive_jepa_model.py:87-88, 98-151` (score ← (14 × score + 2 × comfort) / 16). It needs the previous tick's chosen plan. Examples in a clip are 0.5 s apart on 99.2 % of steps, so the previous example's pick can stand in for it. | as cited |

### 2.4 Evaluation sets (`s6_sets.py`, seed 20261007; `sets_table.csv`)

Slices follow the manifest: highway (`A`), close-pass (`B_pass`), pull-out (`B_raw`), ordinary. **No pull-out example is graded in
either set** (pull-outs come from clips without a map), so the brief's "pull-out (B)" slice is empty; the graded B slice is close-pass.

| set | slice | examples | clips |
|---|---|---|---|
| UNSEEN (graded, `val_full`) | highway | 3,315 | 110 |
| | close-pass | 1,563 | 57 |
| | ordinary | 6,964 | 234 |
| | all | 11,842 | 401 |
| SEEN (sample of graded training examples) | highway | 3,475 | 205 |
| | close-pass | 1,347 | 88 |
| | ordinary | 7,020 | 434 |
| | all | 11,842 | 727 |

- SEEN is drawn from the 21,806 graded training examples (727 clips); each clip contributes in proportion to its graded examples.
- The small set `val/manifest_val.parquet` has 225 graded examples in 8 clips; it is not used (val_full is the larger set).
- Exam overlap: 0 clips dropped from either set. Exam list = `public_2601` in `alpasim/data/scenes/sim_suites.csv` ∪ our 30 trial
  scenes (`RD/01_find_clips/exam_exclusions_strict_our30.txt`) = 913 ids, identical to `RD/01_find_clips/exam_exclusions.txt`.
- For information: every clip of both sets is listed in `sim_suites.csv` under `public_2604` (the mapped release the training data comes
  from). That suite is not on the exam list.
- No clip and no chunk is shared between UNSEEN and the training manifest.

### 2.5 Reproduction of S4 on UNSEEN

**Stored half (S4's own per-example readouts, `RD/06_first_training_run/run/readouts/val_full_pass*.parquet`; `s6_repro_stored.py`, `repro_stored.csv`).**
Code path S4 used: `s4_train.py:94-125` (`evaluate`), tables by `s4_report.py:38-53`.

| number | S4 report | recomputed | difference |
|---|---|---|---|
| pass 9, pick = true best (V2) | 37.9 % | 37.86 % of 11,842 | 0.0 |
| pass 9, highway / close-pass / ordinary | 11.1 / 42.7 / 49.5 % | 11.10 / 42.67 / 49.51 % | 0.0 |
| pass 9, rank agreement (mean) | 0.232 | 0.2322 | 0.0 |
| pass 9, pick within 1.5 m, close-pass (V3, "15 %") | 14.8 % | 14.84 % of 1,570 | 0.0 |
| pass 9, a plan within 1.5 m is on the menu ("93 %") | 93.5 % | 93.50 % | 0.0 |
| shipped, pick = true best ("34 %") | 34.2 % | 34.17 % | 0.0 |
| pass 1, pick = true best | 44.1 % | 44.14 % | 0.0 |

**Fresh half (CPU forward passes in our wrapper, tolerance ±3 points): PASSES, all 11,842 UNSEEN examples, 401 clips**
(`s6_repro_fresh.py`, `repro_fresh_11842.csv`). The gate for the a, b, c tables is open.

| number, same 11,842 examples | S4 stored (GPU, 16-bit) | S6 fresh (CPU, 32-bit) | difference |
|---|---|---|---|
| shipped, pick = true best | 34.2 % | 34.0 % | −0.2 |
| pass 1, pick = true best | 44.1 % | 44.7 % | +0.5 |
| pass 9, pick = true best | 37.9 % | 38.3 % | +0.4 |
| pass 9, close-pass pick within 1.5 m (1,563 examples) | 14.9 % | 16.5 % | +1.6 |
| pass 9, close-pass plan within 1.5 m on the menu | 93.7 % | 94.8 % | +1.1 |
| pass 9, rank agreement, mean (S4 breaks ties by slot order; S6 with S4's definition; S6 with average ranks) | 0.232 | 0.261 | 0.292 with average ranks |

**The aggregates reproduce; the individual picks do not.** The "pick = true best" verdict agrees per example on 95.6 % (shipped),
90.1 % (pass 1) and 92.9 % (pass 9), but the grade of the pick is within 0.001 of S4's on only 57 %, 52 % and 51 %: on about half of
the examples a different plan wins. The top two scores are a median 0.003 apart on a 0–1 scale against a spread of about 0.29 over
the 32 plans. S4 evaluated in 16-bit mixed precision on the GPU (`RD/06_first_training_run/s4_train.py:82`); S6 runs the same weights in 32-bit on
the CPU, and its split forward equals the model's own 32-bit forward exactly (section 7). That number-format rounding flips the winner
is an inference, not isolated (flag 14). The first-1,024 interim tables are kept in `repro_fresh_1024.csv`, `repro_detail_1024.csv`.

## 3. Predictions

- `PREDICTIONS.md` (version 1): sha256 `8791c38f36df15ed2852b9efbcac174b826023b071d6bc3d3d24d646c9d51c1b`, hashed 2026-10-07T22:33:23+05:30.
- `PREDICTIONS_v2.md` (in force): sha256 `c3f4057afbb1c3a4e82a2758be402f9acf44e704fcfb4f2a2342a0732262ab60`, hashed 2026-10-07T22:58:57+05:30.
- Dated note, 2026-10-07: Sid asked for P-a1 and P-c1 to be dropped and P-c1' added "since PREDICTIONS.md is not hashed yet". It had
  been hashed 25 minutes earlier. Nothing from a, b, c had run, so the change was made as a second file; version 1 is untouched.
  Version 2 states what I had already seen from S4's stored readouts when it was written: the UNSEEN side of P-c1' (0.232) and P-x2
  are not blind. No SEEN number had been looked at.

- `PREDICTIONS_v3.md` (adds P-a4; version 2 stays in force unchanged): sha256 `8fcdbd7107a2d507b46589d8b5be0d31fcfa33df8d55a8be8258ee483b984b52`,
  hashed 2026-10-07T23:26:31+05:30, before the gate opened and before any model run. It discloses that P-a4 is partly not blind: the
  all-UNSEEN "graded ≥ 0.9" share (48.8 %) and the close-pass 1.5 m rate (14.8 %) were already known from S4's stored readout; the
  all-UNSEEN 1.5 m rate and the per-slice shares were not.

- `PREDICTIONS_v4.md` (adds the top-2 margin measurement and P-a5; versions 2 and 3 stay in force): hashed 2026-10-08 05:36, sha256 in
  `predictions.sha256`. Half (i) of P-a5 was not blind (flag 15).

**Results so far (UNSEEN only; pass 9 on its own menu unless said).**

| prediction | what it said | measured | verdict |
|---|---|---|---|
| P-x3 | stored readout reproduces 37.9 % exactly; fresh CPU within ±3 points | exact; fresh 38.3 % (+0.4) | **holds** |
| P-a2 | close-pass pick graded ≥ 0.9: above 50 % if the ruler is the story, below 35 % otherwise | 33.5 % (26.1–41.1) | "otherwise" side at the point estimate; the interval crosses 35 %. Only 50.0 % of close-pass menus hold a plan graded ≥ 0.9 |
| P-a3 | lenient numbers for pass 1 ≥ pass 9 | graded ≥ 0.9: 44.1 vs 49.5 %; within 0.05 of best: 61.7 vs 63.0 %; regret 0.108 vs 0.115 | **fails** on graded ≥ 0.9 (pass 9 higher); the other two are within each other's intervals |
| P-a4 | graded ≥ 0.9 minus 1.5 m rate: > 30 points if the ruler is most of the story, < 10 otherwise | all 33.9 (30.4–36.9); highway 30.2; close-pass 17.0 (9.6–24.6); ordinary 39.4 | above 30 on all examples, "between the two readings" on close-pass. A random pick has a 27.8-point gap, so most of the gap is the two rulers, not the scorer |
| P-a5 (i) | > 70 % of examples have a top-2 score gap < 0.01 (not blind) | 84.0 % (82.3–85.6) | holds |
| P-a5 (ii) | in > 30 % of those the two plans differ in true grade by > 0.1 | 11.5 % (10.2–12.8) | **fails**; P-a5 as a whole fails |
| P-x2 | graded ≥ 0.9 at least 15 points above strict; random pick within 10 points of the scorer's lenient number (not blind) | 11.2 points above strict; random pick 10.9 points below | **fails** on both halves, the second narrowly |
| P-c2, second half | "a plan graded ≥ 0.95 on the menu" below 93 % | 65.6 % | holds as written; S4's 93 % was a distance rate, a different quantity |
| P-c1' | rank agreement SEEN > 0.5 and UNSEEN < 0.3 (memorised), or both < 0.3 (can't learn) | SEEN median 0.48 (0.45–0.50), mean 0.37 (0.35–0.39); UNSEEN median 0.39 (0.35–0.42), mean 0.29 (0.26–0.32) | **neither branch holds.** SEEN is below 0.5 on both statistics and above 0.3 on both. The prediction did not say mean or median. Reading: a modest seen-over-unseen gap (about 0.08), not memorisation |
| P-b1 | starved of streets: SEEN beats UNSEEN by > 0.10 in separation on map-dependent checks AND a probe trained on SEEN fails on UNSEEN | scorer total output, SEEN minus UNSEEN: collision +0.13, time-to-collision +0.13, drivable area +0.07, driving direction +0.07, lane keeping +0.07. Probe on UNSEEN: 0.73–0.89 | **fails**: the gap passes 0.10 on two of five checks, and the probe does not fail on UNSEEN |
| P-b2 | not readable from a photo: probe weak even on SEEN for map-dependent checks (< 0.70), map-free readable (> 0.80) | probe on SEEN held-out: drivable area 0.91, driving direction 0.91, lane keeping 0.77; comfort 0.94 | **fails**: every map-dependent check is above 0.70 |
| P-b3 | training loss: probe on UNSEEN beats the scorer's own head on the same check by > 0.10 | progress +0.29, comfort +0.28, lane keeping +0.10, collision +0.09, time-to-collision +0.09, drivable area +0.06, driving direction +0.03; total grade ≥ 0.9 +0.05 | **holds on progress and comfort, at the line on lane keeping, not on the rest.** Caveat: the total output is trained on the total grade, not on any one check, so on single checks the comparison flatters the probe; on the total itself the probe is +0.05 |
| P-x1 | zero-weight per-check outputs separate no better than the total output (difference < 0.05), pass 1 and pass 9 | largest "own minus total": +0.01 (time-to-collision, both passes) | **holds** |
| P-c2 first half | true grades of menu plans are lower than the stored grade of the nearest catalogue curve, especially on highway | true minus nearest-catalogue, mean per example: all +0.006 (−0.000 to +0.011); highway −0.026 (−0.041 to −0.013); close-pass +0.022; ordinary +0.017 | **fails overall; right direction but small on highway.** On highway 34 % of plans grade more than 0.1 below their nearest catalogue curve and 13 % more than 0.1 above; the nearest curve there is 3.1 m away on average, so "nearest" is loose |

## 4. Tables from c, a, b

Part a rulers, all on the same picks (Sid, 2026-10-07): (1) strict match, (2) pick graded ≥ 0.9, (3) pick within 0.05 of the best
grade, with mean and 90th-percentile regret, and (4) **pick within 1.5 m of the human path at 4 s**, S4's ruler (`s4_train.py:116-117`,
`s4_report.py:44`), so S4's 15 % sits next to the grade rulers. Also, per slice: how often the best-graded menu plan is itself within
1.5 m of the human (how far rulers 2 and 4 can disagree at all).

UNSEEN tables (`s6_analyze.py --set=unseen`; `table_a_c_unseen.csv`, `table_menu_unseen.csv`, `table_top2_margin_unseen.csv`,
`table_menu_distance_unseen.csv`; markdown built by `s6_report_tables.py`). SEEN tables follow as 4.S1–4.S5 (`s6_analyze.py --set=seen`, `table_*_seen.csv`), then part b (4.6, 4.7; `s6_partb.py`, `table_b_heads.csv`, `table_b_probes.csv`). The catalogue items of part c are tables 4.8 and 4.9 (`s6_partc_catalogue.py`).
The momentum-aware row uses an offline stand-in for the previous plan (the previous graded example of the same clip, 0.5 s earlier);
it applied on 95.9 % of examples and changed the pick on 32.5 %.

**Table 4.1 — PRIMARY: each checkpoint whole, on its own menu. UNSEEN, all 11,842 graded examples, 401 clips.** Brackets: 95 % interval, clip-level bootstrap.

| scorer | pick = true best, % | pick graded ≥ 0.9, % | pick within 0.05 of best, % | regret, mean | regret, 90th pct | pick within 1.5 m of human, % | rank agreement, median |
|---|---|---|---|---|---|---|---|
| shipped | 34.0 (31.0–37.0) | 42.1 (38.7–45.6) | 66.2 (63.7–68.6) | 0.116 (0.106–0.126) | 0.36 (0.26–0.41) | 11.6 (10.1–13.2) | 0.23 (0.19–0.27) |
| pass 1 | 44.7 (42.0–47.6) | 44.1 (40.6–47.5) | 61.7 (59.3–64.2) | 0.108 (0.099–0.119) | 0.28 (0.24–0.36) | 15.5 (13.7–17.3) | 0.52 (0.49–0.54) |
| pass 9 | 38.3 (35.6–41.3) | 49.5 (45.8–52.7) | 63.0 (60.5–65.3) | 0.115 (0.105–0.125) | 0.29 (0.23–0.40) | 15.6 (14.0–17.2) | 0.39 (0.35–0.42) |
| pass 9, momentum-aware (0.1.4b) | 38.5 (35.7–41.5) | 49.8 (46.2–53.0) | 64.5 (62.1–66.8) | 0.113 (0.103–0.124) | 0.30 (0.23–0.40) | 18.4 (16.7–20.1) | — |
| baseline — random pick (expected) | 33.2 (31.5–35.1) | 38.6 (35.8–41.2) | 49.5 (47.9–51.1) | 0.210 (0.197–0.224) | 0.58 (0.54–0.61) | 10.8 (10.3–11.4) | — |
| baseline — slot 0 (the menu has no order; first slot) | 45.6 (42.8–48.3) | 44.0 (40.5–47.0) | 63.5 (60.8–66.1) | 0.198 (0.181–0.220) | 1.00 (0.88–1.00) | 13.5 (12.0–15.0) | — |
| baseline — plan closest to the human | 50.9 (49.0–52.9) | 52.7 (49.1–56.0) | 78.5 (76.4–80.4) | 0.079 (0.071–0.089) | 0.15 (0.14–0.17) | 89.1 (87.1–91.0) | — |
| ceiling — best-graded plan (closest to human among ties) | 100.0 (100.0–100.0) | 66.8 (63.4–70.1) | 100.0 (100.0–100.0) | 0.000 (0.000–0.000) | 0.00 (0.00–0.00) | 70.6 (67.9–73.5) | — |

**Table 4.1 highway — 3,315 examples, 110 clips.**

| scorer | pick = true best, % | pick graded ≥ 0.9, % | pick within 0.05 of best, % | regret, mean | regret, 90th pct | pick within 1.5 m of human, % | rank agreement, median |
|---|---|---|---|---|---|---|---|
| shipped | 5.1 (3.2–7.1) | 16.5 (11.4–21.9) | 63.5 (58.6–68.3) | 0.108 (0.092–0.127) | 0.22 (0.20–0.38) | 0.9 (0.4–1.4) | -0.11 (-0.20–-0.03) |
| pass 1 | 15.1 (12.0–18.4) | 20.6 (15.5–25.3) | 38.3 (33.8–42.8) | 0.168 (0.145–0.189) | 0.46 (0.40–0.69) | 6.2 (4.7–7.9) | 0.08 (0.01–0.16) |
| pass 9 | 11.9 (8.9–15.0) | 35.6 (29.0–41.2) | 44.8 (39.8–49.5) | 0.149 (0.128–0.171) | 0.42 (0.25–0.55) | 5.4 (3.8–6.9) | 0.06 (-0.04–0.14) |
| pass 9, momentum-aware (0.1.4b) | 12.4 (9.3–15.6) | 36.1 (29.6–41.7) | 46.2 (41.1–51.1) | 0.148 (0.128–0.169) | 0.42 (0.25–0.55) | 6.0 (4.3–7.7) | — |
| baseline — random pick (expected) | 15.8 (13.5–18.1) | 26.9 (21.8–31.6) | 36.4 (33.4–39.3) | 0.208 (0.185–0.232) | 0.52 (0.42–0.60) | 5.7 (5.0–6.4) | — |
| baseline — slot 0 (the menu has no order; first slot) | 21.1 (16.5–25.5) | 33.4 (26.9–39.2) | 48.4 (42.7–53.7) | 0.179 (0.152–0.206) | 0.68 (0.40–0.86) | 10.2 (7.3–13.3) | — |
| baseline — plan closest to the human | 36.7 (32.8–40.5) | 34.2 (27.8–40.0) | 60.3 (55.6–64.6) | 0.118 (0.098–0.142) | 0.28 (0.17–0.66) | 72.6 (66.7–77.5) | — |
| ceiling — best-graded plan (closest to human among ties) | 100.0 (100.0–100.0) | 58.3 (51.1–64.6) | 100.0 (100.0–100.0) | 0.000 (0.000–0.000) | 0.00 (0.00–0.00) | 42.2 (36.7–47.4) | — |

**Table 4.1 close-pass — 1,563 examples, 57 clips.**

| scorer | pick = true best, % | pick graded ≥ 0.9, % | pick within 0.05 of best, % | regret, mean | regret, 90th pct | pick within 1.5 m of human, % | rank agreement, median |
|---|---|---|---|---|---|---|---|
| shipped | 41.8 (36.7–47.2) | 31.5 (24.7–38.2) | 56.0 (50.3–61.4) | 0.150 (0.126–0.176) | 0.50 (0.42–0.73) | 13.9 (10.7–17.3) | 0.31 (0.24–0.36) |
| pass 1 | 47.0 (42.2–51.8) | 31.8 (24.7–39.2) | 59.8 (55.1–64.0) | 0.114 (0.098–0.132) | 0.34 (0.26–0.43) | 15.5 (12.8–18.4) | 0.55 (0.51–0.58) |
| pass 9 | 43.5 (38.7–48.5) | 33.5 (26.1–41.1) | 60.9 (55.6–65.5) | 0.150 (0.119–0.182) | 0.62 (0.43–0.86) | 16.5 (13.7–19.6) | 0.43 (0.37–0.48) |
| pass 9, momentum-aware (0.1.4b) | 43.6 (38.5–48.4) | 34.2 (26.9–42.0) | 63.0 (58.0–67.5) | 0.146 (0.117–0.179) | 0.61 (0.43–0.86) | 18.7 (15.6–21.9) | — |
| baseline — random pick (expected) | 35.8 (32.6–38.7) | 26.2 (20.1–32.5) | 47.5 (43.9–50.8) | 0.267 (0.230–0.307) | 0.66 (0.61–0.72) | 11.0 (10.2–12.0) | — |
| baseline — slot 0 (the menu has no order; first slot) | 51.1 (45.5–56.4) | 31.6 (24.2–39.4) | 62.8 (56.8–68.3) | 0.238 (0.190–0.289) | 0.86 (0.86–1.00) | 13.4 (10.4–16.8) | — |
| baseline — plan closest to the human | 52.1 (48.4–55.9) | 38.7 (30.6–46.6) | 78.2 (74.7–81.5) | 0.103 (0.083–0.126) | 0.43 (0.25–0.50) | 94.8 (92.8–96.6) | — |
| ceiling — best-graded plan (closest to human among ties) | 100.0 (100.0–100.0) | 50.0 (41.5–58.6) | 100.0 (100.0–100.0) | 0.000 (0.000–0.000) | 0.00 (0.00–0.00) | 73.3 (69.5–77.1) | — |

**Table 4.1 ordinary — 6,964 examples, 234 clips.**

| scorer | pick = true best, % | pick graded ≥ 0.9, % | pick within 0.05 of best, % | regret, mean | regret, 90th pct | pick within 1.5 m of human, % | rank agreement, median |
|---|---|---|---|---|---|---|---|
| shipped | 45.9 (42.0–49.8) | 56.7 (52.5–60.7) | 69.7 (66.9–72.6) | 0.112 (0.098–0.128) | 0.34 (0.23–0.50) | 16.2 (14.0–18.5) | 0.35 (0.31–0.39) |
| pass 1 | 58.2 (55.1–61.6) | 58.1 (53.9–62.3) | 73.2 (70.6–75.9) | 0.079 (0.070–0.088) | 0.17 (0.15–0.20) | 19.9 (17.6–22.5) | 0.63 (0.60–0.64) |
| pass 9 | 49.7 (46.1–53.5) | 59.7 (55.7–63.9) | 72.1 (69.5–75.0) | 0.091 (0.079–0.103) | 0.19 (0.16–0.25) | 20.3 (18.2–22.5) | 0.51 (0.47–0.55) |
| pass 9, momentum-aware (0.1.4b) | 49.7 (46.1–53.5) | 59.8 (55.6–64.0) | 73.6 (71.0–76.3) | 0.089 (0.077–0.102) | 0.19 (0.16–0.25) | 24.3 (22.1–26.7) | — |
| baseline — random pick (expected) | 40.9 (39.0–43.1) | 47.0 (43.6–50.4) | 56.1 (54.2–58.0) | 0.199 (0.180–0.218) | 0.57 (0.52–0.60) | 13.2 (12.6–13.8) | — |
| baseline — slot 0 (the menu has no order; first slot) | 56.0 (52.6–59.4) | 51.9 (47.7–55.8) | 70.9 (67.7–73.9) | 0.199 (0.171–0.229) | 1.00 (1.00–1.00) | 15.1 (13.0–17.2) | — |
| baseline — plan closest to the human | 57.4 (55.0–59.9) | 64.7 (60.3–69.0) | 87.2 (85.5–88.9) | 0.056 (0.047–0.066) | 0.14 (0.09–0.14) | 95.6 (94.4–96.6) | — |
| ceiling — best-graded plan (closest to human among ties) | 100.0 (100.0–100.0) | 74.6 (70.5–78.6) | 100.0 (100.0–100.0) | 0.000 (0.000–0.000) | 0.00 (0.00–0.00) | 83.5 (81.1–85.9) | — |

**Table 4.2 — OFF-DESIGN (head swap): another checkpoint's score head on the pass-9 trunk, pass-9 menu and grades. UNSEEN, all.** A head was never trained on this trunk's features; this is not how any checkpoint runs.

| score head | pick = true best, % | pick graded ≥ 0.9, % | pick within 0.05 of best, % | regret, mean | regret, 90th pct | pick within 1.5 m of human, % | rank agreement, median |
|---|---|---|---|---|---|---|---|
| shipped head on pass-9 trunk | 29.3 (27.0–32.0) | 43.0 (39.6–46.3) | 47.6 (45.0–50.3) | 0.133 (0.124–0.142) | 0.32 (0.26–0.40) | 14.7 (13.1–16.3) | 0.16 (0.10–0.21) |
| pass-1 head on pass-9 trunk | 23.3 (20.9–25.9) | 43.5 (40.1–46.7) | 45.4 (42.9–48.2) | 0.120 (0.112–0.130) | 0.26 (0.24–0.29) | 13.0 (11.4–14.7) | 0.07 (0.03–0.12) |
| pass 9 (same rows as primary, for reference) | 38.3 (35.6–41.3) | 49.5 (45.8–52.7) | 63.0 (60.5–65.3) | 0.115 (0.105–0.125) | 0.29 (0.23–0.40) | 15.6 (14.0–17.2) | 0.39 (0.35–0.42) |

**Table 4.3 — what the menu allows (UNSEEN).** Share of examples, %, with 95 % interval; last two columns are counts of the 32 plans.

| menu | slice | examples | a plan graded ≥ 0.95 on the menu | a plan graded ≥ 0.9 | a plan within 1.5 m of the human | a best-graded plan within 1.5 m of the human | plans tied for best, mean | plans graded ≥ 0.9, mean |
|---|---|---|---|---|---|---|---|---|
| shipped | all | 11,842 | 53.3 (49.5–56.9) | 60.4 (56.5–63.9) | 62.0 (58.4–66.1) | 50.8 (47.6–54.5) | 8.8 | 11.9 |
| shipped | highway | 3,315 | 19.1 (13.5–24.6) | 37.7 (30.3–45.0) | 9.1 (5.5–12.5) | 7.1 (4.1–10.0) | 2.5 | 4.8 |
| shipped | close-pass | 1,563 | 49.4 (40.8–57.8) | 51.3 (42.5–59.8) | 84.2 (79.9–87.9) | 63.3 (58.2–68.3) | 10.5 | 9.3 |
| shipped | ordinary | 6,964 | 70.4 (66.1–74.4) | 73.3 (69.1–77.2) | 82.2 (78.8–85.5) | 68.8 (65.2–72.6) | 11.4 | 15.8 |
| pass 1 | all | 11,842 | 59.4 (55.7–62.8) | 60.4 (56.7–63.8) | 84.1 (81.7–86.4) | 57.3 (54.6–60.3) | 10.8 | 9.1 |
| pass 1 | highway | 3,315 | 37.5 (30.4–44.1) | 38.5 (31.4–45.1) | 64.9 (58.8–70.4) | 28.0 (24.1–31.6) | 5.8 | 3.6 |
| pass 1 | close-pass | 1,563 | 49.5 (40.7–58.0) | 50.4 (41.7–59.0) | 89.8 (87.1–92.6) | 60.7 (56.6–65.1) | 11.1 | 7.2 |
| pass 1 | ordinary | 6,964 | 72.1 (68.1–76.3) | 73.0 (69.0–77.1) | 92.0 (90.3–93.5) | 70.5 (67.7–73.2) | 13.1 | 12.2 |
| pass 9 | all | 11,842 | 65.6 (62.1–68.9) | 66.8 (63.4–70.1) | 89.1 (87.1–91.0) | 70.6 (67.9–73.5) | 10.6 | 12.4 |
| pass 9 | highway | 3,315 | 56.5 (49.2–62.8) | 58.3 (51.1–64.6) | 72.6 (66.7–77.5) | 42.2 (36.7–47.4) | 5.1 | 8.6 |
| pass 9 | close-pass | 1,563 | 49.6 (41.0–58.3) | 50.0 (41.5–58.6) | 94.8 (92.8–96.6) | 73.3 (69.5–77.1) | 11.4 | 8.4 |
| pass 9 | ordinary | 6,964 | 73.6 (69.5–77.6) | 74.6 (70.5–78.6) | 95.6 (94.4–96.6) | 83.5 (81.1–85.9) | 13.1 | 15.0 |

**Table 4.4 — top-2 margin (P-a5), UNSEEN.** Score gap = top score minus second (0–1 scale); grade gap = true-grade difference between those two plans.

| scorer | slice | examples | score gap, median | score gap < 0.01, % | of those: grade gap > 0.1, % | of those: top-scored is the worse by > 0.1, % | grade gap > 0.1, all examples, % |
|---|---|---|---|---|---|---|---|
| shipped | all | 11,842 | 0.0031 (0.0029–0.0033) | 84.9 (83.3–86.3) | 9.0 (8.1–10.0) | 4.4 (3.9–5.0) | 10.6 (9.6–11.5) |
| shipped | highway | 3,315 | 0.0022 (0.0020–0.0024) | 94.3 (92.4–96.0) | 11.3 (9.4–13.4) | 5.2 (4.1–6.3) | 11.9 (10.0–14.0) |
| shipped | close-pass | 1,563 | 0.0051 (0.0044–0.0060) | 71.2 (66.5–75.5) | 10.5 (8.3–12.9) | 5.6 (4.1–7.3) | 13.2 (10.8–15.6) |
| shipped | ordinary | 6,964 | 0.0034 (0.0031–0.0036) | 83.5 (81.6–85.4) | 7.5 (6.5–8.5) | 3.8 (3.3–4.4) | 9.3 (8.2–10.4) |
| pass 1 | all | 11,842 | 0.0030 (0.0028–0.0033) | 83.6 (81.6–85.2) | 12.7 (11.2–14.3) | 6.3 (5.4–7.3) | 14.4 (12.9–16.1) |
| pass 1 | highway | 3,315 | 0.0042 (0.0036–0.0048) | 79.5 (75.4–83.2) | 30.3 (26.8–34.3) | 15.4 (13.0–18.0) | 30.5 (27.5–34.1) |
| pass 1 | close-pass | 1,563 | 0.0036 (0.0031–0.0042) | 80.0 (75.4–84.1) | 8.8 (6.7–10.9) | 4.8 (3.6–6.2) | 11.2 (8.9–13.6) |
| pass 1 | ordinary | 6,964 | 0.0025 (0.0023–0.0028) | 86.4 (84.3–88.5) | 5.8 (4.9–6.7) | 2.6 (2.2–3.2) | 7.5 (6.4–8.7) |
| pass 9 | all | 11,842 | 0.0025 (0.0023–0.0027) | 84.0 (82.3–85.6) | 11.5 (10.2–12.8) | 5.7 (5.1–6.4) | 13.9 (12.6–15.3) |
| pass 9 | highway | 3,315 | 0.0032 (0.0028–0.0037) | 83.3 (80.2–86.3) | 21.8 (18.4–25.4) | 10.2 (8.6–12.1) | 24.3 (21.2–27.6) |
| pass 9 | close-pass | 1,563 | 0.0041 (0.0034–0.0050) | 74.7 (69.3–79.4) | 11.2 (8.6–14.1) | 6.1 (4.3–8.1) | 15.2 (12.4–18.2) |
| pass 9 | ordinary | 6,964 | 0.0020 (0.0018–0.0022) | 86.4 (84.5–88.4) | 6.8 (5.8–7.9) | 3.5 (2.9–4.2) | 8.6 (7.5–9.9) |

**Table 4.5 — how different the menus are (UNSEEN).** Mean distance from each plan of one menu to the nearest plan of the other (average over the 8 poses), metres.

| pair | slice | examples | pass-9 plan → nearest plan of the other | other's plan → nearest pass-9 plan |
|---|---|---|---|---|
| pass 9 vs pass 1 | all | 11,842 | 1.25 (1.09–1.43) | 1.92 (1.69–2.16) |
| pass 9 vs pass 1 | highway | 3,315 | 3.07 (2.61–3.58) | 4.73 (4.23–5.28) |
| pass 9 vs pass 1 | close-pass | 1,563 | 0.56 (0.53–0.60) | 0.76 (0.69–0.83) |
| pass 9 vs pass 1 | ordinary | 6,964 | 0.54 (0.52–0.56) | 0.84 (0.78–0.89) |
| pass 9 vs shipped | all | 11,842 | 4.71 (3.91–5.58) | 2.66 (2.20–3.17) |
| pass 9 vs shipped | highway | 3,315 | 14.53 (12.53–16.80) | 7.87 (6.57–9.33) |
| pass 9 vs shipped | close-pass | 1,563 | 0.82 (0.75–0.90) | 0.68 (0.63–0.72) |
| pass 9 vs shipped | ordinary | 6,964 | 0.90 (0.82–1.00) | 0.63 (0.60–0.66) |

**Table 4.S1 — PRIMARY: each checkpoint whole, on its own menu. SEEN, all 11,842 graded examples, 727 clips.** Brackets: 95 % interval, clip-level bootstrap.

| scorer | pick = true best, % | pick graded ≥ 0.9, % | pick within 0.05 of best, % | regret, mean | regret, 90th pct | pick within 1.5 m of human, % | rank agreement, median |
|---|---|---|---|---|---|---|---|
| shipped | 32.7 (30.3–35.0) | 42.5 (39.8–45.3) | 66.1 (64.3–67.9) | 0.114 (0.105–0.122) | 0.36 (0.28–0.38) | 11.9 (10.7–13.1) | 0.21 (0.19–0.24) |
| pass 1 | 43.1 (40.9–45.4) | 45.1 (42.5–47.8) | 60.1 (58.2–62.2) | 0.103 (0.096–0.110) | 0.25 (0.23–0.28) | 15.6 (14.3–17.0) | 0.51 (0.49–0.53) |
| pass 9 | 40.8 (38.2–43.2) | 53.3 (50.9–55.9) | 65.5 (63.5–67.5) | 0.078 (0.072–0.084) | 0.19 (0.18–0.19) | 15.9 (14.5–17.3) | 0.48 (0.45–0.50) |
| pass 9, momentum-aware (0.1.4b) | 40.4 (38.0–42.9) | 53.2 (50.8–55.8) | 65.8 (63.8–67.9) | 0.079 (0.074–0.085) | 0.19 (0.18–0.20) | 17.5 (16.1–18.9) | — |
| baseline — random pick (expected) | 32.4 (31.0–33.9) | 39.4 (37.4–41.4) | 48.2 (46.9–49.6) | 0.208 (0.197–0.218) | 0.54 (0.51–0.58) | 10.8 (10.4–11.2) | — |
| baseline — slot 0 (the menu has no order; first slot) | 44.1 (41.9–46.4) | 46.0 (43.7–48.4) | 62.5 (60.4–64.4) | 0.191 (0.177–0.206) | 0.98 (0.86–1.00) | 14.5 (13.2–15.8) | — |
| baseline — plan closest to the human | 51.0 (49.3–52.7) | 54.4 (51.9–56.9) | 77.6 (75.9–79.3) | 0.075 (0.068–0.082) | 0.15 (0.15–0.15) | 89.1 (87.8–90.3) | — |
| ceiling — best-graded plan (closest to human among ties) | 100.0 (100.0–100.0) | 70.1 (67.8–72.4) | 100.0 (100.0–100.0) | 0.000 (0.000–0.000) | 0.00 (0.00–0.00) | 69.2 (67.2–71.5) | — |

**Table 4.S1 highway — 3,475 examples, 205 clips.**

| scorer | pick = true best, % | pick graded ≥ 0.9, % | pick within 0.05 of best, % | regret, mean | regret, 90th pct | pick within 1.5 m of human, % | rank agreement, median |
|---|---|---|---|---|---|---|---|
| shipped | 4.6 (3.2–6.1) | 14.4 (11.0–17.6) | 67.3 (63.2–71.2) | 0.098 (0.086–0.111) | 0.22 (0.20–0.37) | 1.2 (0.7–1.7) | -0.11 (-0.16–-0.04) |
| pass 1 | 14.3 (11.8–17.0) | 18.9 (15.7–21.9) | 38.7 (35.6–41.7) | 0.148 (0.137–0.161) | 0.39 (0.30–0.44) | 6.2 (4.8–7.7) | 0.11 (0.06–0.16) |
| pass 9 | 12.0 (9.3–14.7) | 36.6 (32.2–40.8) | 47.0 (43.1–50.5) | 0.117 (0.107–0.128) | 0.22 (0.21–0.24) | 5.2 (3.9–6.5) | 0.09 (0.02–0.15) |
| pass 9, momentum-aware (0.1.4b) | 12.2 (9.5–14.8) | 36.5 (32.2–40.7) | 47.1 (43.3–50.9) | 0.117 (0.107–0.129) | 0.22 (0.21–0.24) | 5.8 (4.4–7.3) | — |
| baseline — random pick (expected) | 15.2 (13.3–17.1) | 26.5 (23.2–29.9) | 36.5 (34.1–38.8) | 0.194 (0.178–0.211) | 0.44 (0.37–0.53) | 5.9 (5.4–6.4) | — |
| baseline — slot 0 (the menu has no order; first slot) | 19.8 (16.3–23.4) | 32.7 (28.5–36.9) | 48.1 (43.9–52.2) | 0.165 (0.146–0.185) | 0.54 (0.32–0.61) | 11.0 (8.7–13.2) | — |
| baseline — plan closest to the human | 36.7 (34.0–39.5) | 33.2 (29.2–37.2) | 59.8 (56.0–63.1) | 0.108 (0.096–0.122) | 0.26 (0.18–0.29) | 74.3 (71.1–77.5) | — |
| ceiling — best-graded plan (closest to human among ties) | 100.0 (100.0–100.0) | 59.9 (55.2–64.4) | 100.0 (100.0–100.0) | 0.000 (0.000–0.000) | 0.00 (0.00–0.00) | 43.0 (39.2–46.8) | — |

**Table 4.S1 close-pass — 1,347 examples, 88 clips.**

| scorer | pick = true best, % | pick graded ≥ 0.9, % | pick within 0.05 of best, % | regret, mean | regret, 90th pct | pick within 1.5 m of human, % | rank agreement, median |
|---|---|---|---|---|---|---|---|
| shipped | 43.2 (38.4–48.4) | 37.7 (31.4–44.1) | 62.0 (57.7–66.5) | 0.123 (0.105–0.142) | 0.41 (0.31–0.50) | 13.1 (10.3–15.9) | 0.33 (0.27–0.38) |
| pass 1 | 46.8 (42.5–51.8) | 40.7 (34.2–47.4) | 63.5 (59.5–67.7) | 0.095 (0.080–0.111) | 0.26 (0.21–0.32) | 19.7 (16.1–23.7) | 0.59 (0.55–0.61) |
| pass 9 | 47.1 (42.0–52.5) | 43.7 (37.2–50.4) | 67.1 (63.0–71.0) | 0.072 (0.061–0.083) | 0.17 (0.15–0.19) | 16.8 (13.2–20.0) | 0.55 (0.50–0.58) |
| pass 9, momentum-aware (0.1.4b) | 47.3 (42.2–52.5) | 43.6 (37.0–50.4) | 67.6 (63.7–71.5) | 0.071 (0.061–0.083) | 0.17 (0.15–0.19) | 18.5 (15.1–21.9) | — |
| baseline — random pick (expected) | 37.3 (34.4–40.2) | 31.6 (27.0–36.6) | 49.9 (47.4–52.4) | 0.227 (0.205–0.250) | 0.54 (0.49–0.59) | 12.0 (11.2–12.9) | — |
| baseline — slot 0 (the menu has no order; first slot) | 53.8 (49.6–57.8) | 39.1 (32.9–45.4) | 66.5 (62.5–70.5) | 0.195 (0.160–0.233) | 0.92 (0.86–1.00) | 17.9 (14.3–21.2) | — |
| baseline — plan closest to the human | 51.7 (48.1–55.8) | 44.8 (38.1–51.7) | 78.0 (75.1–80.7) | 0.092 (0.077–0.108) | 0.35 (0.15–0.46) | 96.0 (94.2–97.5) | — |
| ceiling — best-graded plan (closest to human among ties) | 100.0 (100.0–100.0) | 59.5 (52.7–66.6) | 100.0 (100.0–100.0) | 0.000 (0.000–0.000) | 0.00 (0.00–0.00) | 74.8 (71.6–78.0) | — |

**Table 4.S1 ordinary — 7,020 examples, 434 clips.**

| scorer | pick = true best, % | pick graded ≥ 0.9, % | pick within 0.05 of best, % | regret, mean | regret, 90th pct | pick within 1.5 m of human, % | rank agreement, median |
|---|---|---|---|---|---|---|---|
| shipped | 44.6 (41.7–47.6) | 57.4 (54.3–60.3) | 66.3 (63.9–68.7) | 0.120 (0.107–0.131) | 0.36 (0.27–0.49) | 16.9 (15.2–18.7) | 0.32 (0.29–0.35) |
| pass 1 | 56.6 (54.0–59.3) | 58.9 (55.9–61.7) | 70.1 (67.8–72.1) | 0.082 (0.074–0.091) | 0.19 (0.17–0.21) | 19.5 (17.7–21.4) | 0.60 (0.58–0.62) |
| pass 9 | 53.8 (50.8–56.8) | 63.5 (60.3–66.3) | 74.3 (72.0–76.5) | 0.060 (0.053–0.067) | 0.14 (0.14–0.16) | 21.0 (19.2–22.9) | 0.58 (0.56–0.61) |
| pass 9, momentum-aware (0.1.4b) | 53.1 (50.1–56.1) | 63.4 (60.2–66.3) | 74.7 (72.3–76.9) | 0.062 (0.055–0.070) | 0.14 (0.14–0.16) | 23.0 (21.3–25.0) | — |
| baseline — random pick (expected) | 40.0 (38.4–41.6) | 47.3 (44.9–49.7) | 53.7 (52.2–55.3) | 0.211 (0.196–0.226) | 0.58 (0.54–0.61) | 13.0 (12.5–13.5) | — |
| baseline — slot 0 (the menu has no order; first slot) | 54.2 (51.7–56.8) | 54.0 (50.9–56.9) | 68.8 (66.4–71.3) | 0.204 (0.182–0.226) | 1.00 (1.00–1.00) | 15.6 (14.1–17.3) | — |
| baseline — plan closest to the human | 57.9 (55.9–59.8) | 66.7 (63.5–69.4) | 86.3 (84.8–87.7) | 0.056 (0.047–0.064) | 0.14 (0.13–0.14) | 95.0 (94.0–96.0) | — |
| ceiling — best-graded plan (closest to human among ties) | 100.0 (100.0–100.0) | 77.1 (74.4–79.6) | 100.0 (100.0–100.0) | 0.000 (0.000–0.000) | 0.00 (0.00–0.00) | 81.2 (79.1–83.1) | — |

**Table 4.S2 — OFF-DESIGN (head swap): another checkpoint's score head on the pass-9 trunk, pass-9 menu and grades. SEEN, all.** A head was never trained on this trunk's features; this is not how any checkpoint runs.

| score head | pick = true best, % | pick graded ≥ 0.9, % | pick within 0.05 of best, % | regret, mean | regret, 90th pct | pick within 1.5 m of human, % | rank agreement, median |
|---|---|---|---|---|---|---|---|
| shipped head on pass-9 trunk | 29.9 (27.8–32.0) | 46.0 (43.6–48.6) | 48.6 (46.4–50.9) | 0.122 (0.113–0.130) | 0.27 (0.25–0.29) | 15.1 (13.8–16.5) | 0.21 (0.18–0.25) |
| pass-1 head on pass-9 trunk | 24.8 (22.8–26.9) | 46.2 (43.8–48.8) | 45.8 (43.7–48.1) | 0.114 (0.107–0.121) | 0.24 (0.23–0.26) | 12.9 (11.6–14.2) | 0.13 (0.10–0.17) |
| pass 9 (same rows as primary, for reference) | 40.8 (38.2–43.2) | 53.3 (50.9–55.9) | 65.5 (63.5–67.5) | 0.078 (0.072–0.084) | 0.19 (0.18–0.19) | 15.9 (14.5–17.3) | 0.48 (0.45–0.50) |

**Table 4.S3 — what the menu allows (SEEN).** Share of examples, %, with 95 % interval; last two columns are counts of the 32 plans.

| menu | slice | examples | a plan graded ≥ 0.95 on the menu | a plan graded ≥ 0.9 | a plan within 1.5 m of the human | a best-graded plan within 1.5 m of the human | plans tied for best, mean | plans graded ≥ 0.9, mean |
|---|---|---|---|---|---|---|---|---|
| shipped | all | 11,842 | 53.6 (50.6–56.5) | 60.4 (57.5–63.3) | 60.7 (57.4–63.8) | 48.6 (45.7–51.5) | 8.7 | 11.7 |
| shipped | highway | 3,475 | 15.3 (11.9–18.8) | 31.0 (26.1–36.0) | 7.9 (5.6–10.4) | 6.7 (4.6–8.9) | 2.2 | 4.0 |
| shipped | close-pass | 1,347 | 53.5 (46.4–60.7) | 56.4 (49.7–63.6) | 83.7 (79.7–87.5) | 62.6 (57.7–67.1) | 11.5 | 10.4 |
| shipped | ordinary | 7,020 | 72.5 (69.5–75.1) | 75.7 (72.9–78.2) | 82.4 (80.1–84.6) | 66.7 (64.0–69.3) | 11.3 | 15.8 |
| pass 1 | all | 11,842 | 60.3 (57.6–63.0) | 61.8 (59.1–64.6) | 83.8 (82.1–85.3) | 56.0 (53.9–58.2) | 10.5 | 9.3 |
| pass 1 | highway | 3,475 | 34.1 (29.8–38.4) | 35.5 (31.2–39.9) | 67.5 (63.8–71.1) | 27.7 (24.7–30.4) | 5.5 | 3.2 |
| pass 1 | close-pass | 1,347 | 55.6 (48.8–62.5) | 57.7 (51.0–64.7) | 92.2 (90.1–94.2) | 63.1 (59.2–67.2) | 11.7 | 8.4 |
| pass 1 | ordinary | 7,020 | 74.1 (71.4–76.6) | 75.6 (72.8–78.1) | 90.2 (88.8–91.6) | 68.7 (66.4–70.9) | 12.7 | 12.5 |
| pass 9 | all | 11,842 | 68.8 (66.4–71.1) | 70.1 (67.8–72.4) | 89.1 (87.8–90.3) | 69.2 (67.2–71.5) | 10.4 | 12.6 |
| pass 9 | highway | 3,475 | 58.4 (53.8–62.8) | 59.9 (55.2–64.4) | 74.3 (71.1–77.5) | 43.0 (39.2–46.8) | 4.8 | 8.5 |
| pass 9 | close-pass | 1,347 | 57.6 (50.7–64.9) | 59.5 (52.7–66.6) | 96.0 (94.2–97.5) | 74.8 (71.6–78.0) | 11.9 | 10.1 |
| pass 9 | ordinary | 7,020 | 76.1 (73.2–78.6) | 77.1 (74.4–79.6) | 95.0 (94.0–96.0) | 81.2 (79.1–83.1) | 12.8 | 15.1 |

**Table 4.S4 — top-2 margin (P-a5), SEEN.** Score gap = top score minus second (0–1 scale); grade gap = true-grade difference between those two plans.

| scorer | slice | examples | score gap, median | score gap < 0.01, % | of those: grade gap > 0.1, % | of those: top-scored is the worse by > 0.1, % | grade gap > 0.1, all examples, % |
|---|---|---|---|---|---|---|---|
| shipped | all | 11,842 | 0.0031 (0.0030–0.0033) | 83.2 (81.9–84.5) | 9.1 (8.3–9.8) | 4.7 (4.2–5.2) | 11.3 (10.5–12.2) |
| shipped | highway | 3,475 | 0.0021 (0.0020–0.0023) | 93.4 (91.6–94.9) | 10.5 (9.0–12.1) | 5.3 (4.4–6.3) | 11.5 (9.9–13.1) |
| shipped | close-pass | 1,347 | 0.0048 (0.0041–0.0054) | 74.1 (70.5–77.9) | 9.3 (7.3–11.4) | 4.2 (2.9–5.5) | 12.3 (10.2–14.6) |
| shipped | ordinary | 7,020 | 0.0037 (0.0034–0.0039) | 79.9 (78.1–81.6) | 8.2 (7.2–9.1) | 4.4 (3.8–5.0) | 11.0 (9.9–12.1) |
| pass 1 | all | 11,842 | 0.0030 (0.0029–0.0032) | 83.2 (81.8–84.6) | 13.4 (12.1–14.6) | 6.6 (5.9–7.4) | 15.1 (14.0–16.4) |
| pass 1 | highway | 3,475 | 0.0039 (0.0036–0.0043) | 79.9 (76.9–82.9) | 30.0 (27.3–32.5) | 15.0 (13.4–16.7) | 29.9 (27.5–32.4) |
| pass 1 | close-pass | 1,347 | 0.0036 (0.0032–0.0042) | 80.5 (77.3–83.8) | 7.6 (5.7–9.6) | 3.7 (2.6–4.8) | 9.9 (7.9–11.9) |
| pass 1 | ordinary | 7,020 | 0.0025 (0.0024–0.0027) | 85.3 (83.4–87.0) | 6.7 (6.0–7.5) | 3.3 (2.8–3.8) | 8.8 (7.8–9.9) |
| pass 9 | all | 11,842 | 0.0024 (0.0022–0.0026) | 83.1 (81.7–84.5) | 11.0 (9.9–12.2) | 4.8 (4.3–5.4) | 13.4 (12.2–14.5) |
| pass 9 | highway | 3,475 | 0.0029 (0.0026–0.0032) | 82.4 (79.9–84.8) | 23.0 (20.5–25.7) | 9.4 (8.1–10.7) | 24.7 (22.2–27.4) |
| pass 9 | close-pass | 1,347 | 0.0037 (0.0032–0.0044) | 75.4 (71.7–79.1) | 7.9 (5.8–10.2) | 4.2 (3.0–5.5) | 9.9 (8.0–12.2) |
| pass 9 | ordinary | 7,020 | 0.0020 (0.0018–0.0022) | 85.0 (83.2–86.7) | 5.8 (5.0–6.7) | 2.7 (2.3–3.3) | 8.4 (7.3–9.6) |

**Table 4.S5 — how different the menus are (SEEN).** Mean distance from each plan of one menu to the nearest plan of the other (average over the 8 poses), metres.

| pair | slice | examples | pass-9 plan → nearest plan of the other | other's plan → nearest pass-9 plan |
|---|---|---|---|---|
| pass 9 vs pass 1 | all | 11,842 | 1.25 (1.14–1.38) | 1.95 (1.79–2.12) |
| pass 9 vs pass 1 | highway | 3,475 | 2.95 (2.64–3.27) | 4.69 (4.32–5.02) |
| pass 9 vs pass 1 | close-pass | 1,347 | 0.56 (0.53–0.58) | 0.79 (0.74–0.85) |
| pass 9 vs pass 1 | ordinary | 7,020 | 0.55 (0.53–0.57) | 0.82 (0.78–0.86) |
| pass 9 vs shipped | all | 11,842 | 4.91 (4.31–5.57) | 2.75 (2.39–3.12) |
| pass 9 vs shipped | highway | 3,475 | 14.70 (13.17–16.20) | 7.85 (6.87–8.84) |
| pass 9 vs shipped | close-pass | 1,347 | 0.78 (0.73–0.84) | 0.66 (0.63–0.69) |
| pass 9 vs shipped | ordinary | 7,020 | 0.86 (0.81–0.93) | 0.62 (0.60–0.65) |

**Table 4.6 — part b, the scorer's own outputs: separation of plans that pass a check from plans that fail it** (0.5 = coin flip; pooled over plans; 95 % clip-bootstrap interval, 200 resamples). "Total" = the trained total output; "own" = that check's own output, which has zero training weight; collision, drivable area and traffic light have no output of their own. Traffic light is skipped: fewer than 200 failing plans.

| check | needs the map | plans passing, UNSEEN pass 9 | shipped total, UNSEEN | shipped own, UNSEEN | pass 1 total, UNSEEN | pass 1 own, UNSEEN | pass 9 total, UNSEEN | pass 9 own, UNSEEN | pass 9 total, SEEN | pass 9 own, SEEN | shipped own, SEEN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| collision | yes | 93.6 % | 0.76 (0.72–0.80) | — | 0.80 (0.77–0.82) | — | 0.78 (0.75–0.82) | — | 0.91 (0.90–0.92) | — | — |
| drivable area | yes | 90.2 % | 0.72 (0.69–0.76) | — | 0.82 (0.79–0.84) | — | 0.83 (0.80–0.85) | — | 0.90 (0.89–0.91) | — | — |
| driving direction | yes | 90.6 % | 0.77 (0.73–0.81) | 0.48 (0.45–0.53) | 0.83 (0.81–0.85) | 0.76 (0.74–0.78) | 0.85 (0.83–0.87) | 0.49 (0.46–0.53) | 0.92 (0.91–0.93) | 0.38 (0.35–0.41) | 0.44 (0.42–0.46) |
| progress (full) | yes | 45.6 % | 0.38 (0.35–0.40) | 0.40 (0.37–0.42) | 0.63 (0.61–0.64) | 0.56 (0.54–0.57) | 0.56 (0.55–0.58) | 0.44 (0.42–0.46) | 0.59 (0.57–0.61) | 0.45 (0.43–0.46) | 0.41 (0.39–0.43) |
| time-to-collision | yes | 91.7 % | 0.62 (0.56–0.67) | 0.61 (0.58–0.65) | 0.76 (0.73–0.79) | 0.77 (0.74–0.79) | 0.74 (0.71–0.78) | 0.75 (0.71–0.78) | 0.87 (0.86–0.89) | 0.79 (0.77–0.82) | 0.64 (0.61–0.66) |
| lane keeping | yes | 67.8 % | 0.62 (0.60–0.65) | 0.51 (0.48–0.53) | 0.61 (0.58–0.65) | 0.46 (0.43–0.48) | 0.63 (0.61–0.66) | 0.45 (0.42–0.47) | 0.70 (0.68–0.71) | 0.43 (0.41–0.45) | 0.48 (0.46–0.50) |
| comfort | no | 75.4 % | 0.39 (0.35–0.42) | 0.80 (0.77–0.81) | 0.67 (0.65–0.68) | 0.65 (0.63–0.67) | 0.67 (0.65–0.69) | 0.47 (0.43–0.50) | 0.66 (0.64–0.67) | 0.46 (0.44–0.48) | 0.79 (0.77–0.81) |
| total grade ≥ 0.9 | yes | 38.6 % | 0.58 (0.56–0.61) | — | 0.75 (0.73–0.77) | — | 0.76 (0.74–0.77) | — | 0.84 (0.83–0.85) | — | — |

**Table 4.7 — part b, fresh linear probes on pass-9 features, trained on 120,000 plans from 60 % of the SEEN clips.** "Slot" = the 256 numbers per plan the scorer network reads; "feat" = the 1,024 numbers before its final layer (these fits stopped at the 300-iteration limit, so they are lower bounds for that feature set). The held-out SEEN clips were held out from the probe only: the model itself trained on them.

| check | probe on slot: SEEN held-out | probe on slot: UNSEEN | probe on feat: SEEN held-out | probe on feat: UNSEEN | pass 9 total output, UNSEEN | slot probe minus total output, UNSEEN |
|---|---|---|---|---|---|---|
| collision | 0.93 (0.92–0.95) | 0.88 (0.85–0.90) | 0.95 (0.94–0.96) | 0.84 (0.81–0.87) | 0.78 (0.75–0.82) | +0.09 |
| drivable area | 0.91 (0.89–0.93) | 0.88 (0.86–0.91) | 0.92 (0.90–0.93) | 0.87 (0.85–0.89) | 0.83 (0.80–0.85) | +0.06 |
| driving direction | 0.91 (0.89–0.93) | 0.89 (0.86–0.91) | 0.95 (0.93–0.96) | 0.89 (0.87–0.91) | 0.85 (0.83–0.87) | +0.03 |
| progress (full) | 0.86 (0.84–0.87) | 0.85 (0.84–0.86) | 0.88 (0.86–0.89) | 0.85 (0.84–0.86) | 0.56 (0.55–0.58) | +0.29 |
| time-to-collision | 0.89 (0.87–0.91) | 0.83 (0.80–0.86) | 0.92 (0.90–0.94) | 0.81 (0.78–0.85) | 0.74 (0.71–0.78) | +0.09 |
| lane keeping | 0.77 (0.74–0.79) | 0.73 (0.70–0.76) | 0.79 (0.77–0.82) | 0.73 (0.70–0.75) | 0.63 (0.61–0.66) | +0.10 |
| comfort | 0.94 (0.93–0.95) | 0.95 (0.94–0.96) | 0.94 (0.93–0.95) | 0.95 (0.94–0.96) | 0.67 (0.65–0.69) | +0.28 |
| total grade ≥ 0.9 | 0.84 (0.83–0.85) | 0.80 (0.79–0.82) | 0.89 (0.88–0.90) | 0.81 (0.79–0.83) | 0.76 (0.74–0.77) | +0.05 |

**Table 4.8 — part c, calculator spot check.** 400 catalogue curves (20 on each of 20 UNSEEN examples from 20 clips, seeded) re-graded
with the calculator call S6 uses for menus, against S3's stored total grade: largest difference 0.00024, all 400 within 0.001 (the
stored grades are 16-bit). `partc_spotcheck_20x20.csv`.

**Table 4.9 — part c, pass-9 menu plans against their nearest catalogue curve, UNSEEN** (nearest = smallest mean distance over the 8
poses; `table_c_nearest_catalogue_unseen.csv`, per example in `partc_nearest_catalogue_unseen.parquet`). Brackets: 95 % interval.

| slice | examples | plans whose nearest curve has a stored grade | distance to nearest curve, mean | true grade of menu plans, mean | stored grade of nearest curves, mean | true minus nearest, mean | plans graded > 0.1 below nearest | plans graded > 0.1 above nearest | a plan graded ≥ 0.95 on the menu | a curve graded ≥ 0.95 in the catalogue |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 11,842 | 97.7 % | 1.11 m | 0.719 (0.702–0.737) | 0.714 (0.696–0.731) | +0.006 (−0.000 to +0.011) | 14.1 % (12.4–16.0) | 8.6 % (7.9–9.4) | 65.6 % | 83.7 % |
| highway | 3,315 | 99.4 % | 3.08 m | 0.705 (0.670–0.736) | 0.731 (0.696–0.763) | −0.026 (−0.041 to −0.013) | 34.3 % (30.2–38.6) | 12.6 % (11.2–14.0) | 56.5 % | 93.6 % |
| close-pass | 1,563 | 96.3 % | 0.40 m | 0.621 (0.570–0.671) | 0.599 (0.544–0.650) | +0.022 (+0.009 to +0.037) | 7.6 % (6.6–8.7) | 9.1 % (7.2–11.2) | 49.6 % | 62.8 % |
| ordinary | 6,964 | 97.1 % | 0.34 m | 0.748 (0.725–0.771) | 0.731 (0.707–0.754) | +0.017 (+0.012 to +0.023) | 6.0 % (5.4–6.5) | 6.6 % (5.8–7.7) | 73.6 % | 83.7 % |

The last two columns are the generator's ceiling against the catalogue's: a curve graded ≥ 0.95 exists in the catalogue on 83.7 % of
UNSEEN examples (93.6 % on highway) but on the 32-plan menu on 65.6 % (56.5 % on highway). The "catalogue" column is a floor, since
S3's pre-filter left some curves ungraded.

## 5. Flags

1. **Brief premise wrong** (2.2): the scorer is not trained on catalogue curves.
2. **S4's 93 % and 15 % are distances, not grades** (2.2). With true grades, S4's stored readout has a plan graded ≥ 0.95 on the menu
   on 65.9 % of UNSEEN examples at pass 9.
3. **6 outputs, 5 of them untrained** (2.3). Part b.1 can compare 5 checks with a scorer output of their own; collision, drivable area
   and traffic light have none and get the total output instead.
4. **"4 map-free checks" is not in `s3_scorer.py`** (2.3). From the calculator code only comfort is map-free; collision and
   time-to-collision are actor-driven with map-based fault rules; progress needs the mapped centre line, although the brief lists it as
   map-free. Part b.3 will group as: actor / motion checks (collision, time-to-collision, comfort) vs map checks (drivable area,
   driving direction, traffic light, progress, lane keeping), and show progress separately so either reading can be made.
5. **No graded pull-out example exists in SEEN or UNSEEN, so failure B (pull-outs) is outside S6's reach** (2.4).
6. **S4 report disagreement, small:** `RD/06_first_training_run/settings_manifest.json` and `RD/04_training_set/REPORT.md:14` say 4,018 training clips;
   `manifest.parquet` holds 4,015 distinct clips. Not investigated.
7. **V3's denominator** (1,570) includes 7 close-pass examples that are not graded; the graded close-pass count is 1,563.
10. **S4's evaluation ran in 16-bit; the adapter runs in 32-bit.** `s4_train.py:82` wraps the forward pass in 16-bit autocast for
    evaluation as well as training. No autocast or half-precision call exists in `adapters/drivejepa/real_model.py`,
    `adapters/drivejepa014b/momentum.py` or the 0.1.6 patch tree. On the first 1,024 UNSEEN examples the two number formats pick a
    different plan on about half of the examples (2.5). S4's aggregate rates hold; its per-example picks are not what the car would pick.
11. **The pick is close to a coin toss between the top plans** (2.5): median gap between the top two scores 0.002–0.003; under 0.01 on
    about 85 % of examples. Part a reports this gap next to every ruler.
12. **S4's rank agreement breaks ties by slot order** (`s4_train.py:86-88`, ranks from a double argsort). Grades tie often (many plans at
    the same grade), so the number depends on plan order. Part c reports both that definition and the standard one (average ranks).
13. **Session ended at about 05:08 on 2026-10-08 and took the UNSEEN run with it** (24 of 47 shards were on the NAS, none half-written).
    Resumed at 05:26 from the next shard in its own session (`run_s6_all.sh` under `setsid nohup`; finished shards are skipped, never
    redone), with SEEN queued behind it and 6 threads + 2 grading workers (Sid). The load log has a gap from 05:08 to 05:26.
14. **Precision isolation (16-bit vs 32-bit) is DEFERRED, whole item.** S4's per-example files do not store the pick's slot number
    (columns: `dist4, ade, best_dist4, fastest_end, chosen_end, human_end, top1, top1_strict, rank_corr, ref_chosen, ref_best, ref_std,
    dist4_of_ref_best`), so a slot-match rate between S4's picks and the 32-bit picks cannot be computed. A 16-bit rerun on CPU would not
    isolate precision (different code path from S4's GPU run), and the GPU is in use. To do when the GPU is free: re-score 200 UNSEEN
    examples in 16-bit on the GPU and report the share of picks matching the 32-bit picks. Until then "rounding flips the winner" (2.5)
    stays an inference.
15. **P-a5 (top-2 margin) added in `PREDICTIONS_v4.md`, hashed 2026-10-08 05:36; half of it is not blind.** The 70 % threshold on the
    top-2 score gap was set after I had reported 85.3 % on the first 1,024 examples. The true-grade gap between the top two plans had
    not been computed on any example when v4 was hashed.
16. **EXAM CLIPS, INSPECTION ONLY, NOT AN EVALUATION: the two pull-out scenes of the S5 trial (d2715023, b44a4a44), adapter 0.1.6.**
    Read-only on the trial's stored per-replan menus (`07_closed_loop_test/menus/d016_<scene>/`); no model run; used for nothing else in S6
    (`s6_exam_inspect.py`; per replan: `exam_inspection_pullout_replans.csv`; all 32 scores per replan:
    `exam_inspection_pullout_scores32.csv`). Pull-out plan = the trial's "departing plan" (ends ≥ 2 m from the car); best = highest
    scored. 34 replans per scene, one every 0.5 s from handover.

    | per replan (34 each) | d2715023 | b44a4a44 |
    |---|---|---|
    | pick is the top-scored plan | 34 of 34 | 34 of 34 |
    | pull-out plan picked | 0 of 34 | 0 of 34 |
    | pull-out plans among the 32, median | 21 | 24 |
    | rank of the best pull-out plan, median (best to worst) | 12 (10 to 14) | 9 (7 to 11) |
    | score of the pick, median | 0.952 | 0.928 |
    | score of the best pull-out plan, median | 0.383 | 0.853 |
    | pick's score minus best pull-out's, median (smallest to largest) | 0.551 (0.123 to 0.787) | 0.072 (0.041 to 0.141) |
    | gap between the top two scores, median | 0.003 | 0.005 |
    | pick changed by the momentum-aware step | 0 of 34 | 0 of 34 |
    | end distance of the best pull-out plan (range) | 2.0–3.1 m | 2.0–3.4 m |

    Two things the per-replan rows show that the trial's medians do not. On d2715023 the best pull-out plan's score falls over the run:
    0.80 at handover, 0.57 at 5 s, about 0.2–0.35 after 7.5 s, while the pick stays at 0.92–0.99. On b44a4a44 it stays at 0.80–0.88
    throughout, 0.04–0.14 below the pick. In both scenes at least 6 stay-put plans always outrank every pull-out plan, and the
    "best pull-out" plans are short: none ends more than 3.4 m away. These reproduce the trial's own table (rank 12 (10) and 9 (7),
    scores 0.95 / 0.38 and 0.93 / 0.85).
17. **Menu slots are not interchangeable, so "slot 0" is not a neutral baseline** (pass-9 menu, UNSEEN, every fourth shard, 3,072
    examples, no intervals). Per slot, the share of examples where that slot is a true-best plan runs from 0.13 to 0.54, the mean
    grade from 0.52 to 0.81, and the mean end distance from 27 to 62 m. Slots that are often the true best (0, 5, 14, 30: about 0.5)
    are long, fast plans with middling mean grades (0.73–0.74); the slots the scorer picks most (11, 26, 18) are shorter plans. Slot 0
    is the true best on 45.6 % of all UNSEEN examples, above the scorer's 38.3 %, with a much worse regret (0.198 against 0.115) and a
    90th-percentile regret of 1.0. Reading: strict match rewards risky plans that are sometimes exactly right; not tested further.
18. **Highway, strict: the scorer is below a random pick** (pass 9 11.9 % (8.9–15.0) against 15.8 % (13.5–18.1); shipped 5.1 %). On
    regret it is still better than random (0.149 against 0.208).
19. **Second session end, 2026-10-08 morning.** The SEEN run finished on its own at 09:32 (it was detached), but the watcher that was
    to tabulate it died with the session, so the SEEN tables and part b ran at 22:08–22:28 from `run_s6_post.sh` (detached).
20. **Part b, progress check: the pass rule was changed after a first look.** A median split gave no positives for pass 1 (over half
    its plans have full progress), so progress now passes at full progress like the other checks. The first UNSEEN table is kept as
    `table_b_heads_unseen_EPmedian_superseded.csv`. No prediction named a progress threshold.
21. **Part b, probes on the 1,024 pre-output features stopped at the iteration limit** (300), so those columns are lower bounds. The
    256-feature probes converged. Probe training used 120,000 plans from 60 % of the SEEN clips, one fixed regularisation setting, no
    tuning.
22. **"SEEN held-out" in table 4.7 is held out from the probe only.** The model trained on every SEEN clip.
23. **Root disk at 98 % (46 GB free) on 2026-10-08 22:07.** S6's local folder is under 1 MB; nothing was deleted by S6.
8. **Machine over the limit before S6 started anything.** See the log.
9. **Root disk full at 23:31** (`/`, 1.8 TB, 0 bytes available; it holds `/home`, `/tmp` and this tree). Not caused by S6 (S6 had
   written under 1 MB locally). An edit in progress left `s6_dump.py`, `load_gate.sh` and `run_s6_dump.sh` empty; they were rewritten at
   23:44 once 64 GB was free. The three prediction files were re-checked against their hashes afterwards: all match. The load log has a
   gap from 23:31 to 23:44. Intact files were copied to `{runs}/drivejepa_retrain/08_scorer_diagnosis/local_backup_2331/`. Whether S5
   was affected is not known; `07_closed_loop_test/` was not opened.

### Load and swap log (every 2 min from 23:17 on, at Sid's request; 20 min before; full log in `load_log.csv`, written by `load_gate.sh`)

Gate until 23:31: start at 5-minute load < 32 and free swap > 15 GB; stop at free swap < 8 GB. It never opened: free swap moved between
0 and 14 GB all evening with nothing of S6 running, while 45–77 GB of RAM stayed available.
**Gate from 23:31 (Sid): start at 5-minute load < 32 and available RAM > 40 GB; stop at available RAM < 20 GB; pause at load ≥ 32; S6's
own resident memory capped at 12 GB (the run exits above it).** The launcher also refuses to start with under 5 GB free on the root disk.

| time (IST) | load 1 / 5 / 15 min | free swap, GB | available RAM, GB | gate |
|---|---|---|---|---|
| 22:29 (before anything) | 37.6 / 52.1 / 63.3 | 7 | 67 | closed |
| 22:57 | 35.9 / 48.7 / 51.4 | 14 | 57 | closed |
| 23:06 | 66.6 / 49.0 / 48.2 | 14.1 | 55.2 | closed |

## 6. Files written

Local (`RD/08_scorer_diagnosis/`):

| file | sha256 |
|---|---|
| `PREDICTIONS.md` | `8791c38f36df15ed2852b9efbcac174b826023b071d6bc3d3d24d646c9d51c1b` |
| `PREDICTIONS_v2.md` | `c3f4057afbb1c3a4e82a2758be402f9acf44e704fcfb4f2a2342a0732262ab60` |
| `PREDICTIONS_v3.md` | `8fcdbd7107a2d507b46589d8b5be0d31fcfa33df8d55a8be8258ee483b984b52` |
| `set_unseen.parquet` | `c7abca895f743d4354be9d1a5e96c19277cd75291f4e99acc820d2052bfd39f0` |
| `set_seen.parquet` | `0d6920026f83898c0d765aaac2603b18d386035f62cf907c6cd8fb043ab5fc14` |
| `sets_table.csv` | `f5646b41c6e46ad82c57a92a45c9fedc05bc6800d9a96d0553e5c18db705ab35` |
| `repro_stored.csv`, `load_log.csv`, `s6_sets.py`, `s6_repro_stored.py`, `load_gate.sh`, `predictions.sha256` | hashed at the end |

NAS (`{runs}/drivejepa_retrain/08_scorer_diagnosis/`): `dump/unseen/` and `dump/seen/`, 47 shards each (about 97 MB per shard, 9 GB in all); `dump_timing/` (10 examples); `local_backup_2331/`; copies of `PREDICTIONS_v4.md` and `predictions.sha256`.

## 7. CPU hours

Part 0 so far: under 0.2 CPU hour (single-process reads, one hashing pass, two timing runs).

**10-example timing (2026-10-08 01:29–01:30, UNSEEN, CPU, 4 threads + 2 grading workers + 1 loader; `timing10.log`).**

| item | value |
|---|---|
| wall time per example | 3.0 s including the one-off parity double pass; steady parts: encoder 1.49 s, three trunks + head swap 0.12 s, grading three menus 0.46 s, loading 0.02 s |
| CPU time per example | 10.1 CPU-s (includes start-up; about 8 CPU-s steady) |
| peak own memory | 6.9 GB proportional (12.8 GB if shared pages are counted once per process) |
| split forward vs the model's own forward, pass 9 | plans differ by 0.0, scorer outputs by 0.0, same pick |
| stored per example | about 0.38 MB (features, menus, grades, simulated states) → about 9 GB on the NAS for both sets |

**Projection for the full run (23,684 examples = UNSEEN + SEEN), written before launching it:** about 55–65 CPU hours. The worker rule
(half the free cores, at most 16) allows about 4 busy processes at the load seen at 01:30 (25 on 32 cores), so the run uses 3 threads +
1 grading worker + 1 loader; at an expected 3–3.5 s per example that is about 10–11.5 h of wall time per set, 20–23 h for both, longer
if the gate pauses it. UNSEEN runs first because the reproduction check and most tables need it.

First attempt at 01:28 stopped itself on the memory cap before grading anything (`timing10_attempt1_cap_tripped.log`): the cap summed
resident memory per process and read 22.1 GB while the parent peaked at 7.4 GB, because two forked grading workers were each charged for
the parent's model. Grading workers are now spawned clean and the cap is checked on proportional memory (shared pages split between the
processes sharing them). The cap itself is unchanged at 12 GB.

UNSEEN run, measured: launch 1 (3 threads + 1 grading worker, 01:31–05:08, killed with the session) 6,144 examples at 2.05 s each,
about 7 CPU hours by the timing run's ratio (not logged; the launch never wrote its totals). Launch 2 (6 threads + 2 workers,
05:26–06:43) 5,698 examples in 76 min at 0.80 s each, 6.59 CPU hours logged. Analysis and bootstrap: under 0.3 CPU hour. One load
pause of 2 minutes in launch 1; no memory stop. The run came in far under the 55–65 CPU-hour projection: UNSEEN took about 14.
SEEN run (6 threads + 2 workers, 06:43–09:32): 11,842 examples in 2.8 h, 14.33 CPU hours logged, no pause. Tables on SEEN and part b
(22:08–22:28, 2 threads): about 0.7 CPU hour. Part c catalogue checks (22:36–22:44): about 0.4 CPU hour. **Total for S6: about 29.5 CPU hours** against a 55–65 projection.

