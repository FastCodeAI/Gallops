<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H04 · Training the scorer on the sub-scores closes the base's NAVSIM gap

**Navvrat, Siddesh** · NAVSIM, Bench2Drive, HUGSIM · ⏳ running · 22 Oct 2026 · hypothesis

[← all experiments](../../README.md#hypothesis-experiments) · [the account](https://github.com/FastCodeAI/Gallops/blob/drivejepa-wajepa/run3a/docs/drivejepa_wajepa.md) (on branch ⎇) · [experiment.toml](experiment.toml)

> The record is on branch `drivejepa-wajepa/run3a`, not merged yet; its links go to that branch.

**Priority:** 4 · **Builds on:** [H01](../H01_four_camera_base/CARD.md)

## Question

The 4-camera base trails released Drive-JEPA on navtest (0.820 against 0.845 on Navvrat's eval), which fails the gate's NAVSIM bar (0.007). 72% of its failing scenes are a wrong pick with a good plan available, and drivable area is about 1.4 points of the gap. The released recipe trains the scorer on the final score alone. Does training it on the sub-scores close the gap?

## Hypothesis

Retraining the base's scorer with NAVSIM's sub-scores switched on, targeting drivable area, closes the base's NAVSIM gap to released Drive-JEPA. Refuted if the retrained base still fails the gate's NAVSIM pair against released Drive-JEPA: the item decides whether the base passes the gate on NAVSIM.

## Predictions

⚠️ Not pre-registered: Run 3A followed a spec (the user's of 23 Oct, revision 2) that is not committed, the branch has no PREDICTIONS.md, and its mapping to this item is an inference

## What we learned

So far, Run 3A narrowed the base's navtest gap (EPDMS 85.27 -> 87.30, paired +2.03 [+1.65, +2.44]) but stays behind released Drive-JEPA (87.78; paired -0.49 [-0.84, -0.11]); drivable area rose .972 -> .984 (released .986). It also added an agent decoder and an agent look in place of the global look, so the gain is not the sub-score heads' alone; no scorer-only arm is recorded, and the gate has not run.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H04.agent-decoder**<br>the agent decoder Run 3A loads: a sweep of slots x layers; D = 300 slots x 3 layers, epoch 11 of 12 kept | scratch, on H01.four-camera-planner's frozen WA-JEPA token cache | agent decoder (DETR3D-style slots on the token rays) (4.25 M); frozen: WA-JEPA encoder (tokens cached) | navsim `navtrain minus the 51 held-out logs` (train)<br>navsim `navtrain, 51 held-out logs (sha1(log) % 20 == 0)` 4,457 samples (selection) — chose the slot count and depth, and reports the metrics below | [drivejepa_wajepa.toml#agent-decoder](https://github.com/FastCodeAI/Gallops/blob/drivejepa-wajepa/run3a/recipes/drivejepa_wajepa.toml)<br>Drive-JEPA (linhanwang/Drive-JEPA) @ `548bb82` | — |
| **H04.run3a**<br>Run 3A: agent decoder D, agent look (global look removed), 12 scorer heads on the sub-scores, tuned pick; epoch 13 of 20 | H04.agent-decoder | planner from scratch (refiner, agent look, 12 scorer heads, map auxiliary targets), agent decoder D (detection loss, weight 1) (13.22 M); frozen: WA-JEPA encoder (tokens cached) | navsim `navtrain minus the 51 held-out logs (1,141 logs)` 98,831 samples (train)<br>navsim `navtrain, 51 held-out logs` (selection) — chose the epoch, then the pick weights<br>navsim `navtest` 12,146 scenes (eval) — never used to choose anything<br>bench2drive `50 routes` 50 routes (eval)<br>hugsim `419 scenarios` 419 scenarios (eval) | [drivejepa_wajepa.toml#run3a-train](https://github.com/FastCodeAI/Gallops/blob/drivejepa-wajepa/run3a/recipes/drivejepa_wajepa.toml)<br>Drive-JEPA (linhanwang/Drive-JEPA) @ `548bb82` | `{runs}/drivejepa_wajepa/run3a/epoch_ckpts/`<br>sha256 not recorded |

- **H04.agent-decoder:** † Mapped to H04 by inference: it exists to feed Run 3A's agent look. Sweep on the 51 held-out logs (22-23 Oct): 64 x 3, 128 x 3, 64 x 6, 128 x 6, 300 x 3. AdamW 2e-4, weight decay 0.01, batch 64, fp16, cosine over 12 epochs, gradient clip 35. Targets: every vehicle, pedestrian and cyclist within 32 m at time t, nearest 64 kept; navtest never read. D trained 23 Oct 15:56-17:11; its two files are the ones it trained with (verified by timestamps and the code-agent-decoder tree's diff). Checkpoint {runs}/drivejepa_wajepa/agent_decoder/ckpts_D_q300_l3/epoch_11.pt, no sha256 recorded.
- **H04.run3a:** † Mapped to H04 by inference: the branch never names the hypothesis, and Run 3A changes more than the scorer (agent decoder and agent look in place of the global look), so it is not a clean test of the sub-score claim. Spec: the user's of 23 Oct (revision 2), not committed. Trained 24 Oct 17:34 to 25 Oct 10:21; 1,544 steps an epoch at about 2.0 s a step. Epoch 13 chosen by val/score_epoch on the 51 held-out logs (0.9173; epoch 19: 0.9102) under the default pick (gates 1, route 2, no blend); a grid on epoch 13 then locked gates (0.5, 0.5, 1, 0.5, 0.5), route 2, blend 0.5 on 25 Oct (0.9204 against 0.9174; best of 32: 0.9611). drive_jepa_config.py as at launch was not kept (as_run_sha256 null). Closed-loop runs drive with WAJEPA_DJ_FOLLOW=straight. Run 3A's checks 1-3 pass from this repo (27 Oct).

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H04.agent-decoder | offline agent detection recall at 2 m (held-out logs) 0.497 (n=4,457 samples)<br>precision 0.748 at 2 m; recall of moving vehicles within 20 m 0.849; epoch 11 of D | — | ●●○○ moderate · selected-on-report, adapted | docs/drivejepa_wajepa.md, 'The agent decoder' |
| H04.run3a ★ | NAVSIM navtest EPDMS, human filter, locked pick 87.78 → **87.30** (n=12,146 scenes)<br>full navtest, no momentum-aware selection; paired per scene -0.49 [-0.84, -0.11]; CSV score 84.09 against 84.49. The pick was tuned on navtrain's 51 held-out logs, not on navtest | released Drive-JEPA, same evaluation | ●●○○ moderate · adapted, open-loop | docs/drivejepa_wajepa.md (navtest-score, verified 27 Oct) |
| H04.run3a | NAVSIM navtest EPDMS, human filter, locked pick 85.27 → 87.30 (n=12,146 scenes)<br>paired per scene +2.03 [+1.65, +2.44] | H01.four-camera-planner | ●●○○ moderate · adapted, open-loop | docs/drivejepa_wajepa.md |
| H04.run3a | NAVSIM navtest EPDMS, human filter, default pick 87.78 → 87.00 (n=12,146 scenes)<br>the pick training and checkpoint selection used (gates 1, route 2, no blend); CSV score 83.89; no paired interval reported | released Drive-JEPA, same evaluation | ●●○○ moderate · adapted, open-loop | docs/drivejepa_wajepa.md |
| H04.run3a | NAVSIM navtest drivable area (DAC), locked pick .972 → .984 (n=12,146 scenes)<br>the hypothesis' target sub-score; released .986. Run 3A / the 4-camera planner: NC .986 / .980, EP .900 / .879, TTC .981 / .973, LK .977 / .967, EC .406 / .401; no per-sub-score test reported | H01.four-camera-planner | ●●○○ moderate · adapted, open-loop | docs/drivejepa_wajepa.md |
| H04.run3a | Bench2Drive mean driving score 57.60 → 63.29 (n=50 routes)<br>50 routes, 2 Hz, the command fix, one run per route (26 Oct); zero-shot; RC 79.02, 30 routes completed; the 4-camera planner in the same setup 51.70 / 51.95 (two runs); runs are not repeatable (the 4-camera planner's two differ by a mean of 9.1 DS per route) | released Drive-JEPA, 2 Hz | ●●○○ moderate | docs/drivejepa_wajepa.md |
| H04.run3a | HUGSIM HD-Score 33.8 → 40.4 (n=419 scenarios)<br>419 scenarios, command only, one run per scenario; paired +6.6 [+3.6, +9.7]; KITTI-360 and Waymo have no rear camera, so Run 3A drove those with a black rear image | released Drive-JEPA | ●●○○ moderate | docs/drivejepa_wajepa.md |

## Checks that decided something

- **Is the Run 3A overlay the code that trained?** (27 Oct): verified file by file: no file of the package, its configs or the Lightning module changed between the launch (24 Oct 17:34) and the overlay, except drive_jepa_config.py (25 Oct: the locked pick), whose launch version was not kept
- **Run 3A spec checks 1-3, synthetic inputs** (27 Oct): pass from this repo with the 24 Oct numbers: transform error 1.8e-15 m, 768 / 768 waypoints, attention on nothing 1.0000; checks 4-7 ran on 24 Oct but their scripts are not in the repo
- **Regression gate on Run 3A**: not run: paddock has no wajepa_dj_run3a adapter on any ref through 3 Nov

## Evidence outside this repo

- Paddock's wajepa_dj_run3a adapter: on 27 Oct only uncommitted files in one local paddock worktree (rerank-native), per docs/drivejepa_wajepa.md; no paddock ref carries it through 3 Nov, so no gate run exists.
- The agent decoder's and Run 3A's epoch checkpoints, the lane centrelines and the rear-safe teacher list are off-repo at the training site ({runs}/drivejepa_wajepa/, beside the metric cache); no sha256 is recorded on the branch.
