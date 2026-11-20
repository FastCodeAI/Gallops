<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H04 · Training the scorer on the sub-scores closes the base's NAVSIM gap

**Navvrat, Siddesh** · NAVSIM, Bench2Drive, HUGSIM, KITScenes, nuScenes, neuroNCAP, AlpaSim · ⏳ running · 22 Oct 2026 · hypothesis

[← all experiments](../../README.md#hypothesis-experiments) · [the account](README.md) · [experiment.toml](experiment.toml)

**Priority:** 4 · **Builds on:** [H01](../H01_four_camera_base/CARD.md)

## Question

The 4-camera base trails released Drive-JEPA on navtest (0.820 against 0.845 on Navvrat's eval), which fails the gate's NAVSIM bar (0.007). 72% of its failing scenes are a wrong pick with a good plan available, and drivable area is about 1.4 points of the gap. The released recipe trains the scorer on the final score alone. Does training it on the sub-scores close the gap?

## Hypothesis

Retraining the base's scorer with NAVSIM's sub-scores switched on, targeting drivable area, closes the base's NAVSIM gap to released Drive-JEPA. Refuted if the retrained base still fails the gate's NAVSIM pair against released Drive-JEPA: the item decides whether the base passes the gate on NAVSIM.

## Predictions

⚠️ Not pre-registered: the sub-score agent-look model (called Run 3A while it ran) followed a spec (the user's of 23 Oct, revision 2) that is not committed, the branch has no PREDICTIONS.md, and its mapping to this item is an inference

## What we learned

So far, the sub-score agent-look model narrowed the base's navtest gap (EPDMS 85.27 -> 87.30, paired +2.03 [+1.65, +2.44]) but stays behind released Drive-JEPA (87.78; paired -0.49 [-0.84, -0.11]); drivable area rose .972 -> .984 (released .986). It also added an agent decoder and an agent look in place of the global look, so the gain is not the sub-score heads' alone; no scorer-only arm is recorded. The regression gate, run by hand from local paddock worktrees, says HOLD on the gate's current subsets: KITScenes and nuScenes GAIN against released Drive-JEPA; NAVSIM, Bench2Drive (7 routes, command fix: 40.80 against 59.04), neuroNCAP and AlpaSim NEUTRAL; nothing regressed. Against the 4-camera planner every pair is NEUTRAL except Bench2Drive, a REGRESSION (80.24 -> 40.80, t -2.57).

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H04.agent-decoder**<br>the agent decoder that the sub-score agent-look model loads: a sweep of slots x layers; D = 300 slots x 3 layers, epoch 11 of 12 kept | scratch, on H01.four-camera-planner's frozen WA-JEPA token cache | agent decoder (DETR3D-style slots on the token rays) (4.25 M); frozen: WA-JEPA encoder (tokens cached) | navsim `navtrain minus the 51 held-out logs` (train) — agent_targets.gz beside each sample's tokens, from navtrain_agents.pkl (POINTERS.md)<br>navsim `navtrain, 51 held-out logs (sha1(log) % 20 == 0)` 4,457 samples (selection) — chose the slot count and depth, and reports the metrics below | [drivejepa_wajepa.toml#agent-decoder](../../recipes/drivejepa_wajepa.toml)<br>commit `548bb82`<br>Drive-JEPA (linhanwang/Drive-JEPA) @ `548bb82` | `{runs}/drivejepa_wajepa/agent_decoder/ckpts_D_q300_l3/`<br>sha256 (epoch_11.pt) `e887dcda20a74a043e3a7552fe27d7f34596b7a25e5c97a545b3b2421a9bae2a`<br>files: epoch_11.pt `e887dcda20a7` |
| **H04.subscore-agent-look**<br>The sub-score agent-look model (called Run 3A while it ran): agent decoder D, agent look (global look removed), 12 scorer heads on the sub-scores, tuned pick; epoch 13 of 20 | H04.agent-decoder | planner from scratch (refiner, agent look, 12 scorer heads, map auxiliary targets), agent decoder D (detection loss, weight 1) (13.22 M); frozen: WA-JEPA encoder (tokens cached) | navsim `navtrain minus the 51 held-out logs (1,141 logs)` 98,831 samples (train)<br>navsim `teacher list: Drive-JEPA's v2 anchors minus those failing rear safety` 103,288 samples (train) — anchors_scores_index_v2_rearok_lane, beside the metric cache (POINTERS.md)<br>navsim `lane centrelines of the four nuPlan maps, for the labels` 4 maps (train) — one pickle per map (POINTERS.md)<br>navsim `navtrain, 51 held-out logs` (selection) — chose the epoch, then the pick weights<br>navsim `navtest` 12,146 scenes (eval) — never used to choose anything<br>bench2drive `50 routes` 50 routes (eval)<br>hugsim `419 scenarios` 419 scenarios (eval) | [drivejepa_wajepa.toml#run3a-train](../../recipes/drivejepa_wajepa.toml)<br>commit `548bb82`<br>Drive-JEPA (linhanwang/Drive-JEPA) @ `548bb82` | `{weights}/wajepa_dj_run3a/run3a/step_000021616/`<br>sha256 (epoch_13.ckpt) `92185265cee95383dd62fbba760d3a5c3e4d8e8d09de40cbb8f7523190b2cf24`<br>files: epoch_13.ckpt `92185265cee9` |

- **H04.agent-decoder:** † Mapped to H04 by inference: it exists to feed the sub-score agent-look model's agent look. Sweep on the 51 held-out logs (22-23 Oct): 64 x 3, 128 x 3, 64 x 6, 128 x 6, 300 x 3. AdamW 2e-4, weight decay 0.01, batch 64, fp16, cosine over 12 epochs, gradient clip 35. Targets: every vehicle, pedestrian and cyclist within 32 m at time t, nearest 64 kept; navtest never read. D trained 23 Oct 15:56-17:11; its two files are the ones it trained with (verified by timestamps and the code-agent-decoder tree's diff). Checkpoint {runs}/drivejepa_wajepa/agent_decoder/ckpts_D_q300_l3/epoch_11.pt, no sha256 recorded. Trained in a Drive-JEPA checkout at upstream 548bb82 with the overlay, so train_git_sha is that upstream commit. No paddock adapter loads this checkpoint on its own, so `model` names the decoder; it sits at the training site in {data.navsim.exp}/agent_decoder/ckpts_23oct_D_q300_l3/.
- **H04.subscore-agent-look:** † Mapped to H04 by inference: the branch never names the hypothesis, and the sub-score agent-look model changes more than the scorer (agent decoder and agent look in place of the global look), so it is not a clean test of the sub-score claim. Spec: the user's of 23 Oct (revision 2), not committed. Trained 24 Oct 17:34 to 25 Oct 10:21; 1,544 steps an epoch at about 2.0 s a step. Epoch 13 chosen by val/score_epoch on the 51 held-out logs (0.9173; epoch 19: 0.9102) under the default pick (gates 1, route 2, no blend); a grid on epoch 13 then locked gates (0.5, 0.5, 1, 0.5, 0.5), route 2, blend 0.5 on 25 Oct (0.9204 against 0.9174; best of 32: 0.9611). drive_jepa_config.py as at launch was not kept (as_run_sha256 null). Closed-loop runs drive with WAJEPA_DJ_FOLLOW=straight. The sub-score agent-look model's checks 1-3 pass from this repo (27 Oct). Trained in a Drive-JEPA checkout at upstream 548bb82 with the overlay, so train_git_sha is that upstream commit. The checkpoint sits at the training site in {data.navsim.exp}/run3a/epoch_ckpts_20261024_1734/; run3a-checkpoint has not published it, so it has no sidecar yet.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H04.agent-decoder | offline agent detection recall at 2 m (held-out logs) 0.497 (n=4,457 samples)<br>precision 0.748 at 2 m; recall of moving vehicles within 20 m 0.849; epoch 11 of D | — | ●●○○ moderate · selected-on-report, adapted | experiments/H04_subscore_scorer_navsim/README.md, 'The agent decoder' |
| H04.subscore-agent-look ★ | NAVSIM navtest EPDMS, human filter, locked pick 87.78 → **87.30** (n=12,146 scenes)<br>full navtest, no momentum-aware selection; paired per scene -0.49 [-0.84, -0.11]; CSV score 84.09 against 84.49. The pick was tuned on navtrain's 51 held-out logs, not on navtest | released Drive-JEPA, same evaluation | ●●○○ moderate · adapted, open-loop | experiments/H04_subscore_scorer_navsim/README.md (navtest-score, verified 27 Oct) |
| H04.subscore-agent-look | NAVSIM navtest EPDMS, human filter, locked pick 85.27 → 87.30 (n=12,146 scenes)<br>paired per scene +2.03 [+1.65, +2.44] | H01.four-camera-planner | ●●○○ moderate · adapted, open-loop | experiments/H04_subscore_scorer_navsim/README.md |
| H04.subscore-agent-look | NAVSIM navtest EPDMS, human filter, default pick 87.78 → 87.00 (n=12,146 scenes)<br>the pick training and checkpoint selection used (gates 1, route 2, no blend); CSV score 83.89; no paired interval reported | released Drive-JEPA, same evaluation | ●●○○ moderate · adapted, open-loop | experiments/H04_subscore_scorer_navsim/README.md |
| H04.subscore-agent-look | NAVSIM navtest drivable area (DAC), locked pick .972 → .984 (n=12,146 scenes)<br>the hypothesis' target sub-score; released .986. The sub-score agent-look model / the 4-camera planner: NC .986 / .980, EP .900 / .879, TTC .981 / .973, LK .977 / .967, EC .406 / .401; no per-sub-score test reported | H01.four-camera-planner | ●●○○ moderate · adapted, open-loop | experiments/H04_subscore_scorer_navsim/README.md |
| H04.subscore-agent-look | Bench2Drive mean driving score 57.60 → 63.29 (n=50 routes)<br>50 routes, 2 Hz, the command fix, one run per route (26 Oct); zero-shot; RC 79.02, 30 routes completed; the 4-camera planner in the same setup 51.70 / 51.95 (two runs); runs are not repeatable (the 4-camera planner's two differ by a mean of 9.1 DS per route) | released Drive-JEPA, 2 Hz | ●●○○ moderate | experiments/H04_subscore_scorer_navsim/README.md |
| H04.subscore-agent-look | HUGSIM HD-Score 33.8 → 40.4 (n=419 scenarios)<br>419 scenarios, command only, one run per scenario; paired +6.6 [+3.6, +9.7]. Both numbers are from the 22 Oct HUGSIM campaign, which paddock's docs/benchmarks/hugsim.md marks superseded: six waypoints (3 s) of each plan sent, the released model with momentum off and uncropped, its record kept valid: false; this model's run is not stored in paddock (no paddock adapter on main). Paddock's current released Drive-JEPA figure, with momentum on and the whole 4 s plan sent, is 37.1. KITTI-360 and Waymo have no rear camera, so the sub-score agent-look model drove those with a black rear image | released Drive-JEPA | ●●○○ moderate · no-record | experiments/H04_subscore_scorer_navsim/README.md |

## Regression gate

**H04.subscore-agent-look** — 20 Nov 2026, all tier, against released Drive-JEPA (the queue's stored runs): ⏸ **HOLD**

| Benchmark | Metric | Baseline | Candidate | t | MDE | Units | Verdict |
|---|---|---|---|---|---|---|---|
| KITScenes | ADE (3 s), m | 2.5375 | 2.1215 | 2.10 | 0.559 | 117 | 🟢 GAIN |
| nuScenes | L2 averaged over 1, 2 and 3 s, m | 2.4314 | 2.1509 | 2.19 | 0.361 | 143 | 🟢 GAIN |
| NAVSIM | EPDMS | 0.8572 | 0.8534 | -0.43 | 0.026 | 126 | ⚪ NEUTRAL |
| neuroNCAP | NCAP score, quick check (3 pairs) | 3.1794 | 4.1071 | 1.13 | 4.398 | 3 | ⚪ NEUTRAL |
| Bench2Drive | driving score, crash check (route 3717) | 37.6700 | 48.0000 |  |  | 1 | ▫️ TOO FEW UNITS |
| AlpaSim | scene score, crash check (clip 04749bb9) | 0.0000 | 0.0000 |  |  | 1 | ▫️ TOO FEW UNITS |
| neuroNCAP | NCAP score, closed loop (15 pairs) | 3.6792 | 3.6915 | 0.05 | 0.801 | 15 | ⚪ NEUTRAL |
| Bench2Drive | driving score, closed loop (7 routes) | 59.0360 | 40.8048 | -1.14 | 53.853 | 7 | ⚪ NEUTRAL |
| AlpaSim | scene score, closed loop (10 scenes) | 0.4162 | 0.4391 | 0.11 | 0.647 | 10 | ⚪ NEUTRAL |

Run by hand from local paddock worktrees (the queue only runs adapters on a pushed branch), every pair on the gate's current subsets (paddock #105, 23 Oct; the quick checks #113, 26 Oct) against the queue's stored released Drive-JEPA runs, keep-going. Where each pair comes from: KITScenes and nuScenes 28 Oct (manifest wajepa_dj_run3a-openloop); neuroNCAP, quick and closed, 27 Oct (wajepa_dj_run3a-ncap and -ncapq2 for the quick pairs' rerun); AlpaSim, quick and closed, 27 Oct (wajepa_dj_run3a-alpasim2, worktree gate-oct27-alp at 9819e36); NAVSIM 20 Nov (wajepa_dj_run3a-current); Bench2Drive, quick and closed, 20 Nov (wajepa_dj_run3a_b2d-cmdfix, the model with the command fix, FOLLOW_REFERENCE fed as STRAIGHT, WAJEPA_DJ_FOLLOW=straight in its model.yaml). AlpaSim: the gate printed CONFIG DIFFERS on bridge_args, driver_config and driver_profile (this model drove AutoE2E's driver profile: six pinhole cameras, front f 990.8 px, 1920 x 1018, no distortion, global shutter, against Drive-JEPA's front camera at f 1545 px, 1920 x 1080, distortion, rolling shutter; and a 0.5 s newest-frame replan wrapper against drivejepa-0.1.4b's 11-frame planner; an uncommitted edit to paddock's AlpaSim harness, no paddock ref carries it); its two pairs are the same runs re-paired with those three keys allowed (tools.gate compare --allow-config), and the effect of each difference is not measured. Superseded by these pairs, as the gate redrew its subsets on 23 Oct: the 25 Oct gate on paddock b507811's subsets (NAVSIM 1,000 tokens, KITScenes 140 scenes, nuScenes 30 scenes, Bench2Drive 7 of the 50 routes picked by the baselines' spread, at 2 Hz, without the command fix): NAVSIM 0.875 -> 0.879 (t 0.55), KITScenes 3.222 -> 2.159 (t 5.54, GAIN), nuScenes 2.497 -> 1.914 (t 2.98, GAIN), Bench2Drive 58.801 -> 60.391 (t 0.13), all HOLD. Quick-tier crash checks are listed before the closed-loop pairs of the same benchmark. Against H01.four-camera-planner's runs on the same subsets (tools.gate compare): NAVSIM 0.844 -> 0.853 (t 1.22, NEUTRAL), KITScenes 1.969 -> 2.159 (all 223 keyframes, t -1.62, NEUTRAL), nuScenes 2.057 -> 2.151 (t -1.06, NEUTRAL), neuroNCAP 3.280 -> 3.691 (t 0.99, NEUTRAL), AlpaSim 0.450 -> 0.439 (t -0.08, NEUTRAL), Bench2Drive 80.24 -> 40.80 (t -2.57, REGRESSION).

## Checks that decided something

- **Is the sub-score agent-look model's overlay the code that trained?** (27 Oct): verified file by file: no file of the package, its configs or the Lightning module changed between the launch (24 Oct 17:34) and the overlay, except drive_jepa_config.py (25 Oct: the locked pick), whose launch version was not kept
- **The sub-score agent-look model's spec checks 1-3, synthetic inputs** (27 Oct): pass from this repo with the 24 Oct numbers: transform error 1.8e-15 m, 768 / 768 waypoints, attention on nothing 1.0000; checks 4-7 ran on 24 Oct but their scripts are not in the repo
- **Regression gate on the sub-score agent-look model**: run: see H04.subscore-agent-look's gate blocks. Overall HOLD on 25, 27 and 28 Oct; no pair regressed. Paddock's git log holds no wajepa_dj_run3a adapter through 5 Nov, because the gate ran from local worktrees with the adapter uncommitted.

## Evidence outside this repo

- Paddock's wajepa_dj_run3a adapter: on 27 Oct only uncommitted files in one local paddock worktree (rerank-native), per experiments/H04_subscore_scorer_navsim/README.md; no paddock ref carries it through 5 Nov, so the gate ran from local worktrees (see [[runs.gate]]).
- The agent decoder's and the sub-score agent-look model's epoch checkpoints, the lane centrelines and the rear-safe teacher list are off-repo at the training site ({runs}/drivejepa_wajepa/, beside the metric cache); POINTERS.md lists them with bytes and sha256.
