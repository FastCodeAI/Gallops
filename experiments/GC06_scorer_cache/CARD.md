<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# GC06 · Drive-JEPA's scorer trained on cached features, held out scene by scene

**Naren** · offline, neuroNCAP · ⏸ parked (inconclusive) · 9 Oct – 12 Oct 2026 · gap-closing

[← all experiments](../../README.md#gap-closing-experiments) · [the account](README.md) · [experiment.toml](experiment.toml)

**Builds on:** [GC03](../GC03_swerve_finetune/CARD.md) · **Evidence for:** D06, D07, D09 ([decision log](../../README.md#decision-log))

## Question

s1–s3 retrained the scorer on four rendered scenes, and its held-out ranking loss ended where a random ranking lands (3.45–3.58, against log 32 = 3.47): it memorised the scenes it saw, and each step reran the 300 M-parameter encoder. With everything upstream of the scorer frozen and cached, can Drive-JEPA's scorer learn to pick a safe plan on a scene it has never seen?

## What we learned

On camera-blind cached features, a scorer trained leave-one-scene-out ranked a held-out scene's collisions better than chance for the first time (AUC 0.66–0.83 against 0.50–0.59 released; open loop), but it learned plan geometry and ego motion, not the car. Real nuScenes data broke it either way, and its one closed-loop gain on 0103 was a 2 m drift set off by the scene, not an avoidance.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **GC06.v1**<br>cache v1: built at --batch 4, one run of scorer_folds (fixed seed) | released | scorer heads on cached features: a control (s1's target and ranking loss) and sub-score heads; frozen: encoder, queries, refiner | neuroncap-actor-insertion `round 0 (labelled_round0_v5) + GT (labelled_gt) + DAgger round 1 (labelled_round1_s2)` 21,934 ticks (train) — 450 rendered episodes × 64 plans over scenes 0053, 0058, 0064, 0070, 0075; leave-one-scene-out, each scene held out in turn | `gallops.train.scorer_cache` | — |
| **GC06.v2**<br>cache v2: rebuilt at --batch 1, measured by code on branch train/sighted-scorer | released | scorer variants on cached features (attention, plan geometry, probe paths), five seeds; frozen: encoder, queries, refiner | neuroncap-actor-insertion `round 0 (labelled_round0_v5) + GT (labelled_gt) + DAgger round 1 (labelled_round1_s2)` (train) — the same sources as v1; leave-one-scene-out | `gallops.train.scorer_cache` | — |
| **GC06.v3_nusc**<br>cache v3_nusc: real nuScenes train scenes as episodes, graded against where their vehicles really went | released | scorer variants with real scenes in training, masked from the collision label, or at 0.2 weight; frozen: encoder, queries, refiner | nuscenes `train minus every neuroNCAP scene, scene-0061 and the 5 reconstructed scenes` (train) — labelled_nusc; no inserted hazard, every annotated vehicle is graded<br>nuscenes `train, 66 held-out scenes` (held-out) — labelled_nusc_holdout | `gallops.data.nuscenes_real` | — |
| **GC06.relative-0103**<br>closed loop on 0103: the five-seed sub-score scorer may override the released pick (relative rule, margin 0.2) | released | sub-score scorer: MLP over each plan's own tokens, attention over 256 probe tokens, plan geometry; six channels (2.01165 M); frozen: all of Drive-JEPA | neuroncap `scene 0103, seeds 0-2, three runs per scenario` 9 episodes (eval) | commit `cb52979` | `{weights}/drivejepa_subscore/realmask-attngeo-v1/step_000000020/`<br>sha256 not recorded<br>⚠️ sidecar: the sidecar is not in gallops.checkpoint's format (no train_run, no weights_sha256, `unit` where a step unit goes), so `gallops experiments checkpoint` refuses it; nothing records the pick rule a run used: scorer.json gives the absolute rule, and this run drove the relative one |

- **GC06.v1:** Built by scorer_cache at gallops b3caf13; its ticks are mixed within each batch of four. 64 plans a tick: 32 native, 32 from a vocabulary rerun through the refiner's last stage. Gated pick (keep the released plan unless the collision head is under 0.5): kept safe 90–100% per held-out scene, frontal rescued 2–29%, stationary 4–48%; the two Singapore scenes (0053, 0058) are weakest. Result file dj_cache/v1/folds.json, copied to the branch record's results/v1/.
- **GC06.v2:** Rescues rise from 2–26% to 17–36% with four training scenes and have not levelled off. Frontal is limited by what the plans see: more than 2 s before contact the car is 35–45 m away, the released pick reaches 8–13 m and straight swerves 15–25 m. Result files dj_cache/v2/probe_*_s{0..4}.json and learning_curve_*.json, copied to the branch record's results/v2/.
- **GC06.v3_nusc:** With real scenes in training, the rendered frontal lesson disappears (real logs teach 'car ahead, short braking plan, safe'). With real scenes masked from the collision label, the rendered scorer works but fires on 55% of real held-out ticks and keeps only 68% of them safe. At 0.2 weight no middle ground was found. Result files probe_attngeo_real*, probe_attngeo_realmask*, probe_attngeo_realw* sit under the branch record's results/v2/. The real-nuScenes conversion itself is unaffected by the camera bug.
- **GC06.relative-0103:** Run by paddock's drivejepa_subscore adapter, neuroNCAP run 20261012T131011-c43ac5. Its run_config.json records neither the scorer directory nor the rule; these weights are identified as the ones the adapter loaded by the branch record (scripts/final_train.py, results/subscore_weights_sidecar/, byte-identical to this directory's sidecar). Trained 20 epochs on the rendered scenes (round0_v5, gt, round1_s2) and on real nuScenes train for on_road, in_lane and progress only. Five seed files, sha256 seed_0 5f113faf20fb…, seed_1 2a90cd232ae5…, seed_2 81a1813270a3…, seed_3 df4903d8cdce…, seed_4 7debbf4db701…. The absolute rule (run 20261011T160321-c43ac5) scored frontal 2.20 / 2.26 / 2.21, all hit.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| GC06.v1 ★ | offline held-out collision ranking AUC, worst of 5 folds 0.59 → **0.660** (n=5 scenes)<br>each rendered scene held out in turn; 0.660 / 0.706 / 0.695 / 0.828 / 0.751 for 0053 / 0058 / 0064 / 0070 / 0075, against 0.50–0.59 for the released final score; one seed | released final score, best of the same 5 folds | ●●○○ moderate · blind, open-loop | experiments/GC06_scorer_cache/README.md; dj_cache/v1/folds.json |
| GC06.v2 | offline held-out collision ranking AUC, trained on a single scene 0.8<br>'about 0.8'; from code not on main (train/sighted-scorer) | — | ○○○○ unverified · blind, open-loop | experiments/GC06_scorer_cache/README.md; dj_cache/v2/learning_curve_*.json |
| GC06.v3_nusc | offline override rate on real nuScenes held-out ticks (real scenes masked from the collision label) 55%<br>66 held-out real nuScenes train scenes; and keeps only 68% of them safe; from code not on main (train/sighted-scorer) | — | ○○○○ unverified · blind, open-loop | experiments/GC06_scorer_cache/README.md; dj_cache/v2/probe_attngeo_realmask* |
| GC06.relative-0103 | neuroNCAP ncap_score, frontal 4.044839 (n=3 episodes)<br>scene 0103, frontal, seeds 0-2; 2.13 / 5.0 / 5.0. Not an avoidance: every switch happens in the first 3 s, identically in the frontal, stationary and side runs, and drifts the ego about 2 m left; the oncoming car misses an ego already displaced | released (blind): 2.07 / 2.26 / 2.08 | ●○○○ argued · blind, smoke | experiments/GC06_scorer_cache/README.md; paddock branch record/drivejepa-camera-blind, results/neuroncap/drivejepa_subscore/20261012T131011-c43ac5/metrics.json |

## Checks that decided something

- **The released scorer on cached features (v1)**: reproduces its own picks on 99.62% of ticks, max logit drift 0.0017 → cached features stand in for the live forward pass
- **Does the scorer's answer change when only the hazard car changes?** (12 Oct): no: on GT drives, where 30 runs of a scene share every pose, the cached tokens were bit-identical across runs whose images differ by up to 23% of pixels; the cameras were placed relative to LIDAR_TOP, so no image reached the refiner → caches v1, v2 and v3_nusc must be rebuilt; a cache records its key (camera frame, code sha, --batch, vocabulary sha256, checkpoint, calibration source), and scorer_labels and scorer_folds refuse a cache without camera_frame: ego
- **Ticks mixed within a batch**: at batch 1 the rebuilt plans match the driven ones to 5e-5 m, at batch 4 by up to 1.0 m: the vendor's spatial cross-attention takes every element's camera queries from element 0's mask → every pass is one tick (TICKS_PER_PASS); scorer_cache, the trainer, reach and the preflight refuse --batch above 1
- **Is the 0103 closed-loop gain an avoidance?** (12 Oct): no: the relative rule's switches happen in the first 3 s with the same times, offsets and scores in frontal, stationary and side runs → any closed-loop gain must be checked for switches that differ between scenarios before it is credited

## Evidence outside this repo

- The raw record -- scripts and ops chains as run, small result JSONs, the scorer's sidecars: gallops branch record/blind-scorer-experiments, experiments/2026-10-blind-scorer/ (kept for the record, not the current state of anything).
- The v2 measurements were made by code not on main (attention and plan-geometry scorer, probe paths, real / real-mask / 0.2-weight variants, learning curves, five seeds), on gallops branch train/sighted-scorer.
- The feature caches v1, v2 and v3_nusc (25 GB), fold logits and the scorer .pt files stay off the repository at the training site.
- The closed-loop adapter drivejepa_subscore is on paddock branch model/drivejepa-subscore (no PR); its two 0103 runs are on paddock branch record/drivejepa-camera-blind.
