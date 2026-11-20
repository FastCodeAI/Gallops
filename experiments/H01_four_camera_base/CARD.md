<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H01 · The 4-camera stack as the base: WA-JEPA's frozen encoder under Drive-JEPA's planner

**Navvrat** · NAVSIM, Bench2Drive, KITScenes, nuScenes, neuroNCAP, AlpaSim · ⏳ running · 16 Oct 2026 · hypothesis

[← all experiments](../../README.md#hypothesis-experiments) · [the account](README.md) · [experiment.toml](experiment.toml)

**Priority:** 1 · **Evidence for:** D11 ([decision log](../README.md#decision-log))

## Question

Released Drive-JEPA reads only its front camera. Navvrat's 4-camera stack puts Drive-JEPA's planner (path reads plus a global read over all four cameras) on WA-JEPA's frozen encoder over front-left, front, front-right and back. The team agreed it as the base on 19 Oct; the 21 Oct list put gating it first, because every later result needs that baseline. Is it at least as good as released Drive-JEPA where the gate measures?

## Hypothesis

Replacing Drive-JEPA's front-camera ViT-L with WA-JEPA's frozen encoder over four cameras, under Drive-JEPA's planner with path reads and a global read, gives a base at least as good as released Drive-JEPA. Refuted if paddock's regression gate, base against released Drive-JEPA, reports a REGRESSION; the list's first step is the quick tier (NAVSIM, KITScenes, nuScenes, neuroNCAP 0103).

## Predictions

⚠️ Not pre-registered: the 4-camera planner (called Run 2 while it ran) trained 16-17 Oct, before the 21 Oct hypothesis list; its spec (the user's of 15-16 Oct) is not committed and the branch has no PREDICTIONS.md

## What we learned

So far, the 4-camera planner trails released Drive-JEPA on navtest (EPDMS 85.27 against 87.78, human filter), the benchmark that trained it, and on Bench2Drive's 50 routes zero-shot (DS 51.70 and 51.95 against 57.60, one run per route). Its extra inputs are used on NAVSIM (page, no-record). The regression gate, run by hand from local paddock worktrees because no paddock ref carries the adapter, says HOLD on the gate's current subsets: KITScenes and nuScenes GAIN against released Drive-JEPA; NAVSIM, Bench2Drive (7 routes, command fix: 80.24 against 59.04), neuroNCAP and AlpaSim NEUTRAL; nothing regressed.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H01.four-camera-planner**<br>The 4-camera planner (called Run 2 while it ran): Drive-JEPA's planner from scratch on frozen WA-JEPA tokens (b0 f0 l0 r0), footprint + global look; epoch 19 of 20 kept | scratch: no released weights loaded (checkpoint_path ''); WA-JEPA encoder frozen | planner (refiner and scorer), token projector Linear(1024, 256) + camera embedding, global look (7.8 M); frozen: WA-JEPA encoder (V-JEPA 2.1 ViT-L/16, tokens cached) | navsim `navtrain train_logs (Drive-JEPA's own split)` 85,109 scenes (train) — the token cache of all 103,288 navtrain samples, this split and val_logs together; the digest is over its sorted log/token lines (POINTERS.md)<br>navsim `navtrain val_logs (Drive-JEPA's own split)` 18,179 scenes (train) — only logged val/score_epoch; the last epoch was kept, so the recipe declares both as training data<br>navsim `navtest` 12,146 scenes (eval) | [drivejepa_wajepa.toml#run2-train](../../recipes/drivejepa_wajepa.toml)<br>commit `548bb82`<br>Drive-JEPA (linhanwang/Drive-JEPA) @ `548bb82` | `{weights}/wajepa_dj/run2/step_000026580/`<br>sha256 (epoch_19.ckpt) `f8ce667cdcd7233173759dde4b1c1ed0f9c1921608192ec92878db1fc73cf5ad`<br>files: epoch_19.ckpt `f8ce667cdcd7` |

- **H01.four-camera-planner:** Trained 16 Oct 17:48 to 17 Oct 05:33 (experiment wajepa_dj_v2): batch 64, Adam 1e-4, 16-mixed, seed 0, 1.65 s a step. Token dump 16 Oct: navtrain's 103,288 scenes, 0 errors, 5.43 h. The last epoch is kept, not the best (val/score_epoch 0.880 at epoch 19, 0.882 at 17). WA-JEPA code: AFARI-Research/WA-JEPA at bec2966. Trained in a Drive-JEPA checkout at upstream 548bb82 with the overlay: gallops had no trainer then, so train_git_sha is that upstream commit. The checkpoint sits at the training site in {data.navsim.exp}/wajepa_dj_v2/epoch_ckpts_20261016_1748/; run2-checkpoint has not published it, so it has no sidecar yet.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H01.four-camera-planner ★ | NAVSIM navtest EPDMS, human filter 87.78 → **85.27** (n=12,146 scenes)<br>full navtest, no momentum-aware selection; CSV score 81.98 against 84.49. Same run_pdm_score.py and metric cache, released run on a clean upstream tree. Sub-scores the 4-camera planner / released: EC .401 / .478, DAC .972 / .986, EP .879 / .891. The 17 Oct evaluation is labelled 'encoder possibly selected on navtest' | released Drive-JEPA, same evaluation | ●●○○ moderate · adapted, open-loop | experiments/H01_four_camera_base/README.md (navtest-score from the evaluations' CSVs, verified 27 Oct) |
| H01.four-camera-planner | NAVSIM navtest EPDMS, Navvrat's eval 0.845 → 0.820<br>higher on 3,317 scenes, lower on 4,119, the same on 4,710; biggest gaps extended comfort -0.077, drivable area -0.014. Agrees to three digits with the doc's CSV-score column (81.98 / 84.49); that it is the same evaluation is inferred | released Drive-JEPA, same eval | ○○○○ unverified · adapted, open-loop, no-record | team gap-review page (22 Oct): Navvrat's dashboard, checkpoint wajepa_dj_v2 epoch 19 |
| H01.four-camera-planner | Bench2Drive mean driving score 57.60 → 51.70 (n=50 routes)<br>50 routes, 2 Hz, the command fix, one run per route; zero-shot (a NAVSIM-trained planner in CARLA). A second command-fix run gave 51.95 (the two dated 23-24 Oct in this folder's README); 50.07 on 18 Oct before the fix. The two command-fix runs differ by a mean of 9.1 DS per route | released Drive-JEPA, 2 Hz | ●●○○ moderate | experiments/H01_four_camera_base/README.md |

## Regression gate

**H01.four-camera-planner** — 20 Nov 2026, all tier, against released Drive-JEPA (the queue's stored runs): ⏸ **HOLD**

| Benchmark | Metric | Baseline | Candidate | t | MDE | Units | Verdict |
|---|---|---|---|---|---|---|---|
| KITScenes | ADE (3 s), m | 2.5375 | 2.0010 | 3.10 | 0.489 | 117 | 🟢 GAIN |
| nuScenes | L2 averaged over 1, 2 and 3 s, m | 2.4314 | 2.0570 | 3.41 | 0.309 | 143 | 🟢 GAIN |
| NAVSIM | EPDMS | 0.8572 | 0.8440 | -1.28 | 0.029 | 126 | ⚪ NEUTRAL |
| neuroNCAP | NCAP score, quick check (3 pairs) | 3.1794 | 4.0237 | 0.98 | 4.609 | 3 | ⚪ NEUTRAL |
| Bench2Drive | driving score, crash check (route 3717) | 37.6700 | 29.2020 |  |  | 1 | ▫️ TOO FEW UNITS |
| AlpaSim | scene score, crash check (clip 04749bb9) | 0.0000 | 0.0000 |  |  | 1 | ▫️ TOO FEW UNITS |
| neuroNCAP | NCAP score, closed loop (15 pairs) | 3.6792 | 3.2800 | -0.96 | 1.254 | 15 | ⚪ NEUTRAL |
| Bench2Drive | driving score, closed loop (7 routes) | 59.0360 | 80.2351 | 1.08 | 65.786 | 7 | ⚪ NEUTRAL |
| AlpaSim | scene score, closed loop (10 scenes) | 0.4162 | 0.4503 | 0.24 | 0.441 | 10 | ⚪ NEUTRAL |

Run by hand from local paddock worktrees (the queue only runs adapters on a pushed branch), every pair on the gate's current subsets (paddock #105, 23 Oct; the quick checks #113, 26 Oct) against the queue's stored released Drive-JEPA runs, keep-going. Where each pair comes from: KITScenes and nuScenes 28 Oct (manifest wajepa_dj-openloop); neuroNCAP, quick and closed, 27 Oct (wajepa_dj-ncap); AlpaSim, quick and closed, 27 Oct (wajepa_dj-alpasim2, worktree gate-oct27-alp at 9819e36); NAVSIM 20 Nov (wajepa_dj-current); Bench2Drive, quick and closed, 20 Nov (wajepa_dj_b2d-cmdfix, the model with the command fix, FOLLOW_REFERENCE fed as STRAIGHT, WAJEPA_DJ_FOLLOW=straight in its model.yaml). AlpaSim: the gate printed CONFIG DIFFERS on bridge_args, driver_config and driver_profile (this model drove AutoE2E's driver profile: six pinhole cameras, front f 990.8 px, 1920 x 1018, no distortion, global shutter, against Drive-JEPA's front camera at f 1545 px, 1920 x 1080, distortion, rolling shutter; and a 0.5 s newest-frame replan wrapper against drivejepa-0.1.4b's 11-frame planner; an uncommitted edit to paddock's AlpaSim harness, no paddock ref carries it); its two pairs are the same runs re-paired with those three keys allowed (tools.gate compare --allow-config), and the effect of each difference is not measured. Superseded by these pairs, as the gate redrew its subsets on 23 Oct: the 21 Oct gate on paddock b507811's subsets (NAVSIM 1,000 tokens, KITScenes 140 scenes, nuScenes 30 scenes, Bench2Drive 7 of the 50 routes picked by the baselines' spread, at 2 Hz, without the command fix): NAVSIM 0.875 -> 0.863 (t -1.29), KITScenes 3.222 -> 1.969 (t 6.44, GAIN), nuScenes 2.497 -> 1.956 (t 3.03, GAIN), Bench2Drive 58.801 -> 62.757 (t 0.36), all HOLD. Quick-tier crash checks are listed before the closed-loop pairs of the same benchmark.

## Checks that decided something

- **Are the extra inputs used? (open loop, passing navtest scenes)**: global read off -0.038, back camera zeroed -0.025, ego-speed noise -0.027 (no-record: Navvrat's numbers, not re-run by the page's authors) → the back camera stays in the agreed base (19 Oct)
- **Global read off, NAVSIM and Bench2Drive**: NAVSIM 500 scenes 0.870 -> 0.852 (-0.018, 95% CI -0.038 to -0.000), mostly lane keeping (0.978 -> 0.946); Bench2Drive, all 25 routes, 51.1 -> 50.4 (-0.7, CI -12.6 to +11.0), no detectable effect (no-record) → the side-crash question (global read on vs off on neuroNCAP 0103, 0110) became priority 6
- **Why the base fails on navtest (failure report)** (19 Oct): 72% of the stack's failing navtest scenes are a wrong pick with a good plan available; in every largest-difference scene both models had a perfect plan among their 32 (no-record) → the scorer is the weak part: priority 4, train the scorer on the sub-scores (H04)
- **Regression gate, quick tier on the base (the list's item 1)**: run: see H01.four-camera-planner's gate blocks. Overall HOLD on 21, 27 and 28 Oct; no pair regressed. The team's page of 22 Oct says it had not run, and paddock's git log holds no wajepa_dj adapter through 5 Nov, because the gate ran from local worktrees with the adapter uncommitted.

## Evidence outside this repo

- Navvrat's navtest dashboard and 19 Oct failure report (checkpoint wajepa_dj_v2 epoch 19), quoted on the team's gap-review page (updated 22 Oct); not committed to any repository.
- Paddock's wajepa_dj adapter: on 27 Oct only uncommitted files in one local paddock worktree (rerank-native), per experiments/H01_four_camera_base/README.md; no paddock ref carries it through 5 Nov, so the gate ran from local worktrees (see [[runs.gate]]).
- The token cache (about 420 GB, {data.wajepa_tokens}) and the epoch checkpoints ({runs}/drivejepa_wajepa/run2/epoch_ckpts) are off-repo at the training site; POINTERS.md lists them with bytes and sha256 (the token cache by its sample list).
