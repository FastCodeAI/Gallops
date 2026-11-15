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

⚠️ Not pre-registered: Run 2 trained 16-17 Oct, before the 21 Oct hypothesis list; its spec (the user's of 15-16 Oct) is not committed and the branch has no PREDICTIONS.md

## What we learned

So far, Run 2 trails released Drive-JEPA on navtest (EPDMS 85.27 against 87.78, human filter), the benchmark that trained it, and on Bench2Drive's 50 routes zero-shot (DS 51.70 and 51.95 against 57.60, one run per route). Its extra inputs are used on NAVSIM (page, no-record). The regression gate, run by hand from local paddock worktrees because no paddock ref carries the adapter, says HOLD: KITScenes and nuScenes GAIN against released Drive-JEPA, NAVSIM, Bench2Drive (7 routes, 2 Hz) and neuroNCAP NEUTRAL, nothing regressed.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H01.run2**<br>Run 2: Drive-JEPA's planner from scratch on frozen WA-JEPA tokens (b0 f0 l0 r0), footprint + global look; epoch 19 of 20 kept | scratch: no released weights loaded (checkpoint_path ''); WA-JEPA encoder frozen | planner (refiner and scorer), token projector Linear(1024, 256) + camera embedding, global look (7.8 M); frozen: WA-JEPA encoder (V-JEPA 2.1 ViT-L/16, tokens cached) | navsim `navtrain train_logs (Drive-JEPA's own split)` 85,109 scenes (train) — the token cache of all 103,288 navtrain samples, this split and val_logs together; the digest is over its sorted log/token lines (POINTERS.md)<br>navsim `navtrain val_logs (Drive-JEPA's own split)` 18,179 scenes (train) — only logged val/score_epoch; the last epoch was kept, so the recipe declares both as training data<br>navsim `navtest` 12,146 scenes (eval) | [drivejepa_wajepa.toml#run2-train](../../recipes/drivejepa_wajepa.toml)<br>commit `548bb82`<br>Drive-JEPA (linhanwang/Drive-JEPA) @ `548bb82` | `{weights}/wajepa_dj/run2/step_000000019/`<br>sha256 (epoch_19.ckpt) `f8ce667cdcd7233173759dde4b1c1ed0f9c1921608192ec92878db1fc73cf5ad`<br>files: epoch_19.ckpt `f8ce667cdcd7` |

- **H01.run2:** Trained 16 Oct 17:48 to 17 Oct 05:33 (experiment wajepa_dj_v2): batch 64, Adam 1e-4, 16-mixed, seed 0, 1.65 s a step. Token dump 16 Oct: navtrain's 103,288 scenes, 0 errors, 5.43 h. The last epoch is kept, not the best (val/score_epoch 0.880 at epoch 19, 0.882 at 17). WA-JEPA code: AFARI-Research/WA-JEPA at bec2966. Whether run2-checkpoint has published it into {weights}/wajepa_dj/run2/, and the weights' sha256, are not recorded on the branch. Trained in a Drive-JEPA checkout at upstream 548bb82 with the overlay: gallops had no trainer then, so train_git_sha is that upstream commit. The checkpoint sits at the training site in {data.navsim.exp}/wajepa_dj_v2/epoch_ckpts_20261016_1748/; run2-checkpoint has not published it, so it has no sidecar yet.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H01.run2 ★ | NAVSIM navtest EPDMS, human filter 87.78 → **85.27** (n=12,146 scenes)<br>full navtest, no momentum-aware selection; CSV score 81.98 against 84.49. Same run_pdm_score.py and metric cache, released run on a clean upstream tree. Sub-scores Run 2 / released: EC .401 / .478, DAC .972 / .986, EP .879 / .891. The 17 Oct evaluation is labelled 'encoder possibly selected on navtest' | released Drive-JEPA, same evaluation | ●●○○ moderate · adapted, open-loop | experiments/H01_four_camera_base/README.md (navtest-score from the evaluations' CSVs, verified 27 Oct) |
| H01.run2 | NAVSIM navtest EPDMS, Navvrat's eval 0.845 → 0.820<br>higher on 3,317 scenes, lower on 4,119, the same on 4,710; biggest gaps extended comfort -0.077, drivable area -0.014. Agrees to three digits with the doc's CSV-score column (81.98 / 84.49); that it is the same evaluation is inferred | released Drive-JEPA, same eval | ○○○○ unverified · adapted, open-loop, no-record | team gap-review page (22 Oct): Navvrat's dashboard, checkpoint wajepa_dj_v2 epoch 19 |
| H01.run2 | Bench2Drive mean driving score 57.60 → 51.70 (n=50 routes)<br>50 routes, 2 Hz, the command fix, one run per route; zero-shot (a NAVSIM-trained planner in CARLA). A second command-fix run gave 51.95 (the two dated 23-24 Oct in this folder's README); 50.07 on 18 Oct before the fix. The two command-fix runs differ by a mean of 9.1 DS per route | released Drive-JEPA, 2 Hz | ●●○○ moderate | experiments/H01_four_camera_base/README.md |

## Regression gate

**H01.run2** — 21 Oct 2026, quick tier, against released Drive-JEPA, the gate runs of 21 Oct (drivejepa-released-timing-20261021); Bench2Drive: its native-50 2 Hz runs merged to the gate's 7 routes: ⏸ **HOLD**

| Benchmark | Metric | Baseline | Candidate | t | MDE | Units | Verdict |
|---|---|---|---|---|---|---|---|
| NAVSIM | epdms | 0.875 | 0.863 | -1.29 | 0.027 | 136 | ⚪ NEUTRAL |
| KITScenes | ade | 3.222 | 1.969 | 6.44 |  | 140 | 🟢 GAIN |
| nuScenes | l2_avg | 2.497 | 1.956 | 3.03 |  | 30 | 🟢 GAIN |
| Bench2Drive | driving_score | 58.801 | 62.757 | 0.36 |  | 7 | ⚪ NEUTRAL |

Run by hand. Three runs of tools.gate: NAVSIM (worktree gate-wajepa), KITScenes and nuScenes (gate-wajepa-more), and Bench2Drive at the model's 2 Hz by a hand-written manifest (PADDOCK_INFER_HZ, not on paddock main); each printed overall HOLD. Subsets are marked invalid by the gate (a subset of the benchmark); neuroNCAP was blocked that day and ran on 27 Oct.

**H01.run2** — 27 Oct 2026, closed tier, against released Drive-JEPA, the paddock queue's runs of 26 Oct: ⏸ **HOLD**

| Benchmark | Metric | Baseline | Candidate | t | MDE | Units | Verdict |
|---|---|---|---|---|---|---|---|
| neuroNCAP | ncap_score | 3.679 | 3.28 | -0.96 | 1.254 | 15 | ⚪ NEUTRAL |

Run by hand: the queue only runs adapters on a pushed branch. neuroNCAP 15 pairs, one run each; its quick 3 pairs 3.179 -> 4.024, t 0.98, NEUTRAL. AlpaSim, 10 scenes (manifest wajepa_dj-alpasim2): scene_score 0.416 -> 0.450, t 0.24, MDE 0.441, gate verdict CONFIG DIFFERS (our driver profile differs from the baseline's), so it is not a pair here.

**H01.run2** — 28 Oct 2026, quick tier, against released Drive-JEPA, the paddock queue's runs of 23 Oct: ⏸ **HOLD**

| Benchmark | Metric | Baseline | Candidate | t | MDE | Units | Verdict |
|---|---|---|---|---|---|---|---|
| KITScenes | ade | 2.537 | 2.001 | 3.1 | 0.489 | 117 | 🟢 GAIN |
| nuScenes | l2_avg | 2.431 | 2.057 | 3.41 | 0.309 | 143 | 🟢 GAIN |

KITScenes and nuScenes rerun for the PDM sub-scores on the queue's newer subsets (nuScenes 1,000 samples).

## Checks that decided something

- **Are the extra inputs used? (open loop, passing navtest scenes)**: global read off -0.038, back camera zeroed -0.025, ego-speed noise -0.027 (no-record: Navvrat's numbers, not re-run by the page's authors) → the back camera stays in the agreed base (19 Oct)
- **Global read off, NAVSIM and Bench2Drive**: NAVSIM 500 scenes 0.870 -> 0.852 (-0.018, 95% CI -0.038 to -0.000), mostly lane keeping (0.978 -> 0.946); Bench2Drive, all 25 routes, 51.1 -> 50.4 (-0.7, CI -12.6 to +11.0), no detectable effect (no-record) → the side-crash question (global read on vs off on neuroNCAP 0103, 0110) became priority 6
- **Why the base fails on navtest (failure report)** (19 Oct): 72% of the stack's failing navtest scenes are a wrong pick with a good plan available; in every largest-difference scene both models had a perfect plan among their 32 (no-record) → the scorer is the weak part: priority 4, train the scorer on the sub-scores (H04)
- **Regression gate, quick tier on the base (the list's item 1)**: run: see H01.run2's gate blocks. Overall HOLD on 21, 27 and 28 Oct; no pair regressed. The team's page of 22 Oct says it had not run, and paddock's git log holds no wajepa_dj adapter through 5 Nov, because the gate ran from local worktrees with the adapter uncommitted.

## Evidence outside this repo

- Navvrat's navtest dashboard and 19 Oct failure report (checkpoint wajepa_dj_v2 epoch 19), quoted on the team's gap-review page (updated 22 Oct); not committed to any repository.
- Paddock's wajepa_dj adapter: on 27 Oct only uncommitted files in one local paddock worktree (rerank-native), per experiments/H01_four_camera_base/README.md; no paddock ref carries it through 5 Nov, so the gate ran from local worktrees (see [[runs.gate]]).
- The token cache (about 420 GB, {data.wajepa_tokens}) and the epoch checkpoints ({runs}/drivejepa_wajepa/run2/epoch_ckpts) are off-repo at the training site; POINTERS.md lists them with bytes and sha256 (the token cache by its sample list).
