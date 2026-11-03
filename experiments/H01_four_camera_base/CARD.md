<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H01 · The 4-camera stack as the base: WA-JEPA's frozen encoder under Drive-JEPA's planner

**Navvrat** · NAVSIM, Bench2Drive · ⏳ running · 16 Oct 2026 · hypothesis

[← all experiments](../../README.md#hypothesis-experiments) · [the account](https://github.com/FastCodeAI/Gallops/blob/drivejepa-wajepa/run3a/docs/drivejepa_wajepa.md) (on branch ⎇) · [experiment.toml](experiment.toml)

> The record is on branch `drivejepa-wajepa/run3a`, not merged yet; its links go to that branch.

**Priority:** 1 · **Evidence for:** D11 ([decision log](../../README.md#decision-log))

## Question

Released Drive-JEPA reads only its front camera. Navvrat's 4-camera stack puts Drive-JEPA's planner (path reads plus a global read over all four cameras) on WA-JEPA's frozen encoder over front-left, front, front-right and back. The team agreed it as the base on 19 Oct; the 21 Oct list put gating it first, because every later result needs that baseline. Is it at least as good as released Drive-JEPA where the gate measures?

## Hypothesis

Replacing Drive-JEPA's front-camera ViT-L with WA-JEPA's frozen encoder over four cameras, under Drive-JEPA's planner with path reads and a global read, gives a base at least as good as released Drive-JEPA. Refuted if paddock's regression gate, base against released Drive-JEPA, reports a REGRESSION; the list's first step is the quick tier (NAVSIM, KITScenes, nuScenes, neuroNCAP 0103).

## Predictions

⚠️ Not pre-registered: Run 2 trained 16-17 Oct, before the 21 Oct hypothesis list; its spec (the user's of 15-16 Oct) is not committed and the branch has no PREDICTIONS.md

## What we learned

So far, Run 2 trails released Drive-JEPA on navtest (EPDMS 85.27 against 87.78, human filter), the benchmark that trained it, and on Bench2Drive's 50 routes zero-shot (DS 51.70 and 51.95 against 57.60, one run per route). Its extra inputs are used on NAVSIM (page, no-record). The gate has not run on it: paddock has no wajepa_dj adapter on any branch through 3 Nov.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H01.run2**<br>Run 2: Drive-JEPA's planner from scratch on frozen WA-JEPA tokens (b0 f0 l0 r0), footprint + global look; epoch 19 of 20 kept | scratch: no released weights loaded (checkpoint_path ''); WA-JEPA encoder frozen | planner (refiner and scorer), token projector Linear(1024, 256) + camera embedding, global look (7.8 M); frozen: WA-JEPA encoder (V-JEPA 2.1 ViT-L/16, tokens cached) | navsim `navtrain train_logs (Drive-JEPA's own split)` 85,109 scenes (train)<br>navsim `navtrain val_logs (Drive-JEPA's own split)` 18,179 scenes (train) — only logged val/score_epoch; the last epoch was kept, so the recipe declares both as training data<br>navsim `navtest` 12,146 scenes (eval) | [drivejepa_wajepa.toml#run2-train](https://github.com/FastCodeAI/Gallops/blob/drivejepa-wajepa/run3a/recipes/drivejepa_wajepa.toml)<br>Drive-JEPA (linhanwang/Drive-JEPA) @ `548bb82` | `{runs}/drivejepa_wajepa/run2/epoch_ckpts/`<br>sha256 not recorded |

- **H01.run2:** Trained 16 Oct 17:48 to 17 Oct 05:33 (experiment wajepa_dj_v2): batch 64, Adam 1e-4, 16-mixed, seed 0, 1.65 s a step. Token dump 16 Oct: navtrain's 103,288 scenes, 0 errors, 5.43 h. The last epoch is kept, not the best (val/score_epoch 0.880 at epoch 19, 0.882 at 17). WA-JEPA code: AFARI-Research/WA-JEPA at bec2966. Whether run2-checkpoint has published it into {weights}/wajepa_dj/run2/, and the weights' sha256, are not recorded on the branch.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H01.run2 ★ | NAVSIM navtest EPDMS, human filter 87.78 → **85.27** (n=12,146 scenes)<br>full navtest, no momentum-aware selection; CSV score 81.98 against 84.49. Same run_pdm_score.py and metric cache, released run on a clean upstream tree. Sub-scores Run 2 / released: EC .401 / .478, DAC .972 / .986, EP .879 / .891. The 17 Oct evaluation is labelled 'encoder possibly selected on navtest' | released Drive-JEPA, same evaluation | ●●○○ moderate · adapted, open-loop | docs/drivejepa_wajepa.md (navtest-score from the evaluations' CSVs, verified 27 Oct) |
| H01.run2 | NAVSIM navtest EPDMS, Navvrat's eval 0.845 → 0.820<br>higher on 3,317 scenes, lower on 4,119, the same on 4,710; biggest gaps extended comfort -0.077, drivable area -0.014. Agrees to three digits with the doc's CSV-score column (81.98 / 84.49); that it is the same evaluation is inferred | released Drive-JEPA, same eval | ○○○○ unverified · adapted, open-loop, no-record | team gap-review page (22 Oct): Navvrat's dashboard, checkpoint wajepa_dj_v2 epoch 19 |
| H01.run2 | Bench2Drive mean driving score 57.60 → 51.70 (n=50 routes)<br>50 routes, 2 Hz, the command fix, one run per route; zero-shot (a NAVSIM-trained planner in CARLA). A second command-fix run gave 51.95 (the two dated 23-24 Oct in the doc's version on branch drivejepa-wajepa/run2); 50.07 on 18 Oct before the fix. The two command-fix runs differ by a mean of 9.1 DS per route | released Drive-JEPA, 2 Hz | ●○○○ argued | docs/drivejepa_wajepa.md |

## Checks that decided something

- **Are the extra inputs used? (open loop, passing navtest scenes)**: global read off -0.038, back camera zeroed -0.025, ego-speed noise -0.027 (no-record: Navvrat's numbers, not re-run by the page's authors) → the back camera stays in the agreed base (19 Oct)
- **Global read off, NAVSIM and Bench2Drive**: NAVSIM 500 scenes 0.870 -> 0.852 (-0.018, 95% CI -0.038 to -0.000), mostly lane keeping (0.978 -> 0.946); Bench2Drive, all 25 routes, 51.1 -> 50.4 (-0.7, CI -12.6 to +11.0), no detectable effect (no-record) → the side-crash question (global read on vs off on neuroNCAP 0103, 0110) became priority 6
- **Why the base fails on navtest (failure report)** (19 Oct): 72% of the stack's failing navtest scenes are a wrong pick with a good plan available; in every largest-difference scene both models had a perfect plan among their 32 (no-record) → the scorer is the weak part: priority 4, train the scorer on the sub-scores (H04)
- **Regression gate, quick tier on the base (the list's item 1)**: not run as of the page (22 Oct). Paddock's git log, all refs through 3 Nov, holds no wajepa_dj adapter and no gate run of it; the page adds that its adapter predates the camera fix (#62), so it would be blind on neuroNCAP until rebased

## Evidence outside this repo

- Navvrat's navtest dashboard and 19 Oct failure report (checkpoint wajepa_dj_v2 epoch 19), quoted on the team's gap-review page (updated 22 Oct); not committed to any repository.
- Paddock's wajepa_dj adapter: on 27 Oct only uncommitted files in one local paddock worktree (rerank-native), per docs/drivejepa_wajepa.md; no paddock ref carries it through 3 Nov.
- The token cache (about 420 GB, {data.wajepa_tokens}) and the epoch checkpoints ({runs}/drivejepa_wajepa/run2/epoch_ckpts) are off-repo at the training site; no sha256 of either is recorded on the branch.
