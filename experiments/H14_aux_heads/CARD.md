<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H14 · Auxiliary prediction heads on Drive-JEPA's refiner, fine-tuned on nuScenes train

**Shreedhar** · nuScenes · 🗑 dropped · 20 Oct 2026 · hypothesis

[← all experiments](../../README.md#hypothesis-experiments) · [experiment.toml](experiment.toml)

> The record is on commit `1c60e59`, not merged yet; its links go to that commit.

**Evidence for:** D19 ([decision log](../../README.md#decision-log))

## Question

Does training Drive-JEPA's refiner with auxiliary heads (BEV map, objects, lead vehicle, lane change; dropped at inference) give better plans than the identical fine-tune without them, and does it cost the released model anything it already does well?

## Hypothesis

H-gain: fine-tuning the planner with auxiliary losses (T) gives better plans than the identical fine-tune without them (C). Supported only if the heads learn, T beats C on held-out planning by more than C's range across its three seeds and by more than twice the scene-clustered SE of the difference, the sight check passes, and (if run) shuffled labels S do not match T. H-safe: T does not regress against the released model (R) on any benchmark where R is measured; supported only if the gate's T-vs-R check shows no REGRESSION.

## Predictions

⚠️ Not pre-registered: the doc cites experiments/drivejepa_aux/PREDICTIONS.md, which was never written; the decision rules the runs are read against are docs/drivejepa-aux-heads.md step 4, committed at 1ec1823 before any run, unchanged

## What we learned

Cancelled 4 Nov 2026 (D19): the results and their analysis are inconclusive, and future experiments do not build on this one. What it had recorded, open loop and in domain (2,996 keyframes of 102 held-out nuScenes train scenes, epoch 10): the heads move the plan the frozen scorer picks, not the proposals. T2's picked L2 at 3 s is 2.934 m against C's 3.760 and the release's 3.788, beyond C's seed range and two paired SE on every seed; best of 32 is the same in every arm. The lane-change and lane-divider heads learned nothing. Inputs not checked against paddock's, no shuffled-label arm, no gate: nothing here speaks to NAVSIM, closed loop or H-safe.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H14.C**<br>arm C: refiner fine-tune on nuScenes train, no heads (control); seeds 0, 1, 2 | released | refiner path (2.39 M); frozen: encoder, projector, scorer | nuscenes `train, 577 scenes` 16,952 keyframes (train)<br>nuscenes `train, 102 held-out scenes (15%, seed 0)` 2,996 keyframes (selection) — chose each run's kept epoch; the numbers quoted are at epoch 10, which nothing chose | [drivejepa_aux.toml#c-s0](https://github.com/FastCodeAI/Gallops/blob/1c60e59df49114e314b525987561023bb6ad5c7b/recipes/drivejepa_aux.toml)<br>commit `b48f731` | `{weights}/drivejepa_aux/aux-c-s0/`<br>sha256 not recorded |
| **H14.T1**<br>arm T1: as C, plus upstream's BEV map and object heads; seeds 0, 1, 2 | released | refiner path, BEV map head, object head; frozen: encoder, projector, scorer | nuscenes `train, 577 scenes` 16,952 keyframes (train)<br>nuscenes `train, 102 held-out scenes (15%, seed 0)` 2,996 keyframes (selection) | [drivejepa_aux.toml#t1-s0](https://github.com/FastCodeAI/Gallops/blob/1c60e59df49114e314b525987561023bb6ad5c7b/recipes/drivejepa_aux.toml)<br>commit `b48f731` | `{weights}/drivejepa_aux/aux-t1-s0/`<br>sha256 not recorded |
| **H14.T2**<br>arm T2: T1 plus lead-vehicle and lane-change heads; seeds 0, 1, 2 | released | refiner path, BEV map head, object head, lead-vehicle head, lane-change head; frozen: encoder, projector, scorer | nuscenes `train, 577 scenes` 16,952 keyframes (train)<br>nuscenes `train, 102 held-out scenes (15%, seed 0)` 2,996 keyframes (selection) | [drivejepa_aux.toml#t2-s0](https://github.com/FastCodeAI/Gallops/blob/1c60e59df49114e314b525987561023bb6ad5c7b/recipes/drivejepa_aux.toml)<br>commit `b48f731` | `{weights}/drivejepa_aux/aux-t2-s0/`<br>sha256 not recorded |

- **H14.C:** Three seeds (recipe stages c-s0, c-s1, c-s2; train runs aux-c-s0..2), 10 epochs of 529 steps at batch 32; planning loss = the Bench2Drive generator stage's mtd_loss with the release's top-2 plans as teachers; Adam 2e-5 on the refiner path. All nine runs passed the preflight identically and trained from b48f731 in a clean worktree. Numbers at the fixed epoch 10; a run's kept epoch was chosen on the held-out scenes it reports on.
- **H14.T1:** Stages t1-s0..2 (train runs aux-t1-s0..2). Heads at upstream's weights (class 1.0, box 0.1, BEV 1.0), Adam 1e-4; dropped at inference. T1 - C, L2 at 3 s, per seed: -0.46 +/- 0.12, -0.32 +/- 0.12, -0.48 +/- 0.11 (C's seed range 0.009). T1 - C collision: +0.37 +/- 0.40, +0.57 +/- 0.51, +0.20 +/- 0.39 points.
- **H14.T2:** Stages t2-s0..2 (train runs aux-t2-s0..2); new heads at weight 0.5. T2 - C per seed: L2 at 3 s -0.89 +/- 0.11, -0.78 +/- 0.11, -0.81 +/- 0.10; L2 at 1 s -0.22 +/- 0.04, -0.16 +/- 0.04, -0.18 +/- 0.03 (C's range 0.017); collision -1.14 +/- 0.45, -0.67 +/- 0.44, -0.70 +/- 0.42 points (C's range 0.23), not resolved since seed 1 lies inside two SE. Lead gap error 3.5-4.1 m against 15.2 m for the train mean; lane-change F1 0 / 0. Seed 0 is the planned gate candidate.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H14.C | nuScenes picked-plan L2 at 3 s, m 3.788 → 3.760 (n=2,996 keyframes)<br>102 held-out train scenes, every run at epoch 10; mean of three seeds (3.755-3.765); no C-vs-R test reported. Best of 32 ADE 0.197 against R's 0.268, yet C picks the proposal nearest the human on 2.4-2.8% of keyframes against R's 4.7% | R, the released model | ●●○○ moderate · adapted, open-loop | docs/drivejepa-aux-heads.md, Results (2026-10-21) |
| H14.T1 | nuScenes picked-plan L2 at 3 s, m 3.760 → 3.340 (n=2,996 keyframes)<br>102 held-out train scenes, every run at epoch 10; mean of three seeds (3.283-3.444); paired T1 - C beyond C's seed range and two SE on every seed (in the run notes). Best of 32: 0.199 against C's 0.197 | H14.C | ●●○○ moderate · adapted, open-loop | docs/drivejepa-aux-heads.md, Results (2026-10-21) |
| H14.T2 ★ | nuScenes picked-plan L2 at 3 s, m 3.760 → **2.934** (n=2,996 keyframes)<br>102 held-out train scenes, every run at epoch 10; mean of three seeds (2.863-2.989); paired T2 - C -0.89 / -0.78 / -0.81 m by seed, beyond C's seed range (0.009) and two SE on every seed. Released model 3.788. Best of 32 unchanged (0.198 against C's 0.197): the change is in the pick | H14.C | ●●○○ moderate · adapted, open-loop | docs/drivejepa-aux-heads.md, Results (2026-10-21) |
| H14.T2 | nuScenes collision within 3 s, % 3.53 → 2.69 (n=2,996 keyframes)<br>102 held-out train scenes, every run at epoch 10; mean of three seeds (2.47-2.94); not resolved: seed 1's paired difference lies inside two SE. Released model 4.04 | H14.C | ○○○○ unverified · adapted, open-loop | docs/drivejepa-aux-heads.md, Results (2026-10-21) |

## Checks that decided something

- **Preflight, all nine runs** (21 Oct): passed identically: code at b48f731 (251 tests under 3.9), train / held-out disjoint (577 / 102 scenes), live rebuild against the cache 0.0 on 60 of 60, trainer 4.6e-5 m with picks 60 of 60, loss parity identical, sight 11.8 m max. Not checked: the inputs against paddock's (deferred by decision) → the runs trained; no checkpoint is a ship candidate until the paddock-inputs check exists
- **Do the heads learn? (rule 1)** (21 Oct): partly: the BEV map (road, vehicles), boxes and lead vehicle beat their no-camera baselines; lane dividers (IoU 0.000) and lane changes (F1 0, the head always says 'keep') learned nothing → class weights for the rare classes, or drop those heads (proposed in drivejepa-aux-next.md)
- **Which epoch to read** (21 Oct): the held-out planning loss moves 0.01-0.03 between epochs, as much as the gaps between arms, and ranked C first while C's picks were the worst; the kept epoch flatters every arm → quote every run at the fixed epoch 10, which nothing chose (evaluate.py)
- **Regression gate (H-safe)**: not run; the shuffled-label arm S was not run either

## Evidence outside this repo

- PR #20 was closed unmerged on 28 Oct 2026 and its branch train/drivejepa-auxilliary deleted. The account, recipe and code are at that PR's head, 1c60e59 (refs/pull/20/head), which is where this record links.
- Served checkpoints at {weights}/drivejepa_aux/aux-<arm>-s<seed>/step_<n>/ with sidecars, and reports at {runs}/drivejepa_aux/ (report.md, report-epoch10.md, picks-epoch10.md), at the training site (one H200); no sha256 is recorded on the branch.
- The next round is proposed, not run: docs/drivejepa-aux-next.md (PREDICTIONS.md first, then the inputs check against paddock, then the gate on T2 seed 0).
