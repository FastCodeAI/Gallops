<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# GC07 · Sub-score scorer over inserted swerve paths, on the seeing model

**Naren** · neuroNCAP · ⏸ parked (mixed) · 13 Oct 2026 · gap-closing

[← all experiments](../../README.md#gap-closing-experiments) · [the account](https://github.com/FastCodeAI/Gallops/blob/expert/v6-imitation/docs/brake-swerve-expert.md) (on branch ⎇) · [experiment.toml](experiment.toml)

> The record is on branch `expert/v6-imitation`, not merged yet; its links go to that branch.

**Builds on:** [GC06](../GC06_scorer_cache/CARD.md) · **Evidence for:** D08 ([decision log](../../README.md#decision-log))

## Question

After the camera fix, none of the seeing model's 32 candidates swerves round the oncoming car on neuroNCAP 0103 frontal; the only clear plans are hard braking, and braking alone still gets hit. Can a scorer with six sub-score heads, choosing among 32 inserted swerve paths passed through the model's own refiner, avoid the car?

## What we learned

On 0103 it cleared frontal (5.00, 0/3 hit, against 2.36 released), but on 3 runs a shape and by overriding the model from 0.5 s in every scenario, stationary and side included, flipping side 0–16 times an episode. Parked: labels graded each plan as if followed for 4 s, progress is 5/12 of the formula and the choice has no memory, so fast early swerves win; what carried forward is that the manoeuvre must come from the model's own generator, taught by one consistent driver (GC08).

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **GC07.sighted-v1**<br>sub-score scorer, 5 seeds × 20 epochs, on features that include the camera | released Drive-JEPA scorer MLP for the own-token columns; probe and geometry columns at zero | sub-score scorer (no collision, on road, TTC, lane keeping, progress, final) (2.012 M); frozen: encoder, generator, refiner | neuroncap-actor-insertion `5 non-benchmark nuScenes train scenes (0053 0058 0064 0070 0075), rendered with an inserted car` 23,777 ticks (train) — usable ticks: the seeing round-0 collection plus cache v4<br>neuroncap `scene 0103, frontal / stationary / side` 9 runs (eval) — 3 runs per shape | `gallops.train.scorer_train`<br>commit `dbcc105` | `{weights}/drivejepa_subscore/sighted-v1/step_000000020/`<br>sha256 (over 5 files) `b6e0c778e9f0f5f38e0b0388eb77b335cab286ff7b919a5226b5079b0d7224c7`<br>files: seed_0.pt `690ae3d6709d`, seed_1.pt `47a72b5d2c43`, seed_2.pt `5dfaff3e3023`, seed_3.pt `27125b1e205a`, seed_4.pt `1f11144e697e` |

- **GC07.sighted-v1:** Each tick, 32 hand-built paths (the pick or straight ahead shifted 1–4 m either side, plus 4 braking nudges) and 32 straight probe paths go through the refiner's last stage: 64 plans with the model's own features. A fixed formula combines the heads and an override rule (relative or absolute) decides when to leave the model's pick. Preflight passed (fidelity 99.3% after the switch to lossless frames). The checkpoint is a 5-seed ensemble (seed_0.pt … seed_4.pt, averaged); weights_sha256 is the digest over all five, listed in members (it pinned seed_4.pt alone until the #25 review). params_trained_m is the sidecar's params_m (per seed).

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| GC07.sighted-v1 ★ | neuroNCAP ncap_score, frontal 2.36 → **5.00** (n=3 runs)<br>scene 0103, 3 runs, relative rule; 0/3 hit against 3/3; it moved early in every scenario, so the clear frontal runs are not shown to be a reaction to the car | released Drive-JEPA, seeing | ●●○○ moderate · smoke, adapted, no-record | docs/brake-swerve-expert.md §1 (branch); paddock run neuroncap/drivejepa_subscore/20261013T202318-c43ac5 |
| GC07.sighted-v1 | neuroNCAP ncap_score, mean of 3 shapes 3.01 → 4.44 (n=9 runs)<br>scene 0103, 3 runs per shape, relative rule; frontal 5.00 vs 2.36, stationary 5.00 vs 5.00, side 3.33 (1/3 hit) vs 1.67 (2/3 hit). The doc gives the shapes, not the mean; 4.44 is the team page's | released Drive-JEPA, seeing | ●●○○ moderate · smoke, adapted, no-record | paddock run neuroncap/drivejepa_subscore/20261013T202318-c43ac5; baseline docs/brake-swerve-expert.md §5 (branch) |

## Checks that decided something

- **Why the sub-score scorer was parked**: it overrode the model from 0.5 s in every scenario and flipped side 0–16 times an episode: plans were labelled as if followed for the full 4 s (so the released braking plan graded as a crash), progress is 5/12 of the formula, each tick chose with no memory, and after the first swerve the car was in states no training data covered → parked; code and caches kept as diagnostics; the manoeuvre must come from the model's own plan generator, taught by one consistent driver (expert v6, GC08)

## Evidence outside this repo

- neuroNCAP runs on 0103 (3 per shape): paddock run dirs neuroncap/drivejepa_subscore/20261013T202318-c43ac5 (relative rule, the doc's table) and 20261013T205827-fff68a (DRIVEJEPA_SUBSCORE_RULE=absolute, the same nine scores); under the site's paddock runs root, committed to no repository.
- Training inputs: the seeing round-0 collection and scorer cache v4, under the site's runs and cache roots; listed in the sidecar's notes.
