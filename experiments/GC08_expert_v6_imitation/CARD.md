<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# GC08 · Expert v6 (brake, then swerve for oncoming cars) and its imitation rounds 1A, 1B and 2

**Naren** · neuroNCAP, nuScenes · ❌ negative · 13 Oct – 16 Oct 2026 · gap-closing

[← all experiments](../../README.md#gap-closing-experiments) · [the account](https://github.com/FastCodeAI/Gallops/blob/expert/v6-imitation/docs/brake-swerve-expert.md) (on branch ⎇) · [experiment.toml](experiment.toml)

> The record is on branch `expert/v6-imitation`, not merged yet; its links go to that branch.

**Builds on:** [GC07](../GC07_subscore_scorer/CARD.md) · **Evidence for:** D10, D11, D12 ([decision log](../../README.md#decision-log))

## Question

On neuroNCAP the seeing model should brake for a parked and a crossing car, and for an oncoming car brake first, then steer round it. Can it learn that by imitating one consistent expert (v6), proven in an offline copy of the simulator, from the 5 reconstructed training scenes, and does what it learns transfer to a street it has not seen?

## What we learned

Imitating v6 teaches brake-then-swerve on the scenes it trained on (round 1B on new seeds: frontal clear in 10 of 20 runs, against 4 of 50 released), but it did not transfer: round 1A on held-out 0075 learned to brake for the side car and not to swerve, and on 0103 round 2 scored 2.31 against 3.01, all on 3-10 runs a shape. The likely cause, argued and not tested, is too few scenes (5, from 2 cities; NeuRAD measured 7.5 h per new scene on this machine).

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **GC08.v6-r1a-loso0075**<br>round 1A: refiner and scorer imitate v6 on 4 scenes, 0075 held out | released Drive-JEPA (seeing) | refiner, scorer; frozen: encoder | neuroncap-actor-insertion `round 0, 4 of the 5 rendered training scenes (0075 held out)` (train) — driven by the released seeing model, relabelled by v6; the doc gives 8,402 ticks over 150 episodes for round 0, of which 0075's are held out<br>nuscenes `train, real driving (human source)` (train) — 33% kept each epoch; the target is the human's path<br>neuroncap-actor-insertion `scene 0075` (held-out) — training curves, and 10 closed-loop runs per shape<br>neuroncap `scene 0103, frontal / stationary / side` 9 runs (eval) | `gallops.train.swerve`<br>commit `21067a5` | `{weights}/drivejepa_swerve/v6-r1a-loso0075/step_000015684/`<br>sha256 `49bf193783003cb8487eb6cc29856556fa07b4b80c7c6d182cc022b22e51b838`<br>⚠️ sidecar: train_git_sha records 048d49c, the commit at save time; 1A ran 21067a5. trained_on names only the rendered round-0 collection (it also read real nuScenes; notes.sources lists it). --accumulate and --subsample-source are not recorded |
| **GC08.v6-r1b-all**<br>round 1B: as 1A on all 5 scenes, plus the expert-driven collection; epoch-2 checkpoint | released Drive-JEPA (seeing) | refiner, scorer; frozen: encoder | neuroncap-actor-insertion `round 0, 5 rendered training scenes (0053 0058 0064 0070 0075)` 8,402 ticks (train) — driven by the released seeing model, relabelled by v6, 150 episodes<br>neuroncap-actor-insertion `expert-driven collection, the same 5 scenes` 9,234 ticks (train) — 150 episodes driven by v6 through neuroNCAP with the renderer (paddock DRIVEJEPA_DRIVE=expert)<br>nuscenes `train, real driving (human source)` (train) — 33% kept each epoch<br>neuroncap `scene 0103, frontal / stationary / side` 9 runs (eval)<br>nuscenes `open loop, 150 scenes (val, per the team page)` 5,119 samples (eval) | `gallops.train.swerve`<br>commit `048d49c` | `{weights}/drivejepa_swerve/v6-r1b-all/step_000069186/`<br>sha256 `2195fc27212d57a22ca35067d9f2642894343e0fbd60e6cfc6863c96213487eb`<br>⚠️ sidecar: trained_on names only the rendered round-0 collection (it also read real nuScenes and the expert-driven collection; notes.sources lists them). --accumulate and --subsample-source are not recorded |
| **GC08.v6-r2-short**<br>round 2: one epoch of DAgger from 1B on its own 40 episodes, relabelled by v6 | GC08.v6-r1b-all | refiner, scorer; frozen: encoder | neuroncap-actor-insertion `round 2: 1B epoch 2 driving the 5 training scenes, frontal and stationary, seeds 10-13 (40 episodes)` 2,466 ticks (train) — v6 labels: 1,135 brake, 507 brake then swerve, 288 least-impact; plus a quarter of round 0 and of the expert-driven set, a tenth of real nuScenes<br>neuroncap `scene 0103, frontal / stationary / side` 9 runs (eval) | `gallops.train.swerve`<br>commit `b4f3930` | `{weights}/drivejepa_swerve/v6-r2-short/step_000011377/`<br>sha256 `916b930a38ead276c601c3e6ccdbe0b5fbd20835f252c7cc879887cf8a5bdff9`<br>⚠️ sidecar: trained_on names only the rendered round-0 collection; notes.sources lists all four sources (round 0, real nuScenes, expert-driven, round 2). Read from the sidecar; the doc's list covers 1A and 1B only |

- **GC08.v6-r1a-loso0075:** Winner-takes-all L1 to the v6 plan plus scorer BCE and ranking; batch 1 × 8 accumulated, lr 1e-4, brake ticks 2×, brake-then-swerve 4×, 3 epochs. Training curves: nearest-candidate distance to the v6 plan 1.18 → 0.48 m (train), 0.83 → ~0.59 m (held-out 0075); scorer picks the best of 32 on 27–31% of training ticks, 9–22% on 0075, where its loss does not fall. Pinned checkpoint is epoch 1 (the one driven on 0103); epochs 2 and 3 are step_000031485 (ed263d39acfb) and step_000047338 (b1c829eb3dec). The sidecar's step counts one-tick forward passes, not optimizer steps (train/swerve.py increments it per micro-batch).
- **GC08.v6-r1b-all:** Recipe as 1A. Stopped at 93% of epoch 3 on OSError [Errno 5] reading one frame-cache file from the NAS; the epoch-2 checkpoint is the one evaluated. Training curves: nearest-candidate distance to the v6 plan 1.03 → 0.40 m. On 0103 frontal one run braked to 2.6 m/s and moved 3.3 m left (the intended manoeuvre), one swerved 3 m right without braking, one was hit; one stationary run veered 3.1 m left with no need. Every tick stayed on the drivable area. Step counts one-tick forward passes, not optimizer steps.
- **GC08.v6-r2-short:** Continues 1B for one epoch on the round-2 ticks plus a quarter of round 0 and of the expert-driven set and a tenth of the real driving, with brake-then-swerve at 2× (1B: 4×). Three things changed at once from 1B (the weighting, the share of old data, one epoch), so it does not say which one cost the swerve. Step counts one-tick forward passes, not optimizer steps.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| GC08.v6-r1a-loso0075 | neuroNCAP ncap_score, side 0.00 → 5.00 (n=10 runs)<br>scene 0075 (held out of 1A), 10 runs; 10/10 clear against 0/10; it brakes for the side car. Checkpoint epoch not recorded | released Drive-JEPA, seeing | ●●○○ moderate · smoke, adapted, no-record | docs/brake-swerve-expert.md §4 (branch) |
| GC08.v6-r1a-loso0075 | neuroNCAP ncap_score, frontal 1.84 → 1.69 (n=10 runs)<br>scene 0075 (held out of 1A), 10 runs; 0/10 clear either way: no swerve learned on the unseen street; stationary 5.00 for both | released Drive-JEPA, seeing | ●●○○ moderate · smoke, adapted, no-record | docs/brake-swerve-expert.md §4 (branch) |
| GC08.v6-r1a-loso0075 | neuroNCAP ncap_score, frontal 2.36 → 2.01 (n=3 runs)<br>scene 0103, 3 runs, epoch 1; 3/3 hit either way; side 0.00 (3/3 hit) against 1.67 (2/3 hit); stationary 5.00 for both | released Drive-JEPA, seeing | ●●○○ moderate · smoke, adapted, no-record | docs/brake-swerve-expert.md §4 (branch); paddock run neuroncap/drivejepa/20261014T153543-0d6a7b |
| GC08.v6-r1b-all | neuroNCAP ncap_score, mean of 3 shapes 3.01 → 3.07 (n=9 runs)<br>scene 0103, 3 runs per shape, epoch 2; +0.059 over 9 paired runs, paired SE 0.749 (recomputed from the two runs' samples.jsonl, 49c277 against 7153f0, in the #25 review; the doc on #17 gives 1.01, which is the unpaired SE). Frontal 4.21 (1/3 hit) vs 2.36 (3/3), side 0.00 (3/3 hit) vs 1.67 (2/3); the frontal gain and the side loss cancel | released Drive-JEPA, seeing | ●●○○ moderate · smoke, adapted, no-record | docs/brake-swerve-expert.md §4-5 (branch); paddock run neuroncap/drivejepa/20261015T055256-49c277 |
| GC08.v6-r1b-all | neuroNCAP frontal runs with no contact 4 → 10 (n=20 runs)<br>the 5 training scenes, new seeds 10-13; scenes it trained on; different seeds and run counts (10 of 20 against 4 of 50). Mean 2.85; stationary 16 of 20 (4.16) against 46 of 50 | released Drive-JEPA, seeing: 4 of 50 runs, seeds 0-9 | ●●○○ moderate · adapted, no-record | docs/brake-swerve-expert.md §5 (branch) |
| GC08.v6-r1b-all | nuScenes L2 (m) 2.380 → 1.186 (n=5,119 samples)<br>open loop, 5,119 samples; 1B trained on real nuScenes train driving; collisions 1.2% against 1.4%, not significant | released Drive-JEPA, seeing | ●●○○ moderate · open-loop, adapted, no-record | docs/brake-swerve-expert.md §4 (branch); paddock run nuscenes/drivejepa/20261015T074632-198187 |
| GC08.v6-r2-short ★ | neuroNCAP ncap_score, mean of 3 shapes 3.01 → **2.31** (n=9 runs)<br>scene 0103, 3 runs per shape; frontal 1.79 (3/3 hit), side 0.13 (3/3 hit), stationary 5.00. Lost 1B's swerve: widest frontal move 1.1 m against ~3 m; the needless stationary veer is gone (0.6 m against 3.1 m) | released Drive-JEPA, seeing | ●●○○ moderate · smoke, adapted, no-record | docs/brake-swerve-expert.md §5 (branch); paddock run neuroncap/drivejepa/20261016T180740-bf1796 |

## Checks that decided something

- **Expert v6, Gate A: replay**: each round-0 episode's driven plans, replayed through neuro-ncap's own Engine.step, vehicle model, evaluator and collision checker without the renderer (expert/rollout.py), reproduce its ego_poses.json and metrics.json: 150/150 episodes, worst 0.07 mm, metrics exact
- **Expert v6, Gate B: v6 drives all 150 training episodes (trigger T = 3.5 s)**: no contact frontal 42/50, stationary 50/50, side 23/50, against 4/50, 46/50, 25/50 for the released seeing model; 0 off-road ticks, 0 side flips; with memory cleared the same decision on 98.3% / 100% / 97.3% of ticks. Frontal failures are all 0058 (road too narrow to pass; v6 brakes, 1.56); most side failures are a crossing car at a 36–45° bearing, outside the front camera's 32.6° → side is left as it is; v6 approved as the teacher (13 Oct, per the team's status page)
- **Expert v6 in the real engine**: expert-driven collection (paddock's analysis adapter, DRIVEJEPA_DRIVE=expert) replays v6 through neuroNCAP with the renderer and asserts the pose stays within 2 cm of the offline rollout; the first smoke differed by 0.000 mm over 70 ticks and scored the same
- **Probe: does the frozen encoder see what v6 decides?**: linear probe on pooled tokens, v6's decision on a held-out scene, balanced accuracy with the image / ego state alone: 0053 0.29/0.26, 0058 0.45/0.38, 0064 0.67/0.55, 0070 0.53/0.61, 0075 0.93/0.58. Strong on the Boston scenes, weak on Singapore
- **Round 2's init preflight**: a DAgger round driven by the --init weights is fidelity-checked against them once they load: 1B's weights held to what 1B drove on 45/45 ticks, 0.0 m apart

## Evidence outside this repo

- neuroNCAP runs on 0103 (3 per shape), paddock run dirs under the site's paddock runs root, committed to no repository: neuroncap/drivejepa/20261014T153543-0d6a7b (1A epoch 1), 20261015T055256-49c277 (1B epoch 2), 20261016T180740-bf1796 (round 2); and 20261012T205915-7153f0, the released seeing model on seeds 0-2, whose scores match the doc's baseline (the doc names no run).
- nuScenes open loop of 1B epoch 2: paddock run dir nuscenes/drivejepa/20261015T074632-198187, committed to no repository.
- Round 1A on 0075 (10 runs per shape): the gallops collection collect_r1a_eval under {runs}; which 1A epoch drove it is not recorded.
- Labelled data (labelled_r0cam_v6, labelled_expert_v6, labelled_r2_v6, labelled_nusc) and candidate dumps: under {runs}, listed per run in each sidecar's notes.sources.
