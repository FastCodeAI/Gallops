<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H15 · A perception side-car on frozen Drive-JEPA: 3D boxes and each agent's 1 s path, tested as an input to the scorer, to the refiner, and against LoRA on the ViT

**Shreedhar** · NAVSIM, nuScenes · ⏳ running · 3 Nov 2026 · hypothesis

[← all experiments](../../README.md#hypothesis-experiments) · [the account](README.md) · [experiment.toml](experiment.toml)

**Builds on:** [H14](../H14_aux_heads/CARD.md), [H04](../H04_subscore_scorer_navsim/CARD.md)

## Question

Drive-JEPA's planner reads image tokens but no explicit account of the other agents. Can a small network trained on the frozen ViT's tokens detect them in 3D and predict each one's next second, and when its decoded agents are handed to the scorer or the refiner, does the plan improve, how much of the scorer or refiner must change to use them, and does LoRA on the ViT do better than the frozen tap? The scorer's use of named agents is also the route to explanations that can be checked.

## Hypothesis

H-A: a side-car of about 4 M parameters on the frozen tokens beats no-camera and kinematic baselines at detecting agents (recall and precision at 2 m) and at their 1 s paths. H-B1: given its decoded agents, a scorer picks better plans (EPDMS of the pick on held-out navtrain logs) than the identical scorer adaptation without them (arm C), by more than C's seed range and twice the paired, log-clustered SE on every seed; tested with the scorer frozen behind an adapter, with LoRA, fully fine-tuned, and replaced. H-B2: the same for the refiner's proposals (best-of-32 EPDMS). H-B3: LoRA on the ViT beats the frozen tap. H-info: shuffled agents (S) do not recover the gain and ground-truth agents (O) are at least as good. Refuted per question if F does not beat C under any mode; the side-car claim is refuted if it does not beat its baselines.

## Predictions

[PREDICTIONS.md](PREDICTIONS.md), written before the run.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H15.s0**<br>S0: the data passes on NAVSIM navtrain: splits, agent labels with each agent's 1 s path, the token cache and the release's outputs | released | —; frozen: encoder, projector, refiner, scorer | navsim `navtrain, train logs (1,071)` (train)<br>navsim `navtrain, select logs (H04's 51 held-out)` (selection)<br>navsim `navtrain, report logs (70, seeded draw)` (held-out) | [drivejepa_perception.toml#labels](../../recipes/drivejepa_perception.toml)<br>Drive-JEPA (linhanwang/Drive-JEPA) @ `e21f474` + patches | — |
| **H15.a0**<br>A0: which frozen tap carries 3D -- one-layer probes on the post-projector tokens, the ViT's output and block 12 | released | one-layer side-car probe; frozen: encoder, projector, refiner, scorer | navsim `navtrain, select logs (H04's 51 held-out)` (selection) | [drivejepa_perception.toml#a0-vit-out](../../recipes/drivejepa_perception.toml)<br>commit `80f4181` | — |
| **H15.a1**<br>A1: the side-car on the ViT's output, three seeds and 128 queries on seed 0 | released | side-car (4.39 M); frozen: encoder, projector, refiner, scorer | navsim `navtrain, train logs (1,071)` 94,195 samples (train)<br>navsim `navtrain, report logs (70, seeded draw)` 4,636 samples (held-out) | [drivejepa_perception.toml#a1-s0](../../recipes/drivejepa_perception.toml)<br>commit `f05e2ff` | — |
| **H15.a2**<br>A2 (optional): the side-car on a seeded 25% and 50% of the train logs, seed 0 -- is it data-limited? | released | side-car (4.39 M); frozen: encoder, projector, refiner, scorer | — | [drivejepa_perception.toml#a2-50](../../recipes/drivejepa_perception.toml)<br>commit `f3934bf` | — |
| **H15.b1**<br>B1: the side-car's agents as scorer inputs -- four modes (adapter, + LoRA, + full scorer, replacement) x arms C, F, O, S x three seeds | released | scorer adapter, LoRA, the scorer MLP or a replacement scorer, by mode; frozen: encoder, projector, refiner, side-car | navsim `navtrain, report logs (70, seeded draw)` 4,636 samples (held-out) | [drivejepa_perception.toml#b1-full-F](../../recipes/drivejepa_perception.toml)<br>commit `e6af85b` | — |
| **H15.b2**<br>B2: the side-car's agents as refiner inputs -- three modes (adapter, + LoRA, + the refiner path) x arms C and F x three seeds | released | refiner adapter, LoRA on the refiner or the refiner path, by mode; frozen: encoder, projector, scorer, side-car | — | [drivejepa_perception.toml#b2-full-F](../../recipes/drivejepa_perception.toml)<br>commit `e6af85b` | — |
| **H15.b3**<br>B3.1: rank-16 LoRA on the ViT's last four blocks in a branch only the side-car reads, against A1 continued without it | H15.a1 | LoRA on blocks 21-24 (1.05 M), side-car (5.44 M); frozen: encoder blocks 1-20, projector, refiner, scorer | — | [drivejepa_perception.toml#b3-lora](../../recipes/drivejepa_perception.toml)<br>commit `1627ebd` | — |
| **H15.b32**<br>B3.2, reduced: B3.1 seed 0's LoRA shared with the release's planner, which is not retrained | H15.b3 | —; frozen: everything | — | [drivejepa_perception.toml#b3-shared](../../recipes/drivejepa_perception.toml)<br>commit `307a44d` | — |
| **H15.ood**<br>OOD: the side-car zero-shot and every B1 model on nuScenes, H14's 102 held-out train scenes | released | —; frozen: everything | nuscenes `train, H14's 102 held-out scenes` 2,996 keyframes (held-out) | [drivejepa_perception.toml#ood-read-v2](../../recipes/drivejepa_perception.toml)<br>commit `5635f7f` | — |

- **H15.s0:** Nothing trains here. Labels: every annotated agent within 60 m at the current frame, in its ego frame, with its positions at t + 0.5 s and t + 1.0 s joined on track_token and moved through global (docs/drivejepa-perception-sidecar.md, S0.4). Held to NAVSIM's and nuPlan's own code by `navsim_data check`.
- **H15.a0:** 10,000 train rows, 6 epochs each, read on the select logs at the kept epoch (6). Read again with the fixed readout (f2b6570): the as-run 'within 30 m' counted to 35 m (README). The ViT's output was chosen for A1.
- **H15.a1:** 12 epochs over 94,195 train rows, kept epoch chosen on the select logs with the as-run readout, then read again with the fixed one (a1-*-reread). Sight check: zeroed tokens give 27 detections a sample on seed 0 (0 on the others); with another log's real tokens every run's recall falls to 0.046-0.047 (a1-*-sight). Seed 0's kept epoch (11) made the agents phase B reads.
- **H15.a2:** Whole logs drawn with seed 0 (train_sidecar.fraction_of_logs): 25,120 and 49,276 rows; 100% is H15.a1 seed 0. Read with the fixed readout at each run's kept epoch (11 and 12).
- **H15.b1:** Loss bce (chosen on the select logs, arm C, before any arm read an agent; the plan's imitation term made every control worse than the release). F trained from e6af85b, O and S from 1627ebd; the B1 code is the same. Agents: A1 seed 0, epoch 11, 4.86 a row. Verdict by the pre-registered rule (b1-report): NEUTRAL for every arm in every mode, kept and last epoch. B1.6, every F mode at seed 0: deleting the named agent changes the pick 3.1-4.9x as often as a random one, the agent is real 86% of the time; AUC 0.92-0.97, but the controls reach the same (b1-calibration-*), so the calibration is the trained sub-scores', not the agents'.
- **H15.b2:** The release's planning loss (winner-takes-all to the human plus 0.5 x up to four good anchors, drawn as its training drew them), 3 epochs over 94,195 train rows, read at the last epoch; new candidates PDM-scored offline, picked by the frozen release scorer. C trained from 6b0736a, F from e6af85b (the refiner code is the same). Verdict by the pre-registered rule (b2-report): NEUTRAL in every mode, in best of 32 and in the pick. The best mode on select is full; its O and S are recorded when they finish.
- **H15.b3:** From a block-20 cache (30 GB, 20,000 train rows + select + report), 4 epochs, three seeds per arm; both arms start from A1 seed 0's kept epoch. The preflight held the frozen branch to A1: select F1 0.66908 against 0.66940.
- **H15.b32:** The frozen branch first reproduces the release (pick on every row within 1.1 mm, 99.9% of rows within 4.4 mm, picks equal; its first run was refused by one non-picked candidate 0.75 m off, measured by scripts/b3_identity.py). The plan's B3.2 retrains the planner on the adapted tokens; that was not run.
- **H15.ood:** Nothing trains. The release at batch 1 on 2,996 keyframes (inputs through H14's TablesTickDataset); the projection held to the calibration's own on every scene (1.4e-5 px). The release's picked L2 at 3 s, 3.788 m, equals H14's on the same keyframes.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H15.a0 | NAVSIM recall at 2 m within 30 m, front view, all classes (ViT output, 1024-d) 0.249 → 0.343 (n=32,375 targets)<br>select logs; one seed per tap; block 12: 0.269. P1 (best tap >= 0.30) supported. Precision 0.383 against 0.318 | post-projector tokens, 256-d | ●●○○ moderate · open-loop | — |
| H15.a1 | NAVSIM recall at 2 m within 30 m, front view, all classes 0 → 0.631 (n=24,216 targets)<br>report logs; mean of three seeds (0.6300-0.6323); precision 0.736-0.742. P2 supported on all three seeds; 128 queries (seed 0): recall 0.647, precision 0.802 | no camera (blind arm, trained identically) | ●●●○ strong · open-loop | — |
| H15.a1 | NAVSIM recall at 2 m, moving vehicles within 20 m 0.935<br>report logs; mean of three seeds (0.935-0.936); P3 supported (>= 0.65) | — | ●●●○ strong · open-loop | — |
| H15.a1 | NAVSIM lead-vehicle gap error, m 8.53 → 2.28 (n=1,807 samples)<br>report logs, samples with a vehicle ahead in the ego lane; mean of three seeds (2.14-2.36); P5 supported | no-camera prior (train mean gap) | ●●●○ strong · open-loop | — |
| H15.a1 | NAVSIM path FDE at 1 s, matched moving agents, m 1.268 → 1.216 (n=4,231 agents)<br>report logs; mean of three seeds; paired difference -0.049 (SE 0.013), -0.048 (SE 0.014), -0.059 (SE 0.012), beyond 2 SE clustered by log on every seed: P4's support rule met, but 4-5% against the predicted 10% | constant velocity from the side-car's own velocity | ●●○○ moderate · open-loop | — |
| H15.a2 | NAVSIM F1 at 2 m within 30 m, front view, all classes, on half the train logs 0.683 → 0.638<br>report logs; one seed per fraction; 25% of the logs: 0.567. Each doubling adds 0.045-0.07 F1: still rising, so the side-car is data-limited | H15.a1 seed 0, all the train logs | ●●○○ moderate · open-loop | — |
| H15.b1 ★ | NAVSIM pick EPDMS, scorer with the side-car's agents (F) against the same scorer without (C), best mode on select (full) 0.9150 → **0.9142** (n=4,636 samples)<br>report logs; NEUTRAL in all four modes. F - C per seed -0.13, -0.11, -0.00 points (2 SE 0.19, 0.16, 0.13); ground-truth agents (O) NEUTRAL too, so the limit is not perception: NC and TTC are 0.996 on the release's pick in-sample (S0.8). The release scores 0.9140 on these rows (in-sample for it). P7, P8 refuted | H15.b1 full C | ●●○○ moderate · open-loop | — |
| H15.b2 | NAVSIM best-of-32 EPDMS, refiner with the side-car's agents (F) against the same refiner without (C), best mode on select (full) 0.9589 → 0.9589 (n=4,636 samples)<br>report logs; NEUTRAL in all three modes. F - C per seed +0.004, -0.013, +0.001 points (2 SE 0.019, 0.015, 0.010); P12 (+0.2 to +0.5) refuted. The release scores 0.9564 on these rows (in-sample for it); retraining the refiner without agents already raises best of 32 to 0.9575-0.9590 | H15.b2 full C | ●●○○ moderate · open-loop | — |
| H15.b3 | NAVSIM recall at 2 m within 30 m, front view, all classes 0.632 → 0.631<br>report logs; mean of three seeds (0.6290-0.6330); the control (A1 continued, no LoRA) 0.6215-0.6285. P13 (>= +0.05) refuted | H15.a1 seed 0 (the frozen tap) | ●●○○ moderate · open-loop | — |
| H15.b32 | NAVSIM pick EPDMS with B3.1's LoRA in the shared trunk, planner unchanged 0.9140 → 0.9139 (n=4,636 samples)<br>report logs; SE 0.0009; paired difference -0.0002 (SE 0.0009, clustered by log); best of 32 0.9567 against 0.9564; in-sample for R | the frozen branch (= R) | ●●○○ moderate · open-loop | — |
| H15.ood | nuScenes side-car zero-shot recall at 2 m within 30 m, vehicles 0.480<br>102 held-out train scenes; A1 seed 0, never trained on nuScenes; all classes 0.380 at precision 0.608 (navtrain report: 0.632 / 0.742). P15's recall part (>= 0.30) holds | — | ●●○○ moderate · open-loop | — |
| H15.ood | nuScenes picked-plan L2 at 3 s, m, B1 full F (best mode on select) 3.788 → 4.050 (n=2,996 keyframes)<br>102 held-out train scenes; mean of three seeds; minus R +0.234, +0.239, +0.313 m, each beyond 2 SE clustered by scene: P15 refuted. full C (no agents) is as much worse (+0.282, +0.438, +0.352), so the cost is fitting the scorer to navtrain, not the agents; collision rate not lower | R, the released model | ●●○○ moderate · open-loop | — |
