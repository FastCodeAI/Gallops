<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H07 · A LoRA on the encoder closes the domain gap: the KITScenes gain holds in closed loop and on the base's encoder

**Siddesh** · KITScenes, AlpaSim · ⏳ running · 16 Oct 2026 · hypothesis

[← all experiments](../../README.md#hypothesis-experiments) · [the account](https://github.com/FastCodeAI/Gallops/blob/train/drivejepa-kitscenes/docs/drivejepa-kitscenes.md) (on branch ⎇) · [experiment.toml](experiment.toml)

> The record is on branch `train/drivejepa-kitscenes`, not merged yet; its links go to that branch.

**Priority:** 7 · **Evidence for:** D11, D18 ([decision log](../../README.md#decision-log))

## Question

Released Drive-JEPA, trained on NAVSIM, picks poorly on KITScenes: pick ADE (3 s) 2.372 m on 223 held-out moments while its closest candidate is 0.374 m away. A rank-16 LoRA on its encoder, trained by V-JEPA 2's hide-and-guess task on KITScenes frames with no human paths, moved the pick. Does that gain hold in closed loop (AlpaSim) and on the base's WA-JEPA encoder, i.e. does adapting the encoder transfer?

## Hypothesis

A LoRA on the encoder, trained self-supervised on the target domain's frames, closes the domain gap: the KITScenes gain (2.37 -> 1.88 m, open loop) holds in closed loop (AlpaSim, 30 scenes) and when repeated on the base's WA-JEPA encoder. Refuted if the merged-LoRA encoder gives no closed-loop gain over the released model on AlpaSim, or a LoRA on WA-JEPA's encoder does not reproduce the KITScenes gain.

## Predictions

⚠️ Not pre-registered: the encoder LoRA (exp1) trained 16 Oct, before the 21 Oct hypothesis list; the branch has no PREDICTIONS.md

## What we learned

So far, open loop on 223 held-out KITScenes moments, the LoRA alone moved the pick (ADE 2.372 -> 1.879 m) with the menu unchanged (closest candidate 0.374 -> 0.378 m); training the generator and then the scorer took the pick to 0.596 m. The only closed-loop test is of exp3, which also retrained the generator and scorer: the gate FAILs it (AlpaSim 0.4344 -> 0.1158, Bench2Drive 59.0360 -> 0.2686, NAVSIM 0.8572 -> 0.7533). The hypothesis' own tests, the LoRA alone in AlpaSim and a LoRA on WA-JEPA, have not run.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H07.exp1**<br>exp1, encoder only, tuned on KIT: rank-16 encoder LoRA by hide-and-guess; pass 3 of 6 kept | released (740335c800e2) | encoder LoRA (rank 16, every attention and MLP layer), guesser (training only) (6.3 M); frozen: encoder base weights, projector, generator, scorer | kitscenes `train, 482 scenes (default_geo_split_v1_0)` 18,757 frame pairs (train)<br>kitscenes `train, 50 picking scenes held out from training` 103 moments (selection)<br>kitscenes `val 117 + overlap_train_val 23 scenes` 223 moments (held-out) — read once, after the checkpoint was fixed; 221 of 223 below 15 m/s | [scripts in src/](https://github.com/FastCodeAI/Gallops/blob/train/drivejepa-kitscenes/experiments/drivejepa_kitscenes/03_encoder_training/src/)<br>Drive-JEPA (vendored tree with local changes, see POINTERS.md) @ `e21f474` | `{weights}/drivejepa_kitscenes/encoder-only/step_000001761/`<br>sha256 `2568101fb234cc25f4e6781834c72659283cb8c5df5569cc9e42f248b9e2aaf0` |
| **H07.exp2a**<br>exp2a, generator only, trained on KIT: gentle rate (Adam 3e-5), no stay term; 9.5 passes kept | H07.exp1 | generator; frozen: encoder (exp1's LoRA merged), projector, scorer | kitscenes `train, 482 scenes` 13,937 moments (train)<br>kitscenes `train, 50 picking scenes` 1,763 moments (selection)<br>kitscenes `val 117 + overlap_train_val 23 scenes` 223 moments (held-out) | [scripts in src/](https://github.com/FastCodeAI/Gallops/blob/train/drivejepa-kitscenes/experiments/drivejepa_kitscenes/07_generator_only_gentle_rate/src/)<br>Drive-JEPA (vendored tree with local changes, see POINTERS.md) @ `e21f474` | `{weights}/drivejepa_kitscenes/generator-only/step_000004133/`<br>sha256 `e66e505e1a166c019bd80a4f8c5e02581323a36e4a0ec80b1239d1c21fe019c5` |
| **H07.exp2b**<br>exp2b, generator and encoder LoRA trained together; 1 pass kept | H07.exp2a | generator, encoder LoRA (exp1's, unmerged), guesser (training only); frozen: projector, scorer | kitscenes `train, 482 scenes` 13,937 moments (train)<br>kitscenes `train, 50 picking scenes` 1,763 moments (selection)<br>kitscenes `val 117 + overlap_train_val 23 scenes` 223 moments (held-out) | [scripts in src/](https://github.com/FastCodeAI/Gallops/blob/train/drivejepa-kitscenes/experiments/drivejepa_kitscenes/08_generator_and_encoder_together/src/)<br>Drive-JEPA (vendored tree with local changes, see POINTERS.md) @ `e21f474` | `{weights}/drivejepa_kitscenes/generator-and-encoder/step_000000435/`<br>sha256 `c27aae6e392d98e849463445391490dab94ef9414e920cb874b251183adca915` |
| **H07.exp3**<br>exp3, encoder LoRA, generator and scorer trained together; 4.5 passes kept; the one checkpoint gated | H07.exp1 | encoder LoRA, generator, scorer's pred_score head, guesser (training only); frozen: projector, the rest of the scorer | kitscenes `train, 482 scenes` 13,937 moments (train)<br>kitscenes `train, 50 picking scenes` 1,763 moments (selection)<br>kitscenes `val 117 + overlap_train_val 23 scenes` 223 moments (held-out) | [scripts in src/](https://github.com/FastCodeAI/Gallops/blob/train/drivejepa-kitscenes/experiments/drivejepa_kitscenes/10_encoder_generator_scorer/src/)<br>Drive-JEPA (vendored tree with local changes, see POINTERS.md) @ `e21f474` | `{weights}/drivejepa_kitscenes/encoder-generator-scorer/step_000001958/`<br>sha256 `91e13bc38aa76ee386b903a27f582b5459b1ae0d362a0cb090c01db118ebdb98` |

- **H07.exp1:** Guesser-only warm-up, then passes of 587 steps at batch 32 (LoRA AdamW 2e-4, guesser AdamW 1e-4); stopped after 6 passes (3 without gain); pass 3 chosen on the 103 picking moments (pick ADE 2.392 -> 1.873). Training finished 16 Oct 23:45; controls 17 Oct. 03_encoder_training/REPORT.md counts 95 better / 55 worse by more than 10 cm, the page 96 / 55. Merged encoder alone: sha256 31db4e8dbf846a34d42d0042416ddd51ec11c8a36036054d88ffe3de6ee5d854.
- **H07.exp2a:** † An adjacent run, mapped to H07 by inference: it trains the generator, not the encoder. Loss: upstream's winner-takes-all to the human path only (KITScenes has no vocabulary grades). The fast-rate run (Adam 1e-4) was dropped after 1 pass when variety fell 28% (limit 10%); stay-term runs (0.3, 1.0) dropped. Chosen on the 1,763 picking moments (slow closest FDE 0.572 -> 0.346 m); held-out read 20 Oct.
- **H07.exp2b:** † An adjacent run, mapped to H07 by inference. Generator Adam 3e-5, LoRA AdamW 1e-4, plus hide-and-guess; stopped at 3 passes after two turn checks failed in a row; best qualifying check 0.355 m at 1 pass on the picking moments. Its control, generator only trained longer, reached 0.340 m at 0.5 more passes and was dropped.
- **H07.exp3:** † An adjacent run, mapped to H07 by inference. Generator and scorer start from the release, not from exp2. Scorer target: softmax(-d_k / 0.5 m) over the 32 by mean distance to the human path, replacing upstream's safety-and-progress grades. One backward pass of three equal-weight losses; the scorer loss sends zero gradient to all 41 generator tensors (checked before training). Stopped by rule after 7.5 passes; the 4.5-pass check (slow pick ADE 0.602 m on the picking moments) was fixed before the held-out read. Training-format checkpoint p04.5.pt: sha256 dc6c99d5b0298d32bf26f77f14f228969a7efb53822fcdd5feecac32b096eb45. The record pins the merged export the gate loaded, driven by paddock's drivejepa node with DRIVEJEPA_CHECKPOINT; {nas} is the branch's placeholder for the network store.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H07.exp1 ★ | KITScenes pick ADE (3 s), m 2.372 → **1.879** (n=223 moments)<br>paired -0.493 [-0.687, -0.313] (bootstrap over whole scenes); closest-candidate ADE 0.374 -> 0.378 (+0.004 [-0.009, +0.018]): the menu did not move, the pick did | released model, same software | ●●○○ moderate · adapted, open-loop | docs/drivejepa-kitscenes.md, held-out tables (recomputed by the author from the per-moment files) |
| H07.exp2a | KITScenes pick ADE (3 s), m 1.879 → 1.146 (n=223 moments)<br>paired -0.733 [-0.930, -0.545] | H07.exp1 | ●●○○ moderate · adapted, open-loop | docs/drivejepa-kitscenes.md |
| H07.exp2a | KITScenes closest-candidate ADE (3 s), m 0.378 → 0.194 (n=223 moments)<br>paired -0.185 [-0.231, -0.146]; variety fell 2.82 -> 1.63 m | H07.exp1 | ●●○○ moderate · adapted, open-loop | docs/drivejepa-kitscenes.md |
| H07.exp2b | KITScenes pick ADE (3 s), m 1.146 → 1.101 (n=223 moments)<br>paired -0.045 [-0.114, +0.026]: not resolved | H07.exp2a | ○○○○ unverified · adapted, open-loop | docs/drivejepa-kitscenes.md |
| H07.exp3 | KITScenes pick ADE (3 s), m 1.101 → 0.596 (n=223 moments)<br>paired -0.505 [-0.631, -0.383]; against exp2a -0.550 [-0.664, -0.439]; released model 2.372. Slow turns (37): 1.764 -> 0.819. The scorer now ranks by closeness to the human path, not by safety | H07.exp2b | ●●○○ moderate · adapted, open-loop | docs/drivejepa-kitscenes.md |

## Regression gate

**H07.exp3** — 28 Oct 2026, all tier, against released Drive-JEPA (the queue's stored runs): ⛔ **FAIL**

| Benchmark | Metric | Baseline | Candidate | t | MDE | Units | Verdict |
|---|---|---|---|---|---|---|---|
| KITScenes | ADE (3 s), m | 2.5375 | 0.9204 | 8.91 | 0.513 | 117 | 🟢 GAIN |
| nuScenes | L2 averaged over 1, 2 and 3 s, m | 2.4314 | 0.9152 | 14.58 | 0.293 | 143 | 🟢 GAIN |
| NAVSIM | EPDMS | 0.8572 | 0.7533 | -9.31 | 0.032 | 126 | 🔴 REGRESSION |
| neuroNCAP | NCAP score, quick check (3 pairs) | 3.1794 | 2.2194 | -0.44 | 11.792 | 3 | ⚪ NEUTRAL |
| Bench2Drive | driving score, crash check (route 3717) | 37.6700 | 18.4700 |  |  | 1 | ▫️ TOO FEW UNITS |
| AlpaSim | scene score, crash check (clip 1acf2f5d) | 0.9295 | 0.0000 |  |  | 1 | ▫️ TOO FEW UNITS |
| neuroNCAP | NCAP score, closed loop (15 pairs) | 3.6792 | 2.7265 | -1.26 | 2.276 | 15 | ⚪ NEUTRAL |
| Bench2Drive | driving score, closed loop (7 routes) | 59.0360 | 0.2686 | -5.15 | 38.246 | 7 | 🔴 REGRESSION |
| AlpaSim | scene score, closed loop (10 scenes) | 0.4344 | 0.1158 | -2.51 | 0.399 | 10 | 🔴 REGRESSION |

Job 20261027T215457-siddesh-94f0, keep-going; all 9 subsets finished. Every candidate run is marked invalid (site env override DRIVEJEPA_CHECKPOINT), and the baselines ran older paddock code (KITScenes, nuScenes and NAVSIM before Drive-JEPA's momentum default). KITScenes covers the 186 val moments; the 37 overlap_train_val moments are left out. The gate feeds KITScenes a straight command, zero acceleration and sideways speed and a fixed 2.2 m camera height, unlike training: 0.920 m against 0.596 m on the same moments. Bench2Drive: the car moved 0.0 m on 6 of 7 routes. Quick-tier crash checks are listed before the closed-loop pairs of the same benchmark.

## Checks that decided something

- **Random-LoRA control (04)** (17 Oct): five random LoRAs of the same size all pick worse: ADE (3 s) 2.41-2.53 m on the picking moments, against 1.873 for the trained one → the training itself matters
- **Same-software re-run of the released model (04)** (17 Oct): reproduces the earlier evaluation of it to within one changed pick
- **Gate KITScenes against the held-out evaluation (11, Part E)** (28 Oct): same 223 moments and human paths: 0.920 m through the gate against 0.596 m, from the gate's straight command, zero acceleration and sideways speed and fixed 2.2 m camera height; on the 184 truly-straight moments the gap is still 0.27 m. Whether these inputs contribute to the closed-loop failures is not verified
- **Bench2Drive near zero: real or broken? (11, Part B)** (28 Oct): real: simulator and model server healthy on every route, yet the car moved 0.0 m on 6 of 7; that its plans asked for no motion is argued from the controller code, as the plans were not recorded
- **The hypothesis' own tests (page, priority 7)**: not run: the merged LoRA encoder alone in AlpaSim (30 scenes), and a LoRA on WA-JEPA's encoder. No record on the branch (last commit 30 Oct) or in paddock's git log through 3 Nov

## Evidence outside this repo

- Weights, frame caches and per-moment results: off-repo at the training site, each with bytes and sha256 in experiments/drivejepa_kitscenes/POINTERS.md. None has a checkpoint.json sidecar or sits in {weights}.
- Gate job 20261027T215457-siddesh-94f0 in paddock's queue site folder (verdict.json, gate.log, baseline.toml, smoke/manifest.toml); candidate runs under paddock's runs root, <bench>/drivejepa/<run>.
- The vendored Drive-JEPA tree (upstream e21f474) every experiment imports has local changes in no git history: two modified attention files and three untracked files, the KITScenes adapter among them; sha256 in POINTERS.md.
- Siddesh's page 'Drive-JEPA Encoder LoRA on KIT', quoted on the team's 22 Oct page: 96 moments better, 55 worse, 72 within 10 cm; 1.4 GPU-hours (no-record).
