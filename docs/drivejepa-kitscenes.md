# Drive-JEPA retrained on KITScenes: three experiments

Three ways of retraining parts of the released Drive-JEPA model on KITScenes Multimodal, evaluated on
223 held-out moments, and the regression-gate result for the one checkpoint that went through it.
The working record (scripts, step reports, result tables) is
[experiments/drivejepa_kitscenes/](../experiments/drivejepa_kitscenes/README.md).

**In one paragraph.** On held-out KITScenes moments, the path the model finally chooses moved from
2.372 m average error over 3 s (released) to 0.596 m (encoder, generator and scorer trained
together). Most of that came from two things: training the generator, which makes the 32 candidate
paths closer to what a human drove, and retraining the scorer to pick the candidate closest to the
human path. These are **open-loop numbers**: the model's plan never drives a car. The one checkpoint
that was also tested in closed loop, through Paddock's regression gate, **fails it**: it gains on
the two open-loop driving logs (KITScenes, nuScenes) and regresses on NAVSIM, Bench2Drive and AlpaSim.

**These checkpoints are a research record, not for deployment.** None has a `checkpoint.json`
sidecar, none was written into `{weights}`, and the one tested in closed loop failed the gate.
Their locations and sha256 are in
[POINTERS.md](../experiments/drivejepa_kitscenes/POINTERS.md).

---

## The model

Drive-JEPA (the released `drive_jepa_perception_based_agent_vitl_v2` checkpoint) plans 4 s ahead
from the front camera:

1. **Two front frames** (now and 0.5 s earlier) go into the **encoder**, a ViT-L image network
   (304 M parameters).
2. A **projector** (one 1024 → 256 layer) passes the encoder's features on.
3. The **generator** turns them, with the car's own speed and the driving command, into **32
   candidate paths**, refined in 4 rounds. These 32 are the **menu**.
4. The **scorer** gives each candidate a score and the model drives the highest. That choice is the
   **pick**; it is the model's output.

Two numbers follow from this. The **closest candidate** is the menu entry nearest the human path:
it measures how good the menu is. The **pick** is what the scorer chose: it measures the model. The
**pick gap** is the difference between them, i.e. how much the scorer gives away.

Three more terms:

- **LoRA** (low-rank adaptation): a small trainable change added to a frozen network's weight
  matrices, here rank 16 on every attention and MLP layer of the encoder (6.3 M trainable
  parameters, 2 % of the encoder). It can be merged into the weights afterwards.
- **Hide-and-guess**: a self-supervised task that needs no human paths. Patches of the two frames are
  hidden; a small **guesser** network predicts the encoder's features for the hidden patches, and
  the targets come from an **answer key**, a slowly moving copy of the encoder (momentum 0.99925).
  This is V-JEPA 2's training task, with its released predictor as the guesser's starting weights.
- **A pass** is one sweep through the training data.

## Data

KITScenes Multimodal, front ring camera, the official `default_geo_split_v1_0` lists.

| Use | Scenes | Moments | Read for |
|---|---:|---:|---|
| Training | 482 of `train` | 13,937 (every 5th frame with a full 4 s human path); 18,757 frame pairs for hide-and-guess | training |
| Picking | 50 of `train`, drawn with `random.Random(20261016)` and held out from training | 1,763 (generator runs) or 103 (encoder run, the earlier evaluation's rule) | choosing a checkpoint |
| Held out | `val` 117 + `overlap_train_val` 23 | 223 (186 + 37) | the comparison below |
| Not read | `test`, `test-e2e` | — | — |

- `train.txt` lists 534 scenes; 532 are on disk at the training site, and 50 of those were set aside
  for picking. The two lists are `02_encoder_build/out/splits.json`; the held-out moments are
  `09_heldout_comparison/heldout_moments.csv`. The test checks the three are disjoint.
- **The 223 held-out moments were never used to choose anything; each model was evaluated on them
  once, after it was fixed.** In order: the encoder-only model (and a same-software re-run of the
  released model), then the four-model comparison, then the three-part model.
- **The 223 are almost all slow.** Groups below: **slow** is below 15 m/s (221 moments); **slow
  turns** are slow moments whose heading changes by more than 0.3 rad within 3 s (37); fast has 2
  moments and is too few to read.
- `overlap_train_val` scenes are held out by scene, but Paddock's gate notes they share places with
  the training data. Its KITScenes verdict leaves them out.

## The three experiments

| Experiment | Encoder LoRA | Projector | Generator | Scorer | Starts from | Loss |
|---|---|---|---|---|---|---|
| **1. Encoder only, tuned on KIT** | trained | frozen | frozen (released) | frozen (released) | Released model | hide-and-guess only; no human paths |
| **2a. Generator only, trained on KIT** | frozen (experiment 1, merged) | frozen | trained | frozen (released) | Encoder only, tuned on KIT | generator loss vs the human path |
| **2b. Generator and encoder trained together** | trained | frozen | trained | frozen (released) | Generator only, trained on KIT, plus experiment 1's LoRA unmerged | generator loss + hide-and-guess |
| **3. Encoder, generator and scorer trained together** | trained | frozen | trained | pick head trained | **Released generator and scorer** plus experiment 1's LoRA | generator loss + scorer loss + hide-and-guess |

Experiment 3 does **not** start from experiment 2's generator: its generator starts from the
released weights. Experiments 2 and 3 both branch from experiment 1.

### 1. Encoder only, tuned on KIT

The encoder LoRA was trained by hide-and-guess on the 18,757 frame pairs of the training scenes:
first a warm-up that trained only the guesser, then passes of 587 steps at batch 32 (LoRA AdamW
2e-4, guesser AdamW 1e-4). After each pass the whole model planned the 103 picking moments, and the
pass with the lowest pick ADE (3 s) was kept. Training stopped after 6 passes (3 without gain); pass
3 was chosen.

What it changed: **the menu did not move, the pick did.** The scorer, reading the tuned features,
picks faster candidates from the same menu, so the error along the direction of travel falls and the
sideways error barely moves (`03_encoder_training/REPORT.md`). Two controls: five random LoRAs of
the same size all pick worse (ADE (3 s) 2.41–2.53 m on the picking moments, against 1.873 for the
trained one), so the training itself matters; and the released model re-run in the same software
reproduces the earlier evaluation of it to within one changed pick (`04_encoder_checks/`).

### 2. Generator only, then generator and encoder together

**The generator loss.** Upstream trains the generator winner-takes-all: of the 32 candidates, only
the one closest to the human path is pulled towards it (L1 over the 8 waypoints, x, y and heading),
in all 4 refinement rounds. Upstream's loss has a second half that pulls candidates towards
vocabulary paths graded safe by NAVSIM's rule-based scorer; KITScenes has no such grades, so only the
human-path half was used, by calling upstream's function unchanged with empty vocabulary lists
(`05_generator_survey/REPORT.md`).

Runs tried, all on 13,937 training moments, batch 32, checked every half pass on the 1,763 picking
moments:

| Run | Result on the picking moments | Kept? |
|---|---|---|
| Generator only, fast learning rate (Adam 1e-4) | the menu got much closer, but its variety fell 28 % in one pass (limit 10 %), slow turns lost their turning candidates; stopped after 1 pass | dropped |
| Generator only, gentle learning rate (3e-5), 10 passes | slow-moment closest FDE (3 s) 0.572 → 0.346 m at 9.5 passes; turns held at every check | **kept: Generator only, trained on KIT** |
| — with a stay-near-original term, weight 0.3 / 1.0 (pulls the 31 non-closest candidates back to where they started) | keeps the variety, costs accuracy (0.413 / 0.486 m) and blocks the fast menu from reaching high speeds | dropped |
| Generator only, trained longer (control, from the kept checkpoint) | best 0.340 m at 0.5 more passes, within the 0.01 m check-to-check swing; then slowly worse | dropped |
| Generator and encoder trained together (generator Adam 3e-5, LoRA AdamW 1e-4, plus hide-and-guess), from the kept checkpoint | best qualifying check 0.355 m at 1 pass; stopped at 3 passes after two turn checks failed in a row | **kept for the comparison: Generator and encoder trained together** |

Both kept checkpoints were chosen on the picking moments; the held-out numbers below are their first
reading on unseen moments.

### 3. Encoder, generator and scorer trained together

One run trains three parts at once from the released generator and scorer and experiment 1's LoRA,
with one backward pass of the sum of three losses (equal weights):

1. **Generator loss**, as in experiment 2 (generator Adam 3e-5).
2. **Scorer loss** (new). Upstream picks the candidate with the highest last logit of the scorer's
   `pred_score` head (`drive_jepa_model.py:85-91`); the other five logits do not enter the pick. For
   each candidate *k*, *d<sub>k</sub>* is its mean xy distance to the human path over 0.5–3 s. The
   target over the 32 is softmax(−*d<sub>k</sub>* / 0.5 m), and the loss is the cross-entropy between
   that target and the softmax of the last logit over the 32. Only the `pred_score` head is trained
   (Adam 1e-4); the rest of the scorer stays frozen.
3. **Hide-and-guess**, as in experiment 1 (LoRA and guesser AdamW 1e-4).

**The frozen generator replay.** The scorer reads the generator's final state, not the image
features. To keep the scorer loss from training the generator's weights, each step runs the
generator's 4 rounds a second time for the scorer, with the generator's weights, the car's own
state, the starting candidates and the candidates passed between rounds all detached; only the image
features stay attached. Checked before training, in training mode with and without image masking:
the replay gives the same numbers as the normal pass bit for bit; the scorer loss sends exactly zero
gradient to all 41 generator tensors and non-zero gradient to all 10 `pred_score` tensors and all 192
LoRA tensors. So the scorer loss can move the encoder, and through it the menu, but never the
generator's weights.

**Who trains the encoder.** Measured on one fixed batch at each of the 16 checks, the share of the
LoRA's gradient norm from each loss was: generator 64–86 %, scorer 12–35 %, hide-and-guess 1.4–3.4 %
(at the chosen check 76 / 21 / 2.8 %; `10_encoder_generator_scorer/out/tables.md`).

Training stopped by rule after 7.5 passes (no gain in slow pick ADE for 6 checks); the check at 4.5
passes had the lowest slow pick ADE (3 s) on the picking moments (0.602 m) and was fixed before the
held-out moments were read. The turn and menu checks held at every check. The scorer loss rose after
half a pass because the menu tightened and the targets flattened (target entropy 2.53 → 2.88), not
because the scorer got worse.

## Results on the 223 held-out moments (open loop)

All five models, batch 1, no image masking. ADE (3 s) is the mean distance to the human path over
the waypoints at 0.5, 1.0, …, 3.0 s; FDE (*t*) is the distance at time *t*. Closest candidate: the
menu entry with the lowest ADE (3 s). Candidates within 2 m: how many of the 32 have ADE (3 s) under
2 m. Variety: the mean distance between two candidates, over all 496 pairs. Pick gap: pick ADE (3 s)
minus closest candidate ADE (3 s). All in metres; all 223 moments.

| Model | closest candidate ADE (3 s) | closest candidate FDE (3 s) | candidates within 2 m | variety (m) | pick ADE (3 s) | pick FDE (1 s) | pick FDE (2 s) | pick FDE (3 s) | pick gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Released model | 0.374 | 0.664 | 17.5 | 2.83 | 2.372 | 1.314 | 2.679 | 4.206 | 1.997 |
| Encoder only, tuned on KIT | 0.378 | 0.662 | 17.7 | 2.82 | 1.879 | 1.002 | 2.085 | 3.442 | 1.501 |
| Generator only, trained on KIT | 0.194 | 0.379 | 25.9 | 1.63 | 1.146 | 0.319 | 1.223 | 2.644 | 0.952 |
| Generator and encoder trained together | 0.190 | 0.377 | 26.0 | 1.60 | 1.101 | 0.313 | 1.169 | 2.533 | 0.912 |
| Encoder, generator and scorer trained together | 0.203 | 0.388 | 24.7 | 1.78 | 0.596 | 0.201 | 0.617 | 1.354 | 0.394 |

Slow moments and slow turns:

| Model | slow (221): closest ADE (3 s) | slow: pick ADE (3 s) | slow turns (37): closest ADE (3 s) | slow turns: pick ADE (3 s) |
|---|---:|---:|---:|---:|
| Released model | 0.357 | 2.323 | 0.558 | 3.026 |
| Encoder only, tuned on KIT | 0.359 | 1.826 | 0.557 | 2.571 |
| Generator only, trained on KIT | 0.194 | 1.124 | 0.350 | 1.545 |
| Generator and encoder trained together | 0.189 | 1.077 | 0.346 | 1.764 |
| Encoder, generator and scorer trained together | 0.202 | 0.589 | 0.332 | 0.819 |

Paired differences, newer model minus the one it is compared with (negative = newer is better), mean
and 95 % interval from a paired bootstrap over whole scenes (10,000 resamples, seed 0):

| Model | compared with | metric | all 223 | slow (221) | slow turns (37) |
|---|---|---|---|---|---|
| Encoder only, tuned on KIT | Released model | closest candidate ADE (3 s) | +0.004 [-0.009, +0.018] | +0.002 [-0.011, +0.017] | -0.000 [-0.058, +0.071] |
| Encoder only, tuned on KIT | Released model | pick ADE (3 s) | -0.493 [-0.687, -0.313] | -0.497 [-0.696, -0.318] | -0.455 [-0.864, -0.088] |
| Generator only, trained on KIT | Encoder only, tuned on KIT | closest candidate ADE (3 s) | -0.185 [-0.231, -0.146] | -0.165 [-0.199, -0.136] | -0.207 [-0.389, -0.080] |
| Generator only, trained on KIT | Encoder only, tuned on KIT | pick ADE (3 s) | -0.733 [-0.930, -0.545] | -0.702 [-0.888, -0.518] | -1.026 [-1.601, -0.539] |
| Generator and encoder trained together | Generator only, trained on KIT | closest candidate ADE (3 s) | -0.004 [-0.013, +0.004] | -0.005 [-0.013, +0.003] | -0.004 [-0.044, +0.032] |
| Generator and encoder trained together | Generator only, trained on KIT | pick ADE (3 s) | -0.045 [-0.114, +0.026] | -0.047 [-0.115, +0.024] | +0.219 [-0.055, +0.535] |
| Encoder, generator and scorer trained together | Generator only, trained on KIT | closest candidate ADE (3 s) | +0.009 [-0.006, +0.021] | +0.008 [-0.007, +0.021] | -0.019 [-0.086, +0.036] |
| Encoder, generator and scorer trained together | Generator only, trained on KIT | pick ADE (3 s) | -0.550 [-0.664, -0.439] | -0.536 [-0.652, -0.429] | -0.725 [-1.170, -0.331] |
| Encoder, generator and scorer trained together | Generator and encoder trained together | closest candidate ADE (3 s) | +0.013 [-0.004, +0.027] | +0.013 [-0.004, +0.028] | -0.015 [-0.092, +0.053] |
| Encoder, generator and scorer trained together | Generator and encoder trained together | pick ADE (3 s) | -0.505 [-0.631, -0.383] | -0.488 [-0.615, -0.369] | -0.945 [-1.476, -0.479] |

How to read it:

- **Training the generator is what improves the menu**: closest candidate ADE (3 s) 0.378 → 0.194 m.
  Training the encoder with it, or training the scorer too, does not change the menu beyond noise.
- **The menu also narrowed**: variety 2.82 → 1.63–1.78 m. More candidates sit near the human path,
  and fewer cover other options.
- **Training the scorer is what improves the pick**: 1.101 → 0.596 m against the generator and
  encoder model, and on slow turns 1.764 → 0.819 m. The pick gap falls from 0.91 to 0.39 m.
- About 1 in 5 moments (41–48 of 223) still gets a worse pick by more than 10 cm under the three-part
  model than under the two-part ones.
- The pictures of the largest gains and losses are in `09_heldout_comparison/out/` and
  `10_encoder_generator_scorer/out/`.

## Regression gate: Encoder, generator and scorer trained together

**Only this checkpoint went through the regression gate.** The other models were evaluated open
loop only.

**What the gate is.** Paddock's regression gate (`tools/gate.py` with its shared queue, Paddock
`docs/gate.md`) runs a candidate checkpoint through Paddock's normal benchmarks and pairs each
sample with the same sample from a stored baseline run, here the released Drive-JEPA. For each
benchmark it runs a paired t-test grouped by that benchmark's independent unit (scene, log, route or
scenario pair): **REGRESSION** if t ≤ −t<sub>crit</sub>, **GAIN** if t ≥ t<sub>crit</sub>, otherwise
**NEUTRAL**. The overall result is **FAIL** if any benchmark regresses; **SHIP** needs at least one
closed-loop gain (neuroNCAP's 15 pairs or AlpaSim) and no regression; anything else is **HOLD**.
"Closed loop" means the model's plan drives a simulated car and the next frame depends on where it
went. The two crash checks are single-scene runs that only show the pipeline works.

**How it was run.** The trained checkpoint was merged into one full state_dict
(`11_regression_gate/prep_export.py`, sha256 `91e13bc38aa76ee3…`), checked to load strictly in
Paddock's Drive-JEPA node and to give the same plans as the training code to 0.0 m on 5 moments
(`11_regression_gate/PREP.md`), and submitted to the queue on 2026-10-28: job
`20261027T215457-siddesh-94f0`, all tiers, keep-going, Paddock commit `9585f81`. All 9 benchmark
subsets finished. The baselines are the queue's stored runs of the released model.

The gate's overall verdict is **FAIL**.

| Benchmark subset | metric | released model | Encoder, generator and scorer trained together | units | t | verdict |
|---|---|---:|---:|---|---:|---|
| KITScenes, open loop | ADE (3 s), m | 2.5375 | 0.9204 | 117 scenes | 8.91 | GAIN |
| nuScenes, open loop | L2 averaged over 1, 2 and 3 s, m | 2.4314 | 0.9152 | 143 logs | 14.58 | GAIN |
| NAVSIM, open loop, simulated scoring | EPDMS | 0.8572 | 0.7533 | 126 logs | -9.31 | REGRESSION |
| neuroNCAP, quick check | NCAP score | 3.1794 | 2.2194 | 3 scenario pairs | -0.44 | NEUTRAL |
| Bench2Drive, crash check | driving score | 37.6700 | 18.4700 | 1 route | – | TOO FEW UNITS |
| AlpaSim, crash check | scene score | 0.9295 | 0.0000 | 1 scene | – | TOO FEW UNITS |
| neuroNCAP, closed loop | NCAP score | 3.6792 | 2.7265 | 15 scenario pairs | -1.26 | NEUTRAL |
| Bench2Drive, closed loop | driving score | 59.0360 | 0.2686 | 7 routes | -5.15 | REGRESSION |
| AlpaSim, closed loop | scene score | 0.4344 | 0.1158 | 10 scenes | -2.51 | REGRESSION |

Units are the independent units the test groups by. The KITScenes verdict covers the 186 `val`
moments; the 37 `overlap_train_val` moments are left out by the gate. nuScenes and NAVSIM run on
fixed subsets (1,000 and 1,018 samples), neuroNCAP on 15 of its 20 scenario pairs, Bench2Drive on 7
of 220 routes. The gate marks every candidate run that sets its own checkpoint as "invalid" and
notes that the baselines ran older Paddock code; for KITScenes, nuScenes and NAVSIM the baselines
also predate Drive-JEPA's momentum default. Full notes: `11_regression_gate/RESULTS.md`.

What the closed-loop rows contain (`11_regression_gate/RESULTS.md`, read from the run records):

- **Bench2Drive:** the car moved 0.0 m on 6 of the 7 routes. The simulator ran, the model answered
  every request, and the Bench2Drive code is identical to the baseline's, so this is the model's
  behaviour, not a crash. Whether its plans from a standing start asked for no motion is argued from
  the controller code; the plans were not recorded.
- **AlpaSim:** 5 of 10 scenes worse, 4 the same, 1 better; at-fault collisions 2 → 3, off-road 1 → 3.
  On the two scenes that start at rest, the first plan is 0.00 m/s.
- **neuroNCAP:** neutral by the test, but collisions 6 → 8 of 15 (3 avoided, 5 new).
- **NAVSIM:** drivable-area failures 23 → 102 of 1,018 scenarios, and less progress.

**The gate's KITScenes inputs differ from training.** The gate feeds the model a **straight**
driving command at every moment, **acceleration 0**, **sideways speed 0**, and a **fixed 2.2 m
camera height**. Training and the held-out evaluation above used the recorded command, the real
acceleration and sideways speed, and a camera height estimated per scene. This explains **0.920 m
(gate) against 0.596 m (held-out evaluation) on the same 223 moments**: the keys, the human paths
(within 1.6e-14 m) and the 3 s window are identical. The command is a small part of it: on the 184
moments whose true command is straight the gap is still 0.27 m. The split among the other inputs was
not measured. **Whether these input differences contribute to the closed-loop results, for example
the Bench2Drive car not moving on 6 of 7 routes, is NOT verified.** The released model's own
KITScenes number also differs for the same reason (2.537 m through the gate, 2.372 m here).

## Checkpoints

**A research record, not for deployment.** No `checkpoint.json`; nothing in `{weights}`. Sizes,
full sha256 and where each file sits are in
[POINTERS.md](../experiments/drivejepa_kitscenes/POINTERS.md), with the runs tried and dropped.

| Name | Contains | Starts from | sha256 (first 16) | How to load |
|---|---|---|---|---|
| Released model | the full released checkpoint | upstream | `740335c800e24859` | upstream agent |
| Encoder only, tuned on KIT | encoder LoRA, answer-key LoRA, guesser, optimiser state (366 MB) | Released model | `2568101fb234cc25` | merge the `lora` tensors into the released encoder (`02_encoder_build/src/lora.py`, as `03_encoder_training/src/f_final.py` does); the merged encoder alone is also kept (`31db4e8dbf846a34`) |
| Generator only, trained on KIT | the generator's 41 tensors (9.5 MB) | Encoder only, tuned on KIT | `e66e505e1a166c01` | load onto the released model with experiment 1's LoRA merged (`09_heldout_comparison/src/final.py`, model 3) |
| Generator and encoder trained together | generator, encoder LoRA, answer-key LoRA, guesser (148 MB) | Generator only, plus experiment 1's LoRA | `c27aae6e392d98e8` | load generator and merge the LoRA onto the released model (`09_heldout_comparison/src/final.py`, model 4) |
| Encoder, generator and scorer trained together | generator, scorer `pred_score` head, encoder LoRA, answer-key LoRA, guesser (154 MB) | Released generator and scorer, plus experiment 1's LoRA | `dc6c99d5b0298d32` | `10_encoder_generator_scorer/src/final5.py` (`build_ours`) |
| — merged export, as gated | one full state_dict, 495 tensors (1.25 GB) | the line above | `91e13bc38aa76ee3` | Paddock's Drive-JEPA node: `DRIVEJEPA_CHECKPOINT=<file>` |

Only the merged export loads in existing inference code. The others are in the training scripts' own
format and need those scripts and the released checkpoint.

## How to reproduce

Each folder runs through its own `run.sh`, which uses Drive-JEPA's Python 3.9 environment and puts
earlier folders' code on `PYTHONPATH`. Put the placeholders back to real locations first (the
experiment README lists them); the scripts have not been run since they were put in. Each needs the
KITScenes scenes, the released checkpoint, the guesser's starting weights and the vendored Drive-JEPA
tree whose files are listed with sha256 in POINTERS.md. One GPU; the training scripts wait for enough free GPU
memory before starting. The times below are the first and last timestamps of each run's own log, on
a GPU shared with other jobs.

| Step | Command (from the folder) | Log span |
|---|---|---|
| 1. split, players, baseline | `./run.sh src/splits.py`, `src/c_players.py`, `src/f_baseline.py` in `02_encoder_build/` | — |
| 1. frame cache, then the chain: cache check, fixed loss set, warm-up, passes, final evaluation | `./run.sh src/a_cache.py build`, then `./chain.sh` in `03_encoder_training/` (`chain2.sh` resumed it after a crash) | cache 1 h 44 min (from its report); warm-up and passes 1–3 05:30–07:24; passes 4–6 and the final evaluation 23:11–23:31 |
| 1. controls | `./run.sh src/close.py random`, `src/close.py zero223` in `04_encoder_checks/` | — |
| 2. data and survey | `./run.sh src/kitdata.py build` in `06_generator_only_fast_rate/`; `src/g0_gpu.py` in `05_generator_survey/` | — |
| 2. fast learning rate | `./run.sh src/phase1.py all` in `06_generator_only_fast_rate/` | 6 min |
| 2. gentle learning rate, three stay-term weights | `./run.sh src/rerun.py all` in `07_generator_only_gentle_rate/` | 1 h 51 min (43, 34 and 33 min per run) |
| 2. together, with the control | `./run.sh src/phase2.py all` in `08_generator_and_encoder_together/` | 1 h 24 min |
| 1, 2. held-out comparison | `./run.sh src/final.py`, then `src/final_report.py` in `09_heldout_comparison/` | 28 min |
| 3. training | `./run.sh src/allthree.py all` in `10_encoder_generator_scorer/` | 3 h 16 min |
| 3. held-out run, tables | `./run.sh src/final5.py`, then `src/report.py` | 5 min |
| 3. export for the gate | `prep_export.py` (CPU), steps as in `11_regression_gate/PREP.md` | — |
| 3. regression gate | `python -m tools.gate_queue submit --model drivejepa --env DRIVEJEPA_CHECKPOINT=<merged export> --tier all --keep-going`, from a Paddock checkout | queued 03:25, finished 05:22 (IST) |

## Limitations

- **Open-loop numbers only, except the regression gate**, and the gate fails. Nothing here shows the
  retrained model drives better.
- **The trained scorer ranks by closeness to the human path, not by safety.** The released scorer was
  trained on NAVSIM's safety-and-progress grades; experiment 3 replaced that target. A candidate close
  to the human path but unsafe now scores high.
- **No scorer-only control.** The scorer was only ever trained together with the generator and
  encoder, so how much of the pick gain needs the other two is not measured. There is also no run of
  experiment 3 without the scorer loss.
- **The menu narrowed**: variety 2.82 → about 1.7 m (1.63–1.78 m on the held-out moments). Fewer
  alternatives are offered.
- **The held-out set is almost all slow driving** (2 moments at 15 m/s or above), and the gate's
  KITScenes verdict leaves out the 37 `overlap_train_val` moments because they share places with
  training.
- **Checkpoints were chosen on 50 picking scenes from `train`, and each run's choice is the best of
  many checks** (up to 20), so its picking-moment number is optimistic; the held-out numbers are the
  ones to quote.
- **One run per setting**, no seeds repeated.
- **The vendored Drive-JEPA tree the code imports has local, uncommitted changes**: two modified
  attention files and three untracked files, including the KITScenes adapter every experiment uses.
  Their sha256 are in POINTERS.md; they are not in any git history.
- **The committed scripts have not been run** since their machine paths were replaced by placeholders.

## Verified and not verified

| Claim | Evidence | Strength |
|---|---|---|
| The five-model held-out table | recomputed for this record from the two saved per-moment files (menus, picks, human paths) with an independent script; every value equal to 3 decimals; groups 221 / 37 reproduce. The script and its inputs are not committed, so a reviewer cannot repeat it | measured by the author |
| The paired 95 % intervals | as computed by the runs' report scripts; not re-run | as recorded |
| The gate table | equal to the job's `verdict.json` | proof |
| The gate's KITScenes input differences | read from Paddock's code at `9585f81` (`11_regression_gate/RESULTS.md` Part E) | verified by reading (as-run report) |
| Training, picking and held-out scenes are disjoint | `tests/test_drivejepa_kitscenes.py` on the committed lists | proof |
| `test` and `test-e2e` never read | the runs' own data rules and reports | argued |
| The scorer loss never reaches the generator's weights | gradient check before training: zero on 41 of 41 generator tensors | as recorded |
| Checkpoint sha256 and contents | hashed and opened on the CPU for this record | proof |
| The held-out rows were produced by the checkpoints hashed here | no as-run file records a trained checkpoint's sha256; the held-out scripts (`09_heldout_comparison/src/final.py`, `10_encoder_generator_scorer/src/final5.py`) load them by path, and each checkpoint's file time is earlier than the held-out result file that used it | argued |
| The vendored Drive-JEPA tree was not edited | no file in it newer than the first checkpoint; its own git status shows only the changes listed | verified |
| The Drive-JEPA files the code imports | traced by importing the modules and building the model on the CPU | measured; a training-only code path could load more |
| The committed files differ from the as-run ones only in paths and folder names | `AS_RUN.sha256` holds the originals' hashes; the originals stay at the training site, and the replacements are many-to-one, so they cannot be undone from the committed text | by construction; not re-verified |
| Bench2Drive plans asked for no motion | the controller's braking rule and the absence of a crash | argued; plans not recorded |
| Input differences contribute to the closed-loop failures | — | **not verified** |

## Where this came from

The folders were renamed at commit time. The reports and scripts inside use the working names.

| Folder here | Working name |
|---|---|
| `01_encoder_survey/` | `Encoder-LoRA-KIT/s0_recon` |
| `02_encoder_build/` | `Encoder-LoRA-KIT/s1_build` |
| `03_encoder_training/` | `Encoder-LoRA-KIT/s2_train` |
| `04_encoder_checks/` | `Encoder-LoRA-KIT/s3_close` |
| `05_generator_survey/` | `Generator-Joint-KIT/s0_recon` |
| `06_generator_only_fast_rate/` | `Generator-Joint-KIT/s1_phase1` ("phase 1") |
| `07_generator_only_gentle_rate/` | `Generator-Joint-KIT/s2_phase1_rerun` ("phase-1 rerun"; the kept checkpoint is "p9.5", stay-term weight "w") |
| `08_generator_and_encoder_together/` | `Generator-Joint-KIT/s3_phase2` ("phase 2", and "control") |
| `09_heldout_comparison/` | `Generator-Joint-KIT/s4_final` |
| `10_encoder_generator_scorer/` | `Generator-Joint-KIT/s5_all_three` |
| `11_regression_gate/` | `Generator-Joint-KIT/s5_all_three/regression` |

Model names in the reports: "pass 3" or "pass-3 LoRA" is Encoder only, tuned on KIT; "p9.5" is
Generator only, trained on KIT; "phase 2 p1.0" is Generator and encoder trained together; "4.5
passes" or "p04.5" is Encoder, generator and scorer trained together. "1b" is closest candidate FDE
(3 s); "selection" and "MAIN" are the picking moments; "val + overlap" is the held-out set.
