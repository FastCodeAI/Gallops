<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H18 · The sub-score agent-look model's architecture, trained from scratch on NAVSIM + LEAD with a per-waypoint fact scorer

**Navvrat** · NAVSIM, KITScenes, nuScenes, neuroNCAP, Bench2Drive, AlpaSim, offline · ⏳ running · 18 Nov 2026 · hypothesis

[← all experiments](../../README.md#hypothesis-experiments) · [the account](README.md) · [experiment.toml](experiment.toml)

**Builds on:** [H04](../H04_subscore_scorer_navsim/CARD.md)

## Question

The sub-score agent-look model (H04) learns its pick from NAVSIM's sub-scores, so it trains on NAVSIM alone. A scorer that predicts what happens along each path, waypoint by waypoint (hit something, off road, left the route, red light, stop sign, wrong way, comfort; metres gained, lane offset, closeness, cut-in braking), can be labelled on CARLA (LEAD) as well as on NAVSIM. What does the same architecture do when it is trained end to end on both, with a fresh LoRA on the encoder and that fact scorer choosing among its 32 proposals?

## Hypothesis

Stated by this record from the spec's 'What' line, not by the spec, which sets no size and no pass mark: training H04's architecture end to end on NAVSIM and LEAD together, with the per-waypoint fact scorer picking, gives a model that drives NAVSIM as H04 does while the same weights learn CARLA's scenes. The record's evidence against it would be a regression against H04 or released Drive-JEPA on NAVSIM; Navvrat decides what counts.

## Predictions

⚠️ Not pre-registered: the spec (specs/run3b_v2_spec.md, Navvrat's of 19 Nov) sets no pass marks ('No pass marks. Report numbers next to run 3A's where we already have them'), and no PREDICTIONS.md was written before the run

## What we learned

So far, on navtest (12,146 scenes) the fact-scorer model scores EPDMS 86.35 with NAVSIM's human filter: below the sub-score agent-look model (87.30; paired -0.95, 95 % over logs [-1.79, -0.19]) and released Drive-JEPA (87.78; -1.43 [-2.18, -0.66]), above the 4-camera planner (85.27; +1.08 [+0.18, +2.09]). Collisions and time-to-collision are its best of the four (NC .988, TTC .983), extended comfort far higher (EC .655 against .406), ego progress and drivable area lower (EP .823 against .900, DAC .966 against .984). The regression gate, run by hand, says FAIL against released Drive-JEPA: NAVSIM REGRESSION (0.857 -> 0.807, t -6.05), nuScenes GAIN, KITScenes and neuroNCAP NEUTRAL. On AlpaSim (10 scenes) its scene score is 0.132 against released 0.416 (NEUTRAL, t -1.69) and H04's 0.439 (REGRESSION, t -2.29), with progress 0.174 against 0.473. Bench2Drive's pairs are running.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H18.fact-scorer**<br>The fact-scorer model (called Run 3B v2 while it ran): WA-JEPA ViT-L + fresh LoRA, H04's agent decoder and planner from scratch, per-waypoint fact scorer, NAVSIM + LEAD; last step of a 48 h schedule | released WA-JEPA encoder (base frozen, fresh LoRA with B = 0); agent decoder, planner, path reader and heads from scratch: nothing loaded from H04 or Phase 1 | LoRA r32 / alpha 32 on Q and V of all 24 encoder blocks (lr 0.1 x the heads'), agent decoder (300 slots x 3 layers, detection loss on both datasets), planner (32 proposals x 8 waypoints, 4 rounds, footprint look + agent look; min-L1 to the expert + 0.5 x teachers), H04's auxiliary heads (key agents, ego areas, map auxiliary targets; LEAD), path reader: 7 yes/no + 4 number facts per waypoint, ranking loss weight 0.1, expert's-metres head (20.62 M); frozen: WA-JEPA ViT-L base weights | navsim `navtrain minus the 51 held-out logs` 98,831 samples (train) — 4 cameras x 2 frames (t - 0.5 s, t) at 512 x 256 (Phase 1's JPEG q95 copies), H04's cached ego status, lidar2img and agent targets<br>lead `lead_inputs_v2 training side minus the 416 held-out routes` 116,069 samples (train) — CARLA frames f-2 and f, LEAD's camera rig (4 of its 6 x 60 deg cameras); fact labels carla_full_v1 + carla_full_v1_vru with static.prop.dirtdebris* as debris (pack_v2: 442 of 121,235 records changed hit); POINTERS.md<br>navsim `navtrain, 51 held-out logs (sha1(log) % 20 == 0)` 1,008 samples (held-out) — 1,008 of the 4,457, evenly spaced, the same draw every report; nothing was chosen on them<br>lead `the 416 held-out routes` 1,008 samples (held-out) — 1,008 of 5,166, evenly spaced, the same draw every report<br>navsim `navtest` 12,146 scenes (eval) — never read in training or for any choice | commit `548bb82`<br>Drive-JEPA (linhanwang/Drive-JEPA) @ `548bb82` | `{weights}/run3b_v2/run3b_v2b_19nov/step_000023859/`<br>sha256 (trainable_step023859_final.pt) `12ab840ef81366cb2da936372ffd1b5fddb86d026ab156f5022b6ee68b6afac3`<br>files: trainable_step023859_final.pt `12ab840ef813` |

- **H18.fact-scorer:** Spec: Navvrat's of 19 Nov ~02:15, verbatim in specs/run3b_v2_spec.md; earlier drafts (18 Nov, 19 Nov ~02:00) were superseded before any long run. Trained 19 Nov 03:15 to 20 Nov 09:41 on one GPU shared with other users (scripts/run3b/v2/probe_then_launch.sh, after a 5-step eval-path probe): 23,859 optimizer steps, 7.1 passes of 3,358 steps; batch 64 = 32 NAVSIM + 32 LEAD drawn with replacement by weight (stop-sign frames x2, LEAD frames with a pedestrian or cyclist in our path x4). Adam, heads lr 1e-4, LoRA 1e-5, 500-step warm-up, cosine to 0 over the run, one backward and one step per batch. Teacher forcing on the agent look: share 1.0 for the first pass, then linear to 0.2 at 60 % of the run, noise 0.3 m / 3 deg / 0.5 m/s. Augmentation: angle 1.0 deg, position 0.05 m, zoom x[1, 1.1] on half the samples, one camera dropped in 10 %. Facts labelled live on the GPU for every path the planner proposes (scripts/run3b/gpulab); 0 label errors over the run. Score T4, gamma 0.9, P 0.108; pick = argmax over the 32 of the score from predicted facts with the expert's metres from the head. Step-0 check: the encoder's tokens on 200 NAVSIM held-out samples against H04's cache, cosine median 0.992 (local JPEG re-encodes; Navvrat: 'use local re-encodes'). Held-out reports at step 2,000 and after every pass on 1,008 NAVSIM + 1,008 LEAD samples, the same draw each time. The checkpoint evaluated is the last step, the end of the schedule: no checkpoint was selected (pass 6 had the highest held-out pick, 0.876 against 0.866, n 1,008 each). The code was not committed when it ran, so train_git_sha is the Drive-JEPA tree's upstream commit; the code as run is in scripts/, AS_RUN_SHA256.txt holds each file's sha256 as it ran.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H18.fact-scorer ★ | NAVSIM navtest EPDMS, human filter 87.78 → **86.35** (n=12,146 scenes)<br>full navtest, no momentum-aware selection; paired per scene -1.43, 95 % [-1.86, -1.00] over scenes, [-2.18, -0.66] over the 136 logs; CSV score 82.68 against 84.49; 0 failed scenes | released Drive-JEPA, same evaluation | ●●○○ moderate · adapted, open-loop | the navtest CSV (POINTERS.md), table by run3a's navtest_table.py recombination, pairs by scripts/run3b/v2/navtest/paired.py |
| H18.fact-scorer | NAVSIM navtest EPDMS, human filter 87.30 → 86.35 (n=12,146 scenes)<br>paired per scene -0.95, 95 % [-1.38, -0.53] over scenes, [-1.79, -0.19] over logs; 4,475 better / 2,465 same / 5,206 worse. Against H04's default pick (87.00): -0.65 [-1.06, -0.24] / [-1.45, +0.15] | H04.subscore-agent-look, locked pick | ●●○○ moderate · adapted, open-loop | as the headline |
| H18.fact-scorer | NAVSIM navtest EPDMS, human filter 85.27 → 86.35 (n=12,146 scenes)<br>paired per scene +1.08, 95 % [+0.64, +1.52] over scenes, [+0.18, +2.09] over logs | H01.four-camera-planner | ●●○○ moderate · adapted, open-loop | as the headline |
| H18.fact-scorer | NAVSIM navtest ego progress (EP) .900 → .823 (n=12,146 scenes)<br>the fact-scorer model / H04: NC .988 / .986, DAC .966 / .984, DDC .989 / .993, TLC .998 / .998, TTC .983 / .981, LK .977 / .977, HC .979 / .977, EC .655 / .406; released: EP .891, DAC .986, EC .478; no per-sub-score test | H04.subscore-agent-look, locked pick | ●●○○ moderate · adapted, open-loop | the navtest CSV (POINTERS.md) |
| H18.fact-scorer | offline held-out NAVSIM pick EPDMS (expert's metres from the head) 0.922 → 0.866 (n=1,008 samples)<br>best of the 32: 0.959 against H04's 0.961. H04's numbers on the same samples are from its stored pick-tune dump, not re-run. Agents, NAVSIM 10-20 m recall 0.579 against H04's 0.621 on the same samples; LEAD as Bench2Drive recall / precision 0.552 / 0.795 (H04 on LEAD's 4,990 trial frames, quoted: 0.303 / 0.406) | H04.subscore-agent-look, locked pick, the same 1,008 samples | ●●○○ moderate · adapted | eval_step023859_final.json (POINTERS.md) |

## Regression gate

**H18.fact-scorer** — 20 Nov 2026, all tier, against released Drive-JEPA (the queue's stored runs): ⛔ **FAIL**

| Benchmark | Metric | Baseline | Candidate | t | MDE | Units | Verdict |
|---|---|---|---|---|---|---|---|
| KITScenes | ADE (3 s), m | 2.5375 | 2.8596 | -1.46 | 0.625 | 117 | ⚪ NEUTRAL |
| nuScenes | L2 averaged over 1, 2 and 3 s, m | 2.4314 | 2.1173 | 2.37 | 0.374 | 143 | 🟢 GAIN |
| NAVSIM | EPDMS | 0.8572 | 0.8067 | -6.05 | 0.024 | 126 | 🔴 REGRESSION |
| neuroNCAP | NCAP score, quick check (3 pairs) | 3.1794 | 4.0229 | 0.98 | 4.611 | 3 | ⚪ NEUTRAL |
| Bench2Drive | driving score, crash check (route 3717) |  |  |  |  | 1 | — NOT RUN |
| AlpaSim | scene score, crash check (clip 04749bb9) | 0.0000 | 0.0000 |  |  | 1 | ▫️ TOO FEW UNITS |
| neuroNCAP | NCAP score, closed loop (15 pairs) | 3.6792 | 3.9265 | 1.44 | 0.518 | 15 | ⚪ NEUTRAL |
| Bench2Drive | driving score, closed loop (7 routes) |  |  |  |  | 7 | — NOT RUN |
| AlpaSim | scene score, closed loop (10 scenes) | 0.4162 | 0.1317 | -1.69 | 0.529 | 10 | ⚪ NEUTRAL |

Run by hand (the queue only runs adapters on a pushed branch), keep-going, from the worktrees and site config H04's gate ran under; manifests run3b_v2-released (KITScenes, nuScenes, NAVSIM, neuroNCAP), run3b_v2-alpasim and run3b_v2-b2d under the same gate directory. AlpaSim: the gate printed CONFIG DIFFERS on bridge_args, driver_config and driver_profile (the four-camera render this model reads against Drive-JEPA's front-camera driver), as for H01 and H04; its two pairs are the same runs re-paired with those three keys allowed (tools.gate compare --allow-config). Bench2Drive: scored without the comfort fact (RUN3B_SCORE_DOMAIN=lead, as LEAD samples were in training; the spec masks comfort on all of CARLA); running since 20 Nov 13:33. Quick-tier crash checks are listed before the closed-loop pairs of the same benchmark.

## Evidence outside this repo

- Paddock's run3b_v2 adapter: uncommitted files in two local paddock worktrees (gate-oct27 for the gate, gate-oct27-alp for its AlpaSim subsets), as H04's; copies in paddock/ of this folder, sha256 in AS_RUN_SHA256.txt.
- The checkpoint, the train log, the per-pass held-out reports, the navtest CSV and the gate's verdict files are off-repo at the training site; POINTERS.md lists them with bytes and sha256.
- The fact labels (carla_full_v1 + carla_full_v1_vru, the NAVSIM pool), LEAD's inputs (lead_inputs_v2, its ego-status sidecar and auxiliary targets) and the pool pack (pack_v2) are built data from earlier work at the training site, not built by this record's code; POINTERS.md names them.
