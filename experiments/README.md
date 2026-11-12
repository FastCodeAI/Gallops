# Experiments

One folder per experiment, named `<ID>_<slug>`. Each holds:

- `experiment.toml`: the facts. Who ran it, what it started from, what it trained on, which
  checkpoint it wrote (by weights sha256), what it measured and how strongly, and what the
  gate said. The repository README and the card are rendered from these files.
- `CARD.md`: the facts on one page, rendered from `experiment.toml`. The repository README's IDs
  link here. Never edit it by hand; `make check` fails when it differs from the record.
- `README.md`: the account. Why, what was done, what it showed, what was decided.
- `PREDICTIONS.md` (hypotheses): what was expected, written and committed before the first run.
- The scripts as they ran, and the small tables the numbers come from.

The rules — IDs, names, versions, where code and data go — are in
[docs/experiments.md](../docs/experiments.md). Start a new experiment with
`gallops experiments new <ID> <slug> --owner <name>`, which copies [_template/](_template/).

GC05 also keeps its step-by-step record as it ran: one numbered folder per step, explained by
[GC05_physicalai_retrain/RECORD.md](GC05_physicalai_retrain/RECORD.md). Records marked "on branch"
live on a branch that has not been merged yet; their `experiment.toml` is here so the evidence is
indexed now, and the record moves into the folder when the branch merges.

## Index

<!-- experiments:index -->
| ID | Experiment | Track | Owner | When | Status | Folder |
|---|---|---|---|---|---|---|
| [GC01](GC01_rerank_kitscenes/CARD.md) | A re-ranker over Drive-JEPA's 32 candidates, trained on KITScenes train | gap-closing | Naren | 16 Sep – 12 Oct 2026 | ◐ mixed | [GC01_rerank_kitscenes/](GC01_rerank_kitscenes/) |
| [GC02](GC02_selection_without_training/CARD.md) | Selection changes with no training: momentum, the model's own signals, injected swerves | gap-closing | Naren, Siddesh | 23 Sep – 13 Oct 2026 | ◐ mixed | [GC02_selection_without_training/](GC02_selection_without_training/) |
| [GC03](GC03_swerve_finetune/CARD.md) | The swerve finetune: teaching Drive-JEPA to evade on reconstructed scenes with inserted hazards | gap-closing | Naren | 26 Sep – 12 Oct 2026 | ⏸ parked (inconclusive) | [GC03_swerve_finetune/](GC03_swerve_finetune/) |
| [GC04](GC04_bench2drive/CARD.md) | Drive-JEPA retrained on the Bench2Drive base set: Split-v4 and the Final | gap-closing | Navvrat | 29 Sep – 13 Oct 2026 | ✅ positive | [GC04_bench2drive/](GC04_bench2drive/) |
| [GC05](GC05_physicalai_retrain/CARD.md) | Drive-JEPA retrained on NVIDIA Physical AI AV | gap-closing | Siddesh | 1 Oct – 12 Oct 2026 | ✅ positive | [GC05_physicalai_retrain/](GC05_physicalai_retrain/) |
| [GC06](GC06_scorer_cache/CARD.md) | Drive-JEPA's scorer trained on cached features, held out scene by scene | gap-closing | Naren | 9 Oct – 12 Oct 2026 | ⏸ parked (inconclusive) | [GC06_scorer_cache/](GC06_scorer_cache/) |
| [GC07](GC07_subscore_scorer/CARD.md) | Sub-score scorer over inserted swerve paths, on the seeing model | gap-closing | Naren | 13 Oct 2026 | ⏸ parked (mixed) | [GC07_subscore_scorer/](GC07_subscore_scorer/) |
| [GC08](GC08_expert_v6_imitation/CARD.md) | Expert v6 (brake, then swerve for oncoming cars) and its imitation rounds 1A, 1B and 2 | gap-closing | Naren | 13 Oct – 16 Oct 2026 | ❌ negative | [GC08_expert_v6_imitation/](GC08_expert_v6_imitation/) |
| [H01](H01_four_camera_base/CARD.md) | The 4-camera stack as the base: WA-JEPA's frozen encoder under Drive-JEPA's planner | hypothesis | Navvrat | 16 Oct 2026 | ⏳ running | [H01_four_camera_base/](H01_four_camera_base/) |
| [H02](H02_smallest_valid_gate/CARD.md) | Smallest valid gate: subset size per benchmark from the spread of past paired runs | enabler | Naren | 20 Oct – 21 Oct 2026 | ☑️ done | [H02_smallest_valid_gate/](H02_smallest_valid_gate/) |
| [H03](H03_shared_store_and_tags/CARD.md) | One shared folder and group; every checkpoint tagged by weights sha256 with full metadata | enabler | Naren | 21 Oct 2026 | ⏳ running | [H03_shared_store_and_tags/](H03_shared_store_and_tags/) |
| [H04](H04_subscore_scorer_navsim/CARD.md) | Training the scorer on the sub-scores closes the base's NAVSIM gap | hypothesis | Navvrat, Siddesh | 22 Oct 2026 | ⏳ running | [H04_subscore_scorer_navsim/](H04_subscore_scorer_navsim/) |
| [H05](H05_ssm_memory/CARD.md) | A state-space memory layer makes plans steadier and reactions earlier | hypothesis | Naren, Siddesh | 21 Oct – 4 Nov 2026 | ⏸ parked (inconclusive) | [H05_ssm_memory/](H05_ssm_memory/) |
| [H06](H06_global_read/CARD.md) | The global read is what makes a model brake for the crossing car | hypothesis | Navvrat | 19 Oct 2026 | ⏳ running | [H06_global_read/](H06_global_read/) |
| [H07](H07_encoder_lora/CARD.md) | A LoRA on the encoder closes the domain gap: the KITScenes gain holds in closed loop and on the base's encoder | hypothesis | Siddesh | 16 Oct 2026 | ⏳ running | [H07_encoder_lora/](H07_encoder_lora/) |
| [H08](H08_sideways_velocity_fov/CARD.md) | Real sideways velocity and a matched camera field of view are free gains | hypothesis | Naren | 21 Oct 2026 | ⏳ running | [H08_sideways_velocity_fov/](H08_sideways_velocity_fov/) |
| [H09](H09_shared_runner/CARD.md) | A shared runner: one queue, one job at a time, with warm neuroNCAP renderers and a persistent CARLA | enabler | Naren | 21 Oct 2026 | ⏳ running | [H09_shared_runner/](H09_shared_runner/) |
| [H10](H10_pdm_terms/CARD.md) | One set of PDM-style terms on every benchmark, combined by NAVSIM's formula (paddock #102) | enabler | Naren | 21 Oct – 22 Oct 2026 | ☑️ done | [H10_pdm_terms/](H10_pdm_terms/) |
| [H11](H11_hugsim/CARD.md) | HUGSIM trial beside neuroNCAP, checked by reproducing WA-JEPA's published 0.4462 | enabler | Navvrat | 21 Oct 2026 | ⏳ running | [H11_hugsim/](H11_hugsim/) |
| [H14](H14_aux_heads/CARD.md) | Auxiliary prediction heads on Drive-JEPA's refiner, fine-tuned on nuScenes train | hypothesis | Shreedhar | 20 Oct 2026 | ⏳ running | [H14_aux_heads/](H14_aux_heads/) |
| [H15](H15_prediction_memory/CARD.md) | Carrying the planner's output embeddings across moments makes the picker steadier and earlier | hypothesis | Naren, Siddesh | 5 Nov 2026 | 📝 planned | [H15_prediction_memory/](H15_prediction_memory/) |
<!-- /experiments:index -->

## Enablers

What the hypotheses need before they can be decided.

<!-- experiments:enablers -->
| ID | Enabler | Owner | Priority | Status | What it gives the team |
|---|---|---|:-:|---|---|
| [**H02**](H02_smallest_valid_gate/CARD.md) | Smallest valid gate | Naren | 2 | ☑️ done | Smallest drop each subset detects (80% power): neuroNCAP closed tier 0.73 for a small change and 2.85 for a big one, out of 5; its quick tier 2.5 / 6.7, a pipeline check only; NAVSIM about 0.0035; KITScenes 0.51 m for a big change; Bench2Drive only 17–38 points; AlpaSim about 0.23, argued. |
| [**H03**](H03_shared_store_and_tags/CARD.md) | Shared store and checkpoint tags | Naren | 3 | ⏳ running | So far: the paddock account and a paddock group holding the three of us exist on shakaal; paddock's gate queue (#109, 26 Oct) copies every submitted checkpoint into a shared store named by its sha256. |
| [**H09**](H09_shared_runner/CARD.md) | Shared gate runner | Naren | 9 | ⏳ running | So far: the queue landed (paddock #109, 26 Oct) and runs as the paddock account, one job at a time, with each submitted checkpoint copied into a shared store by its sha256 and the code pinned to a pushed commit; by 27 Oct it had made Drive-JEPA's quick- and closed-tier baselines and gated a teammate's candidate. |
| [**H10**](H10_pdm_terms/CARD.md) | PDM-style terms on every benchmark | Naren | 10 | ☑️ done | paddock #102, merged 22 Oct: NAVSIM's five terms print under every gate pair, as far as each benchmark can measure them, and never change a verdict. neuroNCAP has all five by NAVSIM's rules, its collision check agreeing with neuroNCAP's own on 600 of 600 stored runs; NAVSIM has all five; AlpaSim NC, DAC and EP; nuScenes NC, EP and C; Bench2Drive NC and EP; KITScenes and nuPlan EP and C. |
| [**H11**](H11_hugsim/CARD.md) | HUGSIM trial | Navvrat | 11 | ⏳ running | So far: HUGSIM is a paddock benchmark (#118, merged 27 Oct PDT, 28 Oct IST) on 420 of the release's 436 scenarios (16 fail for every model), with upstream PR #57's controller fix. |
<!-- /experiments:enablers -->

## Regression gate log

Every checkpoint we would keep goes through paddock's gate
(`python -m tools.gate`, paddock `docs/gate.md`): candidate minus baseline on
fixed subsets, paired sample by sample, standard errors clustered by scene, pair
or route. Any REGRESSION fails the checkpoint; a closed-loop GAIN with nothing
regressing ships it; anything else holds. Bench2Drive's 7 routes and the quick
tier's 3 neuroNCAP episodes can fail a checkpoint but cannot ship one.

<!-- experiments:gate -->
| Date | Run | Checkpoint | Tier | NAVSIM | KITScenes | nuScenes | neuroNCAP | Bench2Drive | AlpaSim | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 Oct 2026 | [H02.timing](H02_smallest_valid_gate/CARD.md) | `740335c800e2` | all | ⚪ 0.875 → 0.875 | ⚪ 3.222 → 3.222 | ⚪ 2.497 → 2.497 | ⚪ 3.175 → 3.175 | ⚪ | — | ⏸ **HOLD** |
| 28 Oct 2026 | [H07.exp3](H07_encoder_lora/CARD.md) | `91e13bc38aa7` | all | 🔴 0.8572 → 0.7533 | 🟢 2.5375 → 0.9204 | 🟢 2.4314 → 0.9152 | ⚪ 3.6792 → 2.7265 | 🔴 59.0360 → 0.2686 | 🔴 0.4344 → 0.1158 | ⛔ **FAIL** |
<!-- /experiments:gate -->

## Decision log

What the team decided, in date order, on what evidence. It is kept in
[decisions.toml](decisions.toml).

<!-- experiments:decisions -->
| # | Date | Decision | By | Evidence | Led to |
|---|---|---|---|---|---|
| **D01** | 19 Sep 2026 | **Fix the selection, not the head.** First target: Drive-JEPA's choice among its 32 candidates, not its trajectory head. On KITScenes the best candidate is 0.37 m from the human path and the pick 2.36 m; a re-ranker over the frozen model's candidates attacks that gap and proves the train → checkpoint → paddock loop on the way. | Naren | — | [GC01](GC01_rerank_kitscenes/CARD.md) |
| **D02** | 26 Sep 2026 | **neuroNCAP needs both halves trained.** Selection alone cannot fix neuroNCAP: no signal in the released model picks a clear plan on the rescue ticks (0 of 322), and the scorer rates injected swerves below the colliding plan. Fine-tune both halves — propose evasive swerves and choose them — on hazards inserted into scenes outside the benchmark. | Naren | [GC02.signals](GC02_selection_without_training/CARD.md), [GC02.inject](GC02_selection_without_training/CARD.md) | [GC03](GC03_swerve_finetune/CARD.md) |
| **D03** | 6 Oct 2026 | **No run trains until the preflight passes.** Defect 6: swerve runs v1–v4 graded every tick from the first tick's pose. From now on every training run passes the preflight — code committed, grading reproduced, inputs rebuilt faithfully — before its first step, and there is no way to skip it. | Naren | [GC03.v4](GC03_swerve_finetune/CARD.md) | [GC03.v4fix](GC03_swerve_finetune/CARD.md) |
| **D04** | 9 Oct 2026 | **Train the scorer on cached features.** s1–s3 ended at a random ranking loss. Train the scorer alone on cached features from the frozen encoder and refiner, with sub-score heads, so a run takes minutes and can be checked scene by scene. | Naren | [GC03.s1](GC03_swerve_finetune/CARD.md), [GC03.s2](GC03_swerve_finetune/CARD.md), [GC03.s3](GC03_swerve_finetune/CARD.md) | [GC06](GC06_scorer_cache/CARD.md) |
| **D05** | 10 Oct 2026 | **Ship the retrained generator as 0.1.7.** After its 30-scene trial, the second run's merged checkpoint — released encoder, first-run scorer, second-run generator — drives AlpaSim as driver 0.1.7. | Siddesh | [GC05.run2](GC05_physicalai_retrain/CARD.md) | — |
| **D06** | 12 Oct 2026 | **Everything so far was camera-blind.** Paddock and the trainer placed the cameras relative to nuScenes' LIDAR_TOP, so no image reached the model in any swerve or scorer-cache run. Those numbers stand as what they measured, not as evidence about seeing; the caches are to be rebuilt, and the preflight now refuses a model a blanked camera does not move. | Naren | [GC03](GC03_swerve_finetune/CARD.md), [GC06](GC06_scorer_cache/CARD.md) | [GC07](GC07_subscore_scorer/CARD.md) |
| **D07** | 12 Oct 2026 | **A closed-loop gain must be a reaction.** A closed-loop gain counts only once the switches behind it are shown to differ between scenarios: the scorer-cache model's 0103 gain was a drift that happened to clear the car, not an avoidance. | Naren | [GC06.relative-0103](GC06_scorer_cache/CARD.md) | — |
| **D08** | 13 Oct 2026 | **Park the scorer; imitate expert v6.** Park the sub-score scorer: on the seeing model it overrode the pick from 0.5 s in every scenario and flipped side. Teach the manoeuvre through the model's own generator instead, with expert v6 — brake for stationary and side hazards, brake then swerve onto free road for oncoming ones — as the one consistent driver to imitate. | Naren | [GC07.sighted-v1](GC07_subscore_scorer/CARD.md) | [GC08](GC08_expert_v6_imitation/CARD.md) |
| **D09** | 14 Oct 2026 | **One tick per forward pass.** The vendor model mixes ticks within a batch, so every forward pass is now one tick. The batch-8 swerve runs and the batch-4 cache v1 would train differently today; a rerun is a new run. | Naren | [GC03](GC03_swerve_finetune/CARD.md), [GC06.v1](GC06_scorer_cache/CARD.md) | — |
| **D10** | 16 Oct 2026 | **Brake-then-swerve imitation: no transfer.** Concluded negative: the swerve is learned on the five training scenes but lost on unseen streets (round 2 on 0103: 2.31 against 3.01 released), and each new reconstructed scene costs about 7.5 GPU-hours. | Naren | [GC08.v6-r1a-loso0075](GC08_expert_v6_imitation/CARD.md), [GC08.v6-r1b-all](GC08_expert_v6_imitation/CARD.md), [GC08.v6-r2-short](GC08_expert_v6_imitation/CARD.md) | — |
| **D11** | 19 Oct 2026 | **Base: Navvrat's 4-camera stack.** The three tracks merge into one base: Navvrat's 4-camera stack — WA-JEPA's encoder over four views, Drive-JEPA's planner with path reads and a global read. All three tracks found the scorer is the weak part. Still open: close its NAVSIM gap before the gate, rebase its adapter onto the camera fix, repeat the LoRA on its encoder. | team | [H01.run2](H01_four_camera_base/CARD.md), [H07.exp1](H07_encoder_lora/CARD.md), [GC08](GC08_expert_v6_imitation/CARD.md) | [H01](H01_four_camera_base/CARD.md), [H04](H04_subscore_scorer_navsim/CARD.md), [H07](H07_encoder_lora/CARD.md) |
| **D12** | 21 Oct 2026 | **One prioritised list, H01–H11.** The hypothesis phase runs as one list in priority order, proposed 21 Oct: enablers first, then the hypotheses that most directly get the base through the gate. Dropped: the scene-count curve (the imitation overfits, and a scene costs about 7.5 GPU-hours) and more cameras for side crashes (the front camera already sees the crossing car). | team | [GC08](GC08_expert_v6_imitation/CARD.md) | [H01](H01_four_camera_base/CARD.md), [H02](H02_smallest_valid_gate/CARD.md), [H03](H03_shared_store_and_tags/CARD.md), [H04](H04_subscore_scorer_navsim/CARD.md), [H05](H05_ssm_memory/CARD.md), [H06](H06_global_read/CARD.md), [H07](H07_encoder_lora/CARD.md), [H08](H08_sideways_velocity_fov/CARD.md), [H09](H09_shared_runner/CARD.md), [H10](H10_pdm_terms/CARD.md), [H11](H11_hugsim/CARD.md) |
| **D13** | 21 Oct 2026 | **The gate: own scores decide, MDE printed.** Each benchmark's own score keeps deciding the gate; the PDM-style terms are shown beside it and may decide later, one benchmark at a time. The subset sizes stay (20 pairs × 1 seed was not adopted), and every verdict prints its minimum detectable effect. | Naren | [H02.mde](H02_smallest_valid_gate/CARD.md), [H02.timing](H02_smallest_valid_gate/CARD.md) | [H10](H10_pdm_terms/CARD.md) |
| **D14** | 21 Oct 2026 | **neuroNCAP stays; trial HUGSIM beside it.** neuroNCAP stays the crash benchmark; HUGSIM is trialled beside it on a slice, checked by reproducing WA-JEPA's published 0.4462, before deciding whether to adopt it. | team | — | [H11](H11_hugsim/CARD.md) |
| **D15** | 23 Oct 2026 | **Gate subsets redrawn at random.** Every gate subset is drawn at random within groups of turns, speeds and places (seed 20261022), replacing the hand-forced subsets of 21 Oct; neuroNCAP's closed tier becomes 15 pairs × 1 seed. | Naren | [H02.mde](H02_smallest_valid_gate/CARD.md) | [H02](H02_smallest_valid_gate/CARD.md) |
| **D16** | 23 Oct 2026 | **Drive-JEPA cropped at 2 Hz by default.** Drive-JEPA runs with the OpenScene crop and plans twice a second on every benchmark by default, and the queue's baselines are remade under it. | Naren | [H08.crop](H08_sideways_velocity_fov/CARD.md) | — |
| **D17** | 26 Oct 2026 | **Momentum-aware selection on by default.** Paddock's Drive-JEPA adapter selects with momentum by default, as the published method does; Drive-JEPA baselines from here on include it. | Siddesh | — | — |
| **D18** | 28 Oct 2026 | **KITScenes exp 3 fails the gate.** The encoder + generator + scorer checkpoint trained on KITScenes fails the gate (NAVSIM, Bench2Drive and AlpaSim regress); its checkpoints stay a research record, not for deployment. | Siddesh | [H07.exp3](H07_encoder_lora/CARD.md) | — |
| **D19** | 6 Nov 2026 | **H12 and H13 removed.** The nuScenes scene selector (H12) and the Alpamayo teacher (H13) are not this repo's work: #19 and #18 closed unmerged, and their records are removed, as they served no experiment here. The two IDs stay retired, never reused; the code is at those PRs' heads. | Naren | — | — |
<!-- /experiments:decisions -->

## How to read the results

- **Nothing here is yet a closed-loop safety gain** shown on a split nothing was
  chosen on. The strongest result is the re-ranker's open-loop gain on
  KITScenes, and the same weights do worse on neuroNCAP.
- **Every swerve and scorer-cache run was camera-blind.** Paddock and the
  trainer both placed the cameras relative to nuScenes' LIDAR_TOP, so no image
  reached the model; a black frame gave the same plan. Those numbers are what
  the runs measured, but they say nothing about learning to *see* a hazard. The
  trainer is fixed and its preflight now refuses a model that a blanked camera
  does not move.
- **The swerve runs also trained at batch 8, and scorer cache v1 was built at
  batch 4.** The vendor model mixes ticks within a batch, which did not matter
  while it was blind but would from the first run that sees. Every forward pass
  is now one tick, so any rerun trains differently.
- **Several shipped sidecars are known to be wrong about their own run,** and
  were left as written rather than hand-edited. Each is marked ⚠️ on its
  card.
- **Bench2Drive and AlpaSim runs are one run per route or scene,** so a
  difference of a few points is not a result on its own.

| Mark | Means |
|---|---|
| ●●●● proof · ●●●○ strong · ●●○○ moderate · ●○○○ argued · ○○○○ unverified | how strongly the evidence supports the number: paddock's ladder ([docs/conventions.md](../docs/conventions.md)) |
| `adapted` | trained on the benchmark's own domain |
| `blind` | measured before the 12 Oct camera fix: no image reached the model |
| `smoke` | one scene or a few episodes: a pipeline check, not a result |
| `selected-on-report` | the variant was chosen on the split it is reported on |
| `open-loop` | measured against logged driving, not driven |
| `no-record` | the evidence is not yet committed to any repository |
| ✅ positive · ❌ negative · ◐ mixed · ❔ inconclusive · ☑️ done | how a finished experiment came out (☑️ for an enabler) |
| ⏳ running · 📝 planned · ⏸ parked · 🗑 dropped | an experiment that has not finished |
| 🟢 GAIN · ⚪ NEUTRAL · 🔴 REGRESSION · ▫️ no verdict | a regression-gate pair |
| 🚀 SHIP · ⏸ HOLD · ⛔ FAIL | the gate's overall verdict |
| `29d5f9517c2e` | a checkpoint's tag: the first 12 hex digits of its weights' sha256; the card gives it in full. "—" means it was never recorded |
| **GC05**, **H07**, … | an experiment's ID: it opens the experiment's card, `experiments/<ID>_<slug>/CARD.md` |
| on branch ⎇ | the record is on a branch not yet merged |

## Code map

Every module under `src/gallops`, and the runs whose `[runs.code]` used it. `make check` fails on an
**unclaimed** module: maintained code stays in `src/` (it is linted, tested and imported by
recipes), but no trainer exists here without the evidence it produced. The contract and the
command line serve every experiment and are listed as shared.

<!-- experiments:code-map -->
| Module | Used by | Kind |
|---|---|---|
| `gallops.checkpoint` | — | shared |
| `gallops.cli` | — | shared |
| `gallops.compat` | H05.r1 | experiment |
| `gallops.config` | — | shared |
| `gallops.data.nuscenes_real` | GC06.v3_nusc | experiment |
| `gallops.data.rerank` | GC01.kitscenes-train-h128-t1 | experiment |
| `gallops.drivejepa_b2d.batch_fix` | GC04.final | experiment |
| `gallops.drivejepa_b2d.check_data` | GC04.split-v4, GC04.final | experiment |
| `gallops.drivejepa_b2d.head` | GC04.split-v4, GC04.final | experiment |
| `gallops.drivejepa_b2d.host` | GC04.final | experiment |
| `gallops.drivejepa_b2d.provenance` | GC04.split-v4, GC04.final | experiment |
| `gallops.drivejepa_b2d.sidecar` | GC04.split-v4, GC04.final | experiment |
| `gallops.drivejepa_b2d.train_area_head` | GC04.final | experiment |
| `gallops.drivejepa_b2d.train_generator` | GC04.final | experiment |
| `gallops.drivejepa_b2d.train_ipad_head` | GC04.split-v4 | experiment |
| `gallops.drivejepa_b2d.train_split_head` | GC04.split-v4, GC04.final | experiment |
| `gallops.drivejepa_b2d.write_checkpoint` | GC04.split-v4, GC04.final | experiment |
| `gallops.experiments` | — | shared |
| `gallops.expert.check_scenarios` | GC03.stageA | experiment |
| `gallops.expert.draw` | GC03.stageA, GC03.v1, GC03.v2, GC03.v3, GC03.v4, GC03.v4fix, GC03.l4fix, GC03.l5 | experiment |
| `gallops.expert.drivable` | GC03.l5 | experiment |
| `gallops.expert.fastgrade` | GC06.v1, GC06.v2, GC06.v3_nusc | experiment |
| `gallops.expert.grader` | GC03.stageA, GC03.v1, GC03.v2, GC03.v3, GC03.v4, GC03.v4fix, GC03.l4fix, GC03.l5, GC03.s1, GC03.s2, GC03.s3 | experiment |
| `gallops.expert.label` | GC03.stageA, GC03.v1, GC03.v2, GC03.v3, GC03.v4, GC03.v4fix, GC03.l4fix, GC03.l5, GC03.s1, GC03.s2, GC03.s3 | experiment |
| `gallops.expert.lanes` | GC03.l5, GC03.s1, GC03.s2, GC03.s3 | experiment |
| `gallops.expert.rounds` | GC03.s3 | experiment |
| `gallops.expert.scenarios` | GC03.stageA | experiment |
| `gallops.memory.cache` | H05.r1 | experiment |
| `gallops.memory.checks` | H05.r1 | experiment |
| `gallops.memory.inputs` | H05.r1 | experiment |
| `gallops.memory.probe` | H05.r2 | experiment |
| `gallops.memory.probe_control` | H05.r2 | experiment |
| `gallops.memory.ticks` | H05.r1 | experiment |
| `gallops.recipe` | — | shared |
| `gallops.train.export` | GC03.stageA, GC03.v4, GC03.s1, GC03.s2, GC03.s3 | experiment |
| `gallops.train.losses` | GC03.v1, GC03.v2, GC03.v3, GC03.v4, GC03.v4fix, GC03.l4fix, GC03.l5, GC03.s1, GC03.s2, GC03.s3 | experiment |
| `gallops.train.preflight` | GC03.l4fix, GC03.l5, GC03.s1, GC03.s2, GC03.s3 | experiment |
| `gallops.train.reach` | GC03.stageA, GC03.v4 | experiment |
| `gallops.train.rerank` | GC01.kitscenes-train-h128-t1 | experiment |
| `gallops.train.safety_check` | GC03.v4fix, GC03.l4fix, GC03.l5, GC03.s1, GC03.s2, GC03.s3 | experiment |
| `gallops.train.scorer_cache` | GC06.v1, GC06.v2, GC06.v3_nusc | experiment |
| `gallops.train.scorer_folds` | GC06.v1 | experiment |
| `gallops.train.scorer_labels` | GC06.v1, GC06.v2, GC06.v3_nusc | experiment |
| `gallops.train.swerve` | GC03.stageA, GC03.v1, GC03.v2, GC03.v3, GC03.v4, GC03.v4fix, GC03.l4fix, GC03.l5, GC03.s1, GC03.s2, GC03.s3 | experiment |
| `gallops.train.swerve_data` | GC03.stageA, GC03.v1, GC03.v2, GC03.v3, GC03.v4, GC03.v4fix, GC03.l4fix, GC03.l5, GC03.s1, GC03.s2, GC03.s3 | experiment |
<!-- /experiments:code-map -->
