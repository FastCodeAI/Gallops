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
| [H05](H05_ssm_memory/CARD.md) | A state-space memory layer makes plans steadier and reactions earlier | hypothesis | Naren | 21 Oct 2026 | 📝 planned | [H05_ssm_memory/](H05_ssm_memory/) |
| [H06](H06_global_read/CARD.md) | The global read is what makes a model brake for the crossing car | hypothesis | Navvrat | 19 Oct 2026 | ⏳ running | [H06_global_read/](H06_global_read/) |
| [H07](H07_encoder_lora/CARD.md) | A LoRA on the encoder closes the domain gap: the KITScenes gain holds in closed loop and on the base's encoder | hypothesis | Siddesh | 16 Oct 2026 | ⏳ running | [H07_encoder_lora/](H07_encoder_lora/) |
| [H08](H08_sideways_velocity_fov/CARD.md) | Real sideways velocity and a matched camera field of view are free gains | hypothesis | Naren | 21 Oct 2026 | ⏳ running | [H08_sideways_velocity_fov/](H08_sideways_velocity_fov/) |
| [H09](H09_shared_runner/CARD.md) | A shared runner: one queue, one job at a time, with warm neuroNCAP renderers and a persistent CARLA | enabler | Naren | 21 Oct 2026 | ⏳ running | [H09_shared_runner/](H09_shared_runner/) |
| [H10](H10_pdm_terms/CARD.md) | One set of PDM-style terms on every benchmark, combined by NAVSIM's formula (paddock #102) | enabler | Naren | 21 Oct – 22 Oct 2026 | ☑️ done | [H10_pdm_terms/](H10_pdm_terms/) |
| [H11](H11_hugsim/CARD.md) | HUGSIM trial beside neuroNCAP, checked by reproducing WA-JEPA's published 0.4462 | enabler | Navvrat | 21 Oct 2026 | ⏳ running | [H11_hugsim/](H11_hugsim/) |
| [H12](H12_nuscenes_teacher_scenes/CARD.md) | nuScenes teacher-scene selection for the privileged teacher's data collection | enabler | Ruhan | 14 Oct 2026 | ⏳ running | [H12_nuscenes_teacher_scenes/](H12_nuscenes_teacher_scenes/) |
| [H13](H13_alpamayo_teacher/CARD.md) | Alpamayo privileged teacher on neuroNCAP, reusing the tested expert solver | enabler | Ruhan | 14 Oct 2026 | ⏳ running | [H13_alpamayo_teacher/](H13_alpamayo_teacher/) |
| [H14](H14_aux_heads/CARD.md) | Auxiliary prediction heads on Drive-JEPA's refiner, fine-tuned on nuScenes train | hypothesis | Shreedhar | 20 Oct 2026 | 🗑 dropped | [H14_aux_heads/](H14_aux_heads/) |
| [H15](H15_perception_sidecar/CARD.md) | A perception side-car on frozen Drive-JEPA: 3D boxes and each agent's 1 s path, tested as an input to the scorer, to the refiner, and against LoRA on the ViT | hypothesis | Shreedhar | 3 Nov 2026 | ⏳ running | [H15_perception_sidecar/](H15_perception_sidecar/) |
<!-- /experiments:index -->

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
| `gallops.drivejepa_perception.b_analysis` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.b_report` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.cached_release` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.env` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.evaluate` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.explain` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.geometry` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.headroom` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.importance` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.navsim_data` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.navtest` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.nuscenes_data` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.ood` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.pdm_labels` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.preflight` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.replay_agent` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.scorers` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.sidecar` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.targets` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.tokens` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.train_refiner` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.train_scorer` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.train_sidecar` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.train_vit_lora` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.veto` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.views` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
| `gallops.drivejepa_perception.vit_lora` | H15.s0, H15.a0, H15.a1, H15.a2, H15.b1, H15.b2, H15.b3, H15.b32, H15.ood, H15.r1, H15.r2, H15.r3, H15.r4, H15.r5 | experiment |
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
