# Pointers — what H16 produced that is not in the repository

Nothing below is committed: checkpoints by rule (`.gitignore`, `docs/checkpoints.md`), the rest for size. Locations are
templates: `{weights}`, `{runs}` and `{data.NAME}` come from `gallops.toml`. At the training site these have not been moved
into the template layout; the "where it sits" column says where each one is, with `{runs}` the site's run root and
`{ops}` its paddock-ops folder. "Recorded" sha256 were read from the file written beside the checkpoint when it was saved,
not hashed again; the rest were hashed on 6 Nov, when this record was written.

## Weights

| Item | Template location | Where it sits at the training site | Bytes | sha256 |
|---|---|---|---:|---|
| Merged encoder, pass 3 of 3 (`H16.r1`), no sidecar | `{weights}/drivejepa_b2d_encoder/h16-r1-lora/step_000000003/merged_encoder_pass_03.pt` | `{runs}/b2d_drivejepa/phase1/merged/p1_05nov_pass_03/` | 1,215,651,437 | `339d74428a15de9d20976568cae9decbe71f3ac81286b7085a68bd6fe53d5e94` (recorded) |
| Stage 2, epoch 18 of 20 (`H16.r2`), export for paddock, no sidecar | `{weights}/drivejepa_p2/h16-r2-stage2/step_000000018/agent.ckpt` | `{runs}/b2d_drivejepa/phase2/runs/p2_05nov/ckpt_epoch18/` | 1,245,093,039 | `ed07e8a10f6934a20aab3ac3481498a123f7b429ded43ba0087985db7b4fa27a` (recorded) |
| A copy of that file, made 6 Nov 14:44 for the disk load, after all 13 scored routes | none | `{djexp_p3}/Drive-JEPA-cache/p2_epoch18_agent.ckpt` | 1,245,093,039 | not hashed |
| Training state of every epoch, `H16.r2` (model, optimiser, scaler, RNG) | none | `{runs}/b2d_drivejepa/phase2/runs/p2_05nov/ckpt_epochNN/state.pt`, 20 of them; epoch 18 is 3,735,293,277 bytes | | not hashed |
| Training state, `H16.r3`, epochs 1 to 8 so far | none | `{runs}/b2d_drivejepa/phase2/runs/p2ft_06nov/ckpt_epochNN/state.pt` | | not hashed |
| LoRA saves of `H16.r1` (LoRA, slow copy, guesser, optimiser, RNG) | none | `{runs}/b2d_drivejepa/phase1/runs/p1_05nov/save_*/state.pt`; pass 3 is `save_0014_pass_03` | | not hashed |

## Released files the runs read

| Item | Where | sha256 |
|---|---|---|
| Released agent checkpoint | `{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt` | `740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05` (checked by the run) |
| V-JEPA pretraining checkpoint, the guesser's source | `{data.navsim.exp}/Drive-JEPA-cache/vitl_merge_3dataset_e50.pt` | `4649182770ef68f84a001780c6579435345948fd80f186fa3616ab078ced668f` (checked by the run) |

## Built data

| Item | Built by | Where it sits at the training site | Bytes | sha256 |
|---|---|---|---:|---|
| Vocabulary, 256 | `gallops.drivejepa_b2d_retrain.build_vocab` | `{ops}/b2d_drivejepa/vocab/b2d_256.npy` | 123,008 | `86d539a3a498434b61180a04084ab7a8e60c522cbf8378133117e091d23e7ae5` |
| Vocabulary, 512 | same | `.../b2d_512.npy` | 245,888 | `307982862839037fc74bc6572603db0f906b7cebb9d8ba3ed8574bd67ad2b95f` |
| Vocabulary, 1,024 | same | `.../b2d_1024.npy` | 491,648 | `0385e690aa1b4ec79330a5d828ab86e797075aacffef0ef728643b31427ef3cb` |
| **Vocabulary, 2,048, the one trained on** | same | `.../b2d_2048.npy` | 983,168 | `9a39f95a8508f2d524dff6fe6ab74019adf17c2f83651ba6dbc7590130ff7f71` |
| Vocabulary, 4,096 | same | `.../b2d_4096.npy` | 1,966,208 | `b2d21a5681db4a5bc5f5931dfa76cbc0be3a52c114f52b0d9b2d8d6a81ecf065` |
| Vocabulary, 8,192 | same | `.../b2d_8192.npy` | 3,932,288 | `0cbb26c49be2c32f20ba4f8800e43fd3e5fdac75ce7ba85a383d93920891a4c6` |
| The expert's paths | `scripts/vocab/extract_paths.py` | `{ops}/b2d_drivejepa/vocab/paths_cache.npz` | 57,220,044 | `9ba42417f6c8762b3ada11ddcfadf84a7a9fad4885a21c7f3b3d095c9f2f2154` |
| The frames used | same | `.../frames_used.csv` | 2,257,919 | `a8617eea6678e4c8c4b4c4daa7cec703f14c55cbbfcb614e3a7a88fc6b00af3e` |
| **Teacher index, Step A v2, the one trained on** | `gallops.drivejepa_b2d_retrain.teachers` | `{data.b2d_retrain}/teachers_2048_v2.npz` | 20,554,918 | `8f1ed0972e6b18fa643297ca7979376beb19fe2fcc91fd00a938ff94a987a22f` |
| Teacher index, Step A v1, superseded, never trained on | `scripts/phase2/teachers_v1_copy.py` | `{data.b2d_retrain}/teachers_2048.npz` | 19,849,332 | `c065c90c5ae45e10139a5acd5144d87146f79a907793136da0744ef5571b9722` |
| Grade files, v2: every graded path and term | `gallops.drivejepa_b2d_retrain.grade_all` | `{data.b2d_retrain}/grade_v2/<clip>.npz`, 944 | | not hashed: one file per clip |
| Grade files, v1, superseded | `scripts/phase2/grade_all_v1_copy.py` | `{data.b2d_retrain}/grade/<clip>.npz` | | not hashed |
| Scorer context | `scripts/phase2/build_context.py` | `{data.b2d_retrain}/context/<clip>.npz` and `.x.npz`, 1,888 | | not hashed: one pair per clip |
| Ego inputs as the runner sends them | `scripts/phase2/build_inputs.py` | `{data.b2d_retrain}/inputs/ego_inputs.npz` | 11,824,438 | `78fa3d5c23cec4e76b8b4dd5f395c71b572dec21826e0ccae40c5b5569a7a5a6` |
| Route lengths, the expert's distance over each clip | **no script survives** | `{data.b2d_retrain}/inputs/route_len.json` | 66,396 | `a598ae44f2904ba5b421f286218948c75f843cb9b52560e5e329220fb54963a3` |
| Own-lane-blocked flags | `scripts/phase2/blocked_all.py` | `{data.b2d_retrain}/inputs/blocked_all.json` | 6,068,620 | `ec60c231eec14ac31e0a9baafdea03a4dc86dc31a89a1ea380904527715180e2` |
| Held-out moments; Stage 2 reads its lidar2img and img_shape | `scripts/phase1/build_ade_set.py` | `{runs}/b2d_drivejepa/phase1/ade_set.pt` | 2,906,243 | `6cb237a6744bfa6515e9c35957e9bcd1a36caae08d304c346d63ea1cc6382c65` |
| Frame cache, 46,700 frames, about 18 GB | `gallops.drivejepa_b2d_retrain.build_cache` | `{data.b2d_phase1}/frame_cache/<clip>.bin`, `index.json` | | not hashed |

The split itself is tracked: `recipes/drivejepa_b2d_retrain/clips.json`, sha256
`f11e6c7b1ff3739b28d7b7621c17de35180eecfdaea58769d03060efbb0d4916`, byte for byte the file the runs read.

## Code the runs imported that is not added here

Sid's five files are in this repository at e75aaac (branch `train/drivejepa-kitscenes`, PR #24), under
`experiments/drivejepa_kitscenes/`. The byte copies Phase 1 imported match those blobs:

| File at e75aaac | sha256 |
|---|---|
| `02_encoder_build/src/common.py` | `e3e14116b8407dd6c559d5947a3f985346d5da05cde90b67b3c73de5f5d9bcb1` |
| `02_encoder_build/src/lora.py` | `ecbde53759b6b52acb7a098123ee039675e69897cbddfd200ac3b3ac8a85bc1c` |
| `02_encoder_build/src/players.py` | `67d1f0b38bb998d3ce25633cc94e75e9e1fcf1311c71caabb44f3fcb24cd2e03` |
| `03_encoder_training/src/masks2.py` | `9192b3afebad1e150dd0ed477c5f28ae128c6d57202d7653a72b01998c25dbed` |
| `03_encoder_training/src/f_final.py` | `084b830b8982c8232e3391e0f501d984be65ee41a1d63cc16ec219d8ba05a085` |

Site tooling the as-run scripts import, in no repository of ours unless the row says so. The scripts in `scripts/` that
use it are a record, not runnable from here.

| File | Used by | sha256 |
|---|---|---|
| `{ops}/mtd/build_context_b2d.py` | `scripts/phase2/build_context.py` | `6258d18b79cd4e2e8251241bb2c228ef8f90327d3ca5e4e3a56f7856b828d078` |
| `{ops}/mtd/stage4/sca_fix.py`, the batch patch the trainers applied at runtime; `gallops.drivejepa_b2d.batch_fix` is its port | `H16.r2`, `H16.r3` | `55a3cc05b7df63d974a6d055b9b4c77bdffb1a6c33eb8b5512591147af1c9a98` |
| `{ops}/b2d-labels/v6/blocked_v6.py` | `scripts/phase2/blocked_all.py` | `bd81ae4e3d3892a5cd2adb083664a3763d372c9eff066149854088f22b6c03ac` |
| `{ops}/phase0a-labels/labels_b2d.py` | `scripts/vocab/extract_paths.py` | `eec74a0a4559f7ea39c69d32e844308c995daf6e98535134b5156f34d8de7b88` |
| `{ops}/run3a-adapted/phase1/data/geometry.py`, a local git repository at 24aacaa | `scripts/vocab/extract_paths.py` | `221a395886d015a6099c683bdf5335e3922435556a85b6ed6fa556bf79eb28ea` |
| `.../phase1/data/targets.py` | same | `d4879d087738836424c39d1b3ba103db05a41c362d8d3b4ea5b61ebf63b95621` |
| `.../phase1/data/dataset.py` | `scripts/phase1/build_ade_set.py` | `c2126a7dc596b74218a1cabbbcd602fe3b1a863a773392f8ee0dae2725940ae4` |
| `.../phase1/config.yaml` | same | `8eba03065e6f6f2492c2527d5c1108fce9407cf64e5cbe0d537d7e5d017207cb` |

`mtd/scorer.py` is in `src/` as `scorer.py`: its sha256 as it ran is in `AS_RUN_SHA256.txt`.

**The Drive-JEPA tree.** Every run imported `{code.drivejepa}`: upstream `e21f474`, with paddock's local changes, as H07
recorded: `navsim_v1/navsim/common/dataclasses.py`, `navsim_v2/navsim/planning/script/run_pdm_score_sequential.py`,
`spatial_cross_attention.py` and `temporal_self_attention.py` modified, `msda_compat.py` and `scripts/inference/` untracked.

**The paddock adapter.** `drivejepa_p2`, uncommitted, in worktrees `rerank-native` (at f3c118d, itself with uncommitted
changes; the 50 routes) and `gate-oct27` (at 3daf4b9; the gate), and `gate-oct27-alp` (at 9819e36) for AlpaSim. `paddock/` in
this folder holds `policy.py` and both `model.yaml`, machine roots as placeholders; their sha256 as they ran are in
`AS_RUN_SHA256.txt`.

## Run records

| Item | Where it sits at the training site |
|---|---|
| Closed-loop records, per route: `results.json`, the 13 scored | `{runs}/b2d_drivejepa/phase2/closed_loop/repa/bench2drive/drivejepa_p2/` |
| Run 3A's routes, compared with | `{runs_p3}/run3a_cmdfix/repa/bench2drive/wajepa_dj_run3a/` |
| Released model's routes, compared with | `{ops}/native-50/dj_native50.json` |
| Gate state, manifests, per-subset logs | `{runs}/b2d_drivejepa/phase2/gate/gate/smoke/drivejepa_p2-{openloop,ncap,alpasim}/` |
| Gate candidates' run records | `results/{kitscenes,nuscenes,neuroncap}/drivejepa_p2/` in worktree `gate-oct27` |
| Daily logs | `agent_memory/05-11-26.md`, `06-11-26.md` in the training site's workspace |
