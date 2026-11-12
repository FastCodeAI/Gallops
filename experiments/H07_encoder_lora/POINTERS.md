# Pointers — what these experiments produced that is not in the repository

Nothing below is committed: weights by rule (`.gitignore`, `docs/checkpoints.md`); per-moment arrays, frame caches and feature caches because they are derived from KITScenes; logs and tables over 100 KB for size, as in `experiments/drivejepa_retrain/`. Each row gives a location, a size and a sha256, so a copy can be recognised wherever it ends up.

**None of these checkpoints has a `checkpoint.json` sidecar, and none was written into `{weights}`.** They are a research record, not checkpoints for deployment: the one that went through Paddock's regression gate failed it (`docs/drivejepa-kitscenes.md`). The template column is where a copy should go if one is ever made usable; the next column is where each file sits at the training site today.

**Locations are templates, not paths.** `{weights}` and `{runs}` are `[roots.weights]` and `[roots.runs]` from `gallops.toml`. On the training site's store the folders under `encoder_lora_kit/` and `generator_joint_kit/` keep short working names (`s1` … `s5`, `w0`, `phase2`, `control`); the "Where this came from" table in the doc maps them to the folders here. The other placeholders are listed in the experiment README.

## Weights

| Item | Template location | Where it sits at the training site | Bytes | sha256 |
|---|---|---|---:|---|
| Released Drive-JEPA checkpoint (start of every experiment) | `{weights}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt` | `upstream release (`LinhanWang/Drive-JEPA`)` | 3735352913 | `740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05` |
| Starting weights of the hide-and-guess guesser (V-JEPA 2 predictor, not a result) | `{weights}/Drive-JEPA-cache/vitl_merge_3dataset_e50.pt` | `upstream release` | 5127748765 | `4649182770ef68f84a001780c6579435345948fd80f186fa3616ab078ced668f` ‡ |
| **Encoder only, tuned on KIT** — chosen check, 3 passes: encoder LoRA, answer-key LoRA, guesser, optimiser | `{weights}/drivejepa_kitscenes/encoder-only/step_000001761/state.pt` | `{runs}/drivejepa_kitscenes/encoder_lora_kit/s2/checkpoints/pass_03/` | 366078108 | `2568101fb234cc25f4e6781834c72659283cb8c5df5569cc9e42f248b9e2aaf0` |
| — its settings, saved beside it | `{weights}/drivejepa_kitscenes/encoder-only/step_000001761/settings.json` | `{runs}/drivejepa_kitscenes/encoder_lora_kit/s2/checkpoints/pass_03/` | 1953 | `027eb799a25ed3d6b9942983c150e63ee932160dece002798bbb052f561a9b29` |
| — the same LoRA merged into the released encoder (encoder weights only) | `{weights}/drivejepa_kitscenes/encoder-only/step_000001761/merged_encoder_pass_03.pt` | `{runs}/drivejepa_kitscenes/encoder_lora_kit/s2/` | 1215651437 | `31db4e8dbf846a34d42d0042416ddd51ec11c8a36036054d88ffe3de6ee5d854` |
| — warm-up end state, the start of its passes | `{weights}/drivejepa_kitscenes/encoder-only/warmup/state.pt` | `{runs}/drivejepa_kitscenes/encoder_lora_kit/s2/checkpoints/warmup_final/` | 88394419 | `a852fee65dee447062da23a54998b9b89b0c3a90a81e47eb0e25dfc2a0139591` |
| **Generator only, trained on KIT** — gentle learning rate, no stay term, 9.5 passes: generator only | `{weights}/drivejepa_kitscenes/generator-only/step_000004133/p09.5.pt` | `{runs}/drivejepa_kitscenes/generator_joint_kit/s2/w0/ckpt/` | 9536586 | `e66e505e1a166c019bd80a4f8c5e02581323a36e4a0ec80b1239d1c21fe019c5` |
| **Generator and encoder trained together** — 1.0 pass: generator, encoder LoRA, answer-key LoRA, guesser | `{weights}/drivejepa_kitscenes/generator-and-encoder/step_000000435/p01.0.pt` | `{runs}/drivejepa_kitscenes/generator_joint_kit/s3/phase2/ckpt/` | 148342095 | `c27aae6e392d98e849463445391490dab94ef9414e920cb874b251183adca915` |
| **Encoder, generator and scorer trained together** — 4.5 passes: generator, scorer pick head, encoder LoRA, answer-key LoRA, guesser | `{weights}/drivejepa_kitscenes/encoder-generator-scorer/step_000001958/p04.5.pt` | `{runs}/drivejepa_kitscenes/generator_joint_kit/s5/ckpt/` | 153636583 | `dc6c99d5b0298d32bf26f77f14f228969a7efb53822fcdd5feecac32b096eb45` |
| — the same, merged into one full state_dict for Paddock's Drive-JEPA node (what the regression gate loaded) | `{weights}/drivejepa_kitscenes/encoder-generator-scorer/step_000001958/merged_state_dict.ckpt` | `{nas}/siddesh/checkpoints/drivejepa_kit_enc_gen_scorer_p045/ (queue copy: {paddock_checkpoints}/da3038408e58c37c1268f5d018a56609b137a1a14dbd7c991d2ca560a0e77334/)` | 1245120771 | `91e13bc38aa76ee386b903a27f582b5459b1ae0d362a0cb090c01db118ebdb98` |

‡ Not re-hashed for this record: the value is the one the experiment code checks before loading the file (`02_encoder_build/src/common.py`, `E50_SHA256`).

### Runs tried and dropped

| Item | Where it sits at the training site | Bytes | sha256 |
|---|---|---:|---|
| Generator only, fast learning rate — 1 pass (stopped by the variety guard) | `{runs}/drivejepa_kitscenes/generator_joint_kit/s1/ckpt/pass_01.pt` | 28635381 | `498b4aeb2d281dc4006bead21f389bea605efcaa6e53391bc198d7a0d1dd1881` |
| Generator only, stay term 0.3 — 9.5 passes (its best check was 8.5) | `{runs}/drivejepa_kitscenes/generator_joint_kit/s2/w0.3/ckpt/p09.5.pt` | 9536586 | `77b8d922f5fa097d13905260ac2dc25eed6c0cf8d1ce9d5760effbc9efd226c7` |
| Generator only, stay term 1.0 — 9.5 passes (its best check was 8.0) | `{runs}/drivejepa_kitscenes/generator_joint_kit/s2/w1/ckpt/p09.5.pt` | 9536586 | `8e7227b4d30b641b2bf1eb24e81d2a0fce428ca5d948536ccc496f4a96ac7a7b` |
| Generator only, trained longer (the control beside the joint run) — 0.5 pass, its best check. The folder holds 0.5 to 3.5 passes | `{runs}/drivejepa_kitscenes/generator_joint_kit/s3/control/ckpt/p00.5.pt` | 9536650 | `fc6c84d54e8d1c13e07a63aba4ea1af8a10afe6f6c3eb7e412d53fee2a2df4a1` |

Every other check of every run sits beside these, one file per half pass (75 files for the generator runs, 15 for the three-part run, 2.3 GB).

## Held-out results and caches

| Item | Where it sits at the training site | Bytes | sha256 |
|---|---|---:|---|
| Per-moment rows and menus of the four models on the 223 held-out moments | `{runs}/drivejepa_kitscenes/generator_joint_kit/s4/heldout223.pkl` | 4061002 | `cf427722c60df7a0ea9dbaa0c1455a8fd80cc50613f5d6a3ad15e14a706ad80e` |
| The same for Encoder, generator and scorer trained together | `{runs}/drivejepa_kitscenes/generator_joint_kit/s5/heldout223.pkl` | 1063829 | `1993d92860ff9bdb4f5096c0aada256dad1239d522b0ffd4b938f2560f3ae0e0` |
| Frame cache: two front frames per moment, cropped and resized (KITScenes images) | `{runs}/drivejepa_kitscenes/encoder_lora_kit/s2/frame_cache/` | 8477299130 (11 shards) | — |
| Encoder features of the picking moments | `{runs}/drivejepa_kitscenes/encoder_lora_kit/s2/selection_features.pt` | 2025337147 | — |
| Training moments and per-scene calibration for the generator runs | `{runs}/drivejepa_kitscenes/generator_joint_kit/s1/moments.pt`, `scene_meta.pt` | 14049351, 464377 | — |
| Per-check menus on the 1,763 picking moments, every run | `{runs}/drivejepa_kitscenes/generator_joint_kit/s*/…/dumps/` | 115503776 (three-part run alone) | — |

## Files left out of the folders here

Logs, per-moment pickles, and tables over 100 KB, under the folder name they belong to. The location table `02_encoder_build/out/split_locations.json` (per-scene 1 km grid cells of KITScenes poses) is left out because it is derived from where the scenes were recorded.

| File | Bytes | sha256 |
|---|---:|---|
| `02_encoder_build/out/dump_f1.pkl` | 4426 | `f51cf48627bf5790090aa478406e8fa4322b84ab7f250203d2ebbcba8a9f799b` |
| `02_encoder_build/out/dump_f2.pkl` | 4426 | `f51cf48627bf5790090aa478406e8fa4322b84ab7f250203d2ebbcba8a9f799b` |
| `02_encoder_build/out/f1_rows.pkl` | 748 | `aff794fa51909bca3742dd63f0497fa5c7d320966654e008e9e22b1405d65081` |
| `02_encoder_build/out/split_locations.json` | 89120 | `461f42932d0902e6c333205b2f125337f783da427e036a761a007b9c6a8be4fd` |
| `02_encoder_build/run/f_baseline.log` | 4086 | `70f3eb878fc7143a2e508c361b967e121015a10824cd30ef031fab6a1a674d22` |
| `02_encoder_build/run/g_timing.log` | 2413 | `a2629414c057432decba68f63d8150634aea00f13486498af7ca74c0f14e0fa6` |
| `02_encoder_build/run/gpu_checks.log` | 4219 | `ee4cec1d224c6c38b5d8aa485dc72c3acdb39a3c5a51c96c173087cdc6052ee5` |
| `03_encoder_training/logs/a_build.log` | 2715 | `7bf73f3fcb678ea26a3664b2f55ea91437d4ed5741468fa22515e3bb6a38581b` |
| `03_encoder_training/logs/a_gate.log` | 2917 | `bbb68dec8d9f7cde0fcc22e447cff237ddd3782d1da094feb1c5564bfe4d1622` |
| `03_encoder_training/logs/c1_moments.log` | 8879 | `ca8bf2a5e4c61fc8c8d22fe226263afc5d6bdc7f8c8e1958f40977d6d7d7b2db` |
| `03_encoder_training/logs/c2_fixed.log` | 2880 | `890e81dda1dc56da7df0dee7d29ce50749f7f63c9c40ed085cd749d09c621d6e` |
| `03_encoder_training/logs/chain.log` | 20408 | `3e4a3e7a206d10845b1688b68b49f9c1121940a13ca7d47077566c35b0281c82` |
| `03_encoder_training/logs/chain2.log` | 24860 | `1fde04d2d579fd4009fd757ffae4e22a483203a41a0ce345f572075e390ece5c` |
| `03_encoder_training/logs/d_warmup.log` | 5768 | `95e7d5c596ae503a6fe6da71c5265b7972ee23f5e117b1f4a0eefe20887622ff` |
| `03_encoder_training/logs/e_main.log` | 16493 | `0631309686e720b4da50441ad6a2ddf306c3690c68273e3f26750cb875548aee` |
| `03_encoder_training/logs/f_final.log` | 17083 | `75b204b9b60e83d11f67d0860a36e1024649d7def90c1daf61896d5925720e6d` |
| `03_encoder_training/logs/main_steps.jsonl` | 915640 | `0723e5c6bfa400f1fdab84425948417c2e59041955bcbd685768002371b0b20a` |
| `03_encoder_training/logs/plots.log` | 115 | `75c0aee8e18b6b41393d15a9546c1dcb4594469a4e1ea42fed53a6c937080332` |
| `03_encoder_training/logs/warmup_steps.jsonl` | 224974 | `69c9ae84695610e12bf2e749675e24389e34600e1e0e2c67d14be2cfe9b78ed8` |
| `03_encoder_training/out/c_selection_zero_dump.pkl` | 436888 | `10d1ec2793cf4a94d65a8e8e300e5f382c42567c003c3256fe17fe4d17e4db29` |
| `03_encoder_training/out/c_selection_zero_rows.pkl` | 47955 | `b0b7391caab56ea9821390a03632a06f393ffae9debbe4f0d0d2e4f4c83e1f37` |
| `03_encoder_training/out/e_dump_pass_01.pkl` | 436888 | `ecb729146f8bcd9d2c5d9865bf8a17880e6df9fe9e4db885b831937ded1bb00a` |
| `03_encoder_training/out/e_dump_pass_02.pkl` | 436888 | `8333bcf5b7fb7df375a51ab575b6e6c2a60f28d4ea424d64f164146c524a66e4` |
| `03_encoder_training/out/e_dump_pass_03.pkl` | 436888 | `25decdb047e7f8151488bb066af59fa887599b64062c5334c35e752d43b155d8` |
| `03_encoder_training/out/e_dump_pass_04.pkl` | 436888 | `79a2de42bdba5ff2bfd21371d13c90b5a3e2e24ca4307d1bf3243df610969818` |
| `03_encoder_training/out/e_dump_pass_05.pkl` | 436888 | `02724a4297c8865eea12f9684189a3ecd71e185e4f4958bc719d998c367bdb56` |
| `03_encoder_training/out/e_dump_pass_06.pkl` | 436888 | `d8da4acd3ffc981b626d0442f4892b6acb1792c14efe8954a75de2d46ccff1e3` |
| `03_encoder_training/out/f_final_dump.pkl` | 946782 | `270cc850075419be13dd288669b7242fad52553883bef742ba50530684431d12` |
| `03_encoder_training/out/f_final_rows.pkl` | 110633 | `9d3ab40ba7384b65fcd0688a34a6329a42f79bb267d540f752ccf3eeb1c4fd2e` |
| `04_encoder_checks/logs/check2.log` | 8514 | `8fcf2fd2ced030f7c4c9fd4132567d5a117c0880a8dbc4a602a33a60d61c9738` |
| `04_encoder_checks/out/random_seed1_dump.pkl` | 436888 | `b802d68adf06378e7d29cc7807984f314fe236eb89510b8ab31830298dc8acb8` |
| `04_encoder_checks/out/random_seed2_dump.pkl` | 436888 | `59c90d94fffa43f63a792b3771e9518ecaa9b09fca19c892566f4d631a730b91` |
| `04_encoder_checks/out/random_seed3_dump.pkl` | 436888 | `35b04b7fea53ffdbb1eb3e743c28d521fbc8bf3f5af95a34bbb559d4b9f54e32` |
| `04_encoder_checks/out/random_seed4_dump.pkl` | 436888 | `60930648e78934886f3e48867693f9290987f95fc87ea1994edd2f96ed5358db` |
| `04_encoder_checks/out/random_seed5_dump.pkl` | 436888 | `464b5fdd5a472e7a6090a16ddeff8b45727409956933927d6ae5e67513fd643e` |
| `04_encoder_checks/out/zero223_dump.pkl` | 946782 | `3efe758d531ffa9c1b7bb761fb99587af72a3d3877fefdd1e54458ac652e6561` |
| `04_encoder_checks/out/zero223_rows.pkl` | 103239 | `00b3e619a782be01b05000d8b842823f93a13878700aa73f25303e6fa59eab9b` |
| `05_generator_survey/logs/d_horizon.log` | 1565 | `e6d5af272dfd591ab501eee96c847492e1827e5b5eb5156a43154924f3032d24` |
| `05_generator_survey/logs/memory.log` | 7914 | `2cf0dfc0f9fec491d7873ce2b52323eeaace1a3f8ab81d2d3766588e87d2eb14` |
| `05_generator_survey/logs/menu.log` | 13829 | `eecf423ae98d082e9da70dc6d3a72f5f6230afbe1972f451645a8b111bb72db8` |
| `05_generator_survey/logs/parts.log` | 10085 | `4eef3d9fc5aa85c5002cdf4f319b920b1c381dc5c6d5f10e5f559f75934df1ac` |
| `05_generator_survey/out/g_menu_per_moment.pkl` | 17044 | `55047b9cbf7bbf49d07f2880bb60ca9da24454f03584ccd041efd74093077abc` |
| `06_generator_only_fast_rate/logs/data_build.log` | 2890 | `9bf1d027be39b3e3d24444a5e8a1fba2961916051a7473f56825d32adfc187bf` |
| `06_generator_only_fast_rate/logs/phase1.log` | 8727 | `fccb6d774b578b73da9bdfdbd82cbb2604a0233e243f0ba5ce274d8f16268fce` |
| `06_generator_only_fast_rate/logs/phase1_attempt1_g2_counter_bug.log` | 8036 | `a38ebc789bbb09ac8a18e19e898ecee741e5e01fceb101435ba328d74c0dfb44` |
| `07_generator_only_gentle_rate/logs/rerun.log` | 17944 | `bb503782e6911464f616ff7ee84d065a75e6470f1c7d766c21a2c3d3e056d31d` |
| `08_generator_and_encoder_together/logs/phase2.log` | 30024 | `8855187b9653ead0e46db2601aaa612edfb67679781da3f5516bb598e07d04b0` |
| `08_generator_and_encoder_together/logs/phase2_attempt1_swap_stall.log` | 10192 | `daf9b52e1d43c5df6424dceb5a46f300b65a1141f75940a4e9f41add8003a8cf` |
| `08_generator_and_encoder_together/logs/report.log` | 8631 | `38d813dcf34c1888a067e0391c96b22f864f819e1778d66d76114191c087234c` |
| `09_heldout_comparison/logs/final.log` | 15410 | `afdd7109df84a166d2bc2ab775bce22a33ede993ccbbe2fac55788492efa4693` |
| `09_heldout_comparison/logs/g1_regate.log` | 469 | `d3b10249827a1b1d925ed6d67981ab241e05271b7febb74911e332b6d94ef22e` |
| `09_heldout_comparison/logs/report.log` | 14331 | `50e0ecceed3a6c160bd29921cf4d57daf8eb4f369d27e3df7ccf9dac29edbe4a` |
| `09_heldout_comparison/logs/smoke.log` | 9040 | `38de54c596bc65eec01d4c926d966ec8e8b531869edf7d49eec3d39c7f8038a7` |
| `10_encoder_generator_scorer/logs/allthree.log` | 38517 | `1108366d84870b3200add4bfb0751cc4efa80a9900020635aabdcf32f91d8818` |
| `10_encoder_generator_scorer/logs/final5.log` | 12543 | `4f0f9a82dda3ac8188ce5e98e0fffd8036a262ed82ef174ce3b6a897c87ce1bf` |
| `10_encoder_generator_scorer/logs/report.log` | 18732 | `69d911366028bdcf4e6588322014134fe4ea24a02f6f7315d2b8a16e44101aee` |
| `10_encoder_generator_scorer/logs/smoke.log` | 9311 | `52d4a71c3c853ed0f5b1390b91a377486a1075ec0f7dad12bc66b7eeb34900a4` |
| `10_encoder_generator_scorer/out/run.json` | 279647 | `e6c1c3e0d59a85253753985de69ede76ad855bb7d33ce08cbe48598f483b582a` |
| `11_regression_gate/out/prep_compare.json` | 2191 | `8f123ae6969ca018dd3c19183b485724296595243a5da08493ba846f12696686` |
| `11_regression_gate/out/prep_compare.log` | 7646 | `b6f9fc7c2119c4a7dc69e52fc8b5948141ab9a882fe0d670e829c9dafeb48a77` |
| `11_regression_gate/out/prep_export.json` | 1440 | `44597307c30f61885aadbe2f4b7ba0b7c37259058fccc7849190d5837d87b61a` |
| `11_regression_gate/out/prep_export.log` | 5920 | `41a2e707f528de46d789497512efe4b1a635b1fcfd2e99f508d343d60d27d5cf` |
| `11_regression_gate/out/prep_load.json` | 185 | `d2661efa625b20ba8cd400b3963cf02dbea3ffc1033d77b8e4306d7fa6d438ad` |
| `11_regression_gate/out/prep_load.log` | 4663 | `d3bd6c73b51fbdac950f4f3255c105945d26de619bc81c1128a8d9287e4f472f` |

## Drive-JEPA files the code imports

Every experiment imports Drive-JEPA from the vendored tree that Paddock's KITScenes runner uses (`{drivejepa_repo}`, upstream commit `e21f474`). Nothing in that tree was edited by this work. The list below was traced on 2026-10-30 by importing the experiments' modules and building the model and the hide-and-guess encoder on the CPU, recording every module file and every file opened under the tree. It covers what those imports and constructors load; a code path that only runs during training could load more.

`git` says what the tree's own git reports for each file: **clean** = as committed upstream; **modified** = a local change in that checkout; **untracked** = a file not in its git at all. The modified and untracked ones are not ours and exist only in that checkout.

| File | git | Bytes | sha256 |
|---|---|---:|---|
| `navsim_v2/data/8192.npy` | clean | 3932288 | `cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/bevformer/msda_compat.py` | untracked | 1019 | `7b85b17414c23ba4cd8942ccce636acf7e54054efd2b638f17cace2568e1a6c0` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/bevformer/spatial_cross_attention.py` | modified | 18831 | `96c53f83476f4ffaa48b39def951d09980174bc480f06e4182550fcfcd96bfa8` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/bevformer/temporal_self_attention.py` | modified | 11849 | `624b05fd028046db5b65f701fd3c8111daa7b4964123ba117809d7cf550dae8b` |
| `navsim_v2/scripts/inference/kitscenes_adapter.py` | untracked | 15591 | `f516db5b28888ce3a99147060eb8542bd8975d3a1752542335b2fb36c721b42b` |
| `navsim_v2/scripts/inference/run_kitscenes_inference.py` | untracked | 13251 | `466e9747e9bdd7ab40d8f18dd2b8fa2d60e302e5bf32d4ee452177ae5cf72705` |
| `navsim_v2/navsim/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/agents/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/agents/abstract_agent.py` | clean | 4220 | `a6447a16654449fe34bdb662bae92dd6a63d2ddd125b8db0828d1b77d615ae42` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/bevformer/bev_feature_build.py` | clean | 3173 | `c450bda28b0f4d5a050c97814917000f4b090d27a68c1700450fb544c5d93436` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/bevformer/bev_refiner.py` | clean | 5235 | `eff9fd20790810f7f92c9f93cb957542da5a2af3af7f7020d4555b5859e5bcd8` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/bevformer/custom_base_transformer_layer.py` | clean | 11111 | `c2e0c1d7ecac01efa795fa843da6f9cbb8871d385f5dabbf9eb74e33079e49b7` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/bevformer/decoder.py` | clean | 13972 | `5420e8968736e1df5148b2f4e6eddde8a9c12dc3ed04fb9361afe0ac06400c43` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/bevformer/encoder.py` | clean | 19308 | `07ae0ff7eacac7ff374fffbc3a8e03fcc84a3462e59570984aac14112e02cd9c` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/bevformer/grid_mask.py` | clean | 4073 | `d29eb57fbbccf75ef8eac527ca2566a28671e9e55b33e17d216bae50ab1688c7` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/bevformer/multi_scale_deformable_attn_function.py` | clean | 5903 | `03ad15be08b83fa2cb33defff6c22653b7eb1bab6fe33210cf677fabcc11781b` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/bevformer/simple_image_encoder.py` | clean | 2637 | `6fa90fdf6bc1b6ad7ae12cddb76d66f1fbc5f9cf730cfc95763dfddd7eb82a83` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/bevformer/transform3d.py` | clean | 14795 | `4f739d8d0d490efce9ea1e142bbe4f6c461ed3fca0aff256be6b1a8c123b6a0f` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/bevformer/transformer_decoder.py` | clean | 2484 | `2ba11bb5936fdcbe9d133929b2e62cb0cac6c0d9abb55cf6e5d9ab30c78e2591` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_agent.py` | clean | 14913 | `cf925ecee1a7545aef606c0e2d4a6eb7774e7d0ed2ae803eba5a1231c3417b39` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_config.py` | clean | 4390 | `61892398ad0fa4e300cc46ca8cd1f9447d2c67354023fea5b878d6cdf8608d23` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_features.py` | clean | 16144 | `d7dce17a098903f91ade9af2e80c85837e2772d8c0943367844457f9d86615c5` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py` | clean | 6203 | `dbcda6eef826c247e240702b34e0a23153a8c19547fa77ef5ff2a400ec955804` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/compute_navsim_score.py` | clean | 6697 | `df2d84fb9021276561ddb10d02c0218d594dad42de3afd2c49c475616e472f36` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/map_head.py` | clean | 2239 | `bf4eba2b3c58c4feef16376e6d07611b278f56e9a01460359b0a83c6c9c79eda` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/scorer.py` | clean | 2219 | `d97eaae3d08765f776984d8456c0fb50f6062451d5fe5c99571252b91543b273` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/train_pdm_scorer.py` | clean | 34076 | `f78280921784dd09d2937dcaf7242e66bd50a21ca1ca18524a3675e6f47330d8` |
| `navsim_v2/navsim/agents/drive_jepa_perception_based/traj_refiner.py` | clean | 962 | `c0d8fbe3fecb1e6123f15aee1a286efb1961342017a515bc63d41487defe8ee2` |
| `navsim_v2/navsim/agents/transfuser/transfuser_config.py` | clean | 3534 | `14866ffa5ff2d8ad3cd2cafe359038220adf977f83b27b7da42f248671d40516` |
| `navsim_v2/navsim/agents/transfuser/transfuser_features.py` | clean | 17073 | `1cf800e61d1e4cc78748f9672fd20e03c380b19e83e06b1b486dc8e5b44167d9` |
| `navsim_v2/navsim/agents/transfuser/transfuser_loss.py` | clean | 5554 | `b870de5844ce86a08ef322e46e3695025cd4ded30c062fc66ac4ac4974e60b33` |
| `navsim_v2/navsim/common/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/common/dataclasses.py` | clean | 29328 | `312ba866b271dbc71ffa18e80f084c66fcfe643d7c60c30603af2925c6d53d9f` |
| `navsim_v2/navsim/common/dataloader.py` | clean | 11718 | `89f3fde80ea3bb9d4e01bf0f2a02fe0dceb663d52028aba0eada41cefe6d8115` |
| `navsim_v2/navsim/common/enums.py` | clean | 3690 | `05ac678d1b2afeef3345b47b37e107dcab7dad9af98ef039e4ea00d7895d0974` |
| `navsim_v2/navsim/common/utils.py` | clean | 1015 | `a2ae70a67411bf27f4c193bf54ce23d176fb8f8767c15d1c31a6e86c2b565733` |
| `navsim_v2/navsim/evaluate/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/evaluate/pdm_score.py` | clean | 9875 | `cdf42bca71c743e2a11bb41d98eb3f0013d8b85542f592edd28fbf66aeabcabd` |
| `navsim_v2/navsim/planning/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/planning/metric_caching/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/planning/metric_caching/metric_cache.py` | clean | 2103 | `0ceafc65c0bedbce877b6ce77fa68be54363db0e17c0a1797939370912399922` |
| `navsim_v2/navsim/planning/scenario_builder/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/planning/scenario_builder/navsim_scenario_utils.py` | clean | 8157 | `f69892934edc9fc25558a45813bfa89ff0d389280939f3038021463103fd55df` |
| `navsim_v2/navsim/planning/simulation/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/planning/simulation/planner/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/observation/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/observation/pdm_object_manager.py` | clean | 8397 | `c8ef532af2082451694b67adf30eed31ace87f8fab5fdefe2740cc13ff47b780` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/observation/pdm_observation.py` | clean | 18576 | `7d6f42f9148bfd1f0067f86de7a6d493f3f3f83f0be39f5137c000437e2c44e5` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/observation/pdm_occupancy_map.py` | clean | 10417 | `83aeb24090abcb26e563153cac6a50e7e41894d8353091aa4d3dd04bc88cd702` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/scoring/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py` | clean | 18616 | `944a32f5051ecf84a39f0fe487109752ce234366da0f3d6e1935251cd777d694` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py` | clean | 35492 | `c3d4ab71716caac036e3ca440660da871f0e09439cde327febe4a10318756bd6` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer_utils.py` | clean | 2323 | `110b51aab2005318c0b262ce06a80926d5bdf9588f526ed47294ef8f5d6f30c6` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/simulation/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/simulation/batch_kinematic_bicycle.py` | clean | 7387 | `253beb3c1e7005a8c18a2aa33f43cbf6181829bc6aeda57f74d7bbc91f081ee0` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/simulation/batch_lqr.py` | clean | 21576 | `03c4d1b781edfba61dd2eee4f9308cc197e8db264de3df3936b6ff11e30238e7` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/simulation/batch_lqr_utils.py` | clean | 11654 | `a140e4132c6ec6017041d5185ae6f2aad3dbbe2274fbaaea4fadd05168fa3718` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py` | clean | 3376 | `ae07b533ada10bc37e973e5685a46a7cafc51de8d922ed02add558452ff04807` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/simulation_v2/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/simulation_v2/batch_kinematic_bicycle.py` | clean | 7387 | `253beb3c1e7005a8c18a2aa33f43cbf6181829bc6aeda57f74d7bbc91f081ee0` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/simulation_v2/batch_lqr.py` | clean | 21516 | `cdc7375c018704e45b83419fb4a9b437cb26ff5978429b5e0eccaef05599682b` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/simulation_v2/batch_lqr_utils.py` | clean | 12112 | `0b696b9886d7edb4c2c849fbce8978428831fb4ac07b18db7a71e45695c370dd` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/simulation_v2/pdm_simulator.py` | clean | 3382 | `300b223b02f676d6fe49c3b9640d22074b6953d18845e65ece8b91389a67db07` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/utils/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/utils/pdm_array_representation.py` | clean | 11469 | `7b43958e85c259a2e2d9a1f62bcfb61745e3bb6ed4fff8b29316493c3c8516a4` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/utils/pdm_enums.py` | clean | 3714 | `019c93c0a2337c8bc3a8c36a35a1c48b0c63a78975cc2c3472443f01d8b1a038` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/utils/pdm_geometry_utils.py` | clean | 7475 | `7a7ee642b4ab4bc9ce7b535f761c37da7c0744077de407c96a58545ae55d3e20` |
| `navsim_v2/navsim/planning/simulation/planner/pdm_planner/utils/pdm_path.py` | clean | 4041 | `febeb11b0661b1a2eb2d8ddbc3346fdd8c2b57fe0e03e4dc0c8811c78e24c71c` |
| `navsim_v2/navsim/planning/training/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/planning/training/abstract_feature_target_builder.py` | clean | 1423 | `9eefbb84e66d050e68737cd58ead1b3f341905fed96b18245bec59594a947cad` |
| `navsim_v2/navsim/traffic_agents_policies/__init__.py` | clean | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `navsim_v2/navsim/traffic_agents_policies/abstract_traffic_agents_policy.py` | clean | 8092 | `feaf44f99ba409da2725717495aa2a876d6f79cd339abbf0ed8d615ebbe77076` |
| `navsim_v2/navsim/traffic_agents_policies/log_replay_traffic_agents.py` | clean | 2851 | `bb797f602b1fbcf447a245165c8c0115eef6f3c2ade4f9617fdff35836630a08` |
| `navsim_v2/vjepa2/configs/eval/vitl/in1k.yaml` | clean | 3762 | `4b687c0fa3419cd433b78cf3e5accb3af0c2de114e700f78cea7792a1e549234` |
| `navsim_v2/vjepa2/evals/image_classification_frozen/modelcustom/vit_encoder.py` | clean | 3195 | `326db83c0293230fe0b8cad270ebee14091f4a50d8de515bed39db342cdb49a4` |
| `navsim_v2/vjepa2/src/masks/multiseq_multiblock3d.py` | clean | 9262 | `17a27c016b0e92143806c61495d66c1bc705c5bc9f946f40c1781700969b9c95` |
| `navsim_v2/vjepa2/src/masks/utils.py` | clean | 660 | `833f111a0fa5ffdbd3a6412e2dace2517c3c178f49c14f8bb631d9f6a070dfd0` |
| `navsim_v2/vjepa2/src/models/predictor.py` | clean | 9230 | `a9a77d1d06e6c858329be169e881c07bcc238594ff63cd56dafdc13bfbb5c7fb` |
| `navsim_v2/vjepa2/src/models/utils/modules.py` | clean | 22472 | `bb82b27f5d39cbe2062fefc96734203454bc38338dc6ab66c91a0df3b909cadf` |
| `navsim_v2/vjepa2/src/models/utils/patch_embed.py` | clean | 1343 | `1823c40b34410e70eb313360e21678e18a0ee5465b5d47de9f3d967c074f5a9d` |
| `navsim_v2/vjepa2/src/models/utils/pos_embs.py` | clean | 3553 | `e6d3b3863ade47b3734a3b443f1bcc93eab02e39bf596ea199517f68ccab443e` |
| `navsim_v2/vjepa2/src/models/vision_transformer.py` | clean | 14623 | `f740a0d1f515688224ba65ad53108565b3499b1a8345f107ab8057d252d694b0` |
| `navsim_v2/vjepa2/src/utils/tensors.py` | clean | 1832 | `782b58bd2af456e184750e5318ab773105108383f61b280fe4c7a90f46add2c8` |
