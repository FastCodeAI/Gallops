# Pointers — what this experiment produced that is not in the repository

Nothing below is committed: checkpoints and arrays by rule (`.gitignore`,
`docs/checkpoints.md`), the training set because it is derived from NVIDIA
Physical AI AV, the run folders because they hold rendered frames, and the
large tables for size. Each row gives a location, a size and a sha256, so a
copy can be recognised wherever it ends up.

**Locations are templates, not paths.** `{weights}` and `{runs}` are
`[roots.weights]` and `[roots.runs]` from `gallops.toml`. At the site that ran
this experiment the artefacts have not been moved into the `{weights}` layout:
the "where it sits" column says where each one is, relative to that site's
store for this experiment, written `{runs}/drivejepa_retrain`. Folders under it
are written with this repository's folder names; on the store itself each
still carries its working name, which the "Where this came from" table in
`docs/drivejepa-retrain.md` gives. Lay artefacts out as the template column
says when copying.

The sidecar for the retrained checkpoint is `checkpoint.json`, beside the
weights; `docs/drivejepa-retrain.md` reproduces it.

## Weights

| Item | Template location | Where it sits at the training site | Bytes | sha256 |
|---|---|---|---:|---|
| Retrained checkpoint, adapter 0.1.7 (merged: released encoder + first-run scorer + second-run generator) | `{weights}/drivejepa_physicalai/second-run-generator/step_000000002/drive_jepa_physicalai_generator.ckpt` | `{runs}/drivejepa_retrain/12_second_training_run/ckpt/` | 1273002536 | `29d5f9517c2ea8789f35c51f0934493e0727bde2d40b573ecc3eb33f5922d916` |
| Extended trajectory vocabulary, 10,240 anchors | `{weights}/drivejepa_physicalai/second-run-generator/step_000000002/vocab_10240.npy` | `{runs}/drivejepa_retrain/12_second_training_run/ckpt/` | 4915328 | `4b0bc3c9a55c1787cd54d4a27c8bfd7119bbe6ae2cbe1bf1cfe3343da639e785` |
| Second run, the pass that was chosen, trainer format | `{weights}/drivejepa_physicalai/second-run-generator/step_000000002/pass_02.pt` | `{runs}/drivejepa_retrain/12_second_training_run/gen/final_P0/` | 75388279 | `38ecab1586a28cdf794a2eac08f09692eeee78187912c8185872f38eb7a5c220` |
| Intermediate checkpoint, not shipped: the first run's chosen pass, merged | `{weights}/drivejepa_physicalai/first-run/step_000000009/drive_jepa_s4_pass09.ckpt` | `{runs}/drivejepa_retrain/06_first_training_run/candidate/` | 1273002795 | `8a71c3c0d84a9758de46c7b46eaf18556a62ac4837feea7271040e7a0caacdd2` |
| First run, chosen pass, trainer format (start of the second run) | `{weights}/drivejepa_physicalai/first-run/step_000000009/pass_09.pt` | `{runs}/drivejepa_retrain/06_first_training_run/run/ckpt/` | 85177023 | `b3130f1f2155f5a9a5c06191831dfba0e0c07a9841689b39d386c12061328a5a` |
| Released Drive-JEPA checkpoint (start of the first run; encoder source) | `{weights}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt` | upstream release (`LinhanWang/Drive-JEPA`) | 3735352913 | `740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05` |

## Training set

Derived from NVIDIA Physical AI AV; not redistributable from here.

| Item | Template location | Bytes | sha256 |
|---|---|---:|---|
| Training manifest, 126,981 samples | `{runs}/drivejepa_retrain/04_training_set/manifest.parquet` | 1893313 | `ce834ee735fbd745e8b073f069803c7df1ebc64d86f48b6947a85f674ad7f1be` |
| Training labels | `{runs}/drivejepa_retrain/04_training_set/labels.npy` | 24660608 | `0638133e31fb992c433cdd4172e384407efa2fae33988370b291ebee3d92791b` |
| Label index | `{runs}/drivejepa_retrain/04_training_set/labels_index.parquet` | 2094235 | `0f7f1ec8834958ea92e73e0de4f157b694070854cdf71b6b789c43488992e95f` |
| Validation manifest (`val_full`), 24,657 examples | `{runs}/drivejepa_retrain/04_training_set/val_full/manifest_val_full.parquet` | 399189 | `06d4941900307fd5d7639e6b65dd824515a8d5792725afc39cbcb18c74a2bd0c` |
| Validation labels | `{runs}/drivejepa_retrain/04_training_set/val_full/labels.npy` | 4930496 | `6c5ce23cb1622aefdb6baf7c7abe0895021a40257d61ce49fb273be4622b420d` |
| Smoke validation manifest, 608 examples | `{runs}/drivejepa_retrain/04_training_set/val/manifest_val.parquet` | 18424 | `defe77d9e7b099abc4f682bece3697f3cfd50a7695287786844a8819d1a92ddc` |

## Tables left out for size

Per-clip and per-sample tables over 100 KB (over 1 MB for the 30-scene closed-loop test). The scripts that write them are in the folder named.

| File | Template location | Bytes | sha256 |
|---|---|---:|---|
| `14_after_retraining_diagnosis/part2_why_it_swerves/menu_proposals.csv` | `{runs}/drivejepa_retrain/14_after_retraining_diagnosis/part2_why_it_swerves/menu_proposals.csv` | 464811 | `75448c7ba12f61a31b9301efcfc54aea0670951a2b1d05c7aec84423ccee5be6` |
| `14_after_retraining_diagnosis/part3_swerve_fixes/fixA_alteration.csv` | `{runs}/drivejepa_retrain/14_after_retraining_diagnosis/part3_swerve_fixes/fixA_alteration.csv` | 994941 | `a1b914b5d548156b792c97afc08edbc4b05f0eb44505908fb46439ba6bc59c60` |
| `14_after_retraining_diagnosis/part3_swerve_fixes/fixB_picks.csv` | `{runs}/drivejepa_retrain/14_after_retraining_diagnosis/part3_swerve_fixes/fixB_picks.csv` | 389081 | `9429d4c08a7b3f710833bd72af7288b0d744f9bc9845b9addc4f0660a59c8aaa` |
| `14_after_retraining_diagnosis/part3_swerve_fixes/long_hold.csv` | `{runs}/drivejepa_retrain/14_after_retraining_diagnosis/part3_swerve_fixes/long_hold.csv` | 173703 | `6047bf3db3cd59435d6429ed04dbc920ed51515bb20601becdf27b22b2325856` |
| `14_after_retraining_diagnosis/part3_swerve_fixes/testbed.csv` | `{runs}/drivejepa_retrain/14_after_retraining_diagnosis/part3_swerve_fixes/testbed.csv` | 3095416 | `9bb1b85385a26a261017eaaa7b8e5e1295f60e95e27d255c615bf3ebf8cf1037` |
| `14_after_retraining_diagnosis/part4_why_it_stays_stopped/proposals.csv` | `{runs}/drivejepa_retrain/14_after_retraining_diagnosis/part4_why_it_stays_stopped/proposals.csv` | 1513634 | `742ccd78121662bbc1aa7630ec9e916c100d297a2c381ff5e93bcba34cf36f81` |
| `14_after_retraining_diagnosis/part4_why_it_stays_stopped/replans.csv` | `{runs}/drivejepa_retrain/14_after_retraining_diagnosis/part4_why_it_stays_stopped/replans.csv` | 112623 | `4b55a640d24d44514dec3a3fd1ba90fc677a6f7828cedce9fcfd041dcd0707bc` |
| `14_after_retraining_diagnosis/part5_stuck_fixes/antistall_first_firings.csv` | `{runs}/drivejepa_retrain/14_after_retraining_diagnosis/part5_stuck_fixes/antistall_first_firings.csv` | 139595 | `9fcdf217b34e716ce52db66f12467e5f81ad70cc5fffdf9db6982ed44b902c09` |
| `14_after_retraining_diagnosis/part5_stuck_fixes/scorer_geometry.csv` | `{runs}/drivejepa_retrain/14_after_retraining_diagnosis/part5_stuck_fixes/scorer_geometry.csv` | 692684 | `07ad388a0b2e3f1e09accb4e5c67c96dda549fa29a4cda7094b2011048484f48` |
| `01_find_clips/a1_tree_26.04.json` | `{runs}/drivejepa_retrain/01_find_clips/a1_tree_26.04.json` | 940096 | `46583741de5dd4c3839fb7c1bb8717130c2d05aeb11e08c0d662c77689d4fb75` |
| `01_find_clips/d2_anchors.csv` | `{runs}/drivejepa_retrain/01_find_clips/d2_anchors.csv` | 5845150 | `c90bea29d9e73b08c99f33ac99e1a358286d183fd3dceeee290cab00dc106152` |
| `01_find_clips/d2_anchors_n60.csv` | `{runs}/drivejepa_retrain/01_find_clips/d2_anchors_n60.csv` | 233465 | `a204dfcdcfbb326c91edad322d3fab5af51b6929bfce19c21a06e225ad10875e` |
| `01_find_clips/d2_scenes.csv` | `{runs}/drivejepa_retrain/01_find_clips/d2_scenes.csv` | 180636 | `92041309315c76f6905ea20cca52d7741024ca470824d6925265a7620fdfec78` |
| `01_find_clips/d3_anchors.csv` | `{runs}/drivejepa_retrain/01_find_clips/d3_anchors.csv` | 4484286 | `905f8320fc8128fc6e7ca892b8d80d7ec306db4b86aa6bc446710564b6130992` |
| `01_find_clips/d3_scenes.csv` | `{runs}/drivejepa_retrain/01_find_clips/d3_scenes.csv` | 162808 | `65339e9694bd860af893402ed3665d6fc1100651c8830ede41550e3a21217e37` |
| `01_find_clips/download_manifest.csv` | `{runs}/drivejepa_retrain/01_find_clips/download_manifest.csv` | 4980758 | `cc5684d81220f1f441a8a51b9236c863ba113f357657f6c3a419438edc1e585a` |
| `01_find_clips/r3_raw_clips.csv` | `{runs}/drivejepa_retrain/01_find_clips/r3_raw_clips.csv` | 10538642 | `c6b7e70c80311011b73b347eb2241b55de35ecfee44893af74b17cbcc19f2e32` |
| `01_find_clips/r3_raw_moveoffs.csv` | `{runs}/drivejepa_retrain/01_find_clips/r3_raw_moveoffs.csv` | 916668 | `28d7de2bd1c557f3cfddb043e0eeff884c620e7d9ce7da183319497f38aa5d5c` |
| `01_find_clips/r3_raw_moveoffs_ego.csv` | `{runs}/drivejepa_retrain/01_find_clips/r3_raw_moveoffs_ego.csv` | 2552762 | `bb5aa6c869424dbf0d0c2a625edd6bca3a10f5b848f1066a351de7fd50f76263` |
| `01_find_clips/r3_raw_tree_sizes.json` | `{runs}/drivejepa_retrain/01_find_clips/r3_raw_tree_sizes.json` | 855779 | `f69e10a5f10ae291edef2d38ffe30bf86d355b6e17e192fe408879328092c999` |
| `01_find_clips/raw_train.txt` | `{runs}/drivejepa_retrain/01_find_clips/raw_train.txt` | 5684251 | `e2917ab0e8e1a599c03c21a813fe09ae5bb47d23515cab817723a9d7b27267ce` |
| `01_find_clips/raw_val.txt` | `{runs}/drivejepa_retrain/01_find_clips/raw_val.txt` | 3364462 | `24bb4e7f610e0fe45eaa26f8b54e32234d7ad9797c9da45e98db5236ec2b9fa5` |
| `02_build_samples/samples.csv` | `{runs}/drivejepa_retrain/02_build_samples/samples.csv` | 276970 | `8a70d54215aa45fc1d2449225855f4580ddba2fdd0e9c2410b7ce8d32f896a8b` |
| `03_grade_candidates/download_manifest.csv` | `{runs}/drivejepa_retrain/03_grade_candidates/download_manifest.csv` | 198897 | `08c24da1f669c4365f20a02c255417f0405396eb00bf732e4eccc76922472d3b` |
| `03_grade_candidates/pilot_grades.csv` | `{runs}/drivejepa_retrain/03_grade_candidates/pilot_grades.csv` | 2426657 | `33f19f24046b27c6dd42e1c8931a662b48df25041832aba837bc9c07087e4fa9` |
| `03_grade_candidates/pilot_grades_s2.csv` | `{runs}/drivejepa_retrain/03_grade_candidates/pilot_grades_s2.csv` | 2418699 | `2ca4f50fa4440f2cf4923acf8342b6d28a25a4f9dffd62290a13c7554bfe20f0` |
| `03_grade_candidates/s21_vocab_anchors.csv` | `{runs}/drivejepa_retrain/03_grade_candidates/s21_vocab_anchors.csv` | 678698 | `8c5a5b9849e71ac9d21a762af31ee75f610428d27b38eb3e3d5e81c4fbab9df3` |
| `03_grade_candidates/samples.csv` | `{runs}/drivejepa_retrain/03_grade_candidates/samples.csv` | 137370 | `69217f52a7c9ca107c4699fb42cc575f0b8e0c3e41fe693c691be81998fed5db` |
| `04_training_set/download_manifest.csv` | `{runs}/drivejepa_retrain/04_training_set/download_manifest.csv` | 7420784 | `4b84f3340ca3ea1662f95587d557fa52943e0bff618ba947eb9f5d9c84040c73` |
| `04_training_set/vocab_added.csv` | `{runs}/drivejepa_retrain/04_training_set/vocab_added.csv` | 199491 | `88cce8307d82075a0a7437a09d5cbf365725b80501db8a4856587bb200ce96ce` |
| `09_three_hypotheses/h2_exam_inspection_scores32.csv` | `{runs}/drivejepa_retrain/09_three_hypotheses/h2_exam_inspection_scores32.csv` | 120993 | `3438e2f3e69b76366f309b984521849b578837a65e9e88c34f06f61b087fe062` |
| `10_three_quick_looks/l1_sample_plans.csv` | `{runs}/drivejepa_retrain/10_three_quick_looks/l1_sample_plans.csv` | 716490 | `4f303f81b59c6df6559927d26812fad41730771d5d6c1c8fea00628b27d41224` |
| `11_scorer_head_and_pullout_grades/c4_exam_inspection_plans32.csv` | `{runs}/drivejepa_retrain/11_scorer_head_and_pullout_grades/c4_exam_inspection_plans32.csv` | 208192 | `4b82ede589c867b1bd17e74a3bd819a917388c2ea6fd3084f437258203d51b59` |

## Run folders

Whole directories, so no single hash. Each run's own log carries the adapter
version and the checkpoint sha256 it verified before driving.

| Item | Template location | Contents |
|---|---|---|
| 30-scene closed-loop test, adapter 0.1.7 | `{runs}/drivejepa_retrain/07_closed_loop_test_017/runs/drivejepa_017_<scene>/` | 30 folders, 4.9 GB; rollouts with rendered frames |
| 30-scene closed-loop test, intermediate checkpoint (adapter 0.1.6) | `{runs}/alpasim_bulk/runs/drivejepa/drivejepa_016_<scene>/` | 30 folders, 4.9 GB |
| 30-scene reference, adapter 0.1.4b | `{runs}/alpasim_bulk/baseline_014b/run_<scene>/` | 30 folders |
| Second run: all passes and pilot arms | `{runs}/drivejepa_retrain/12_second_training_run/gen/` | 8 run folders, one or two `pass_NN.pt` of 75 MB each |
| First run: all passes | `{runs}/drivejepa_retrain/06_first_training_run/run/ckpt/` | `pass_00.pt` to `pass_10.pt`, 840 MB |
| Second run: an evaluation of the finished checkpoint on unseen clips, made after it was chosen | `{runs}/drivejepa_retrain/12_second_training_run/phase2/` | `phase2_tables.md` (33 KB) and 9 scripts. It reads the shipped checkpoint and played no part in producing or choosing it. Two of its tables, the second run's own targets and the scorer's picks, are committed as `12_second_training_run/unseen_clip_evaluation.md`; the rest is **not in this repository**. `phase2_tables.md` sha256 `6780de352273c9759576165a1845a661f88777eebab730d1b648e39c918aa266` |
