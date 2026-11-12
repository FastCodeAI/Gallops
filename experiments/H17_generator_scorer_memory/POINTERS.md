# Pointers: what this experiment produced that is not in the repository

Nothing below is committed: checkpoints and arrays by rule (`.gitignore`, `docs/checkpoints.md`), the token
cache because it is derived from NVIDIA Physical AI AV, logs by convention, and tables over 100 KB for size.
Each row gives a location, a size in bytes and a sha256, so a copy can be recognised wherever it ends up.

**Two kinds of location.** A *template location* starts with `{weights}`, which is `[roots.weights]` in
`gallops.toml`: where a copy belongs in this repository's layout. Nothing has been moved there yet. *Where
it sits* is a label, not a variable `gallops.config` expands: `{experiment_store}` is the training site's
network store for this experiment, `{retrain_store}` that of the earlier retrain; the README's
placeholder table lists every label. Neither store is under that site's `[roots.runs]`. Hashes were
computed on 12 Nov 2026 from the files on those stores.

## The starting model

Drive-JEPA 0.1.8b, pass 0.6: an earlier retrain on Physical AI AV whose own record is not in this
repository yet. It is identified here by its sha256 only.

| Item | Where it sits | Bytes | sha256 |
|---|---|---:|---|
| Starting model, merged checkpoint | `{retrain_store}/018b/release/drive_jepa_018b_p06.ckpt` | 1273003357 | `3d566eb211b5ce0f556680196a46fc0a74da29091ae62ac588910a48d62b8e66` |

## Checkpoints

Trainer format: the generator, the scorer and the memory blocks, without the frozen encoder. No
`checkpoint.json` sidecar was written, because none of these was a candidate for paddock.

### H17.r2, first training run, memory carried

Template location `{weights}/drivejepa_physicalai_memory/h17-r2-carried/`. Where it sits:
`{experiment_store}/runs/pilot_seed0/carry/`. The file names give the pass (`p01.2` is pass
1.2); the chosen check is `ckpt/p01.2.pt`, step 5,218, which the record pins. `resume.pt` is the trainer's
resume state.

| File | Bytes | sha256 |
|---|---:|---|
| `ckpt/p00.0.pt` | 35039459 | `c37436407ec16910957fac9913854f40bd94991af59fd03de8630c335abff25e` |
| `ckpt/p00.2.pt` | 35039587 | `709f0564f1b6b649ceac5237b26f93f212fde1b0da8e4e4abcd5f1dfca29e36a` |
| `ckpt/p00.4.pt` | 35039587 | `311c3bd69b0bcfc7557d29fa3665a84413a551bf2f7b89fa9158cd26796df4f3` |
| `ckpt/p00.7.pt` | 35039587 | `787d03b985722e841156ca91249e8d7de5185aab9186a48c0fcf98a5f5db9842` |
| `ckpt/p00.9.pt` | 35039587 | `bb504871221b98a09a4dba3a82e774ee666337234f42b8a6077821f1be31ccee` |
| `ckpt/p01.0.pt` | 35039587 | `5ce54b3a4bfee24cdb7c3d1566333dcae0180557850e5e5de39363556b58ce6e` |
| `ckpt/p01.2.pt` | 35039779 | `70b37896e5b1369a093195cfa4c46b62749200fdb29b6712e0f24d5ba604a5cd` |
| `ckpt/p01.4.pt` | 35039779 | `2dd063b98b723ca5d26dd6aba3d4f1ae64a35c6c7002393c723dcc22d6752bbb` |
| `ckpt/p01.6.pt` | 35039779 | `d029f5e5c260c42d156ae4e11d64bab989a0628208712333fbc0f98a1557dc40` |
| `ckpt/p01.9.pt` | 35039779 | `dcb8faf3a0df4595b68a2b673a7d26fab3ed796145505ed79c29cd0927a131f7` |
| `ckpt/p02.0.pt` | 35039779 | `f3f02ccaefa7cc28f339951e67d97b3a79d37474a628480166a99460ea336f44` |
| `resume.pt` | 105148041 | `f412a26e23b7ce22cce51a00942198f1cf7a62232e280a0c715b9f2d4e669649` |

### H17.r2, first training run, memory wiped at every moment

Template location `{weights}/drivejepa_physicalai_memory/h17-r2-wiped/`. Where it sits:
`{experiment_store}/runs/pilot_seed0/wiped/`. The chosen check is `ckpt/p01.2.pt`.

| File | Bytes | sha256 |
|---|---:|---|
| `ckpt/p00.0.pt` | 35039395 | `c3fced798f9870db52d66d12e3e0bf53353391c38b70a579ca8bc138f6a50f39` |
| `ckpt/p00.2.pt` | 35039587 | `7ba41a8c1dac67a780c11ca57ef49a35318b4ee1ae6d2ef8a7827f3920edc412` |
| `ckpt/p00.4.pt` | 35039587 | `986bcfb2d41792b5d1c9b0e4fb003c0fbc6368300d572543aa9e18bbae0ebb4b` |
| `ckpt/p00.7.pt` | 35039587 | `6eafc977197b60a4d8704e7adb5a02b523075811b7f9b032eff701825dff986a` |
| `ckpt/p00.9.pt` | 35039587 | `2d2a077be9f5d69a0f25ddb8d523e0e24dec0040e113bbb91470ca5c53b29f3e` |
| `ckpt/p01.0.pt` | 35039587 | `c9edfc8a626d94b03bebfa8d475fd51ae8480a2ebe5b9cdcc703056ef841592a` |
| `ckpt/p01.2.pt` | 35039587 | `369779a425b47a7282be87566a2b5deef10068050a98bd514f6c3d917bd7bbf5` |
| `ckpt/p01.4.pt` | 35039587 | `b9d1f2b2b4430d2395c7124107e828e7dbe61d4cacad4596f838b1719e4c2a15` |
| `ckpt/p01.6.pt` | 35039587 | `62b2cef366fa7f0c92ef35560c49d4c3b6d2f4810ffe319b85fc4b0684310252` |
| `ckpt/p01.9.pt` | 35039587 | `ab10ede37ca6e8c5a07082bba0e21f9de1584616a629279185838f85490b919a` |
| `ckpt/p02.0.pt` | 35039587 | `c2f6269516855b8eeedb1c9327654cd2d09f832e057049c31fc133f760244bed` |
| `resume.pt` | 98111036 | `d0b6937daf9bf0e962d97c47ed1e04795770806699effafe1f5b28670a1c19a0` |

### H17.r3, the memory-collapse fix, 930 steps

Template location `{weights}/drivejepa_physicalai_memory/h17-r3-collapse-fix/`. Where it sits:
`{experiment_store}/runs/build4_short/`. The record pins `ckpt/step0930.pt`.

| File | Bytes | sha256 |
|---|---:|---|
| `ckpt/step0310.pt` | 35061136 | `594556a7ba727e479a2e417d0e62a59c386d202abb815e6707bebadb51bdac01` |
| `ckpt/step0620.pt` | 35061136 | `aa556e22efb18293f65b0c9f7eb677c078c63bab116ec68f18e440b42867a7c8` |
| `ckpt/step0930.pt` | 35061136 | `645157874f555f922160cde67040e1f8f067954d0adf95ce2839dd6d3755d9fa` |
| `resume.pt` | 105198333 | `e8976704078ebe941222770a1186ebdbe01d9f1eaf0ea39b40c1a111ac36ecde` |

### H17.r4, the recall lesson, 930 steps

Template location `{weights}/drivejepa_physicalai_memory/h17-r4-recall-lesson/`. Where it sits:
`{experiment_store}/runs/build5_short/`. The record pins `ckpt/step0930.pt`, the checkpoint
the headline result was measured on.

| File | Bytes | sha256 |
|---|---:|---|
| `ckpt/step0310.pt` | 36748734 | `88dc7d6cb3617c1037086ff547af041ed93047888fcef9a97ecd1e6e84cfcf89` |
| `ckpt/step0620.pt` | 36748734 | `0a0291d2e4245867ccf848084e9763d3252a4d672fad2fb5b3fe7a43b04f6536` |
| `ckpt/step0930.pt` | 36748734 | `84359783ef502e1ccc3a2dcff013bf45d215f9fb11e046cc8110c64581434e21` |
| `resume.pt` | 110139953 | `b688396305885b14fad21b9cea99fe67620cb9a52129dbe82816566d7d3ff5e9` |

## The token cache and the slices

The frozen encoder's output at every moment of four slices, 88.6 GB in all, one file per clip. The per-clip
files are not hashed one by one; each slice is identified by its index file, which lists every clip file
and moment.

| Slice | Where it sits | Clip files | Index file | Bytes | sha256 of the index |
|---|---|---:|---|---:|---|
| training | `{experiment_store}/cache/training/` | 4015 | `index_training.parquet` | 2049402 | `ac5561cdb42ecafb6af91cca71d066e591aebc4dc51be3294011753ba7921613` |
| picking | `{experiment_store}/cache/picking/` | 84 | `index_picking.parquet` | 50400 | `97c5a1dfb6e3b946060bb76a1272625c636d6ae64fc271a120efe39ee0be044f` |
| graded heldout | `{experiment_store}/cache/graded_heldout/` | 355 | `index_graded_heldout.parquet` | 187178 | `52422a939a35e1c402bd1f1b85eb69469ed3341d623e59d0f7a68f183a7f3654` |
| raw heldout | `{experiment_store}/cache/raw_heldout/` | 300 | `index_raw_heldout.parquet` | 166245 | `2de890c7facf71eb6946fe8f01f0c889ed683518365d7a99610c633ad38361f6` |

| Slice manifest | Where it sits | Bytes | sha256 |
|---|---|---:|---|
| `manifest_graded_heldout.parquet` | `{experiment_store}/slices/manifest_graded_heldout.parquet` | 172964 | `006ec7c81396683f0017c63353b13aa7867f4a21fa41d53048dc52d08c9b36d3` |
| `manifest_raw_heldout.parquet` | `{experiment_store}/slices/manifest_raw_heldout.parquet` | 155896 | `2a916fc04eb45df8c704c775b4965d976b5c5a63c24419c56ce98fafbb197d9f` |
| `manifest_heldout.parquet` | `{experiment_store}/raw_heldout/manifest_heldout.parquet` | 155896 | `2a916fc04eb45df8c704c775b4965d976b5c5a63c24419c56ce98fafbb197d9f` |
| `manifest_heldout_tags.parquet` | `{experiment_store}/raw_heldout/manifest_heldout_tags.parquet` | 256907 | `f48769797ae723eca992d920aeaf731d0285bc80424743007195b651af403c61` |

## Tables and arrays left out

Tables over 100 KB, and every `.parquet` and `.pt` file whatever its size. Each sits in the working
folder's `out/` at the training site; the folder named is the step it belongs to here.

| File | Step folder | Bytes | sha256 |
|---|---|---:|---|
| `b2_train_clips.csv` | `01_survey` | 456382 | `53a08640b4549732e6e7d6f93a4c41b18dd6dd02422cf76897eb5ea76de11e14` |
| `b4b_fetch_status.csv` | `01_survey` | 157250 | `d4dfdcaa0c528c8b7847b9b2422df4c3023eda409c994fb79e7add6a39d040b3` |
| `b4b_fetch_verify.csv` | `01_survey` | 282131 | `1e1128e7ed396c0585d8ccf5914d20dc3254c754653ff903ce9fbb391fae1ece` |
| `b4b_ordinary_clips.csv` | `01_survey` | 259711 | `a96790fec6060298ca06baae7d0ba4413e425e896c827f60fddefbf8eb218f5e` |
| `memory_moments_graded_heldout.parquet` | `02_data_slices_and_token_cache` | 5946 | `cc3f8c29240c410ef0d77b31266a4f307a2debc38ac23e0dbc768a0769a73711` |
| `memory_moments_heldout.parquet` | `02_data_slices_and_token_cache` | 4503 | `52257b99bebc175b42265cc156f1c4c942df3c30d85a6c3eccea52db3d41f28c` |
| `memory_moments_picking.parquet` | `02_data_slices_and_token_cache` | 3569 | `6e150d9d7639d9ea1373be9d1a49a5d7875286b177ee4a699babdf8fbe9fe340` |
| `memory_moments_raw_heldout.parquet` | `02_data_slices_and_token_cache` | 6497 | `008c4d2f25f44022cb0fc6c5c840eb3d36e2c100f0eac12bf600da164f76919a` |
| `memory_moments_train.parquet` | `02_data_slices_and_token_cache` | 35842 | `d1e9a56f796b4dfe412bd02130c8ae68cc6e4bfeac3f9de5a0ba44f24683d0f7` |
| `s4_sample_moments.parquet` | `02_data_slices_and_token_cache` | 24981 | `f1ecf4b11f2b626f99e4afd30c0b35407f5250562d48623aab62731148cd47b3` |
| `slice_clips.json` | `02_data_slices_and_token_cache` | 175997 | `d50629feeeebfc547b863c040f87679d52fa81cb434f1072d1406eb56300191b` |
| `attempt_before_slot_emb/s6_t4.json` | `03_memory_model_and_trainer` | 102731 | `204c9368ee3c543d1519f9e15e375e04509dad6308bf79ea6b8df12fd2d25cba` |
| `s6_t4.json` | `03_memory_model_and_trainer` | 102022 | `7fe3549f4870ee818d552cea989c02da538af26ff8fb80c45a080bffd537006d` |
| `s6_t4_v2.json` | `03_memory_model_and_trainer` | 120992 | `51727ead952a5860b929efe45426b5ee17b6ddcdd11ef5cb624e3e88586dfed8` |
| `s6_t4_v3.json` | `03_memory_model_and_trainer` | 143795 | `0c9c6068a2cf3578cf5d4fd9faee0fcfd78fbd8c525c3aa28c3458af5fae5a08` |
| `recall_target_stats.pt` | `06_recall_lesson` | 67909 | `9c39e24375bedd6d16901c1bedbaa2576e06f52bcb5554ced41ef8699c41d357` |
| `link2_results.json` | `07_does_the_memory_hold_driving_information` | 117563 | `fd6d02e25ca719007199aba8efded25d718981011de4cc40be3d09604cbd3109` |

## Logs

81 log files, 977538 bytes together, in the working folder's `logs/` at the
training site. The reports cite them as `logs/<name>`. The logs of the 12-scene closed-loop check are
not listed: that check was still writing them when this was recorded.

| File | Bytes | sha256 |
|---|---:|---|
| `b1_b3.log` | 2914 | `eda6481a3d8baf9cf457306d5397658af6d0c727949b4f02de699c23ab0320f4` |
| `b2_graded_heldout.log` | 921 | `10c452a09ff552b550f246b776155475fb4a91ce1faab7ba66c7070397d96d10` |
| `b2_heldout.log` | 934 | `bc94df18f1f2e449c22915d4fef111ec80e9c581aa45a4f732d3a9e8f3ffb3b2` |
| `b2_raw_heldout.log` | 894 | `59e22eec6aa57674a884bd91d62aa7662f0080e53082003c342bf96d1fd99a53` |
| `b2_train.log` | 944 | `bc8af055430ac2d463e468c11b191d97289e1d2b7f525ceee5cc55a8e837a8cd` |
| `b2_train_full.log` | 2124 | `d5104c2949a5f003c09e4a1d2383c421d920e3d8e6cec5dc0f75b2b60f4ab5f5` |
| `b3_timerange.log` | 46112 | `4eab8b876011c9a6efec0a4e2af4bf12ecc138ede9ecb976e4248638c6e18384` |
| `b4_braw.log` | 1855 | `09d272a26cdd7283dbf53f009c47b1376660301c186a74e9b2c1bfba72006f2e` |
| `b4_extra.log` | 641 | `81a9b3c452863b05bdb8958c30ba4be7d93545f2f8695a43a0eddd9917c13885` |
| `b4_nurec.log` | 2862 | `de195402c0afa2899b3ec0ed452bcbdab669ff746394098578e11b0e3d4cc324` |
| `b4b_braw.log` | 2166 | `d75e1d6b42e15bf815b87fc02128ff772890bff124b2a314c2363d89fc8f1a81` |
| `b4b_fetch.log` | 823 | `f7cf06d5fb9274d300f0825f379e0c94ab43eb18c34788ba7d13bafa273a6718` |
| `b4b_fetch_plan.log` | 253 | `c121fcaa9b89c4c350aa888cbe174ef9ffba0961050e7d076107f6c9ea540b4d` |
| `b4b_heldout.log` | 2321 | `df1c8dadda95150190f7c1cdbd2218b508c862c31d4513652e5f278a66831f9d` |
| `b4b_moments.log` | 6601 | `472d1ea5e3242ecddd3a690df40e64d22e25ccc3df5fb45798d8fd0e28efdd96` |
| `b4b_nurec.log` | 3205 | `b0a680fd24bb3f9f6d48798440f2d392b9152efc1ff9428844881826a0b5a95b` |
| `b4b_ordinary.log` | 8172 | `72c16c6c044a2ddd4c0878da616f08742fdae11658f500a5370b29fc68d9ff4c` |
| `b4b_verify.log` | 13694 | `1530da7c16b202b4ffd5b943f11087604e9dcd5c9308c6db33f84bb4ead7bba9` |
| `b4c_graded_heldout.log` | 2454 | `27be1737ef8a51882aa7f1913233190423174ca80ab08c092588b506db4d1786` |
| `b4c_picking.log` | 1733 | `d2c3a9601c81eea71661ea09500366b065fea0d03624b1251df789599e936c89` |
| `b4c_raw_heldout2.log` | 2345 | `3612e004210840117dad17dcdbaf1c274e9c684dc96d9af32806d15b01042930` |
| `build4_before.log` | 6154 | `d8f7133ac8cfd985a2df9b5655507f74b289cbef5222999c9d1b4eff742b6570` |
| `build4_chain.log` | 1345 | `0578921b093eb1d38e56e3c9c22f114576840dd9b547f3b21b17ad3dab04ab96` |
| `build4_chain_stdout.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `build4_short.log` | 31247 | `25ffbcbf2262b196381b03a6bacff5167e286289bb174fb47d2a05c109f446a4` |
| `build4_tests.log` | 57909 | `bc97826bcef05ed308f4cbf045937ecabac33713b6f47b504d5f05980eca588e` |
| `build5_before.log` | 6520 | `56bea88fc0ba69b7ed1dbe069c1d7a423725276784203a555d3377f6c54361e4` |
| `build5_before_attempt1.log` | 4739 | `346fe9e5007998de4244db6597bca767e702acc9de5b361652657c9170be46ae` |
| `build5_before_attempt2.log` | 6146 | `d604f95b9e9eb668ff92b65ff05f02f76e5b5ebee720d16edd912a0743dd01f6` |
| `build5_chain.log` | 8060 | `67d206020d44ac1052b16ed521f8121b9f7f178496a44ad0873eed1f919cea6e` |
| `build5_measure.log` | 9416 | `e2110e0687302238d7846adac922db2958dce0dd561a124acf366a8a0a40ed93` |
| `build5_short_part1.log` | 29701 | `0dd0bf792ae9746cdd18612305a0848c4c45ce9b6e2fbc84578e4b23c6cc87ed` |
| `build5_short_part2.log` | 31222 | `74abdbe17355034c6ad7cf6308b42e730051b73bfa6410f3ea9cf7ed47ec1675` |
| `build5_stats.log` | 431 | `c26661127d5810434ba3e096f9ba808732310cbdf7e90c3ed31284469d8f7b9c` |
| `build5_tests.log` | 65914 | `cef1acf6080f9e185360d217304766566011f97afcd9102dac33ef6db8470a00` |
| `c1_params.log` | 5192 | `38b7a7a20760b392e38aeb3478786419ffec2cc312634f1842f1c638a8043677` |
| `ceiling_chain.log` | 763 | `ca14e707ed01368efbf702bb586d094693fdc87f20a4e354ab45578749733ad6` |
| `ceiling_extract.log` | 1212 | `45520f89724ca5fb29a595ddbe1ee1a4f68acf6e8c5f21a8b8f8ee1657213117` |
| `ceiling_mlp_long.log` | 3645 | `ad381ab3fe0d0961a9bced465757471143de1467c9bba0ac28f7e592ad8d2c8e` |
| `ceiling_probe.log` | 13315 | `9dd57d708be6c9f32c45b777cfa2954a93442446ae452c68b00ef57756124943` |
| `ceiling_stats.log` | 881 | `21637aad9396af603cb99a886c8377b46feb581a60c52fce6677cb3732b18d7f` |
| `ceiling_summary.log` | 116 | `c71143cc3c079697d2d8e0e1626dded225ef6c0a185fdd96f624ec3329c70439` |
| `link2_chain.log` | 1569 | `138aceaa76f0318b92c0468afdea450dc80c00979e23b3a954aafad0992d6049` |
| `link2_probe.log` | 16691 | `8cf1862ecd2e548606b9af591658f5303ddbd363de22914701fa3cdc4f0c9f95` |
| `link2_replay.log` | 9613 | `522ed308ab75cf548e606793467e276c173dd09bec990ef84552d075fb0c1e0d` |
| `link2_stats.log` | 1240 | `f20d76fc08f4dc54db47e357cfebc49b6e7078deab446d2f89c998cf09b62bc9` |
| `link2_summary.log` | 114 | `738b77f6ab288c5a63fc9cb24df13e2f477451c408b32b00c43ac18e7fdd362a` |
| `pilot_carry.log` | 38985 | `995400bb29f86903a1fd6299d998c4b618c4740b9a83f7e9b86729b7fb44ac05` |
| `pilot_carry_attempt1_import_crash.log` | 5399 | `eae339470cf259fa27a4c729792e6f31272e0c663577fdcc333eb3015825c2d2` |
| `pilot_carry_attempt2_3pass_stopped.log` | 27692 | `233f954b05c93d6bc6886a31dfb0a40e158109d6f0d2240ce1642785afeb6408` |
| `pilot_carry_attempt3_memory_gate.log` | 537 | `06d5c4c086e0a4ca8cb25ce4dede935d551e86085460961645043c4028645dd7` |
| `pilot_carry_part1_crashed_p1.2_check.log` | 42632 | `fc8c69dcefdea1fbf9be6d8e2e76659e3cd76fc51b0576b258cb434cbfdf65ab` |
| `pilot_chain.log` | 1639 | `2c2819cd82c704a398b7a30d196584ed2598fd834f2fe1a3c4bc106ffc3301cd` |
| `pilot_final.log` | 29563 | `d598d00cd808042521d1119e791cf10898179df1203b22126529404846bc6e0b` |
| `pilot_step0.log` | 255 | `966bd8582404c5217155865b0ca8ea75f7d3970b90598d8c6412a11dff1b381d` |
| `pilot_wiped.log` | 50048 | `c0a6a596802e1d2dcdb1baf50398e99ee495b5784c4324b92cad8d890a88a937` |
| `pilot_wiped_attempt1_import_crash.log` | 5399 | `bb7e473e161e1331fceac0f973722d16e32cc86c458d1e641b5a6183ef622cbe` |
| `pilot_wiped_attempt2_3pass_stopped.log` | 27692 | `64b8e94d1a0de0dd004652d865715a5897aeb3e050ae344fa07997e55fa1eb4b` |
| `pilot_wiped_attempt3_memory_gate.log` | 537 | `06d5c4c086e0a4ca8cb25ce4dede935d551e86085460961645043c4028645dd7` |
| `probe_manifests.log` | 12005 | `7f7413039cfefe8db8d4cdfdd2706748b3d5f0035db650a93da398c2ce0a75b0` |
| `probe_timing.log` | 8870 | `f9415d106d6dafabdc55d1a7598b11139225a033e3d0b123784342f1bc6caf0e` |
| `s1_graded_heldout.log` | 1937 | `cf274408ce48d7e8c343e739fa5e103db133bfde0ac63f8d6df84e6ec4df6366` |
| `s3_moments_graded_heldout.log` | 634 | `eff0632bcca778d34e28c80e4526b2badc1f4d933c9781e64596e426ebb370d7` |
| `s3_moments_picking.log` | 628 | `ba6eff01fc2be40abc807c1a8e93502a27a6215702a431880d3a5d829ba85ceb` |
| `s3_moments_raw_heldout2.log` | 625 | `b942488869ba19686750d4543e852cc4cac93be656a5d4a427c11d9d61a6677c` |
| `s4_cache_sample.log` | 14187 | `1b9f4cc2b3e99943475f40f467ba69b803590340e9913e03479d339c0f9d8947` |
| `s4_sample_list.log` | 350 | `9b390bc9ad584412d253611384c128a81d855909db91ecf5517753cb42d6f72b` |
| `s5_cache_chain.log` | 515 | `a4a593fa97d9d8d98b0c7f0db815e408d38ec8f37c94d73b4da81f7c5632050f` |
| `s5_cache_check.log` | 23450 | `78a6fa702cb0e8fb94770224cbdd326fb3f00a0c1c09f50778cd242f5db6faec` |
| `s5_cache_graded_heldout.log` | 10649 | `c4b9840f4d38964902863c3f909df9742d8a9ccf099c7fe79bfb2a3f38773726` |
| `s5_cache_picking.log` | 10627 | `0c41eafa5f6193fb4ec9e15b48a10c9dc1d8e8e7f7d95cec22639bb67793429f` |
| `s5_cache_raw_heldout.log` | 10597 | `aed410a392775fb1ffca14819eb8e37ec959990569fbf3d3390712494f758acf` |
| `s5_cache_training.log` | 11459 | `da582c0f6d8a9483bfb1fe6e993c5919a76dc77ec4a86e712e85d5c44e9afa7a` |
| `s5_cache_training_batch8_STOP.log` | 10103 | `f06d02f1d80afb8b69a942cef716e389f8304301e7c9f6923b5e7557e26afc68` |
| `s6_all.log` | 54382 | `e3a536173bd4d6f05934f791c69c6d1fda8762f2cd6458b27bb6a196ffcd4e28` |
| `s6_all_attempt1_save_dump_crash.log` | 29280 | `25f61f6fbfef33877c527773e10c26b2f8620cca8b0072b808c420ac9c129296` |
| `s6_all_attempt2_killed_by_session_restart.log` | 47278 | `62be2e15fdf7fafa2ef6a237ee357b0771bd50965e0d6bb9ce9c3eece3669feb` |
| `s6_chain.log` | 658 | `9d6c257fe42ea3ec2aa221b655a348757ae80902731192ef9e5cf5ff919f4695` |
| `s6_lidar2img_constancy.log` | 301 | `306a204e178168a689559b8eac4235f9188d3f7ab4837ffaf697aa523442cba7` |
| `s6_rest_before_slot_emb.log` | 51782 | `c95cd6de99918174f6b45cbd0695d8cbf3852b38048b2d1b3cd06a6940f64179` |
| `s6_t2map.log` | 29720 | `d839693fe99a1df789c0edb90e84f74e7fe592cd5769394b8fee1ca66b6692bf` |

## Run folders

Whole directories, so no single hash.

| Item | Where it sits | Contents |
|---|---|---|
| The memory saved at each moment, for the test of what the memory holds | `{experiment_store}/runs/link2/` | replays and reader outputs |
| Pooled past views and reader outputs, for the test of what the past view holds | `{experiment_store}/runs/ceiling/` | arrays and reader outputs |
| Step-0 reference dump of the model checks | `{experiment_store}/runs/build3_tests/` | one dump per check |
| 12-scene closed-loop check | `{experiment_store}/runs/cl12/` | not finished when this was recorded; nothing from it is hashed or cited |
