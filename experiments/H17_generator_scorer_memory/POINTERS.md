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

### H17.carried-vs-wiped, first training run, memory carried

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

### H17.carried-vs-wiped, first training run, memory wiped at every moment

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

### H17.collapse-fix, the memory-collapse fix, 930 steps

Template location `{weights}/drivejepa_physicalai_memory/h17-r3-collapse-fix/`. Where it sits:
`{experiment_store}/runs/build4_short/`. The record pins `ckpt/step0930.pt`.

| File | Bytes | sha256 |
|---|---:|---|
| `ckpt/step0310.pt` | 35061136 | `594556a7ba727e479a2e417d0e62a59c386d202abb815e6707bebadb51bdac01` |
| `ckpt/step0620.pt` | 35061136 | `aa556e22efb18293f65b0c9f7eb677c078c63bab116ec68f18e440b42867a7c8` |
| `ckpt/step0930.pt` | 35061136 | `645157874f555f922160cde67040e1f8f067954d0adf95ce2839dd6d3755d9fa` |
| `resume.pt` | 105198333 | `e8976704078ebe941222770a1186ebdbe01d9f1eaf0ea39b40c1a111ac36ecde` |

### H17.recall-lesson, the recall lesson, 930 steps

Template location `{weights}/drivejepa_physicalai_memory/h17-r4-recall-lesson/`. Where it sits:
`{experiment_store}/runs/build5_short/`. The record pins `ckpt/step0930.pt`, the checkpoint
the headline result was measured on.

| File | Bytes | sha256 |
|---|---:|---|
| `ckpt/step0310.pt` | 36748734 | `88dc7d6cb3617c1037086ff547af041ed93047888fcef9a97ecd1e6e84cfcf89` |
| `ckpt/step0620.pt` | 36748734 | `0a0291d2e4245867ccf848084e9763d3252a4d672fad2fb5b3fe7a43b04f6536` |
| `ckpt/step0930.pt` | 36748734 | `84359783ef502e1ccc3a2dcff013bf45d215f9fb11e046cc8110c64581434e21` |
| `resume.pt` | 110139953 | `b688396305885b14fad21b9cea99fe67620cb9a52129dbe82816566d7d3ff5e9` |

## Per-moment picks, grades and run records

What every pick grade in the record is computed from: for each scored checkpoint, the path the scorer chose
at each picking-set moment (`<name>.npz`) and the grades of all 32 paths (`<name>_grades.npy`), and each run's
`run.json`. `06_recall_lesson/interval_after_review.py` reads the six dump pairs below. Where they sit:
`{experiment_store}/`, then the path given.

| File | Bytes | sha256 |
|---|---:|---|
| `runs/pilot_seed0/carry/dumps/p01.2.npz` | 9329536 | `e5972d11bc0ec558288cc6b76a7e8e044a2a7638d31aa3c9ed6bd402413ef146` |
| `runs/pilot_seed0/carry/dumps/p01.2_grades.npy` | 1303040 | `7cc23d762208b1a715520eb37cf9c35e72aedd81620410e515ba53cb1c03a227` |
| `runs/pilot_seed0/wiped/dumps/p01.2.npz` | 9332753 | `c049917f176d9a37e84c30422d9df440ddc50c22f52396149e66bb80288d3e4a` |
| `runs/pilot_seed0/wiped/dumps/p01.2_grades.npy` | 1303040 | `5e01b53b65aaadd2ef0152cee1c93daf29f129194b3ecba79ef43106a1bbaed1` |
| `runs/build4_short/dumps/step0930.npz` | 9323163 | `fc9ca410d7eec0e9d8852cd6cbd6a4ace547520705807e547f7b70e658a8abd9` |
| `runs/build4_short/dumps/step0930_grades.npy` | 1303040 | `a6467f4910f0252a5fbea50bac72044b175077873d9f0454c1f2eebe266f14d1` |
| `runs/build4_short/dumps/step0930_wiped.npz` | 9322759 | `f8d69dfe9489ce5493c017b7fe5a3442bdc432d5356ac38aad4aadf9049d47d9` |
| `runs/build4_short/dumps/step0930_wiped_grades.npy` | 1303040 | `0eb601a310dd184538bbb84e3756913b4af102e4a91c7e1e22b824be9ea87029` |
| `runs/build5_short/dumps/step0930_carried.npz` | 9328637 | `36acaf4b02cc3cab6d4f622238f7c1e5f15cfff578ed873e271cecee4b024d20` |
| `runs/build5_short/dumps/step0930_carried_grades.npy` | 1303040 | `51c6b5c4636189e8652c4fb8ef167fd7619121356cb0cc042827a2d6bed4c8e9` |
| `runs/build5_short/dumps/step0930_reset.npz` | 9328284 | `3415bcaa4730e9654b4269d01461e9dd89741b4d20fe7c59a8e0a7927fbd0284` |
| `runs/build5_short/dumps/step0930_reset_grades.npy` | 1303040 | `88a70a5eb439363bcbca1bceb42ab2ca3a11abd379b1c6142903af049aa99cae` |
| `runs/pilot_seed0/carry/run.json` | 3198457 | `f7611c7947a4b52e9ecac231f75d2673da586f47c6cc53e2c0cc069b9afffe4f` |
| `runs/pilot_seed0/wiped/run.json` | 2936808 | `5d734580b17bc8d103b62e487edd0abe30d0d35f3bb55b58b8c9fd739f3b8491` |
| `runs/build4_short/run.json` | 322527 | `e72546db4d709af078b37fa440bc39e575655eca08a0dab0a26764c8b12f55cf` |
| `runs/build5_short/run.json` | 490038 | `01287c17733a3b86d5444f7f267b9cf5c4c059cc15fcada498b59035d454d061` |

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
training site. The reports cite them as `logs/<name>`. The closed-loop steps' logs are listed further down.

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
| Closed-loop drives | `{experiment_store}/runs/cl12/`, `{experiment_store}/runs/cl18/`, `{local_store}/runs/new40/` | every drive folder is hashed one by one below |

## Closed-loop drives

One folder per drive, as the simulator wrote it: the rollout file, the camera video, the annotated replay and
its trace, the logs and the configs. The record keeps only each drive's score, per-tick metrics, gates and
provenance, under its step folder's `drives/`. A folder's sha256 is the sha256 of the lines
`<sha256 of file>  <path inside the folder>`, one per file, sorted by path: `find . -type f | LC_ALL=C sort | sed 's|^./||' |
xargs sha256sum | sha256sum` run inside it. Hashed on 21 Nov 2026. Folders whose name ends in
`.simfail_*`, `.harnessfail_*` or `.killed_*` are the three drives that did not finish (step 12's report).
125 folders, 21375679269 bytes together.

### Step 09, memory carried

Where they sit: `{experiment_store}/runs/cl12/runs/`

| Folder | Files | Bytes | sha256 |
|---|---:|---:|---|
| `memory_carry_04749bb9` | 67 | 106423846 | `b22cb6d2130fb8449fed3bba179f06d332a077d8687a90b497439f518c98dad4` |
| `memory_carry_048b974e` | 63 | 157227170 | `31c15a135ad7134a9aeb51b3c49fbac3dca81c9e5f0e1fa6b96f2336702a29f4` |
| `memory_carry_5f064b02` | 63 | 217706101 | `6fa2e26bba2dcd4ebe6bbc65641f82d3ab8cee6e34daeac30061f47a83174f3b` |
| `memory_carry_8023dce5` | 65 | 160024131 | `f810b3cfea0deb68b894ae34860005f1e8c2b02da9599a1a35ee1d9b5412f79d` |
| `memory_carry_b1b170d0` | 67 | 142499234 | `fca5252809b1795719f3c175944538384ee653110d09ca0a3216c97c373dfc15` |
| `memory_carry_bae463d3` | 64 | 176451191 | `0dde40eb79e68913651e3ec969dbc2485c541cafae2db24b31fabe589c42d552` |
| `memory_carry_bd07fb91` | 66 | 144399229 | `3b0411e688279c60caca77f0d9c41ab9e93b9f3230dde19921a9d0e475cdb0fc` |
| `memory_carry_cdd9448c` | 66 | 175860804 | `cf46922ee1ea32828875c75683717c254ecbb9dc570e025b7977bae5994c0303` |
| `memory_carry_d1a25a99` | 66 | 224982198 | `ae9198a4adcd7babea3203101fc2967ddb08eb39655622778ccfe12778a5880f` |
| `memory_carry_d2715023` | 67 | 251275371 | `9b9bb1aa250a57762d5df2c99df796da868934e4f3f593a9f34127b40e7ed1b3` |
| `memory_carry_e121e37d` | 66 | 171817997 | `39f655eaaa94a13aee3c13e9b155ebc3b5ea89f09b395195f84f71fc06839a7a` |
| `memory_carry_e3ed554f` | 63 | 235310261 | `26ab8499b69ff1a63bfbf805bf843926563bd375ea7f8c5cf417b0250932cac9` |

### Step 09, memory reset every call

Where they sit: `{experiment_store}/runs/cl12/runs/`

| Folder | Files | Bytes | sha256 |
|---|---:|---:|---|
| `memory_reset_04749bb9` | 66 | 99650024 | `3b48775a752c250b864c8dd6eacb7f62b9858c75d96fe87493b74d37175b7ff5` |
| `memory_reset_048b974e` | 63 | 161407586 | `fc3ce9aadb65205916d7255b9a94264b360a3849eb6427d7ab261cef657826b6` |
| `memory_reset_5f064b02` | 65 | 215273160 | `5b92cdf32b205e1d0a3f42863e015088dbb2928d208d276f663ec0e4a4333317` |
| `memory_reset_8023dce5` | 66 | 158852973 | `d00c9f4931c73664ac821b679c430563112ec1d238073fe18cc193fc06981d3e` |
| `memory_reset_b1b170d0` | 67 | 138549259 | `5cdb958b2c4b9d04cdff7c48e8ea8e4edbe76a30b36465f3a3da7bd639488c9e` |
| `memory_reset_bae463d3` | 63 | 165526336 | `9b3f4418d7b5fa3731cc0d6f35b2c82ac8f6cb1fb22d3888bbef56cfea2e2e3b` |
| `memory_reset_bd07fb91` | 65 | 138753980 | `0cdfe126473091850c03851040a82edb635cfdf8bbd85a6be7b891e9b12cf7d3` |
| `memory_reset_cdd9448c` | 66 | 175887544 | `872aade3c92e777a9c7cd6ccb73a587f4501bae111fcdaa30d6388ceed56ad5c` |
| `memory_reset_d1a25a99` | 66 | 228556809 | `7dde8393967b8cbda8bfdf44130045a5f1025f56df2bebf99aa0ceee6da6e093` |
| `memory_reset_d2715023` | 66 | 247249396 | `8dd7f175a6f7a5115c365178808799a64c2092ea69740ed952b4c8669c6f6ffd` |
| `memory_reset_e121e37d` | 66 | 171471759 | `551624a4c4bf7a9aa8b2bfdcd9df639949cabdf7d31c3ee203d09cfc46751b8b` |
| `memory_reset_e3ed554f` | 63 | 235061958 | `7ee2e7b7cdd4d528b04763d2d72e0dfb40cc9aca10db89fcc71794183990ff27` |

### Step 09, the starting model (driven by its own lane, 31 Oct)

Where they sit: `{retrain_store}/018b/closed_loop/runs/`

| Folder | Files | Bytes | sha256 |
|---|---:|---:|---|
| `drivejepa_018bp06_04749bb9` | 66 | 97327802 | `2d5c2192adef892ec56ef7c7aa220c5a9d2c698c026c5bb21485c216c2583516` |
| `drivejepa_018bp06_048b974e` | 66 | 163518769 | `7e08e15184baf1ca18cfbf4129d3762b0cfbb01a4c61660dec251ba7a8b03c75` |
| `drivejepa_018bp06_5f064b02` | 63 | 214402258 | `5b3c2f0a86c461c0a97da3e7170055563920cfc9ac4483e52a7e2e46ec98504c` |
| `drivejepa_018bp06_8023dce5` | 66 | 149037060 | `b5c7f262c711e056c4ecc2af8d4cea97fdaf93668b8f83cdb7f45d6cd1af43ae` |
| `drivejepa_018bp06_b1b170d0` | 66 | 133429147 | `00887db852884b67230c9e485d58491e512ef8bbd4a4d3e2453bde72f9c4c2f8` |
| `drivejepa_018bp06_bae463d3` | 63 | 162465196 | `224ec28052ce44ed0b3503c9fc183dc0c60e0b41163ece4e61e6532bb1c2957e` |
| `drivejepa_018bp06_bd07fb91` | 65 | 144977685 | `e1a51807a15c222e87629c4c11e09cacc99dfee461ee8aad6ca393d44547cd39` |
| `drivejepa_018bp06_cdd9448c` | 63 | 172535211 | `c01b2b828d572abb2cde97d1a8dd8c91f6c93207e2523d23075cade1d84df3eb` |
| `drivejepa_018bp06_d1a25a99` | 65 | 221916233 | `a0c2f11c8b16814872ca23fa8186d8509a2e5d93287e9372c537d7ae689145b4` |
| `drivejepa_018bp06_d2715023` | 66 | 240259812 | `7e694c96bbdb03803741ad51bc74a9695555ce63dce44bc85319bba73cd8927d` |
| `drivejepa_018bp06_e121e37d` | 66 | 170299625 | `558e7f2960ffba047ca48b377fb0c4c970fe18c98ccfc6ed9b4349a29d3f2bb9` |
| `drivejepa_018bp06_e3ed554f` | 63 | 235827753 | `70685ade4935b79c2b0d2227436b5bd783197e50fa62ebe90d8054162f65f4da` |

### Step 09, image check: the starting model re-driven on 12 Nov

Where they sit: `{experiment_store}/runs/cl12/imagecheck_018bp06/runs/`

| Folder | Files | Bytes | sha256 |
|---|---:|---:|---|
| `drivejepa_018bp06_04749bb9` | 67 | 97774909 | `ea8e51de44fd025a639c12a0a5c0a079ae438d6ccc2a8e245afd5f8a32b8bcbe` |
| `drivejepa_018bp06_8023dce5` | 66 | 149146952 | `ebc7ef8a0738bfe1eebe928b6d46889844e65e348a7bd4c7e1393930dd3f2089` |
| `drivejepa_018bp06_d2715023` | 66 | 240501064 | `e5e51acf181911d8dbc2b14005a55fb394e0b59e26bccbb4950812c8a77f5740` |
| `drivejepa_018bp06_e3ed554f` | 63 | 238125434 | `b70150f656ad519b55a49f7af0d7e1895fb09191900d2e37ec9964ab560d07df` |

### Step 10, memory carried

Where they sit: `{experiment_store}/runs/cl18/runs/`

| Folder | Files | Bytes | sha256 |
|---|---:|---:|---|
| `memory_carry_01d503d4` | 66 | 112391642 | `6ebec1e52772518a6bcd894ce9ed9eb3a4315b403d9d728fc345aa291d1425bb` |
| `memory_carry_023b7fcc` | 66 | 135247386 | `1cca2048ad1199251c51f4618b773a4c9ca33b21daaf882870b7ee941d874c90` |
| `memory_carry_0245ff75` | 68 | 201633573 | `4a40846a2e56a3e715302604f44645ac735ec6242ebe6e2c798028c9ac345ca9` |
| `memory_carry_026d6a39` | 67 | 177459906 | `9172ec6902fdf1f3bc9acbbe89cd640cb1eb8546ccd01bfe0f50a3fe19657d8a` |
| `memory_carry_02e075b9` | 66 | 153593333 | `3887997669710c357b3e8add56520bed1b97fefd1d517bd3ad4f5ef4cab480d1` |
| `memory_carry_02eadd92` | 63 | 237426379 | `fe03f2ffbb953479e5c9062c7c31e9cc557a90433d4c4b601133425b82564743` |
| `memory_carry_032b6f21` | 68 | 172062021 | `b58641bfe2493bc34a1ec626e40040c9c9d92e3d51ba58e1a09acc730dca7d15` |
| `memory_carry_04394343` | 66 | 207303695 | `4f2377af646fd3fa9acaa7e96dc1c2c482240984750d79dd10ab7238bb5e4513` |
| `memory_carry_16b83346` | 66 | 204337359 | `8051a3fd60016f8dedee02fa1db4ccb98aaadbe79fb9609506d4f775b1eda52e` |
| `memory_carry_1acf2f5d` | 66 | 183084849 | `c33af4cc03e0f87351633763372895221f8901689f28bd60f0afd304cea6bec6` |
| `memory_carry_2e1ce314` | 66 | 157030316 | `c40e85617aa0fa4578786cc277a73224184e083c6c0a77cf56d411d79d27fc94` |
| `memory_carry_6725df2d` | 66 | 130781687 | `762818511f56dc74c1e8e09101932b278c1a485ca68ce75415e700e4aef72cb2` |
| `memory_carry_71d7d03b` | 66 | 214247995 | `5344a6c6dc8353de0659c3c0ee60b4f785c25ed664cf812060b3a9fd4f0b4cac` |
| `memory_carry_791d68ce` | 63 | 159639549 | `f0794639d281d0363a281056647ea715e68aa2ea15b06914fb6a3b239c83d01f` |
| `memory_carry_89f2eeb8` | 66 | 192289726 | `7dcddacffb87f4c064d8e812bab37091c33f316597ebbac674f1b5f2ce000870` |
| `memory_carry_98dd56b2` | 67 | 184072244 | `4856ee5cef45fdb19ecba2ab0d82523e16136beeedecf668006c43647ec6b3a2` |
| `memory_carry_acd22430` | 66 | 121692231 | `7813e3b3d2b2bc41675b94cf66387a892903fec67278acb6086e6767e67a369d` |
| `memory_carry_b44a4a44` | 63 | 189402630 | `df515f526ffa17d2e7cf64770ce6a58a51610c997c567bbee738d84deb889c57` |

### Step 10, the starting model

Where they sit: `{experiment_store}/runs/cl18/018bp06/runs/`

| Folder | Files | Bytes | sha256 |
|---|---:|---:|---|
| `drivejepa_018bp06_01d503d4` | 67 | 108432156 | `ad40363c0aea8060908ff27424c80daab7f557c78eb854591f1298f8a0cdf286` |
| `drivejepa_018bp06_023b7fcc` | 66 | 134565398 | `d36fd273504f79464f8f02f85a06b683d80813736e495bf36acc150d8497124a` |
| `drivejepa_018bp06_0245ff75` | 66 | 196557683 | `e58a46d46cba94405fa92b0e01932cb3be00dc325ed05789d5d95a7fc2c282d0` |
| `drivejepa_018bp06_026d6a39` | 68 | 180669310 | `aa3e4806ad1df5daf5aa0a2101b58ea672fe224e6489373304774fe949ce3f95` |
| `drivejepa_018bp06_02e075b9` | 66 | 153380823 | `92b3cb9923c05d3b42eb547901686aba3d60a9c1b5f02dd633919a431436bcb4` |
| `drivejepa_018bp06_02eadd92` | 63 | 237287213 | `49d24454da2223297fcae7113322671e3f712ff0ac0eaa97fb5916d5d06a1507` |
| `drivejepa_018bp06_032b6f21` | 66 | 170674851 | `e2ab6ac48620b1d56622142c448962d9362e309ef145739cfc81049cdf527cc1` |
| `drivejepa_018bp06_04394343` | 64 | 214867375 | `8e4e14017f7c6f345e60c4817941ee698418750b94c1dddcbca2c45d98868505` |
| `drivejepa_018bp06_16b83346` | 66 | 194152333 | `d7ca79d64a01e22cb00be67d922e38e9cdb69ccf1a2e1acabcfed842c2ca90eb` |
| `drivejepa_018bp06_1acf2f5d` | 63 | 183087334 | `b959b394f6fe433f0bec6862e0e90e748d14db182d9bc249b4a515300124ff77` |
| `drivejepa_018bp06_2e1ce314` | 66 | 156810114 | `299f9193b0be54883cf578741cc8e92296f33cbb5e76c11ea7562c096f6296a6` |
| `drivejepa_018bp06_6725df2d` | 66 | 131169575 | `b465e844e719cac5daf9e3144e9b1b727322d42f4d32766e2f6510a5fb97cfd8` |
| `drivejepa_018bp06_71d7d03b` | 66 | 213019611 | `30f9b6e5cd37fb21e591d3394470754b3e76db536efd1e8f2608d1fc9c06e85f` |
| `drivejepa_018bp06_791d68ce` | 63 | 152636791 | `c334d92f4e2611b126ea86a9d530f61af5c120acd78c3dbc38846c39293198bb` |
| `drivejepa_018bp06_89f2eeb8` | 63 | 192193465 | `6555546f275ad17ed2a7d47be080a6f7565d1bc35033291549fb888afde5a495` |
| `drivejepa_018bp06_98dd56b2` | 64 | 182305371 | `e59280ddf3bbcee92de329a3079ded919458678e52256f2c6141060e819bc0dd` |
| `drivejepa_018bp06_acd22430` | 66 | 120451536 | `152a77c183d2c2b092c8ac10f20cb632caa27d8ca351e336e9d153c50457aa24` |
| `drivejepa_018bp06_b44a4a44` | 63 | 188035921 | `4729818c895ed27e55262fe008946dd13ba2cbc208844181d38aa4ea9be59de9` |

### Step 12, memory carried

Where they sit: `{local_store}/runs/new40/runs/`

| Folder | Files | Bytes | sha256 |
|---|---:|---:|---|
| `memory_carry_29815599` | 63 | 198001509 | `aba6cbdd1fdef8772c1e3e828e601160bf44780aaef133ec90cd22ebdc975471` |
| `memory_carry_320c11f3` | 68 | 152797592 | `57faa0b9484a78791d62fa077ac9deef8276c2e714457cb611978fac07be174e` |
| `memory_carry_34226720` | 65 | 137231089 | `336b77a34835b2c88210599699e102f4b4fdbb14be423d2fdcbb9532305e0878` |
| `memory_carry_44409f00` | 66 | 146208268 | `f6fa4e91b44b7e2d8d3c80e8b828ffd14be50a05109569dbc4a63a487e3fd88e` |
| `memory_carry_44409f00.harnessfail_symlink_not_mounted` | 35 | 197635 | `7d3f50e0295526eef9e8096bffb087c9f15cd0e50861499624178696561363b6` |
| `memory_carry_49e0ea24` | 66 | 141559386 | `8ea70bd3f33ef5c29773575db81e133ab2d540f94ffd81463ac9db7a05777040` |
| `memory_carry_4a495c2d` | 66 | 170300179 | `5fcb793deb0c9172ceceacccd9ac9de95d0bcce11913c80b0061ac9e562ec1dd` |
| `memory_carry_5c1b7555` | 66 | 88592670 | `9b1b4abb90e394274b3f950340c679a285218f0119f3561bc30d8b7385175765` |
| `memory_carry_68e53160` | 65 | 155581927 | `a12a02a591b62427c0bfe29a1ba1103b198869607d8d50fe8c04ae06d5df1b71` |
| `memory_carry_6ae687bb` | 66 | 150330771 | `6a5784d6a541a0fba970aca6e9d6ccc283d6be3f3e5ed558578a222bed5e12e7` |
| `memory_carry_7747986c` | 63 | 123497973 | `c96c29a2fcd47bbde71012b711361f355341dd0e6d163fbd4f56291ae9f14741` |
| `memory_carry_8578f2c7` | 69 | 188482650 | `32afed2ef61c23bf08e23568214cb8a9c62ffca21b009b006e0b4e8169421387` |
| `memory_carry_8fd8f4f0` | 66 | 217383156 | `2955c5e50a3f4cfc0d9d9703a2e40440889649f0f097c85a5dec6b698f5f1fd2` |
| `memory_carry_9d8fca3b` | 67 | 149381452 | `4dfd16b5b282611e06909308353d29832c3f785935edc35692ef0cf7f7a3d9d9` |
| `memory_carry_9e6753c2.simfail_route_foldback` | 44 | 265405060 | `72e29c87faef56d5807701610100b7ceb5cfc339c8ef0bff1f145deb14899d02` |
| `memory_carry_a32549eb` | 66 | 108408977 | `78c2426ecfaad6444449022bdfc5fce74a9c3cbab03e7284a752c91ee5d3a05f` |
| `memory_carry_c438a779` | 65 | 179877714 | `a4d03b2d72abd7b4f2fb724c1cc36f53e963c9036446d9f8aa6d6e2f71ff65b9` |
| `memory_carry_dd0f84df` | 66 | 265188159 | `ce68f083fb42ee84643636966a74a454c87b7519aafc0b162750405320247bea` |
| `memory_carry_e883822a` | 66 | 173158643 | `d6f493eecf863cd7c1867631d62da6ca17cccc96e343481884b53a6ffa005368` |
| `memory_carry_f005ac5f` | 65 | 195902299 | `c6a589837bdf90fb0156982043b21fed6a95e37052cb1bf8cadf69cd42d36c90` |
| `memory_carry_f3a87e7a` | 66 | 163123396 | `59ee65572ebc5232f28bc2201d42b02e09e36c3812b87363302459b2e92065f6` |
| `memory_carry_f890fe10` | 66 | 249831497 | `6ad0a5d91ca73b2f2f974ed4efb1eea72dd6bd2c48505b9e494b85f9a62650e2` |

### Step 12, the starting model

Where they sit: `{local_store}/runs/new40/018bp06/runs/`

| Folder | Files | Bytes | sha256 |
|---|---:|---:|---|
| `drivejepa_018bp06_29815599` | 66 | 204349152 | `b2f3ff621411d6d4dd66ae55f5cbe8c7acce5cfd10d653a6defe54d255b86340` |
| `drivejepa_018bp06_320c11f3` | 66 | 149436397 | `f0f0030cefc7ed70efd5a65f78eb494526eb0fc41b3ce7a81b943fd037e61af6` |
| `drivejepa_018bp06_34226720` | 66 | 135796622 | `d363f7a7ad98f532911de6c82fbf436089cfc7776d2657a490c53fa03cf24ef3` |
| `drivejepa_018bp06_44409f00` | 66 | 145849432 | `c88ba74e5d26fa02a0f098333f24fb6b1c9a72a84587bc387fc06fde4f352865` |
| `drivejepa_018bp06_49e0ea24` | 66 | 141499706 | `6547a336e65143871d4c780d97c3b1c849855044002c671d149f0eae8c441187` |
| `drivejepa_018bp06_4a495c2d` | 66 | 170404617 | `72a3d7b9c28f0d90ad7a7bcba9f3ace92766af2fda8be281c3d07e69f56c0f29` |
| `drivejepa_018bp06_5c1b7555` | 66 | 88522135 | `87ea4b38d13e17940e02d35c67e517745e9d9ebfe6fc7776e1a290a71314b9b8` |
| `drivejepa_018bp06_68e53160` | 64 | 147642602 | `4113117d7b79abc131d93fb492b63687efa76e7d3ea4a12da7ac15dea4fc6ce6` |
| `drivejepa_018bp06_6ae687bb` | 63 | 146945698 | `2fd0d165de6889b2ecebb16db4757a993367468c5dee13eb2fe83e958b311764` |
| `drivejepa_018bp06_7747986c` | 63 | 127213674 | `cb367d7c5b7dc9a7aea3b3af1fae4da123e94fd83e99b9364c594375d7494734` |
| `drivejepa_018bp06_8578f2c7` | 66 | 186010464 | `d3a13da49564a498be957e00b6ce1ec819a58350f59190a71319736f3d180150` |
| `drivejepa_018bp06_8fd8f4f0` | 66 | 220256721 | `8b8690aecdfffac1fbbabbc8d78e1c0430811dad0234533da57fb262cffea2de` |
| `drivejepa_018bp06_9d8fca3b` | 66 | 148977233 | `a603ff619a328c7a5d89016f2112a158d31ef0ac2ecdae24f675501d65dd2643` |
| `drivejepa_018bp06_a32549eb` | 68 | 110068683 | `aed032690e330e4e68d6a8928a2c2d0c65676c43fd7848792cce2eae6ecca5a0` |
| `drivejepa_018bp06_c438a779` | 65 | 166996926 | `2c3a7b6674decd842279dee77e8442befa6f126f2b3e113b7ff5843a727f7c4c` |
| `drivejepa_018bp06_dd0f84df` | 66 | 267078892 | `55820c5a3b6c3db5414b76343a37c987cc8986f34e7895f42ab4ea58df940212` |
| `drivejepa_018bp06_e883822a` | 63 | 173450466 | `9cc71ad2cf9b0b639473163ca07657454f2cc56a0f979ae833724e651c522ed2` |
| `drivejepa_018bp06_f005ac5f` | 65 | 189024128 | `7540941dfdeb2c100290245159c69fe0c47f016f3889e99eb64d72abcadbc6ec` |
| `drivejepa_018bp06_f3a87e7a` | 66 | 162465690 | `cd681f3ca303d886fb0081315c06aef81a31777d3b46d243b5bada5498acf016` |
| `drivejepa_018bp06_f890fe10` | 63 | 193121151 | `ccb159dc9524969101d8887ac999c6f4bc288e06c25dea9c608ec7c5b14ff5be` |

### Step 12, image check after the rebuild of 20 Nov

Where they sit: `{local_store}/runs/new40/imagecheck/018bp06/runs/`, `{local_store}/runs/new40/imagecheck/runs/`

| Folder | Files | Bytes | sha256 |
|---|---:|---:|---|
| `018bp06/runs/drivejepa_018bp06_04749bb9` | 67 | 98680659 | `7291ca26e6a4955df162a3bf3bb2f0cd4d8cba97b1d0509c1343093c3f01e925` |
| `018bp06/runs/drivejepa_018bp06_8023dce5` | 66 | 150033813 | `eaaf84bbcd91ec500edbd99d643400f31af00555443abe646702b2701e073698` |
| `018bp06/runs/drivejepa_018bp06_d2715023` | 66 | 241106252 | `041c45c0a79f8a0aa2bd17fbb4b7d1416639241810ae5873f1b66cf8d53f8a4c` |
| `018bp06/runs/drivejepa_018bp06_e3ed554f` | 63 | 237057011 | `d06caa685dc1393988d7727e9f92805de39823d8cc05d2793443d416068c10d0` |
| `018bp06/runs/drivejepa_018bp06_e3ed554f.killed_watchdog_20261120T0042` | 32 | 1194172 | `89fd2c83bc4985becf3e8c7507716ae35778a0ca94d7a91511c42ee7b41798ee` |
| `runs/memory_carry_0245ff75` | 68 | 201600064 | `899a8e6e377f0dec33a93bea710ae33c79240d1b6d75f994c69dde6342542d09` |
| `runs/memory_carry_bae463d3` | 63 | 174698299 | `f1a83955086d220f6200b97a45c5205df59874f3f07536b0a886df7109d3606b` |

## Closed-loop tables, arrays and the replacement scene left out

| Item | Where it sits | Bytes | sha256 |
|---|---|---:|---|
| Plans unpinned, for the 2-core check (step 10, `out/cl18_thread_all.npz`) | `{working_folder}/out/cl18_thread_all.npz` | 308768 | `f31f152bba8745207e51e54b0c9bb7058202544c0df7b4deb33bbbe89133ed26` |
| Plans pinned to 2 cores (step 10, `out/cl18_thread_two.npz`) | `{working_folder}/out/cl18_thread_two.npz` | 308768 | `9cc10f5adbdb2b9cf0ba4c802acfd4b9d104d5d77b77161545ee202e0a1ef15a` |
| The withdrawn re-score, per drive (step 11, `out/fulldrive_rescore.json`) | `{working_folder}/out/fulldrive_rescore.json` | 101108 | `7cb187272569eb12dc38bbbaee6168192e3d0c65bf7e6e474946188f6ed05135` |
| The replacement scene 44409f00, as downloaded; equal to the dataset repository's record (`tables/scene_draw/replacement_download.json`) | `{local_store}/new40_scenes/usdz/6ffea280-aabf-4695-903b-37dabe37ca73.usdz` | 1871395562 | `8f92e1c94a3b4ccb9ea07f22758ec6c8172e5b667da47acd98e5e1727289bec1` |

## The simulator images

Docker image ids on the training site. They are not files; the id is the sha256 of the image's configuration.

| Image | Used by | Id |
|---|---|---|
| `alpasim-base:0.134.0`, the build of 31 Oct 2026 | step 09's 12 drives of the starting model (made by its own lane); deleted from the machine before 3 Nov | not recorded |
| `alpasim-base:0.134.0`, built 12 Nov 2026 | step 09 (the memory drives and the image check) and step 10; deleted from the machine before 17 Nov | `sha256:9c5909dafcd5574dca1c783e6abc29dfbdfcc9fc2860725dd80030475b8f8881` |
| `alpasim-base:0.134.0`, rebuilt 20 Nov 2026 | step 12 | `sha256:ab16bc78a4686dcd63c2b77a77fe585df3846528a61b552db6b71f9553983a44` |
| `nvcr.io/nvidia/nre/nre-ga:26.04`, the renderer | all steps | `sha256:97f43e7130c5636ce3e80ea3184d97f56a87fdd989b05cce42230881dbdea284` |

## Closed-loop logs

In the working folder's `logs/` at the training site. The reports cite them as `logs/<name>`.

| File | Bytes | sha256 |
|---|---:|---|
| `cl12_check_only_carry.log` | 2111 | `5408fef6cc29a5fa6a24cd4933c6c698a37c2ed1df4955a3519bed1efc6d2b4e` |
| `cl12_check_only_carry_worker.log` | 4198 | `822391604d7ca5434cfed78617a7904222d020aeaa186708412f64cb5a4b234e` |
| `cl12_check_only_reset.log` | 2111 | `bd4a15371a5c2eb7e69f4d263d2fd208a909940736083f0273767611fa56ce30` |
| `cl12_check_only_reset_worker.log` | 4198 | `2d362089062edbfe7dd32cce43d1a39b681b697a05f2c537954cea53a7df33db` |
| `cl12_imagecheck_cmp.log` | 783 | `c58db0481fd0fcaa4e3ad08bd5697c0293c56c79fc6f1f8d94a4b7e0d2af9f28` |
| `cl12_proof_Ac.log` | 229 | `fd7a30ff14276ae51a4db2378f538401c163b6df5d845760f1ab7f3ab462e00c` |
| `cl12_proof_Ac_p06a_worker.log` | 3666 | `c5a3761965634b5f3235db74769598f45621cde3f9a3d247a814af43648fe4c4` |
| `cl12_proof_Ac_p06b_worker.log` | 3666 | `c5a3761965634b5f3235db74769598f45621cde3f9a3d247a814af43648fe4c4` |
| `cl12_proof_A_first_run.log` | 1170 | `b4c9497848769a5dfb52467a07e59d98cdb8f09a9ef5c235640c1b696c694c0e` |
| `cl12_proof_A.log` | 1232 | `2cbf43645025038a4b0955152574bdff228e9ab35173754fd6b6d390f2a22373` |
| `cl12_proof_A_memory_worker.log` | 17197 | `9f6b5a523384a5782153cce8663f4997b89697ec98c4fa97956a22be39ce69aa` |
| `cl12_proof_A_p06_worker.log` | 14664 | `2bc4c5e5294bbfa2a186828292372fc6146b2b93830978cdf648722859fbf8b6` |
| `cl12_proof_A_shadow_off.log` | 1670 | `5fda269697c9073abd6131a9268d6759cc8efb1f65d7cf7018918c8acb1b31d9` |
| `cl12_proof_B_carry_worker.log` | 8554 | `97bfd0da8434c01f66c93400187af06fb5da0e7fccbd54369a43b09cffaa4562` |
| `cl12_proof_B.log` | 1024 | `b26eb0f335ce786c454f5aeb36b2317e8ea3169127664ff0e6b576b6fbe700c2` |
| `cl12_proof_B_reset_worker.log` | 8554 | `f533925c2fcb9f16bced7ccfef39cac18c22b131c37cd56f316274605f3a6930` |
| `cl12_proof_build.log` | 449 | `7b358dbac840305bd0c6d9b0a4ab051ded37a01303db736c89a68ea4e766f2ec` |
| `cl12_proof_C.log` | 1184 | `aba9e011edeef73092004e4584d15d361af4e911491cbfb81a263efdb976367c` |
| `cl12_proof_C_worker.log` | 9640 | `1e6c9f1108648dd4e0a38bf1d11f2b552c5fca3c57cfbe82dd5210f941c9e9ce` |
| `cl12_proofs_chain.log` | 2257 | `cf2ceb9e3b3fbc585c57a2a7f0edab7dd257937823ce865cd896db61dedf5e5f` |
| `cl12_queue.log` | 14339916 | `11362c893799c4ee2de581f1a4fb4fd35cf75fd3abe176485c83f37e9bd81175` |
| `cl12_table.log` | 9928 | `744e2b5079cd728a9d3beaa6c51332607786530b016c8ca194147924de5bf498` |
| `cl18_queue.log` | 17565036 | `a4992f9e0fcb9c910e93bba96769bb50b01bafdbdcae62eee71f59bba2e3d67a` |
| `cl18_relaunch.log` | 234 | `0ec01f1e7c8b8c08a30bd21a4cec0325b5505acc07618fe1b7f23b3f93a0ec4e` |
| `cl18_table.log` | 7412 | `181f0e7dde9554712a096780489de2a69eb7cb297dd9b12dfa69fa5e5c6c7c37` |
| `cl18_thread_all_carry_worker.log` | 4277 | `feadbaccaef5c3b6318a70256e71877b433811e52ce96f4452c6e4ff6728191a` |
| `cl18_thread_all.log` | 63 | `443e531ef35ce208a7a09ed24fdcb4de254df6ae5744a41a218c5439f74ae6dc` |
| `cl18_thread_all_p06_worker.log` | 3666 | `c5a3761965634b5f3235db74769598f45621cde3f9a3d247a814af43648fe4c4` |
| `cl18_thread_compare.log` | 604 | `50ebabf438017bc8dcdb1ac1f6a5448c3121fca0648b94688621eefa4b220a88` |
| `cl18_thread_two_carry_worker.log` | 4277 | `feadbaccaef5c3b6318a70256e71877b433811e52ce96f4452c6e4ff6728191a` |
| `cl18_thread_two.log` | 62 | `81ae350b024da79b073f2dd742b205deffd0ac23248918c5fcbf2c829a141287` |
| `cl18_thread_two_p06_worker.log` | 3666 | `c5a3761965634b5f3235db74769598f45621cde3f9a3d247a814af43648fe4c4` |
| `cl20_compare_page.log` | 5777 | `4632d4809f7c6f15bd0b7601aba0ccdbebda2ed52d1dedb7179f76c5e0633747` |
| `cl20_imagecheck_cmp.log` | 1179 | `c4e67dddf3a6ce1d20fc4af05c9dadc9d6d65a3533c43e804aa8cb071b646cdc` |
| `cl20_imagecheck_queue.log` | 3470325 | `dd231dfae2e14566917cbd1cb35040f3619ffcbd86cd7a87f123847b608daa99` |
| `cl20_queue.log` | 20462172 | `c2c3433f23b07e2fd4a4849008cfb3ed548336060d70cf67070d49a01487a38d` |
| `cl20_table.log` | 3206 | `e16176d1dca5306ece5b154560ff9b5c097b4296b13f6ac723a27431821a68ba` |
| `fulldrive_017.log` | 3065 | `f1a5aecf9c943448ad86c666f8cab0201e6a183dacbf7a3b3532116ea62e97a0` |
| `fulldrive_all.log` | 10963 | `c6a78fd0668b7410e76a35ab764df725c15fddffed5a6b9b75595436e670f291` |
| `fulldrive_prereg.sha256` | 370 | `bcde59e393b75630506808e16cb2ff5e25fd87e0e7d1bb86b23402a7e63258fe` |
| `new40_compare_page.log` | 6341 | `e59f718910cfaa2aeb26463817165064f538cbaa792a2f82ee19eac0305d507d` |
| `new40_dryrun_018bp06.log` | 2323 | `ea087dab95cc8d2d4a7a316b9c2c64b442642040847e9a3f50613c14111d1aae` |
| `new40_dryrun_carry.log` | 2590 | `8092259d15aed408faada74f79cfcd928107fbb4e081043a3b88f5f55aa3e899` |
| `s13_compare_page_cl18_final.log` | 5114 | `f25eddf6e940aebb0245becda352cdc26f03ec636e735fbb7fbd6b1cdceed31b` |
| `s13_compare_page_cl18.log` | 5310 | `024a53a6722fc254975ec7a6cf41f64cc252182c97d5a5a1718e343a1ab7325f` |
| `s13_compare_page.log` | 3067 | `78dde9e81b863b0131c7a763b04c2b5d5c12b33ebc10d689ef9bbc65c79fa245` |
| `s13_compare_page_rebuild.log` | 2923 | `7faec6bdf7e55c6d50a2db7d5b13933a6b5919b618aecb9e32b851f8e1ee3ae0` |
| `s13_page_check.log` | 10956 | `7baa9b10df90ba8c443cc15e0d3c6f5589e2b90c3f53ad16c57f1d6fc50839e0` |
| `cl12_proofs_before_freeze/cl12_proof_A.log` | 1251 | `309a2f20c92af31e5734d6698d385415bdb35a2f74ba515aa86f354281c605cc` |
| `cl12_proofs_before_freeze/cl12_proof_B.log` | 1024 | `9cabbeb8a2628c314a8a67736f840e6b8459d4da50d7f1728e21ece98a0f987b` |
| `cl12_proofs_before_freeze/cl12_proof_C.log` | 1184 | `aba9e011edeef73092004e4584d15d361af4e911491cbfb81a263efdb976367c` |

The comparison page that shows every drive side by side, with its rendered videos, is served from
`{local_store}/compare_page/` (built by `s20_compare_page.py`); it is a view of the drive folders above, not evidence of its own.
