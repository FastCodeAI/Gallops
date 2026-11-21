# Pointers: what this experiment produced that is not in the repository

Nothing below is committed: checkpoints and arrays by rule (`.gitignore`, `docs/checkpoints.md`), the token copy and
the history side file because they are derived from NVIDIA Physical AI AV, logs by convention, the per-tick drive
metrics because they are binary, and the trainer's run files for size (1.07 MB each). Each row gives a location, a
size in bytes and a sha256, so a copy can be recognised wherever it ends up.

**Two kinds of location.** A *template location* starts with `{weights}`, which is `[roots.weights]` in
`gallops.toml`: where a copy belongs in this repository's layout. Nothing has been moved there yet. *Where it sits*
is a label, not a variable `gallops.config` expands: `{approach4_store}` is the training site's bulk data disk folder
for this experiment, `{staging}` a folder on its local disk (the network store was read-only when the history was
built), `{retrain_store}` the store of the earlier retrain, `{experiment_store_h17}` H17's store. The README's
placeholder table lists every label. Hashes were computed on 20 and 21 Nov 2026 from the files where they sit.

## The starting model

Drive-JEPA 0.1.8b, pass 0.6, as in H17's POINTERS.md.

| Item | Where it sits | Bytes | sha256 |
|---|---|---:|---|
| Starting model, merged checkpoint | `{retrain_store}/018b/release/drive_jepa_018b_p06.ckpt` | 1273003357 | `3d566eb211b5ce0f556680196a46fc0a74da29091ae62ac588910a48d62b8e66` |

## Checkpoints

Trainer format: the generator, the scorer, the motion encoder, the "no history" vector and the two links, without
the frozen encoder. No `checkpoint.json` sidecar was written, because none of these was a candidate for paddock.
The file names give the pass (`p00.2` is pass 0.2); checks are every 930 steps. `resume/last.pt` is the trainer's
last timed save. The two step-0 files differ in bytes (each copy wrote its own; what differs was not examined); both reproduce the
starting model exactly (`04_model_and_trainer/tables/step0_check.json`).

### H19.r1, history shown

Template location `{weights}/drivejepa_physicalai_motion_history/h19-r1-history-shown/step_<n>/`. Where it sits:
`{approach4_store}/runs/arm_A/`. The chosen check is `ckpt/p00.2.pt`, which the record pins. `ckpt/p00.0.pt` drove the wiring proof.

| File | Bytes | sha256 |
|---|---:|---|
| `ckpt/p00.0.pt` | 30021317 | `d2ff692cfbad8bd94d2f7ff4f5725fc4768daca1a4c665294910682ab73dcdc5` |
| `ckpt/p00.2.pt` | 90106758 | `d1a40752613e0dad4e678bc44cba904dc314fc7b57ebfd5dcd79c592450b4f4e` |
| `ckpt/p00.4.pt` | 90185094 | `f5871b545327eeb91acc513635a232567d32deeb3c54fc2d9f7456c4a150807c` |
| `ckpt/p00.6.pt` | 90263494 | `00b445f28e2839194eab71b2b57023f06cf2def73e5fe69aa07f45bb79dcf254` |
| `ckpt/p00.8.pt` | 90341766 | `28ddc9e89ceaf68748bace6555409ba3be74a3229c51d8d3ce5dc3a54b0c0e7f` |
| `ckpt/p01.0.pt` | 90420230 | `92ac68b9cc1d7cb6c8994df3fb422abb0126263dad9b8dfa59778b02c7cc0c6e` |
| `ckpt/p01.2.pt` | 90498502 | `96909357ef6f788413bbe899215da2da038402516684d3854a679e1db87c0fea` |
| `resume/last.pt` | 90466113 | `18162c9d17e9f5fead55c8831bcc101490449cb5712638667b1061663ea36868` |
| `run.json` | 1065094 | `4061f12eceecfb276224fcf1d46ae3044a7e50ffc00218e47007779b845997dd` |

### H19.r2, history hidden

Template location `{weights}/drivejepa_physicalai_motion_history/h19-r2-history-hidden/step_<n>/`. Where it sits:
`{approach4_store}/runs/arm_B/`. The chosen check is `ckpt/p00.6.pt`, which the record pins.

| File | Bytes | sha256 |
|---|---:|---|
| `ckpt/p00.0.pt` | 30021317 | `d74776b63624c00e026ed27a381b09e5f3b05fde7b90ecd5a9ba4c39decb831d` |
| `ckpt/p00.2.pt` | 90106758 | `59d199936727e7c6f734b7e6b7f64f8ff89315f564645660c36e3ae7f7bde72b` |
| `ckpt/p00.4.pt` | 90185094 | `41c62d6e8f22793d518d65d61a9d7d644b055acc617a487556e60ac91865b04e` |
| `ckpt/p00.6.pt` | 90263494 | `35c83bf67e89b4579cb3d8f77fc45862da2f0b2070caecc1f5193caed940dd9a` |
| `ckpt/p00.8.pt` | 90343750 | `bbd41f9e86a58ec5a4810e8bf82ac3968f5eb69ce7ede42e71be22668d3f505c` |
| `ckpt/p01.0.pt` | 90422150 | `ca6918f6da8d33bbc4b9a7f003a58b81018c75c3e81e1e575a1cfe3ff6ed3f1c` |
| `ckpt/p01.2.pt` | 90500550 | `720a87b0a43153ccfff4be0f09fdb3febb8b27dcef3a8bffb97381508854220a` |
| `resume/last.pt` | 90464833 | `a48e1fb12336faec2a338029f85a197372830d8b73fd333d10900320cc3a2d74` |
| `run.json` | 1064652 | `1b97ee2c18e52d2e2f8fefddfeff58df8e912d9201cc2dfdac41593dc253c079` |

## The trainer's run files and per-check dumps

`run.json` is each copy's own record: every check's numbers, the selection, the losses and the random states. It
is the source of the offline results; the step report's tables were written from it by `report_arm.py`.
`dumps/` holds each check's candidate plans (`.npz`) and their grades (`_grades.npy`); `logs/` the trainer's log.

| File | Where it sits | Bytes | sha256 |
|---|---|---:|---|
| `arm_A/dumps/p00.0_grades.npy` | `{approach4_store}/runs/` | 1303040 | `59181e46f58ebc65ca2387228e16d8b88e199f413b8049263e7efb3ef1d19d24` |
| `arm_A/dumps/p00.0.npz` | `{approach4_store}/runs/` | 7211166 | `1f67e2bbecdaaa659c80584d630ff7f36cf8ca64a1f6b99b4aa276b47be15a06` |
| `arm_A/dumps/p00.2_grades.npy` | `{approach4_store}/runs/` | 1303040 | `6290da94d11d79318906af3a3419f76ea350c94f7fa1d4e4ee4e174c4ff28b53` |
| `arm_A/dumps/p00.2.npz` | `{approach4_store}/runs/` | 7206957 | `a7051a617943cdcc03abdbf36fe18c64eb32058e40dbb7da0d4ad3036d618fe6` |
| `arm_A/dumps/p00.4_grades.npy` | `{approach4_store}/runs/` | 1303040 | `a0d5cb2b327daef697f6e520dc02e4304135ec3ca5aefcf695c99f6689713af9` |
| `arm_A/dumps/p00.4.npz` | `{approach4_store}/runs/` | 7212859 | `ebab707ca0ff961934948fab1dce583309ab0d8538448e6d1ea3489f3ba64c37` |
| `arm_A/dumps/p00.6_grades.npy` | `{approach4_store}/runs/` | 1303040 | `57bf932df74674c492c2431cb445f74f7e24043e89623914eaffcc9cd839c5c9` |
| `arm_A/dumps/p00.6.npz` | `{approach4_store}/runs/` | 7218314 | `b2e3f34e49363f2f26eb8efb89631a46e3ba2e2cff4b7c5f987775750d4a3d35` |
| `arm_A/dumps/p00.8_grades.npy` | `{approach4_store}/runs/` | 1303040 | `8b1a538507014d853eb8a44ffee1a1346c1cb0aeb04ef83ab3659134c2527194` |
| `arm_A/dumps/p00.8.npz` | `{approach4_store}/runs/` | 7211808 | `b9e108bccd75c3ce55aededa31db79ff40af236e7666c120901b5829cee974ad` |
| `arm_A/dumps/p01.0_grades.npy` | `{approach4_store}/runs/` | 1303040 | `21d2a2001d25c1ff749c4bec74a61b099d06b9377d5c015d4061d33e5cf55423` |
| `arm_A/dumps/p01.0.npz` | `{approach4_store}/runs/` | 7206507 | `09cdec1a5ec878de4486de0d7af11bc2f025f20300854ce28c3ec49b33c8d18b` |
| `arm_A/dumps/p01.2_grades.npy` | `{approach4_store}/runs/` | 1303040 | `cb504445c30145738e796d422c595df8e7ed8036c41dee49850b7306684c7ba4` |
| `arm_A/dumps/p01.2.npz` | `{approach4_store}/runs/` | 7214057 | `db8061f0a9bf21233439a3aac445771eff0b3aceb2f61e1f6766fd59d595de14` |
| `arm_A/logs/train_a4_A.log` | `{approach4_store}/runs/` | 46892 | `2711f525a755bcefc7c542d6d9286e2291dae7cb8168abbaae1a91f723e607f9` |
| `arm_B/dumps/p00.0_grades.npy` | `{approach4_store}/runs/` | 1303040 | `59181e46f58ebc65ca2387228e16d8b88e199f413b8049263e7efb3ef1d19d24` |
| `arm_B/dumps/p00.0.npz` | `{approach4_store}/runs/` | 7211166 | `1f67e2bbecdaaa659c80584d630ff7f36cf8ca64a1f6b99b4aa276b47be15a06` |
| `arm_B/dumps/p00.2_grades.npy` | `{approach4_store}/runs/` | 1303040 | `46fdb9d652ac00477bb4389b7a26784d6cef854b27b47608d47fc899e0d2449c` |
| `arm_B/dumps/p00.2.npz` | `{approach4_store}/runs/` | 7203588 | `4079f6626789cf9bb7b4f8fed5963fe24cb5c701fc671eed99894fb2c2ca2168` |
| `arm_B/dumps/p00.4_grades.npy` | `{approach4_store}/runs/` | 1303040 | `0a175d75fff06abf45133f5fc5955100cdb955a9421b862b9576c03f53df4495` |
| `arm_B/dumps/p00.4.npz` | `{approach4_store}/runs/` | 7210398 | `01cb703dd2480e0f951a147b8cc57bf77b98c62138f1e3f7578f62066cc0c0e7` |
| `arm_B/dumps/p00.6_grades.npy` | `{approach4_store}/runs/` | 1303040 | `f6f7834a89d900db2f5fff2cca725f8d1c75e01d44eef2d9ff3e43863e720521` |
| `arm_B/dumps/p00.6.npz` | `{approach4_store}/runs/` | 7211502 | `274ab8d4cf0390b0b3231fda7206eab506d953490ca61f7e509ab820af9302f0` |
| `arm_B/dumps/p00.8_grades.npy` | `{approach4_store}/runs/` | 1303040 | `1db64931ae2c28b0066721da31553c8a78da83668d02b2fd4b9edc58e4908c6e` |
| `arm_B/dumps/p00.8.npz` | `{approach4_store}/runs/` | 7212281 | `b1e7d832cb23e77f8b8d5ef51e09da630b32a6dbfca0a2ef054bc9c82df31548` |
| `arm_B/dumps/p01.0_grades.npy` | `{approach4_store}/runs/` | 1303040 | `bb784b1328c5d6b171e66bfc2329a794f07e933583436976b28b6083a1ae7d5d` |
| `arm_B/dumps/p01.0.npz` | `{approach4_store}/runs/` | 7211638 | `bffd70c0a57f98e333bb1470a0ddb5caa0f503fe97d8e3e4002355a41bf44b4c` |
| `arm_B/dumps/p01.2_grades.npy` | `{approach4_store}/runs/` | 1303040 | `7021443678ac543d19184a3b88c663a71edb40249e713a3ba990a0621797627a` |
| `arm_B/dumps/p01.2.npz` | `{approach4_store}/runs/` | 7214898 | `8b20ba0eb77a098c01fce9c8b4c159e8e3344930ef34e1e76b953cdbe53799e9` |
| `arm_B/logs/train_a4_B.log` | `{approach4_store}/runs/` | 51104 | `0b0e7f41caa0cb75ca9bffc9d3c2574279d57bc52175403f531eeb8163526098` |
| `arm_A_stdout.log` | `{approach4_store}/runs/` | 74419 | `9b2ee1b601b7a41c2b22bee0efdfcd72e5a3e14262df7a7f267b50f647a8e89a` |
| `arm_B_stdout.log` | `{approach4_store}/runs/` | 136604 | `bb0c91711e0be3d882af5fbeef8df42e93effd80bcbb0c30441f3c88202955ae` |
| `queue.log` | `{approach4_store}/runs/` | 4634 | `038c3f341a55c521d1b92cbeb18fb5b471abd45d8a0ecbad381bea50dbea8087` |

Also kept, not evidence for any result, sizes only: `arm_A_aborted_statusbug/` (the first start, stopped at check 0
by a bug in the status line; 38,574,300 bytes), `resume_test/` and `resume_test_1_nondeterministic/` (the resume
test after and before the deterministic backward; 270,077,317 bytes each).

## The history side files

Built by `03_history_data/build_side_file.py` and `build_picking_history.py` with `ego_history.py`. A copy is the
same side file only if every file's bytes are identical. The listing digest is the sha256 of the output of
`ls | sort | xargs sha256sum` run inside the folder (the same as `sha256sum *.npz` in the C locale).

| Item | Where it sits | Files | Bytes | sha256 |
|---|---|---:|---:|---|
| History, training slice (126,981 moments) | `{staging}/ego_history/` | 4015 | 35582626 | listing digest `98c6a08eee1e5b0c4b33916dc792fcaab6cfae6ab837b880e5d55b9df1b371e6` |
| History, picking set | `{staging}/ego_history_picking/` | 84 | 576344 | listing digest `8ba10f5970365d0bb19340dd7d19fd760204ddd9878341e656d15fe9f6a8d6a1` (files below) |
| History diagnostics, one parquet per clip | `{staging}/diag/` | 4015 | 27896127 | listing digest `ed7e262b37a95f7565fdefbabcf0bf15bf8d18ecb6581b7310c24276bd7f3f5b` |

The two 4,015-line listings are kept beside the data, not here, for size. The picking set's 84 files:

| File | sha256 |
|---|---|
| `0238dfe4-6545-46be-933d-8eb2ba29358c.npz` | `11280ee034ccea3c2ee8317348e2d9a4dcfc0ed95c70de65f5d67d156bddefc0` |
| `02ce6aee-45fb-4d4d-921c-4836f9a64be0.npz` | `10bf0d2519a5361958467f07430a522367ebed7817020a0da29dcd6b099d6083` |
| `0418c913-fead-437c-a856-bc397fc606cb.npz` | `3d73aed6daa27b1cae763fb83e2365558b635e7a3036d2bf5c1cb989b93b5174` |
| `05cc65a8-13f4-4e6c-b98d-0cd442536bfd.npz` | `f3881bd8b9751938651cfedaf1f04724f43cb88f64da706fa32c8887fe7e3a94` |
| `0ec477bb-58d9-4868-8441-235a1d2df0dd.npz` | `fd9c622c47ad6a8bf2c6c84c65bdab6ee1b098d2c7d01566a2241b12667827d7` |
| `0ed7cd93-8375-4912-90a0-3e83293b33f7.npz` | `0677ff32c58e13970169a621fa778d3fdc653b697e583cbe2b24018ca19a758e` |
| `101ea55e-281a-4b81-a3b4-6476013e74bf.npz` | `98fb5f8780a5cb198eff8dd94972a1b2b25c4e3b18560ca7fa09ec7196f46cf3` |
| `10e5a754-e4ba-4861-9ec8-b87a5d8e9b4a.npz` | `d649af7461b372af5ccfb5e49e5735baa9389b48bcca8c84e31a73a027767d07` |
| `10e91619-c0e2-4e87-abbb-40791a7ecb6b.npz` | `f7d06146f605dbe68b59caf69b0aabd1c3a09b026155e85224daa0454a21387d` |
| `11af607b-ca54-4826-aa5d-43aa0d03f297.npz` | `fd34e42f53376acc25c00c7db6860087b0e9598c9b68efa819d93bfc7dbc6076` |
| `17b15bc8-fca3-4bd9-a6ff-0357421e90f1.npz` | `fc7aabb9f6aa7a3610ed130ede118b3be45a30d2cc67b6369d4343dc3b630e46` |
| `17ca33eb-7b09-4f93-93b3-9583ab693228.npz` | `135c7c7933ed0c6df15e3803afff7d9c7e07b67c2ee6816af6fe91f390fb3fee` |
| `19585c5c-d523-49f4-bf15-59291e6c9278.npz` | `fce890e6464363137c4b54cd461c10e05b1e2dd2f93c083777e6259f1cf60892` |
| `1c9774e9-02ec-4ae4-a404-c057c9aa88fe.npz` | `91bb937153143de2df2129b7d4703518af37c2d5f01b59145b650d58a01b80f1` |
| `22fa5c92-6e6f-4f4e-bbfa-9a1490f9cd38.npz` | `75173b6c115941dff99434a576a9dfd48437de41968551f42b9d796471bd9e4d` |
| `24c76d62-f44c-497e-968c-320c09295d16.npz` | `04867f14c8b2fcda5174f49b7a8bf825e60ee1b258de2273e480366fdc1a896e` |
| `2ab960cf-8ded-4331-85e1-49b61b635dc1.npz` | `24fcdaa316c8d413d2796f8a05f86e5141dc0665f364b92bea38bc90fc9e6132` |
| `2b3f2239-3b12-435e-8c9e-623151c5fc04.npz` | `4052cb702b8b662668aa7b5ad3edbf5725650ecc94b9e10f662bbf7a31058a3f` |
| `2cdb2ef7-bd15-47e4-aea6-43ccf1dbc686.npz` | `c8eddcbcefadfd2c07889f286dddfc985d528b2eb63857eeebfbedb00fc54627` |
| `2deb1c4b-2a76-4c0d-a79e-3b4b7105bd27.npz` | `cd5009feef4fa6305dfcef27f5e16a17414aedb73dcfa3354dc62bd92c815cfc` |
| `2e7644c3-3e3d-4f4e-a350-090c284b06d2.npz` | `dc46b38dba6c8d8d111abb0ac55d07701ba7aabf372f9d85d04d9213b02337e7` |
| `2eaf1872-7780-43df-bdf1-9f48975e6436.npz` | `92612b9a72eda6dfd420f4c35e17528066cde9d37c8c6ea85bb8ab32b7c6f387` |
| `319cdf0b-0ce0-4437-9eae-903a0c129c12.npz` | `72150e0192969e8b631fbbc4ce288296f1669b05ba53aa33c92615536928ccfc` |
| `34fa9b56-fa47-4a86-a7df-f26846b240e0.npz` | `77dc7388def2c2dbe1bf4a3a3df7d77d3dc13cb0945a51ea04a06bcc9dc964f7` |
| `37d060d6-1628-434c-8e32-97faf9cdcfe2.npz` | `a85050ffc63ad751106a8a0b7d3bb332854f84efafe4abc163b5b7a416fc3dd9` |
| `39c63dab-8634-4a0e-9f56-d98f733dbbc8.npz` | `a597e413d79ef049dbcd63e8be2d9b50cbb32504a591ea4e890d00116997d80b` |
| `3cc99a13-8bc2-493e-995b-0097f8efa0a0.npz` | `efafc1430b71bc8eb0c42401b68bce94c739f718b244f6b20a03f2e89d9166e2` |
| `3f3a65b1-5534-4b69-8d2a-f5f441f209ac.npz` | `f2c17f66efc0b567ee45e95f864bea4b3a40803cb5360a0edb283c1c9b9bb3ba` |
| `4086cd81-2226-4e9d-bb5d-8ea0eceaebf3.npz` | `b07018bbcd81ee2936e3b4dbb6f1411c2179e2152fe774ef9a753672813c679c` |
| `4132330a-e7ee-4a0a-a88c-223871d0c442.npz` | `b20f4eaebe91f9f42f3050f49650e2b69b8a3944ae6c972f02faeb3b5a2a041f` |
| `435719b4-ba2f-433c-8490-1f2d8ed05f0d.npz` | `4ed2c89225301e02a29756e8d0687b5c17ae1dbad0fd0c77232ad08f439fce12` |
| `48dc7e8a-14ab-4bc3-8aac-dbb20cba057c.npz` | `c5e3a0dbc418e6d0aba26c7d677f61f06ac836ab955059aa84d47a04ee553282` |
| `4a4f4f1f-43b1-4324-a34b-0079394fa729.npz` | `91594a92d064638ab42c58c66fd538192c52158a0eee3362a7688ca76cc386b2` |
| `504ee5fd-7588-4ec5-b8f3-2bde1b06c05a.npz` | `e19a6143b179f89c3dcaabc10a18b59d353a39852cb9f9414e0d3b3d75dbf740` |
| `519a1bf9-b9e2-405a-a69c-f2026f4e1058.npz` | `cd34fde83c6a56989feb3c44a85d97024c295a5ef51dd4d4cf6bda46c4ff2edf` |
| `5ab16901-2bc2-4004-a0b2-0bb1071e96cf.npz` | `5a169a2ccde94290fdcfdbca53254b7ad8e6e1b9895ae206e5b040e951330c6e` |
| `5eea18db-eb06-4daa-af77-b7d09bcd1037.npz` | `e791322c6f4b64029eeb8e91ecf3b09932bda3730c3aab0ba79f2a0191cdc32a` |
| `630a2f71-b5de-4661-be26-9592bbe8f727.npz` | `2c0b8d8e554ed84353ce91752a131a24620d7d2f911a4d4ced971aa18ae78c55` |
| `65c66b19-c723-4d6a-a3f1-06eeb60f525c.npz` | `f4f65250ac630b13127ee410e253eb83425d6bba67a83fa964d11596f73d53df` |
| `65e82a93-d72b-468a-af09-af81531106bb.npz` | `034c9189510c0ce0b80885e2bdaedc4021121d939ff4fa6824a73d97a9b9f6ed` |
| `6819efbc-d0f5-4498-a997-1ca0355c87fd.npz` | `d171740ef91f9eb78bd1cfbaedd16d934f7eafd0c934004a683fcaaddbb20332` |
| `6ac28ce5-ca21-4610-9f25-ddba46d91d51.npz` | `3ad79e814b1b6d69f31c6292ef3351d08e7c0521e6ae26133b731de20e3d79ed` |
| `6b43ab3f-0462-428a-8dee-ac2a41d7ad67.npz` | `bb83fa00ec3946ba72814bd82c3149375cec44c569b4c0ee169c60c11d6a893c` |
| `6c9a7235-eb49-4061-abb6-9581269cb940.npz` | `63c1863b66e6b43bb6e703313d5707221adb482e24bddfed7f9bcb5846854ee4` |
| `71a090b3-d8e3-4d6f-8acf-f888bc1aad48.npz` | `f2fb7c779155ba300b193b07908b456112dd222d8043097753b6517a43e1b960` |
| `7308521a-9425-422b-a240-47babc87215e.npz` | `3102c77d31e3476ebd9fd4ab4a46552806f9a87766ec1ce8b2feba69a5bdf8c6` |
| `7c31c021-4616-49db-8baa-e9128a25933e.npz` | `f5358cf90ca4421c928ab630ce816f5afdd76e8b0d4b31f09bffaf49a4684342` |
| `7fb3b33e-85d4-4d3e-b999-5eee301c2308.npz` | `faf19deab91c1b077dc0981a749b49f9ff424ab1b242cbb1535e69f4b99932c7` |
| `85137ffb-08fd-4a4d-bd92-c3a36ebc083e.npz` | `0b18572b0cf9adc9d3d089c35330ffd3c44d210d69682abdb7e54b8ec71cca74` |
| `864ec5ea-9eaa-4d80-a846-8bf5c982f140.npz` | `8288b991b471a1d98b34648a6613851f7699d179d47b4cfe20da80ae6bc0ffdb` |
| `8995a9ff-c92b-4c46-96c1-d94aba91d299.npz` | `98ae142abc5959be9c40071289437ccb1cacc43ea5cf168087cd275b14bc9d7e` |
| `8b2e569e-8d01-4c53-9878-3b45fff48c72.npz` | `a7ecf0ba96144371cf513c49e0e006557e2e16df0338ed9313d034e080c2669d` |
| `8f3902f9-864d-4acb-9907-9a6da8b242df.npz` | `c41d3ef9bc25cb5230145d3e00736e3fbc6fbb0a589fe1ed27850b936c796b3b` |
| `91194ec4-75de-4418-9e49-0d706d84c7a7.npz` | `8bc5a01a43e67a052a6837becca9b5b3c82c060ac2c455f373bbebc7fac55986` |
| `92263c6a-d890-4b9e-a40d-c44d553ab8f0.npz` | `60fc615b57cdf6ec5753314e19cd06a527eb55ace572e3277236c7c80f9acf20` |
| `93f81daa-813a-4417-a0af-c86ddb820ed4.npz` | `33597a649e0326c2081d9c7aa3f8958db05fe3febed0b60aee28f4c97aeb7018` |
| `95acae18-ec63-47c0-9830-c79ebe2ea2bf.npz` | `f91ff1c0fd3796dd4ee13d41378ef2be5ac9326275f773ecd97e5a5866c89660` |
| `98694f91-f6a4-44f3-aa7c-3095c91e4a78.npz` | `424f40b8557b63723649365679d61f1a44b72f66f9bad9ceb2da9c78147afb07` |
| `9bab530f-065f-48a5-9e8d-0ef420818c02.npz` | `7a859d8ebf346712e50424d6c77102eccd16b88b5345d8270035846b68057d9c` |
| `9c8e6289-e412-4457-849f-0890a6e1f64f.npz` | `db29f57ed03df513da8f2a03f2adad5c5ddec1007172ebdd71ac5c244092a44c` |
| `9efa4ece-5030-42a5-9da0-1ff3d1c6ae93.npz` | `38ff81d1bbc4cb584692e9272fc1a2017b603700951f74d5d98bb230db6bd4a2` |
| `a7733a20-3b3a-48a3-a7fc-77379f654094.npz` | `a4cbd455cd1618eb7b3646a79f5def7df68736772399f3cd4726d0922e748711` |
| `a7f2f48f-d409-4222-8932-3d2208bc2ca0.npz` | `ba3b9ea24e89eb76224d494feae07b14b473dd88327e0448b835d602fae9126e` |
| `b3942662-f316-4a91-b4da-fce76f1249e6.npz` | `1fff316f1fe59bc74e7c7a6979ab79757014785ce6071ca376a28a3cfb5a3717` |
| `bd523a23-0fea-43a1-b6e7-675516d0dba5.npz` | `e2b4467d02ffe7aeb285e82773e816fd3cc0940342aefd7c622419d4dccd6a62` |
| `ca7accbc-3505-4fe3-9ef8-ad1e5a5041e4.npz` | `3ffebb84339d37d0cab87eb95eb5ef4a16c83b386d377f155f567601c5d510c5` |
| `cb1b00aa-b3a5-40b2-99d1-5175d6f70e13.npz` | `1c499beaeee8546b33c2c93045db8bf6acaf3dae0d999ccd40fe2cfeffebf1c0` |
| `ce75af77-b53c-41b1-8524-390e317d4683.npz` | `9ef4605c5bed52b4583377a8b06334636380310faa7934cd6d2eb83b9b9a2483` |
| `d1ba973a-5f8b-4cc1-9016-9f1205b5ded8.npz` | `770407beb26fd5bb63fdf3ecc33f4319a45617eba73babce5ec27ff99fd8b903` |
| `db4bc13c-b192-4d75-ba60-13d204983245.npz` | `ea5ef5fa4d5fb28c2f0f5257ce84de534ebeccfd139961499c9655d25bd93fad` |
| `dd56464d-86a7-4861-8723-cb06ef86747e.npz` | `3849cfafb1e1a503c5385d97e39788811ea18563146a9e11f7085bc6cdc22c1d` |
| `e44934d1-4c65-4a6e-928c-fa2b1a0669da.npz` | `045ca708c0e9c6e5039507e26bd792a24316454fc46f2b4c13193e7fa9cb92ae` |
| `e55e991b-50f1-48db-8114-f717f6f0f494.npz` | `bd677a298655f1fa4e499b02ee9e401e719350c02eec161f4d8ad22c546b1ce0` |
| `e848c843-d275-4f60-a194-61cc7f699124.npz` | `b922ed44a69d30565df2547182c6827b981a1cf1075c85daa483162fb9ca5a29` |
| `e9335cc9-4273-4c4a-b975-051551836cae.npz` | `c0bb80ef271e9e9e51adcb5c2bba7fa36bba0a72919adee1d37d6ab3bbc3b166` |
| `eb9d5707-1ed9-4f65-9b88-bd5495f6f586.npz` | `e50ff45ce723db813d508b11fb80c710fe205e5b651737982b2331cc283747b0` |
| `ede8c19c-d344-4835-8613-4692b0c605f4.npz` | `41371b5c89ae0af3039f6e05665327901c46920ca1d2b73548905d096ec904e8` |
| `f36e44f4-f3f6-4163-a262-c3e445a9b44e.npz` | `31ce028195e35b5ec6496505dc1dd1d9c2592bfecd9d7909027c6c0c1a2aa10d` |
| `f5e30435-0c70-4a1b-9f18-184ef794a28d.npz` | `fbed9e8609a9337974a55c25599799adb7df7ca026a68f38b7de8e4b4b4f6573` |
| `f5e4e873-4c83-427c-a2d2-3c5179b2bae5.npz` | `e83fbc4330a23c3499d461cbfef385bebf2e6b9f9704016c1929346975739d31` |
| `f68b3f5c-2030-4a38-985d-c319da26845a.npz` | `4e8a5df26810ea9ea1fb1fcd51c020138b7019dbd633aa9e510ff3c8387a1f4c` |
| `f69a9538-6694-48aa-952e-5b27b3f915fd.npz` | `a41831c5af650ec171fd122e0c4a79295e397284681693e319faa1671afe9703` |
| `fab24f4f-5d90-4460-bd85-7850faf12739.npz` | `f63c540169001afd4e27485f35381f8b26b3cfa3d7d06b506a03a328990faa25` |
| `fb9530e3-a98a-4c98-9fb4-afef6ae993f3.npz` | `46d0abc540c13b430af665ff897e2a959097559238de7931b5c2734169de0a09` |

## The token copy

A local copy of H17's token cache in two arrays (`copy_tokens.py`). `copy.json` records 640 of 640 sampled rows
bit-equal to H17's cache and all 126,981 pooled rows equal.

| File | Where it sits | Bytes | sha256 |
|---|---|---:|---|
| `tokens.npy` | `{approach4_store}/tokens/` | 66574614656 | `f090733a6dfa103eb8010aa1530ce5ad604af04c961b25ffdb46ed5a21b77f72` |
| `pooled.npy` | `{approach4_store}/tokens/` | 4290942080 | `ca35c14158bbaa7745081fbc7a209f63f6086c546211eef36c34ce84d82de71d` |
| `index.parquet` | `{approach4_store}/tokens/` | 30190040 | `c008dfa759236b31016b1f5a55b4544a4727a8349e524db16cff506398e5d49e` |
| `camera.npz` | `{approach4_store}/tokens/` | 1189202 | `dc05320237568ede65d49e8cb728762767c69311c590a11c78ccd5605533dea3` |
| `camera_picking.npz` | `{approach4_store}/tokens/` | 25626 | `1af3a93518fa663dc236e8cc95e7c557b0b197f90df95d39df2b83e9fe30c1c3` |
| `copy.json` | `{approach4_store}/tokens/` | 719 | `41c28a4b63abf2191f578f34c561b3be4be9721d955a73ff37a4b84aafddf2f1` |

## Closed-loop drives

One folder per drive under `{approach4_store}/cl/`: `runs/a4_armA_<scene>` (history shown), `runs/a4_armB_<scene>`
(history hidden), `runs/a4_step0A_e3ed554f` (the wiring proof), and `imagecheck/runs/drivejepa_018bp06_<scene>` (the
starting model on the rebuilt image). Each holds the rollout, per-tick metrics, videos and logs. The score file
and gate files of each are committed in `06_closed_loop_30_scenes/drives/`; the per-tick metrics are pinned here.
`_preflight_dryrun/` holds the launcher's dry runs (5,480,042 bytes; not evidence).

| Drive folder | Bytes | File | sha256 |
|---|---:|---|---|
| `runs/a4_armA_01d503d4` | 108520505 | `aggregate/results-summary.json` | `04b48c57f457374273255303fac37f70f8fe04cd88bd685e1233f69a32d3cbe5` |
| `runs/a4_armA_023b7fcc` | 134372404 | `aggregate/results-summary.json` | `5a5e75630088610723c0b380489bd8c3fe2e7e49aa18e135ad3f5c2ee48dac4a` |
| `runs/a4_armA_0245ff75` | 196481364 | `aggregate/results-summary.json` | `cc715800e17103e8e230950fa367d655d1ab70fbbe1cfef32b69fa7c32c771ca` |
| `runs/a4_armA_026d6a39` | 176561250 | `aggregate/results-summary.json` | `04cadb84998b9fae4804d38bdaaca5a14bac0350a0ba2ccf4c45be22fecb7e14` |
| `runs/a4_armA_02e075b9` | 153973337 | `aggregate/results-summary.json` | `f05528c96cc9e0c1513d2642d17ba153cf20462df49e06a5192fb266ac10706e` |
| `runs/a4_armA_02eadd92` | 253688242 | `aggregate/results-summary.json` | `e5f229e56eff059fa2119720aab883febbb4e7a14afaf11bdc17c647bf80cf15` |
| `runs/a4_armA_032b6f21` | 170106636 | `aggregate/results-summary.json` | `6414e20863469f47dec72a10cd21f2c2d717ce907e66e56dee61746acb14ce71` |
| `runs/a4_armA_04394343` | 214538945 | `aggregate/results-summary.json` | `77c2d9da30377d29551f4f9007349df46654adbb9bab08f0016e49f69c3661d6` |
| `runs/a4_armA_04749bb9` | 97988338 | `aggregate/results-summary.json` | `b712014687703e53e69ba6f8e9e23af7c26cbf68142355451a99978d2ac70ba7` |
| `runs/a4_armA_048b974e` | 168552674 | `aggregate/results-summary.json` | `1e9944f8705856f6d7df9b4b9f5ed859e4cbc66b8eabac4a4d4e32df7c2080e5` |
| `runs/a4_armA_16b83346` | 203298592 | `aggregate/results-summary.json` | `d17673dc1b1f165fce99bb9017398b3cb4e37c5e15bdccc257a4ebcc9fea3534` |
| `runs/a4_armA_1acf2f5d` | 180665765 | `aggregate/results-summary.json` | `0407cb11b1e602684bb29a4045f09550a68088d1dd593cf78e9710fa6074c1d3` |
| `runs/a4_armA_2e1ce314` | 156793530 | `aggregate/results-summary.json` | `b20f5768db8a4a3ca2818aab4609621c96b8376bec1605bb84a4918a765a3b29` |
| `runs/a4_armA_5f064b02` | 216003486 | `aggregate/results-summary.json` | `be69c85a2125e0707cf4314530e1f437dd71e9b2939acb7c5b2e18ec8a8597b8` |
| `runs/a4_armA_6725df2d` | 130829679 | `aggregate/results-summary.json` | `cf90811ec50a84aecbf56d2bef6dbb0dc434babbdd83dec18639627c38767908` |
| `runs/a4_armA_71d7d03b` | 213483823 | `aggregate/results-summary.json` | `9681ef00a4b33934fafa702d07bdc35fd4cf74c0ba54d28f839d3f72d4dddc9a` |
| `runs/a4_armA_791d68ce` | 164709510 | `aggregate/results-summary.json` | `19e845a6da5f08f598de68df56e5f022f3afaba956d64f63269ec949fb0040a9` |
| `runs/a4_armA_8023dce5` | 160908102 | `aggregate/results-summary.json` | `82596bdc234a9c70fb5b17743ad39cf2ab0fa960369c905febcfa65d3557525b` |
| `runs/a4_armA_89f2eeb8` | 192146803 | `aggregate/results-summary.json` | `a2296f36f342201540d5113cbb0c95bb4afbc76c62f6fb34b82d2f8b84f01551` |
| `runs/a4_armA_98dd56b2` | 179900453 | `aggregate/results-summary.json` | `dfe1e53d7d10c0a288736f51f8221cad80ad18a47b6588b09e1e61e7150616f6` |
| `runs/a4_armA_acd22430` | 117316483 | `aggregate/results-summary.json` | `6c7b85005e35cbb09ac2720e0080dccbba0cf28bbaaab9b46b4c62dad473a367` |
| `runs/a4_armA_b1b170d0` | 133570961 | `aggregate/results-summary.json` | `3c24b96e112de2f3ca2e4ffb7535983c669f1ecc585e6b2e59dd6c7c584c5b97` |
| `runs/a4_armA_b44a4a44` | 190115156 | `aggregate/results-summary.json` | `9c03a312e50221ea41541c1c9d5b0734d5e7bdf8cd406fb2db716c98ea8bb0cc` |
| `runs/a4_armA_bae463d3` | 178455828 | `aggregate/results-summary.json` | `f0b8b3cafa2ce014b02d8b6de6b048f5e94b2a84f146253fb359a27e679496af` |
| `runs/a4_armA_bd07fb91` | 147376409 | `aggregate/results-summary.json` | `156c06b1dc9896b42d6692cf5fbc1d44c17d5f0fb5cc8dab34da8b1815c2cc52` |
| `runs/a4_armA_cdd9448c` | 176923511 | `aggregate/results-summary.json` | `2a3ec610b31176e8c0e2f8709af7cc0230e130506bb5d628dfa879178d43a9c9` |
| `runs/a4_armA_d1a25a99` | 230591318 | `aggregate/results-summary.json` | `eb5cd78295482593774b2db49c639ae010e1fb937880423afc6a38eb9c18ca81` |
| `runs/a4_armA_d2715023` | 225879167 | `aggregate/results-summary.json` | `925465979e23812a69a7dea4ff6f0ba2e9375e92bb6afab1cb90a8b309f8337d` |
| `runs/a4_armA_e121e37d` | 174022142 | `aggregate/results-summary.json` | `b417f385154334f9d83a9a72b53a797c8a6343d93fd98c5933e92807b1cf938f` |
| `runs/a4_armA_e3ed554f` | 239892490 | `aggregate/results-summary.json` | `67ba67763aee4dccf13a77b48e61478ea765cbc0c259bd26a7f063f1bc753b69` |
| `runs/a4_armB_01d503d4` | 110022397 | `aggregate/results-summary.json` | `45605b0fad8785bf99aa615a034337e909fbb4f0faee99dac47f2abcb0ce01b0` |
| `runs/a4_armB_023b7fcc` | 133516803 | `aggregate/results-summary.json` | `d150fdf7cae6afb326b974016fa0e9375de0e2cba10334e51b1646808715a7c5` |
| `runs/a4_armB_0245ff75` | 196733862 | `aggregate/results-summary.json` | `fc9a4a9bd4f843441cb92871c2e804ce99f1432f8b9c59bd5532c233f24fa0a1` |
| `runs/a4_armB_026d6a39` | 174240735 | `aggregate/results-summary.json` | `f368e9c04e1b025c224b52264d078182e83d4f1964554aa09ea36246734ca900` |
| `runs/a4_armB_02e075b9` | 153309975 | `aggregate/results-summary.json` | `c84641151b1737e9fda93e02884c47f2e3d062007504db03a73fe9c2db490964` |
| `runs/a4_armB_02eadd92` | 234544611 | `aggregate/results-summary.json` | `07707a97dfe46a8657b7ee4e1671400c3d776c6b4c0d6599706e17ca7eed8081` |
| `runs/a4_armB_032b6f21` | 169297719 | `aggregate/results-summary.json` | `762f4ffce8945a047af0a68547c0af5b8210902a79eae61eab7ab3562c7e9442` |
| `runs/a4_armB_04394343` | 217233272 | `aggregate/results-summary.json` | `9a077fa2de8ac263a7fce9c11904b391fb97723ffb4623cff160540f614f1eba` |
| `runs/a4_armB_04749bb9` | 98231755 | `aggregate/results-summary.json` | `b63b86957ce6c4b69d100307f44890fb9ce611ed0ce323a7770087993cb795fe` |
| `runs/a4_armB_048b974e` | 157184480 | `aggregate/results-summary.json` | `45cc3ee2956797c947d54145f40d6366858b951f9658b6946a605e526f68d463` |
| `runs/a4_armB_16b83346` | 190970680 | `aggregate/results-summary.json` | `2d464ecea8cb03d49f81885e72255d40ebc827278ea2afda99d82169d028549f` |
| `runs/a4_armB_1acf2f5d` | 181743959 | `aggregate/results-summary.json` | `97a3c2baad52dca34736ad76b414b2352052a3a7fad55da58ba9fac40dc19aab` |
| `runs/a4_armB_2e1ce314` | 156631569 | `aggregate/results-summary.json` | `42e66817ba809a6de6c04b67eb805ed8ae4b579cd070c5c4061eb257ffb04693` |
| `runs/a4_armB_5f064b02` | 217171282 | `aggregate/results-summary.json` | `0a1b28a9a294d92abadfddc6dc7080a8540a6a7ccb3d4fe9aa79b54fcdbd1b8b` |
| `runs/a4_armB_6725df2d` | 131360677 | `aggregate/results-summary.json` | `559f58dc4c5f637ced98c5a060ae5c3db4602cbb683293c30c67868ef0551a39` |
| `runs/a4_armB_71d7d03b` | 213438236 | `aggregate/results-summary.json` | `69e8c7b5d7e10526773a53899fb72eb9d17dfe9882900fa84922e9fa4e5818dd` |
| `runs/a4_armB_791d68ce` | 174762556 | `aggregate/results-summary.json` | `a92ae024022e174a32ff8c014fbe77bd3cb91113362ae2b981775137880486c7` |
| `runs/a4_armB_8023dce5` | 151258715 | `aggregate/results-summary.json` | `11763a71ddfb4be0a3dc243f466650f698efafa8303f7e7b05a853d22bf246fe` |
| `runs/a4_armB_89f2eeb8` | 192071790 | `aggregate/results-summary.json` | `07068e2fa1dc89e6edae850f21e4c1729e1fbf9f8e5209c046aa20e6c5094fc4` |
| `runs/a4_armB_98dd56b2` | 180668804 | `aggregate/results-summary.json` | `dd2ad0592b8011ee018f1f6bba418b7b4761aacdb590439f5ea82a0b96468aaa` |
| `runs/a4_armB_acd22430` | 118199528 | `aggregate/results-summary.json` | `d96c5b0f7402cbae1dd51d0c8f844e49eb47d5886bd2529af5b59f5dffe01e17` |
| `runs/a4_armB_b1b170d0` | 134681420 | `aggregate/results-summary.json` | `a1d6e3704a89bc72b833fd281859e9d91f1b5d31e4ca182c373062c02447a5f9` |
| `runs/a4_armB_b44a4a44` | 185736688 | `aggregate/results-summary.json` | `28c26ac51a2c43063684c2444a7c2763669472832bf509b9f1b8183c9fdbe7d7` |
| `runs/a4_armB_bae463d3` | 164061234 | `aggregate/results-summary.json` | `8365dd96df7fd6480910087090493937c5e5c6321383e9dcafa6d6336e48df78` |
| `runs/a4_armB_bd07fb91` | 144981489 | `aggregate/results-summary.json` | `ea20777ff82486c20eea67e7b5cd58902e8b538cd0b51f63948f8866b565a2d4` |
| `runs/a4_armB_cdd9448c` | 172826593 | `aggregate/results-summary.json` | `1cec98608dfc8d40040780be25a868cb1d6041a437efd490f12235260f965bdc` |
| `runs/a4_armB_d1a25a99` | 218729342 | `aggregate/results-summary.json` | `6ac472dd6e53feaaa50795d83ef41b4b63f10aba751ec45d8c3b495845a006cf` |
| `runs/a4_armB_d2715023` | 243587911 | `aggregate/results-summary.json` | `90fcf1a46f4fc52c56030fe093cfb4bde483177cd43863aeadbbe2b035ca4bce` |
| `runs/a4_armB_e121e37d` | 170906228 | `aggregate/results-summary.json` | `d748a30b7bcd2e81d5be9c38fc5c5d98bf5deb2e6cbc728854450c0895a957da` |
| `runs/a4_armB_e3ed554f` | 237251100 | `aggregate/results-summary.json` | `3e586a011384513e23dde1174bd269f576d956bf906651b29430b8f1190f95f8` |
| `runs/a4_step0A_e3ed554f` | 236547836 | `aggregate/results-summary.json` | `a6829297e127f294db62f099843de602c44324e43c461281892396e29e29b782` |
| `runs/a4_armA_01d503d4` | 108520505 | `rollouts/…/metrics.parquet` | `75d18f40dd78d652eab63777ec585aee8f0ff2eaa834c9559077ac1c0f274f5a` |
| `runs/a4_armA_023b7fcc` | 134372404 | `rollouts/…/metrics.parquet` | `014162b5c57f226838c3e4f0265e8f276524c60599d69a6b5d6d55670da109dd` |
| `runs/a4_armA_0245ff75` | 196481364 | `rollouts/…/metrics.parquet` | `00e7f9a602ff8835be00f8c4cece67261e2ef7fd733300fd4ef37b6406e079fa` |
| `runs/a4_armA_026d6a39` | 176561250 | `rollouts/…/metrics.parquet` | `0fc00df4bd4369b2f342e2f0a8544bcb9f3061b4be77dc9e05d67adc02db5014` |
| `runs/a4_armA_02e075b9` | 153973337 | `rollouts/…/metrics.parquet` | `b0065be9c44304eafebd0a4a257a0577e3911afc0d08b910e68603d64beb2d8d` |
| `runs/a4_armA_02eadd92` | 253688242 | `rollouts/…/metrics.parquet` | `a92cd62608dd21b5647ab6f2ffc5d2fd1ee0e7c9b608e68d8b2c21fc3c1fdc4e` |
| `runs/a4_armA_032b6f21` | 170106636 | `rollouts/…/metrics.parquet` | `cc8ced9ef83770b2e9f5dc53f671edfc3c9606c311dabc3c72cdccdeee7d7796` |
| `runs/a4_armA_04394343` | 214538945 | `rollouts/…/metrics.parquet` | `3942cdb7d3078ceb0a46886448a5a626e545c595e5a5d050bd9d2dc4c59832fe` |
| `runs/a4_armA_04749bb9` | 97988338 | `rollouts/…/metrics.parquet` | `4efcab2d495cf874c288ac2d1c78d2d47e6618a19e79c59268c531e0b1d68c77` |
| `runs/a4_armA_048b974e` | 168552674 | `rollouts/…/metrics.parquet` | `561c544e26e1e6c7d6b917021184828e8e7b8dd8abfc13b2fefab612479ee4de` |
| `runs/a4_armA_16b83346` | 203298592 | `rollouts/…/metrics.parquet` | `ebb4cc0188d7e2a0ed54e77130fe4fd70671fb15194def89e7e155f86b7d91d1` |
| `runs/a4_armA_1acf2f5d` | 180665765 | `rollouts/…/metrics.parquet` | `ccf550c8b85828c5d1d1170912561b1653be6ff2247821fcc45b060d6129f874` |
| `runs/a4_armA_2e1ce314` | 156793530 | `rollouts/…/metrics.parquet` | `f2593c61dce700d616fe6b04b532165f6be2cad29de2a0db8bedc6ca2f15b2db` |
| `runs/a4_armA_5f064b02` | 216003486 | `rollouts/…/metrics.parquet` | `867de80826b0da4c4f4bb1eb3d03593ca29532bbc6cfb9faf350f0ca5d146ca3` |
| `runs/a4_armA_6725df2d` | 130829679 | `rollouts/…/metrics.parquet` | `25ed3f945268e94e4d9ca3b68d4179efb24c349404b2c529c821ff3488f09caa` |
| `runs/a4_armA_71d7d03b` | 213483823 | `rollouts/…/metrics.parquet` | `b599f6266de19b18c7011efc9660a2dd77dbfee23c143a276c7079e14f7cae5a` |
| `runs/a4_armA_791d68ce` | 164709510 | `rollouts/…/metrics.parquet` | `29032f1bbe3ea8373d2fa3c8f40a70d8155781b382759dcad5e64e53ccef557e` |
| `runs/a4_armA_8023dce5` | 160908102 | `rollouts/…/metrics.parquet` | `de13c49b87e728a1a01bb219f2d0ec47d3a57e423542160122aecf9316849470` |
| `runs/a4_armA_89f2eeb8` | 192146803 | `rollouts/…/metrics.parquet` | `06ad743346c36a2ffaaa7dc95ca5235adf1657b799f9dfa25d8db2b206709042` |
| `runs/a4_armA_98dd56b2` | 179900453 | `rollouts/…/metrics.parquet` | `727af359054a47c9a97905cfe313a2d8d784c8237864f76e43b977204cc9bd96` |
| `runs/a4_armA_acd22430` | 117316483 | `rollouts/…/metrics.parquet` | `f8f54fc62b93370e6bcc9598c0c439a3998b87c60f86c499ae13c781e4938a61` |
| `runs/a4_armA_b1b170d0` | 133570961 | `rollouts/…/metrics.parquet` | `7c029aeb4200dbb22009926eb93fbdb4e329aeea17bf2190ca85cd1bf5c38394` |
| `runs/a4_armA_b44a4a44` | 190115156 | `rollouts/…/metrics.parquet` | `5d4cc0d63bbc13cf9401f4b0ee5b03543573755de1cf3cea32495f06d24ae018` |
| `runs/a4_armA_bae463d3` | 178455828 | `rollouts/…/metrics.parquet` | `837119c80ab0e1b97ef8c91ad7b589b0d196fc03aba6e564c4f8b079d63f056b` |
| `runs/a4_armA_bd07fb91` | 147376409 | `rollouts/…/metrics.parquet` | `532b7d90604054653132876aef45a6e91e5f8c66dc9b96607f2e7b61b45ce91f` |
| `runs/a4_armA_cdd9448c` | 176923511 | `rollouts/…/metrics.parquet` | `f676b906cc71a7b52a8ea197f7d4d94e242decb29b340645a909e5fe0ffd4255` |
| `runs/a4_armA_d1a25a99` | 230591318 | `rollouts/…/metrics.parquet` | `2d8c6fadf76e3f5c45a6363abc7703b30f51c6534bd747728c929881fee87a0d` |
| `runs/a4_armA_d2715023` | 225879167 | `rollouts/…/metrics.parquet` | `4715d86393088d18a574c5b42e3b625e9ff3e952dc42f766470a46148ce62fcb` |
| `runs/a4_armA_e121e37d` | 174022142 | `rollouts/…/metrics.parquet` | `b5ad98b7ed74afb4b12617c0895a99f8c13539ee499b3a0e882da507b835c044` |
| `runs/a4_armA_e3ed554f` | 239892490 | `rollouts/…/metrics.parquet` | `08ac1b858a71f5129303a47779492f74242ade3ef3da58ff593340ad3278e28d` |
| `runs/a4_armB_01d503d4` | 110022397 | `rollouts/…/metrics.parquet` | `d4b440b9b1c7c8a619b0748ef898edf56601f6dd4b34d0cc8fa96541a9467945` |
| `runs/a4_armB_023b7fcc` | 133516803 | `rollouts/…/metrics.parquet` | `1fba1521ff0369d2a775869067ce5abe08f53387f7c4d1a3f7b987fba9da3361` |
| `runs/a4_armB_0245ff75` | 196733862 | `rollouts/…/metrics.parquet` | `a4fbb0a62a7e77a8fe77132735073b9f850e2fa242001bcef6e4833505e72870` |
| `runs/a4_armB_026d6a39` | 174240735 | `rollouts/…/metrics.parquet` | `2e4e9ebba56c90e0f1ce74b63de30ca4ce21fd2f77695d76c328ef92bdad4ab0` |
| `runs/a4_armB_02e075b9` | 153309975 | `rollouts/…/metrics.parquet` | `812e70c390f364dda02a9e526aad9e5c9a154b42729b2eab191b6bbc6743d802` |
| `runs/a4_armB_02eadd92` | 234544611 | `rollouts/…/metrics.parquet` | `2705ad865ae4a3fc0b4da03858d5cd7d31371847c37b7793a72c4ab2545206a5` |
| `runs/a4_armB_032b6f21` | 169297719 | `rollouts/…/metrics.parquet` | `b6ec79eb81f95772275c04af57b43eba028a20baf0ecabea5d76058b8d90133c` |
| `runs/a4_armB_04394343` | 217233272 | `rollouts/…/metrics.parquet` | `8c79c7e49432c6ea475fdf4b276183eb106e5a1c4c30ea2b8d5f7459d9c91707` |
| `runs/a4_armB_04749bb9` | 98231755 | `rollouts/…/metrics.parquet` | `492b3eda1ffd997f28e32e1279a72a7d48788ef3e5300735f37a14bc7db7e01f` |
| `runs/a4_armB_048b974e` | 157184480 | `rollouts/…/metrics.parquet` | `9741a522ef5f754cc6014f42c6d0068802b416220a0842d19596bc45f89672c1` |
| `runs/a4_armB_16b83346` | 190970680 | `rollouts/…/metrics.parquet` | `e2c25b5505e03a246bceb35e7f020067eacb6730b32d610ed22c1f6e49cfd407` |
| `runs/a4_armB_1acf2f5d` | 181743959 | `rollouts/…/metrics.parquet` | `1177fd4b0a1ab2bec87f4c32750f021bb2ccdd78aa1e4b0c04567703c5bf9b25` |
| `runs/a4_armB_2e1ce314` | 156631569 | `rollouts/…/metrics.parquet` | `c7df90639e42bd908a952a6d42fe68db5919d5d0b12a30121ef567ea6ed3d27a` |
| `runs/a4_armB_5f064b02` | 217171282 | `rollouts/…/metrics.parquet` | `277704566089251d2f246b8bb0489515dc95c24559c8bbebdfa1dbdd3f0177b2` |
| `runs/a4_armB_6725df2d` | 131360677 | `rollouts/…/metrics.parquet` | `31ef14e1cce5c8bf840927409eecb249d6e146653cc0642a3676b2215c8417f2` |
| `runs/a4_armB_71d7d03b` | 213438236 | `rollouts/…/metrics.parquet` | `bf34096744f69f8137ea490ae9d763a017110ea7188276a652a0b769ae4e9ba0` |
| `runs/a4_armB_791d68ce` | 174762556 | `rollouts/…/metrics.parquet` | `7ab052d7163344a210895d951161a34dd99026217a9b24120c49f96f91b598d4` |
| `runs/a4_armB_8023dce5` | 151258715 | `rollouts/…/metrics.parquet` | `073e5a0ab49ab76480dc223fc8d0a1b431af2035776285ebdd6069d6ba38d97d` |
| `runs/a4_armB_89f2eeb8` | 192071790 | `rollouts/…/metrics.parquet` | `67aa6d1f6b060cfc3b82e4c8af535244b2dc25e23685521fd5a5f4bfffdf7b36` |
| `runs/a4_armB_98dd56b2` | 180668804 | `rollouts/…/metrics.parquet` | `4977c74ea6c84534eb2312eb7ce640fa64ff33ef3ef13e3e6c26db0e4550618c` |
| `runs/a4_armB_acd22430` | 118199528 | `rollouts/…/metrics.parquet` | `62957ff89a2fb843232cd6856ff723784cc2aa62954c3d5fec58b3123734fb23` |
| `runs/a4_armB_b1b170d0` | 134681420 | `rollouts/…/metrics.parquet` | `18e4a67720313570c72ba06f0dd5bba8bbf597315da055bf4a0d65068ae1bce5` |
| `runs/a4_armB_b44a4a44` | 185736688 | `rollouts/…/metrics.parquet` | `79086c45dfade0483e81ab20a9a2def342d0f02fe588f41d0a6e5f526b3cd493` |
| `runs/a4_armB_bae463d3` | 164061234 | `rollouts/…/metrics.parquet` | `49d33c6f935325b5a3188db8e3e8ce594f3375ab34857ec61cca08d0e95de3a0` |
| `runs/a4_armB_bd07fb91` | 144981489 | `rollouts/…/metrics.parquet` | `551f9307c3770f80cbe5fcd08342661d786565c8143afd43bc290e9d9f816367` |
| `runs/a4_armB_cdd9448c` | 172826593 | `rollouts/…/metrics.parquet` | `b6d75ef25e39e9ef7b7d4d793daf11a7592c5fbf824bc23e8b3b2743a863ae83` |
| `runs/a4_armB_d1a25a99` | 218729342 | `rollouts/…/metrics.parquet` | `6391f3a4c02a136534af3e91a87bdeea974d0e5d682c30e19f75b0a0a3aed281` |
| `runs/a4_armB_d2715023` | 243587911 | `rollouts/…/metrics.parquet` | `79900b4b376c12448b621cbbf5a026ef2472a66a9f35a3a154a524fca1885d14` |
| `runs/a4_armB_e121e37d` | 170906228 | `rollouts/…/metrics.parquet` | `a397d2086578b3413565d4355d243ffd8fa481d27b15f67664a04d69fd3ffbac` |
| `runs/a4_armB_e3ed554f` | 237251100 | `rollouts/…/metrics.parquet` | `78d9bf27b87f04eaa890c9443573c3591f2ab47218b0df51cc75dd69c5abbfc7` |
| `runs/a4_step0A_e3ed554f` | 236547836 | `rollouts/…/metrics.parquet` | `84b04d8fc03f98d0dc594cf36818791f8f44749db7fb8c8987766595e8785110` |
| `imagecheck/runs/drivejepa_018bp06_026d6a39` | 180674868 | `aggregate/results-summary.json` | `005a0086226f6e45587f971ac9af5f1b9693a279894fc7f84b2b86325f02ab75` |
| `imagecheck/runs/drivejepa_018bp06_02e075b9` | 153302311 | `aggregate/results-summary.json` | `ab8b6a6e64e2a090c6941c814c5b8a72ff6b198eb568ed8eede960d795a65a37` |
| `imagecheck/runs/drivejepa_018bp06_791d68ce` | 152617502 | `aggregate/results-summary.json` | `9ab7db8f23ee0ac5f19794e47782ca20d10c31df8bc8bcd2a9385b6b0ed462fd` |
| `imagecheck/runs/drivejepa_018bp06_89f2eeb8` | 192143084 | `aggregate/results-summary.json` | `55c5c429074111db0977894807f797b1876c5adb433ee293de5896b3fa587c26` |
| `imagecheck/runs/drivejepa_018bp06_026d6a39` | 180674868 | `rollouts/…/metrics.parquet` | `81b31d9d06679a578fc3c135590d872dfa4c15309dfced4a5a83c6793788d452` |
| `imagecheck/runs/drivejepa_018bp06_02e075b9` | 153302311 | `rollouts/…/metrics.parquet` | `863b97b4e75de2215d491845183cbbb9f2d9f9daf542c180268d6185fd07ed20` |
| `imagecheck/runs/drivejepa_018bp06_791d68ce` | 152617502 | `rollouts/…/metrics.parquet` | `f0ad3fee6e8045895668708466f9d4e8287dccf2a32857d9529963d4d8bf6f71` |
| `imagecheck/runs/drivejepa_018bp06_89f2eeb8` | 192143084 | `rollouts/…/metrics.parquet` | `384f52df44f048358c5ddbcaf7d7ae7fbfb6d0608d219a0bc836f42dec91412f` |

## Simulator images

Saved with `docker save` on 2026-11-20 after both were built; the tar sha256 was recorded at that time.

| Image | Image id | Where it sits | Bytes | tar sha256 |
|---|---|---|---:|---|
| `alpasim-base:0.134.0` | `sha256:ab16bc78a4686dcd63c2b77a77fe585df3846528a61b552db6b71f9553983a44` | `{approach4_store}/images/alpasim-base_0.134.0_ab16bc78.tar` | 12088414208 | `be5ea7379f65d9314f8a49d610cdc7bf70f167face63348ad3ea75db84630108` |
| `nvcr.io/nvidia/nre/nre-ga:26.04` | `sha256:97f43e7130c5636ce3e80ea3184d97f56a87fdd989b05cce42230881dbdea284` | `{approach4_store}/images/nre-ga_26.04_97f43e71.tar` | 14287742464 | `84f2813eb6e4d2677de76ca8bc63e6c4980893d2949091cd897d6161a5a549e6` |

## Logs

The working folder's `logs/` (cited by the reports as `MH/logs/<name>` or `logs/<name>`).

| File | Bytes | sha256 |
|---|---:|---|
| `logs/a4_worker_check.log` | 4427 | `09deb4553666d1987f2d945591c5a85b5bfa46f08e3c72841f223b650ee7bebc` |
| `logs/checks.log` | 4561 | `ba5ecdbaf89c8dd24f31d676d6fe8c76c34c186ab69c286f31638fd5146ea726` |
| `logs/copy_tokens_finish.log` | 1095 | `32677c68a76dad75882d82ebc93317d45460df27d9f682f63f803b6128e2c47b` |
| `logs/copy_tokens.log` | 2454 | `58cec7276698f4e6f3f7fd398c8b0eb1130cc2db787ca536054ca765e9b4d7ad` |
| `logs/data_check_eq.log` | 5149 | `33330bfb7fd4294eb329d56ff4a865206767b170b85fa767ec06a8b4a9f00ddb` |
| `logs/data_check.log` | 6028 | `73bc57a2b9707e084e4f065dedcf6d185491f9b8f6a29c9092a43baf8c429968` |
| `logs/det_probe_det.log` | 20822 | `5579eee7a4c8163ed60c6daba4c973a42fbc8e6eb1de9bfbeb314c7021f9fc95` |
| `logs/det_probe_plain2.log` | 21256 | `2b87029473069fd17f625beb62e57d64b21534e5868de2b67226c27cb217f5f6` |
| `logs/det_probe_plain3.log` | 21264 | `967ffafadb7fdec7c06f346f74d9acbe0692155fa83b27420c41a510c85c1db9` |
| `logs/det_probe_plain.log` | 24414 | `50532cfe06d467f2be8fded28981460e108fb9b4a38a22c27f1d67ce571e1f38` |
| `logs/drive_dry_run.err` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `logs/dry_run_A.log` | 22047 | `c00762171ffa0d9318d4888135f1e893cfca4555c569bfe9876efacca96cd5ac` |
| `logs/dry_run_B.log` | 22044 | `f2cbfbd6a8299b543ba962be2cc2cc131681c8fdb86b288a41243eabd74e6523` |
| `logs/local_reader_bench.log` | 149 | `a87ecdfb7e7d6fba3497ca2b95eeb816519ffce3c4c7aa1862785c1f801dbfd5` |
| `logs/reader_bench.log` | 1017 | `f77dcff17ee6afcee70a3c64c9ab6a278951d50344e06c953261851bce8c652b` |
| `logs/resume_test2.log` | 58811 | `43ab5aa5cca73ad351dfdac9e7fe267dbb569e582b8182e8bdd6abf956b5b330` |
| `logs/resume_test.log` | 58830 | `9ae3034e816929a4a9dc5f9ba872cb121ffc11a0aacf36946e02eda6866e1d84` |
| `logs/side_file_full.log` | 3500 | `8f25e998c641a71e850f9cef78c2de50048fec7bd01eecd9565c7565dfbac3fe` |
| `logs/side_file_full.stdout` | 3500 | `8f25e998c641a71e850f9cef78c2de50048fec7bd01eecd9565c7565dfbac3fe` |
| `logs/side_file_time.log` | 355 | `1e771d1d43bcd2d0cda9d086fa6c1ad8e94e6704b6c3ec89da61b1ed7a3f408d` |
| `logs/step0_check.log` | 28236 | `11aa31ed5d352175c50f62ae8cc39dfa59e5a499a8988c5bb51137e661ce1dde` |
| `logs/test_ego_history.log` | 524 | `78ee4a189dd0a11765536e3e8228ddfbda4e8841d0b7da63146bf15a0f61e8cc` |
