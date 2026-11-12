# Pointers: what H05 produced that is not in the repository

Nothing below is committed: arrays and checkpoints by rule (`.gitignore`, `docs/checkpoints.md`), and the
rest because it is derived from NAVSIM navtrain. Each row gives a size in bytes and a sha256, so a copy can
be recognised wherever it ends up. Hashes were computed on 12 Nov 2026 from the files on the training
site's store.

**Locations are templates, not paths.** `{data}/memory_cache` is the `cache` entry of the `navtrain`
dataset in `gallops.toml`, the folder the record's `source` fields name.

## The saved outputs of the released model (H05.r1)

`{data}/memory_cache/release/`: one row per moment, 126,694 moments, float32. `manifest.json` names the
released checkpoint by its sha256 (`740335c8…2d30d05`) and the commit that built the cache.
[H15](../H15_prediction_memory/README.md) reads these files.

| File | What it holds | Bytes | sha256 |
|---|---|---:|---|
| `done.npy` | which moments are written | 126822 | `97873cae7dfb2f4c08cec61c3664917ff48844b271cbe71a6faa5390409f00b8` |
| `ego.npy` | the ego state, 11 numbers per moment | 5574664 | `7f5e5a23060afbea87d6f8854ff1416671e31f00de12a0d00390776b883f8d64` |
| `encoder.npy` | the encoder map, (512, 256) per moment | 66424144000 | `c19694978865d2b84b966dba534aa6189a4193748bad121b1db90ef7169bd73a` |
| `logits.npy` | their six score logits, (32, 6) per moment | 97301120 | `f9ea658fcc31c2ddb0400a032229a5d07d36399dbca2070408974f9f7a794c19` |
| `manifest.json` | how the cache was built | 618 | `dea18e58eecc08ee280983743501ffa1baa8f87e58864c55a715c1e149d227ad` |
| `progress.json` | the build's progress record | 52623 | `5cfe9a94163493417e0e2f3ee8454f433fe171b3252d6a0e0cfef3f4f7056ca7` |
| `proposals.npy` | the 32 proposed paths, (32, 8, 3) per moment | 389204096 | `da5fbe346b469b496da1521fd7884a3125be9ba3e088e4d0528ae297291e61ba` |
| `scorer_in.npy` | the scorer's input, (32, 256) per moment | 4151509120 | `0957c69c8028cde22e6c31e4f3c49328ce0ce2dcb343e5f7114f9969a556f42a` |

## The cache's sample checks (H05.r1)

`{data}/memory_cache/checks/`. The two determinism dumps are the same bytes, which is the check.

| File | What it holds | Bytes | sha256 |
|---|---|---:|---|
| `contact.json` |  | 2590 | `e1a5bb8a278ee1fe7e3ff9144fa53cf358536bfef8969566024e9eb3d62f3439` |
| `determinism.json` |  | 260 | `b56109f9fe719fc9b602cc4e32060c2dea95254765db7a750affe2efa5960db7` |
| `determinism_1.npz` |  | 11221574 | `24aa6cf7b2427a3a4e34d59d16c782f4f4cb42b205c280b83075158e14195d05` |
| `determinism_2.npz` |  | 11221574 | `24aa6cf7b2427a3a4e34d59d16c782f4f4cb42b205c280b83075158e14195d05` |
| `fidelity.json` |  | 45016 | `ea93bb45f00a7091eb0f7b824b9ee006b5b0dab4b1a12c03707fc34f6fe42505` |
| `rate.json` |  | 425 | `e3280da5b3bbe9c84a436e76e49403411c78b8077b6c17df29aa0504bb1a86d7` |
| `sight.json` |  | 2140 | `9f03dbff19d6158910e1af97ff85b857634432565aa1150149025598eb80762f` |
| `timing.json` |  | 340 | `34decbcb3d06722865c21d2c632d3bf505aaf67a5802b49ed1aa44edd9df28ce` |
| `workers.json` |  | 611 | `a1a3c775daa2f4fc53fc357a4171d7fd3c517519cb668a65601e3f6a2a2c38c5` |

## The probe and its control (H05.r2)

`{data}/memory_cache/probe/`. `probe.json`, `c/control_c.json` and `k1/refit_k1.json` are the files the
record's results cite.

| File | What it holds | Bytes | sha256 |
|---|---|---:|---|
| `c/control_c.json` | the control: camera past against the car's own past | 37113 | `bcdc5cd5f0b6fb6438c1e6c9f644e23435ae32b231d997a5be8818a809288492` |
| `c/control_c_run.log` |  | 1359 | `acb90a1396530dd32261b41426981573c68d18ef4f43f75c2cb8632bd101b7dd` |
| `k1/refit_k1.json` | the refit against the model's own view | 10619 | `4f2efc9d090ca02b086b76bb6a51f93e4575f4f1c732499fa22ff905c30b73f6` |
| `pca.npz` | the 128-number summary fitted on training moments | 4360968 | `42fc844902dc51293128f64a97452178c160cf8fee7f07db5b361848a36e670a` |
| `pooled.npy` | the pooled encoder map per moment | 4281243776 | `72519772fd34f0cdb35d34421644272a9ebca883be4570ed516eacdb250786da` |
| `probe.json` | the probe's results for 1, 2, 4 and 8 moments | 48227 | `560d169ef780328a86e86366422c81307b4b9aea481a245ec49e6846553ac348` |
| `probe_error_vs_k.png` |  | 132831 | `be88f4be1486d9210b42c842061904bca88186fd34b972cb7ec82e735c20c869` |
| `probe_error_vs_k_braking.png` |  | 110202 | `529214aa383337047e2a53ec1f6a18a7e689c9f111d3c7718ea5603b9ce53082` |
| `probe_run.log` |  | 2981 | `daa416325cdecc84d32169a3df945ac945f826e8e658e3172ed6a2ca275ed7a5` |
| `refit_k1_run.log` |  | 918 | `c62e2ceb4c89a75af78f64fdc5b3d409f14b87cd54f1a4073ce7e68e03cd563d` |
| `targets.npz` | the human's speed and braking over the next 3 s | 2204012 | `2a8d571130a04f768abb0831a679fed6480a272b0c377be3043fc2e2f6c76bb6` |
