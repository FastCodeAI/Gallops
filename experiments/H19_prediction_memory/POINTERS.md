# Pointers: what H19 produced that is not in the repository

Nothing below is committed: arrays and checkpoints by rule (`.gitignore`, `docs/checkpoints.md`), and the
rest because it is derived from NAVSIM navtrain. Each row gives a size in bytes and a sha256, so a copy can
be recognised wherever it ends up. Hashes were computed on 12 Nov 2026 from the files on the training
site's store.

**Locations are templates, not paths.** `{data.navtrain.cache}` is the `cache` key of
`[datasets.navtrain]` in `gallops.toml`, which `gallops.config` resolves. The record's `source` fields
write the same folder as `{data}/memory_cache`.

**These are the files the weights were trained on.** The hashes were taken on 12 Nov, a week after
training, and the trainer of 5 Nov wrote no digest into its sidecars (it does now). What ties them: every
input file here was last written before the first of the six checkpoints (the latest, `scores.npy`, at
06:44 on 5 Nov; the first checkpoint at 06:49), by the store's modification times.

## Checkpoints (H19.r3-a-s0 to H19.r3-b-s2)

`{weights}/drivejepa_memory/`, in the layout `docs/checkpoints.md` gives. Each `memory.pt` digest is the
`weights_sha256` the record pins; `a` is the memory carried, `b` the same layer wiped every moment.

| File | Bytes | sha256 |
|---|---:|---|
| `h15-r3-a-L4-s0/step_000001638/checkpoint.json` | 1810 | `9caff956d295387bfa968107e07586b5ff64b96ce9d6a09765480e06c06aa6cb` |
| `h15-r3-a-L4-s0/step_000001638/memory.pt` | 1336629 | `0679ea453b9714dbc524d4d8435619572fac784ca006b9935adffbec6a5456b4` |
| `h15-r3-a-L4-s1/step_000001638/checkpoint.json` | 1811 | `d5803cffed511ce849a368353f1ed8eabf48bf4a81bc64fb9ab8a734a6faeda8` |
| `h15-r3-a-L4-s1/step_000001638/memory.pt` | 1336629 | `c23859b7798b95c68afeb926eadeba668b44c5a38867f38c7f61f35b54dbfdd7` |
| `h15-r3-a-L4-s2/step_000001638/checkpoint.json` | 1811 | `28ee3135843000aa80711e08343bb6ba07d8c73df02ee357cc1168c174a14f97` |
| `h15-r3-a-L4-s2/step_000001638/memory.pt` | 1336629 | `df6298b2f15e95ea6b3ac60d2957de6533b7763f1fcae1df110c18b226c0e409` |
| `h15-r3-b-L0-s0/step_000001638/checkpoint.json` | 1808 | `3c0233669990474e8661eabdd1054964ad5e8f2748214d2f85f9355d25a1c5e2` |
| `h15-r3-b-L0-s0/step_000001638/memory.pt` | 1336629 | `dfd944c1c3f1ceabae1cfdfc2bf63c55c19872b887ad2ffd3909da2c29ed8bca` |
| `h15-r3-b-L0-s1/step_000001638/checkpoint.json` | 1807 | `508ae84a1773a3658ddf899fabbc3962dc8b4bd2c583258777fc8583284308a4` |
| `h15-r3-b-L0-s1/step_000001638/memory.pt` | 1336629 | `7db7b315efef84aa418e44dd69a2fb5141374ff8d5f20425bf574e0aa7b110ca` |
| `h15-r3-b-L0-s2/step_000001638/checkpoint.json` | 1810 | `8683b58ddee9e7427fc203059c19887539abd96db2366b19e11db7b4f0359205` |
| `h15-r3-b-L0-s2/step_000001638/memory.pt` | 1336629 | `deb69754a231eb352fc5924686d8ab220a980633a117996d15749bc3b0152004` |

## True grades of the 32 paths (H19.r1)

`{data.navtrain.cache}/grades/`: one row per graded moment, 91,977 moments.

| File | What it holds | Bytes | sha256 |
|---|---|---:|---|
| `done.npy` | which moments are graded | 92105 | `898009884744c81b18f4512d55c8796dd6a74fa0b6a6b35a926670a47f2a7101` |
| `forgiving.npy` | lane keeping and traffic light the benchmark's way | 23546240 | `0cc6990866359d9b1cf178606de19ac0b9d269bc96d27bcaf57ade69e07124ea` |
| `human.npy` | the human driver's path, graded the same way | 3679208 | `0d02e9e8c17ba0dd703d6acda47b045175ceb974377cd1b6bffb17d936e03c8d` |
| `manifest.json` | how the grades were built | 1149 | `dd4a888584e418255fe11b41fe13ad9492fe74bc98864c6a80f48d70b782f221` |
| `origin.npy` | the full-precision origin of each moment | 1471760 | `904d826921c83846bf76f477a58659c260ac4f299049a0d2d899786cd3d296bb` |
| `progress.json` | the build's progress record | 45430 | `f2f20996d3561eb027eea11327dae558c17504d8aeea9e0f235f098f4e486177` |
| `row.npy` | which cache row each graded moment is | 735944 | `e83cf08fa50ff5968d1ef58cd5f987dae8aae55c213834d292d1cc49318cd576` |
| `scores.npy` | the strict sub-scores and the final grade of each path | 105957632 | `0b5e3ab1fd24b2fb27ae7be17f983c4741b4314d74480d105751dd6e916ae938` |
| `states.npy` | each path's simulated states, as offsets from `origin.npy` | 5309648384 | `8f26a214e2705f1abce1fe71c3afab23612eac6256d24d0a3f6192d4167f61af` |

## Matching each path to the previous moment's closest path (H19.r2)

`{data.navtrain.cache}/align/`.

| File | What it holds | Bytes | sha256 |
|---|---|---:|---|
| `manifest.json` |  | 176 | `b940e98b712df2784983a26deeb87aa732345fae48ccc28214c0037920e90d32` |
| `match.npy` |  | 8108544 | `5cde0ee8683a680d5469590e812b0010f6a92e0995b5681641b370a02fa818bd` |
| `match_m.npy` |  | 16216960 | `b12a44e77748a2bfdeeb4c0f826d93ce59f1f466da233429034391bd4a079cc9` |
| `poses.npy` |  | 16216960 | `819bbe06521f909daf63118d499b6ce55a5f2a042dd31f66e312316378293ef3` |
| `prev.npy` |  | 1013680 | `1a93f83a82ae67760e6c9c060d46f819250652c9bf9975006e777ce8282d5654` |

## Selection, preflight and evaluation outputs (H19.r3)

`{data.navtrain.cache}/h15/`. `evaluate.json` is the file the record's offline results cite.

| File | What it holds | Bytes | sha256 |
|---|---|---:|---|
| `evaluate.json` |  | 8750 | `4596308be318e11bc3b2c86ab01ad53473c4be46ae324fa0f4ad80d877319eee` |
| `final.json` |  | 6661 | `e040d0fc43c10ed9be65f38be5232219261519ffcc824237c425a4b49055c11d` |
| `preflight.json` |  | 6853 | `f6d1f6fff9bab1cb90b5804dbefac825231cf14529980535195cc222508f7a08` |
| `selection.json` |  | 5295 | `9807ef2f858a03735b8e36a6af0f5533463df83b2b8cc3a748585a919fa8e7da` |

## The adapter test's array

`paddock/src/paddock/adapters/models/drivejepa/tests/fixtures/memory_fixture.npz`: random inputs and the
scores gallops' memory layer gives them, which pin paddock's copy of the layer to gallops'. It is an array,
so it is not committed. `scripts/make_memory_fixture.py <out.npz>` rebuilds it under the training
interpreter; the adapter's tests that need it skip when it is absent. The copy that was used is in the
throwaway paddock checkout the gate ran from, and in this branch's history at `1825bce`.

| File | Bytes | sha256 |
|---|---:|---|
| `memory_fixture.npz` | 549893 | `3498cf739bcda8a6f1f75aa3d7742ef65b102e5191350ab674c8b5b62dfe6aa7` |

## The gate's logs

The gate's printed output for each tier and arm. The verdict files (`paddock/results/gate/*.json`) and the
two comparison tables are committed; the logs are not. Two copies exist: the logs as they were written,
in a folder named `paddock-h15-runs` beside `{runs}` on the training site's network store (written
`{nas_store}/paddock-h15-runs/`: a label, not a variable `gallops.config` resolves), and the copies with machine paths replaced
by placeholders that this branch held until `1825bce` (`git show 1825bce:experiments/H15_prediction_memory/paddock/results/gate/<name>`).

| File | Bytes, as committed at `1825bce` | sha256, as committed | Bytes, on the store | sha256, on the store |
|---|---:|---|---:|---|
| `gate_quick_a.log` | 12538 | `6c200dcdfeb986b4c0cebbd74d1cf6b4dfbe4551e47da05fea281bfac37d4f44` | 13167 | `330e3033b56e391a3a9436d55d7c1e1ce4782efda38443ecce5c34f1e64f19f3` |
| `gate_closed_a.log` | 6868 | `d5ab7691f6549c3794a8d56dd664f4c0564a06e21e2019d550af0be0728ea20b` | 8509 | `347009426c2534dd495c4d8190a3f74d02cbbc4f101574dc1545a545fc3a2600` |
| `gate_quick_b.log` | 13461 | `349085e912bb9f5472a4bda0831ec93706660d36b4e2ebdadc838ecd88f17ffd` | 16392 | `645b5e48903c2d41d7a5b20b18b59fc682fc806211fa182d58d1cf9967c8ce32` |
| `gate_closed_b.log` | 7293 | `31de104fe9a8a9c67a0a1e09af9bc57ebfe27f625547cedb2684bfbe987b9df0` | 8934 | `2b0d10163cd2ff484f3a62d202ae11822d3b44074c1a05c0888a56db5c53a013` |
| `gate_closed_b_recheck.log` | 6249 | `75f969cd12f349d034341d77fa376545a55ad5ab8a45dacea0d5d78d5475976f` | 6334 | `b031467ff907cc1953b77ca9cb8e601d5836738b53070487c5dfd95b5059715a` |
| `gate_quick_released-alpasim.log` | 3546 | `6c53f4118a3ab271dfedbdb302798d8f93e9e409ef818a7ac81ecfb9b4d26ce4` | 4088 | `731db21b368800591b26851fc2cd6d2e6c8b7bae8308508af93b72dbc85c6f53` |
