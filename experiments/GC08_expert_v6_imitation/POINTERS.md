# Pointers — what GC08 produced that is not in the repository

Nothing below is committed: checkpoints by rule (`docs/checkpoints.md`), and the rest because it is
built data or a run record that lives on the training site's store. Each row gives a location, a size
and a sha256, so a copy can be recognised wherever it ends up.

**Locations are templates, not paths.** `{weights}` and `{runs}` are `[roots.weights]` and
`[roots.runs]` from `gallops.toml`. Paddock's run folders are given relative to paddock's own runs
root.

**A folder's digest** covers every file in it, so one line pins the whole set:

```bash
cd <folder> && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum
```

## Weights

The pins are copied from `experiment.toml`. Paddock loads `agent.ckpt`.

| Location | File | Bytes | sha256 |
|---|---|---:|---|
| `{weights}/drivejepa_swerve/v6-r1a-loso0075/step_000015684/` | `agent.ckpt` | 1,245,089,903 | `cd4fc891749717b707ec4847b27370dcd27e57ce348ebeecbb4698abc92f9ca5` |
| `{weights}/drivejepa_swerve/v6-r1a-loso0075/step_000015684/` | `model.pt` | 1,245,091,311 | `49bf193783003cb8487eb6cc29856556fa07b4b80c7c6d182cc022b22e51b838` |
| `{weights}/drivejepa_swerve/v6-r1b-all/step_000069186/` | `agent.ckpt` | 1,245,089,903 | `69061f9ced807d6baa988379ccbe35fb46129bc6d4ffb98c0f4810a37868d06e` |
| `{weights}/drivejepa_swerve/v6-r1b-all/step_000069186/` | `model.pt` | 1,245,091,311 | `2195fc27212d57a22ca35067d9f2642894343e0fbd60e6cfc6863c96213487eb` |
| `{weights}/drivejepa_swerve/v6-r2-short/step_000011377/` | `agent.ckpt` | 1,245,089,903 | `305892faf6bbb864dfffb06422029ce73b63bd0316deb437c4c48981da62f33e` |
| `{weights}/drivejepa_swerve/v6-r2-short/step_000011377/` | `model.pt` | 1,245,091,311 | `916b930a38ead276c601c3e6ccdbe0b5fbd20835f252c7cc879887cf8a5bdff9` |

## Labels and candidate dumps

Ticks are lines in the label files; usable ones have `usable_for_imitation` set, and they are what
the trainer read.

| Location | What it is | Files | Bytes | Ticks (usable) | Folder digest |
|---|---|---:|---:|---|---|
| `{runs}/labelled_r0cam_v6/` | round 0, relabelled by v6 | 150 | 17,668,548 | 8,402 (7,932) | `6534a97e02175e7cb8a3493cea4f5ddd1b4827d9d942b5a3dcd59afdfc5e1d42` |
| `{runs}/labelled_expert_v6/` | the expert-driven collection, labelled | 150 | 20,086,473 | 9,558 (9,234) | `5a356c179c959c3f951d3cff95dc4179cb04394b76a5cc39767d45dae7fdc65f` |
| `{runs}/labelled_r2_v6/` | round 2, relabelled by v6 | 40 | 5,190,899 | 2,466 (2,402) | `14dd32e70911567a9cd16d6024c364f445ad7888ea679057bcdfc0acbeb5bdee` |
| `{runs}/labelled_nusc/` | real nuScenes driving, the human source | 613 | 11,215,521 | 18,015 (18,015) | `ba54386f0b59bc47f825dce31f0e3cd16f35d6440ee3e176f21e7b6df8a8668c` |
| `{runs}/collect_dumps_r0cam/` | candidate dumps, round 0 | 150 | 205,118,601 | — | `6224a8edf0cbb35a179215f83cf9359c97fc93630742692eb30d4396e0fbae18` |
| `{runs}/collect_dumps_expert/` | candidate dumps, expert-driven | 151 | 227,499,705 | — | `84a1e94140ab76b6f00d38ec4b0c12ab148c14aef555eb9f5902d4cafcf60400` |
| `{runs}/collect_dumps_r2/` | candidate dumps, round 2 | 40 | 59,338,008 | — | `fdbc44d76bd6f6928ed72b2e06756132dbe34bb8b04ea6873ed2d89507f92cdb` |
| `{runs}/collect_dumps_r1a_eval/` | candidate dumps, round 1A on 0075 | 30 | 44,183,857 | — | `bf541865c4e64d7996ac906faf5be4e8d0fc8f3b31a95eea95d881fb04b17f47` |

## Inputs every v6 label read

The routes come from the scene packs and the road from the rasters. The labels and Gate B read
`drivable/`; `drivable_all/` links the same 5 files beside the real scenes' rasters.

| Location | What it is | Files | Bytes | Folder digest |
|---|---|---:|---:|---|
| `{runs}/scene_packs/` | the 5 scenes' logged data and routes, from `rollout pack` and `routes` | 5 | 988,922 | `2ad03270617702b5a73ff4116466734ea5abbebdbfdf114711f05c1d9ef8f4df` |
| `{runs}/drivable/` | a road raster per scene | 5 | 56,690 | `5647dd4682737e106ff8ccb9adfaa0989b10e4b8ca2bfd8c439afb5bc5af7775` |
| `{runs}/expert_plans_v3/` | v6's plans and poses per episode, which the expert-driven collection drove | 150 | 7,109,787 | `b023380a8cd95f5a77e050b0db10c3c2e6ffaf02e5192c65b2454d9c75d851df` |

## Collections

Rendered episodes. **These are not hashed:** most of their bytes are camera frames, and the folders above
pin what was trained on them.

| Location | What it is | Files | Bytes |
|---|---|---:|---:|
| `{runs}/collect_r0cam/` | round 0, driven by the released seeing model | 9,302 | 13,981,390,428 |
| `{runs}/collect_expert/` | driven by v6 | 10,458 | 15,756,538,746 |
| `{runs}/collect_r2/` | round 2, driven by 1B epoch 2 | 2,706 | 4,057,976,651 |
| `{runs}/collect_nusc/` | real nuScenes, in the collector's layout | 3,395 | 141,262,385 |
| `{runs}/collect_r1a_eval/` | round 1A on held-out 0075 | 1,986 | 206,816,936 |

## Paddock runs

The evidence behind the closed-loop and open-loop numbers. Committed to no repository, which is what
the results' `no-record` flag says.

| Run | What it is | `samples.jsonl` bytes | sha256 |
|---|---|---:|---|
| `neuroncap/drivejepa/20261012T205915-7153f0` | released seeing model, seeds 0-2, the 0103 baseline | 2,318 | `4d7891935cabc0b6d3c38805d771b3b6c95d466db664688f59d60c62b92009d6` |
| `neuroncap/drivejepa/20261014T153543-0d6a7b` | 1A epoch 1 on 0103 | 2,330 | `344d4a47c8920460d0d8cd90f67edb3662e5ee0f51cbf06ca63109a3818c6916` |
| `neuroncap/drivejepa/20261015T055256-49c277` | 1B epoch 2 on 0103 | 2,274 | `681ce77a7d99cf4b1267a434657136d4472347ddc2d4c0c11a4d218cb666caa2` |
| `neuroncap/drivejepa/20261016T180740-bf1796` | round 2 on 0103 | 2,366 | `bd5126723b3e757c871d8d94dcc8333a59a2740e63af32df7a41f5432b151903` |
| `nuscenes/drivejepa/20261015T074632-198187` | 1B epoch 2, nuScenes open loop | 6,662,200 | `5846a2b64b70b665e14a5a8e5a41cc3a1d2cd247f1600b772ec30306e7ce34af` |

## Paddock code

Copied in [`paddock/`](paddock/README.md), so nothing depends on one working tree. Where each piece
came from:

| Code | Where it was | Pin |
|---|---|---|
| `drivejepa_ncapdump` with `DRIVEJEPA_DRIVE` = `model` or `gt` | branch `analysis/drivejepa-ncap-candidates`, not on `main` or any PR | `b1c57796b80c568167e1859134d4eeb7e024879d` |
| the parent adapter, with the camera-frame fix | branch `drivejepa-ncapdump-camfix`, never pushed | `f4f5efc0acd7fee592aab0198407dc09a6c2aba1` |
| `policy.py` with `DRIVEJEPA_DRIVE=expert` | uncommitted, in the worktree `paddock-dj-ncapdump-camfix` | 18,573 bytes, sha256 `ecd6641be8256a2ed78dbfdb39534ab3700c289275667763e466467866249375` |

## Evidence

Reports and logs under paddock's runs root, where the gates and the collection wrote them. The
scripts that made them are in [`scripts/`](scripts/README.md), and the probe's output is
[`scripts/probe_v6.json`](scripts/probe_v6.json).

| Location | What it holds | Bytes | sha256 |
|---|---|---:|---|
| `ops/dj_expert/gateA_r0cam.json` | Gate A: 150 of 150 episodes, worst 0.069 mm | 56,697 | `17bbb86afd8626e68e20b260e71c176dba55443d09bb79dcef6ef4d9d2fa6746` |
| `ops/dj_expert/gateB_v3_T3.5.json` | Gate B: the account's table and agreement figures, and every tick's plan | 9,114,255 | `bea45e729e19ab754195291af315adc6437803543bedfe579a81619e568c3379` |
| `ops/dj_collect/collect_expert.log` | the expert-driven collection: 150 of 150, the 2 cm assertion never firing | 482,940 | `25c711333a2d70c5d8b1bc5d871a92793422c8a6a465bfa096408a79cf137ed2` |

The review of #17 reran samples at `67ad95f`: Gate A on 6 episodes, worst 0.049 mm, and Gate B on 48
of the 50 frontal episodes, 8 hits, all 0058 at 1.56.
