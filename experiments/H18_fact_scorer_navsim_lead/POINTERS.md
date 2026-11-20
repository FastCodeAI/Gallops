# Pointers — what H18 produced that is not in the repository

Nothing below is committed: checkpoints by rule (`.gitignore`, `docs/checkpoints.md`), the rest for size. Locations are
templates: `{weights}`, `{runs}` and `{data.NAME}` come from `gallops.toml`. At the training site these have not been
moved into the template layout; the "where it sits" column says where each one is.

## Weights

| Item | Template location | Where it sits at the training site | Bytes | sha256 |
|---|---|---|---:|---|
| The fact-scorer model, step 23,859 of 23,859 (`H18.fact-scorer`): the trainable tensors only (LoRA A/B of the 24 blocks, agent decoder, planner, path reader; 434 tensors), no sidecar | `{weights}/run3b_v2/run3b_v2b_19nov/step_000023859/trainable_step023859_final.pt` | `{data.paddock_data}/run3b/runs_v2/run3b_v2b_19nov/trainable_step023859_final.pt` | 82,635,691 | `12ab840ef81366cb2da936372ffd1b5fddb86d026ab156f5022b6ee68b6afac3` |
| WA-JEPA's released encoder, the frozen base the checkpoint above adds to (H01's) | `{weights}/WA-JEPA/model_state_dict.pt` | `{data.paddock_dump}/weights/WA-JEPA/model_state_dict.pt` | 1,575,763,741 | `e1d3be013de7dbaa5a6c30925a0d2f91e6ce1a6b17a75ca73fe01a1abe6d6408` |

The run's other checkpoints (step 2,000 and the end of passes 1 to 7) sit beside the last one, 82,635,691 bytes each,
not hashed: none was evaluated beyond the trainer's held-out reports.

## Training record

| Item | Where it sits at the training site | Bytes | sha256 |
|---|---|---:|---|
| Arguments | `{data.paddock_data}/run3b/runs_v2/run3b_v2b_19nov/args.json` | 553 | `052cf810785f813b8e9ddd2009b9ff295643bc0f2c3838e345a70011cfaf81d8` |
| Loss weights, split sizes, schedule | `.../setup.json` | 944 | `7c81d15f83795cfa376c65548f7823151d9b93a75f2c3e124715047bc89d7130` |
| Per-step losses and timings | `.../train_log.jsonl` | 61,388,542 | `0e96b569861193ab510865f5fc38a950b53ab4495ae272b850c3beb11aab6589` |
| Held-out report, step 2,000 | `.../eval_step002000_smoke.json` | 101,544 | `cc1a19d5e0447ec4a4e615b39abb9ac18ca005d3888a093114550e2942eefb03` |
| Held-out report, pass 1 | `.../eval_step003358_pass1.json` | 101,593 | `77dd243238caf1cd0cf1860c575df4a24384d9cb3c8140dab045ca3eb071a212` |
| Held-out report, pass 2 | `.../eval_step006716_pass2.json` | 101,395 | `29e1b7f10bd309f6271c1625bcd516817cc5eb1d65e137763963fbad5f095867` |
| Held-out report, pass 3 | `.../eval_step010074_pass3.json` | 101,368 | `f3435a33007e6f6eea0c3dcfd089f791ab0b2a92b21db9d9c325d093814c38f2` |
| Held-out report, pass 4 | `.../eval_step013432_pass4.json` | 101,438 | `adbdaa83c3920edb49ad51208fe191dc4725cf2a06990012705713eaefd4da4e` |
| Held-out report, pass 5 | `.../eval_step016790_pass5.json` | 101,381 | `77764bb4cd645b8f0f430c8b7eb1a60829629f972d07bdb63a3f163e1871e9bf` |
| Held-out report, pass 6 | `.../eval_step020148_pass6.json` | 101,438 | `35b63451425d6ece1c07ad4679c4d3d3f0fa73c03974c1b9ddcebb225d5cf627` |
| Held-out report, pass 7 | `.../eval_step023506_pass7.json` | 101,430 | `a88676bd5b64ac5286de2833fdc761bc2daa35f0c9edf13fd0db92aa2e576f86` |
| Held-out report, last step | `.../eval_step023859_final.json` | 101,450 | `553e7afdc7718349f7c81e0798cae4183e764ad80ffa02b605067f3b6afb8d0e` |

## Evaluation

| Item | Where it sits at the training site | Bytes | sha256 |
|---|---|---:|---|
| navtest, all 12,146 scenes | `{data.paddock_data}/run3b/navtest/eval_run3b_v2_final/2026.11.20.10.34.16/2026.11.20.11.27.57.csv` | 1,042,396 | `54c73441ec9851da1985c7a87802727bd2574c08b31f1f69a68611604e4a09a4` |
| Gate verdict, 20 Nov, against released Drive-JEPA | `{runs}/gate-run3b/verdict_run3b_v2.json` | 37,327 | `ef0e399a3a87f0e18c5f1433f03a1deef328a893b21cab7dd8135d333f93ee73` |
| Gate manifest | `{data.p3_renesas}/paddock-runs/gate-oct27/gate/smoke/run3b_v2-released/manifest.toml` | 1,574 | `eaa8f0887ed1bf59a14e7625d5fe4acca533acbcb8706908fbaf7de5a4d2802b` |
| Paired with H04's gate run, nuScenes | `{runs}/gate-run3b/compare_nuscenes_run3b_vs_run3a.json` | 4,732 | `e08c8595ddc3fc8c06776f2840974659ae6b2e38f4b7aaab1605d9421369c707` |
| Paired with H04's gate run, KITScenes | `{runs}/gate-run3b/compare_kitscenes_run3b_vs_run3a.json` | 3,401 | `d43cbf30caccb5b55c2355cd5a6fc1e54d5ad5028e033472ff7902d2b2a5e8c1` |
| Paired with H04's gate run, neuroNCAP 15 pairs | `{runs}/gate-run3b/compare_neuroncap_run3b_vs_run3a.json` | 7,667 | `6669295f9d74a853908097469964c7e06709a941e16de9f7e301b1920c8464da` |
| Paired with H04's gate run, neuroNCAP quick | `{runs}/gate-run3b/compare_neuroncapq_run3b_vs_run3a.json` | 7,530 | `3b7729a9519d7e2b687e8e702157db1cbf8e329a523df9372a5def5a3784f3a0` |
| Paired with H04's gate run, NAVSIM (83 shared tokens; not used) | `{runs}/gate-run3b/compare_navsim_run3b_vs_run3a.json` | 7,634 | `524c182d368eeb0bcc942e11e74d4f1692a19bed5fbf987e7f98b40fd6c49be1` |
| Gate verdict, AlpaSim, 20 Nov (CONFIG DIFFERS as printed) | `{runs}/gate-run3b/verdict_run3b_v2_alpasim.json` | 12,739 | `12f0a73e34037ed979c376ef32ac46a266a24e8a617c51fdf066577d52426fd3` |
| AlpaSim pair against released, driver keys allowed | `{runs}/gate-run3b/compare_alpasim_run3b_vs_released_profile_allowed.json` | 5,788 | `45d2f173153def3eade0b07f7bf6047002b79e097c11cfb7cb8202b961fe4915` |
| AlpaSim pair against H04's run | `{runs}/gate-run3b/compare_alpasim_run3b_vs_wajepa_dj_run3a.json` | 4,379 | `167f2644793a2cf6494a4a1720b9b9c839942ab85de6166d032be6325030d3c2` |
| AlpaSim pair against H01's run | `{runs}/gate-run3b/compare_alpasim_run3b_vs_wajepa_dj.json` | 4,353 | `851e92face26461d1f3a5684702ba2870ea03cef7a693fc1de515666a6318b88` |
| AlpaSim crash check against released, driver keys allowed | `{runs}/gate-run3b/compare_alpasim_quick_run3b_vs_released_profile_allowed.json` | 5,952 | `cb003864b870f3a16c6f1add8fd33b3503d6446bf9c2391e51767379c9cd6f19` |
| Gate verdict, Bench2Drive, 20 Nov | `{runs}/gate-run3b/verdict_run3b_v2_b2d.json` | 10,468 | `05a09c16dda079064c63133479be16e645a81216e67745ddf33d0192d7bbe117` |
| Gate manifest, Bench2Drive | `{data.p3_renesas}/paddock-runs/gate-oct27/gate/smoke/run3b_v2-b2d/manifest.toml` | 604 | `2df9003f2ece27202bf3143bd6646dc1353a12fdb15ec50335d46e22c3646472` |
| Gate manifest, AlpaSim | `{data.p3_renesas}/paddock-runs/gate-oct27/gate/smoke/run3b_v2-alpasim/manifest.toml` | 575 | `4485195204785b17640be6647d67814172c8dcea9dab76ce91157af56e790b78` |

Gate runs (paddock run ids): navsim `20261120T045940-856e6e`, nuscenes `20261120T050004-387aa8`, kitscenes
`20261120T050356-48399a`, neuroncap `20261120T050645-7d6522`, neuroncap quick `20261120T045955-0d22fb`, alpasim `20261120T060100-20d216` (quick `20261120T055820-b7d92a`), bench2drive `20261120T081237-5e8510` (quick `20261120T080323-60e8d7`).

## Built data it trained on (from earlier work, not built here)

| Item | Where it sits at the training site | Note |
|---|---|---|
| NAVSIM images, 256 × 512 JPEG q95 copies | Phase 1's image root (`{data.paddock_data}`) | the frames H04's token cache used, re-encoded |
| H04's per-sample features (ego status, lidar2img, agent targets) | `{data.wajepa_tokens}` (H01's POINTERS) | read as is |
| LEAD inputs, boxes, ego-status sidecar, auxiliary targets | `{data.paddock_data}/lead_inputs_v2/` | `lead_inputs_v2`, `lead_aux` |
| Fact labels and pool pack, LEAD hit labels with `static.prop.dirtdebris*` as debris | `{data.paddock_data}/run3b/pack_v2/` (NAVSIM files link to `pack/`) | from `carla_full_v1` + `carla_full_v1_vru` and the NAVSIM pool |
| LEAD routes that overlap the Bench2Drive 220 (excluded from training by `scripts/run3b/r3b/data.py`) | `{runs}/run3a/facts/lead/carla/lead360_overlap.json` | 1,996,965 bytes, sha256 `3af110e2e6eae5b0f8f327fea2ae646aa01ff3cabefe3c5a59e77658bada0992`; 0 of the 7,905 training routes are in it |
