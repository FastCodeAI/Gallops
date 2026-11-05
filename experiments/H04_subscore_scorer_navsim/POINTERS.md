# Pointers — what H04 produced that is not in the repository

Nothing below is committed: checkpoints by rule (`.gitignore`, `docs/checkpoints.md`), the rest for size. Locations are
templates: `{weights}`, `{runs}` and `{data.NAME}` come from `gallops.toml`. At the training site these have not been moved
into the template layout; the "where it sits" column says where each one is.

## Weights

| Item | Template location | Where it sits at the training site | Bytes | sha256 |
|---|---|---|---:|---|
| Agent decoder D, 300 slots × 3 layers, epoch 11 of 12 (`H04.agent-decoder`) | `{runs}/drivejepa_wajepa/agent_decoder/ckpts_D_q300_l3/epoch_11.pt` | `{data.navsim.exp}/agent_decoder/ckpts_23oct_D_q300_l3/epoch_11.pt` | 17,042,431 | `e887dcda20a74a043e3a7552fe27d7f34596b7a25e5c97a545b3b2421a9bae2a` |
| Run 3A, epoch 13 of 20 (`H04.run3a`), trainer format, no sidecar yet | `{weights}/wajepa_dj_run3a/run3a/step_000000013/epoch_13.ckpt` | `{data.navsim.exp}/run3a/epoch_ckpts_20261024_1734/epoch_13.ckpt` | 158,899,987 | `92185265cee95383dd62fbba760d3a5c3e4d8e8d09de40cbb8f7523190b2cf24` |

## Built data

| Item | Built by | Template location | Bytes | sha256 |
|---|---|---|---:|---|
| Agents of every cached sample | `gallops.drivejepa_wajepa.extract_agents` | `{runs}/drivejepa_wajepa/agent_decoder/navtrain_agents.pkl` | 224,308,243 | `a007bade184f5324a95b3521e85357b75a7ba5cabd2349960e479dbb6311d96d` |
| Lane centrelines, Singapore | `gallops.drivejepa_wajepa.lane_centrelines` | `{runs}/drivejepa_wajepa/run3a/lane_centrelines/sg-one-north.pkl` | 1,816,686 | `3140b80ebbdc143c7776a8a2b045a2749fa51526b0502f8e15d84a5f1aa3db6e` |
| Lane centrelines, Boston | same | `.../us-ma-boston.pkl` | 2,301,828 | `10ed4e7ac8c98570f0f04cbea7a2b58b9f30fc18ab304255a21f1a18976613ab` |
| Lane centrelines, Las Vegas | same | `.../us-nv-las-vegas-strip.pkl` | 4,088,752 | `125afb9fa266308d7fc5079432474acbf1ab6873544b177fbcdac3f3fe842703` |
| Lane centrelines, Pittsburgh | same | `.../us-pa-pittsburgh-hazelwood.pkl` | 891,689 | `d95e2a0102d7e2c98e1eeff7286e2812fa791e3214842389c4fe9d3fe79e70b1` |
| Teacher list, 103,288 samples, 1,192 log folders | `gallops.drivejepa_wajepa.teacher_rearok` | beside the metric cache, `anchors_scores_index_v2_rearok_lane` | 857,681,920 | not hashed: one small file per sample |
| Agent targets, `agent_targets.gz` in every sample's cache folder | `gallops.drivejepa_wajepa.agent_targets` | `{data.wajepa_tokens}` | inside the token cache (H01's POINTERS) | not hashed |
