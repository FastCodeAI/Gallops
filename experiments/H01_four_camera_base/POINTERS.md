# Pointers — what H01 produced that is not in the repository

Nothing below is committed: checkpoints by rule (`.gitignore`, `docs/checkpoints.md`), the cache for size. Locations are
templates: `{weights}`, `{runs}` and `{data.NAME}` come from `gallops.toml`. At the training site the checkpoint has not
been moved into the `{weights}` layout; the "where it sits" column says where it is.

## Weights

| Item | Template location | Where it sits at the training site | Bytes | sha256 |
|---|---|---|---:|---|
| The 4-camera planner, epoch 19 of 20 (`H01.four-camera-planner`), trainer format, no sidecar yet | `{weights}/wajepa_dj/run2/step_000000019/epoch_19.ckpt` | `{data.navsim.exp}/wajepa_dj_v2/epoch_ckpts_20261016_1748/epoch_19.ckpt` | 93,329,139 | `f8ce667cdcd7233173759dde4b1c1ed0f9c1921608192ec92878db1fc73cf5ad` |

## Built data

| Item | Built by | Location | Size | Identified by |
|---|---|---|---|---|
| Token cache: 103,288 navtrain samples, float16 tokens (4 × 512 × 1,024) plus light features and targets | `gallops.drivejepa_wajepa.dump_tokens` | `{data.wajepa_tokens}` | about 378 GiB | sha256 of the sorted `log/token` lines, one per sample, each ending in a newline: `d993bbde8fc0c492e2b9a3f18d66777daba84ce35a3ab8afb8091b4d2bb1c0e0`. The tokens themselves are not hashed; `verify-tokens` checks them against a live recompute |

The 4-camera planner's other inputs (the training metric cache, Drive-JEPA's teacher scores) are Drive-JEPA's own preprocessing, listed under
"The data" in [README.md](README.md).
