# Pointers — what GC07 produced that is not in the repository

Nothing below is committed: checkpoints by rule (`docs/checkpoints.md`), and the rest because it is
built data or a run record that lives on the training site's store. Each row gives a location, a size
and a sha256, so a copy can be recognised wherever it ends up.

**Locations are templates, not paths.** `{weights}` and `{runs}` are `[roots.weights]` and
`[roots.runs]` from `gallops.toml`. Paddock's run folders are given relative to paddock's own runs
root.

## Weights

The seed pins are copied from `experiment.toml`; the checkpoint is the five seeds together, pinned by
their digest there. All of them sit in `{weights}/drivejepa_subscore/sighted-v1/step_000000020/`.

| File | Bytes | sha256 |
|---|---:|---|
| `seed_0.pt` | 8,051,353 | `690ae3d6709da2dc7ce528d1cbeb44e8db0d57eaeb38d6cc0716e18246c8cf35` |
| `seed_1.pt` | 8,051,353 | `47a72b5d2c4314bd7387df5e849519248723b3c2712e3f9748ea892ecc2eccbe` |
| `seed_2.pt` | 8,051,353 | `5dfaff3e3023eb9e9d6761098ebd0116c515a30d9091789e3cf3ec6bf7f9c7e0` |
| `seed_3.pt` | 8,051,353 | `27125b1e205abd8be14e9a03f25e95028be8b62fd8839cd387f863fd6f3687f5` |
| `seed_4.pt` | 8,051,353 | `1f11144e697efed38dfe50390141ee94b8f3f8dfe056d7008921daf1cbe9429c` |
| `scorer.json` | 1,242 | `107df2130731a019548e338cb10cc48fac02876ffc4646d0a45038a124bfea90` |

## Training inputs

**These are not hashed:** they are 20 GB and 14 GB, and the sidecar's notes name them.

| Location | What it is | Files | Bytes |
|---|---|---:|---:|
| `dj_cache/v4/`, under the site's cache root | scorer cache v4 | 1,207 | 20,937,536,487 |
| `{runs}/collect_r0cam/` | the seeing round-0 collection | 9,302 | 13,981,390,428 |

## Paddock runs

Committed to no repository, which is what the results' `no-record` flag says.

| Run | What it is | `samples.jsonl` bytes | sha256 |
|---|---|---:|---|
| `neuroncap/drivejepa_subscore/20261013T202318-c43ac5` | 0103, relative rule, the account's table | 2,219 | `7887998e6104c17cec0d8dc2f2845a8fc5aabc4e5b72e0b3a8178c3605733cff` |
| `neuroncap/drivejepa_subscore/20261013T205827-fff68a` | 0103, `DRIVEJEPA_SUBSCORE_RULE=absolute` | 2,219 | `ec2825340076b5680ec4c257d0e1da8d253337eb1e736c38508bf8c71a7c92e0` |

## Paddock code

Neither repository imports the other, so what ran on paddock's side has to be named here. Without
these branches another site can train this scorer but cannot load it.

| Code | Where it is | Pin |
|---|---|---|
| the `drivejepa_subscore` adapter | branch `model/drivejepa-subscore`, not on `main` or any PR | `027bc9f82996d0ee4f824eee76425d6ef973e09c` |
| the adapter and its result records | branch `record/drivejepa-camera-blind`, not on `main` or any PR | `779214431d452510743647d0d60d0647dbf79e91` |
