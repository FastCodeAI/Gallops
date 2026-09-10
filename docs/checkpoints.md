# Checkpoints, and the contract with paddock

**A checkpoint directory plus a `checkpoint.json` beside it is the entire
interface between gallops and paddock.** Neither repo imports the other. Neither
knows the other's layout. They share one directory root and one file format.

```
{weights}/<model>/<train_run>/step_<n>/
    checkpoint.json
    model.safetensors
```

`{weights}` is `[roots.weights]` from `gallops.toml`, and it must be **the same
directory** as paddock's `[roots.weights]`. That is the only coupling, and it is
site configuration on both sides, so neither repo contains a path.

## Why a sidecar rather than inspecting the weights

- **Only the trainer knows the step and its own commit.** A weights file does
  not carry the git sha of the code that produced it, or which split it saw.
  Those are the facts that make a result reproducible, and they are lost the
  moment they are not written down at the time.
- **Hashing 20 GB per evaluation run is not acceptable.** Paddock computes a run
  id *before* it starts a model node; making that wait on a checkpoint digest
  adds minutes to every run. Compute the digest once here, cache it in the
  sidecar.
- **It survives `rsync`.** Identity by path, size and mtime breaks the moment a
  checkpoint is copied between machines — which is exactly when two sites are
  about to disagree about what they measured.

## Writing one

```python
from gallops.checkpoint import Checkpoint, TrainedOn, write
from gallops.config import SiteConfig

cfg = SiteConfig.load()
directory = cfg.checkpoint_dir(model="gallops-v0", train_run="nuscenes-base", step=90_000)

save_weights(directory / "model.safetensors")
write(directory, Checkpoint(
    model="gallops-v0",
    step=90_000,
    train_git_sha=git_sha(),
    train_run="nuscenes-base",
    trained_on=(TrainedOn("nuscenes", "train"),),
    selection_split="val",
    params_m=71.4,
    precision="fp32",
))
```

**Write the sidecar in the same function that saves the weights.** A sidecar
written later is one that gets skipped when a job is killed, and an unlabelled
checkpoint is worth less than no checkpoint at all — because it looks usable.

`write()` writes to a temporary file and renames, so a reader never sees a
half-written sidecar. During a sweep there are readers running while a trainer
is writing.

## The fields, and what each one disables when absent

| field | required | what it is for |
|---|---|---|
| `model` | ✅ | family, not run. Becomes the paddock adapter name, so it must be stable across steps and safe as a directory name. |
| `step` | ✅ | orders checkpoints within a run. |
| `train_git_sha` | ✅ | without it a result cannot be reproduced even with the data and the seed. |
| `train_run` | ✅ | groups a sweep in a report. |
| `trained_on` | — | the contamination declaration. **Empty means "not recorded", not "trained on nothing"** — a reader cannot tell those apart, so write it down. |
| `selection_split` | — | which split chose this checkpoint. Reporting a headline from it is selection bias, not a result. |
| `params_m`, `macs_g` | — | absent means paddock cannot fill the deployment column for this model. |
| `precision` | — | pairs an fp32 and an int8 export into a paired delta. |
| `weights_sha256` | — | strongest identity, when someone has paid for it. |
| `notes` | — | free-form. A training run knows things we have not made fields for yet. |

`trained_on` dataset names must match paddock's `datasets.yaml` keys —
`nuscenes`, `kitscenes`, `navsim` — because that is what makes contamination
detectable at scoring time. A name that does not match is a check that silently
cannot run.

**`trained_on` is a self-declaration, not a measurement.** Nothing verifies it,
the same way nothing verifies a pinned sha in paddock's `repos.yaml`. Real
verification is sample-id overlap between the training manifest and what paddock
hashed into `sequences_sha256`, and it belongs in this repo, not that one.

## Why contamination matters immediately

Every number in paddock's matrix today is **zero-shot** — its nuScenes benchmark
states outright that none of the four models was trained on nuScenes.

nuScenes train is the obvious first training corpus. nuScenes val is the obvious
first evaluation, because it is what every published paper reports. **The moment
we do both, our row stops being comparable to the four above it** and the table
becomes quietly dishonest, unless something on the page says so.

That is why `trained_on` is in the contract from the first commit rather than
added later. Added later, it has to be back-filled by hand into result files
that paddock forbids hand-editing.

## Versioning

`SIDECAR_VERSION` bumps only when an existing field **changes meaning**. Adding
an optional field does not need a bump, because readers keep unknown keys in
`notes` rather than raising — a newer trainer must not break an older reader in
the middle of a sweep.

## Checking one

```bash
gallops show {weights}/gallops-v0/nuscenes-base/step_000090000
```

Read `trained_on` before spending GPU hours scoring a checkpoint. That is the
field that decides whether the result you are about to produce can go in the
table.
