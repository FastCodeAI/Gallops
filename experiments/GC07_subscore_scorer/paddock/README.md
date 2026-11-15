# What ran in paddock

`drivejepa_subscore` is the adapter that loads this experiment's weights: Drive-JEPA with the camera
fixed, 32 inserted swerves, and the scorer `gallops.train.scorer_train` writes. Its `build_scorer`
must match `subscorer.layout()`, which `tests/test_subscorer.py` pins.

It is here because paddock holds only the released models; anything trained lives in gallops. The
files sit at the paths they had in paddock, at `027bc9f` on paddock's branch
`model/drivejepa-subscore`, unchanged.

| Path | What it is |
|---|---|
| `src/paddock/adapters/models/drivejepa_subscore/` | the adapter |
| `docs/models/drivejepa_subscore.md` | the model page |

`record/drivejepa-camera-blind` (`7792144`) holds an earlier version. This one is what GC07's runs
used: only it has `DRIVEJEPA_SUBSCORE_RULE`, which run `20261013T205827-fff68a` sets.

They are a record. Gallops' `make check` neither lints nor runs them.

## Running them again

Paddock runs them; they are not kept there. Check out paddock at `027bc9f`, where all of them are,
or copy `src/` from here at the same paths into a checkout that has the camera-frame fix. Paddock
finds an adapter by its folder. Do not commit them there.

Checked on 6 Nov 2026: after that copy onto `027bc9f`, paddock's `load_models()` lists
`drivejepa_subscore`, and the adapter compiles. It was not run: that takes the renderer and the GPU.
