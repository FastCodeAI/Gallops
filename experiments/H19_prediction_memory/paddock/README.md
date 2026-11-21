# What ran in paddock for H15's gate

H15's memory never went into paddock: paddock holds only the released models' checkpoints (paddock
rule of 4 Nov). For the gate it was added to a throwaway paddock worktree off paddock `origin/main`
`0fae534`, never committed there and never pushed. These are the files as they ran, at the paths they
had there.

| Path | What it is |
|---|---|
| `src/paddock/adapters/models/drivejepa/memory.py` | new: the memory layer transcribed from `gallops.memory.model` and `.align`, the checkpoint loader (refuses a directory without `checkpoint.json`, or whose weights' sha256 disagrees with it) and the per-path state kept by timestamp |
| `src/paddock/adapters/models/drivejepa/policy.py` | released `policy.py` plus the memory: hooks on the scorer add the nudge to the final logit inside the model's own forward, so momentum acts on the nudged score; `_catch_up` runs the scored frame's own chain at plan time; `/diagnostics` reports the memory |
| `src/paddock/adapters/models/drivejepa/model.yaml` | released `model.yaml` plus `DRIVEJEPA_MEMORY: ""` (empty: the released node, bit for bit) |
| `src/paddock/adapters/models/drivejepa/tests/test_memory.py` | the adapter's memory tests (14). Those that pin this copy to gallops' (same inputs, same scores to 1e-6) read `fixtures/memory_fixture.npz`. That array is not committed: `../scripts/make_memory_fixture.py` writes it, [POINTERS.md](../POINTERS.md) gives its sha256, and the tests skip when it is absent. That is the one change to this file since it ran |
| `drivejepa_memory.diff` | the change to paddock's tracked files against `0fae534` |

They are a record. Gallops' `make check` neither lints nor runs them.

`results/` holds the paddock run records and the gate's verdict files and comparison tables. The gate's
printed logs are not committed; [POINTERS.md](../POINTERS.md) lists them.

## Running them again

Check out paddock at `0fae534` and copy `src/` from here over it. Build the test array first:
`<drive-jepa python> experiments/H15_prediction_memory/scripts/make_memory_fixture.py
src/paddock/adapters/models/drivejepa/tests/fixtures/memory_fixture.npz`; without it, nine of the 14
memory tests skip and the run looks green. Then run the gate with
`--model drivejepa --env DRIVEJEPA_MEMORY=<checkpoint directory>` (the H15 report,
`MEMORY_H15_GATE.md`, has the exact commands and the site config). Momentum stays on, as `model.yaml`
sets it. Do not commit them there.
