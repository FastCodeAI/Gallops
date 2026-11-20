# The simulator adapter that drove the memory model

The memory model drove AlpaSim through this adapter in all three closed-loop steps (09, 10, 12). It is a copy
of the starting model's own adapter (`drivejepa018bp06`) with the memory added. It never went into a paddock
branch: it ran from a personal working tree, `{paddock_sid}/Alpasim/adapters/drivejepamemory/`. These are
its files as they ran.

**As they ran.** Before every drive the queue checked the adapter against its build freeze,
`drivejepamemory/launch/tables/build_memory.sha256`. On 21 Nov 2026 every file it lists still matched, so
the files copied here are the ones that drove. Machine paths in them were replaced by placeholders
(the record's README lists them; `../AS_RUN.sha256` holds each changed file's hash as it was).

| Path | What it is |
|---|---|
| `drivejepamemory/memory.py` | the memory: 16 slots, updated after each pick, kept per drive session, set to its start state when a session starts or closes, or when two calls are not about 0.5 s apart in simulated time |
| `drivejepamemory/planner.py`, `bridge.py`, `constants.py` | the starting model's planner, bridge and constants with the memory wired in; `constants.py` pins the starting model by sha256, and the runner pins the memory checkpoint |
| `drivejepamemory/batch_fix.py`, `divider.py`, `momentum.py` | unchanged from the starting model's adapter |
| `drivejepamemory/launch/` | the runner, the model server and the extra gates for one drive; `tables/` holds the build freeze and the three adapter proofs run before the first drive |
| `drivejepamemory_vs_drivejepa018bp06.diff` | the change against the starting model's adapter, `launch/` left out. It replaces `launch/tables/adapter_018bp06_to_memory.diff`, which was written before the last change to `bridge.py` and misses it |

**Not copied: four links** to the shared Drive-JEPA adapter, which both models use unchanged: `src/` (the
vendored upstream model code), `__init__.py`, `fake_model.py` and `real_model.py`, all under
`{paddock_sid}/Alpasim/adapters/drivejepa/`.

They are a record. Gallops' `make check` neither lints nor runs them.
