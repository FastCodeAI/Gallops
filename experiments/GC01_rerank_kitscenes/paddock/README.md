# What ran in paddock

These files are copied from paddock commit `a263819`, at the paths they had
there. Paddock removed them in paddock #121, because paddock holds only the
released models' checkpoints; anything trained lives in gallops.

| Path | What it is |
|---|---|
| `src/paddock/adapters/models/drivejepa_rerank/` | the adapter: released Drive-JEPA, with this selector choosing among its 32 candidates. `model.yaml` pins the checkpoint. |
| `tools/models/drivejepa/paddock_candidates.py` | dumps Drive-JEPA's 32 candidates per sample through paddock's own KITScenes or nuScenes input path. `gallops.data.rerank dataset` reads the dump. |
| `tools/models/drivejepa/rerank.py` | `report`: scores a checkpoint on a held-out dump. |
| `docs/models/drivejepa_rerank.md` | the model page: every number, the rebuild steps and the limits. |
| `results/` | the two runs paddock tracked: KITScenes `20260921T210621-d7ed08` (ADE@3s 0.973 m, 223 samples) and nuScenes `20261012T220930-6194bd` (ADE@3s 0.949 m, 5,119 samples). |

One change from paddock: the model page's link to a paddock page that is not
copied here points at that page on GitHub, at `a263819`.

They are a record. Gallops' `make check` neither lints nor runs them.

## Running them again

Paddock runs them; they are not kept there. Either check out paddock at
`a263819`, where all of them are, or copy `src/` and `tools/` from here into a
paddock checkout at the same paths. Paddock finds an adapter by its folder, so
`paddock run --model drivejepa_rerank --bench kitscenes` then works. Do not
commit them there, and copy the results they write under paddock's `results/`
back here.

Checked on 4 Nov 2026 against paddock #121's branch, where the adapter is gone:
after the copy, `paddock list models` lists `drivejepa_rerank`, the adapter's
16 tests pass in paddock's interpreter, and `rerank.py report --help` runs.
`paddock_candidates.py` runs in Drive-JEPA's own environment (Python 3.9), as
its docstring says; it was not run here.
