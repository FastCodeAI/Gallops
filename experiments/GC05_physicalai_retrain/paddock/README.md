# What ran in paddock

These files are copied from paddock commit `a263819`, at the paths they had
there. They leave paddock because it holds only the released models'
checkpoints: paddock #121 (merged 3 Nov 2026) removed the driver, and paddock
#132 removes the two run records. #132 was still open on 4 Nov 2026, so until
it merges both runs are also on paddock's `main`, under the model name
`drivejepa`.

| Path | What it is |
|---|---|
| `tools/models/drivejepa/alpasim_driver/` | Drive-JEPA as a native AlpaSim driver, version 0.1.7. It serves only the retrained checkpoint: it checks the file's size and sha256 and refuses anything else. |
| `results/alpasim/drivejepa/legacy-drivejepa_017_20hz/` | its AlpaSim run on the benchmark's first 10 scenes, mean `scene_score` 0.9956. |
| `results/alpasim/drivejepa/legacy-drivejepa_017_20hz_30scenes/` | the same driver on the 30 scenes, 0.7697. |

Paddock PR #68 imported both runs under the model name `drivejepa`, so
paddock's leaderboard shows them as the released model's AlpaSim score. That
is why paddock #132 takes them out.

One change from paddock: the driver README's links to paddock pages point at
those pages on GitHub, at `a263819`.

They are a record. Gallops' `make check` neither lints nor runs them.

## Running it again

The driver imports paddock's AlpaSim connection code
(`paddock.adapters.benchmarks.alpasim.scaffold`), which stays in paddock. Copy
`tools/` from here into a paddock checkout at the same path, and launch it as
its README says. Do not commit it there. Its tests need AlpaSim's interpreter;
from that checkout:

```bash
<alpasim python> -m pytest tools/models/drivejepa/alpasim_driver/tests -q
```

Not run here: this needs AlpaSim's environment, which this copy was not
checked against.
