# What ran in paddock

These files are copied from paddock commit `a263819`, at the paths they had
there. Paddock removed them because it holds only the released models'
checkpoints: the driver in paddock #121, the two run records in paddock
#132.

| Path | What it is |
|---|---|
| `tools/models/drivejepa/alpasim_driver/` | Drive-JEPA as a native AlpaSim driver, version 0.1.7. It serves only the retrained checkpoint: it checks the file's size and sha256 and refuses anything else. |
| `results/alpasim/drivejepa/legacy-drivejepa_017_20hz/` | its AlpaSim run on the benchmark's first 10 scenes, mean `scene_score` 0.9956. |
| `results/alpasim/drivejepa/legacy-drivejepa_017_20hz_30scenes/` | the same driver on the 30 scenes, 0.7697. |

Paddock PR #68 imported both runs under the model name `drivejepa`, so
paddock's leaderboard showed them as the released model's AlpaSim score. That
is why they left paddock.

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
