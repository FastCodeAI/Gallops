# What ran in paddock

`drivejepa_ncapdump` collected every rendered episode GC08 trained on: released Drive-JEPA, plus a
per-tick dump of all 32 candidates. `DRIVEJEPA_DRIVE` says who drives the car while it collects:
`model`, `gt` (the logged human path), or `expert` (v6's plans), which made the expert-driven
collection that 1B and round 2 trained on.

It is here because paddock holds only the released models; anything trained lives in gallops. The
files sit at the paths they had in paddock.

| Path | What it is | Taken from |
|---|---|---|
| `src/paddock/adapters/models/drivejepa_ncapdump/` | the adapter | `policy.py` from the working tree of paddock branch `drivejepa-ncapdump-camfix`, where the `expert` mode was never committed; it was last changed before the collection began and not since. The rest are unchanged from `b1c5779` |
| `src/paddock/adapters/models/drivejepa/policy.py` | the adapter it builds on, with the camera-frame fix | `f4f5efc`, a local commit that was never pushed |
| `docs/models/drivejepa_ncapdump.md` | the model page, which predates the `expert` mode | `b1c5779` |

One change from paddock: the model page's link to a page not copied here points at it on GitHub, at
`b1c5779`.

They are a record. Gallops' `make check` neither lints nor runs them.

## Running them again

Paddock runs them; they are not kept there. Check out paddock at `b1c5779` (branch
`analysis/drivejepa-ncap-candidates`) and copy `src/` from here at the same paths. Paddock finds an
adapter by its folder. Do not commit them there. The `expert` mode also needs the plan files and the
pointer file that [../scripts/README.md](../scripts/README.md) describes.

Checked on 6 Nov 2026: after that copy, paddock's `load_models()` lists `drivejepa_ncapdump`, and the
adapter compiles. It was not run: that takes the renderer and the GPU.
