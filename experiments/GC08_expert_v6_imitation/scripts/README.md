# Scripts, as they ran

The chain behind Gate B, the probe and the expert-driven collection, from the training site's `ops/`
folders. Committed as run, with machine paths replaced by placeholders:

| Placeholder | Stands for |
|---|---|
| `{runs}` | gallops' runs root |
| `{paddock_runs}` | paddock's runs root |
| `{repos}` | the folder holding the checkouts |
| `{ncap_env}` | the neuro-ncap environment |

In the order they ran:

| Script | What it did |
|---|---|
| `run_gateB_v3.sh` | Gate B at T = 3.5 s, one process per scene, merged into `gateB_v3_T3.5.json` |
| `probe_v6.py` | the probe table in the account; `probe_v6.json` is its output |
| `export_expert_plans.py` | split that report into `expert_plans_v3/`, one file of plans and poses per episode |
| `run_collect_expert.sh` | the expert-driven collection, then lane references and v6 labels |
| `collect_expert.py` | one engine call per episode, writing which episode it is to the file the adapter reads |
| `collect.py`, `main_offset.py` | what `collect_expert.py` imports and calls, shared with the other collections |

The adapter read its plans through three entries in the site config, which is not committed:
`DRIVEJEPA_DRIVE = "expert"`, `DRIVEJEPA_EXPERT_DIR` (`{runs}/expert_plans_v3`) and
`DRIVEJEPA_EXPERT_CURRENT` (`expert_current.txt`, beside these scripts). The adapter is in
[../paddock/](../paddock/README.md).

**Two are not exactly as run:**

- `export_expert_plans.py` ran inline and was never saved, so it is recovered from that session's log.
  Re-run on 6 Nov 2026 it reproduced all 150 plan files byte for byte.
- `collect.py` is as it stood on 16 Oct. It was edited after the collection finished on 15 Oct, and
  the version that ran is not kept.

They are a record. Gallops' `make check` neither lints nor runs them, and they were not re-run here
beyond that one export.
