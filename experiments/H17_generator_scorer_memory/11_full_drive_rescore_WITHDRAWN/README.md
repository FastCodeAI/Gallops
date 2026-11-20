# Full-drive re-score: withdrawn

**Withdrawn, because the renderer is not valid more than 4 m from the recorded path. Kept as it ran.**

This step re-scored every closed-loop drive to its last tick, instead of stopping the score window at the
first crash or at 4 m from the human path. Past 4 m the simulator renders a scene it has no recording of, so
what happens there is not evidence about the driver. The re-score's numbers are not used anywhere in the
record. The standard score, which stops at 4 m, is the one used.

| File | What it is |
|---|---|
| `PREDICTIONS.md` | the rule, written before any drive was re-scored |
| `REPORT.md` | the result as it was reported at the time |
| `s19_fulldrive_rescore.py`, `s19_fulldrive_tables.py`, `s19_watch_and_run.sh` | the scripts as they ran |
| `tables/` | the per-drive table, the reference model's re-score and the report's tables; the full per-drive json (101 KB) is in [POINTERS.md](../POINTERS.md) |
