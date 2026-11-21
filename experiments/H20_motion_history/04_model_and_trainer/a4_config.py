"""Approach 4: every location in one place. Change STAGE (the side-file root) when the side file moves to the NAS; nothing else reads it.

STAGE        the 2026-11-17 side file (PREREG_APPROACH4.md §1): <STAGE>/ego_history/<clip>.npz, 4,015 training clips
PICK_HISTORY history of the 2,000 picking-set moments (same builder; build_picking_history.py)
TOKENS       local copy of approach 3's training token cache (copy_tokens.py; Sid's exception for this lane, 2026-11-17)
PICK_CACHE   approach 3's picking-slice token cache (84 clip files, read whole: only the checks read it)
RUNS         training outputs (the NAS refuses writes from shakaal since 2026-11-16)
"""
import os
from pathlib import Path

STAGE = Path(os.environ.get("A4_STAGE", "{staging}"))
SIDE = STAGE / "ego_history"
PICK_HISTORY = STAGE / "ego_history_picking"
TOKENS = Path("{approach4_store}/tokens")
PICK_CACHE = Path("{experiment_store_h17}/cache/picking")
RUNS = Path("{approach4_store}/runs")
