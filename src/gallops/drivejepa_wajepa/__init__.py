"""Drive-JEPA's planner on frozen WA-JEPA tokens, trained on NAVSIM navtrain: Run 2 and Run 3A.

The stages of recipes/drivejepa_wajepa.toml. Both runs trained inside a Drive-JEPA checkout with navsim's own
run_training.py; what differs from upstream is a set of files in navsim's agent packages. Those files are
kept here, exactly as they ran, under overlay/<run>/ with a manifest of their hashes, and `code` rebuilds the
tree a run trained in: upstream Drive-JEPA at a pinned commit with the overlay laid over it. Every other
stage runs inside that tree, in Drive-JEPA's own interpreter (Python 3.9), so nothing here imports navsim at
module level.

Lineage, data and what was verified: experiments/H01_four_camera_base/README.md.
"""
