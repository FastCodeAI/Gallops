"""Drive-JEPA retrained on Bench2Drive: the training behind Split-v4.

Split-v4 scores a mean driving score of 68.1 on the 50-route Bench2Drive set
(30 Sep 2026). It is the released Drive-JEPA NAVSIM-v2 checkpoint (ViT-L) with
only the scorer head retrained, in three stages, each starting from the one
before: ipad-base-start -> split-full -> split-v4aug. The backbone and the
proposal generator are untouched, so the 32 proposals are the released model's
and only the pick changes.

The stages, in order and with their arguments, are `recipes/drivejepa_b2d.toml`;
`gallops recipe drivejepa_b2d` prints them for this machine. What each stage
reads and how the port was verified is in docs/drivejepa_b2d.md.

Every module here is a stage run as `python -m gallops.drivejepa_b2d.<stage>`,
so nothing is imported at package level.
"""
