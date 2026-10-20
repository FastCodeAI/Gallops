"""Auxiliary prediction heads for Drive-JEPA: labels from nuScenes, the heads, one training step.

docs/drivejepa-aux-heads.md is the plan. The modules split by what they need:

    bev, targets   numpy only -- the label grid and the label rules, tested by `make check`
    nusc           nuScenes tables and map (numpy; the devkit only for the map)
    inputs         the model's inputs for a nuScenes keyframe (swerve_data's TickDataset, fed from tables)
    heads, step    torch -- the heads, their losses and one training step

Everything here runs inside Drive-JEPA's own interpreter (Python 3.9), so it stays valid
Python 3.9, and nothing is imported at package level.
"""
