"""Drive-JEPA retrained on Bench2Drive: the training behind two closed-loop results.

Both start from the released Drive-JEPA NAVSIM-v2 checkpoint (ViT-L) and retrain
parts of it on the Bench2Drive base set; the backbone is never trained.

  Split-v4  Bench2Drive 50 routes, mean driving score 68.1 (30 Sep 2026). Only
            the scorer head changes: ipad-base-start -> split-full -> split-v4aug.
  Final     73.6 (6 Oct 2026). A fine-tuned proposal generator (MTD-H teachers),
            a split head from split-v4aug retrained on the new candidates, an
            area head, and a progress guard in the policy (no training).

The stages, in order and with their arguments, are `recipes/drivejepa_b2d.toml`;
`gallops recipe drivejepa_b2d` prints them for this machine. What each stage
reads and how the port was verified is in experiments/GC04_bench2drive/README.md.

Every module here is a stage run as `python -m gallops.drivejepa_b2d.<stage>`,
and the generator stage runs inside Drive-JEPA's own Python 3.9 interpreter. So
nothing is imported at package level, and the modules that run there
(train_generator, batch_fix, provenance) stay valid Python 3.9 --
tests/test_recipe.py checks it.
"""
