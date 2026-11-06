"""Drive-JEPA's training rebuilt on Bench2Drive, stage by stage (experiment H16).

Unlike gallops.drivejepa_b2d (GC04), which retrains parts of the released model, this rebuilds what
Drive-JEPA's own two stages rest on, from Bench2Drive's data and scoring terms:

  frame cache   the 944 clips' front camera, crop `native`        build_cache, frames
  encoder       a LoRA by hide-and-guess, then merged             train_lora, merge_lora, ade
  vocabulary    k-means on the expert's 4 s paths                 build_vocab
  grade         Bench2Drive's penalties over 4 s, teacher index   scorer, grade, grade_all, teachers
  Stage 2       generator and scorer, from scratch or released    samples, train_stage2, export_agent

The stages, in order and with their arguments, are `recipes/drivejepa_b2d_retrain.toml`. Everything here runs
inside Drive-JEPA's own Python 3.9 interpreter, so nothing is imported at package level, and every module
stays valid Python 3.9: tests/test_drivejepa_b2d_retrain.py checks it.

These modules are the as-run scripts with imports and paths changed and the formatting this repository's lint
asks for. No stage has been run from this repository: experiments/H16_drivejepa_b2d_retrain/README.md says
what was compared with what ran.
"""
