# What ran in paddock

The run records of this experiment's neuroNCAP runs, which paddock drove with
`DRIVEJEPA_CHECKPOINT` pointed at a swerve-finetune checkpoint. Paddock holds
only the released models' checkpoints, so their results live here.

| Runs | What they are |
|---|---|
| ten runs on scene 0103, 30 Sep to 8 Oct | the smokes, 6 episodes each, one per checkpoint; each `experiment.toml` result names its run |
| `20261008T021909-55974f` | s1 on the full set, 200 episodes, `ncap_score` 3.671 |
| `20261022T042520-207714`, `20261022T234613-207714`, `20261023T023837-207714` | the s1 run re-scored with PDM terms added, from its stored episodes; nothing was driven again, and the three agree |

Where they come from:

- The first eleven are the same files, byte for byte, as on paddock branch
  `record/drivejepa-camera-blind` (commit `7792144`), which is kept and not
  for merging.
- The three re-scores were never committed anywhere; they are copied from the
  paddock checkout on shakaal where they were written.

Only `run.json` and `metrics.json` are here. The per-episode records, node
logs and videos stay in `{runs}/neuroncap/drivejepa/<run id>/` at the training
site.

The s1 run is recorded `valid: true` only because it was made before paddock's
runner marked any `DRIVEJEPA_CHECKPOINT` override invalid. It is a finetune's
result, not the released model's, and it was never ranked.
