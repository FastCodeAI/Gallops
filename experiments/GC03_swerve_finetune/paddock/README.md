# What ran in paddock

The run records of this experiment's neuroNCAP runs, which paddock drove with
`DRIVEJEPA_CHECKPOINT` pointed at a swerve-finetune checkpoint. Paddock holds
only the released models' checkpoints, so their results live here.

| Runs | What they are |
|---|---|
| ten runs on scene 0103, 30 Sep to 8 Oct | the smokes, 6 episodes each, one per checkpoint; each `experiment.toml` result names its run |
| `20261008T021909-55974f` | s1 on the full set, 200 episodes, `ncap_score` 3.671 |
| `20261022T042520-207714`, `20261022T234613-207714`, `20261023T023837-207714` | the s1 run re-scored with PDM terms added, from its stored episodes; nothing was driven again. The three agree on `ncap_score`, collisions and impact speed, but not on the PDM terms (below) |

Where they come from:

- The first eleven are the same files, byte for byte, as on paddock branch
  `record/drivejepa-camera-blind` (commit `7792144`), which is kept and not
  for merging.
- The three re-scores were never committed anywhere; they are copied from the
  paddock checkout on shakaal where they were written.

The three re-scores' PDM terms are not the same:

| PDM term | `042520`, `234613` | `023837` |
|---|---|---|
| `pdm_nc` | 0.7368 | 0.7228 |
| `pdm_dac` | 0.9572 | 0.9564 |
| `pdm_ttc` | 0.6562 | 0.6456 |
| `pdm_ep` | 0.5571 | 0.5647 |
| `pdm_comfort` | 0.8138 | 0.9419 |
| `pdm_style_score` | 0.5857 | 0.5962 |

Each `run.json` names the paddock commit it was scored at: `5f30732` with
uncommitted changes, `3baa7be` clean, and `f15cee7` with uncommitted changes.
`5f30732` predates the commit of neuroNCAP's PDM terms (its tree held them
uncommitted), and its numbers equal `3baa7be`'s exactly. The PDM code
(`src/paddock/metrics/`, `neuroncap/pdm_terms.py`) is identical at `3baa7be`
and `f15cee7`, so `023837` differs because of `f15cee7`'s uncommitted changes,
and those cannot be rebuilt. Quote `20261022T234613-207714`: it is the only
re-score made from a clean commit.
Nothing in gallops quotes a PDM term from these runs yet.

Only `run.json` and `metrics.json` are here. The per-episode records, node
logs and videos stay in `{runs}/neuroncap/drivejepa/<run id>/` at the training
site.

The s1 run is recorded `valid: true` only because it was made before paddock's
runner marked any `DRIVEJEPA_CHECKPOINT` override invalid. It is a finetune's
result, not the released model's, and it was never ranked.
