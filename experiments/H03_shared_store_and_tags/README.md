# H03 — Shared store and checkpoint tags

**Owner:** Naren (setup), all · **Priority:** 3 of 11 · **Started:** 21 Oct 2026 · **Status:** running (enabler)

The facts are in [experiment.toml](experiment.toml). This page transcribes the sources named at each
claim and adds no analysis.

## Why

Item 3 of the 21 Oct priority list: "One shared folder and group for the three of us; every checkpoint
gets a tag (weights sha256) and full metadata". Why then: "We could not run each other's models twice
this week; gallops' checkpoint file already records step, commit and training data". Cost: "Hours; the
group needs the machine admin's OK". What it decides: "Anyone can run anyone's checkpoint, and every
result traces to one" (Renesas Progress and Plan, 22 Oct 2026, "Next steps", item 3).

## What is done

- **The group.** On 3 Nov the machine's group table lists a `paddock` account and a `paddock` group
  whose members are `naren`, `navrant` and `siddesh` (read with `getent`). When it was created, and by
  whom, is not recorded in either repo. paddock's `docs/queue.md` setup steps say they were followed
  "in the order it worked on shakaal" (commit `de335b4`, 24 Oct), and the queue's first job is dated 23
  Oct (see [H09](../H09_shared_runner/README.md)).
- **A shared checkpoint store, by content.** paddock #109 (merged 26 Oct, `ef3f004`): `gate_queue
  submit` copies the checkpoint into a shared store "under a name made from its content (its sha256)",
  with the small files beside it, such as gallops' `checkpoint.json`, so the job measures what was
  submitted even if the original is overwritten (paddock `docs/queue.md`). This covers checkpoints sent
  to the gate, not every checkpoint trained.
- **The tag and its metadata, in gallops.** This registry: one `experiment.toml` per experiment, each
  checkpoint pinned by `weights_sha256` with its model, training run, step, file, location (as a
  template, never a machine path), training data and code; `gallops experiments check` refuses a digest
  claimed by two runs; `gallops experiments checkpoint <step dir>` prints the block, hashing the weights
  every time and refusing a sidecar that disagrees (`src/gallops/experiments.py`, `src/gallops/cli.py`,
  `checkpoint.sha256_file` in `src/gallops/checkpoint.py`). Past checkpoints whose digest was never
  recorded say so; a checkpoint that goes through the gate must carry one.

## What is next

Merge the registry and record each existing checkpoint in it. Not found in any source: whether
checkpoints outside the gate queue (training outputs) sit in a folder the whole group can read.

## How it will be decided

It is an enabler: done when anyone in the group can run anyone's checkpoint and every reported result
names one by its weights sha256 (the item's own "Decides" column).
