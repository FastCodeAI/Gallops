<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H03 · One shared folder and group; every checkpoint tagged by weights sha256 with full metadata

**Naren** ·  · ⏳ running · 21 Oct 2026 · enabler

[← all experiments](../../README.md#enablers) · [the account](README.md) · [experiment.toml](experiment.toml)

**Priority:** 3

## Question

We could not run each other's models twice in the week of 19 Oct. One shared folder and group for the three of us, and a tag on every checkpoint (its weights sha256) with full metadata, so that anyone can run anyone's checkpoint and every result traces to one. Gallops' checkpoint file already records step, commit and training data; the group needs the machine admin's OK.

## What we learned

So far: the paddock account and a paddock group holding the three of us exist on shakaal; paddock's gate queue (#109, 26 Oct) copies every submitted checkpoint into a shared store named by its sha256. This registry pins every checkpoint by weights_sha256 with model, run, step, data and code, and `gallops experiments checkpoint` prints that block from a checkpoint directory. Not every past checkpoint has its digest recorded: those without one say so, and a gated checkpoint without one fails make check.

## Checks that decided something

- **The shared group**: the paddock account and group exist on shakaal; the group's members are naren, navrant and siddesh
- **A shared checkpoint store, by content (paddock #109)** (26 Oct): `gate_queue submit` copies a checkpoint file or directory into the shared store under a name made from its sha256, with the small files beside it (gallops' checkpoint.json), so a job measures what was submitted even if the original is overwritten → merged 26 Oct; anyone in the paddock group can submit
- **Checkpoint tags in this registry** (3 Nov): a [runs.checkpoint] carries weights_sha256 where it is known (64 hex digits, each claimed by one run only; required once a checkpoint is gated), model, train_run, step, file and location as a template; `gallops experiments checkpoint <dir>` prints the block from a sidecar and hashes the weights when the sidecar has no digest (with --hash it hashes anyway and reports a sidecar that disagrees) → the tagging half of H03; the shared group is the other half

## Evidence outside this repo

- The shared checkpoint store is paddock's gate-queue store ([queue] store in the site paddock.toml); paddock docs/queue.md says how it is set up.
