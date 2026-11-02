# H09 — Shared gate runner

**Owner:** Naren · **Priority:** 9 of 11 · **Started:** 21 Oct 2026 · **Status:** running — the queue is
in use; warm simulators are not built (enabler)

The facts are in [experiment.toml](experiment.toml). This page transcribes the sources named at each
claim and adds no analysis.

## Why

Item 9 of the 21 Oct priority list (Renesas Progress and Plan, 22 Oct 2026): "A shared runner: one queue
that runs one job at a time, with warm neuroNCAP renderers and a persistent CARLA owned by the queue".
Why then: "Nothing coordinates between users today: GPU waits reserved nothing until #100's GPU lock
(merged 22 Oct), a crashing CARLA killed its user's other CARLAs, AlpaSim used fixed ports, and Navvrat
is not in the docker group". Cost: it "waits on the machine admin: Naren sends the request for a paddock
service account and a group for the three of us; then PR 3". It decides "whether runs from all three of
us can share the GPU and simulators safely". The page's "PR 3" section listed the admin request: an
account named paddock, a group for the three of us, docker for the account, and shared fast-disk folders.

## What was done

- **Safe beside other jobs** — paddock #100, merged 22 Oct (`ee62a6b`): a crashed Bench2Drive route kills
  only its own CARLA; one launch at a time per GPU across all users; AlpaSim runs started together use
  separate host ports.
- **The account and group** — on 3 Nov the machine lists a `paddock` account and a `paddock` group with
  members `naren`, `navrant` and `siddesh` (see [H03](../H03_shared_store_and_tags/README.md)). When the
  admin created them is not recorded. The `paddock` account is in the `docker` group; `navrant` still is
  not (`getent group docker`, 3 Nov).
- **The queue** — paddock #109, merged 26 Oct (`ef3f004`), `tools/gate_queue.py` and `docs/queue.md`: one
  queue of gate runs for the site; one worker, running as the `paddock` account under systemd, runs jobs
  one at a time with `tools.gate smoke`. `submit` checks each `--env` key against the model's
  `model.yaml` at the job's commit, copies the checkpoint into a shared store under its sha256, and pins
  the code to a commit on GitHub. The queue keeps its own baselines, made by the same account, and remakes
  them when a model's own settings change on `main`.
- **Simulators side by side** — #111 (26 Oct, `31e3714`) runs neuroNCAP, Bench2Drive and AlpaSim at once
  when the card holds them; #115 (26 Oct, `058a88d`) pins AlpaSim's Docker containers to the lane's cores.
  The worker re-reads its `[queue]` settings (lanes, ports) before each job.
- **In use** — read from the queue's job files on 3 Nov: 8 jobs between 23 and 27 Oct, 7 submitted by
  `naren` and 1 by `siddesh`. Drive-JEPA baseline jobs finished on 23 Oct (quick tier), 26 Oct (quick, at
  `main` with momentum on) and 26–27 Oct (closed tier: neuroNCAP 15 pairs, Bench2Drive 7 routes, AlpaSim
  10 scenes); two tests with the released checkpoint as the candidate ended HOLD; two jobs were cancelled;
  Siddesh's KITScenes candidate (encoder LoRA, generator and scorer pick head trained together, `--tier
  all`) ended FAIL on 27 Oct. #116's commit message records that the queue's AlpaSim baseline run first
  held 8 of 10 scenes, which led to the scene swaps described in [H02](../H02_smallest_valid_gate/README.md).

## What is not done

Warm simulators. "Simulators start cold for every benchmark run. Keeping the neuroNCAP renderers and
CARLA running between jobs is the next step, and changes how runs are made, so it comes with its own
check that no score moves" (paddock `docs/queue.md`, Limits). The same section notes that nothing is
cleaned up yet, and that a checkpoint is code (it is unpickled), so the group is the trust boundary.

## How it will be decided

It is an enabler: done when runs from all three of us share the GPU and simulators safely through one
queue, warm simulators included, with the check that no score moves when they are kept warm.
