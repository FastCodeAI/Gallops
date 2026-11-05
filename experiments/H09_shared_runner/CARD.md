<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H09 · A shared runner: one queue, one job at a time, with warm neuroNCAP renderers and a persistent CARLA

**Naren** · NAVSIM, KITScenes, nuScenes, neuroNCAP, Bench2Drive, AlpaSim · ⏳ running · 21 Oct 2026 · enabler

[← all experiments](../../README.md#enablers) · [the account](README.md) · [experiment.toml](experiment.toml)

**Priority:** 9

## Question

Can runs from all three of us share the GPU and simulators safely? Nothing coordinated between users: GPU waits reserved nothing until #100's GPU lock (merged 22 Oct), a crashing CARLA killed its user's other CARLAs, AlpaSim used fixed ports, and Navvrat was not in the docker group. The plan: one queue that runs one job at a time, each job naming a checkpoint in a shared store, with warm neuroNCAP renderers and a persistent CARLA owned by the queue. It waited on the machine admin for a paddock service account and a group for the three of us.

## What we learned

So far: the queue landed (paddock #109, 26 Oct) and runs as the paddock account, one job at a time, with each submitted checkpoint copied into a shared store by its sha256 and the code pinned to a pushed commit; by 27 Oct it had made Drive-JEPA's quick- and closed-tier baselines and gated a teammate's candidate. Not built: warm simulators. Every benchmark run still starts its simulator cold.

## Checks that decided something

- **Safe beside other jobs (paddock #100)** (22 Oct): a crashed Bench2Drive route kills only its own CARLA; one launch at a time per GPU across all users (a lock over the last free-memory check and the model load); AlpaSim runs started together use separate host ports → merged 22 Oct (ee62a6b)
- **The gate queue (paddock #109)** (26 Oct): one queue for the site; one worker, running as the paddock account, runs jobs one at a time with tools.gate smoke; submit checks each --env key against model.yaml at the job's commit, copies the checkpoint into a shared store named by its sha256, and pins the code to a commit on GitHub; the queue keeps its own baselines, remade when a model's settings change on main → merged 26 Oct (ef3f004)
- **The queue in use, 23–27 Oct** (27 Oct): 8 jobs (7 by naren, 1 by siddesh): Drive-JEPA baseline jobs done on 23 Oct (quick), 26 Oct (quick, at main with momentum on) and 26–27 Oct (closed: neuroNCAP 15 pairs, Bench2Drive 7 routes, AlpaSim 10 scenes); two released-as-candidate tests ended HOLD; two jobs cancelled; Siddesh's KITScenes candidate (encoder LoRA + generator + scorer pick head, --tier all) ended FAIL on 27 Oct
- **Simulators side by side (paddock #111, #115)** (26 Oct): neuroNCAP, Bench2Drive and AlpaSim run at once when the card holds them, each counted against GPU memory until its lane ends; AlpaSim's Docker containers are pinned to the lane's cores → merged 26 Oct (31e3714, 058a88d); the worker re-reads [queue] (lanes, ports) before each job
- **Warm simulators**: not built: simulators start cold for every benchmark run; keeping neuroNCAP renderers and CARLA up between jobs is the next step, and changes how runs are made, so it comes with its own check that no score moves

## Evidence outside this repo

- Queue jobs: job.json and result.json under the queue root's jobs/ ([queue] root in the site paddock.toml), on shakaal.
