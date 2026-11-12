# How Paddock's regression gate works

Teaching notes written from the code at Paddock **origin/main 9585f81** (2026-10-27). Every `path:line` is relative
to the Paddock repository at that commit. Facts the repository does not state are marked **NOT STATED**, and facts
that could not be checked are marked **UNVERIFIED**. This file describes; it recommends nothing.

---

## D1. What a regression gate is, and the end-to-end flow

A **regression** is a new version that is worse than the old one. A **regression gate** is an automatic check that
runs before a new model checkpoint (or a code change) replaces an old one.

Paddock's gate answers one question: "does the candidate lose to its baseline anywhere, by more than chance?"
(`docs/gate.md:3-7`).
- The **candidate** is the new checkpoint.
- The **baseline** is the stored result of the old one, here the released Drive-JEPA weights.

The gate never runs a model itself except through Paddock's normal `paddock run` command. It never edits a result. It
reads the per-sample records that two runs already wrote and compares them sample by sample (`docs/gate.md:5-7`).
It exists because a single average can hide a loss, and two averages can differ by chance. The gate pairs samples and
asks whether the difference is larger than the noise.

### The flow, one step at a time

1. **Command.** Either submit to the shared queue (`python -m tools.gate_queue submit --model drivejepa --env
   DRIVEJEPA_CHECKPOINT=<file>`, `docs/queue.md:16-23`), or run by hand (`python -m tools.gate smoke --model drivejepa
   --env DRIVEJEPA_CHECKPOINT=<file> --baseline <baselines.toml> [--tier quick|closed|all] --execute`,
   `docs/gate.md:273-279`).
   - Without `--execute`, the gate only prints the plan (`docs/gate.md:326-328`).
   - The queue's worker runs the same `tools.gate smoke --execute` as the `paddock` account, from its own checkout
     of the submitted commit (`docs/queue.md:91-105`).
2. **Derived site config.** A "site config" (`paddock.toml`) says where data, environments and outputs live on this
   machine. The gate writes a modified copy for the candidate, `<state>/paddock.toml` (`tools/gate.py:1393-1441`).
   It changes:
   - the `--env` values: merged into `[models.<model>.env]` and into every per-benchmark env table, so the
     checkpoint path reaches the model and is recorded;
   - AlpaSim's scene set: pinned to `public_2601_gate10`;
   - NAVSIM: given 2 model nodes;
   - neuroNCAP: its renderer moved to `--renderer-port` and its debug views turned off;
   - nuScenes: its index cache shared.
   None of these is in the run config, so none changes what is measured (`tools/gate.py:1397-1408`).
3. **One `paddock run` per benchmark subset.** Each subset becomes `taskset -c <cores> nice <python> -m paddock.cli
   run --bench <b> --model drivejepa <subset flags>`, with the subset flags from `smoke_args`
   (`tools/gate.py:1278-1326`; `docs/gate.md:326-328`).
   - Before each launch the gate waits until `nvidia-smi` shows enough free GPU memory: the model's `vram_gb`, plus
     the simulator's share, plus 2 GB (`tools/gate.py:1146-1168`). For Drive-JEPA that is 14 GB open loop and
     22 GB closed loop (31 GB for AlpaSim).
   - Lanes (`--cores`, repeatable) let several subsets of one tier run side by side (`docs/gate.md:281-297`).
4. **Per-sample records.** Each run writes:
   - `samples.jsonl`, one JSON line per sample, under `{roots.runs}/<bench>/<model>/<run_id>/`;
   - `run.json` (settings, provenance, validity) in the running checkout's `results/<bench>/<model>/<run_id>/`
     (`tools/gate.py:24-27`; `docs/gate.md:517-520`).
5. **Pairing.** The candidate's records are joined with the baseline's on `sample_id`. Only samples both runs hold
   count (`tools/gate.py:701-704`).
6. **Verdicts.** Each benchmark pair gets one verdict from a t-test whose noise estimate is grouped by the
   benchmark's independent unit (D3), plus override verdicts when the comparison is not trustworthy
   (`tools/gate.py:639-806`).
7. **Overall result.** FAIL, SHIP or HOLD (`tools/gate.py:812-834`; D4).
8. **Manifest.** The gate writes `<state>/manifest.toml`, pairing each finished candidate run with its baseline, and
   scores it with `check` (`docs/gate.md:348-349`; `tools/gate.py:1603-1631`).

### Where each output lands

| output | where | source |
|---|---|---|
| gate state folder `<state>` | `{roots.runs}/gate/smoke/<label>/`, where label = model + a digest of `--env` | `docs/gate.md:351-352`; `tools/gate.py:1840-1843` |
| derived site config | `<state>/paddock.toml` | `tools/gate.py:1843` |
| progress and lane PIDs | `<state>/state.json` | `docs/gate.md:317-318` |
| one log per subset | `<state>/<bench>.log` | `docs/gate.md:346-347` |
| the pairing | `<state>/manifest.toml` | `docs/gate.md:348-349` |
| per-sample records | `{roots.runs}/<bench>/<model>/<run_id>/samples.jsonl` | `tools/gate.py:24-27` |
| run settings | `<checkout>/results/<bench>/<model>/<run_id>/run.json` (not to be committed) | `docs/gate.md:517-520` |
| verdict table | printed to stdout | `tools/gate.py:862-920` |
| queue job (if queued) | `{paddock_site}/queue/jobs/<job>/`: `job.json`, `gate.log`, `result.json`, `verdict.json`, `baseline.toml`, `smoke/` | `docs/queue.md:49-60` |

On shakaal, the queue's `{roots.runs}` is `{paddock_runs}` (`{paddock_site}/paddock.toml:11`).

---

## D2. The benchmarks

Some terms first:
- **Open loop:** the model is shown recorded sensor data and asked for a plan. The plan is scored against what the
  human driver did. The car does **not** follow the model's plan, so a mistake never changes what the model sees next.
- **Closed loop:** a simulator moves the car according to the model's plan. Mistakes change what the model sees next,
  and can lead to collisions.

The gate has **subsets**: fixed, tracked lists of samples per benchmark in `tools/gate_smoke/<name>.toml`, each drawn
once and headed by how it was drawn (`docs/gate.md:211-214`).
- Since 2026-10-22, most subsets are drawn by `tools/gate_subsets.py`.
- Each benchmark is split into groups that change how a model must drive (turn, speed band, city, crash type). Every
  group gets the same share, chosen at random within the group (`docs/gate.md:216-224`).
- So the gate's mean weighs rare driving more than the full benchmark does. It is "a test for what broke, not an
  estimate of the full score" (`docs/gate.md:223-224`).

**What every benchmark shares on the model side** (Drive-JEPA through Paddock):
- **Cameras:** four, CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT and CAM_BACK
  (`src/paddock/adapters/models/drivejepa/policy.py:76-82`; `model.yaml:42`).
- **History:** frames picked by timestamp, about 0.5 s apart (`policy.py:251-263`). Two front frames 0.5 s apart feed
  the encoder, and the other stage reads the four cameras at the current frame (`model.yaml:57-64`).
- **Image geometry:** each camera is cropped and resized onto the 64° OpenScene camera the checkpoint was trained on
  (`model.yaml:31-39`; `policy.py:224-249`), then shrunk to 512×256 (`model.yaml:56`).
- **Ego status:** sideways speed and sideways acceleration are always sent as 0: `ego_velocity=[speed, 0]`,
  `ego_acceleration=[accel, 0]` (`policy.py:303-304, 383-384`).
- **The plan:** 8 poses 0.5 s apart, 4 s in total (`model.yaml:65-67`).
- **Momentum-aware selection** is on (`model.yaml:24-30`). It nudges the scorer towards the plan picked 0.5 s
  earlier, and acts only where the node has such a plan, which means closed loop. Open-loop samples arrive through
  `/infer_sequence`, which resets the node. A previous plan is used only if it is 0.45–0.55 s old
  (`momentum.py:43-57`).

### KITScenes (open loop): subset `kitscenes`, quick tier

- **Dataset.**
  - Paddock's docs call it "European driving from a roof pod" (`docs/benchmarks/kitscenes.md:69-70`). The full
    release is about 4.2 TB over 1,006 sequences (`datasets.yaml:23-25`).
  - The organisation, city and year are **NOT STATED** in the repository.
  - Sensors: a six-camera ring (`docs/benchmarks/kitscenes.md:213, 227-234`). The ring front camera is 3504×2272 and
    87°×63° (`kitscenes.md:303-305`). There is a top lidar used as the base frame (`kitscenes/harness.py:575-583`).
- **What the gate uses.** All 223 keyframes of the shared sample list (`kitscenes_e2e_final.pkl`, from the site's
  `[benchmarks.kitscenes] sample_list`), over 140 scenes. Nothing is drawn.
  - The list's own split puts them in two groups: `val` (186) and `overlap_train_val` (37). The second group is
    "places that are also in KITScenes' training data".
  - The verdict reads only `val`, 117 scenes. The 37 are compared alone in a note (`tools/gate_smoke/kitscenes.toml`
    header and `decide_on`; `tools/gate.py:937-975`).
- **What the model receives.**
  - Each sample carries 48 history frames at 10 Hz plus the keyframe (`kitscenes/harness.py:36-40`). The node picks
    frames 0.5 s apart from them.
  - **Driving command:** `command_source` defaults to `straight` on every frame (`kitscenes/harness.py:93-107,
    393-420`). With `gt`, the keyframe gets the list's one-hot, which tells 39 of 223 keyframes to turn
    (`kitscenes.md:182-186`). That one-hot is a ±5 m lateral offset 4 s ahead.
  - **Speed:** from differenced poses (`kitscenes/loader.py:99-104`).
  - **Acceleration:** 0. `canbus = np.zeros(18)` with only the velocity set (`kitscenes/harness.py:559-560`).
  - The model is asked once per keyframe.
- **Primary metric: ADE (3 s).** The average displacement error is the mean straight-line distance between predicted
  and recorded positions over the first 6 waypoints, 0.5–3.0 s:
  ADE = (1/6) Σₖ₌₁⁶ ‖p̂ₖ − pₖ‖₂, in metres (`kitscenes/metrics.py:47-50`; `kitscenes/harness.py:41-42`).
  **Lower is better** (`src/paddock/results/report.py:17`).
- **Other metrics:** FDE at 1, 2 and 3 s (the distance at that one time), longitudinal and lateral mean absolute
  error, inference time, `pdm_ep`, `pdm_comfort` (`kitscenes/benchmark.yaml:5`).
- **Unit:** the scene (`tools/gate.py:243-247`).
- **Baseline now:** queue run `20261023T195836-e90d98`, released weights, ADE (3 s) **2.521 m** over all 223 (mean
  of its `samples.jsonl`).

### nuScenes (open loop): subset `nuscenes`, quick tier

- **Dataset.** Recorded in Boston seaport and Singapore, with 6 cameras (`docs/benchmarks/navsim.md:45-46`) at
  1600×900 (`policy.py:35`). The organisation and year are **NOT STATED**.
  - The official val split is 150 scenes and 6,019 keyframes at 2 Hz, of which 5,119 are scorable
    (`docs/benchmarks/nuscenes.md:21-23`).
- **What the gate uses.** 1,000 of the 5,119, from 143 of the 150 scenes. They are spread evenly over 34 groups of
  city area × maneuver (left, right, straight) × speed band (slow < 3 m/s, medium, fast ≥ 8 m/s), random within each
  group, seed 20261022 (`tools/gate_smoke/nuscenes.toml` header; `docs/gate.md:244`).
- **What the model receives.**
  - The scored keyframe plus up to 4 earlier ones (2 s), so 4 frames 0.5 s apart (`nuscenes.md:25-28`).
  - Speed and acceleration from the real CAN bus (`nuscenes/loader.py:385-387`). Sideways parts are zeroed by the node.
  - Command from the recorded future: lateral offset at 3 s ≥ 2 m is left, ≤ −2 m is right, else straight
    (`nuscenes.md:33-34`; `nuscenes/loader.py:205-230`).
  - Asked once per keyframe.
- **Primary metric: L2 (1/2/3 s), `l2_avg`.** The mean of the distance between predicted and recorded position at
  exactly 1 s, 2 s and 3 s: (‖e₁ₛ‖ + ‖e₂ₛ‖ + ‖e₃ₛ‖)/3, in metres (`nuscenes/harness.py:47, 109-119`). **Lower is better.**
- **Other metrics:** L2 and collision rate at each time, UniAD-style cumulative variants, `ade_3s`, `pdm_ep`,
  `pdm_comfort` (`nuscenes/benchmark.yaml:9`).
- **Unit:** the scene (`tools/gate.py:246`).
- **Baseline now:** `20261023T200508-261359`, L2 (1/2/3 s) **2.431 m** on the 1,000.

### NAVSIM v2 (open loop, scored by re-simulation): subset `navsim`, quick tier

- **Dataset.** OpenScene, a 2 Hz redistribution of nuPlan (Motional). Cities: Las Vegas, Pittsburgh, Boston and
  one-north (Singapore). 8 cameras (`docs/benchmarks/navsim.md:41-46`). The year is **NOT STATED**.
  - The test split `navtest` has 136 logs and 12,146 scenarios (`navsim.md:20`).
- **What the gate uses.** 1,018 tokens (scenarios) from 126 logs: 1,000 drawn evenly over 35 groups of city ×
  driving command × speed band, then every touched log topped up to 2 because the scorer dies on one. Seed 20261022
  (`tools/gate_smoke/navsim.toml` header).
- **Open loop, but scored by re-simulation.** The model's 4 s plan is driven by a bicycle model and tracker inside
  NAVSIM's scorer, but the model is not asked again along the way (`navsim.md:3-10`).
- **What the model receives.**
  - The scorer's four 2 Hz frames, 1.5 s of history (`navsim.md:396`).
  - The log's driving command (`navsim/agent.py:291-294`).
  - Speed and acceleration from the ego status. Sideways parts are zeroed by the node (`navsim.md:279-285`).
  - Asked once per scenario.
- **Primary metric: EPDMS** (extended PDM score), computed by NAVSIM's own devkit:
  - EPDMS = (no at-fault collision × drivable area × traffic light × driving direction) × weighted mean of {ego
    progress 5, time to collision 5, lane keeping 2, history comfort 2, two-frame comfort 2}.
  - It runs from 0 to 1, and **higher is better** (`navsim.md:23-26`).
- **Unit:** the log, read from the runs' plan dumps. A sample with no known log gets NO UNIT, never a verdict
  (`tools/gate.py:252-256, 923-928`; `docs/gate.md:35-38`).
- **Pairs only with a NAVSIM smoke of the same list:** the two-frame comfort term compares each scenario with the
  next scored one in its log (`docs/gate.md:452-454`).
- **Baseline now:** `20261023T202240-de2f00`, EPDMS **0.857**.

### neuroNCAP (closed loop): subsets `neuroncap_quick` (quick tier) and `neuroncap` (closed tier)

- **Dataset.** NeuRAD neural-rendering reconstructions of nuScenes scenes (`docs/benchmarks/neuroncap.md:3-6`), in
  Boston (`neuroncap.md:315-316`). A safety-critical actor is inserted in each scenario.
  - 20 official scenario-scene pairs over 14 scenes. A full sweep is 10 seeded runs of each, 200 episodes
    (`neuroncap.md:353-357`).
  - Crash types: frontal, side, stationary.
- **What the gate uses.**
  - **Quick tier:** 3 pairs, one per crash type (frontal-0103, side-0103, stationary-0783), 1 run each (seed 0). It
    can fail the gate but never ship it (`tools/gate_smoke/neuroncap_quick.toml`).
  - **Closed tier:** 15 of the 20 pairs, 1 run each: all 5 frontal, all 5 side, and 5 of 10 stationary drawn at
    random (`tools/gate_smoke/neuroncap.toml`).
- **Closed loop.** The model is asked at 10 Hz and returns 6 waypoints at 0.5–3.0 s. A tracker drives one 0.1 s step
  of each plan, with acceleration clamped to ±6 m/s² (`neuroncap.md:44-46, 69-96`).
- **What the model receives.**
  - The 4 rendered nuScenes cameras.
  - The sources of speed, acceleration and command on the neuro-ncap side are **NOT STATED** in Paddock's files read
    here. The node zeroes the sideways parts.
  - Momentum is active (a plan 0.45–0.55 s old exists).
- **Primary metric: NCAP score.** Per episode, 5 if the collision is avoided, else 4 · max(0, 1 − v_impact /
  v_reference), where v_impact is the speed at impact and v_reference the speed it would have hit at without
  braking. Range 0–5, **higher is better**, averaged over episodes (`neuroncap.md:136-137`;
  `neuroncap/harness.py:84-92`).
- **Other metrics:** collision rate, impact speed, PDM-style terms (`neuroncap/benchmark.yaml:5`).
- **Unit:** the scenario-scene pair, e.g. `frontal-0103` (`tools/gate.py:234-244`).
- **Baselines now:** quick `20261026T195633-2a9a76`, NCAP **3.179** (3 episodes); closed `20261026T232704-7bcf0e`,
  NCAP **3.679** (15 episodes).

### Bench2Drive (closed loop): subsets `bench2drive_quick` (quick tier) and `bench2drive` (closed tier)

- **Dataset.** Thinklab-SJTU's benchmark on the CARLA 0.9.15 driving simulator: 220 short routes in Towns 12, 13 and
  15, covering 44 scenario types × 5 (`docs/benchmarks/bench2drive.md:3-6`). Synthetic, not recorded.
- **What the gate uses.**
  - **Closed tier:** 7 of 220 routes, one per group (left turn, right turn, and straight routes grouped by hazard:
    cut-in or hard brake, highway or high speed, junction, obstacle in lane, pedestrian or cyclist). Seed 20261022.
    `can_ship = false`, because 7 routes detect only a 17–38 point drop (`tools/gate_smoke/bench2drive.toml`).
  - **Quick tier:** route 3717 only (the shortest of the 7, 72 m), a crash check (`bench2drive_quick.toml`).
- **Closed loop.** CARLA ticks at 20 Hz. Drive-JEPA is asked every 0.5 s (`planning_hz: 2.0`), and the last plan is
  followed in between by a PID controller (`bench2drive.md:11-13, 87-92`; `model.yaml:46-49`).
- **What the model receives.**
  - Cameras at 800×450 (`bench2drive/agent.py:98-101`).
  - **Command:** the route's road option. Lane-follow becomes "unknown" (`agent.py:112-120, 302-313`).
  - Acceleration is the difference of successive speeds (`agent.py:383, 401-402`).
  - Momentum is active.
- **Primary metric: driving score** = route completion (%) × infraction penalty (a product of penalties for
  collisions, red lights and similar). Range 0–100, **higher is better** (`bench2drive.md:19, 28-30`;
  `bench2drive/harness.py:256-286`).
- **Unit:** the route (`tools/gate.py:249`).
- **Baseline now:** `20261026T233957-56d38c`, driving score **59.04** on the 7.

### AlpaSim (closed loop): subsets `alpasim_quick` (quick tier) and `alpasim` (closed tier)

- **Dataset.** NVIDIA's AlpaSim simulator, driving inside NuRec neural reconstructions of real recorded clips from
  the `public_2601` suite, 913 scenes (`docs/benchmarks/alpasim.md:3-4, 16`; `docs/gate.md:459-466`). Country, year
  and sensor rig are **NOT STATED**.
- **What the gate uses.** 10 scenes (scene set `public_2601_gate10`).
  - They were drawn on 2026-10-27, seed 20261027, from the 50 scenes already downloaded on shakaal (not from all 913),
    over 6 turn × speed groups read from each scene's recorded drive.
  - Four were then swapped within their group: two were not on the driver's scene list, and on two every model left
    the road at once (`tools/gate_smoke/alpasim.toml` header; `docs/gate.md:459-479`).
  - The quick tier runs one of the 10 (`straight/medium`) as a crash check (`alpasim_quick.toml`).
- **Closed loop.** AlpaSim steps at 50 ms (20 Hz) for 400 steps (about 20 s), one rollout per scene.
  - The first 3.0 s replay the recorded drive ("warm-up"), with the model called during it.
  - Drive-JEPA replans every 0.5 s. Between replans the cached plan is re-anchored to the car's current pose
    (`configs/driver/paddock_bridge_drivejepa.yaml:20, 42-48, 55`; `alpasim/drivejepa_rhythm.py:14-17, 73-75`).
- **What the model receives.**
  - One front camera, `camera_front_wide_120fov`, rendered at 1920×1080 and handed over as a pinhole camera with
    f = 1545. The lens distortion is rendered and **not undone** (`paddock_bridge_drivejepa.yaml:13-14, 50-72`).
  - The other three camera slots are placeholders copied from the front (`alpasim/harness.py:173-183`).
  - Frame pair: the newest frame and the one 0.5 s back (±10 %).
  - Speed and acceleration forward only (`drivejepa_rhythm.py:37-42, 352-353`).
  - **Command:** from the first route waypoint at least 20 m ahead; more than 3 m left is left, more than 3 m right is
    right, else straight; no route is unknown (`drivejepa_rhythm.py:82-84, 159-178`).
  - Momentum is active.
- **Primary metric: scene score.**
  progress_score = min(clamp(progress, 0, 1) / 0.8, 1.0);
  scene_score = 0 if an at-fault collision or off-road happened, else progress_score (`alpasim.md:206-210`).
  - progress is the fraction of the recorded route covered within the scored window.
  - The window drops the warm-up and is cut at the first collision of any kind or off-road, or when the car is 4 m
    from the recorded path (`alpasim.md:175-185`).
  - Range 0–1, **higher is better**. Reaching 80 % progress already gives 1.0 (`alpasim.md:292`).
- **Other metrics:** uncapped progress, collision rate, at-fault collision, off-road, open-loop collision, fraction
  of 20 s driven, distance to the recorded path (`alpasim/benchmark.yaml:20-21`).
- **Unit:** the scene (clip) (`tools/gate.py:250`).
- **Baseline now:** `20261027T005941-d3f828`, scene_score **0.434** on the 10. Per scene (from its records):

  | scene | score | progress | at-fault | any collision | off-road |
  |---|---|---|---|---|---|
  | e121e37d | 0.941 | 0.753 | 0 | 0 | 0 |
  | bae463d3 | 0.000 | 0.172 | 0 | 0 | 1 |
  | d1a25a99 | 0.331 | 0.265 | 0 | 0 | 0 |
  | cdd9448c | 0.961 | 0.769 | 0 | 0 | 0 |
  | 02e075b9 | 1.000 | 1.000 | 0 | 0 | 0 |
  | 98dd56b2 | 0.158 | 0.126 | 0 | 1 | 0 |
  | 1acf2f5d | 0.929 | 0.744 | 0 | 0 | 0 |
  | b44a4a44 | 0.000 | 0.002 | 1 | 1 | 0 |
  | 8023dce5 | 0.000 | 0.686 | 1 | 1 | 0 |
  | d2715023 | 0.024 | 0.019 | 0 | 0 | 0 |

---

## D3. The statistics, step by step

1. **Pairing on `sample_id`.** Both runs' `samples.jsonl` are read into dictionaries keyed by `sample_id`. Only the
   shared ids are used. Samples with a missing or non-finite value are left out and counted in a note
   (`tools/gate.py:701-722, 728-729`).
2. **The sign convention: candidate minus baseline, signed so that positive means better.**
   - For a metric where lower is better (ADE, L2), the gate multiplies by −1 (`tools/gate.py:658-660, 721`).
   - So "better by 0.71" on ADE means the candidate's ADE is 0.71 m **lower**.
   - The overall difference is the mean of these per-sample differences, which equals the gap between the two means
     (`tools/gate.py:741`).
3. **Grouping by unit.** Each difference goes into a bucket for its independent unit (scene, route, log, scenario
   pair, clip; `UNIT` at `tools/gate.py:243-257`).
   - Why: samples from one scene move together. Two keyframes 4 s apart in one scene share the road, the weather and
     the other cars, so they are not two independent pieces of evidence.
   - Treating them as independent once turned a neuroNCAP "not separable" into t = −4.25 (`docs/gate.md:31-35`).
4. **Cluster-robust standard error (CR1).** The standard error (SE) says how much the mean difference would wobble
   if the experiment were repeated. The "cluster-robust" version measures that wobble from how much whole units
   disagree, not individual samples (`tools/gate.py:743-747`):
   - n = number of paired samples, G = number of units, d̄ = mean difference.
   - For each unit g, the residual is eₘ = (sum of its differences) − (its sample count) × d̄. That is how far the
     unit's total is from what the average would predict.
   - var = G/(G − 1) × Σ eₘ² / n², and SE = √var. The factor G/(G − 1) is the small-sample correction that gives
     "CR1" its name.
5. **t value.** t = d̄ / SE: how many standard errors the difference is away from zero (`tools/gate.py:750-751`).
   If SE = 0 and the difference is a real constant shift, t = ±∞. A float-noise difference stays at t unset, which
   counts as NEUTRAL (`tools/gate.py:752-756`).
6. **Degrees of freedom.** df = G − 1 (`tools/gate.py:748`). The count of units decides it, not the count of samples.
7. **Critical value.** t crit is the two-sided 5 % value of Student's t at df. Paddock has no scipy, so it uses a
   table, interpolated linearly in 1/df between rows (`tools/gate.py:115-151, 200-216`). Selected rows:

   | df | 1 | 2 | 3 | 4 | 5 | 6 | 9 | 14 | 30 | 60 | 120 | ∞ |
   |---|---|---|---|---|---|---|---|---|---|---|---|---|
   | t crit | 12.706 | 4.303 | 3.182 | 2.776 | 2.571 | 2.447 | 2.262 | 2.145 | 2.042 | 2.000 | 1.980 | 1.960 |

   In gate terms: 3 neuroNCAP quick pairs give 4.303; 7 Bench2Drive routes 2.447; 10 AlpaSim scenes 2.262; 15
   neuroNCAP pairs 2.145; KITScenes' 117 scenes, nuScenes' 143 and NAVSIM's 126 logs about 1.98 (`docs/gate.md:243-251`).
8. **The verdict.** REGRESSION if t ≤ −t crit; GAIN if t ≥ t crit; NEUTRAL otherwise (`tools/gate.py:494-501`).
   NEUTRAL means "not separable at this many units", not "the same" (`docs/gate.md:75`).
9. **MDE (minimum detectable effect)** = (t crit + t₈₀) × SE. It is the smallest true difference the pair would catch
   4 times in 5 (`tools/gate.py:219-227`; `docs/gate.md:63-69`). A NEUTRAL with a large MDE says little.
10. **Overrides.** These replace GAIN, NEUTRAL or REGRESSION when the comparison cannot be trusted:

| override | when | source |
|---|---|---|
| NO RECORDS | one side has no `samples.jsonl` on this site, or no shared samples with a value. Means are shown, nothing is claimed | `tools/gate.py:681-698, 732-735` |
| TOO FEW UNITS | fewer than 2 units: no spread to measure | `tools/gate.py:195, 758-762` |
| NO UNIT | some samples have no unit (NAVSIM with no plan dump naming their log) | `tools/gate.py:767-768` |
| INCOMPLETE | the candidate lost samples to a failure (`dataset.n_failed`, or `status: partial`), or lost every sample. The kept samples would be the easy ones | `tools/gate.py:447-458, 769-779, 688-692` |
| CONFIG DIFFERS | the two runs' `config` differs on a key other than `model_env`, `declared_env` and the subset keys (`limit`, `scene`, `runs`, `scene_set`, `pairs`, `sample_filter`, …). `allow_config` can name a key the candidate is meant to change | `tools/gate.py:401-445, 785-795` |
| SUBSET DIFFERS | the pair names its subset and one of the runs does not hold all of it | `tools/gate.py:1002-1072` |

The order matters. Later checks in `_compare` overwrite earlier ones, so CONFIG DIFFERS beats INCOMPLETE, which
beats NO UNIT (`tools/gate.py:767-795`). SUBSET DIFFERS replaces only a decided verdict (`tools/gate.py:1036-1039`).

### Worked example (real numbers on disk)

The released Drive-JEPA weights, KITScenes, compared with themselves under two image geometries:
- baseline: uncropped "native", run `20261023T015507-adf741`;
- candidate: the OpenScene crop, run `20261023T071315-e90d98`.

This is the measurement behind PR #106. The gate's own output is in
`{naren_workspace}/renesas/paddock-runs/ops/gate/pr106_rerun_20261023/rerun.out:58-68`:

```
bench        metric   baseline candidate better by      SE      t t crit     MDE   units  verdict
kitscenes    ade         3.249     2.537     0.711   0.201   3.54   1.98   0.568     117  GAIN
             note: 37 sample(s) of overlap_train_val left out of this verdict ...; on them alone, better by 0.649, t 1.78
```

Recomputed by hand from the two `samples.jsonl` files (under
`{naren_workspace}/renesas/paddock-runs/kitscenes/drivejepa/`), step by step:

1. **decide_on.** Keep the 186 `val` keyframes. They fall in G = 117 scenes: 75 scenes hold 1 keyframe, 28 hold 2,
   6 hold 3, 4 hold 4, 3 hold 5 and 1 holds 6.
2. **Means.** Baseline ADE (3 s) 3.2487 m, candidate 2.5375 m.
3. **Signed differences.** ADE is lower-is-better, so each sample contributes −(candidate − baseline). Their mean is
   d̄ = **+0.7112 m**: the candidate is 0.71 m closer to the human path on average.
4. **Residuals per scene.** Three examples:
   - scene 00fc1b9b has one keyframe, difference +0.869, residual 0.869 − 1 × 0.711 = +0.157;
   - scene 02be7d74 has one keyframe, +0.723, residual +0.012;
   - scene 03ed2585 has two keyframes, −1.175 and −2.097. Its residual is (−3.272) − 2 × 0.711 = −4.694, a scene
     where the crop made things worse on both keyframes.
5. **SE.** var = 117/116 × Σ eₘ² / 186², which gives **SE = 0.2011 m**. Treating the 186 keyframes as independent
   would give 0.2025 here: nearly the same, because most scenes hold one keyframe.
6. **t** = 0.7112 / 0.2011 = **3.54**, on df = 116. Interpolating between the df 60 and df 120 rows gives t crit ≈ 1.98.
7. **Verdict.** 3.54 ≥ 1.98, so **GAIN**. MDE = (1.98 + 0.845) × 0.201 ≈ 0.568 m.

This matches the gate's line to the printed precision. The queue's current KITScenes baseline (`…195836-e90d98`) is
a later run of the same cropped setting, which is why its run id ends in the same `e90d98`.

---

## D4. Tiers and the overall verdict

- **Tiers** (`tools/gate.py:1082-1085`; `docs/gate.md:226-239`):
  - `quick`, the default: kitscenes, nuscenes, navsim, neuroncap_quick, bench2drive_quick, alpasim_quick.
  - `closed`: neuroncap, bench2drive, alpasim.
  - `all`: quick, then closed.
- **Run order.** With one lane, a tier runs cheapest first, so a regression stops the gate soonest. With several
  lanes, the longest runs first, using `EXPECTED_MIN` (`tools/gate.py:1093-1132`).
- **When the closed tier is skipped.** Under `--tier all`, the closed tier starts only if every quick subset finished
  and none was REGRESSION or INCOMPLETE (`tools/gate.py:1960-1983`; `docs/gate.md:310-313`).
- **Stop at the first regression.** After a finished subset is REGRESSION or INCOMPLETE, nothing new launches.
  Running subsets finish (`tools/gate.py:1960-1961`).
- **`--keep-going`** turns both rules off (`tools/gate.py:2154-2156`).
- **Overall** (`tools/gate.py:812-834`; `docs/gate.md:83-94`):
  - **FAIL** if any pair is REGRESSION, Bench2Drive and neuroncap_quick included.
  - **HOLD** if any subset is unfinished, or any pair is not GAIN, NEUTRAL or REGRESSION. A crash check that ran
    cleanly (TOO FEW UNITS by design) is exempt from this.
  - **SHIP** if neither of the above applies and at least one closed-loop pair that may ship is a GAIN. Only
    neuroNCAP's 15-pair subset and AlpaSim may ship; Bench2Drive and neuroncap_quick have `can_ship = false`.
  - **HOLD** otherwise. **The quick tier alone can therefore never SHIP** (Naren's howto,
    `…/ops/gate/regression-gate-howto.md:45`).
- **Exit codes:**

| code | when |
|---|---|
| `smoke` 0 | finished, not FAIL |
| `smoke` 1 | FAIL, or something unfinished (`tools/gate.py:2096, 2104`) |
| `smoke` 2 | refused to start: unknown model, overlapping `--cores`, renderer port inside a lane's ports (`tools/gate.py:1838, 1854, 1862`) |
| `smoke` 130 | stopped by SIGTERM or Ctrl-C after lanes were torn down (`tools/gate.py:2083`; `docs/gate.md:314-319`) |
| `check`/`compare` | 1 on FAIL, else 0 (`tools/gate.py:2222`) |

---

## D5. Known limits stated by the code or docs

- **False regressions on an unchanged model.** Each pair is tested at 5 % two-sided. Across pairs, an unchanged model
  is called a REGRESSION somewhere about **10 %** of the time on the quick tier's 4 decided pairs, and about **16 %**
  on `--tier all`'s 7, if the pairs were independent. "A FAIL on one pair, alone, is a reason to look at that pair's
  numbers, not proof the candidate is worse" (`docs/gate.md:47-52`).
- **Few units means low power.**
  - On 3 units, a fixed |t| ≥ 2 rule would fire 18 % of the time by chance, so the bar is 4.30 (`docs/gate.md:40-45`).
  - Bench2Drive's 7 routes detect only a 17–38 point drop (`tools/gate_smoke/bench2drive.toml`).
- **AlpaSim scene noise (near-ties).** With one rollout per scene, a single scene can move by up to about 0.55
  between two runs of the same code. Two of the 32 candidates are nearly tied after the handover, and a tiny
  difference flips the pick. The 30-scene mean moved by 0.013. "For this model, compare means, or flag only when
  several scenes move" (`docs/gate.md:488-495`). This is guidance, not code.
- **Metrics recorded but not gated.**
  - Only one metric per benchmark decides: the run's primary metric, unless a manifest pair sets `metric`
    (`tools/gate.py:654-655`).
  - PDM-style rows are printed under each pair but "never change a verdict or the overall line". An unchanged model
    would show at least one wrong-direction row about 35 % (quick) or 51 % (all) of the time if the rows were
    independent (`docs/gate.md:118-176`). The rows: NC = no at-fault collision, DAC = drivable area compliance,
    TTC = time to collision, EP = ego progress, C = comfort.
  - On AlpaSim, NC is `collision_at_fault`, DAC is `offroad` and EP is `progress` (`docs/gate.md:150`).
  - KITScenes' 37 `overlap_train_val` keyframes are reported in a note only (`docs/gate.md:262-265`).
  - The two crash checks never give a verdict (`docs/gate.md:229-235`).
  - Bench2Drive's `outside_route_lanes` and its 20 Hz `metric_info.json` are written but not read (`docs/gate.md:199-201`).
- **Saturating metric.** AlpaSim's scene score is capped at 1.0 once progress reaches 80 %
  (`docs/benchmarks/alpasim.md:206-210, 292`). Two runs that cover 80 % and 100 % of the route score the same.
  Uncapped `progress` is recorded beside it.
- **Subset choices.**
  - AlpaSim draws only from the 50 scenes downloaded on shakaal, not the whole 913-scene suite. Thirty of those 50
    are the scenes behind the published 30-scene results (`docs/gate.md:459-466`).
  - The subsets weigh rare driving more than the full benchmarks, so the gate's means are not estimates of full
    scores (`docs/gate.md:223-224`).
- **Invalid runs.** A checkpoint override (`DRIVEJEPA_CHECKPOINT`) marks every candidate run `valid: false`, so it
  never reaches the leaderboard (`docs/gate.md:496-501`). Subset runs are `valid: false` too. KITScenes (all 223) and
  AlpaSim (its whole pinned scene set) are the exceptions (`docs/gate.md:256-260`).
- **AlpaSim is not CPU-capped by the lane:** its Docker containers are pinned by the launcher, not by `taskset`
  (`docs/gate.md:293-295, 502-504`).
- **Code differences are noted, not refused.** If the baseline and candidate ran different Paddock commits, the gate
  says so in a note; the adapter change is part of the difference (`docs/gate.md:111-115`).

---

## D6. Glossary

| term | meaning |
|---|---|
| ADE (3 s) | average displacement error: mean distance between predicted and recorded positions over the 6 waypoints at 0.5–3.0 s, metres; lower is better |
| AlpaSim | NVIDIA's closed-loop driving simulator over neural reconstructions (NuRec) of recorded clips |
| at-fault collision | a collision the simulator attributes to the ego car |
| baseline | the stored run the candidate is compared with (here the released Drive-JEPA weights) |
| baseline manifest | TOML file naming each subset's baseline run (`[baseline.<subset>] run = "drivejepa:<id>"`) |
| Bench2Drive | closed-loop benchmark of 220 short routes in the CARLA simulator |
| candidate | the new checkpoint or code being tested |
| can_ship | per-subset flag; when false, a GAIN there never makes the overall result SHIP |
| closed loop | the simulator drives the car with the model's plan, so mistakes change what the model sees next |
| CR1 cluster-robust SE | standard error computed from how much whole units (scenes, routes) disagree, with the G/(G−1) correction |
| crash check | one-sample subset in the quick tier that only checks the model can drive at all |
| decide_on | subset field naming the groups that decide the verdict (KITScenes: `val`) |
| declared_env | the env a model's own `model.yaml` sets, recorded per run; a difference is a note, not CONFIG DIFFERS |
| degrees of freedom (df) | here, number of units − 1 |
| derived site config | the gate's modified copy of `paddock.toml` for the candidate |
| driving command | one-hot hint of the intended maneuver: left, straight, right, unknown |
| driving score | Bench2Drive's route completion × infraction penalty, 0–100 |
| EPDMS | NAVSIM v2's extended PDM score, 0–1 |
| encoder | the image network (ViT-L, with LoRA in our checkpoint) |
| FAIL / SHIP / HOLD | overall results (D4) |
| generator | the trajectory head that makes the 32 candidates |
| group (stratum) | category used to draw a subset evenly (turn, speed, city, crash type) |
| lane | one `--cores` CPU list with its own port range; lanes run subsets side by side |
| L2 (1/2/3 s) | nuScenes' mean of position errors at exactly 1, 2 and 3 s |
| MDE | minimum detectable effect, (t crit + t₈₀) × SE |
| menu | the 32 candidate trajectories |
| model_env | env overrides a run was given (e.g. `DRIVEJEPA_CHECKPOINT`), recorded per run |
| momentum-aware selection | scoring step that favours plans close to the one picked 0.5 s earlier; closed loop only |
| NCAP score | neuroNCAP's per-episode 0–5 score: 5 if the crash is avoided, else 4 · max(0, 1 − impact speed / reference speed) |
| neuroNCAP | closed-loop collision-avoidance benchmark on NeuRAD reconstructions of nuScenes |
| NAVSIM | benchmark on OpenScene/nuPlan logs; open-loop plans scored by re-simulation |
| nuScenes | recorded dataset (Boston, Singapore), used open loop |
| open loop | the model is scored against recorded driving; its plan is never driven |
| OpenScene crop | crop and resize of each camera onto the 64° camera Drive-JEPA trained on |
| override verdict | NO RECORDS, TOO FEW UNITS, NO UNIT, INCOMPLETE, CONFIG DIFFERS, SUBSET DIFFERS |
| pairing | joining two runs' samples on `sample_id` |
| PDM rows | NC, DAC, TTC, EP, C terms shown under each pair; never decide |
| pick | the candidate the scorer chooses |
| progress (AlpaSim) | fraction of the recorded route covered in the scored window, uncapped |
| projector | the layer between encoder features and the planner |
| queue | shared job runner (`tools/gate_queue.py`) that runs gate jobs one at a time as the `paddock` account |
| regression | a candidate worse than its baseline beyond chance (t ≤ −t crit) |
| run.json / samples.jsonl | a run's settings record / its per-sample records |
| sample_id | the identifier a sample is paired on |
| scene score | AlpaSim's 0–1 score: 0 on at-fault collision or off-road, else min(progress / 0.8, 1) |
| scorer | the head that scores the 32 candidates |
| site config | `paddock.toml`: where data, environments and outputs live on a machine |
| standard error (SE) | how much the mean difference would vary if the test were repeated |
| subset | fixed tracked sample list for one benchmark in `tools/gate_smoke/` |
| t / t crit | difference divided by SE / the 5 % two-sided cutoff for the given df |
| tier | `quick` (every checkpoint) or `closed` (closed-loop subsets); `all` runs both |
| unit | the independent grouping the SE is clustered on: scene, route, log, scenario pair, clip |
| valid | run flag; false keeps a run off the leaderboard |
| warm-up (AlpaSim) | first 3.0 s replaying the recorded drive before the model's plans are driven |
