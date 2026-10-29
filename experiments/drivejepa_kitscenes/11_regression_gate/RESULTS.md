# Gate results: encoder, generator and scorer trained together (KITScenes, 4.5 passes)

Read-only analysis of gate job `20261027T215457-siddesh-94f0` (submitted 2026-10-28 03:25 IST, finished 05:22 IST,
tier `all`, keep-going, paddock commit `9585f816c05f`). Nothing was run, submitted or edited outside this folder.
All numbers are also in `results.json`, built by `build_results.py`, which reads only the files named below.

Sources: the job folder `{paddock_site}/queue/jobs/20261027T215457-siddesh-94f0/` (`verdict.json`, `gate.log`,
`baseline.toml`, `smoke/manifest.toml`, `smoke/*.log`), per-sample records `{paddock_runs}/<bench>/drivejepa/<run>/`,
`run.json` under each checkout's `results/`, and our held-out dump `{runs}/drivejepa_kitscenes/generator_joint_kit/s5/heldout223.pkl`.

**Overall: FAIL.** All 9 subsets finished (`unfinished: []`, every run `status: complete`, `n_failed: 0`).

---

## Part F: Summary for Sid

On the two open-loop sets it was trained for, the model is much closer to the human driver: KITScenes ADE (3 s) went
from 2.537 m to 0.920 m and nuScenes L2 (1/2/3 s) from 2.431 m to 0.915 m, both clear GAINs. On every benchmark where
it has to drive, it does worse. NAVSIM EPDMS fell from 0.857 to 0.753, mostly from leaving the drivable area (102 vs
23 scenarios fail it) and less progress. Bench2Drive is trustworthy in the sense that the runs did not break: the
simulator ran, the model server answered every request and the Bench2Drive code is identical to the baseline's. But the
car never moved on 6 of the 7 routes (0.0 m travelled), so the near-zero score reflects the model's plans from a standing
start, not a pipeline failure. On AlpaSim the drop is spread over 5 of 10 scenes. Two of them flipped from about 1.0 to 0
through new at-fault collisions, and three lost about half their score because the car drifted off the recorded path
early. One scene got slightly better and four stayed the same. neuroNCAP is NEUTRAL by the gate's test, but the
candidate hit the target in 8 of 15 scenarios against the baseline's 6, often faster than the reference speed. The
KITScenes gap (gate 0.920 m vs our 0.596 m) comes from the same weights given different inputs on the same 223 moments.
The command input explains only a small part: the gap is still 0.27 m on the 184 moments where "straight" is the true
command. The rest must come from the other inputs: acceleration 0, sideways speed 0, fixed 2.2 m camera height. A single
CPU run would be needed to apportion the gap among them (E3).

---

## Part A: Verdict table, confirmed

### A1. Per subset (from `verdict.json` and `gate.log`)

Signed so positive "better by" is better. Units: scenes (KITScenes), logs (nuScenes, NAVSIM), scenario pairs
(neuroNCAP), routes (Bench2Drive), scenes (AlpaSim).

| subset | metric | baseline | candidate | better by | SE | t | t crit | MDE | units | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| kitscenes | ADE (3 s), m | 2.537 | 0.920 | +1.617 | 0.182 | 8.91 | 1.98 | 0.513 | 117 | **GAIN** |
| nuscenes | L2 (1/2/3 s), m | 2.431 | 0.915 | +1.516 | 0.104 | 14.58 | 1.98 | 0.293 | 143 | **GAIN** |
| navsim | EPDMS | 0.857 | 0.753 | −0.104 | 0.011 | −9.31 | 1.98 | 0.032 | 126 | **REGRESSION** |
| neuroncap_quick | NCAP | 3.179 | 2.219 | −0.960 | 2.198 | −0.44 | 4.30 | 11.792 | 3 | NEUTRAL (cannot ship) |
| bench2drive_quick | driving score | 37.67 | 18.47 | −19.20 | – | – | – | – | 1 | TOO FEW UNITS (crash check) |
| alpasim_quick | scene score | 0.929 | 0.000 | −0.929 | – | – | – | – | 1 | TOO FEW UNITS (crash check) |
| neuroncap | NCAP | 3.679 | 2.727 | −0.953 | 0.755 | −1.26 | 2.15 | 2.276 | 15 | NEUTRAL |
| bench2drive | driving score | 59.036 | 0.269 | −58.767 | 11.407 | −5.15 | 2.45 | 38.246 | 7 | **REGRESSION** (cannot ship) |
| alpasim | scene score | 0.434 | 0.116 | −0.319 | 0.127 | −2.51 | 2.26 | 0.399 | 10 | **REGRESSION** |

**Run ids.**
- Candidate run ids come from `smoke/manifest.toml`; baseline run ids from `baseline.toml`. All are in `results.json` → `run_ids`.
- Baseline commits: KITScenes, nuScenes and NAVSIM ran at d889539; neuroNCAP-quick at 54306af; neuroNCAP and
  Bench2Drive at 0687080; AlpaSim at 0687080 and 1561b66, stitched.

**ADE window confirmed as 3 s.**
- `kitscenes/harness.py:42` sets `EVAL_HORIZON_S = 3.0`. `score()` at `:591` calls
  `metrics.displacement_errors(..., INTERVAL_S=0.5, EVAL_HORIZON_S)`.
- `metrics.py:48-50` averages the first `round(3.0/0.5) = 6` points.
- Recomputing a gate record from its `prediction`/`ground_truth` fields over 6 points gives exactly its `ade`.

**nuScenes `l2_avg` window.** It is the mean of L2 at 1, 2 and 3 s (`nuscenes/harness.py:17,47`).

**Notes the gate printed (verbatim in `results.json` → `A.pairs[*].notes`):**

- **Every pair:**
  - "candidate run is marked invalid: … site env override may change what was measured: DRIVEJEPA_CHECKPOINT,
    DRIVEJEPA_CHECKPOINT_SOURCE"
  - "candidate sets … and the baseline does not: the baseline ran model.yaml's default for it"
  - "the runs used different paddock code (<baseline sha> vs 9585f816c): an adapter change between them is part of the
    difference"
- **KITScenes, nuScenes and NAVSIM only:** "model.yaml env differs: DRIVEJEPA_MOMENTUM None -> 'on'". The baselines
  predate the momentum default.
- **KITScenes, overlap_train_val note:** "37 sample(s) of overlap_train_val left out of this verdict (… places in the
  training data); on them alone, better by 1.593, t 3.76".
- **nuScenes / NAVSIM:** both runs are subsets (`--limit 1000` named samples, and `--limit 1018`).
- **neuroNCAP:** "15 of the 20 official scenario-scene pairs; a subset" (3 of 20 for the quick check).
- **Bench2Drive:** "7 of the 220 routes this site runs; a subset". The baseline run is also flagged for a site override
  of DRIVEJEPA_CHECKPOINT.
- **AlpaSim:**
  - "baseline run is marked invalid: … resumed: 9 episode(s) reused from an earlier attempt".
  - The crash checks add "the runs cover different sample sets; only the shared samples are paired", "too few units: 1
    independent unit" and "unpaired samples left out: 6 (Bench2Drive) / 9 (AlpaSim) baseline-only".

**Code differences behind the "different paddock code" notes** (`diff -r` of `src/` between checkouts, read-only):

| area | change from 0687080 to 9585f81 |
|---|---|
| Bench2Drive adapter | identical |
| drivejepa model folder | only `model.yaml` differs: new metadata keys `camera_hz`, `ego_history_*` and `plan_points_hz`, plus comments. `registry.py` only parses those keys, and no runtime code reads them (grep). |
| neuroNCAP | harness, launcher and renderer changed |
| AlpaSim | launcher and scene list changed |

### A2. PDM-style rows (never decide the gate)

| subset | NC at-fault | DAC drivable | TTC | EP progress | C comfort | worse beyond t crit |
|---|---|---|---|---|---|---|
| kitscenes | – | – | – | 0.820 → 0.925 (GAIN) | 0.435 → 0.559 (GAIN) | none |
| nuscenes | col_avg 0.021 → 0.002 (GAIN) | – | – | 0.864 → 0.948 (GAIN) | 0.412 → 0.584 (GAIN) | none |
| navsim | 0.993 → 0.988 | **0.977 → 0.900** | **0.988 → 0.968** | **0.867 → 0.737** | 0.977 → 0.974 | DAC t −6.11, TTC t −3.27, EP t −10.67, PDM-style t −8.72 |
| neuroncap_quick | 0.417 → 0.361 | 1.000 → 0.978 | 0.320 → 0.322 | 0.383 → 0.208 | 0.857 → 1.000 | none |
| neuroncap | 0.637 → 0.558 | **0.998 → 0.836** | 0.582 → 0.480 | 0.448 → 0.404 | 0.918 → 1.000 (GAIN) | DAC t −2.75 |
| bench2drive | collision rate 0.286 → 0.143 | – | – | **route completion 72.2 → 1.38** | – | route completion t −5.96 |
| alpasim | collision at fault 0.200 → 0.300 | off-road 0.100 → 0.300 | – | progress 0.454 → 0.292 | – | none (EP t −2.02 vs 2.26) |

Bench2Drive's lower collision rate comes from a car that does not move; it is not safer driving.

### A3. The two crash checks

**bench2drive_quick (route 3717 alone).**
- Both runs exited 0.
- The candidate drove 41.4 m at up to 10.3 m/s and reached route completion 58.17 %. It hit a layout object and a
  vehicle, left its lane, and was then blocked: infraction penalty 0.3175, driving score 18.47.
- The baseline's own 3717, inside its 7-route run, reached 37.67 / 37.67.
- No crash: `n_crashed 0`, status "Failed - Agent got blocked". The same route in the full candidate run reached only
  9.65 % (8.4 m), so this route varies a lot between runs (B4).

**alpasim_quick (clip 1acf2f5d).**
- Exited 0. Scene score 0 from an at-fault collision at progress 0.709, after 0.455 of the 20 s.
- The baseline's 1acf2f5d scored 0.929.
- The full AlpaSim run reproduced the same numbers for this clip: score 0, progress 0.7091, distance to path 2.738 m,
  driven fraction 0.455.

---

## Part B: Bench2Drive near zero: real or broken?

### B1. Per route (from each route's `results.json` → `_checkpoint.records[0]`)

**Candidate, run `20261027T222440-49bb8c`.**

| route | status | completion % | penalty | driving score | infractions | game s | system s |
|---|---|---|---|---|---|---|---|
| 3255 ParkingCrossingPedestrian | Agent got blocked | 0 | 1.0 | 0 | blocked | 60.05 | 154.8 |
| 3482 VehicleOpensDoorTwoWays | Agent got blocked | 0 | 1.0 | 0 | blocked | 60.05 | 641.0 |
| 3717 VehicleTurningRoute | Agent got blocked | 9.65 | 0.195 | 1.88 | layout collision, outside lanes, min speed, blocked | 65.45 | 257.4 |
| 17598 SequentialLaneChange | TickRuntime | 0 | 1.0 | 0 | none | 200.05 | 2301.2 |
| 18356 ParkingCutIn | Agent got blocked | 0 | 1.0 | 0 | blocked | 60.05 | 234.2 |
| 24258 HazardAtSideLaneTwoWays | Agent got blocked | 0 | 1.0 | 0 | blocked | 60.05 | 115.6 |
| 28111 T_Junction | Agent got blocked | 0 | 1.0 | 0 | blocked | 60.05 | 120.6 |

**Baseline, run `20261026T233957-56d38c`.**

| route | status | completion % | penalty | driving score | infractions | game s | system s |
|---|---|---|---|---|---|---|---|
| 3255 | Agent got blocked | 70.97 | 0.60 | 42.58 | 1 vehicle collision, 13 min speed, blocked | 80.1 | 186.3 |
| 3482 | TickRuntime | 31.99 | 1.0 | 31.99 | 5 min speed | 200.05 | 577.4 |
| 3717 | Agent got blocked | 37.67 | 1.0 | 37.67 | 7 min speed, blocked | 77.75 | 275.1 |
| 17598 | TickRuntime | 65.01 | 1.0 | 65.01 | 10 min speed | 200.05 | 781.6 |
| 18356 | Completed | 100 | 1.0 | 100 | 19 min speed | 24.1 | 75.6 |
| 24258 | Completed | 100 | 0.36 | 36.0 | 2 vehicle collisions, 19 min speed | 30.25 | 61.3 |
| 28111 | Completed | 100 | 1.0 | 100 | 17 min speed | 17.3 | 41.2 |

**What the statuses mean.**
- "Agent got blocked" is the leaderboard's no-motion test: game time stops at 60.05 s.
- "TickRuntime" is the 200 s route time limit. The baseline also hits it, on 3482 and 17598.
- The candidate's wall times (samples.jsonl `duration_s`, including simulator start-up) are 501, 749, 497, 2501,
  436, 278 and 168 s.

### B2. Did the car move?

Computed from the 20 Hz ego locations in `metric_info/*/metric_info.json`:

| route | candidate distance | candidate top speed | candidate time stopped | baseline distance | baseline top speed |
|---|---|---|---|---|---|
| 3255 | **0.0 m** | 0.00 m/s | 60.0 of 60.0 s | 83.5 m | 9.49 m/s |
| 3482 | **0.0 m** | 0.00 | 60.0 / 60.0 | 41.6 m | 10.02 |
| 3717 | 8.4 m | 4.06 | 61.8 / 65.4 | 29.4 m | 6.50 |
| 17598 | **0.0 m** | 0.01 | 200.0 / 200.0 | 77.9 m | 11.94 |
| 18356 | **0.0 m** | 0.00 | 60.0 / 60.0 | 131.3 m | 10.88 |
| 24258 | **0.0 m** | 0.00 | 60.0 / 60.0 | 130.8 m | 11.12 |
| 28111 | **0.0 m** | 0.00 | 60.0 / 60.0 | 84.6 m | 10.28 |

**Did the plans ask to move? Not recorded, so UNVERIFIED.**
- The adapter writes per-tick plans and controls only when `PADDOCK_RECORD_DIR` is set (`bench2drive/agent.py:18,60,203`).
  The gate does not set it, and no `log.csv` exists.
- What the code does show:
  - At speed 0, the controller brakes only if the plan's mean speed over the first 4 s is below 0.4 m/s
    (`pid.py`: `BRAKE_SPEED = 0.4`, `CONTROL_HORIZON_S = 4.0`).
  - An HTTP error or timeout from the model server raises in `run_step` (`agent.py:431-433`). The leaderboard would then
    record "Agent crashed"; every route instead shows `n_crashed 0`.
  - So the server returned a plan every time, and those plans asked for less than 0.4 m/s. This is argued from the
    code, not observed.
- Supporting, from AlpaSim: on the two clips that start at rest (b44a4a44, d2715023) the candidate's first plan after
  warm-up averages 0.00 m/s over 3 s (C2).
- **What would confirm it:** one route re-run with `PADDOCK_RECORD_DIR` set, which writes `v_des` per tick. Not run.

### B3. Run health

- **run.json:** `status: complete`, `dataset.n_expected 7`, `n_samples 7`, `n_failed 0`, `n_reused 0`. All 7 routes
  have records, with `entry_status: Finished` and `eligible: true`.
- **Model server log** (`drivejepa_node_9800.log`, 48 lines): model loaded, deprecation warnings only, no error and no
  traceback.
- **Momentum:** `momentum_counts` shows 1136 replans served (1129 fed). Momentum changed the pick on only 3 of 1136, so
  momentum did not cause the standstill. The baseline's momentum changed 72 of 1262.
- **CARLA:** each `evaluator.log` shows `load_world failed, attempts=0` (twice for 3255) followed by success. The
  baseline logs show the same retries (0–4 per route). Agent ticks run continuously to the route's end; no CARLA error
  or crash is reported.

### B4. Verdict: **REAL**

**Evidence that the runs worked:**
- The simulator ticked for the full time on every route (60 s, or 200 s on 17598).
- The model server answered every request: no crash status, no server error, 1136 replans.
- The Bench2Drive adapter is byte-identical to the baseline commit, and the drivejepa runtime code is unchanged.
- The same pipeline moved the same car 41 m on route 3717 in the crash check an hour earlier.

**What the result is:** a car that never leaves its start position. That is what the model's plans from a standing start
produce under this controller.

**Two qualifications:**
- The content of the plans is UNVERIFIED (B2).
- Route 3717 varied from 58 % to 9.65 % between two runs of identical code and weights, so per-route numbers are noisy.
  The full run overlapped neuroNCAP and AlpaSim on the shared GPU, but CARLA runs synchronously, so timing should not
  change the plans. That is argued, not checked.

---

## Part C: AlpaSim per scene

### C1. All 10 scenes (baseline run `20261027T005941-d3f828` vs candidate run `20261027T230105-487c55`)

Columns: scene score, uncapped progress, at-fault collision, any collision, off-road, distance to the recorded path
(m), and fraction of 20 s driven. Each cell is baseline → candidate.

| clip | score | progress | at fault | any coll. | off-road | dist (m) | 20 s frac | class |
|---|---|---|---|---|---|---|---|---|
| 02e075b9 | 1.000 → **0.000** | 1.000 → 0.309 | 0 → **1** | 0 → 1 | 0 → 0 | 3.30 → 2.40 | 0.84 → 0.135 | worse, **>0.55** |
| 1acf2f5d | 0.930 → **0.000** | 0.744 → 0.709 | 0 → **1** | 0 → 1 | 0 → 0 | 1.29 → 2.74 | 0.84 → 0.455 | worse, **>0.55** |
| e121e37d | 0.941 → 0.444 | 0.753 → 0.355 | 0 → 0 | 0 → 0 | 0 → 0 | 3.98 → 4.00 | 0.84 → 0.225 | worse |
| cdd9448c | 0.961 → 0.478 | 0.769 → 0.382 | 0 → 0 | 0 → 0 | 0 → 0 | 4.01 → 4.01 | 0.668 → 0.275 | worse |
| d1a25a99 | 0.331 → 0.000 | 0.265 → 0.325 | 0 → 0 | 0 → 0 | 0 → **1** | 4.01 → 3.24 | 0.45 → 0.208 | worse |
| 98dd56b2 | 0.158 → 0.234 | 0.126 → 0.187 | 0 → 0 | 1 → 0 | 0 → 0 | 0.27 → 4.05 | 0.27 → 0.278 | better |
| bae463d3 | 0.000 → 0.000 | 0.173 → 0.169 | 0 → 0 | 0 → 0 | 1 → 1 | 0.52 → 0.31 | 0.328 → 0.265 | same |
| b44a4a44 | 0.000 → 0.000 | 0.002 → 0.007 | 1 → 1 | 1 → 1 | 0 → 0 | 0.00 → 0.15 | 0.33 → 0.33 | same |
| 8023dce5 | 0.000 → 0.000 | 0.687 → 0.476 | 1 → 0 | 1 → 0 | 0 → **1** | 1.68 → 0.17 | 0.193 → 0.10 | same |
| d2715023 | 0.024 → 0.003 | 0.019 → 0.002 | 0 → 0 | 0 → 0 | 0 → 0 | 0.29 → 0.01 | 0.84 → 0.84 | same |

**Counts.**
- 5 scenes worse, 4 the same, 1 better (|change| ≤ 0.05 counts as the same).
- Above the 0.55 near-tie level: 2 scenes (02e075b9, 1acf2f5d), both from new at-fault collisions.
- At-fault collisions: 2 → 3. Off-road: 1 → 3.

**Spread.** About 1.93 of the 3.19 total score lost comes from the two collision scenes. Another 0.98 comes from
e121e37d and cdd9448c, where the candidate stopped scoring after 4.5–5.5 s with no collision or off-road, at distance
4.00 m from the recorded path. That reads as the run being cut at 4 m from the recorded path. The 4 m rule is from
earlier notes, not re-read in code this session: UNVERIFIED.

### C2. Replans per scene (`<clip>.replans.jsonl`)

**Replan counts.** Every scene in both runs has 39 replans: 5 warm-up and 34 after handover. Momentum was fed on 38 of
39 (all but the first).

**How often momentum changed the pick, per scene: UNVERIFIED.** The replans file records only the final
`chosen_index` and `momentum_fed`, not the pick without momentum.
- Run level for the candidate: run.json `momentum_counts` shows 99 of 390 picks changed.
- The baseline's run.json counts cover only one resumed scene (0 of 39).
- The table below gives how often the chosen menu index changed between consecutive replans instead. This is not the
  same thing.

| clip | pick-index changes (B / C) | speed sent at first replan after warm-up | planned speed, first 0.5 s (B / C) | planned speed, 3 s average (B / C) |
|---|---|---|---|---|
| 02e075b9 | 18 / 28 | 8.23 m/s | 7.94 / 8.35 | 8.72 / 8.33 |
| 1acf2f5d | 28 / 29 | 6.38 | 6.13 / 6.52 | 6.39 / 5.76 |
| e121e37d | 28 / 23 | 12.98 | 11.33 / 12.95 | 10.40 / **12.42** |
| cdd9448c | 27 / 28 | 2.46 | 2.42 / 2.61 | 3.67 / 3.55 |
| d1a25a99 | 31 / 31 | 5.06 | 0.07 / **5.15** | 0.04 / **3.92** |
| 98dd56b2 | 25 / 28 | 2.33 | 2.91 / 2.47 | 4.02 / 3.60 |
| bae463d3 | 34 / 27 | 1.36 | 1.01 / 1.43 | 2.00 / 1.54 |
| b44a4a44 | 7 / 5 | 0.00 | 0.08 / 0.03 | 0.02 / **0.00** |
| 8023dce5 | 28 / 18 | 2.10 | 1.96 / 2.25 | 2.39 / 1.99 |
| d2715023 | 6 / 0 | 0.00 | 0.13 / 0.01 | 0.03 / **0.00** |

**How the columns were computed.**
- The first replan after warm-up is replan 6 in every scene.
- "Planned speed, first 0.5 s" is |first plan point| / 0.5 s. "3 s average" is |sixth point| / 3 s, a straight-line
  distance over the planned path.

**Reading.**
- On the two clips that start at rest, the candidate plans 0.00 m/s, matching Bench2Drive's standing-start behaviour.
- On d1a25a99 the baseline planned a stop and the candidate kept going, then went off-road.
- On e121e37d the candidate planned about 2 m/s faster than the baseline.

### C3. Where the candidate recordings live

Under `{paddock_runs}/alpasim/drivejepa/20261027T230105-487c55/`. Each clip has `<clip>.replans.jsonl`, plus
`<clip>/rollouts/<clip>/<uuid>/rollout.asl` with `metrics.parquet` beside it:

| clip | rollout uuid | size |
|---|---|---|
| 02e075b9 | 16479a9e-d25d-11f1-8e13-fd96eb5cb910 | 102 MB |
| 1acf2f5d | 7ad04370-d25e-11f1-a497-69519f79f8c4 | 152 MB |
| 8023dce5 | 8e84ae6e-d25f-11f1-8d88-998ac43a08fe | 134 MB |
| 98dd56b2 | affaec86-d25d-11f1-ac58-77072210ee8e | 160 MB |
| b44a4a44 | 0c6fafc8-d25f-11f1-bc07-53223a1613e8 | 162 MB |
| bae463d3 | 414d3502-d25b-11f1-ae06-553afc642827 | 155 MB |
| cdd9448c | 6ab37360-d25c-11f1-a708-4598fbb98897 | 142 MB |
| d1a25a99 | cbc6547a-d25b-11f1-a4ca-23875e1d0c57 | 180 MB |
| d2715023 | 086d4772-d260-11f1-981d-2f031bbb362b | 204 MB |
| e121e37d | a8b58a06-d25a-11f1-b304-0f58678caf4a | 142 MB |

Full paths are in `results.json` → `C.scenes[*].rollout_asl`. The crash-check rollout of 1acf2f5d is under run
`20261027T221606-98bbe7`.

---

## Part D: neuroNCAP and NAVSIM detail

### D1. neuroNCAP, 15 scenarios (baseline run `7bcf0e` vs candidate run `aff320`)

| scenario | NCAP B → C | collision B → C | impact speed B → C (m/s) | reference speed |
|---|---|---|---|---|
| stationary-0099 | 5.00 → 5.00 | no → no | 0 → 0 | 7.84 |
| stationary-0101 | 5.00 → 5.00 | no → no | 0 → 0 | 9.65 |
| stationary-0331 | 5.00 → 5.00 | no → no | 0 → 0 | 7.59 |
| stationary-0796 | 5.00 → **0.95** | no → **yes** | 0 → 9.03 | 11.86 |
| stationary-0966 | 5.00 → **2.69** | no → **yes** | 0 → 2.39 | 7.31 |
| frontal-0103 | 2.10 → 1.66 | yes → yes | 6.30 → 7.77 | 13.28 |
| frontal-0106 | 1.67 → **0.00** | yes → yes | 5.65 → 11.59 | 9.71 |
| frontal-0110 | 1.37 → **5.00** | yes → **no** | 11.86 → 0 | 18.05 |
| frontal-0346 | 1.32 → **5.00** | yes → **no** | 7.44 → 0 | 11.11 |
| frontal-0923 | 1.29 → 0.60 | yes → yes | 14.26 → 17.87 | 21.01 |
| side-0103 | 2.44 → **5.00** | yes → **no** | 3.17 → 0 | 8.11 |
| side-0108 | 5.00 → **0.00** | no → **yes** | 0 → 7.12 | 2.15 |
| side-0110 | 5.00 → **0.00** | no → **yes** | 0 → 10.22 | 9.49 |
| side-0278 | 5.00 → **0.00** | no → **yes** | 0 → 7.02 | 4.77 |
| side-0921 | 5.00 → 5.00 | no → no | 0 → 0 | 2.98 |

**Collision counts.** The baseline collides in 6 of 15 and the candidate in 8 of 15. The candidate avoids 3 crashes the
baseline had (frontal-0110, frontal-0346, side-0103) and adds 5 new ones (stationary-0796, stationary-0966, side-0108,
side-0110, side-0278).

**Impact speed vs reference.** In 4 of the candidate's crashes the impact speed exceeds the reference speed, which
scores 0 (frontal-0106, side-0108, side-0110, side-0278). The baseline never exceeds it. The scoring rule:
5 if the crash is avoided, else 4·max(0, 1 − v_impact/v_ref).

**Drivable area.** The DAC regression (0.998 → 0.836) comes from stationary-0099 (0.53), 0331 (0.63), 0796 (0.36),
side-0921 (0.59) and frontal-0923 (0.59).

### D2. NAVSIM EPDMS sub-scores (1018 scenarios; same ids in both runs)

| sub-score | baseline mean | candidate mean | baseline < 1 (of which 0) | candidate < 1 (of which 0) |
|---|---|---|---|---|
| no at-fault collision (hard) | 0.993 | 0.988 | 7 (7) | 13 (12) |
| drivable area (hard) | 0.977 | **0.900** | 23 (23) | **102 (102)** |
| driving direction (hard) | 0.986 | 0.970 | 24 (4) | 45 (17) |
| traffic light (hard) | 0.997 | 0.999 | 3 (3) | 1 (1) |
| ego progress | 0.867 | **0.737** | 565 (31) | 728 (113) |
| time to collision | 0.988 | 0.968 | 12 (12) | 33 (33) |
| lane keeping | 0.960 | 0.946 | 41 (41) | 55 (55) |
| comfort (history) | 0.977 | 0.975 | 23 (23) | 26 (26) |
| **EPDMS** | 0.857 | 0.753 | 683 (69) | 827 (158) |

**Totals.**
- Scenarios failing at least one hard gate: 55 → 147.
- Worst terms: drivable area (79 more scenarios fail), ego progress (82 more at zero) and time to collision (21 more fail).

**EPDMS of 0 with every term at 1.** This happens in 33 baseline and 30 candidate scenarios (e.g. `81f6d1d0bc835cb3`,
which is 0 in both). Not explained here: UNVERIFIED. It may be the scorer's second stage or a filter. Reading
`navsim_scorer.log` / the devkit's EPDMS aggregation would settle it. It affects both runs about equally.

---

## Part E: KITScenes, gate 0.920 m vs our evaluation 0.596 m

### E1. Same moments, same window: CONFIRMED

**Same moments.** The gate's 223 (scene, keyframe) keys equal our dump's 223 keys and the released run's.
**Same ground truth.** The largest difference over the first 6 points is 1.6e-14 m.
**Same window.** Our dump's `ade` re-computes exactly as the mean of 6 points at 0.5 s, the same function and horizon
(`displacement_errors` at 3 s, `final5.py` docstring line 13). Our dump's trajectories equal its rows bit for bit.

| moments | ours ADE (3 s) | gate ADE (3 s) | released via gate | gate worse than ours | same menu index picked | correlation |
|---|---|---|---|---|---|---|
| val (decides the gate) | **0.610** | 0.920 | 2.537 | 128 / 186 | 89 / 186 | 0.50 |
| overlap_train_val | **0.527** | 0.846 | 2.439 | 23 / 37 | 19 / 37 | 0.60 |
| all 223 | 0.596 | 0.908 | 2.521 | 151 / 223 | 108 / 223 | 0.52 |

The gate's own log line "n=223 ade mean=0.908152" matches the all-223 row.

**Momentum.** Momentum was not fed on any KITScenes moment (`momentum_fed: false` on 223 of 223), so it is not part of
the gap.

### E2. Every input difference

| input | gate (paddock 9585f81) | our held-out evaluation |
|---|---|---|
| driving command at the keyframe | **straight**: `[benchmarks.kitscenes] command_source` is unset in the site config, so the default is `"straight"` (`kitscenes/harness.py:393-395, 413`) | **ground truth** from the recorded future, ±5 m lateral at +4 s (`build_sample(..., command_mode="gt")`, `09_heldout_comparison/src/final.py:228`; the rule is in the adapter `_driving_command`, paddock's vendored copy `drivejepa/kitscenes_adapter.py:271-290`) |
| command on history frames | straight (`harness.py:364`) | the same ground-truth one-hot on all four ego statuses (adapter `build_sample`, `kitscenes_adapter.py:296-320`) (UNVERIFIED which slot reads it; the model reads the newest) |
| acceleration | **0**: `canbus = np.zeros(18)`, only velocity set (`harness.py:559-560`); node puts `[accel, 0]` (`policy.py:303-304`) | real body-frame ax, ay from central differences (`kitscenes_adapter.py:128-154`) |
| sideways speed | **0**: node builds `[speed, 0]` (`policy.py:302-303, 383`) | real vy (`kitscenes_adapter.py:128-154`) |
| forward speed | `L.ego_speed(poses, index)`, a scalar speed (`harness.py:560`) | central-difference body-frame vx (same adapter lines) (UNVERIFIED numeric equality) |
| camera height above road | fixed **2.2 m** (`run_config.json` `ground_offset_m: 2.2`; `harness.py` comment at `:580-584`) | estimated per scene from the lidar ground return, falling back to 2.2 m (`KITScenesSceneAdapter(path)` with no offset → `estimate_ground_offset()`, `kitscenes_adapter.py:194-228`; `f_baseline.py:105`) |
| image geometry | OpenScene crop, then bilinear resize to 1920×1080 in the node (`policy.py:224-249`) | the adapter's `_crop_box` crop (`kitscenes_adapter.py:163, 241-242`), then the devkit features. Matches by code; pixel equality UNVERIFIED (RECON2 §d) |
| history | 48 source frames requested (`HISTORY_FRAMES = 48`, `harness.py:39`); the node keeps 4 statuses 0.5 s apart (`model.yaml` `ego_history_steps: 4`, `ego_history_hz: 2.0`) | 4 statuses at stride 5 frames (0.5 s) (`kitdata.py:88` note; adapter) |
| momentum | on, but never fed (E1) | none |
| attention kernel | `DRIVE_JEPA_MSDA=auto` on the GPU node | as run 2026-10-21 (UNVERIFIED; not recorded in `final_state.json`) |

### E3. Is there an existing "straight" record? **No.**

**No record exists.**
- Every evaluation of this checkpoint in our folders uses `command_mode="gt"`: `final5.py:98`, `09_heldout_comparison/final.py:174,228`,
  and `prep_export.py:224` (the compare step in PREP).
- A grep for a `"straight"` command mode in `experiments/drivejepa_kitscenes` finds none.

**What existing data does give** (gap = gate − ours, grouped by each moment's true command from the sample list):

| true command | moments | ours | gate | gap |
|---|---|---|---|---|
| straight | 184 | 0.562 | 0.829 | **0.267** |
| left | 21 | 0.752 | 1.298 | 0.546 |
| right | 18 | 0.768 | 1.263 | 0.495 |

**What this shows.**
- On the 184 moments whose true command already is "straight", the command input is the same on both sides. These
  moments still lose 0.27 m, so most of the 0.31 m overall gap comes from the other inputs in E2: acceleration, sideways
  speed, camera height and the speed computation.
- The turning moments lose about 0.25 m more each. If that extra is all command, the command accounts for about
  0.25 × 39 / 223 ≈ 0.04 m of the 0.31 m overall. This is an estimate under that assumption, not a measurement.

**What would settle it.** One CPU run, not done: the held-out loop of `final5.py` on the 223 moments with
`command_mode="straight"` and everything else unchanged, using the Drive-JEPA CPU env, about 103 s of GPU-equivalent
work, longer on CPU. Its ADE (3 s) minus 0.596 is the command's share. A second run that also zeroes acceleration and
sideways speed would isolate the rest.
