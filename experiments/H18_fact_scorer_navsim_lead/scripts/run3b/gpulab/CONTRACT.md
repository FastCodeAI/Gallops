# gpulab: run 3B live labels on the GPU (18 Nov 2026, Navvrat's go: "yeah lets do that ... so that training becomes feasible in 2 days")

Goal: replace the CPU live labeller (run3b/r3b/labeller.py, 14 NAVSIM + 14 CARLA worker processes, ~77 CPU-s per
training step of 64 samples x 32 planner paths) with a batched torch (float64, CUDA) labeller that gives THE SAME labels.
The CPU code stays the reference and is never edited.

## What is labelled live (per training step)
64 samples (32 NAVSIM + 32 LEAD/CARLA), each with P = 32 planner paths: poses (P, 8, 3) float64, rear axle, ego frame at t,
waypoints 0.5 .. 4.0 s. Outputs per sample (the CPU reference = labeller.py `_nav_env().label` / `_carla_env().label`):
  hit, offroad, left_route, stop, wrong_way (P, 8) bool
  red (P, 8) int8 (1 red, -1 unknown, 0)
  metres, lane_off, ttc, cutin_brake (P, 8) float  (ttc: inf allowed; cutin_brake nan -> 0)
  comfort_step (P, 8) bool (NAVSIM: facts comfort max over the 6 >= 1; CARLA: all False)
  NAVSIM only, aux = run 3A's get_sub_score outputs for the same poses: key_corners, key_valid, ego_areas, map_aux
Rows: the CPU code labels row 0 (PDM-Closed / LEAD's logged future) + the P paths and returns rows 1..; the GPU code only
needs the P paths (check nothing for rows 1.. depends on row 0 beyond the shared start state).

## Layout (each part owns one folder; never edit another part's folder, run 3A, facts/, carla_*/, or run3b/r3b)
  gpulab/rollout/     extract_rollout(kind, ctx) -> dict (CPU, numpy, pose-independent)
                      simulate(kind, scenes, poses) -> states (B, P, 41, 11) float64 cuda  (= cns.before_score_proposals rows 1..)
                      ego_coords(kind, states) -> (B, P, 41, 5, 2)  (= scorer._ego_coords: BBCoordsIndex order, last = centre)
  gpulab/mapfacts/    extract_map(kind, ctx) -> dict
                      label_map(kind, scenes, states, coords) -> dict: offroad, wrong_way, left_route, lane_off, metres, red,
                      stop, ego_areas (B, P, 41, n_area) bool (scorer._ego_areas, EgoAreaIndex order); NAVSIM: aux map_aux,
                      aux ego_areas
  gpulab/agentfacts/  extract_agents(kind, ctx) -> dict
                      label_agents(kind, scenes, states, coords, ego_areas) -> dict: hit, ttc, cutin_brake, comfort_step;
                      NAVSIM: aux key_corners, key_valid
kind is "nav" or "carla". scenes = list (len B) of the dicts the extract function returned, one per sample. Extract runs in
the CPU prep workers that already hold ctx (= labeller._nav_env()/_carla_env() context(meta)); it must not depend on poses.
Keep scene dicts compact (plain numpy arrays; no shapely / nuPlan objects; they cross process queues).

## Development inputs
- Contexts and the CPU reference: `import labeller as LB` (run3b/r3b); `context, label = LB._nav_env()` (or `_carla_env()`)
  in-process. NAVSIM and CARLA MUST be separate processes (carla_facts patches run 3A constants on import).
- Metas: see run3b/tests/test_labeller.py (NAVSIM from pack/navsim_meta.pkl, LEAD from lead_inputs_v2/routes/*.pkl.gz).
- Poses for tests: anchors data/8192.npy[:, 4::5] (+ anchors_lead_K1024 for CARLA) and noisy copies, including paths that
  stop, reverse-ish, swerve off road, cross stop lines and hit things (use scenes where the CPU reference says yes).
- mapfacts/agentfacts develop on CPU-reference states (cns.before_score_proposals / the CARLA car model, scorer._ego_coords,
  scorer._ego_areas) so they do not wait for rollout; integration swaps in the GPU ones.

## Acceptance (each part reports, in its folder, verify_<part>.json + a short REPORT.md)
- >= 200 NAVSIM + >= 200 CARLA samples x 32 paths, train and held-out.
- bool / int facts: number and share of (path, step) cells that differ from the CPU reference, per fact, every mismatch
  listed with its cause; target 0 except true boundary ties (a point exactly on an edge, a contact at exactly 0 gap).
- numbers: max and p99.9 abs diff; target <= 1e-6 (float64).
- speed: GPU label time for 64 samples x 32 paths (excluding extract), and extract CPU-s per sample.
- GPU memory peak for 64 x 32.

## Box rules (shared machine; a training run is live on this GPU, pgid 3646049: never touch it)
- GPU: torch.cuda.set_per_process_memory_fraction so a test process stays <= 6 GB.
- CPU: tests `nice -n 10 ionice -c3`, <= 4 processes; RAM <= 6 GB per part.
- Never read {data.nas}. Never grep/find/du recursively on {data.hdd} big trees (reading named files is fine).
- Kill only explicit PIDs you started; never pkill/pgrep -f patterns.
- No git, no downloads, no edits outside your folder. Ops files only under {runs}/run3b/gpulab/.
