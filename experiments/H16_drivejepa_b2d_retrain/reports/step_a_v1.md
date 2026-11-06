# Phase 2 Step A: the Bench2Drive grade (05 Nov 2026)

Numbers: `step_a_numbers.json`. Grade files: `paddock-data/b2d_drivejepa_phase2/grade/<clip>.npz`, 944 clips. Teacher index: `paddock-data/b2d_drivejepa_phase2/teachers_2048.npz`, sha256 c065c90c5ae45e10139a5acd5144d87146f79a907793136da0744ef5571b9722.

## Before grading

- **17.06 m/s path:** `HighwayCutIn_Town06_Route300_Weather14`, frames 75-90. Not a glitch: the simulator's own speed reaches 17.27 m/s and distance over time 17.24 m/s, with a smooth profile. `b2d_2048.npy` is unchanged.
- **Context:** all 944 clips rebuilt to a new folder with `mtd/build_context_b2d.build_clip`, 39,148 frames. That covers the 147 training clips and the 8 held-out clips without context. Blocked flags recomputed by the existing job, 0 differences on the 17,219 frames shared with the old table.

## Choices the spec left open

- **No expert mask** on collisions or lanes, so the expert is graded by the same rules. The earlier stand-in masked them.
- **Stop sign:** Bench2Drive's 0.1 m/s is instantaneous, and a path only has 0.5 s average speeds. Stopped here means a 0.5 s segment under 1.0 m/s. On 9,698 stop-clip frames the expert's 8 poses give 289 failures at 0.1 and 11 at 1.0; its 10 Hz log gives 12.
- **Waypoints** for the stop test: nearest lane point, then straight along the lane, standing in for `wp.next`.
- **Scorer margins** (0.3 m on static objects) and the follower model are unchanged. Followers count as vehicle infractions.
- **No clean path** on a frame counts as d_ref 0, so P = 1.

## Results

| | Training | Held-out |
|---|---|---|
| Frames | 36,897 | 2,251 |
| Feasible paths, median | 499 | 424 |
| d_ref under 0.5 m | 26.7 % | 35.0 % |
| Expert reaches 0.95 | 93.9 % | 93.6 % |
| Expert mean grade | .974 | .970 |

Expert below 0.95, frames where each term fires (training 2,257 frames, held-out 145): vehicle 1,100 / 63, outside 827 / 70, static 453 / 8, pedestrian 41 / 8, stop 11 / 0, red 0 / 8, progress 0 / 0.

| Teacher supply | Training >= 1 | >= 4 | Median | Held-out >= 1 | >= 4 | Median |
|---|---|---|---|---|---|---|
| All | .981 | .912 | 44 | .969 | .868 | 52 |
| Straight | .973 | .943 | 59 | .961 | .925 | 57 |
| Turns | .954 | .669 | 7 | .934 | .528 | 4 |
| Stopped | .998 | .952 | 44 | .982 | .880 | 68 |

Terms among feasible paths, share where each fires (training / held-out): outside .525 / .540, progress under 1 .446 / .380, vehicle .338 / .336, red .072 / .073, static .061 / .045, stop .022 / .023, pedestrian .007 / .015. Clean paths .258 / .264. Teachers .110 / .133.

Speed: 21,171,222 paths in 1,353 s, 15,723 paths/s on two processes, GPU shared with M1. 3 frames have no feasible path.
