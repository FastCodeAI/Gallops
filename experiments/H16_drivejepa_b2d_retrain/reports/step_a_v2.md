# Phase 2 Step A, spec v2 (05 Nov 2026)

Numbers: `step_a_v2_numbers.json`, `three_frames.json`, `logs/check_expert_log.log`. Grade files: `paddock-data/b2d_drivejepa_phase2/grade_v2/<clip>.npz`, 944 clips. Teacher index: `paddock-data/b2d_drivejepa_phase2/teachers_2048_v2.npz`, sha256 8f1ed0972e6b18fa643297ca7979376beb19fe2fcc91fd00a938ff94a987a22f. v1 files are untouched.

Route length, the expert's distance over the clip: min 36.8, median 108.8, max 158.7 m; 4 clips under 50 m.

| | Training | Held-out |
|---|---|---|
| Expert reaches 0.95 | 94.4 % (v1 93.9) | 95.1 % (v1 93.6) |
| Expert mean grade | .981 | .983 |
| Expert moves under 0.5 m | 24.5 % | 32.1 % |

Expert below 0.95 (training 2,072 frames, held-out 110), frames where each term fires: vehicle 1,100 / 63, outside 642 / 35, static 453 / 8, pedestrian 41 / 8, stop 11 / 0, red 0 / 8, progress 0 / 0.

| Teacher supply | Training >= 1 | >= 4 | Median | Held-out >= 1 | >= 4 | Median |
|---|---|---|---|---|---|---|
| All | .964 | .916 | 53 | .958 | .882 | 64 |
| Straight | .973 | .949 | 71 | .968 | .938 | 68 |
| Turns | .864 | .670 | 8 | .761 | .518 | 4 |
| Stopped | .984 | .954 | 52 | .985 | .899 | 79 |

Terms among feasible paths, share where each fires (training / held-out): outside .526 / .540, progress under 1 .471 / .412, vehicle .338 / .336, red .072 / .073, static .061 / .045, stop .022 / .023, pedestrian .007 / .015. Clean .258 / .264. Teachers .127 / .157.

Speed: 29,193 paths/s on two processes, after M1 finished (v1: 15,723 beside M1).

## The expert's collisions against its log

39,148 frames, frames with a hit. Path = the grade's view (8 poses, scorer margins). Log = 10 Hz log, same margins. True = 10 Hz log, real boxes, no margin.

| | Path | Log | True | Path and log | Path and true | Path, not true |
|---|---|---|---|---|---|---|
| Pedestrian | 49 | 17 | 17 | 17 | 17 | 32 |
| Vehicle | 1,131 | 1,121 | 763 | 981 | 657 | 474 |
| Static | 461 | 406 | 44 | 402 | 44 | 417 |

42 frames carry a vehicle infraction with no physical contact on the path: the follower rule.

Ego box after the rear-axle shift: the expert's path through the shift equals the context's expert to within 0.0 m. The context's centre against the anno box centre, over 240 frames: mean -0.008 m along, -0.000 m across; largest 0.045 m and 0.038 m. Anno half extents 2.4462 x 0.9184 m, scorer 2.446 x 0.918 m. Rear-axle offset 1.3886 m. Actor location = box centre.

## Three frames: expert passes through the oncoming lane

A path stops when its last 0.5 s segment is under 1.0 m/s (244 of the 2,048). One frame per scenario, the most oncoming distance on the expert's path, ego at 3 m/s or more.

| Frame | | Expert S | Best stopping path | Best path | Teachers (stop / go) |
|---|---|---|---|---|---|
| HazardAtSideLaneTwoWays Route1147, 18000, 3.1 m/s | v1 | .054 | .537 | 1.000 | 1 (0 / 1) |
| | v2 | .781 | .588 | .966 | 1 (0 / 1) |
| AccidentTwoWays Route1103, 30500, 3.5 m/s | v1 | .226 | .623 | .820 | 0 |
| | v2 | .822 | .568 | .938 | 0 |
| VehicleOpensDoorTwoWays Route1203, 21500, 3.3 m/s | v1 | .248 | 1.000 | 1.000 | 13 (1 / 12) |
| | v2 | .930 | 1.000 | 1.000 | 22 (3 / 19) |
