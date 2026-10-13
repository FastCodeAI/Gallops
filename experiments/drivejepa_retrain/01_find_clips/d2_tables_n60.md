Scenes analysed: 60 of 1447 pool scenes (load errors 0); anchors 1916; clip duration min/median/max 19.1/20.0/20.0 s; scenes with < 32 anchors: 3.

### D-0 — speed envelope

| window-min speed m/s | anchors | scenes |
|---|---|---|
| 0–2 | 474 | 30 |
| 2–4 | 146 | 32 |
| 4–6 | 249 | 38 |
| 6–8 | 278 | 36 |
| 8–10 | 147 | 28 |
| 10–12 | 173 | 26 |
| 12–14 | 119 | 18 |
| 14–16 | 69 | 13 |
| 16–18 | 57 | 8 |
| 18–20 | 31 | 6 |
| 20–22 | 17 | 4 |
| 22–24 | 12 | 3 |
| 24–26 | 33 | 3 |
| 26–28 | 16 | 3 |
| 28–30 | 20 | 2 |
| 30–32 | 17 | 3 |
| 32–34 | 28 | 1 |
| 34–36 | 0 | 0 |
| 36–38 | 30 | 1 |

Top bin reached: 36–38 m/s. Samples > 50 m/s: scenes with vmax > 50 = 0; anchors with window-min ≥ 50 = 0.

Per scene: max speed median 12.7, max 36.9 m/s; max speed held 4.5 s median/p90/p99 11.3/22.7/35.0, max 36.7 m/s.

| uuid | max held 4.5 s m/s | max m/s | clip_ratings ego_speed |
|---|---|---|---|
| 06b3e399-a820-49ef-9e69-134cfdb7652a | 36.69 | 36.89 | high |
| 0009402a-a514-443b-9a4c-0e792f5ae581 | 33.87 | 34.4 | high |
| 0051f106-98bf-4001-825b-dff634ec8f18 | 31.98 | 32.21 | high |
| 0208defb-0502-4293-ba29-9273b1a4ee7f | 30.6 | 31.43 | high |
| 05855d57-2dbf-4ca7-a04d-94feb64a6e11 | 26.89 | 27.41 | high |
| 000ff49d-aa30-46ee-af57-b4a0c1143f55 | 23.78 | 24.32 | high |
| 02183dc6-33b5-4f43-b8a3-986af09da831 | 22.57 | 23.13 | high |
| 0026708c-a386-4dd2-82b1-286da952c68f | 20.88 | 22.16 | high |
| 00040136-e651-4abd-991d-0655ccda9430 | 18.7 | 20.53 | high |
| 01132ab5-90d1-4fb4-b823-ce013ec527ec | 18.43 | 18.81 | high |

### D-A — high speed (whole 4.5 s window ≥ 16 m/s, no cap)

Distinct scenes 12; anchors 261; anchors per qualifying scene median 24 (min 6, max 32); scenes qualifying on every anchor 6; per-scene window min/max speed range 16.0–36.9 m/s (per-scene values in d2_scenes.csv).

| bin m/s | scenes | anchors | brake a | follow a | curve a | cruise a | brake sc | follow sc | curve sc | cruise sc |
|---|---|---|---|---|---|---|---|---|---|---|
| 16–18 | 8 | 57 | 6 | 20 | 0 | 34 | 3 | 2 | 0 | 5 |
| 18–20 | 6 | 31 | 8 | 13 | 0 | 16 | 2 | 2 | 0 | 3 |
| 20–22 | 4 | 17 | 4 | 4 | 0 | 11 | 2 | 1 | 0 | 3 |
| 22–24 | 3 | 12 | 1 | 0 | 0 | 11 | 1 | 0 | 0 | 3 |
| 24–26 | 3 | 33 | 0 | 16 | 0 | 17 | 0 | 2 | 0 | 3 |
| 26–28 | 3 | 16 | 0 | 0 | 0 | 16 | 0 | 0 | 0 | 3 |
| 28–30 | 2 | 20 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 2 |
| 30–32 | 3 | 17 | 0 | 4 | 0 | 13 | 0 | 1 | 0 | 2 |
| 32–34 | 1 | 28 | 0 | 28 | 0 | 0 | 0 | 1 | 0 | 0 |
| 34–36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 36–38 | 1 | 30 | 0 | 0 | 0 | 30 | 0 | 0 | 0 | 1 |

Tags, pool total (anchors / scenes): brake 19 / 3; follow 85 / 5; curve 0 / 0; cruise 168 / 10. Multi-tag anchors 11.

Bins with ≥ 1 scene: 10. Highest bin with ≥ 5 distinct scenes: 18–20 m/s.

| rating | no D-A anchor | has D-A anchor | share with D-A |
|---|---|---|---|
| extremely slow | 1 | 0 | 0 |
| high | 2 | 12 | 0.86 |
| medium | 24 | 0 | 0 |
| slow | 21 | 0 | 0 |

Share of D-A scenes rated 'high': 1.00.

### D-B — move-off from standstill

| set | events | distinct scenes |
|---|---|---|
| all move-offs | 8 | 8 |
| lateral ≥ 1.0 m (4 s from move onset), no actor condition | 2 | 2 |
| stationary actor 0–8 m ahead, ±2 m | 0 | 0 |
| **lateral subset** (lateral ≥ 1.0 m AND actor) | 0 | 0 |
| alt: lateral ≥ 1.0 m measured 4 s from t_depart, no actor condition | 4 | 4 |
| alt: lateral from t_depart AND actor | 0 | 0 |

Standstill observed from clip start (stop onset censored): 7 of 8. Stop-to-depart delay median 1.3 s. Lateral (from move onset) median 0.35 m, p90 1.59 m. 4 s window truncated by clip end: 0. Actor classes: {}. Rating of move-off scenes: {'slow': 6, 'extremely slow': 1, 'medium': 1}.

### D-D — overlap

| set | scenes |
|---|---|
| A (D-A anchors) | 12 |
| B (any move-off) | 8 |
| B lateral subset | 0 |
| A ∩ B | 0 |
| neither A nor B (ordinary candidates) | 40 |

### D-E — spot-check list (front mp4 at `sample_set/26.04_release/<uuid>/camera_front_wide_120fov.mp4`, t = seconds from clip start)

| D-A bin | uuid | t s | window-min m/s | tags |
|---|---|---|---|---|
| 16–18 | 000ff49d-aa30-46ee-af57-b4a0c1143f55 | 10 | 17.3 | cruise |
| 16–18 | 0580c069-aeb6-44e0-8bbc-940757e4071a | 3 | 17.2 | cruise |
| 16–18 | 00040136-e651-4abd-991d-0655ccda9430 | 2.5 | 17.3 | brake |
| 18–20 | 01132ab5-90d1-4fb4-b823-ce013ec527ec | 3 | 18.2 | cruise |
| 18–20 | 0026708c-a386-4dd2-82b1-286da952c68f | 11.5 | 19.6 | brake+follow |
| 18–20 | 02183dc6-33b5-4f43-b8a3-986af09da831 | 5 | 18.4 | brake |
| 20–22 | 0026708c-a386-4dd2-82b1-286da952c68f | 10.5 | 20.5 | brake+follow |
| 20–22 | 02183dc6-33b5-4f43-b8a3-986af09da831 | 3.5 | 21.3 | brake |
| 20–22 | 0051f106-98bf-4001-825b-dff634ec8f18 | 4.5 | 21.2 | cruise |
| 22–24 | 0051f106-98bf-4001-825b-dff634ec8f18 | 6.5 | 23 | cruise |
| 22–24 | 02183dc6-33b5-4f43-b8a3-986af09da831 | 2.5 | 22.5 | cruise |
| 22–24 | 000ff49d-aa30-46ee-af57-b4a0c1143f55 | 15 | 23.1 | cruise |
| 24–26 | 0208defb-0502-4293-ba29-9273b1a4ee7f | 2 | 25.1 | cruise |
| 24–26 | 0051f106-98bf-4001-825b-dff634ec8f18 | 8 | 25 | cruise |
| 24–26 | 05855d57-2dbf-4ca7-a04d-94feb64a6e11 | 10.5 | 25.5 | follow |
| 26–28 | 05855d57-2dbf-4ca7-a04d-94feb64a6e11 | 2.5 | 26.5 | cruise |
| 26–28 | 0208defb-0502-4293-ba29-9273b1a4ee7f | 4.5 | 27.3 | cruise |
| 26–28 | 0051f106-98bf-4001-825b-dff634ec8f18 | 10 | 27.4 | cruise |
| 28–30 | 0208defb-0502-4293-ba29-9273b1a4ee7f | 9.5 | 28.8 | cruise |
| 28–30 | 0051f106-98bf-4001-825b-dff634ec8f18 | 12 | 29.3 | cruise |
| 30–32 | 0208defb-0502-4293-ba29-9273b1a4ee7f | 15 | 30.6 | cruise |
| 30–32 | 0051f106-98bf-4001-825b-dff634ec8f18 | 14.5 | 31.4 | cruise |
| 30–32 | 0009402a-a514-443b-9a4c-0e792f5ae581 | 1.5 | 31.5 | follow |
| 32–34 | 0009402a-a514-443b-9a4c-0e792f5ae581 | 9.5 | 33.3 | follow |
| 36–38 | 06b3e399-a820-49ef-9e69-134cfdb7652a | 8 | 36.5 | cruise |

D-B (lateral subset first, then filled with the largest-lateral move-offs without a qualifying actor):

| uuid | t_stop_start | t_move | t_depart | lat4_signed | actor | actor_cat | actor_gap |
|---|---|---|---|---|---|---|---|
| 066bb895-3f45-4b92-a67a-382e5fd52516 | 0 | 8.6 | 10.1 | 2.17 | False |  |  |
| 00442956-c080-4b2e-b9c7-647973969c86 | 0 | 8 | 10.1 | 1.34 | False |  |  |
| 0494c646-c537-4c6b-98b9-96c7e91fd0a8 | 0 | 3.5 | 4.6 | -0.82 | False |  |  |
| 031f43ff-bc87-4cf3-99a1-6aef48e6628f | 0 | 7.8 | 8.7 | -0.57 | False |  |  |
| 03376794-ae4a-4d16-a804-4efa954eec5f | 0 | 7.9 | 8.9 | -0.14 | False |  |  |

### Pre-registered

| id | statement | result | verdict |
|---|---|---|---|
| P-A | ≥ 30 distinct scenes with D-A anchors, spanning ≥ 4 bins | 12 scenes, 10 bins | THIN |
| P-A2 | highest 2 m/s bin with ≥ 5 scenes (descriptive) | 18–20 m/s | — |
| P-B | ≥ 30 move-off events | 8 events in 8 scenes | THIN |
| P-BL | ≥ 15 lateral-subset events | 0 events in 0 scenes | THIN |
