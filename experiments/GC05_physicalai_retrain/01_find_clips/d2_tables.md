Scenes analysed: 1447 of 1447 pool scenes (load errors 0); anchors 46252; clip duration min/median/max 19.1/20.0/20.0 s; scenes with < 32 anchors: 44.

### D-0 — speed envelope

| window-min speed m/s | anchors | scenes |
|---|---|---|
| 0–2 | 10783 | 655 |
| 2–4 | 4141 | 719 |
| 4–6 | 4839 | 762 |
| 6–8 | 4790 | 742 |
| 8–10 | 3960 | 625 |
| 10–12 | 3680 | 483 |
| 12–14 | 2544 | 347 |
| 14–16 | 1729 | 248 |
| 16–18 | 1492 | 182 |
| 18–20 | 1110 | 136 |
| 20–22 | 1369 | 121 |
| 22–24 | 1048 | 101 |
| 24–26 | 1124 | 98 |
| 26–28 | 896 | 82 |
| 28–30 | 969 | 75 |
| 30–32 | 644 | 55 |
| 32–34 | 408 | 32 |
| 34–36 | 460 | 25 |
| 36–38 | 248 | 15 |
| 38–40 | 18 | 2 |

Top bin reached: 38–40 m/s. Samples > 50 m/s: scenes with vmax > 50 = 0; anchors with window-min ≥ 50 = 0.

Per scene: max speed median 12.8, max 40.8 m/s; max speed held 4.5 s median/p90/p99 11.1/26.3/36.0, max 39.0 m/s.

| uuid | max held 4.5 s m/s | max m/s | clip_ratings ego_speed |
|---|---|---|---|
| 3a107de6-a55c-4f0a-81aa-e9af966f81d2 | 39.05 | 40.8 | high |
| 4cb05d65-b2b9-48fc-8f14-d0507d6cda9e | 38.53 | 38.65 | high |
| 170a3280-64dc-488a-bd73-0a5f6fc5617a | 37.86 | 38.29 | high |
| aa5f29c2-e81e-4d45-88d8-88db3d6ca701 | 37.63 | 40.63 | high |
| 43b053e4-ae10-4b88-8338-6f108cfa805d | 37.59 | 38.11 | high |
| 32be03e4-a072-4da5-b597-114cffa52af8 | 37.12 | 38.31 | high |
| 82f67f40-acba-4083-991b-f63355300952 | 36.7 | 37.52 | high |
| 06b3e399-a820-49ef-9e69-134cfdb7652a | 36.69 | 36.89 | high |
| d0d8a2b6-ac68-4e21-ae93-b30284b3fe2f | 36.51 | 36.68 | high |
| 1f0db45a-1da6-4d5a-aa58-7905dfc2051d | 36.51 | 37.76 | high |

### D-A — high speed (whole 4.5 s window ≥ 16 m/s, no cap)

Distinct scenes 404; anchors 9786; anchors per qualifying scene median 32 (min 1, max 32); scenes qualifying on every anchor 253; per-scene window min/max speed range 16.0–40.8 m/s (per-scene values in d2_scenes.csv).

| bin m/s | scenes | anchors | brake a | follow a | curve a | cruise a | brake sc | follow sc | curve sc | cruise sc |
|---|---|---|---|---|---|---|---|---|---|---|
| 16–18 | 182 | 1492 | 225 | 336 | 197 | 859 | 71 | 44 | 38 | 107 |
| 18–20 | 136 | 1110 | 152 | 180 | 150 | 724 | 47 | 29 | 30 | 90 |
| 20–22 | 121 | 1369 | 147 | 383 | 231 | 752 | 40 | 33 | 23 | 84 |
| 22–24 | 101 | 1048 | 111 | 365 | 178 | 512 | 29 | 33 | 22 | 61 |
| 24–26 | 98 | 1124 | 79 | 399 | 178 | 551 | 19 | 34 | 21 | 68 |
| 26–28 | 82 | 896 | 55 | 190 | 61 | 632 | 16 | 21 | 9 | 62 |
| 28–30 | 75 | 969 | 71 | 113 | 16 | 800 | 13 | 15 | 2 | 63 |
| 30–32 | 55 | 644 | 34 | 111 | 13 | 486 | 9 | 12 | 2 | 45 |
| 32–34 | 32 | 408 | 22 | 54 | 0 | 332 | 6 | 5 | 0 | 28 |
| 34–36 | 25 | 460 | 44 | 17 | 0 | 399 | 4 | 3 | 0 | 25 |
| 36–38 | 15 | 248 | 6 | 2 | 0 | 240 | 1 | 1 | 0 | 15 |
| 38–40 | 2 | 18 | 3 | 0 | 0 | 15 | 1 | 0 | 0 | 2 |

Tags, pool total (anchors / scenes): brake 949 / 126; follow 2150 / 120; curve 1024 / 86; cruise 6302 / 322. Multi-tag anchors 605.

Bins with ≥ 1 scene: 12. Highest bin with ≥ 5 distinct scenes: 36–38 m/s.

| rating | no D-A anchor | has D-A anchor | share with D-A |
|---|---|---|---|
| extremely slow | 30 | 0 | 0 |
| high | 70 | 368 | 0.84 |
| medium | 470 | 36 | 0.07 |
| slow | 473 | 0 | 0 |

Share of D-A scenes rated 'high': 0.91.

### D-B — move-off from standstill

| set | events | distinct scenes |
|---|---|---|
| all move-offs | 258 | 254 |
| lateral ≥ 1.0 m (4 s from move onset), no actor condition | 59 | 59 |
| stationary actor 0–8 m ahead, ±2 m | 6 | 6 |
| **lateral subset** (lateral ≥ 1.0 m AND actor) | 3 | 3 |
| alt: lateral ≥ 1.0 m measured 4 s from t_depart, no actor condition | 107 | 105 |
| alt: lateral from t_depart AND actor | 4 | 4 |

Standstill observed from clip start (stop onset censored): 181 of 258. Stop-to-depart delay median 1.3 s. Lateral (from move onset) median 0.16 m, p90 2.13 m. 4 s window truncated by clip end: 8. Actor classes: {'automobile': 5, 'person': 1}. Rating of move-off scenes: {'slow': 198, 'medium': 49, 'extremely slow': 6, 'high': 1}.

Lateral-subset events:

| uuid | t_stop_start | t_move | t_depart | lat4_signed | actor_cat | actor_gap | actor_lat |
|---|---|---|---|---|---|---|---|
| fab24f4f-5d90-4460-bd85-7850faf12739 | 6 | 8 | 10.2 | 2.47 | automobile | 7.63 | -0.21 |
| 81ce115e-0c2a-4c27-9e1a-6d6716cf0ee1 | 0 | 5.7 | 8.8 | 1.64 | automobile | 6.07 | -0.5 |
| febd36c4-abb1-498c-9e79-7f4d90d0285f | 5.7 | 8.6 | 10 | 1.1 | person | 5.53 | -1.24 |

### D-D — overlap

| set | scenes |
|---|---|
| A (D-A anchors) | 404 |
| B (any move-off) | 254 |
| B lateral subset | 3 |
| A ∩ B | 6 |
| neither A nor B (ordinary candidates) | 795 |

### D-E — spot-check list (front mp4 at `sample_set/26.04_release/<uuid>/camera_front_wide_120fov.mp4`, t = seconds from clip start)

| D-A bin | uuid | t s | window-min m/s | tags |
|---|---|---|---|---|
| 16–18 | 96cc7239-9216-4579-aa86-40f9f36280ea | 5.5 | 16.6 | cruise |
| 16–18 | 28f20041-1346-4c7c-b76f-02b000c81324 | 13 | 16.2 | follow |
| 16–18 | 1182b9c0-56b4-4c28-b7c3-5c9de0256699 | 3 | 17.1 | follow |
| 18–20 | 132ae6cb-1e58-42aa-8bf3-4381eb958e43 | 16 | 19.9 | cruise |
| 18–20 | 1a89339b-48d2-42c9-9b6b-277dd098112a | 5.5 | 19.1 | brake |
| 18–20 | 1a394766-ea99-4cf5-98de-d1821222612b | 3.5 | 18.8 | brake |
| 20–22 | 87a3b6d4-fd0f-487f-b409-f8ef2fe53681 | 9 | 21.9 | cruise |
| 20–22 | a9162b2d-b0f6-4c56-9642-c96a11e69fdb | 11 | 21.4 | curve |
| 20–22 | 743997a7-f90e-44cc-873a-2b4f38496ea7 | 8 | 20.2 | follow |
| 22–24 | 1a89339b-48d2-42c9-9b6b-277dd098112a | 1 | 22.7 | brake+curve |
| 22–24 | dd9fd9d1-f3cd-4bae-8093-b1273eddba08 | 13 | 22.1 | follow |
| 22–24 | 5a23ec1c-1c6f-4da6-ae62-c31677896960 | 15 | 23.3 | cruise |
| 24–26 | 7ee3f88d-0da8-4e7f-a0a2-e2798392cb34 | 11 | 25.1 | cruise |
| 24–26 | 504ee5fd-7588-4ec5-b8f3-2bde1b06c05a | 12.5 | 24.7 | follow |
| 24–26 | 63ac00e5-2b62-4b85-acd5-b1594313ffc9 | 1 | 25.7 | follow |
| 26–28 | 6fc87bb8-794b-4884-a1f8-50193c9b6357 | 6.5 | 26.4 | cruise |
| 26–28 | 3dbcdf90-5dc0-4797-94ff-e6bdf47adc14 | 16 | 26.2 | brake |
| 26–28 | bd6387a0-ca7d-4c02-a684-90c141acbc19 | 9.5 | 27.2 | cruise |
| 28–30 | 4d4352f8-8565-42a8-99ac-8b2c790d2591 | 8 | 28.8 | cruise |
| 28–30 | 3d8253d9-9382-40e7-964c-5cafa6f24efc | 2 | 29 | cruise |
| 28–30 | ca97c826-dfb7-4fec-8087-b9fb69987d7b | 0.5 | 29.7 | cruise |
| 30–32 | c131bc24-ae60-4d16-a60c-2f4501763269 | 13 | 30.9 | cruise |
| 30–32 | 4d496419-40e7-4822-ae8a-1af44a22bd83 | 0.5 | 30 | cruise |
| 30–32 | d8ba63a8-1ac8-453f-9f33-a8511914621a | 9 | 30.6 | cruise |
| 32–34 | 4fdce89c-8dc9-4c3a-95e5-1a6c4da202e2 | 4 | 32.9 | cruise |
| 32–34 | 92e0d11d-184c-4614-881c-2325ba7bae5a | 10.5 | 32.6 | cruise |
| 32–34 | 7e27eb97-a5db-45f1-bac2-3a62089c1183 | 7.5 | 32.7 | follow |
| 34–36 | 3d31d122-c1fe-46b9-bcc7-9e8c6c93bd7f | 2 | 34.6 | cruise |
| 34–36 | d2a7ae02-b6ed-4ffc-b245-be1b4b676c8f | 8.5 | 34.9 | brake |
| 34–36 | b45734f4-01d2-48f7-9296-2ae79a7a89b9 | 3 | 34.3 | cruise |
| 36–38 | 1f0db45a-1da6-4d5a-aa58-7905dfc2051d | 14 | 36.2 | cruise |
| 36–38 | 06b3e399-a820-49ef-9e69-134cfdb7652a | 8 | 36.5 | cruise |
| 36–38 | 47650c67-906f-4a73-a012-be29fad601ff | 15 | 36.3 | cruise |
| 38–40 | 3a107de6-a55c-4f0a-81aa-e9af966f81d2 | 7 | 38.6 | brake |
| 38–40 | 4cb05d65-b2b9-48fc-8f14-d0507d6cda9e | 4 | 38.1 | cruise |

D-B (lateral subset first, then filled with the largest-lateral move-offs without a qualifying actor):

| uuid | t_stop_start | t_move | t_depart | lat4_signed | actor | actor_cat | actor_gap |
|---|---|---|---|---|---|---|---|
| fab24f4f-5d90-4460-bd85-7850faf12739 | 6 | 8 | 10.2 | 2.47 | True | automobile | 7.63 |
| 81ce115e-0c2a-4c27-9e1a-6d6716cf0ee1 | 0 | 5.7 | 8.8 | 1.64 | True | automobile | 6.07 |
| febd36c4-abb1-498c-9e79-7f4d90d0285f | 5.7 | 8.6 | 10 | 1.1 | True | person | 5.53 |
| d54d0ed1-a494-42c6-bfb0-ffde27c7ef57 | 0 | 8.2 | 9.3 | 12.32 | False |  |  |
| fd01e9ab-05d6-445a-b974-02a33793ab49 | 0 | 4.8 | 5.7 | 10.44 | False |  |  |

### Pre-registered

| id | statement | result | verdict |
|---|---|---|---|
| P-A | ≥ 30 distinct scenes with D-A anchors, spanning ≥ 4 bins | 404 scenes, 12 bins | ENOUGH |
| P-A2 | highest 2 m/s bin with ≥ 5 scenes (descriptive) | 36–38 m/s | — |
| P-B | ≥ 30 move-off events | 258 events in 254 scenes | ENOUGH |
| P-BL | ≥ 15 lateral-subset events | 3 events in 3 scenes | THIN |
