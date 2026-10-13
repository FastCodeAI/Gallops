Clips scanned: 149773 of 153624 raw_train clips (97.5 %; the other 3851 have no `egomotion.offline`/`obstacle.offline` per `feature_presence.parquet`), 1576 chunks; scan errors 0; clip duration median 20.0 s.

Actor attachment: 4941 lateral events, status {'ok': 4941}; bumper offset source {'vehicle_dimensions': 4940, 'fallback_4.07': 1}; F min/median/max 3.66/4.06/4.09 m.

| set | events | distinct clips | per 1,000 clips scanned |
|---|---|---|---|
| all move-offs | 19398 | 19134 | 129.52 |
| lateral ≥ 1.0 m | 4941 | 4935 | 32.99 |
| lateral ≥ 1.0 m, heading change ≤ 45° | 3257 | 3255 | 21.75 |
| lateral + stationary box 0–8 m ahead of bumper, ±2 m (before turn filter) | 373 | 373 | 2.49 |
|   of which dropped as turns (> 45°) | 88 | 88 | 0.59 |
| **pull-out events (after removing turns)** | 285 | 285 | 1.9 |
| for reference: same with the rig-origin gap | 186 | 186 | 1.24 |

Rate 1.90 pull-out events per 1,000 scanned clips. Whole raw_train was scanned (no sampling): actual total 285; extrapolated to all 153624 raw_train clips incl. those without offline labels: 292.

Pull-out events: actor classes {'automobile': 277, 'person': 2, 'trailer': 2, 'protruding_object': 1, 'bus': 1, 'heavy_truck': 1, 'stroller': 1}; bumper gap median 3.21 m; lateral median 2.01 m; |heading change| median 16.8°; stop observed from clip start 141; lateral toward left 216 / right 69.

Countries of pull-out clips: {'United States': 163, 'Germany': 52, 'France': 16, 'Sweden': 15, 'Finland': 7, 'Portugal': 5}; in nurec_train: 2 clips.

Cross-check on the 737 nurec_train clips also scanned from raw: move-offs from clipgt (usdz) 144, from raw egomotion.offline 142, same (uuid, t_move ± 0.5 s) 137.

Spot-check — 10 pull-out events (seed 20261003):

| uuid | chunk | t_stop_start | t_move | t_depart | lat4_signed | dyaw4_deg | acat_bum | gap_bum | alat_bum |
|---|---|---|---|---|---|---|---|---|---|
| 35c86436-c06d-4c2f-8189-36546f201e86 | 2402 | 0 | 3.7 | 5.1 | 2.66 | 4.86 | automobile | 3.95 | -0.2 |
| 695863ef-7a28-4e3a-8faf-a677efc60a2b | 2706 | 10.5 | 13.8 | 19 | 1.73 | 40.74 | automobile | 4.52 | -1.3 |
| 7ce8500b-7b9c-4f73-96bd-1f5f2990a5da | 718 | 0 | 6.5 | 10.9 | 1.07 | 26.56 | automobile | 2.23 | -0.26 |
| 81ce115e-0c2a-4c27-9e1a-6d6716cf0ee1 | 386 | 0 | 5.7 | 8.8 | 1.64 | 18.71 | automobile | 2.03 | -0.47 |
| 8a4e8a71-24f9-48e0-b564-448de181c549 | 400 | 5.4 | 8.1 | 9.8 | -2.22 | -24.41 | automobile | 4.51 | 1.63 |
| 8f35f7db-f9ec-4d03-acc6-d91709b4fc8b | 776 | 0 | 2.6 | 4.2 | 2.65 | 6.89 | automobile | 3.34 | 0.04 |
| a3f56c5e-0c28-4bc9-a407-203b3198139b | 2831 | 0 | 2.9 | 4.1 | -2.64 | -34.72 | automobile | 3.3 | 1.04 |
| a75c5079-8a11-4cad-bdb1-fccd91678385 | 744 | 3.3 | 8.9 | 10 | 1.78 | 2.95 | automobile | 0.87 | -1.63 |
| b63218e7-9a27-4d0f-9cca-4f9565c74e8f | 403 | 0 | 7 | 10.4 | 1.84 | 40.89 | automobile | 3.71 | 0.16 |
| be89c713-08e2-49cf-9497-15e4adc9e376 | 478 | 4.9 | 7.5 | 9.3 | 3.74 | 40.33 | automobile | 2.22 | -1.91 |

| id | statement | result | verdict |
|---|---|---|---|
| P-RBL | raw_train ≥ 100 pull-out events after removing turns | 285 events in 285 clips | ENOUGH |