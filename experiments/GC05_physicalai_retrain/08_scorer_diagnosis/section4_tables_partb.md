**Table 4.6 — part b, the scorer's own outputs: separation of plans that pass a check from plans that fail it** (0.5 = coin flip; pooled over plans; 95 % clip-bootstrap interval, 200 resamples). "Total" = the trained total output; "own" = that check's own output, which has zero training weight; collision, drivable area and traffic light have no output of their own. Traffic light is skipped: fewer than 200 failing plans.

| check | needs the map | plans passing, UNSEEN pass 9 | shipped total, UNSEEN | shipped own, UNSEEN | pass 1 total, UNSEEN | pass 1 own, UNSEEN | pass 9 total, UNSEEN | pass 9 own, UNSEEN | pass 9 total, SEEN | pass 9 own, SEEN | shipped own, SEEN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| collision | yes | 93.6 % | 0.76 (0.72–0.80) | — | 0.80 (0.77–0.82) | — | 0.78 (0.75–0.82) | — | 0.91 (0.90–0.92) | — | — |
| drivable area | yes | 90.2 % | 0.72 (0.69–0.76) | — | 0.82 (0.79–0.84) | — | 0.83 (0.80–0.85) | — | 0.90 (0.89–0.91) | — | — |
| driving direction | yes | 90.6 % | 0.77 (0.73–0.81) | 0.48 (0.45–0.53) | 0.83 (0.81–0.85) | 0.76 (0.74–0.78) | 0.85 (0.83–0.87) | 0.49 (0.46–0.53) | 0.92 (0.91–0.93) | 0.38 (0.35–0.41) | 0.44 (0.42–0.46) |
| progress (full) | yes | 45.6 % | 0.38 (0.35–0.40) | 0.40 (0.37–0.42) | 0.63 (0.61–0.64) | 0.56 (0.54–0.57) | 0.56 (0.55–0.58) | 0.44 (0.42–0.46) | 0.59 (0.57–0.61) | 0.45 (0.43–0.46) | 0.41 (0.39–0.43) |
| time-to-collision | yes | 91.7 % | 0.62 (0.56–0.67) | 0.61 (0.58–0.65) | 0.76 (0.73–0.79) | 0.77 (0.74–0.79) | 0.74 (0.71–0.78) | 0.75 (0.71–0.78) | 0.87 (0.86–0.89) | 0.79 (0.77–0.82) | 0.64 (0.61–0.66) |
| lane keeping | yes | 67.8 % | 0.62 (0.60–0.65) | 0.51 (0.48–0.53) | 0.61 (0.58–0.65) | 0.46 (0.43–0.48) | 0.63 (0.61–0.66) | 0.45 (0.42–0.47) | 0.70 (0.68–0.71) | 0.43 (0.41–0.45) | 0.48 (0.46–0.50) |
| comfort | no | 75.4 % | 0.39 (0.35–0.42) | 0.80 (0.77–0.81) | 0.67 (0.65–0.68) | 0.65 (0.63–0.67) | 0.67 (0.65–0.69) | 0.47 (0.43–0.50) | 0.66 (0.64–0.67) | 0.46 (0.44–0.48) | 0.79 (0.77–0.81) |
| total grade ≥ 0.9 | yes | 38.6 % | 0.58 (0.56–0.61) | — | 0.75 (0.73–0.77) | — | 0.76 (0.74–0.77) | — | 0.84 (0.83–0.85) | — | — |

**Table 4.7 — part b, fresh linear probes on pass-9 features, trained on 120,000 plans from 60 % of the SEEN clips.** "Slot" = the 256 numbers per plan the scorer network reads; "feat" = the 1,024 numbers before its final layer (these fits stopped at the 300-iteration limit, so they are lower bounds for that feature set). The held-out SEEN clips were held out from the probe only: the model itself trained on them.

| check | probe on slot: SEEN held-out | probe on slot: UNSEEN | probe on feat: SEEN held-out | probe on feat: UNSEEN | pass 9 total output, UNSEEN | slot probe minus total output, UNSEEN |
|---|---|---|---|---|---|---|
| collision | 0.93 (0.92–0.95) | 0.88 (0.85–0.90) | 0.95 (0.94–0.96) | 0.84 (0.81–0.87) | 0.78 (0.75–0.82) | +0.09 |
| drivable area | 0.91 (0.89–0.93) | 0.88 (0.86–0.91) | 0.92 (0.90–0.93) | 0.87 (0.85–0.89) | 0.83 (0.80–0.85) | +0.06 |
| driving direction | 0.91 (0.89–0.93) | 0.89 (0.86–0.91) | 0.95 (0.93–0.96) | 0.89 (0.87–0.91) | 0.85 (0.83–0.87) | +0.03 |
| progress (full) | 0.86 (0.84–0.87) | 0.85 (0.84–0.86) | 0.88 (0.86–0.89) | 0.85 (0.84–0.86) | 0.56 (0.55–0.58) | +0.29 |
| time-to-collision | 0.89 (0.87–0.91) | 0.83 (0.80–0.86) | 0.92 (0.90–0.94) | 0.81 (0.78–0.85) | 0.74 (0.71–0.78) | +0.09 |
| lane keeping | 0.77 (0.74–0.79) | 0.73 (0.70–0.76) | 0.79 (0.77–0.82) | 0.73 (0.70–0.75) | 0.63 (0.61–0.66) | +0.10 |
| comfort | 0.94 (0.93–0.95) | 0.95 (0.94–0.96) | 0.94 (0.93–0.95) | 0.95 (0.94–0.96) | 0.67 (0.65–0.69) | +0.28 |
| total grade ≥ 0.9 | 0.84 (0.83–0.85) | 0.80 (0.79–0.82) | 0.89 (0.88–0.90) | 0.81 (0.79–0.83) | 0.76 (0.74–0.77) | +0.05 |
