## Main table (223 held-out moments; mean, with median in brackets for ALL)

Groups: ALL (n = 223), SLOW (n = 221), FAST (n = 2, too few to read), SLOW TURNS (n = 37).

| Model | ALL: closest candidate ADE (3 s) | ALL: closest candidate FDE (3 s) | ALL: candidates within 2 m | ALL: variety | SLOW: closest candidate ADE (3 s) / FDE (3 s) | FAST: closest candidate ADE (3 s) / FDE (3 s) | SLOW TURNS: closest candidate ADE (3 s) / FDE (3 s) | scorer's pick ADE (3 s) | scorer's pick FDE (1 s) | scorer's pick FDE (2 s) | scorer's pick FDE (3 s) | SLOW TURNS tails vs Encoder only, tuned on KIT: worse > 0.5 m on FDE · largest increase (moment) · mean FDE of worst 10 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Released model | 0.374 (0.301) | 0.664 (0.421) | 17.5 | 2.83 | 0.357 / 0.628 | 2.337 / 4.696 ‡ | 0.558 / 1.129 | 2.372 (1.392) | 1.314 | 2.679 | 4.206 | 3 · +0.78 m (`873eebf4` t = 130) · 5.16 |
| Encoder only, tuned on KIT | 0.378 (0.307) | 0.662 (0.426) | 17.7 | 2.82 | 0.359 / 0.624 | 2.523 / 4.949 ‡ | 0.557 / 1.192 | 1.879 (1.246) | 1.002 | 2.085 | 3.442 | — · 5.75 |
| Generator only, trained on KIT | 0.194 (0.146) | 0.379 (0.265) | 25.9 | 1.63 | 0.194 / 0.380 | 0.156 / 0.320 ‡ | 0.350 / 0.701 | 1.146 (0.925) | 0.319 | 1.223 | 2.644 | 1 · +0.60 m (`8dcfc2ba` t = 40) · 2.80 |
| Generator and encoder trained together | 0.190 (0.138) | 0.377 (0.243) | 26.0 | 1.60 | 0.189 / 0.377 | 0.267 / 0.329 ‡ | 0.346 / 0.750 | 1.101 (0.833) | 0.313 | 1.169 | 2.533 | 3 · +0.78 m (`6f47eb66` t = 490) · 2.85 |

‡ too few to read (fewer than 20 moments).

## Paired differences (second model minus first; negative = the newer model is better)

95 % interval: paired bootstrap, 10,000 resamples of whole scenes, seed 0.

| Pair | Metric | Group | n | mean (m) | median (m) | better / worse by > 10 cm | 95 % interval (m) |
|---|---|---|---|---|---|---|---|
| Encoder only, tuned on KIT vs Released model | closest candidate ADE (3 s) | ALL | 223 | +0.004 | +0.002 | 19 / 19 | [-0.009, +0.018] |
| Encoder only, tuned on KIT vs Released model | closest candidate ADE (3 s) | SLOW | 221 | +0.002 | +0.002 | 19 / 18 | [-0.011, +0.017] |
| Encoder only, tuned on KIT vs Released model | closest candidate ADE (3 s) | FAST ‡ | 2 | +0.186 | +0.186 | 0 / 1 | [-0.006, +0.379] |
| Encoder only, tuned on KIT vs Released model | closest candidate ADE (3 s) | SLOW TURNS | 37 | -0.000 | -0.014 | 8 / 3 | [-0.058, +0.071] |
| Encoder only, tuned on KIT vs Released model | scorer's pick ADE (3 s) | ALL | 223 | -0.493 | -0.053 | 96 / 55 | [-0.687, -0.313] |
| Encoder only, tuned on KIT vs Released model | scorer's pick ADE (3 s) | SLOW | 221 | -0.497 | -0.053 | 95 / 54 | [-0.696, -0.318] |
| Encoder only, tuned on KIT vs Released model | scorer's pick ADE (3 s) | FAST ‡ | 2 | +0.050 | +0.050 | 1 / 1 | [-0.176, +0.276] |
| Encoder only, tuned on KIT vs Released model | scorer's pick ADE (3 s) | SLOW TURNS | 37 | -0.455 | -0.050 | 14 / 7 | [-0.864, -0.088] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | closest candidate ADE (3 s) | ALL | 223 | -0.185 | -0.143 | 138 / 8 | [-0.231, -0.146] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | closest candidate ADE (3 s) | SLOW | 221 | -0.165 | -0.142 | 136 / 8 | [-0.199, -0.136] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | closest candidate ADE (3 s) | FAST ‡ | 2 | -2.368 | -2.368 | 2 / 0 | [-3.331, -1.404] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | closest candidate ADE (3 s) | SLOW TURNS | 37 | -0.207 | -0.115 | 21 / 2 | [-0.389, -0.080] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | scorer's pick ADE (3 s) | ALL | 223 | -0.733 | -0.398 | 144 / 58 | [-0.930, -0.545] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | scorer's pick ADE (3 s) | SLOW | 221 | -0.702 | -0.383 | 142 / 58 | [-0.888, -0.518] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | scorer's pick ADE (3 s) | FAST ‡ | 2 | -4.250 | -4.250 | 2 / 0 | [-5.749, -2.750] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | scorer's pick ADE (3 s) | SLOW TURNS | 37 | -1.026 | -0.594 | 29 / 8 | [-1.601, -0.539] |
| Generator and encoder trained together vs Generator only, trained on KIT | closest candidate ADE (3 s) | ALL | 223 | -0.004 | -0.007 | 7 / 9 | [-0.013, +0.004] |
| Generator and encoder trained together vs Generator only, trained on KIT | closest candidate ADE (3 s) | SLOW | 221 | -0.005 | -0.008 | 7 / 8 | [-0.013, +0.003] |
| Generator and encoder trained together vs Generator only, trained on KIT | closest candidate ADE (3 s) | FAST ‡ | 2 | +0.112 | +0.112 | 0 / 1 | [+0.095, +0.128] |
| Generator and encoder trained together vs Generator only, trained on KIT | closest candidate ADE (3 s) | SLOW TURNS | 37 | -0.004 | -0.010 | 4 / 6 | [-0.044, +0.032] |
| Generator and encoder trained together vs Generator only, trained on KIT | scorer's pick ADE (3 s) | ALL | 223 | -0.045 | -0.070 | 98 / 65 | [-0.114, +0.026] |
| Generator and encoder trained together vs Generator only, trained on KIT | scorer's pick ADE (3 s) | SLOW | 221 | -0.047 | -0.070 | 97 / 64 | [-0.115, +0.024] |
| Generator and encoder trained together vs Generator only, trained on KIT | scorer's pick ADE (3 s) | FAST ‡ | 2 | +0.236 | +0.236 | 1 / 1 | [-0.297, +0.769] |
| Generator and encoder trained together vs Generator only, trained on KIT | scorer's pick ADE (3 s) | SLOW TURNS | 37 | +0.219 | +0.086 | 7 / 17 | [-0.055, +0.535] |

## Picking moments vs held-out moments (mean, m)

Picking moments: the 1,763 moments of the 50 picking scenes the checkpoints were chosen on. Held-out: the 223 moments above.

| Model | Group | closest candidate ADE (3 s): picking → held-out | closest candidate FDE (3 s): picking → held-out |
|---|---|---|---|
| Encoder only, tuned on KIT (reference) | ALL | 0.617 → 0.378 | 1.093 → 0.662 |
| Encoder only, tuned on KIT (reference) | SLOW | 0.322 → 0.359 | 0.572 → 0.624 |
| Generator only, trained on KIT | ALL | 0.202 → 0.194 | 0.391 → 0.379 |
| Generator only, trained on KIT | SLOW | 0.180 → 0.194 | 0.346 → 0.380 |
| Generator and encoder trained together | ALL | 0.201 → 0.190 | 0.405 → 0.377 |
| Generator and encoder trained together | SLOW | 0.177 → 0.189 | 0.355 → 0.377 |
| gain of Generator only, trained on KIT over Encoder only, tuned on KIT | ALL | -0.415 → -0.185 | -0.702 → -0.283 |
| gain of Generator only, trained on KIT over Encoder only, tuned on KIT | SLOW | -0.142 → -0.165 | -0.226 → -0.244 |
| gain of Generator and encoder trained together over Encoder only, tuned on KIT | ALL | -0.415 → -0.189 | -0.688 → -0.286 |
| gain of Generator and encoder trained together over Encoder only, tuned on KIT | SLOW | -0.144 → -0.170 | -0.217 → -0.246 |

