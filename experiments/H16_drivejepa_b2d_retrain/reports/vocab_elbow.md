# Vocabulary size: elbow check (05 Nov 2026)

Navvrat asked for an elbow check from 512 up, and whether more clusters only split speed or feed a mode collapse. Same 36,897 training paths and k-means settings as `../build_vocab.py`, sizes 256 to 8,192, 3 seeds each (33 runs; `elbow.py`, `runs/`). Held-out = the LoRA run's 47 clips. "Moving" = more than 0.1 m in 4 s (1,655 held-out paths). Numbers are the mean of 3 seeds unless stated. Charts: `elbow_05nov.png`.

| Size | Held-out ADE, moving (m) | Training ADE, moving (m) | Held-out minus training (m) | Saved per doubling (m) | Turns, held-out (m) | 9+ m/s, held-out (m) | Along / across error (m) | Entries from one clip | Single-path entries | Entries within 1 m (median) | No entry within 1 m | Straight-entry speed gap (m/s) |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 512 | 0.492 | 0.506 | −0.014 | 0.066 | 1.041 | 1.018 | 0.391 / 0.197 | 0.7 % | 2 | 4 | 9.1 % | 0.052 |
| 768 | 0.454 | 0.455 | −0.001 | 0.067 | 0.955 | 0.963 | 0.358 / 0.184 | 1.6 % | 10 | 7 | 7.2 % | 0.032 |
| 1,024 | 0.426 | 0.420 | +0.006 | 0.059 | 0.874 | 0.929 | 0.335 / 0.175 | 2.1 % | 18 | 8 | 5.9 % | 0.025 |
| 1,536 | 0.392 | 0.375 | +0.017 | 0.048 | 0.807 | 0.853 | 0.308 / 0.161 | 4.7 % | 66 | 12 | 4.5 % | 0.018 |
| 2,048 | 0.372 | 0.342 | +0.030 | 0.040 | 0.772 | 0.799 | 0.294 / 0.151 | 7.3 % | 137 | 15 | 3.8 % | 0.014 |
| 3,072 | 0.348 | 0.295 | +0.054 | 0.040 | 0.725 | 0.763 | 0.275 / 0.142 | 14.3 % | 422 | 22 | 2.9 % | 0.009 |
| 4,096 | 0.332 | 0.260 | +0.072 | 0.034 | 0.683 | 0.745 | 0.262 / 0.136 | 22.3 % | 892 | 28 | 2.1 % | 0.007 |
| 6,144 | 0.312 | 0.206 | +0.106 | 0.034 | 0.647 | 0.712 | 0.245 / 0.129 | 37.7 % | 2,278 | 40 | 1.5 % | 0.005 |
| 8,192 | 0.298 | 0.163 | +0.135 | | 0.626 | 0.703 | 0.235 / 0.123 | 50.2 % | 4,063 | 52 | 1.3 % | 0.003 |
| NAVSIM 8,192 | 0.636 | 0.633 | | | | | | | | | | |

"Saved per doubling" is from this size to the next. Training ADE is one seed. Clip-bootstrap 95 % interval on held-out ADE is about ±0.05 m at every size; the gain between neighbouring sizes is paired and its interval excludes zero at every step.

## What the numbers show

- **No sharp elbow.** Held-out error falls as a smooth power law (about K^−0.18). Inertia is no guide: it is measured on the training paths and keeps falling as entries become single paths.
- **Where it changes:** held-out and training ADE agree up to about 1,024, then part. From there entries increasingly fit individual paths: entries from a single clip go 2.1 % → 7.3 % → 22.3 % → 50.2 % at 1,024 / 2,048 / 4,096 / 8,192.
- **Speed or shape:** along-path error stays about 2× across-path error at every size (1.98 at 512, 1.91 at 8,192), so more clusters do not buy speed alone. But straight entries are already 0.025 m/s apart at 1,024.
- **Allocation:** one entry (2 at 4,096 and up) holds the stationary paths, which are 16-22 % of all paths. Paths at 9+ m/s are 4.5 % of paths and about 11.5 % of entries, so k-means does not drop the fast mode.
- **Seed agreement rises with size for the wrong reason:** at 8,192 the median entry is 0.0 m from an entry of another seed, because both are the same single path.

## Limits

The real teacher pool depends on a grader that isn't built, so "entries within 1 m" is a stand-in. 47 held-out clips, 1,655 moving paths. A vocabulary can only show the conditions for mode collapse; collapse itself is a training outcome.
