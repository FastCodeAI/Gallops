<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H13 · Alpamayo privileged teacher on neuroNCAP, reusing the tested expert solver

**Ruhan** · neuroNCAP · ⏸ parked · 14 Oct – 5 Nov 2026 · enabler

[← all experiments](../README.md#enablers) · [the account](https://github.com/FastCodeAI/Gallops/blob/d617893b6fbf64172e22d1edd362548ec3e5305f/src/gallops/expert/alpamayo_teacher.py) (on commit ⎇) · [experiment.toml](experiment.toml)

> The record is on commit `d617893`, not merged yet; its links go to that commit.

**Evidence for:** D19 ([decision log](../README.md#decision-log))

## Question

Can Alpamayo get privileged teacher labels on neuroNCAP (stop, swerve, brake, least impact) from the expert gallops already tests, draw.teach(), reshaped to Alpamayo's six-waypoint contract, rather than from a second, untested expert ported from the research repo?

## What we learned

A thin adapter reshapes draw.teach()'s plan to six waypoints at 0.5 s, records labels per tick the way label.py does for Drive-JEPA, and flags a tick-to-tick swerve direction flip rather than forcing commitment. No live neuroNCAP episode drove it. Parked: PR #18 closed unmerged on 5 Nov 2026, as this repo is not training Alpamayo.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H13.r1**<br>the adapter as committed (19 Oct); unit tests only, no live neuroNCAP episode | paddock alpamayo15_teacher (14 Oct), moved to gallops on 19 Oct (paddock 6cd1975) | — | — | — | — |

- **H13.r1:** Not a training run: it records the code the branch adds. ALPAMAYO_WAYPOINT_DT = 0.5 is assumed from draw.WAYPOINT_DT and still to be checked against the live alpamayo15_teacher adapter's model.yaml (module docstring). The research repo's Frenet-frame candidate generation and the paddock adapter's go-around commitment state machine are deliberately not ported.

## Checks that decided something

- **Unit tests (tests/test_alpamayo_teacher.py)** (19 Oct): cover truncating 8 poses to 6, refusing a smaller draw.POSES, clear road and stationary obstacle, direction-flip flagging, and the per-tick JSONL record by camera; no end-to-end run against a live neuroNCAP episode

## Evidence outside this repo

- Paddock's half: the privileged ground-truth channel (Observation.actors / scene_name) is on paddock branch ruhan/privileged-channel and the earlier alpamayo15_teacher adapter on ruhan/alpamayo-teacher-planner (14 Oct); neither is on paddock main through 3 Nov.
