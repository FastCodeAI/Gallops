# H05 — State-space memory

**Owner:** Naren · **Priority:** 5 of 11 · **Started:** 21 Oct 2026 (planned) · **Status:** planned (hypothesis)

The facts are in [experiment.toml](experiment.toml). Everything below is transcribed from Renesas
Progress and Plan, 22 Oct 2026 ("Next steps", item 5, and "Naren · hypothesis to test"); nothing has run.

## Why

- The model's view of time is short: two frames, 0.5 s apart. On neuroNCAP the side crashes are runs
  where it never starts braking although the car was in view for 2 s or more.
- Extended comfort is the base's largest NAVSIM gap (−0.077), and the released model's 0.477 is below
  constant velocity's 0.567: consecutive plans disagree.
- Bench2Drive's stall trap (stuck after a first stop) is a lack of state about why it stopped.
- A state-space layer costs the same per tick however long it has run, so it fits a 2 Hz closed-loop
  driver; a transformer over a long window does not.

It is layer 4 of the team's "Memory: do, on-policy" call (design decision 6), whose layers are on-policy
DAgger, the previous plan as input with 30–50% dropout, carrying slots or scene tokens across ticks, and
a Mamba-style state model that needs sequence training. That call says to judge memory on closed loop
only, because open loop feeds logged history.

## The hypothesis

A state-space layer (Mamba-style: a learned recurrence with a fixed-size state, linear in sequence
length) placed between the frozen encoder and the planner carries the scene forward; the planner's
slots read that state each tick. Expected: earlier, steadier reactions in closed loop and better NAVSIM
extended comfort, with no open-loop loss.

What could make it fail, as the page lists it:

- **Extra capacity, not memory.** A gain could come from the added layer alone. Control: the same layer
  with its state reset every tick.
- **Copying the past.** With its own history it can learn to repeat what it was doing and react late.
  Open loop hides this, so judge on closed loop, and train on closed-loop rounds, not only logs.
- **State leaking between scenes.** The state must reset on `/reset`; the AlpaSim bridge never calls it
  between scenes, so fix that before AlpaSim numbers.

## The plan

| Step | What | Cost | Decides |
|---|---|---|---|
| 0 | Does history beyond 0.5 s carry signal? Cache pooled encoder features per tick for a slice of navtrain and the rendered episodes; fit a small probe on the last k ticks (k = 1, 2, 4, 8) to predict the next seconds of human speed and, on rendered episodes, time to collision. If it stops improving at k = 2, there is little for memory to add. | a feature cache (GPU, hours) plus a CPU probe | whether to build the layer at all |
| 1 | Build the layer in gallops: a diagonal state-space block in plain PyTorch (no custom CUDA kernels), over per-tick pooled scene tokens plus ego status; the plan slots attend to its output. Encoder frozen, features cached; sequences of 8–16 ticks (4–8 s at 2 Hz). | about a day of code | — |
| 2 | Three arms on the same data: (a) state carried across ticks, (b) the same layer reset every tick, (c) released Drive-JEPA. NAVSIM EPDMS and extended comfort on the gate's 1,000 scenarios. | a few GPU-hours per arm | memory versus capacity: (a) must beat (b) |
| 3 | Closed loop: the paddock adapter keeps the state between calls and clears it on reset. The gate's closed tier on (a) and (b): neuroNCAP (when braking starts, impact speed, side 0103 and 0110), Bench2Drive stall routes, AlpaSim. | the closed tier took 2 h 21 min without AlpaSim (20 Oct), per checkpoint | keep it only if (a) beats (b), nothing regresses, and reactions come no later |
| 4 | Port to the 4-camera base if it holds. | — | — |

Training lives in gallops; paddock only runs the checkpoint. Momentum-aware selection (AlpaSim 0.421 →
0.543) is the cheap no-training baseline to beat.

## Where it stands (3 Nov 2026)

Step 0 has not started: no commit in gallops or paddock since 21 Oct touches a state-space layer, a
feature cache for it or the probe. Two things the plan relies on have changed in paddock since: the
gate's NAVSIM subset is now 1,018 tokens drawn within groups (#105, merged 23 Oct), and momentum-aware
selection is on by default for Drive-JEPA (#95, merged 26 Oct), so the released arm (c) is now the
momentum-on model. On the page's third risk: paddock's AlpaSim bridge on `main` says "Every scene
starts with /reset" (`adapters/benchmarks/alpasim/bridge.py`, from #93, merged 21 Oct); whether that
covers the path the page meant was not checked.

## How it will be decided

Step 0 decides whether to build it; steps 2 and 3 decide whether to keep it, through paddock's
regression gate, with (b) as the control that separates memory from capacity.
