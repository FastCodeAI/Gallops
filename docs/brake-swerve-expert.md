# Brake, and brake then swerve: expert v6 and the first imitation rounds

Status at 2026-10-16. This is the first training work on Drive-JEPA **with the
camera reaching the model** (gallops #8, paddock #62). Everything in
[swerve-finetune.md](swerve-finetune.md) and [scorer-cache.md](scorer-cache.md)
was camera-blind.

The goal: on neuroNCAP, brake for a parked car and for a crossing car, and for
an oncoming car brake first, then steer round it.

Two attempts, in order:

1. **A sub-score scorer over inserted swerve paths.** It avoided the oncoming car
   on 0103 but swerved everywhere, and was parked.
2. **A redesigned expert (v6), proven in an offline copy of the simulator, which
   the model then imitates.** Rounds 1A, 1B and a short DAgger round 2 are below.

**Where it stands:** the model learns to brake then swerve on the scenes it
trained on, and with 5 reconstructed scenes that does not transfer reliably to
an unseen street. See [Conclusion](#conclusion).

## 1. Sub-score scorer with inserted swerves (parked)

Drive-JEPA makes 32 candidate plans per tick and a scorer picks one. After the
camera fix, on 0103 frontal, none of the 32 swerves; the only clear ones are
hard braking, and braking alone still gets hit by an oncoming car.

What was tried:

- **Insertion.** Each tick, 32 hand-built paths (the model's pick or straight
  ahead, shifted 1–4 m to either side, plus 4 braking nudges) and 32 straight
  probe paths reaching far ahead go through the last stage of the model's own
  refiner, in the slots of their nearest native plans. That is 64 plans with
  the model's own features.
- **Scorer.** Six heads grade every plan: no collision, on road, time to
  collision, lane keeping, progress, final. A fixed formula combines them and an
  override rule (relative or absolute) chooses when to leave the model's pick.
- **Code:** `train/subscorer.py` (the module, inputs, targets, rules and the
  `layout()` contract with paddock's `drivejepa_subscore`), `train/scorer_train.py`
  (trainer and its preflight), `train/scorer_folds.py --variant probe-geo`.
- **Run:** `{weights}/drivejepa_subscore/sighted-v1`, 5 seeds × 20 epochs, on the
  seeing round-0 collection plus cache v4, 23,777 usable ticks over the 5 scenes.
  Preflight passed (fidelity 99.3% after the switch to lossless frames).

Result on neuroNCAP 0103, 3 runs per shape, relative rule:

| | Frontal | Stationary | Side |
|---|---|---|---|
| Released, seeing | 2.36, 3/3 hit | 5.00 | 1.67, 2/3 hit |
| Sub-score scorer | **5.00, 0/3 hit** | 5.00 | 3.33, 1/3 hit |

**Why it was parked.** It overrode the model from 0.5 s in every scenario,
stationary and side included, and the chosen side flipped left and right 0–16
times per episode. The causes:

- Plans were labelled as if followed for the full 4 s, so the released braking
  plan graded as a crash although closed loop re-plans every 0.1 s.
- Progress is 5/12 of the formula, so fast swerves beat braking.
- Each tick chose from scratch, with no memory of the last choice.
- After the first swerve the car was in states no training data covered.

The code and caches stay as diagnostics. What carried forward: the manoeuvre has
to come from the model's own plan generator, taught by one consistent driver.

## 2. Expert v6

`expert/teacher.py`. It decides from how the hazard moves (its true future, never
the scenario name):

| Hazard | What v6 does |
|---|---|
| Parked | Brake to stop at least 2 m short, the stop anchored to the car and the rate recomputed each tick; stay stopped while blocked |
| Crossing | Brake until it has cleared, then resume |
| Oncoming | Brake at about 4 m/s² to a 3 m/s floor, then move sideways on the side that needs the least further move, using any free drivable road (an empty oncoming lane included, never off the road). Only a swerve that clears is committed; otherwise a tight pass, otherwise brake |

What makes it consistent:

- **World-anchored manoeuvre.** The sideways offset is a function of distance
  along a fixed lane route, and the speed schedule runs on the clock. Each tick
  continues the last tick's plan 0.1 s on (a test holds this to 0.2 m).
- **Side chosen once,** changed only if that side becomes impossible.
- **Acts only on what the camera shows:** the hazard is at least 16 px wide in
  the model's crop, or in frame and under 1.5 s from contact, or the car is
  already off its lane by more than 0.75 m. Checked on two frames, now and
  0.5 s back, as the model sees them.
- **Every collision-class actor is guarded** (+0.3 m a side, +0.5 m an end) and
  held at its last pose for 10 s after its log ends.
- **Road eroded by 0.3 m,** with the raw area as fallback when the car is
  already inside that margin.

### Gates, before anything trained

`expert/rollout.py` drives neuro-ncap's own `Engine.step`, vehicle model,
evaluator and collision checker without the renderer.

- **Gate A, replay:** each round-0 episode's driven plans, replayed, reproduce
  its `ego_poses.json` and `metrics.json`. 150/150 episodes, worst 0.07 mm,
  metrics exact.
- **Gate B, v6 drives all 150 training episodes (trigger T = 3.5 s):**

  | | Frontal | Stationary | Side |
  |---|---|---|---|
  | v6, no contact | 42/50 | 50/50 | 23/50 |
  | released seeing model, no contact | 4/50 | 46/50 | 25/50 |

  0 off-road ticks, 0 side flips. Relabelling every state with the expert's
  memory cleared gives the same decision on 98.3% / 100% / 97.3% of ticks.
  The 8 frontal failures are all frontal-0058, where the road is too narrow to
  pass on it; v6 brakes and scores 1.56. Most side failures are a crossing car
  that never enters the camera in time: it holds a bearing of 36–45° and the
  front camera sees 32.6° each side. Side is left as it is.
- **The real engine agrees.** Expert-driven collection (paddock's analysis
  adapter, `DRIVEJEPA_DRIVE=expert`) replays v6 through neuroNCAP with the
  renderer and asserts the pose stays within 2 cm of the offline rollout; the
  first smoke differed by 0.000 mm over 70 ticks and scored the same.
- **Probe:** a linear probe on the frozen encoder's pooled tokens, predicting
  v6's decision on a held-out scene, balanced accuracy with the image against
  ego state alone: 0053 0.29/0.26, 0058 0.45/0.38, 0064 0.67/0.55, 0070
  0.53/0.61, 0075 0.93/0.58. Strong on the Boston scenes, weak on Singapore.

`expert/label.py --expert v6 --packs <dir>` labels a collection with v6 and
refuses an episode with no `lane_reference.json`.

## 3. Trainer changes

- `--accumulate N`: every forward pass is one tick (`TICKS_PER_PASS`), so the
  batch is N single-tick passes with gradients summed.
- Frozen modules stay in eval mode, so GridMask no longer runs inside the frozen
  encoder while the heads train.
- Real nuScenes driving as a `human` source: the target is the human's path;
  there are no dumps, so grading and fidelity skip it.
- `--subsample-source I=F` keeps a random fraction of a source each epoch.
- The held-out scene is matched by exact id.
- **Frame cache** (`train/frame_cache.py`, `GALLOPS_FRAME_CACHE`): each episode's
  frames, cropped and resized as the model reads them, one array per episode,
  read whole into a 64-episode LRU. A round-1A step spent 1.4–2.4 s of 2.5–3 s
  decoding lossless PNGs. Reads retry on `OSError`, after round 1B died on one
  transient NAS read error.
- **`init` sources:** a DAgger round driven by the `--init` weights is
  fidelity-checked against those weights once they load, with the same limits
  as the start-of-run check.

## 4. Rounds 1A and 1B

Both start from the released checkpoint, train the refiner (winner-takes-all L1
to the v6 plan) and the scorer (BCE plus ranking), batch 1 × 8 accumulated,
lr 1e-4, brake ticks 2×, brake-then-swerve ticks 4×, 3 epochs.

| Run | Scenes | Data |
|---|---|---|
| `{weights}/drivejepa_swerve/v6-r1a-loso0075` | 4, 0075 held out | round 0 driven by the released seeing model and relabelled by v6 (8,402 ticks over 150 episodes), real nuScenes at 33% |
| `{weights}/drivejepa_swerve/v6-r1b-all` | all 5 | as 1A, plus the expert-driven collection (150 episodes, 9,234 ticks) |

1B stopped at 93% of epoch 3 on `OSError: [Errno 5] Input/output error` reading
one frame-cache file from the NAS; the epoch-2 checkpoint (`step_000069186`) is
the one evaluated.

### Training curves

From each run's `metrics.jsonl`:

- **The plan generator learns and carries over.** The distance from the v6 plan
  to the nearest of the 32 candidates fell from 1.18 to 0.48 m in training (1A)
  and 1.03 to 0.40 m (1B). On held-out 0075 it fell from 0.83 to about 0.59 m.
- **The scorer does not carry over.** On training ticks its pick is the best of
  the 32 on 27–31% of ticks (from 10%); on held-out 0075 it stays at 9–22% and
  its loss there does not fall.

### Closed loop

| | Frontal | Stationary | Side |
|---|---|---|---|
| 0075, released seeing, 10 runs | 1.84, 0/10 clear | 5.00, 10/10 | 0.00, 0/10 clear |
| 0075, round 1A (held out), 10 runs | 1.69, 0/10 clear | 5.00, 10/10 | **5.00, 10/10 clear** |
| 0103, released seeing, 3 runs | 2.36, 3/3 hit | 5.00 | 1.67, 2/3 hit |
| 0103, round 1A epoch 1, 3 runs | 2.01, 3/3 hit | 5.00 | 0.00, 3/3 hit |
| 0103, round 1B epoch 2, 3 runs | **4.21, 1/3 hit** | 5.00 | 0.00, 3/3 hit |

On 0103, 1B against the released seeing model is +0.06 with a standard error of
1.01 over the 9 paired runs: the frontal gain and the side loss cancel. Of 1B's
frontal runs, one braked to 2.6 m/s and moved 3.3 m left, which is the intended
manoeuvre; one swerved 3 m right without braking; one was hit. One stationary
run veered 3.1 m left with no need. Every tick stayed on the drivable area.

**Normal driving, nuScenes open loop (5,119 samples):** round 1B epoch 2 has L2
1.186 m against 2.380 for the released seeing model, collisions 1.2% against
1.4% (not significant).

### What is verified and what is argued

- **Verified:** Gates A and B; the engine-versus-rollout agreement; the training
  preflight on every run (code, grading, fidelity on the round-0 source, sight);
  the closed-loop and nuScenes numbers above, each from one paddock run.
- **Argued, not tested:** that real driving in the mix is why 1B no longer
  brakes when the human did not on nuScenes; that the 0103 side loss comes from
  the camera limit, not from training.
- **Small samples:** 3 runs per shape on 0103 and 10 on 0075. None of these is a
  result on the full neuroNCAP set.

### Sidecars that are wrong about their run

Left as written, not hand-edited:

- Both `checkpoint.json` files record `train_git_sha` 048d49c, the commit at
  save time. 1A ran 21067a5 (it started before the frame cache existed); 1B
  ran 048d49c. Main has since fixed the trainer to record the starting commit.
- `trained_on` names only the rendered round-0 collection. Both runs also read
  real nuScenes, and 1B the expert-driven collection; `notes.sources` lists them.
- `--accumulate` and `--subsample-source` are not recorded.

## 5. Round 2, a short DAgger round

Round 1B epoch 2 drove the 5 training scenes, frontal and stationary only, on
new seeds 10–13: 40 episodes. On these, which it trained on, frontal had no
contact in 10 of 20 runs (mean 2.85) and stationary in 16 of 20 (4.16). The
released seeing model, on the same scenes with seeds 0–9, had 4 of 50 frontal
and 46 of 50 stationary.

v6 labelled them (2,466 ticks: 1,135 brake, 507 brake then swerve, 288
least-impact). `{weights}/drivejepa_swerve/v6-r2-short` continues 1B for one
epoch on all of them plus a quarter of round 0 and of the expert-driven set and
a tenth of the real driving, with brake-then-swerve at 2× (1B: 4×). The new
`init` preflight held 1B's weights to what 1B drove: 45/45 ticks, 0.0 m apart.

| 0103, 3 runs per shape | Frontal | Stationary | Side | Mean |
|---|---|---|---|---|
| Released, seeing | 2.36, 3/3 hit | 5.00 | 1.67, 2/3 hit | 3.01 |
| Round 1B epoch 2 | 4.21, 1/3 hit | 5.00 | 0.00, 3/3 hit | 3.07 |
| Round 2 | 1.79, 3/3 hit | 5.00 | 0.13, 3/3 hit | 2.31 |

Round 2 lost 1B's swerve: its widest frontal move was 1.1 m against about 3 m,
and two frontal runs braked to about 1 m/s and were hit, as the released model
is. The needless stationary veer is gone (widest 0.6 m against 3.1 m). Three
things changed at once from 1B (the weighting, the share of old data, one
epoch), so this does not say which one cost the swerve.

## Conclusion

**Shown:**

- The camera fix made Drive-JEPA see: on the full neuroNCAP set, 4.13 against
  3.57 blind, 49 collisions in 200 against 91 (paddock #71).
- Expert v6 does what we want in simulation for everything the front camera can
  see (Gate B), and the real engine agrees with the offline copy.
- Imitating it teaches brake-then-swerve on the scenes it trained on: 10 of 20
  frontal runs clear on new seeds, against 4 of 50 for the released model.

**Not shown:** that it transfers. With 5 reconstructed scenes from 2 cities, the
behaviour on an unseen street is inconsistent: 1A on held-out 0075 learned to
brake for the side car (10/10) but not to swerve (0/10); 1B on 0103 cleared
frontal 2 of 3 times, one of them a swerve without braking; round 2, a small
retrain from 1B, cleared it 0 of 3. The scorer's training curves say the same:
its pick is the best plan on 27–31% of training ticks and 9–22% on the held-out
scene, where its loss never falls.

**Most likely cause, argued and not tested:** too few scenes. The other open
explanations are the scorer's design and labels judged as if each plan were
followed for its full 4 s. A learning curve would separate them: train on 1, 2,
3 and 4 scenes and score the held-out one each time; a held-out score that rises
with every added scene points at data.

**Separate from data:** side. The crossing car holds a bearing of 36–45° and the
front camera sees 32.6° to each side, so it is out of view until it is too late.
More front-camera data cannot fix that; more cameras can.

**What would move it:** more reconstructed scenes (NeuRAD measured 7.5 h each on
this machine), with more than two cities, before more training rounds on these
five. Results here rest on 3 runs per shape on 0103 and 10 on 0075.
