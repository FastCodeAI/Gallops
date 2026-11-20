# CLAUDE.md: Run 3B (v2, 19 Nov) — run 3A + fresh LoRA + fact scorer, NAVSIM + LEAD

(Navvrat's spec, revised version pasted 19 Nov ~02:15, verbatim. Earlier: CLAUDE_v2a_19nov.md (19 Nov ~02:00, planner and decoder loaded from run 3A), CLAUDE_v1_18nov.md (18 Nov).)

**Replaces the 18 Nov version.** What changed: no Phase 1 weights; the original WA-JEPA encoder with a fresh LoRA; 2 input frames; planner and agent decoder from scratch; teacher forcing on the agent look; auxiliary heads on LEAD; camera augmentation.

**What:** run 3A's architecture, trained end to end from scratch (encoder aside) on NAVSIM + LEAD with the new fact scorer.

No pass marks. Report numbers next to run 3A's where we already have them (no new work on run 3A). Navvrat decides.

## 1. Fixed (ask before changing)

| | |
|---|---|
| **Encoder** | WA-JEPA ViT-L as in run 3A, base frozen. **Fresh LoRA**: rank 32, alpha 32, on Q and V of all 24 blocks, B zero-initialised, so at step 0 the encoder is run 3A's. LoRA learning rate = 0.1 × the heads' (KNOB) |
| **Input** | Run 3A's: 4 cameras, **2 frames** (one tubelet step, 4 × 512 tokens), 512 × 256. LEAD: frames f−2 and f from `lead_inputs_v2` |
| **Agent decoder** | Run 3A's architecture (300 slots, 3 layers), its classes (vehicle, pedestrian, bicycle, none), **trained from scratch**. Detection loss on both datasets. LEAD targets: `lead_inputs_v2` mapped to these classes (parked cars are vehicles), \|x\|, \|y\| ≤ 32 m |
| **Planner** | Run 3A's architecture, **trained from scratch**: 32 proposals × 8 waypoints, 4 rounds, footprint look + agent look. No map look, no map decoder, no light head |
| **Loaded** | Only the WA-JEPA encoder. Nothing from run 3A's or Phase 1's checkpoints |
| **Scorer** | New, from scratch (section 2). Run 3A's scorer heads are not used |
| **Data** | NAVSIM navtrain minus the 51 held-out logs. LEAD training side (`lead_inputs_v2`, labels `carla_full_v1` + `carla_full_v1_vru`) minus the 416 held-out routes |
| **Batch** | 64 = 32 NAVSIM + 32 LEAD |
| **Sampling** | Stop-sign frames ×2 on both. LEAD frames with a pedestrian or cyclist in our path ×4. Shifted frames at their natural share |
| **Augmentation** | Both datasets: small random shift of camera position, yaw and field of view with `lidar2img` and the rays updated to match; one whole camera dropped at random (rear included) in 10% of samples (KNOBS; Phase 1's values where it had them) |
| **Never** | navtest, navhard, Bench2Drive's 220 evaluation routes, the 380 overlap routes, HUGSIM scenes |

**Step-0 check:** with LoRA's B at zero, the encoder's tokens on 200 NAVSIM held-out samples equal run 3A's cached tokens (cosine reported). Stop if they don't.

## 2. The scorer

**Paths scored per sample:** the expert's path, the planner's 32 proposals (labelled live on the GPU, as now), and the balanced draw from the pool (per fact up to 16 failing + 16 passing; union capped at 128, KNOB).

**Path reader (every path, the planner's 32 included, detached):**
1. Each waypoint (x, y, heading, step index) becomes a token.
2. Footprint look: run 3A's path-anchored read of the camera tokens.
3. Agent look: run 3A's block (section 3 says what it is fed).
4. Self-attention over the path's 8 waypoint tokens.
5. Fact heads on each waypoint token.

**Fact heads, per waypoint:**

| Head | Type | Mask |
|---|---|---|
| Hit something · off road · left the route · stop sign · wrong way | yes / no | |
| Red light | yes / no | NAVSIM: unknown colour |
| Comfort | yes / no | all of CARLA |
| Metres gained so far · lane offset · closeness (s, cap 4) · cut-in braking (m/s², cap 8) | number | |

**Scene head:** the expert's metres so far, 8 numbers.

**The score** (computed, never a head): T4 from `scorer_stats_v2`, γ 0.9, 5% / 0.5 m tolerance, quality = closeness × cut-in × lane × wrong way (0.5) × comfort (NAVSIM only) × overshoot term, penalty on hit, off road, red light, stop sign, left the route, P = 0.108 to start. True expert's metres in training; the head's at test time. Report both.

**Pick:** the proposal with the highest score from predicted facts.

## 3. Teacher forcing on the agent look

Run 3A's agent look reads, per slot: the decoder's feature, and its box, velocity and P(none) (detached).

- **Teacher-forced sample:** for every true agent, take the slot the detection loss matched to it (one-to-one matching). Keep that slot's feature; replace its box and velocity with the true ones; set P(none) to 0 for matched slots and 1 for the rest.
- **Free sample:** the decoder's own outputs, as in run 3A.
- Chosen per sample, the same choice for the planner's and the path reader's agent look. Share of teacher-forced samples: 1.0 for the first pass (the agent decoder starts from scratch, so its outputs are noise), then falling linearly to 0.2 at 60% of the run, then constant (KNOBS).
- Add small noise to the true boxes (position σ 0.3 m, heading σ 3°, speed σ 0.5 m/s; KNOB) so the looks don't learn to expect perfect boxes.
- **Every held-out report gives the scorer's and the pick's numbers twice:** with true agents and with the decoder's. The gap is what vision costs us.

## 4. Losses

| Part | Loss |
|---|---|
| Planner | Run 3A's on both datasets: min-L1 to the expert + 0.5 × min-L1 to 4 teachers. NAVSIM: run 3A's teachers. LEAD: the 4 clean pool paths with the highest true score (T4, P 0.108), ties by distance to the expert |
| Auxiliary heads | Run 3A's (key agents, ego areas, map aux) on **both** datasets. LEAD labels from `lead_aux` with the three fixes: debris out, empty key labels masked, objects appearing after t included |
| Agent decoder | Run 3A's detection loss, both datasets |
| Expert's-metres head | L1 |
| Yes / no facts | Binary cross-entropy per waypoint |
| Number facts | Smooth L1 per waypoint |
| Ranking | On the computed score: pairs of paths in a sample whose true scores differ by more than 0.02, logistic loss on the score difference. Weight 0.1 (KNOB) |

- The scorer does not push the planner: proposals are detached before the path reader.
- Scorer gradients reach the path reader, fact heads, agent decoder and LoRA.
- Loss weights: run 3A's values for its losses, 1 for the new ones, except ranking. Log every loss and its share.

## 5. Speed

- Heads and all losses run once per step on the 64 samples; the encoder may run in groups for memory.
- Compile the heads with CUDA graphs (`torch.compile(mode="reduce-overhead")`), fixed shapes.
- bf16 in the encoder.
- Report s/step, GPU memory, our share of the GPU, and whether MPS is on, before the long run.

## 6. The run

- Start directly. Report at step 2,000 without stopping; stop only on NaN, a loss not falling, a stalled loader or label errors.
- Length: what fits in 48 hours at the measured step time; learning-rate schedule set to that. Checkpoint every pass (trainable weights only, new path each time).
- **Report at step 2,000 and after every pass, on the held-out sets:**
  1. Step time and every loss.
  2. Agent recall / precision by range ring and direction, NAVSIM and LEAD. Start: run 3A on LEAD 0.303 / 0.406 overall, 0.443 at 10–20 m; on NAVSIM 0.62 at 10–20 m.
  3. Facts: within-scene pairwise accuracy per yes / no head, error per number head; with true and with decoded agents. Run 3A's heads on NAVSIM: collision 0.972, drivable 0.959, direction 0.967, light 0.959, progress 0.816.
  4. Expert's-metres head: error at 2 s and 4 s; on LEAD split by "expert stops" against the rest.
  5. The pick: its true score against the best of the 32; share where it is a true top; share of picks that stop while the expert moves; with true and predicted expert's metres; with true and decoded agents.
  6. Planner: nearest-proposal distance to the expert and the spread of the 32, both datasets; on NAVSIM held-out the EPDMS of the pick next to run 3A's 0.920 (best of 32: 0.961).
  7. LEAD teachers: frames with 4 / 1–3 / 0 teachers.

Evaluation on navtest, Bench2Drive (LEAD's camera rig), the light-swap test and HUGSIM gets its own spec.

## Rules

- Don't change run 3A, the fact labels, `lead_inputs_v2` or `lead_aux`. New code in `run3b`; never overwrite a checkpoint, cache or result.
- No invented thresholds or verdicts.
- Log in `agent_memory/<date>.md`.

---
## Run notes (mine)
- 19 Nov 02:04 g3 (v1 spec) stopped at step 148, superseded. v2 code goes in run3b/v2/ (data/, eval/, model/).
- Reused from v1 work: GPU live labels incl. LEAD aux (gpulab/service.py; agentfacts key agents v2 + key_mask; mapfacts CARLA
  aux), grouped/sync-free step ideas (r3b/train2.py), profile (artifact GBPZYw5Dj3hoLk24KBy5wk).
- Phase 1 augmentation values: angle jitter 1.0 deg (rotation, image warped by K R K^-1), position jitter 0.05 m (calibration
  only, image not warped), camera dropout 0.1 (phase1/data/augment.py). No FOV value in Phase 1.
- 19 Nov ~02:15 revised spec (above): planner + agent decoder from scratch, only the encoder loaded; step-0 = token cosine;
  teacher forcing 1.0 for the first pass. Navvrat, same message: "Treat every static.prop.dirtdebris* as debris (not a hit,
  not a key agent), in the hit rule and in the key-agent labels. Report how many stored hit labels change" -> "run them asap
  and then the training".
- Step-0 check (19 Nov ~02:35, v2/data/check_a_tokens.py): frozen WA-JEPA on our NAVSIM images vs run 3A's cached tokens,
  200 held-out samples: per-sample cosine min 0.9793 / p1 0.9835 / median 0.9923 / max 0.9974; cause = local images are
  Phase 1's 256x512 JPEG q95 re-encodes (the cache came from the 1920x1080 NAS originals; on 48 samples with local originals
  the cache builder's path is bit-exact). Navvrat: "Use local re-encodes".
- Navvrat 19 Nov: "FOV jitter: zoom each image by a random factor between 1.00 and 1.10 (crop the centre, resize back to
  512 x 256), and update the intrinsics, lidar2img and rays to match. One factor per sample, the same for all four cameras.
  Applied to 50% of samples." -> v2/data/augment2.py zoom_max 0.10, zoom_p 0.5 (backup augment2_before_zoom.py).
- dirtdebris*: debris_v2/count_hits.py -> 442 of 121,235 stored LEAD records change hit (3,682 pool path rows, 7,111 cells; 0
  expert / logged rows), all from static.prop.dirtdebris02 (859 objects; dirtdebris01 1,191 already debris); old rule
  reproduces all 121,235 stored labels. New LEAD pack: {data.paddock_data}/run3b/pack_v2 (build_pack_v2.py). Live
  labels: gpulab/service.py sets ctx['deb'] = every dirtdebris* id of the route (hit rule + key-agent observation).
- 19 Nov 02:43 run3b_v2_19nov (11.06 s/step, uncompiled ViT) stopped itself at step 98: live-label error = mapfacts
  extract asserted <= 30 rings per polygon (int32 ring bitmask in the point-in-polygon kernel); its stop-report then crashed
  (eval lambda keyword). Fixes: mapfacts kernel uses exterior parity + XOR of hole parities (holes disjoint; no ring cap;
  verify 64 + 64 cases 0 differences; backup mapfacts_before_xor_rings.py); eval lambda; eval failure no longer stops
  training; CUDA-graph step marks + cloned eval outputs; compiled ViT blocks; heads reduce-overhead; AgentLook without its
  diagnostic float() stats (bit-identical, swapped in-process); recompile_limit 64; FlashAttention-2 forced in the encoder;
  inductor cache v2/inductor_cache.
- 19 Nov 03:15 run3b_v2b_19nov: 7.24 s/step -> 23,859 steps (7.11 passes) in 48 h. Report at step 2,000 (~07:20).
