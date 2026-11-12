# Gate G0: read-only findings (2026-10-21)

Code of record: DEVKIT = {drivejepa_repo}/navsim_v2 (upstream e21f474).
Paths below are relative to DEVKIT/navsim/agents/drive_jepa_perception_based/ unless they say otherwise.

## a) How upstream picks the winner

drive_jepa_model.py:85-91
```python
pdm_score = torch.sigmoid(pred_logit)[:, :, -1]
if 'past_ego_simulated_states' in features and features['past_ego_simulated_states'] is not None:
    pdm_score = self.calibrate_score(features, proposals, pdm_score)
token = torch.argmax(pdm_score, dim=1)
trajectory = proposals[torch.arange(batch_size), token]
```
- Only the **last** pred_score logit is used, through a sigmoid, which preserves order. No other pred_score
  logit, no pred_col_agent and no pred_area output takes part, and there are no weights or thresholds.
- The single exception is the comfort re-score (`calibrate_score`, drive_jepa_model.py:98-151, `(14·score + 2·comfort)/16`).
  It runs only when `past_ego_simulated_states` is not None. On KIT it is always None:
  `abstract_agent.py:85` sets it from the `compute_trajectory` argument, which defaults to None, and the KIT runner
  calls `compute_trajectory(agent_input)`. The evaluation dumps set it to None explicitly
  (experiments/drivejepa_kitscenes 03_encoder_training/src/evalsets.py:46) and pick with `argmax(pdm_score)` (evalsets.py:55; 02_encoder_build/src/f_baseline.py:65).
- **Conclusion:** upstream picks on the last pred_score logit alone. This run uses upstream's rule unchanged, so no
  second "upstream pick rule" column is needed.
- Note: a sigmoid in fp32 rounds to exactly 1.0 for logits above about 17. If training pushes several logits that
  high, `argmax` returns the first tied index. Softmax cross-entropy ignores a shift common to all 32 logits, so the
  level can drift. Each check records how often the top `pdm_score` is tied.

## b) pred_score head

- score_module/scorer.py:13,15 `self.pred_score = MLP(tf_d_model=256, tf_d_ffn=1024, 6)`.
- bevformer/transformer_decoder.py:65-78 MLP = Linear(256→1024) → LayerNorm → ReLU → Linear(1024→1024) →
  LayerNorm → ReLU → Linear(1024→6). All 6 logits **share both hidden layers**; only the last Linear's rows are separate.
  Training for the last logit therefore also moves the other five (the drift is reported at each check).
- What the 6 logits are: drive_jepa_agent.py:146-148 trains them against `target_scores[..., -6:]`, and
  score_module/compute_navsim_score.py:88-90 stacks target_scores as [no at-fault collision, drivable area,
  traffic light, driving direction, ego progress, time to collision, lane keeping, comfort, final]. So the 6 logits are
  [driving direction, ego progress, time to collision, lane keeping, comfort, **final PDM score**].
  Upstream config sets sub_score_weight = 0 and final_score_weight = 1 (drive_jepa_config.py:51-52), so the released
  model was trained only on the last logit as well.
- **Input to the scorer** (scorer.py:41-42): `proposal_feature = bev_feature.reshape(B, 32, 8, 256).amax(-2)`, then
  `pred_score(proposal_feature)`. The candidate coordinates (`proposals`) are used only for their shape. The scorer
  **never sees image features directly**: `bev_feature` is the generator's final query state after the 4th refinement
  (drive_jepa_model.py:64-73; traj_refiner.py:23-29, where `Bev_refiner(proposals, bev_feature, image_feature)` mixes in
  image features and the current candidates). Image features reach the scorer only through the generator.

## c) KIT human path length

- Every one of the 13,937 training moments (and the 1,763 picking moments) has future_poses of shape (8, 3) at
  0.5 s spacing (future_interval_s = 0.5 for all), with 8 of 8 waypoints finite. Checked from
  {runs}/drivejepa_kitscenes/generator_joint_kit/s1/moments.pt.
- They are real frames, not padding: kitscenes_adapter.py:354-355 (DEVKIT/scripts/inference) indexes
  `keyframe + 5·(i+1)` with no clipping, and the moment rule `t + 40 <= N − 1` (05_generator_survey/out/d_horizon.json) keeps
  all 8 inside the scene; 0 moments were dropped for bad timing.
- Fewer than 6 valid waypoints: 0 moments. The gate passes.

## d) "Generator and encoder trained together" on the 1,763 picking moments

08_generator_and_encoder_together/out/run_phase2.json, check `pass == 1.0` (the chosen checkpoint, `best: 1.0`), `summary`:

| group | closest candidate ADE (3 s) | scorer pick ADE (3 s) | closest candidate FDE (3 s) |
|---|---|---|---|
| ALL | 0.20137 | 1.37373 | 0.40544 |
| SLOW | 0.17748 | 1.11501 | 0.35519 |
| FAST | not recorded | not recorded | 1.43563 |
| SLOW TURNS | **not recorded** | **not recorded** | 0.53054 |

On SLOW TURNS, 07_generator_only_gentle_rate/src/rerun.py:143-157 (`summarise`) stores only closest FDE (3 s), variety and
better/worse counts. Closest candidate ADE and scorer pick ADE are not stored, so the SLOW TURNS half of the
comparison flag has no reference number. Following the brief, it is not recomputed.
