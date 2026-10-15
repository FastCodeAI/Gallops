"""S3 scoring wrapper around the vendored Drive-JEPA training scorer (train_pdm_scorer.PDMScorer; nothing vendored is edited).

DECLARED DEPARTURES FROM STANDARD EPDMS (decided by Sid, S2.1 review):
 D2  ego box = the real vehicle (raw calibration/vehicle_dimensions: length, width, rear_axle_to_bbox_center, wheelbase) instead of the
     Pacifica default (train_pdm_scorer.py:107). The box comes from MetricCache.ego_state's vehicle parameters.
 D3  driving-direction gate exemption: a plan whose gate is below 1 is exempted (gate set to 1, upstream aggregation re-run) when
     (a) its simulated ego speed is <= 12 m/s for the whole 4 s, (b) a stationary actor (< 0.5 m/s) lies within 15 m ahead of the
     front bumper and +-3.5 m laterally at the anchor (MetricCache attribute d3_stationary_ahead), and (c) its time outside every
     on-route lane is <= 4.5 s. Every exemption is returned to the caller for logging.
"""
import numpy as np
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from navsim.agents.drive_jepa_perception_based.score_module import compute_navsim_score as CNS
from navsim.agents.drive_jepa_perception_based.score_module.train_pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, EgoAreaIndex, StateIndex
TERMS = ["NC", "DAC", "TLC", "DDC", "EP", "TTC", "LK", "HC", "score"]
D3_MAX_SPEED, D3_AHEAD_M, D3_LAT_M, D3_MAX_OPP_S = 12.0, 15.0, 3.5, 4.5

def vehicle_params(length, width, rear_axle_to_center, wheelbase, height=1.5):
    return VehicleParameters(width=float(width), front_length=float(rear_axle_to_center + length / 2), rear_length=float(length / 2 - rear_axle_to_center),
                             cog_position_from_rear_axle=float(wheelbase) / 2, wheel_base=float(wheelbase), vehicle_name="physicalai_hyperion", vehicle_type="gen1", height=float(height))
_SC = {}
def scorer_for(vp):
    k = (round(vp.width, 4), round(vp.front_length, 4), round(vp.rear_length, 4))
    if k not in _SC: _SC[k] = PDMScorer(CNS.proposal_sampling, CNS.config, vehicle_parameters=vp)
    return _SC[k]

def score(mc, poses, d3=True):
    """poses (N, 8, 3) in the rig frame at the anchor. Returns scores (N, 9) in TERMS order and a dict with the D3 bookkeeping."""
    vp = mc.ego_state.car_footprint.vehicle_parameters; sc = scorer_for(vp)
    sim = CNS.before_score_proposals(mc, np.asarray(poses, dtype=np.float32), mc.ego_state)                    # compute_navsim_score.py:33-50 (reference plan at index 0)
    final = sc.score_proposals(sim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map, mc.map_parameters, None, mc.past_human_trajectory)
    ddc0 = sc._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION].copy(); std_final = final.copy()
    speed = np.hypot(sc._states[..., StateIndex.VELOCITY_X], sc._states[..., StateIndex.VELOCITY_Y]).max(axis=1)
    opp_s = sc._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC].sum(axis=1) * CNS.proposal_sampling.interval_length
    ex = np.zeros(len(final), dtype=bool)
    if d3 and bool(getattr(mc, "d3_stationary_ahead", False)):
        ex = (ddc0 < 1.0) & (speed <= D3_MAX_SPEED) & (opp_s <= D3_MAX_OPP_S)
        if ex.any():
            sc._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, ex] = 1.0; final = sc._aggregate_pdm_scores()   # train_pdm_scorer.py:234-270
    scores, _, _, _ = CNS.after_score_proposals(sc, final, mc.ego_state, True)
    return scores, dict(exempt=ex[1:].copy(), ddc_standard=ddc0[1:].copy(), score_standard=std_final[1:].copy(), opp_s=opp_s[1:].copy(), max_speed=speed[1:].copy(),
                        progress=sc._progress_raw[1:].copy(), ref_progress=float(sc._progress_raw[0]), nondrivable=sc._ego_areas[1:, :, EgoAreaIndex.NON_DRIVABLE_AREA].mean(axis=1))
