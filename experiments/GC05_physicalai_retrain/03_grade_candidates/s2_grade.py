"""S2 grader: builds an upstream MetricCache per anchor from the NuRec scene and grades plans with the vendored Drive-JEPA
training scorer (score_module/train_pdm_scorer.py via compute_navsim_score.py). Nothing upstream is edited or re-implemented:
simulation = compute_navsim_score.before_score_proposals, scoring = scorer.score_proposals, breakdown = after_score_proposals.
Run in the conda drive-jepa env with cwd = navsim_v2 (see s2_run_*.py)."""
import math, pathlib, numpy as np
import s2_common as C, s2_scene as SC
K = SC.K
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from navsim.common.dataclasses import Trajectory
from navsim.common.enums import SceneFrameType
from navsim.planning.metric_caching.metric_cache import MetricCache, MapParameters
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from navsim.agents.drive_jepa_perception_based.score_module import compute_navsim_score as CNS

TERMS = ["NC", "DAC", "TLC", "DDC", "EP", "TTC", "LK", "HC", "score"]     # compute_navsim_score.py:90-92 column order
VP = get_pacifica_parameters(); T_BASE_US = 1_000_000_000                  # positive, arbitrary clock origin for nuPlan TimePoints
VEH = ("automobile", "truck", "bus", "vehicle", "trailer", "train"); PED = ("person", "pedestrian"); BIKE = ("rider", "bicycle", "motorcycle", "cyclist")
def obj_type(cat):
    c = (cat or "").lower()
    if any(k in c for k in PED): return TrackedObjectType.PEDESTRIAN
    if any(k in c for k in BIKE): return TrackedObjectType.BICYCLE
    if any(k in c for k in VEH): return TrackedObjectType.VEHICLE
    return TrackedObjectType.GENERIC_OBJECT

def ego_state_at(S, i, vp=None):   # vp: nuPlan VehicleParameters; None = Pacifica (S2 / upstream default). S3 passes the real vehicle.
    c, s = math.cos(S["yaw"][i]), math.sin(S["yaw"][i]); v = np.gradient(S["xy"], K.DT, axis=0)[i]
    vx, vy = c*v[0] + s*v[1], -s*v[0] + c*v[1]
    return EgoState.build_from_rear_axle(rear_axle_pose=StateSE2(float(S["xy"][i, 0]), float(S["xy"][i, 1]), float(S["yaw"][i])), rear_axle_velocity_2d=StateVector2D(float(vx), float(vy)),
                                         rear_axle_acceleration_2d=StateVector2D(float(S["acc"][i]), 0.0), tire_steering_angle=0.0, time_point=TimePoint(T_BASE_US + int(round(S["t"][i] * 1e6))), vehicle_parameters=vp or VP)

def detections(S, i, n=41):
    """n DetectionsTracks at 10 Hz from the anchor (clipgt obstacle tracks, clipgt frame). Speed below 0.5 m/s (1 s baseline) -> velocity 0."""
    out = []
    for k in range(n):
        tq = S["t"][i] + k * K.DT; objs = []
        for tid, tr in S["tracks"].items():
            j = K.idx_at(tr, tq)
            if j is None: continue
            box = OrientedBox(StateSE2(float(tr["xy"][j, 0]), float(tr["xy"][j, 1]), float(tr["yaw"][j])), max(tr["L"], 0.2), max(tr["W"], 0.2), 1.5)
            md = SceneObjectMetadata(timestamp_us=T_BASE_US + int(round(tq * 1e6)), token=f"{tid}_{k}", track_id=None, track_token=str(tid)); ty = obj_type(tr["cat"])
            if ty in (TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE):
                sp = float(tr["v"][j]) if tr["v"][j] >= 0.5 else 0.0
                objs.append(Agent(tracked_object_type=ty, oriented_box=box, velocity=StateVector2D(sp * math.cos(tr["mdir"][j]), sp * math.sin(tr["mdir"][j])), metadata=md))
            else:
                objs.append(StaticObject(tracked_object_type=ty, oriented_box=box, metadata=md))
        out.append(DetectionsTracks(TrackedObjects(objs)))
    return out

def drivable_map(S, i, radius=120.0):
    from shapely.geometry import Point
    c = Point(*S["xy"][i]).buffer(radius); tok, typ, geo = [], [], []
    for lid, l in S["lanes"].items():
        if l["poly"].intersects(c): tok.append(lid); typ.append(SemanticMapLayer.LANE_CONNECTOR if lid in S["connectors"] else SemanticMapLayer.LANE); geo.append(l["poly"])
    for br in S.get("bridges", []):                                   # S2.1 A2: bridged on-route gaps count as on-route lane surface
        if br["poly"].intersects(c): tok.append(br["token"]); typ.append(SemanticMapLayer.LANE); geo.append(br["poly"])
    for iid, p, _ in S["inter"]:
        if p.intersects(c): tok.append(f"int_{iid}"); typ.append(SemanticMapLayer.INTERSECTION); geo.append(p)
    for k, p in enumerate(S["drivable"]):
        if p.intersects(c): tok.append(f"drivable_{k}"); typ.append(SemanticMapLayer.DRIVABLE_AREA); geo.append(p)
    return PDMDrivableMap(tok, typ, geo)

def build_cache(S, i, token, file_path=None, vp=None):
    """MetricCache for the anchor at grid index i. Returns (cache, info) or (None, reason)."""
    n = len(S["t"])
    if i + 40 >= n: return None, "no 4 s future"
    P, cur = SC.centerline_at(S, i)
    if P is None or len(P) < 3: return None, "ego not on any lane"
    h = np.arctan2(np.gradient(P[:, 1]), np.gradient(P[:, 0])); centerline = PDMPath([StateSE2(float(x), float(y), float(a)) for (x, y), a in zip(P, h)])
    ego = ego_state_at(S, i, vp); det = detections(S, i)
    obs = PDMObservation(CNS.proposal_sampling, CNS.proposal_sampling, 100, observation_sample_res=1, extend_observation_for_ttc=False)     # metric_cache_processor.py:218-224
    obs.update_detections_tracks(det, [[] for _ in det], {}, compute_traffic_light_data=True)                                               # no light states in the usdz -> no red-light polygons
    # reference ("PDM-Closed") trajectory: STAND-IN = the human path (no nuPlan lane graph to run PDM-Closed on)
    ref = InterpolatedTrajectory([ego_state_at(S, j, vp) for j in range(i, i + 41)])
    j0 = max(0, i - 15); past = InterpolatedTrajectory([ego_state_at(S, j, vp) for j in range(j0, i + 1, 5)]) if i - j0 >= 5 else None             # 1.5 s at 2 Hz (metric_cache_processor.py:333-335)
    rel = np.array([np.r_[K.ego_frame(S["xy"][i + 5 * k], S["xy"][i], S["yaw"][i]), S["yaw"][i + 5 * k] - S["yaw"][i]] for k in range(1, 9)])
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    mc = MetricCache(file_path=file_path, log_name=S["uuid"], timepoint=ego.time_point, scene_type=SceneFrameType.ORIGINAL, trajectory=ref,
                     human_trajectory=Trajectory(poses=rel.astype(np.float32), trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5)), past_human_trajectory=past, ego_state=ego,
                     observation=obs, centerline=centerline, route_lane_ids=sorted(S["route_ids"]), drivable_area_map=drivable_map(S, i, radius=120.0 if not S.get('fix') else max(120.0, 60.0 + 4.5 * float(S['speed'][i]) * 1.5)),
                     past_detections_tracks=[], current_tracked_objects=[det[0]], future_tracked_objects=det[1:], map_parameters=MapParameters(map_root="", map_version="26.04", map_name=f"nurec-clipgt:{S['uuid']}"))
    return mc, dict(cur_lane=cur, human_rel=rel, n_agents0=len(det[0].tracked_objects.tracked_objects), has_past=past is not None)

def grade(mc, poses, diag_no_ddc=False):
    """poses (N,8,3) rig frame at the anchor. Returns (N,9) in TERMS order. Mirrors compute_navsim_score.get_sub_score:57-76 without the anchors index file."""
    sim = CNS.before_score_proposals(mc, np.asarray(poses, dtype=np.float32), mc.ego_state)
    final = CNS.scorer.score_proposals(sim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map, mc.map_parameters, None, mc.past_human_trajectory)
    scores, _, _, _ = CNS.after_score_proposals(CNS.scorer, final, mc.ego_state, True)
    areas = CNS.scorer._ego_areas[1:].copy()          # (N, 41, 3) multiple lanes / non-drivable / oncoming
    diag = {}
    if diag_no_ddc:
        # DIAGNOSTIC ONLY (S2.1 B2), never written to a cache: same scorer state, driving-direction gate forced to 1, upstream
        # aggregation re-run (train_pdm_scorer.py:234-270) so progress masking is recomputed consistently.
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
        CNS.scorer._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, :] = 1.0; f2 = CNS.scorer._aggregate_pdm_scores()
        diag = dict(score_no_ddc=f2[1:].copy(), ep_no_ddc=CNS.scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, 1:].copy())
    return scores, dict(**diag, progress_raw=CNS.scorer._progress_raw[1:].copy(), ref_progress=float(CNS.scorer._progress_raw[0]), ref_score=float(final[0]), areas=areas.mean(axis=1))
