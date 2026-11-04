"""C1: each navtest disagreement re-examined on our own simulation of the pick and of the human.

For the pick (row 1 after the PDM reference) and the human, lane keeping is recomputed two ways on
the training scorer's own simulated ego boxes:
  coarse = the training grader's loop (every 5th state, i.e. 0.5 s; fail after 4 in a row)
  fine   = the evaluator's loop (every state, 0.1 s; fail after 20 in a row)
"""

import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point

from gallops.config import SiteConfig
from gallops.memory import grade

cfg = SiteConfig.load()
grade.init_worker(cfg)
cns = grade._VENDOR["cns"]
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory  # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (  # noqa: E402
    BBCoordsIndex,
    MultiMetricIndex,
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # noqa: E402

regrade = json.load(open(sys.argv[1]))
run = Path(sys.argv[2])
picks = {}
for line in (run / "trajectories.jsonl").read_text().splitlines():
    if line.strip():
        p = json.loads(line)
        picks[p["token"]] = np.asarray(p["model_waypoints"], np.float64)
root = Path(cfg.resolve("{data.navsim.exp}")) / "Drive-JEPA-cache" / "metric_cache_v2"
names = list(grade.SHARED_WITH_CSV)


def lane_keeping(sc, step, need):
    out = []
    for p in range(sc._num_proposals):
        run_, ok = 0, 1.0
        for t in range(0, sc.proposal_sampling.num_poses + 1, step):
            pt = Point(*sc._ego_coords[p, t, BBCoordsIndex.CENTER])
            if sc._drivable_area_map.is_in_layer(pt, layer=SemanticMapLayer.INTERSECTION):
                continue
            run_ = run_ + 1 if pt.distance(sc._centerline.linestring) > sc._config.lane_keeping_deviation_limit else 0
            if run_ >= need:
                ok = 0.0
                break
        out.append(ok)
    return out


def score(mc, simulated):
    final = cns.scorer.score_proposals(
        simulated, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
        mc.map_parameters, None, mc.past_human_trajectory,
    )
    s = cns.after_score_proposals(cns.scorer, final, mc.ego_state, True)[0]
    multi = cns.scorer._multi_metrics[:, 1:].copy()
    return s, lane_keeping(cns.scorer, 5, 4)[1:], lane_keeping(cns.scorer, 1, 20)[1:], multi


out = []
for t in regrade["per_token"]:
    bad = {k: round(d, 4) for k, d in t["diff"].items() if abs(d) > 1e-3}
    if not bad:
        continue
    mc = cns.load_metric_cache(grade.metric_cache_file(root, t["log"], t["token"]))
    ego, smp = mc.ego_state, cns.simulator.proposal_sampling
    # the pick, exactly as grade_poses simulates it
    sim_pick = cns.before_score_proposals(mc, picks[t["token"]][None], ego)
    s_pick, lk_coarse_pick, lk_fine_pick, _ = score(mc, sim_pick)
    # the human, against the same reference row
    ref = get_trajectory_as_array(mc.trajectory, smp, ego.time_point)[:, :3]
    hum = get_trajectory_as_array(transform_trajectory(mc.human_trajectory, ego), smp, ego.time_point)[:, :3]
    ext = np.zeros((2, *ref.shape[:-1], 11))
    ext[0, :, :3], ext[1, :, :3] = ref, hum
    s_h, lk_coarse_h, lk_fine_h, multi_h = score(mc, cns.simulator.simulate_proposals(ext, ego))
    csv_val = {k: t["ours"][names.index(k)] - d for k, d in t["diff"].items()}
    out.append(
        {
            "token": t["token"],
            "metric": list(bad),
            "ours": {k: round(t["ours"][names.index(k)], 4) for k in bad},
            "csv": {k: round(csv_val[k], 4) for k in bad},
            "pick_lane_keeping_coarse_fine": [lk_coarse_pick[0], lk_fine_pick[0]],
            "human_lane_keeping_coarse_fine": [lk_coarse_h[0], lk_fine_h[0]],
            "human_scores": dict(zip(grade.SCORE_NAMES, [round(float(x), 4) for x in s_h[0]])),
            "human_multiplicative": [float(x) for x in multi_h[:, 0]],
            "human_traffic_light": float(s_h[0][names.index("traffic_light_compliance")]),
            "pick_regraded_matches_stored": bool(np.allclose(s_pick[0], t["ours"], atol=1e-6)),
        }
    )
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(len(out), "tokens written")
