#!/usr/bin/env python
"""T05 Part 2, step 2: build the paths to inject at the probe ticks (car frame at the tick: x forward, y left, heading; 8 poses 0.5 s
apart) with their box-check classes. -> cache/probe_paths.json.   nurec-venv/bin/python t05_probe_paths.py"""
import json, math
import numpy as np, pandas as pd
import t05_common as F
T = F.T; C = F.C
TICKS = {"d2715023": (0.0, 1.0, 5.0), "b44a4a44": (0.0, 3.0, 5.0)}; K8 = 0.5 * np.arange(1, 9)


def shifted(P, s):
    """Proposal P (8,3) with its end moved s metres to the left, the shift scaled along the path by arc-length fraction; the heading
    is turned by the change of each segment's direction."""
    p9 = F.plan9(P); arc = np.r_[0, np.cumsum(np.hypot(*np.diff(p9[:, :2], axis=0).T))]; q = p9.copy(); q[:, 1] += s * arc / max(arc[-1], 1e-9)
    a0 = np.arctan2(np.diff(p9[:, 1]), np.diff(p9[:, 0])); a1 = np.arctan2(np.diff(q[:, 1]), np.diff(q[:, 0])); q[1:, 2] += np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0)); return q[1:]


def main():
    D = pd.read_pickle(F.HERE / "cache" / "props_all.pkl"); D = D[D.version == "0.1.6"]; out = []
    for s, ticks in TICKS.items():
        o = T.load("0.1.6", s); M = T.menus("0.1.6", s); ho = o["ho"]; hs = T.human_start(o); hs_tick = math.ceil(hs / 0.5 - 1e-9) * 0.5
        for rel in ticks:
            tn = int(ho + rel * 1e6); z = M[tn]; ego = o["egot"][tn][1:4]; g = D[(D.scene == s) & (np.abs(D.t_after_handover_s - rel) < 1e-6)].set_index("proposal"); P = z["proposals"].astype(float); ch = int(z["chosen"])
            us = g[g.leave & g.prompt & g.box_free]; pl = g[g.leave & g.prompt]; base_kind = "best usable escape" if len(us) else "best leave+prompt (no usable escape on this menu)"; base = int((us if len(us) else pl).score_final.idxmax())
            lg = g[g.leave & ~g.prompt]; slots = dict(chosen_stay_slot=ch, base_escape_slot=base, best_late_go_slot=int(lg.score_final.idxmax()) if len(lg) else int(g.rank_final.idxmax()), lowest_ranked_slot=int(g.score_final.idxmin()))
            used = []
            for k, v in list(slots.items()):
                if v in used: v = int(g.drop(index=used).rank_final.sub(16).abs().idxmin()); slots[k] = v      # keep four different slots
                used.append(v)
            hw = T.gt_at(o, tn + K8 * 1e6); hfut = C.to_rig(hw, ego)                                           # (i) the human's own future from this tick, in the car's frame
            th = min(rel, hs_tick); hw0 = T.gt_at(o, ho + (th + np.r_[0, K8]) * 1e6); htr = C.to_rig(hw0, hw0[0])[1:]   # (ii) the human's pull-out (from min(tick, human-start tick)) placed at the car's pose
            paths = {"i_human_future_from_tick": hfut, "ii_human_pullout_transplanted": htr, "iii_base_escape_shift_0.0m": P[base]}
            for sh in (0.5, 1.0, 1.5, 2.0): paths[f"iii_base_escape_shift_{sh:.1f}m"] = shifted(P[base], sh)
            paths["iv_stay_zeros"] = np.zeros((8, 3)); paths["v_chosen_stay_path"] = P[ch]
            cls = {k: F.classify(o, tn, ego, v) for k, v in paths.items()}
            out.append(dict(scene=s, t_after_handover_s=rel, tn=tn, idx=int(z["idx"]), human_start_s=hs, human_pullout_origin_s=th, human_offset_from_car_m=float(np.hypot(*(T.gt_at(o, [tn])[0, :2] - ego[:2]))), base_kind=base_kind, slots=slots,
                            paths={k: np.asarray(v).tolist() for k, v in paths.items()}, classes={k: {a: (b if not isinstance(b, (np.floating, np.bool_)) else b.item()) for a, b in c.items()} for k, c in cls.items()}))
            print(s, rel, slots, base_kind, {k: ("leave" if c["leave"] else "stay") + ("/prompt" if c["prompt"] else "") + ("/free" if c["box_free"] else "/overlap") + f" cl={c['min_clearance_m']:.2f}" for k, c in cls.items()})
    json.dump(out, open(F.HERE / "cache" / "probe_paths.json", "w"), default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)


if __name__ == "__main__":
    main()
