#!/usr/bin/env python
"""T05 step 1: per-proposal table (t04 classes + clearance) for every post-handover replan of all 30 scenes under 0.1.6 and of the four
t04 scenes under 0.1.4b. -> cache/props_all.pkl.   CUDA_VISIBLE_DEVICES="" nurec-venv/bin/python t05_menu.py"""
import multiprocessing as mp
import numpy as np, pandas as pd
import t05_common as F
T = F.T


def one(job):
    ver, s = job; o = T.load(ver, s); M = T.menus(ver, s); ho = o["ho"]; rows = []; te, ve, st = F.standing_since(o)
    for tn, z in M.items():
        rel = (tn - ho) / 1e6
        if rel < -1e-6 or tn not in o["egot"]: continue
        e = o["egot"][tn]; ego = e[1:4]; P = z["proposals"].astype(float); Sf = z["pdm_score"].astype(float); Sp = z["S_plain"].astype(float); rf, rp = T.ranks(Sf), T.ranks(Sp); ch = int(z["chosen"]); k = int(np.searchsorted(te, tn))
        for i in range(32):
            rows.append(dict(scene=s, version=ver, replan_idx=int(z["idx"]), tn=tn, t_after_handover_s=rel, proposal=i, **F.classify(o, tn, ego, P[i]), score_plain=Sp[i], rank_plain=int(rp[i]), score_final=Sf[i], rank_final=int(rf[i]), chosen=bool(i == ch),
                             ego_speed_mps=float(ve[k]), standing_s=float(st[k]), standing_s_from_handover=float(min(st[k], rel)) if st[k] >= 0 else -1.0, human_speed_mps=float(np.interp(rel, o["gt_t"], o["gt_v"]))))
    return rows


if __name__ == "__main__":
    jobs = [("0.1.6", s) for s in T.scenes()] + [("0.1.4b", s) for s in T.FOUR]
    with mp.Pool(8) as pool: res = pool.map(one, jobs, chunksize=1)
    D = pd.DataFrame([r for x in res for r in x]); D.to_pickle(F.HERE / "cache" / "props_all.pkl"); print(len(D), D.groupby("version").scene.nunique().to_dict())
    # agreement with t04's own table on the four scenes
    P4 = pd.read_csv(F.HERE.parent / "part4_why_it_stays_stopped" / "proposals.csv"); m = D.merge(P4, on=["scene", "version", "replan_idx", "proposal"], suffixes=("", "_t04"))
    print("rows compared with t04 proposals.csv:", len(m), {c: int((m[c] != m[c + "_t04"]).sum()) for c in ("leave", "prompt", "box_free", "box_free_margin0p3", "chosen")})
    print("box_free vs clearance>0 disagreements:", int((D.box_free != (D.min_clearance_m > 0)).sum()), "| margin0p3 free but clearance<=0.3:", int((D.box_free_margin0p3 & (D.min_clearance_m <= 0.3)).sum()), "| not free at 0.3 but clearance>0.3:", int((~D.box_free_margin0p3 & (D.min_clearance_m > 0.3)).sum()))
