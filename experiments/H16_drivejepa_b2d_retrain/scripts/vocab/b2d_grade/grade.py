"""Vocabulary size by Bench2Drive's scoring, not ADE (Navvrat 05 Nov: "ade is not a metric, b2d scores are"). CPU only.
For sampled frames, every feasible vocabulary path is graded offline with Bench2Drive's terms, on the only scorer that has run on these frames (mtd/scorer.py, read-only):
  grade = route share x 0.6^collision x 0.7^red x (1 - share of the path outside lanes or wrong way)
    route share  metres along the expert's path, capped at the expert's own: going further costs nothing, as in Bench2Drive (scorer.progress cuts overshoot, so it is not used)
    collision    collision_reactive with the expert mask, as MTD ran it. 0.6 is the vehicle coefficient: the scorer does not say what was hit (pedestrian 0.5, static 0.65 in Bench2Drive)
    red          the scorer's port of RunningRedLightTest
    outside      per 0.1 s step, off lane or oncoming
  Not in it: stop signs (no rule in this scorer), yield to EV, blocked, timeouts: route-level or scenario state. Other actors are log replay beyond the follower model.
Feasible = MTD's filter: first-step speed within 2 m/s of the ego's, first-step heading under 15 deg. A vocabulary path scored 0.95 or more is a teacher (Drive-JEPA's rule).
Vocabulary paths are rear-axle poses; the scorer's are the actor centre, so they are shifted back with the same offset (geometry.B2D_REAR_AXLE_OFFSET).
   $PY grade.py <train frames per clip> <train clips> <held-out frames per clip> <out.jsonl>"""
import json, os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
torch.set_num_threads(6)
sys.path.insert(0, "{ops}/mtd"); import scorer
sys.path.insert(0, "{ops}/run3a-adapted")
from phase1.data import geometry as G
OFF = G.B2D_REAR_AXLE_OFFSET
V = "{ops}/b2d_drivejepa/vocab"; CTX = "{ops}/mtd/context/b2d"
EM = ("collision", "lane", "direction")
NAVSIM = "{workspace_root}/wa-jepa/Drive-JEPA-upstream/navsim_v2/data/8192.npy"
VOCABS = [(f"b2d_{k}", f"{V}/b2d_{k}.npy") for k in (512, 1024, 2048, 4096, 8192)] + [("navsim_8192", NAVSIM)]


def to_centre(p):
    q = p.copy().astype(np.float64)
    q[..., 0] = p[..., 0] + OFF * np.cos(p[..., 2]) - OFF
    q[..., 1] = p[..., 1] + OFF * np.sin(p[..., 2])
    return q.astype(np.float32)


VOC = {n: to_centre(np.load(f)[:, 4::5, :]) for n, f in VOCABS}


def route_share(ex, cand):
    """ex (8, 3), cand (C, 8, 3) -> metres along the expert polyline of each end pose, capped at the expert's length, as a share of it."""
    path = np.vstack([[0, 0], ex[:, :2]]).astype(np.float64)
    v = np.diff(path, axis=0); L2 = (v * v).sum(1); seg = np.sqrt(L2); cum = np.r_[0, np.cumsum(seg)]; dexp = cum[-1]
    if dexp < 0.5:
        return np.ones(len(cand))
    p = cand[:, 7, :2].astype(np.float64)
    rel = p[:, None] - path[None, :-1]
    tt = np.where(L2 < 1e-12, 0, np.clip((rel * v[None]).sum(-1) / np.maximum(L2, 1e-12), 0, 1))
    q = path[None, :-1] + tt[..., None] * v[None]
    b = np.linalg.norm(p[:, None] - q, axis=-1).argmin(1); i = np.arange(len(p))
    return np.minimum((cum[b] + tt[i, b] * seg[b]) / dexp, 1.0)


def grade(o, f, ex, cand):
    n = len(cand)
    col, red = 1 - o["collision_reactive"][f, :n].numpy(), 1 - o["red"][f, :n].numpy()
    bad = (o["off"][f, :n] | o["onc"][f, :n]).float().mean(-1).numpy()
    return route_share(ex, cand) * 0.6 ** col * 0.7 ** red * (1 - bad), col, red, bad


def frames_of(clip, n):
    z = np.load(f"{CTX}/{clip}.npz", allow_pickle=True)
    fid, ok = z["frame_id"], z["expert_ok"]
    num = np.array([int(str(f).rsplit("|", 1)[1]) // 100 for f in fid])
    idx = np.where(ok & (num % 5 == 0))[0]
    return fid[idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]] if len(idx) else []


def main():
    ntr, ctr, nho = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    sp = json.load(open("{ops}/b2d_drivejepa/phase1/splits/clips.json"))
    have = {f[:-4] for f in os.listdir(CTX)}
    ho = sorted(set(sp["heldout"]) & have)
    tr = sorted(set(sp["train"]) & have)
    tr = [tr[i] for i in np.linspace(0, len(tr) - 1, ctr).astype(int)] if ctr else []
    out = open(f"{V}/b2d_grade/{sys.argv[4]}", "a")
    t0, nf = time.time(), 0
    for split, clips, n in (("heldout", ho, nho), ("train", tr, ntr)):
        for ci, clip in enumerate(clips):
            fids = list(frames_of(clip, n))
            if not fids:
                continue
            c0 = scorer.load_ctx(f"{CTX}/{clip}.npz", fids)
            ctx = scorer.collate([c0], device="cpu")
            ex, v0s = c0["expert"].astype(np.float32), c0["v0"]
            with torch.no_grad():
                g = scorer.prepare(ctx, "cpu")
                eo = scorer.score(ctx, torch.from_numpy(ex[:, None]), "cpu", prepared=g, expert_mask=EM, steps=True)
                rows = [dict(split=split, clip=clip, frame=str(f), v0=float(v0s[i]), exp_len=float(np.hypot(*np.diff(np.vstack([[0, 0], ex[i, :, :2]]), axis=0).T).sum()),
                             exp_turn=float(abs(np.degrees(np.arctan2(np.sin(ex[i, -1, 2]), np.cos(ex[i, -1, 2]))))),
                             exp_grade=float(grade(eo, i, ex[i], ex[i][None])[0][0]), v={}) for i, f in enumerate(fids)]
                for name, Vc in VOC.items():
                    vi = np.hypot(Vc[:, 0, 0], Vc[:, 0, 1]) / 0.5; dy = np.degrees(np.abs(Vc[:, 0, 2]))
                    surv = [np.where((np.abs(vi - v0s[i]) <= 2.0) & (dy < 15.0))[0] for i in range(len(fids))]
                    C = max(1, max(len(s) for s in surv))
                    cand = np.zeros((len(fids), C, 8, 3), np.float32)
                    for i, s in enumerate(surv):
                        cand[i, :len(s)] = Vc[s]
                    o = scorer.score(ctx, torch.from_numpy(cand), "cpu", prepared=g, expert_mask=EM, steps=True)
                    for i, s in enumerate(surv):
                        if len(s) == 0:
                            rows[i]["v"][name] = dict(feasible=0, best=0.0, teachers=0, spread=0.0, clean=0)
                            continue
                        gr, col, red, bad = grade(o, i, ex[i], cand[i, :len(s)])
                        t = gr >= 0.95
                        e = cand[i, :len(s)][t][:, 7, :2]
                        spread = float(np.linalg.norm(e - e.mean(0), axis=1).mean()) if len(e) > 1 else 0.0
                        lat = float(e[:, 1].max() - e[:, 1].min()) if len(e) > 1 else 0.0
                        rows[i]["v"][name] = dict(feasible=int(len(s)), best=float(gr.max()), teachers=int(t.sum()), spread=spread, lat=lat, clean=int(((col == 0) & (red == 0) & (bad == 0)).sum()),
                                                  safe_share=float(((col == 0) & (red == 0) & (bad == 0)).mean()))
            for r in rows:
                out.write(json.dumps(r) + "\n")
            out.flush()
            nf += len(rows)
            print(f"{split} {ci + 1}/{len(clips)} {clip[:40]} frames {nf} {time.time() - t0:.0f} s", flush=True)
    print("GRADE_DONE", nf, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
