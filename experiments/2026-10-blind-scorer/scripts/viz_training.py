"""Plot training samples from cache v2: every plan coloured by its label, beside the held-out prediction.

Ticks are chosen by a fixed rule (no cherry-picking): for each scene x shape, episode run_0 of round0_v5
(released model drove); frontal/stationary: the tick where the released pick hits with time-to-hit closest
to 3 s and the hazard in view; side (pick never hits): the in-view tick where the target is closest to
20 m. Plus one GT (human drove) and one round1 (s2b drove) frontal tick for 0075.

Predictions: the 5-seed average of the probe+geometry scorer from the fold where this scene was HELD OUT.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "{gallops}/src")
from gallops.expert.drivable import DrivableArea
from gallops.expert.grader import actor_at, corners
from gallops.expert.lanes import load_references, reference_at

C = Path("{dj_cache}/v2")
R = Path("{runs}")
CALIB = Path("{paddock_runs}/ops/ncap_viewer/work")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
STATUS = {"clear": "#0ca30c", "off-lane": "#fab219", "off-road": "#ec835a", "hit": "#d03b3b"}
ORDER = ("clear", "off-lane", "off-road", "hit")
PICK_RGB = "#e040fb"  # the trained scorer's (gated) pick
sig = lambda z: 1 / (1 + np.exp(-z))

P = np.load(C / "probe_input.npz")
episodes = P["episode"]
first = {}
for i, e in enumerate(episodes):
    first.setdefault(str(e), i)
logits = {}
for scene in ("0053", "0058", "0064", "0070", "0075"):
    L = [np.load(C / f"probe_attngeo_s{s}.logits.npz") for s in range(5)]
    te = L[0][f"test_idx_{scene}"]
    mean_sig = np.mean([sig(l[f"logits_{scene}"]) for l in L], 0)
    logits[scene] = (te, mean_sig)


def predicted(global_idx, scene):
    te, s = logits[scene]
    k = np.searchsorted(te, global_idx)
    assert te[k] == global_idx
    return s[k]  # (64, 6) sigmoids


def category(hit, road, dev):
    if not np.isnan(hit): return "hit"
    if not np.isnan(road): return "off-road"
    if dev > 1.75: return "off-lane"
    return "clear"


def targets(lab, i, p):
    nc = float(np.isnan(lab["hit5_at"][i, p])); dac = float(np.isnan(lab["off_road_at"][i, p]))
    ttc = float(np.isnan(lab["ttc_at"][i, p])); lk = float(lab["max_deviation"][i, p] <= 1.75)
    ep = float(min(1.0, lab["travel"][i, p] / max(lab["human_travel"][i], 1e-3)))
    return [nc, dac, ttc, lk, ep, nc * dac * (5 * ep + 5 * ttc + 2 * lk) / 12]


def formula(s):
    return s[..., 0] ** 4 * s[..., 1] * (5 * s[..., 4] + 5 * s[..., 2] + 2 * s[..., 3]) / 12


def project(xy, cam2ego, K):
    plan = np.vstack([[0.0, 0.0], xy])
    ego2cam = np.linalg.inv(cam2ego); cam2img = np.eye(4); cam2img[:3, :3] = K
    s = np.linspace(0, 1, len(plan) * 20)
    x = np.interp(s, np.linspace(0, 1, len(plan)), plan[:, 0]); y = np.interp(s, np.linspace(0, 1, len(plan)), plan[:, 1])
    pts = np.stack([x, y, np.full_like(x, 0.25), np.ones_like(x)], 1)
    cam = (ego2cam @ pts.T).T; img = ((cam2img @ ego2cam) @ pts.T).T
    uv = img[:, :2] / img[:, 2:3]
    return uv, (cam[:, 2] > 0.5) & (uv > -400).all(1) & (uv < [2000, 1300]).all(1)


def line(draw, uv, keep, rgb, w):
    run = []
    for p, k in zip(uv, keep):
        if k: run.append(tuple(p))
        elif len(run) > 1: draw.line(run, fill=rgb, width=w); run = []
        else: run = []
    if len(run) > 1: draw.line(run, fill=rgb, width=w)


def hexrgb(h, a=255): return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) + (a,)


def choose(src, name):
    f = C / src / f"{name}__run_0.npz"
    z, lab = np.load(f), np.load(str(f).replace(".npz", ".labels.npz"))
    ep = Path(str(z["frame_dir"])); poses = json.loads((ep / "ego_poses.json").read_text())
    tg = [a for a in json.loads((ep / "actors.json").read_text()) if a["is_target_actor"]][0]
    best = None
    for i, ts in enumerate(z["timestamp"]):
        if not z["in_frame"][i]: continue
        rel = int(z["pdm_score"][i].argmax()); th = lab["hit5_at"][i, rel]
        if name.startswith("side"):
            st = actor_at(tg, float(ts))
            if st is None: continue
            e2w = np.asarray(poses[str(int(ts))]); gap = abs(np.linalg.norm(st[0] - e2w[:2, 3]) - 20)
        else:
            if np.isnan(th): continue
            gap = abs(th - 3.0)
        if best is None or gap < best[0]: best = (gap, i)
    return (f, z, lab, best[1]) if best else None


def render(src, name, f, z, lab, i):
    from PIL import Image, ImageDraw
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.patches import Polygon
    ts = int(z["timestamp"][i]); shape, scene = name.split("-"); ep = Path(str(z["frame_dir"]))
    g = first[f"{src}/{f.stem}"] + i
    plans = np.concatenate([z["natives"][i], z["vocab"][i][:32]])  # 64 candidates
    assert np.allclose(plans, P["plans"][g]), "tick index mismatch between cache and training input"
    probes = z["vocab"][i][32:]
    cats = [category(lab["hit5_at"][i, p], lab["off_road_at"][i, p], lab["max_deviation"][i, p]) for p in range(64)]
    s = predicted(g, scene); rel = int(z["pdm_score"][i].argmax())
    fs = formula(s); alt = int(fs.argmax()); veto = s[rel, 0] < 0.5; pick = alt if veto else rel
    e2w = np.asarray(json.loads((ep / "ego_poses.json").read_text())[str(ts)])
    stem = f"{src}_{name}_{ts}"
    # camera
    calib = json.loads((CALIB / f"calib_scene-{scene}.json").read_text()); cam2ego, K = np.asarray(calib["cam2ego"]), np.asarray(calib["K"])
    img = Image.open(ep / "CAM_FRONT" / f"{ts}.jpg").convert("RGB"); over = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(over)
    for pr in probes:
        uv, k = project(pr[:, :2], cam2ego, K)
        for (u, v), kk in zip(uv[::20], k[::20]):
            if kk: dr.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(90, 200, 255, 150))
    for p in range(64):
        uv, k = project(plans[p, :, :2], cam2ego, K)
        line(dr, uv, k, hexrgb(STATUS[cats[p]], 110 if p < 32 else 220), 2 if p < 32 else 4)
    for p, col, w in ((rel, (255, 255, 255, 255), 7), (pick, hexrgb(PICK_RGB), 5)):
        uv, k = project(plans[p, :, :2], cam2ego, K); line(dr, uv, k, (0, 0, 0, 200), w + 4); line(dr, uv, k, col, w)
    Image.alpha_composite(img.convert("RGBA"), over).convert("RGB").resize((1200, 675)).save(OUT / f"{stem}_cam.jpg", quality=85)
    # BEV
    actors = json.loads((ep / "actors.json").read_text())
    vehicles = [a for a in actors if a["dims"][1] >= 1.2 and len(a["timestamps"]) >= 2]
    area = DrivableArea.load(R / "drivable" / f"scene-{scene}.npz"); logged, lanes = load_references(ep); ref, _ = reference_at(ts, logged, lanes)
    rot, org = e2w[:2, :2], e2w[:2, 3]; toe = lambda w: (np.asarray(w)[:, :2] - org) @ rot
    surface, ink, muted = "#1a1a19", "#ffffff", "#c3c2b7"
    fig, ax = plt.subplots(figsize=(6.2, 8.6), facecolor=surface); ax.set_facecolor(surface)
    xs = np.arange(-8, 92, 0.5); ys = np.arange(-20, 20, 0.5); gx, gy = np.meshgrid(xs, ys, indexing="ij")
    drv = area.contains(np.stack([gx.ravel(), gy.ravel()], 1) @ rot.T + org).reshape(gx.shape)
    ax.pcolormesh(-gy, gx, np.where(drv, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#33332f"]), shading="auto", zorder=0)
    re = toe(ref); ax.plot(-re[:, 1], re[:, 0], color=muted, lw=1, ls=(0, (4, 4)), zorder=1)
    for a in vehicles:
        st = actor_at(a, float(ts))
        if st is None: continue
        tg = a.get("is_target_actor"); box = toe(corners(st[0], st[1], a["dims"][1], a["dims"][0]))
        ax.add_patch(Polygon(np.c_[-box[:, 1], box[:, 0]], closed=True, fill=tg, fc="#d9d6cc" if tg else "none", ec=ink if tg else "#7d7c76", lw=1.3 if tg else 0.8, zorder=3))
        if tg:
            for kk in (1, 2, 3, 4, 5):
                later = actor_at(a, float(ts) + kk * 1e6)
                if later is None: continue
                b2 = toe(corners(later[0], later[1], a["dims"][1], a["dims"][0])); c0 = toe(later[0][None])[0]
                ax.add_patch(Polygon(np.c_[-b2[:, 1], b2[:, 0]], closed=True, fill=False, ec=ink, lw=0.7, ls=(0, (2, 2)), alpha=0.7, zorder=3))
                ax.text(-c0[1] + 1.6, c0[0], f"+{kk}s", color=muted, fontsize=7, va="center", zorder=6)
    for pr in probes:
        ax.plot(-pr[:, 1], pr[:, 0], ".", color="#5ac8ff", ms=2.5, alpha=0.6, zorder=2)
    for p in range(64):
        q = np.vstack([[0, 0], plans[p, :, :2]])
        ax.plot(-q[:, 1], q[:, 0], color=STATUS[cats[p]], lw=0.8 if p < 32 else 1.6, alpha=0.55 if p < 32 else 0.9, zorder=4)
    for p, col, w in ((rel, ink, 2.6), (pick, PICK_RGB, 2.0)):
        q = np.vstack([[0, 0], plans[p, :, :2]]); ax.plot(-q[:, 1], q[:, 0], color=surface, lw=w + 2, zorder=6); ax.plot(-q[:, 1], q[:, 0], color=col, lw=w, zorder=7)
    egob = corners(np.array([0.5, 0.0]), 0.0, 4.018, 1.85); ax.add_patch(Polygon(np.c_[-egob[:, 1], egob[:, 0]], closed=True, fill=False, ec=ink, lw=1.1, zorder=5))
    ax.set_xlim(-18, 18); ax.set_ylim(-6, 70); ax.set_aspect("equal"); ax.tick_params(colors=muted, labelsize=7)
    ax.set_xticks([-15, -10, -5, 0, 5, 10, 15]); ax.set_xticklabels(["15L", "10L", "5L", "0", "5R", "10R", "15R"]); ax.grid(color="#3a3a37", lw=0.4); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_color("#3a3a37")
    fig.tight_layout(); fig.savefig(OUT / f"{stem}_bev.png", dpi=100, facecolor=surface); plt.close(fig)
    # table rows: released pick, model pick, best-by-label, and a few swerves
    best_label = int(np.argmax([targets(lab, i, p)[5] for p in range(64)]))
    rows = []
    for tag, p in (("released pick", rel), ("model pick" + (" (override)" if veto else " (= released)"), pick), ("best by label", best_label),
                   ("swerve: straight +2.5 m", 32 + 14 + 6), ("swerve: straight -2.5 m", 32 + 14 + 7), ("swerve: brake +1.0 m", 32 + 28)):
        rows.append({"plan": tag, "index": p, "travel": float(lab["travel"][i, p]), "label": targets(lab, i, p), "pred": [float(x) for x in s[p]], "formula": float(fs[p]), "cat": cats[p]})
    return {"stem": stem, "src": src, "episode": name, "t": float((ts - z["timestamp"][0]) / 1e6), "speed": float(z["speed"][i]), "human_travel": float(lab["human_travel"][i]),
            "rel_hit": None if np.isnan(lab["hit5_at"][i, rel]) else float(lab["hit5_at"][i, rel]), "veto": bool(veto),
            "counts_native": {c: cats[:32].count(c) for c in ORDER}, "counts_swerve": {c: cats[32:].count(c) for c in ORDER}, "rows": rows}


report = []
picks = [("round0_v5", f"{shape}-{scene}") for scene in ("0053", "0058", "0064", "0070", "0075") for shape in ("frontal", "stationary", "side")]
picks += [("gt", "frontal-0075"), ("round1_s2", "frontal-0075")]
for src, name in picks:
    got = choose(src, name)
    if got is None:
        print(src, name, "no qualifying tick"); continue
    report.append(render(src, name, *got)); print(src, name, "t", round(report[-1]["t"], 1), flush=True)
(OUT / "report.json").write_text(json.dumps(report, indent=1))

# label statistics over the whole training set, per scene x shape
stats = {}
safe = np.isnan(P["hit5_at"]) & np.isnan(P["off_road_at"])
for scene in ("0053", "0058", "0064", "0070", "0075"):
    for shape in ("frontal", "stationary", "side"):
        m = (P["scene"] == scene) & (P["shape"] == shape)
        if not m.any(): continue
        stats[f"{shape}-{scene}"] = {"ticks": int(m.sum()), "hit": float((~np.isnan(P["hit5_at"][m])).mean()), "off_road": float((~np.isnan(P["off_road_at"][m])).mean()),
                                      "ttc_fail": float((~np.isnan(P["ttc_at"][m])).mean()), "off_lane": float((P["max_deviation"][m] > 1.75).mean()),
                                      "any_safe_tick": float(safe[m].any(1).mean()), "human_speed": float(np.median(P["human_travel"][m]) / 4),
                                      "native_off_road": float((~np.isnan(P["off_road_at"][m][:, :32])).mean())}
(OUT / "stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=0)[:3000])
