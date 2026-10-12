"""Draw the reviewed 32-path vocabulary (vocab_review/vocab.py) on the same six ticks as the page."""
import importlib.util, json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, "{gallops}/src")
import swerve_vocab as sv

from gallops.expert import fastgrade
from gallops.expert.drivable import DrivableArea
from gallops.expert.lanes import load_references, reference_at
from gallops.expert.grader import actor_at, corners

spec = importlib.util.spec_from_file_location("v", "vocab_review/vocab.py"); V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
SW = Path("swerves"); R = sv.R
reports = {}
for f in SW.glob("report_*.json"):
    for r in json.loads(f.read_text()):
        reports[r["episode"]] = r
index = sv.dump_index()
GROUPS = [("pick", "pick's path, current speed"), ("straight", "straight ahead, current speed"), ("brake", "brake 3 m/s², nudge")]
summary = {}
for name, r in reports.items():
    shape, scene_no = name.split("-"); scene = f"scene-{scene_no}"
    ep = R / "collect" / name / f"run_{r['run']}"; ts = r["tick"]
    rows = sv.dump_rows(ep, index); row = rows[ts]
    natives = np.asarray(row["proposals"], float); pi = int(np.argmax(row["pdm_score"])); pick = natives[pi]
    paths, meta = V.vocabulary(natives, pi, float(row["speed"]), float(row["accel"]))
    poses = json.loads((ep / "ego_poses.json").read_text()); e2w = np.asarray(poses[str(ts)])
    actors = json.loads((ep / "actors.json").read_text())
    vehicles = [a for a in actors if a["dims"][1] >= 1.2 and len(a["timestamps"]) >= 2]
    targets = {a["uuid"] for a in actors if a["is_target_actor"]}
    area = DrivableArea.load(R / "drivable" / f"{scene}.npz"); logged, lanes = load_references(ep); ref, _ = reference_at(ts, logged, lanes)
    g = fastgrade.grade_many(np.concatenate([natives, paths]), e2w, ts, vehicles, ref, area)
    cats = []
    for k in range(32, 64):
        cats.append("hit" if not np.isnan(g["hit_at"][k]) else "off-road" if not np.isnan(g["off_road_at"][k]) else "off-lane" if g["max_deviation"][k] > 1.75 else "clear")
    nat_safe = int((np.isnan(g["hit_at"][:32]) & np.isnan(g["off_road_at"][:32])).sum())
    summary[name] = {"counts": {c: cats.count(c) for c in sv.ORDER}, "native_safe": nat_safe,
                     "groups": {gname: {c: sum(1 for m, cc in zip(meta, cats) if m["base"] == gname and cc == c) for c in sv.ORDER} for gname, _ in GROUPS},
                     "max_lat": max(m["peak_lat_accel"] for m in meta), "min_radius": min(m["min_radius_m"] for m in meta)}
    # camera: the 14 "pick" paths at current speed
    from PIL import Image, ImageDraw
    calib = json.loads((sv.CALIB / f"calib_{scene}.json").read_text()); cam2ego, K = np.asarray(calib["cam2ego"]), np.asarray(calib["K"])
    img = Image.open(Path(json.loads((R / "labelled_round0_v5" / f"{name}__run_{r['run']}.jsonl").read_text().splitlines()[0])["frame"]).parent / f"{ts}.jpg").convert("RGB")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(over)
    for p in natives:
        uv, keep = sv.project(p[:, :2], cam2ego, K); sv.draw_line(dr, uv, keep, (235, 235, 235, 70), 2)
    for p, m, c in zip(paths, meta, cats):
        if m["base"] != "pick": continue
        uv, keep = sv.project(p[:, :2], cam2ego, K); sv.draw_line(dr, uv, keep, (0, 0, 0, 150), 8); sv.draw_line(dr, uv, keep, sv.hex_rgb(sv.STATUS[c]), 5)
    uv, keep = sv.project(pick[:, :2], cam2ego, K); sv.draw_line(dr, uv, keep, (0, 0, 0, 200), 11); sv.draw_line(dr, uv, keep, (255, 255, 255, 255), 7)
    Image.alpha_composite(img.convert("RGBA"), over).convert("RGB").resize((1200, 675)).save(SW / f"{r['stem']}_v2_cam.jpg", quality=86)
    # BEV: three panels by base
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.patches import Polygon
    surface, ink, muted, grid_ink = "#1a1a19", "#ffffff", "#c3c2b7", "#3a3a37"
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 7.4), facecolor=surface)
    xs = np.arange(-8, 48, 0.4); ys = np.arange(-16, 16, 0.4); gx, gy = np.meshgrid(xs, ys, indexing="ij")
    world = np.stack([gx.ravel(), gy.ravel()], 1) @ e2w[:2, :2].T + e2w[:2, 3]; drv = area.contains(world).reshape(gx.shape)
    ref_e = sv.to_ego(ref, e2w)
    for ax, (gname, title) in zip(axes, GROUPS):
        ax.set_facecolor(surface)
        ax.pcolormesh(-gy, gx, np.where(drv, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#33332f"]), shading="auto", zorder=0)
        ax.plot(-ref_e[:, 1], ref_e[:, 0], color=muted, lw=1.0, ls=(0, (4, 4)), zorder=1)
        for a in vehicles:
            st = actor_at(a, float(ts))
            if st is None: continue
            box = sv.to_ego(corners(st[0], st[1], a["dims"][1], a["dims"][0]), e2w); tg = a["uuid"] in targets
            ax.add_patch(Polygon(np.c_[-box[:, 1], box[:, 0]], closed=True, fill=tg, fc="#d9d6cc" if tg else "none", ec=ink if tg else "#7d7c76", lw=1.4 if tg else 0.9, zorder=3))
            if tg:
                for k in (1, 2, 3, 4):
                    later = actor_at(a, float(ts) + k * 1e6)
                    if later is None: continue
                    c0 = sv.to_ego(later[0][None], e2w)[0]; box = sv.to_ego(corners(later[0], later[1], a["dims"][1], a["dims"][0]), e2w)
                    ax.add_patch(Polygon(np.c_[-box[:, 1], box[:, 0]], closed=True, fill=False, ec=ink, lw=0.8, ls=(0, (2, 2)), alpha=0.75, zorder=3))
                    ax.text(-c0[1] + 1.4, c0[0], f"+{k} s", color=muted, fontsize=8, va="center", zorder=6)
        ego = corners(np.array([0.5, 0.0]), 0.0, 4.018, 1.85); ax.add_patch(Polygon(np.c_[-ego[:, 1], ego[:, 0]], closed=True, fill=False, ec=ink, lw=1.2, zorder=4))
        for p in natives:
            q = np.vstack([[0, 0], p[:, :2]]); ax.plot(-q[:, 1], q[:, 0], color="#8f8e88", lw=0.7, alpha=0.5, zorder=2)
        cnt = {c: 0 for c in sv.ORDER}
        for p, m, c in zip(paths, meta, cats):
            if m["base"] != gname: continue
            cnt[c] += 1; q = np.vstack([[0, 0], p[:, :2]])
            ax.plot(-q[:, 1], q[:, 0], color=surface, lw=3.6, zorder=4); ax.plot(-q[:, 1], q[:, 0], color=sv.STATUS[c], lw=2.0, zorder=5); ax.plot(-q[-1, 1], q[-1, 0], "o", ms=3.5, color=sv.STATUS[c], zorder=5)
        q = np.vstack([[0, 0], pick[:, :2]]); ax.plot(-q[:, 1], q[:, 0], color=surface, lw=4.6, zorder=6); ax.plot(-q[:, 1], q[:, 0], color=ink, lw=2.4, zorder=7)
        ax.set_xlim(-14, 14); ax.set_ylim(-6, 44); ax.set_aspect("equal"); ax.tick_params(colors=muted, labelsize=8)
        for s in ax.spines.values(): s.set_color(grid_ink)
        ax.set_xticks([-10, -5, 0, 5, 10]); ax.set_xticklabels(["10 L", "5 L", "0", "5 R", "10 R"]); ax.grid(color=grid_ink, lw=0.5); ax.set_axisbelow(True)
        ax.set_title(title, color=ink, fontsize=10.5, loc="left", pad=20)
        ax.text(0, 1.012, "   ".join(f"{c} {cnt[c]}" for c in sv.ORDER if cnt[c]), transform=ax.transAxes, color=muted, fontsize=8.5, va="bottom")
    axes[0].set_ylabel("metres ahead", color=muted, fontsize=9); fig.tight_layout(w_pad=1.2); fig.savefig(SW / f"{r['stem']}_v2_bev.png", dpi=100, facecolor=surface); plt.close(fig)
    print(name, summary[name], flush=True)
(SW / "v2_summary.json").write_text(json.dumps(summary, indent=1))
