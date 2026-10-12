#!/usr/bin/env python
"""S5 TRIAL copy of baseline_014b/b014b_videos.py for the 0.1.6 arm (diff: tables/s5_videos_diff.txt -- output folder, arm label "016" with the
same overlays as 0.1.4b). Original docstring: DRIVEJEPA_014B site -- copy of s5/s5_videos.py for the 0.1.4b arm (diff: paths, label, the 0.1.5 overlays replaced by the
§3.5 ones: the pick WITHOUT momentum dashed cyan when it differs, and a "§3.5 changed pick" banner on those replans).
Original docstring: S5 §6 -- one overlay video per run (0.1.4 recorded or 0.1.5 + veto), H.264, 720 px wide, 20 fps, one frame per tick.
Overlays burned in, in the LIVE rig frame at the replan tick (the proposal-video GT-frame fix, make_proposal_videos.py:197-210,
gt_knots_in_frame with the controller_request pose): 32 proposals thin blue shaded by pdm_score, driven pick thick red,
human future (t0+0.5..+4.0 s) green, sim time + ego speed (pose differences), and for 0.1.5: L's pick outlined (orange),
the old picker's pick (argmax pdm over all 32) dashed cyan when it differs from the driven pick, a VETO banner on the
replan interval where the veto fired, STOPPED when < 1 m/s after handover (shown on both arms so stalls can be compared),
and a red flash on collision ticks (rollout metrics.parquet collision_any > 0).
  cd paddock-sid/Alpasim && alpasim/.venv/bin/python <this file> LABEL=RUN_DIR:ARM [...]   (menus from s5_menus.py must exist)
"""
from __future__ import annotations

import asyncio, glob, io, json, math, pathlib, subprocess, sys, time
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont

S5 = pathlib.Path(__file__).resolve().parent; VID = S5 / "videos"
sys.path.insert(0, "{runs}/paddock_sid/analysis/drivejepa/step2"); import make_proposal_videos as MV  # noqa: E402
W, Hh = 720, 406; SCL = W / 1920.0; FPS = 20   # 406: yuv420p needs even dimensions (720x405 broke the pipe)
ARM_LABEL = {"014": "0.1.4 recorded", "014b": "0.1.4b (paper 3.5 momentum)", "016": "0.1.6 (S4 weights + 3.5 momentum)"}
RED, GREEN, ORANGE, WHITE, CYAN = (0xE2, 0x4B, 0x4A), (0x22, 0xC5, 0x5E), (0xF9, 0x8C, 0x1A), (245, 245, 245), (0, 220, 255)


def font(sz, bold=False): return ImageFont.truetype(MV.FONT_PATH_B if bold else MV.FONT_PATH, sz)


def collision_ticks(run):
    f = glob.glob(str(run / "rollouts" / "clipgt-*" / "*" / "metrics.parquet"))
    if not f: return set()
    m = pd.read_parquet(f[0]); g = m[(m.name == "collision_any") & (m["values"] > 0)]; return set(int(t) for t in g.timestamps_us)


def dashed(draw, pts, colour, width, on=6, off=5):
    for a, b in zip(pts[:-1], pts[1:]):
        if a is None or b is None: continue
        ax, ay = a; bx, by = b; L = math.hypot(bx - ax, by - ay); n = max(int(L // (on + off)), 1); s = 0.0
        for _ in range(n + 1):
            t0 = min(s / L, 1.0) if L else 1.0; t1 = min((s + on) / L, 1.0) if L else 1.0
            draw.line([(ax + (bx - ax) * t0, ay + (by - ay) * t0), (ax + (bx - ax) * t1, ay + (by - ay) * t1)], fill=colour, width=width); s += on + off
            if s >= L: break


def polyline(draw, pts, colour, width, radius=0):
    MV.draw_polyline(draw, pts, colour, width, radius)


def render(label, run, arm):
    t = time.time(); asl = glob.glob(str(run / "rollouts" / "*" / "*" / "rollout.asl"))[0]
    D = asyncio.run(MV.read_scene(asl)); ticks, meta, gt, cam = D["ticks"], D["meta"], D["gt"], D["front_cam"]; ho = meta["render_start_us"] + 3_000_000
    menus = {int(np.load(p)["tn"]): dict(np.load(p)) for p in sorted((S5 / "menus" / label).glob("replan_*.npz"))}
    assert len(menus) >= 30, (label, len(menus), "menus missing -- run s5_menus.py first")
    proj = MV.Projector(cam); coll = collision_ticks(run); rs = json.load(open(run / "aggregate" / "results-summary.json"))["rollouts"][0]
    out = VID / f"{label}.mp4"; import imageio_ffmpeg; ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-loglevel", "warning", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "24", "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE); fnt, fnt_b, fnt_big = font(13), font(14, bold=True), font(30, bold=True)
    active = None; prev_xy = None; prev_tn = None; flash_until = -1; n = 0
    for i, tk in enumerate(ticks):
        if tk["tn"] in menus: active = tk["tn"]
        ego_now = tk.get("ego"); v = np.nan
        if ego_now is not None and prev_xy is not None: v = float(np.hypot(*(ego_now[:2, 3] - prev_xy)) / max((tk["tn"] - prev_tn) / 1e6, 1e-3))
        if ego_now is not None: prev_xy, prev_tn = ego_now[:2, 3].copy(), tk["tn"]
        img = Image.open(io.BytesIO(tk["frame"])).convert("RGB") if tk["frame"] is not None else Image.new("RGB", (cam["w"], cam["h"]), (0, 0, 0))
        img = img.resize((W, Hh), Image.BILINEAR); d = ImageDraw.Draw(img)
        P2 = lambda pts: [((u * SCL, vv * SCL) if p is not None else None) for p in pts for u, vv in [p if p is not None else (0, 0)]]  # noqa: E731
        veto = False
        if active is not None and ego_now is not None:
            m = menus[active]; ego_f = ticks[[k["tn"] for k in ticks].index(active)]["ego"]; props = m["proposals"][:, :, :2]; S = m["pdm_score"]; chosen = int(m["chosen"]); shipped = int(m["shipped"]); veto = bool(m["changed"]); l_idx = chosen
            now = np.stack([MV.reanchor(props[j], ego_f, ego_now) for j in range(len(props))]); lo, hi = float(S.min()), float(S.max())
            for j in np.argsort(S):
                if j == chosen: continue
                polyline(d, P2(proj.project(now[j])), MV.blue_shade((S[j] - lo) / (hi - lo) if hi > lo else 1.0), 1)
            gk = MV.gt_knots_in_frame(gt, active, ego_f)
            if gk: polyline(d, P2(proj.project(MV.reanchor(np.array(gk), ego_f, ego_now))), GREEN, 3)
            if arm in ("014b", "016"):
                if shipped != chosen: dashed(d, P2(proj.project(now[shipped])), CYAN, 2)
            polyline(d, P2(proj.project(now[chosen])), RED, 4, 3)
        rel = (tk["tn"] - ho) / 1e6
        d.rectangle([0, 0, W, 22], fill=(0, 0, 0)); d.text((6, 4), f"{ARM_LABEL[arm]}  {run.name[:22]}  t {rel:+6.2f} s  ego {(f'{v:4.1f}' if np.isfinite(v) else '  --')} m/s  tick {i:3d}{'  force-GT' if tk['tn'] < ho else ''}", fill=WHITE, font=fnt)
        if arm in ("014b", "016") and veto: d.rectangle([W // 2 - 96, 28, W // 2 + 96, 52], fill=(0, 110, 140)); d.text((W // 2 - 88, 32), "\u00a73.5 changed pick", fill=WHITE, font=fnt_b)
        if tk["tn"] >= ho and np.isfinite(v) and v < 1.0: d.rectangle([W - 130, 30, W - 8, 56], fill=(30, 30, 30)); d.text((W - 122, 34), "STOPPED", fill=(255, 210, 60), font=fnt_b)
        if tk["tn"] in coll: flash_until = i + 10
        if i <= flash_until:
            ov = Image.new("RGB", (W, Hh), (255, 0, 0)); img = Image.blend(img, ov, 0.28); d = ImageDraw.Draw(img); d.text((W // 2 - 70, Hh // 2 - 20), "COLLISION", fill=WHITE, font=fnt_big)
        if tk["tn"] == ho: d.rectangle([0, 0, W - 1, Hh - 1], outline=WHITE, width=4); d.text((W // 2 - 90, Hh // 2 - 20), "HANDOVER", fill=WHITE, font=fnt_big)
        proc.stdin.write(img.tobytes()); n += 1
    proc.stdin.close(); rc = proc.wait(); size = out.stat().st_size if out.exists() else 0
    js = dict(label=label, run=str(run), arm=arm, n_frames=n, fps=FPS, n_menus=len(menus), collision_ticks=len(coll), bytes=size, score=rs["score"], status=rs["status"], failure_reason=rs.get("failure_reason"), secs=round(time.time() - t, 1))
    json.dump(js, open(VID / f"{label}.json", "w"), indent=1); print(f"{label}: {n} frames, {len(menus)} menus, {size / 2**20:.1f} MiB, ffmpeg rc {rc} ({js['secs']:.0f}s)", flush=True); return js


if __name__ == "__main__":
    VID.mkdir(exist_ok=True)
    for spec in sys.argv[1:]:
        label, rest = spec.split("="); run, arm = rest.rsplit(":", 1)
        if (VID / f"{label}.json").is_file(): print(f"{label}: exists, skip"); continue
        render(label, pathlib.Path(run), arm)
