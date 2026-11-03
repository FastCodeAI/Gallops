#!/usr/bin/env python
"""S1 checks C1-C5 + descriptive tables. Run: cd paddock-sid/Alpasim && CUDA_VISIBLE_DEVICES="" alpasim/.venv/bin/python <this>"""
import gzip, pickle, random, sys, pathlib, json
import cv2, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import s1_convert as S  # noqa: E402
C, P, K = S.C, S.P, S.K
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
sm = pd.read_csv(HERE / "samples.csv"); clips = pd.read_csv(HERE / "pilot_clips.csv"); C.assert_train(clips.uuid); assert set(sm.uuid) == set(clips.uuid)
out = []; W = lambda s="": (out.append(s), print(s))
def tbl(df):
    f = lambda v: "" if (isinstance(v, float) and np.isnan(v)) or v is None else (f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v))
    W("| " + " | ".join(map(str, df.columns)) + " |"); W("|" + "---|" * len(df.columns))
    for r in df.itertuples(index=False): W("| " + " | ".join(f(v) for v in r) + " |")
    W()
# ---------- descriptive
W("### Samples"); W()
g = sm.groupby("type").agg(clips=("uuid", "nunique"), anchors=("token", "size"), speed_med=("speed", "median"), speed_max=("speed", "max")).reset_index(); tbl(g)
W(f"Frame timing: |frame − target| max {max(sm.dt_older_us.abs().max(), sm.dt_newer_us.abs().max())/1e3:.1f} ms (older {sm.dt_older_us.abs().max()/1e3:.1f}, newer {sm.dt_newer_us.abs().max()/1e3:.1f}); anchors > 20 ms: {int(((sm.dt_older_us.abs() > 20000) | (sm.dt_newer_us.abs() > 20000)).sum())}; pair gap − 500 ms: max |{(sm.pair_gap_us - 500000).abs().max()/1e3:.2f}| ms; egomotion lag (backward difference) max {sm.dyn_lag_us.max()/1e3:.0f} ms.")
W(f"ego_status rows per sample: {sm.n_status_rows.value_counts().sort_index().to_dict()} (4 = upstream count; fewer at t = 0.5 / 1.0 s).")
W(f"Command (human future path as route): {sm.command.value_counts().to_dict()}; by type: " + "; ".join(f"{t} {d.command.value_counts().to_dict()}" for t, d in sm.groupby("type")) + ".")
okrg = sm[~sm.command_routegen.str.startswith("ERR")]; W(f"AlpaSim RouteGeneratorRecorded comparison: available on {len(okrg)}/{len(sm)} anchors (fails on {sm[sm.command_routegen.str.startswith('ERR')].uuid.nunique()} clips); same command on {int((okrg.command == okrg.command_routegen).sum())}/{len(okrg)}.")
cr = sm[(sm.type == "A") & (sm.speed >= 20)]; W(f"Acceleration slot (backward difference of 10 Hz velocity): all anchors ax p1/p50/p99 {sm.ax.quantile(.01):.2f}/{sm.ax.median():.2f}/{sm.ax.quantile(.99):.2f} m/s², |ax| > 4 on {int((sm.ax.abs() > 4).sum())} anchors; ay p1/p99 {sm.ay.quantile(.01):.2f}/{sm.ay.quantile(.99):.2f}."); W()
# ---------- C1
W("### C1 — bit identity"); W()
rng = random.Random(20261004); pick = rng.sample(range(len(sm)), 5); rows = []; cache = {}
def clipdata(u):
    if u not in cache:
        ch = int(clips.set_index("uuid").loc[u, "chunk"]); d, ts, ego, ft, R_N, t_N = S.load_clip(u, ch); cache.clear(); cache[u] = (d, ts, ego, S.build_maps(ft, R_N), t_N)
    return cache[u]
for i in pick:
    r = sm.iloc[i]; d, ts, ego, maps, t_N = clipdata(r.uuid)
    fr = S.decode(d / f"{r.uuid}.{C.FRONT}.mp4", [int(r.frame_older), int(r.frame_newer)])
    new = P._pipeline_a(cv2.cvtColor(S.rectify(fr[int(r.frame_newer)], maps), cv2.COLOR_BGR2RGB)); old = P._pipeline_a(cv2.cvtColor(S.rectify(fr[int(r.frame_older)], maps), cv2.COLOR_BGR2RGB))
    with gzip.open(S.CACHE / r.uuid / r.token / "pad_feature.gz", "rb") as f: feat = pickle.load(f)
    d1 = float(np.abs(feat["camera_feature_1"].numpy() - new).max()); d2 = float(np.abs(feat["camera_feature_2"].numpy() - old).max())
    rows.append(dict(token=r.token, frame_newer=int(r.frame_newer), frame_older=int(r.frame_older), max_abs_diff_cf1=d1, max_abs_diff_cf2=d2, shape=str(tuple(feat["camera_feature_1"].shape)), dtype=str(feat["camera_feature_1"].dtype)))
c1 = pd.DataFrame(rows); tbl(c1); C1 = float(max(c1.max_abs_diff_cf1.max(), c1.max_abs_diff_cf2.max())); W(f"C1 max abs difference over 5 samples (stored sample vs fresh decode → rectify → adapter `_pipeline_a`): **{C1}**."); W()
# ---------- C2
W("### C2 — geometry PNGs"); W()
DIST = S.DIST_V; KV = S.K_V; RV = S.R_V
def project(pts_rig, t_cam):
    pc = (RV.T @ (pts_rig - t_cam).T).T; ok = pc[:, 2] > 0.5
    uv = np.full((len(pc), 2), np.nan)
    if ok.any():
        n = pc[ok, :2] / pc[ok, 2:3]; r2 = (n ** 2).sum(1); ok2 = r2 < 0.9 ** 2      # outside the distortion model's monotonic range -> treat as outside
        uvk = cv2.projectPoints(pc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), KV, DIST)[0].reshape(-1, 2); uvk[~ok2] = np.nan; uv[ok] = uvk
    return uv
def inside(uv): return np.isfinite(uv[:, 0]) & (uv[:, 0] >= 0) & (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] < 1080)
def inside_crop(uv): return inside(uv) & (uv[:, 1] >= K.CROP_ROWS) & (uv[:, 1] < 1080 - K.CROP_ROWS)
mo = pd.read_csv(C.S0 / "r3_raw_moveoffs.csv"); mo = mo[(mo.actor_bum == True) & (mo.dyaw4_deg.abs() <= 45)].drop_duplicates("uuid").set_index("uuid")
(HERE / "c2_png").mkdir(exist_ok=True); rows = []; tot = dict(n=0, out=0, out_crop=0)
for r in clips.itertuples():
    d, ts, ego, maps, t_N = clipdata(r.uuid); s = sm[sm.uuid == r.uuid].reset_index(drop=True)
    for q in s.itertuples():      # all anchors: count only
        with gzip.open(S.CACHE / r.uuid / q.token / "pad_target.gz", "rb") as f: lab = pickle.load(f)["trajectory"].numpy()
        uv = project(np.c_[lab[:, :2], np.zeros(8)], t_N); tot["n"] += 8; tot["out"] += int((~inside(uv)).sum()); tot["out_crop"] += int((~inside_crop(uv)).sum())
    t_pick = (int(np.ceil(mo.loc[r.uuid, "t_depart"] * 2) / 2 * 1e6) if r.type == "B" else 8_000_000); q = s.iloc[(s.t_us - t_pick).abs().argmin()]
    with gzip.open(S.CACHE / r.uuid / q.token / "pad_target.gz", "rb") as f: lab = pickle.load(f)["trajectory"].numpy()
    img = cv2.cvtColor(S.rectify(S.decode(d / f"{r.uuid}.{C.FRONT}.mp4", [int(q.frame_newer)])[int(q.frame_newer)], maps), cv2.COLOR_BGR2RGB)
    pts = np.c_[lab[:, :2], np.zeros(8)]; uvN = project(pts, t_N); uvV = project(pts, S.t_V)
    small = cv2.resize(img[K.CROP_ROWS:-K.CROP_ROWS], K.RESIZE_WH, interpolation=cv2.INTER_LINEAR); sx = 512 / 1920; sy = 256 / (1080 - 2 * K.CROP_ROWS)
    fig, ax = plt.subplots(1, 2, figsize=(19, 5.6), gridspec_kw=dict(width_ratios=[1920 / 1080, 2]))
    ax[0].imshow(img); ax[0].plot(uvN[:, 0], uvN[:, 1], "o-", c="lime", ms=5, lw=1.5, label="label-1, real camera position"); ax[0].plot(uvV[:, 0], uvV[:, 1], "x--", c="magenta", ms=5, lw=1, label="label-1, declared camera position")
    ax[0].set_xlim(0, 1920); ax[0].set_ylim(1080, 0); ax[0].legend(loc="upper right", fontsize=8); ax[0].set_title(f"{r.type} {r.uuid[:8]} t={q.t_us/1e6:.1f}s v={q.speed:.1f} m/s cmd={q.command} — rectified 1920×1080", fontsize=9)
    ax[1].imshow(small); ax[1].plot(uvN[:, 0] * sx, (uvN[:, 1] - K.CROP_ROWS) * sy, "o-", c="lime", ms=4, lw=1.2); ax[1].set_xlim(0, 512); ax[1].set_ylim(256, 0); ax[1].set_title("model input 512×256 (camera_feature_1)", fontsize=9)
    for a_ in ax: a_.axis("off")
    fn = HERE / "c2_png" / f"{r.type}_{r.uuid[:8]}_t{q.t_us//100000:03d}.png"; fig.tight_layout(); fig.savefig(fn, dpi=80); plt.close(fig)
    rows.append(dict(type=r.type, uuid=r.uuid[:8], t_s=q.t_us / 1e6, speed=round(q.speed, 1), command=q.command, label_x8=round(float(lab[-1, 0]), 1), label_y8=round(float(lab[-1, 1]), 1), pts_outside_1080p=int((~inside(uvN)).sum()), pts_outside_input=int((~inside_crop(uvN)).sum()), png=f"c2_png/{fn.name}"))
tbl(pd.DataFrame(rows)); W(f"All {len(sm)} anchors: {tot['out']}/{tot['n']} projected label points fall outside the 1920×1080 rectified image, {tot['out_crop']}/{tot['n']} outside the cropped model input (projection with the real camera position, ground at rig z = 0)."); W()
# ---------- C3
W("### C3 — menu ceiling (A clips, human speed ≥ 20 m/s)"); W()
a = sm[(sm.type == "A") & (sm.speed >= 20.0)]; ok = (a.fastest_end <= 18.8)
tbl(a.groupby("uuid").agg(anchors=("token", "size"), human_speed_med=("speed", "median"), fastest_med=("fastest_end", "median"), fastest_max=("fastest_end", "max"), chosen_step1_med=("chosen_step1", "median"), chosen_end_med=("chosen_end", "median"), n_over_18_8=("fastest_end", lambda x: int((x > 18.8).sum()))).reset_index().assign(uuid=lambda d: d.uuid.str[:8]))
C3 = float(ok.mean()) if len(a) else float("nan"); W(f"C3: fastest-of-32 end speed ≤ 18.8 m/s on {int(ok.sum())}/{len(a)} anchors = **{100*C3:.1f} %** (prediction ≥ 90 %). Pool: fastest median {a.fastest_end.median():.2f}, max {a.fastest_end.max():.2f} m/s; human median {a.speed.median():.1f} m/s."); W()
# ---------- C4
W("### C4 — city speed match (ordinary clips, human speed 5–15 m/s)"); W()
o = sm[(sm.type == "O") & (sm.speed >= 5) & (sm.speed <= 15)].copy(); o["ratio"] = o.chosen_step1 / o.speed
tbl(o.groupby("uuid").agg(anchors=("token", "size"), human_speed_med=("speed", "median"), ratio_med=("ratio", "median"), ratio_min=("ratio", "min"), ratio_max=("ratio", "max")).reset_index().assign(uuid=lambda d: d.uuid.str[:8]))
C4 = float(o.ratio.median()); W(f"C4: median(chosen-plan first-step speed / human speed) = **{C4:.3f}** over {len(o)} anchors in {o.uuid.nunique()} clips (prediction in [0.8, 1.2]); IQR {o.ratio.quantile(.25):.2f}–{o.ratio.quantile(.75):.2f}; share below 0.8: {100*(o.ratio < 0.8).mean():.0f} %."); W()
# ---------- C5
W("### C5 — pull-outs at standstill (descriptive)"); W()
b = sm[sm.type == "B"].copy(); b["t_move"] = b.uuid.map(mo.t_move); b["t_depart"] = b.uuid.map(mo.t_depart); b["t"] = b.t_us / 1e6
st = b[(b.speed < 0.3) & (b.t <= b.t_depart)]
tbl(st[["uuid", "t", "t_move", "t_depart", "command", "n_lat1", "n_lat1_moving", "max_prop_abs_y", "max_prop_len", "chosen_len", "chosen_max_abs_y", "label_len", "label_y8"]].assign(uuid=lambda d: d.uuid.str[:8]).round(2))
W(f"C5: standstill anchors before departure {len(st)} in {st.uuid.nunique()} clips; with ≥ 1 proposal reaching ≥ 1 m sideways: {int((st.n_lat1 > 0).sum())}; with ≥ 1 such proposal that also travels ≥ 2 m: {int((st.n_lat1_moving > 0).sum())}; chosen plan moves ≥ 2 m on {int((st.chosen_len >= 2).sum())}; human label moves ≥ 2 m within 4 s on {int((st.label_len >= 2).sum())}."); W()
verdict = "CONVERTER-OK" if (C1 == 0 and C3 >= 0.9 and 0.8 <= C4 <= 1.2) else "NOT OK: " + ", ".join(n for n, f in [("C1", C1 != 0), ("C3", not C3 >= 0.9), ("C4", not 0.8 <= C4 <= 1.2)] if f)
W(f"VERDICT: {verdict}"); json.dump(dict(C1=C1, C3=C3, C4=C4, verdict=verdict), open(HERE / "checks.json", "w"))
open(HERE / "checks_tables.md", "w").write("\n".join(out))
