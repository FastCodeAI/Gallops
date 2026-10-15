#!/usr/bin/env python
"""Step 0 readout. Read-only on the runs. For each of the two scenes reads the oracle run (NAS 13_best_possible_pick/runs) and the 0.1.7
trial run (NAS 07_closed_loop_test_017/runs) with the same code and writes tables/facts.txt: ONE fact per line, `scene | key | value | source`,
so every number in the report can be cited as tables/facts.txt:<line>. Definitions are PREREG.md's.

  alpasim/.venv/bin/python -B oracle_readout.py            (scenes without a finished oracle run are skipped and listed)
"""
from __future__ import annotations
import glob, json, math, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import extract_actors as EA  # noqa: E402
NAS = pathlib.Path("{runs}/drivejepa_retrain"); ORC = NAS / "13_best_possible_pick" / "runs"; T17 = NAS / "07_closed_loop_test_017" / "runs"; SCENES = ["d2715023", "b44a4a44"]; T = HERE / "tables"


def drive(run):
    """ego track from the run's own log + score and scored window from its results summary."""
    o = EA.load(run); e = np.array(o["ego"]); ho = o["render_start_us"] + o["force_gt_us"]; t = (e[:, 0] - ho) / 1e6; i0 = int(np.argmin(np.abs(t))); x0, y0, yaw0 = e[i0, 1:4]
    dx, dy = e[:, 1] - x0, e[:, 2] - y0; c, s = math.cos(yaw0), math.sin(yaw0); fwd = c * dx + s * dy; lat = -s * dx + c * dy; dist = np.hypot(dx, dy); v = np.hypot(e[:, 4], e[:, 5])
    rs = json.load(open(pathlib.Path(run) / "aggregate" / "results-summary.json"))["rollouts"][0]; win = float(rs["metrics"]["duration_frac_20s"]) * 20.0; w = (t >= 0) & (t <= win + 1e-6); a = t >= 0
    def first(mask): return float(t[np.argmax(mask)]) if mask.any() else float("nan")
    path = float(np.sum(np.hypot(np.diff(e[a, 1]), np.diff(e[a, 2]))))
    return dict(t=t, fwd=fwd, lat=lat, dist=dist, v=v, ho=ho, egodriver=o["egodriver"], asl=o["asl"], summary=str(pathlib.Path(run) / "aggregate" / "results-summary.json"),
                score=float(rs["score"]), status=rs["status"], failure=rs.get("failure_reason"), window_s=win, progress=float(rs["metrics"]["progress_clipped_rel"]), dist_traveled=float(rs["metrics"]["dist_traveled_m"]),
                at_fault=float(rs["metrics"]["collision_at_fault"]), any_coll=float(rs["metrics"]["collision_any"]), offroad=float(rs["metrics"]["offroad"]),
                max_dist_win=float(dist[w].max()), max_abs_lat_win=float(np.abs(lat[w]).max()), max_fwd_win=float(fwd[w].max()), max_dist_all=float(dist[a].max()), max_abs_lat_all=float(np.abs(lat[a]).max()), end_fwd=float(fwd[-1]), end_lat=float(lat[-1]),
                t_first_4m=first(a & (dist > 4.0)), t_first_lat_2m=first(a & (np.abs(lat) > 2.0)), t_first_1ms=first(a & (v > 1.0)), max_speed=float(v[a].max()), path_len=path, t_end=float(t[-1]))


def main():
    F = []; out = {}; skipped = []
    def fact(sc, key, val, src): F.append(f"{sc} | {key} | {val if isinstance(val, str) else (f'{val:.4g}' if isinstance(val, float) else val)} | {src}")
    for sc in SCENES:
        run = ORC / f"drivejepa_017oracle_{sc}"
        if not (run / "aggregate" / "results-summary.json").is_file(): skipped.append(sc); continue
        O = drive(run); B = drive(T17 / f"drivejepa_017_{sc}"); steps = [json.loads(l) for l in open(run / "oracle_steps.jsonl")]; slog = str(run / "oracle_steps.jsonl")
        for arm, D in (("oracle", O), ("0.1.7", B)):
            for k in ("egodriver", "score", "status", "failure", "window_s", "progress", "dist_traveled", "at_fault", "any_coll", "offroad"): fact(sc, f"{arm} {k}", D[k] if D[k] is not None else "none", D["summary"] if k != "egodriver" else D["asl"])
            for k in ("max_dist_win", "max_abs_lat_win", "max_fwd_win", "max_dist_all", "max_abs_lat_all", "end_fwd", "end_lat", "t_first_4m", "t_first_lat_2m", "t_first_1ms", "max_speed", "path_len", "t_end"): fact(sc, f"{arm} {k}", D[k], D["asl"] + " (controller_request state, oracle_readout.py drive())")
        ho = O["ho"]; st = np.array([(s["time_now_us"] - ho) / 1e6 for s in steps]); post = st >= -1e-6; inwin = post & (st <= O["window_s"] + 1e-6)
        wide = np.array([(not s["fallback"]) and abs(s["picked_wp8_lat"]) > 1.5 for s in steps]); fb = np.array([s["fallback"] for s in steps]); pk = np.array([s["picked"] for s in steps]); sp = np.array([s["scorer_pick"] for s in steps])
        fact(sc, "steps logged", len(steps), slog); fact(sc, "steps from handover", int(post.sum()), slog); fact(sc, "steps in scored window", int(inwin.sum()), slog)
        fact(sc, "P0c wide picks from handover", int(wide[post].sum()), slog); fact(sc, "P0c share from handover", float(wide[post].mean()), slog); fact(sc, "P0c wide picks in window", int(wide[inwin].sum()), slog); fact(sc, "P0c share in window", float(wide[inwin].mean()) if inwin.any() else float("nan"), slog)
        full = post & np.array([s["future_frames"] == 40 for s in steps]); fact(sc, "P0c wide picks / steps, steps from handover whose 4 s lie inside the actor log", f"{int(wide[full].sum())} / {int(full.sum())}", slog)
        fact(sc, "fallback steps from handover", int(fb[post].sum()), slog); fact(sc, "steps where oracle pick = scorer pick, from handover", int((pk == sp)[post].sum()), slog)
        fact(sc, "median collision-free plans per menu, from handover", float(np.median([s["n_collision_free"] for s, p in zip(steps, post) if p])), slog)
        fact(sc, "median picked grade (m progress), from handover", float(np.median([max(s["grade"]) for s, p in zip(steps, post) if p])), slog)
        fact(sc, "median |lat| wp8 of the picked plan, from handover", float(np.median([abs(s["picked_wp8_lat"]) for s, p in zip(steps, post) if p])), slog)
        fact(sc, "median wp8 distance of the picked plan, from handover", float(np.median([math.hypot(s["picked_wp8_long"], s["picked_wp8_lat"]) for s, p in zip(steps, post) if p])), slog)
        # the widest collision-free plan on each menu: is a wide pull-out there at all, and collision-free?
        wf = [max([abs(s["proposals"][j][7][1]) for j in range(32) if s["nc_strict"][j] == 1.0] or [0.0]) for s in steps]
        fact(sc, "menus from handover with a collision-free plan wider than 1.5 m at wp8", int(sum(w > 1.5 for w, p in zip(wf, post) if p)), slog)
        for n in (1, 20, 40):
            k = min(n, len(steps)) - 1; s = steps[k]; fact(sc, f"step {s['step']} (asked {n}) widest |lat| at wp8 over 32 plans / car speed / t after handover", f"{s['menu_widest_abs_lat_wp8']:.3f} m / {s['car_speed']:.3f} m/s / {st[k]:+.2f} s", f"{slog}:{k + 1}")
        still = [s for s in steps if s["car_speed"] < 0.1]
        if still: fact(sc, "widest |lat| at wp8 on menus with the car under 0.1 m/s: min / median / max over steps", f"{min(s['menu_widest_abs_lat_wp8'] for s in still):.2f} / {np.median([s['menu_widest_abs_lat_wp8'] for s in still]):.2f} / {max(s['menu_widest_abs_lat_wp8'] for s in still):.2f} m (n={len(still)})", slog)
        # controller following: speed the picked plan asks over its first 0.5 s vs the car's speed 0.5 s later
        ask = np.array([math.hypot(s["picked_wp1_long"], s["picked_wp1_lat"]) / 0.5 for s in steps]); got = np.array([steps[i + 1]["car_speed"] if i + 1 < len(steps) else np.nan for i in range(len(steps))]); m = post & np.isfinite(got)
        fact(sc, "mean speed asked by the picked plan's first 0.5 s / mean car speed at the next replan, from handover", f"{ask[m].mean():.3f} / {got[m].mean():.3f} m/s", slog)
        i0 = int(np.argmax(post)); ex = []
        for k in range(i0, min(i0 + 10, len(steps))):
            s = steps[k]; g = np.array(s["grade"]); ex.append(f"step {s['step']} t {st[k]:+.2f}s | car ({s['car_x']:.2f}, {s['car_y']:.2f}) {s['car_speed']:.2f} m/s | picked slot {s['picked']}{' FALLBACK' if s['fallback'] else ''} grade {g.max():.2f} m, {s['n_collision_free']}/32 collision-free | wp1 long/lat {s['picked_wp1_long']:+.2f}/{s['picked_wp1_lat']:+.2f} m | wp8 long/lat {s['picked_wp8_long']:+.2f}/{s['picked_wp8_lat']:+.2f} m | scorer slot {s['scorer_pick']} (wp8 {s['scorer_pick_wp8_long']:+.2f}/{s['scorer_pick_wp8_lat']:+.2f}) | [{slog}:{k + 1}]")
        g = json.load(open(run / "ext20_gates.json")); fact(sc, "run gates ok", str(g.get("ok")), str(run / "ext20_gates.json")); fact(sc, "run gates failed", ";".join(k for k, v in g["gates"].items() if not v["ok"]) or "none", str(run / "ext20_gates.json"))
        out[sc] = dict(oracle={k: v for k, v in O.items() if not isinstance(v, np.ndarray)}, trial_017={k: v for k, v in B.items() if not isinstance(v, np.ndarray)}, excerpt=ex,
                       P0a=bool(O["max_dist_win"] > 4.0) if sc == "d2715023" else None, P0b=bool(O["max_abs_lat_win"] > 2.0) if sc == "b44a4a44" else None, p0c_wide=int(wide[post].sum()), p0c_n=int(post.sum()))
        (T / f"excerpt_{sc}.txt").write_text("\n".join(ex) + "\n")
    if len(out) == len(SCENES):
        n = sum(out[s]["p0c_n"] for s in SCENES); w = sum(out[s]["p0c_wide"] for s in SCENES); fact("pooled", "P0c wide picks / steps from handover / share", f"{w} / {n} / {w / n:.3f}", "both step logs")
        fact("d2715023", "P0a (moves > 4 m in the scored window)", str(out["d2715023"]["P0a"]), "max_dist_win above"); fact("b44a4a44", "P0b (lateral > 2 m in the scored window)", str(out["b44a4a44"]["P0b"]), "max_abs_lat_win above"); fact("pooled", "P0c (> 50 %)", str(w / n > 0.5), "line above")
    (T / "facts.txt").write_text("\n".join(F) + "\n"); json.dump(dict(scenes=out, skipped=skipped), open(T / "readout.json", "w"), indent=1, default=float)
    for i, l in enumerate(F, 1): print(f"{i:3d}  {l}")
    for sc in out: print(f"--- excerpt {sc}"); print("\n".join(out[sc]["excerpt"]))
    print("skipped (no finished oracle run):", skipped)


if __name__ == "__main__":
    main()
