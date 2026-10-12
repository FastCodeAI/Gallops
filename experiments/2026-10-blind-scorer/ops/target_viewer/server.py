"""Target viewer: every tick of scorer cache v2, every plan's trajectory, sub-score targets and held-out predictions.

    taskset -c 0-1 python server.py --port 8932

Data is read-only: cache v2 ({dj_cache}/v2: plans, labels), the rendered episodes (frames,
poses, actors, lane references), the drivable rasters, and the 5-seed probe+geometry fold logits (each
tick's prediction comes from the fold where its scene was held out).
"""
import argparse, base64, json, sys, threading
from functools import lru_cache
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np

sys.path.insert(0, "{gallops}/src")
from gallops.expert.drivable import DrivableArea
from gallops.expert.grader import actor_at, corners
from gallops.expert.lanes import load_references, reference_at

CACHES = [Path("{dj_cache}/v2"), Path("{dj_cache}/v3_nusc")]
R = Path("{runs}")
CALIB = Path("{paddock_runs}/ops/ncap_viewer/work")
NUSC = Path("{data.nuscenes}")
HERE = Path(__file__).parent
FRAME_ROOTS = [R / "collect", R / "collect_gt", R / "collect_round1-s2", R / "collect_nusc", NUSC / "samples"]
SCENES = ("0053", "0058", "0064", "0070", "0075")
#: which 5-seed fold runs supply the predictions shown ("attngeo" = rendered-only training)
PRED_TAG = __import__("os").environ.get("VIEWER_PRED_TAG", "attngeo")
sig = lambda z: 1 / (1 + np.exp(-z))

print("loading index and predictions...", flush=True)
CACHE_OF, FIRST, COUNT, PLANS, PRED = {}, {}, {}, {}, {}
for cache in CACHES:
    red = cache / "probe_input.npz"
    if not red.exists():
        print(f"  {red} missing; {cache.name} not served", flush=True)
        continue
    P = np.load(red)
    eps = P["episode"]
    PLANS[cache] = P["plans"]
    for i, e in enumerate(eps):
        e = str(e); FIRST.setdefault(e, (cache, i)); COUNT[e] = COUNT.get(e, 0) + 1; CACHE_OF[e.split("/")[0]] = cache
    pred = np.full((len(eps), 64, 6), np.nan, dtype=np.float32)
    v2 = cache.name == "v2"
    for name in (SCENES if v2 else ("nusc_holdout",)):
        files = [cache.parent / "v2" / f"probe_{PRED_TAG}{'' if v2 else ''}_s{s}.logits.npz" for s in range(5)]
        if not v2:
            files = [Path("{dj_cache}/v2") / f"probe_{PRED_TAG}_s{s}.logits.npz" for s in range(5)]
        if not all(f.exists() for f in files):
            continue
        L = [np.load(f) for f in files]
        if f"test_idx_{name}" not in L[0]:
            continue
        te = L[0][f"test_idx_{name}"]
        off = 0 if v2 else -len(np.load(CACHES[0] / "probe_input.npz", mmap_mode="r")["scene"])
        pred[te + off] = np.mean([sig(l[f"logits_{name}"]) for l in L], 0)
    PRED[cache] = pred
print(f"ready: {len(COUNT)} episodes, predictions from {PRED_TAG}", flush=True)


def targets(lab):
    nc = np.isnan(lab["hit5_at"]).astype(np.float32); dac = np.isnan(lab["off_road_at"]).astype(np.float32)
    ttc = np.isnan(lab["ttc_at"]).astype(np.float32); lk = (lab["max_deviation"] <= 1.75).astype(np.float32)
    ep = np.clip(lab["travel"] / np.maximum(lab["human_travel"][:, None], 1e-3), 0, 1).astype(np.float32)
    return np.stack([nc, dac, ttc, lk, ep, nc * dac * (5 * ep + 5 * ttc + 2 * lk) / 12], -1)


def formula(s):
    return s[..., 0] ** 4 * s[..., 1] * (5 * s[..., 4] + 5 * s[..., 2] + 2 * s[..., 3]) / 12


def r2(a, nd=2):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def nan2none(a):
    a = np.asarray(a, dtype=float)
    return [None if np.isnan(x) else round(float(x), 2) for x in a.ravel()]


@lru_cache(maxsize=8)
def area(scene):
    f = R / "drivable" / f"scene-{scene}.npz"
    return DrivableArea.load(f if f.exists() else R / "drivable_nusc" / f"scene-{scene}.npz")


@lru_cache(maxsize=8)
def calib(scene):
    f = CALIB / f"calib_scene-{scene}.json"
    if f.exists():
        c = json.loads(f.read_text()); cam2ego, K = np.asarray(c["cam2ego"]), np.asarray(c["K"])
    else:  # real scenes: the per-scene calibration nuscenes_real precomputed
        c = json.loads((R / "calib_nusc" / f"scene-{scene}.json").read_text())["CAM_FRONT"]
        cam2ego, K = np.asarray(c["pose"]), np.asarray(c["intrinsics"])
    ego2cam = np.linalg.inv(cam2ego); cam2img = np.eye(4); cam2img[:3, :3] = K
    return {"ego2cam": ego2cam.tolist(), "ego2img": (cam2img @ ego2cam).tolist()}


def episode_list():
    out = []
    for src in sorted(p for c in CACHES if c in PLANS for p in c.iterdir() if p.is_dir() and not p.name.startswith("__")):
        for f in sorted(src.glob("*.npz")):
            if f.name.endswith(".labels.npz") or ".tmp" in f.name: continue
            name = f"{src.name}/{f.stem}"
            if name not in FIRST: continue
            shape_scene, run = f.stem.split("__"); shape, scene = shape_scene.split("-")
            out.append({"name": name, "source": src.name, "shape": shape, "scene": scene, "run": run, "ticks": COUNT[name]})
    return out


GRID_X = np.arange(-10, 90, 0.5); GRID_Y = np.arange(-30, 30, 0.5)


@lru_cache(maxsize=24)
def episode(name):
    src, stem = name.split("/")
    cache = CACHE_OF[src]
    z = np.load(cache / src / f"{stem}.npz"); lab = np.load(cache / src / f"{stem}.labels.npz")
    ep = Path(str(z["frame_dir"])); scene = stem.split("__")[0].split("-")[1]
    poses = json.loads((ep / "ego_poses.json").read_text())
    actors = json.loads((ep / "actors.json").read_text())
    vehicles = [a for a in actors if a["dims"][1] >= 1.2 and len(a["timestamps"]) >= 2]
    logged, lanes = load_references(ep); ar = area(scene)
    _, g0 = FIRST[name]; n = len(z["timestamp"]); assert COUNT[name] == n
    labs = {k: lab[k][:, :64] for k in ("hit5_at", "hit_at", "hit_target_at", "off_road_at", "ttc_at", "max_deviation", "travel", "off_path_at")}
    labs["human_travel"] = lab["human_travel"]
    tg = targets(labs); preds = PRED[cache][g0:g0 + n]
    assert np.allclose(np.concatenate([z["natives"], z["vocab"][:, :32]], 1), PLANS[cache][g0:g0 + n]), "cache/training index mismatch"
    meta = [json.loads(m) for m in z["vocab_meta"]]
    kinds = [f"native {k}" for k in range(32)] + [f"{m['base']} {m['offset_m']:+.1f} m" for m in meta[:32]]
    ticks = []
    for i, ts in enumerate(z["timestamp"]):
        ts = int(ts); e2w = np.asarray(poses[str(ts)]); rot, org = e2w[:2, :2], e2w[:2, 3]
        toe = lambda w: ((np.asarray(w)[..., :2] - org) @ rot)
        rel = int(z["pdm_score"][i].argmax()); s = preds[i]
        if np.isnan(s).any():  # trained on (no held-out prediction): show the released pick only
            fs = np.full(64, np.nan); veto = False; pick = rel
        else:
            fs = formula(s); alt = int(fs.argmax()); veto = bool(s[rel, 0] < 0.5); pick = alt if veto else rel
        boxes = []
        for a in vehicles:
            st = actor_at(a, float(ts))
            if st is None or np.linalg.norm(st[0] - org) > 100: continue
            item = {"t": bool(a.get("is_target_actor")), "box": r2(toe(corners(st[0], st[1], a["dims"][1], a["dims"][0])))}
            if item["t"]:
                fut = []
                for k in (1, 2, 3, 4, 5):
                    later = actor_at(a, float(ts) + k * 1e6)
                    if later is not None: fut.append({"k": k, "box": r2(toe(corners(later[0], later[1], a["dims"][1], a["dims"][0])))})
                item["future"] = fut
            boxes.append(item)
        ref, kind = reference_at(ts, logged, lanes); re = toe(ref); re = re[(np.abs(re) < 120).all(1)]
        gx, gy = np.meshgrid(GRID_X, GRID_Y, indexing="ij")
        mask = ar.contains(np.stack([gx.ravel(), gy.ravel()], 1) @ rot.T + org)
        ticks.append({
            "ts": ts, "t": round((ts - int(z["timestamp"][0])) / 1e6, 2), "speed": round(float(z["speed"][i]), 2),
            "in_frame": bool(z["in_frame"][i]), "usable": bool(z["usable"][i]), "rel": rel, "pick": pick, "veto": veto,
            "best": int(tg[i, :, 5].argmax()), "human_travel": round(float(labs["human_travel"][i]), 1),
            "frame": f"/frame?p={ep}/CAM_FRONT/{ts}.jpg",
            "plans": r2(np.concatenate([z["natives"][i], z["vocab"][i][:32]])[..., :2]), "probes": r2(z["vocab"][i][32:, :, :2], 1),
            "hit5": nan2none(labs["hit5_at"][i]), "hit_target": nan2none(labs["hit_target_at"][i]), "off_road": nan2none(labs["off_road_at"][i]),
            "ttc": nan2none(labs["ttc_at"][i]), "dev": r2(labs["max_deviation"][i]), "travel": r2(labs["travel"][i], 1),
            "target": r2(tg[i], 3), "pred": [nan2none(row) for row in s], "formula": nan2none(fs), "pdm": r2(z["pdm_score"][i][:32], 3),
            "actors": boxes, "lane": r2(re, 1), "lane_kind": kind,
            "drivable": base64.b64encode(np.packbits(mask.astype(np.uint8))).decode(),
        })
    fold = scene if cache.name == "v2" else ("held-out real scene" if src == "nusc_holdout" else "trained on")
    return json.dumps({"name": name, "scene": scene, "fold": fold, "kinds": kinds, "calib": calib(scene),
                       "grid": {"x0": float(GRID_X[0]), "y0": float(GRID_Y[0]), "step": 0.5, "nx": len(GRID_X), "ny": len(GRID_Y)},
                       "ticks": ticks}).encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, body, ctype, code=200):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if ctype.startswith("image/") else "no-store"); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                return self.send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/episodes":
                return self.send(json.dumps(episode_list()).encode(), "application/json")
            if u.path == "/api/episode":
                return self.send(episode(q["name"][0]), "application/json")
            if u.path == "/frame":
                p = Path(q["p"][0]).resolve()
                if not any(str(p).startswith(str(r.resolve()) + "/") for r in FRAME_ROOTS) or p.suffix != ".jpg":
                    return self.send(b"forbidden", "text/plain", 403)
                return self.send(p.read_bytes(), "image/jpeg")
            return self.send(b"not found", "text/plain", 404)
        except Exception as exc:  # report, keep serving
            return self.send(f"{type(exc).__name__}: {exc}".encode(), "text/plain", 500)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8932); a = ap.parse_args()
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()
