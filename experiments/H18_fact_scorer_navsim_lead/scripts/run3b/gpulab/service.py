"""run 3B live labels on the GPU (gpulab, CONTRACT.md): the same interface as r3b/labeller.Labeller + train2's slot methods.

  prep workers (CPU, spawned): the CPU labeller's own context (labeller._nav_env / _carla_env context(meta)), then the three
      pose-independent extracts (rollout.extract_rollout, mapfacts.extract_map, agentfacts.extract_agents) -> scene dicts
  server (one spawned process, owns the GPU work): routes prep jobs to the workers, keeps the scenes, and labels each request
      once all its scenes are in: rollout.simulate + ego_coords -> mapfacts.label_map -> agentfacts.label_agents, per kind,
      the request's samples batched; results in the CPU labeller's dtypes and keys (+ "seconds")
Each part was checked against the CPU labeller (rollout / mapfacts / agentfacts verify_*.json); verify_service.py checks the
whole service against labeller.Labeller end to end."""
import os
import queue
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R3B = "{runs}/run3b/r3b"
KEYS = ("hit", "offroad", "left_route", "red", "stop", "wrong_way", "metres", "lane_off", "ttc", "cutin_brake")


def _paths():
    for p in (str(HERE / "agentfacts"), str(HERE / "mapfacts"), str(HERE / "rollout"), R3B):
        if p not in sys.path:
            sys.path.insert(0, p)


_DIRT = {}


def _dirt_all(zrel):
    """LEAD actor ids whose type_id starts with static.prop.dirtdebris, over the whole route (carla_full_v1's own DEBRIS list
    has dirtdebris01 only; this widens it for the live hit rule and the key-agent observation, both of which read ctx['deb'])."""
    if zrel not in _DIRT:
        import label_carla as LC
        rt = LC.route(zrel)
        _DIRT[zrel] = frozenset(str(b["id"]) for bl in rt.boxes for b in bl
                                if str(b.get("type_id", "")).startswith("static.prop.dirtdebris"))
        if len(_DIRT) > 64:
            _DIRT.pop(next(iter(_DIRT)))
    return _DIRT[zrel]


def prep_main(kind, prep_q, scene_q):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        _paths()
        import labeller as LB
        context, _ = LB._nav_env() if kind == "nav" else LB._carla_env()
        _paths()  # _nav_env / _carla_env put their own paths first; ours go back in front
        import agentfacts as A
        import mapfacts as M
        import rollout as R
    except Exception:  # noqa: BLE001
        scene_q.put(("fatal", kind, traceback.format_exc()))
        return
    scene_q.put(("ready", kind, os.getpid()))
    while True:
        job = prep_q.get()
        if job is None or job[0] == "stop":
            return
        _, key, m = job
        try:
            t0 = time.process_time()
            ctx = context(m)
            if kind == "carla":  # Navvrat 19 Nov: every static.prop.dirtdebris* is debris (not a hit, not a key agent)
                ctx["deb"] = _dirt_all(m["zrel"])
            t1 = time.process_time()
            sc = dict(kind=kind, rollout=R.extract_rollout(kind, ctx), map=M.extract_map(kind, ctx),
                      agents=A.extract_agents(kind, ctx, meta=m, aux=True))
            sc["seconds"] = dict(context=t1 - t0, extract=time.process_time() - t1)
            scene_q.put(("scene", key, sc))
        except Exception:  # noqa: BLE001
            scene_q.put(("error", key, traceback.format_exc()[-2000:]))


def _label_kind(kind, scenes, poses32, R, M, A, torch):
    """scenes: list of scene dicts of one kind; poses32 (n, P, 8, 3) float32 -> list of per-sample dicts (numpy)."""
    poses = torch.as_tensor(np.ascontiguousarray(poses32), device="cuda").to(torch.float64)
    states = R.simulate(kind, [s["rollout"] for s in scenes], poses)
    coords = R.ego_coords(kind, states)
    mo = M.label_map(kind, [s["map"] for s in scenes], states, coords, poses=poses)  # poses for both kinds: aux on LEAD too (19 Nov)
    ao = A.label_agents(kind, [s["agents"] for s in scenes], states, coords, mo["ego_areas"])
    mn = M.as_labeller_dtypes(mo)
    an = {k: ao[k].cpu().numpy() for k in ("hit", "ttc", "cutin_brake", "comfort_step")}
    has_aux = "key_corners" in ao and "aux" in mn  # NAVSIM always; LEAD once mapfacts returns its aux (19 Nov)
    if has_aux:
        for k in ("key_corners", "key_valid", "key_mask"):
            if k in ao:
                an[k] = ao[k].cpu().numpy()
    outs = []
    for i in range(len(scenes)):
        o = {k: mn[k][i] for k in ("offroad", "wrong_way", "left_route", "stop", "red", "metres", "lane_off")}
        o.update(hit=an["hit"][i], ttc=an["ttc"][i], cutin_brake=an["cutin_brake"][i], comfort_step=an["comfort_step"][i])
        if has_aux:
            o["aux"] = dict(key_corners=an["key_corners"][i], key_valid=an["key_valid"][i],
                            ego_areas=mn["aux"]["ego_areas"][i], map_aux=mn["aux"]["map_aux"][i])
            if "key_mask" in an:
                o["aux"]["key_mask"] = an["key_mask"][i]
        outs.append(o)
    return outs


def server_main(req_q, scene_q, out_q, worker_qs, mem_gb):
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    try:
        _paths()
        import torch
        torch.cuda.set_per_process_memory_fraction(mem_gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory)
        import agentfacts as A
        import mapfacts as M
        import rollout as R
    except Exception:  # noqa: BLE001
        out_q.put(("fatal", "server", traceback.format_exc()))
        return
    out_q.put(("ready", "server", os.getpid()))
    scenes, asked, reqs = {}, set(), deque()

    def ask(key, m):
        if key in scenes or key in asked:
            return
        kind = "nav" if m["ds"] == 0 else "carla"
        ws = worker_qs[kind]
        ws[key[1] % len(ws)].put(("prep", key, m))
        asked.add(key)

    while True:
        busy = False
        while True:  # requests first
            try:
                job = req_q.get_nowait()
            except queue.Empty:
                break
            busy = True
            if job[0] == "stop":
                for ws in worker_qs.values():
                    for q in ws:
                        q.put(("stop", None, None))
                return
            if job[0] == "prep":
                for key, m in job[1]:
                    ask(key, m)
            else:  # ("label", [(key, meta, poses32)])
                for key, m, _ in job[1]:
                    ask(key, m)
                reqs.append(job[1])
        while True:
            try:
                msg = scene_q.get_nowait()
            except queue.Empty:
                break
            busy = True
            if msg[0] in ("scene", "error"):
                scenes[msg[1]] = msg
                asked.discard(msg[1])
            elif msg[0] == "fatal":
                out_q.put(msg)
        # label every request whose scenes are all in (in arrival order)
        for _ in range(len(reqs)):
            req = reqs.popleft()
            if not all(k in scenes for k, _, _ in req):
                reqs.append(req)
                continue
            busy = True
            t0 = time.time()
            by_kind = {}
            for key, m, p in req:
                msg = scenes.pop(key)
                if msg[0] == "error":
                    out_q.put(("error", key, "prep: " + msg[2]))
                    continue
                by_kind.setdefault(msg[2]["kind"], []).append((key, msg[2], p))
            for kind, items in by_kind.items():
                try:
                    outs = _label_kind(kind, [s for _, s, _ in items], np.stack([p for _, _, p in items]), R, M, A, torch)
                    dt = (time.time() - t0) / max(len(req), 1)
                    for (key, sc, _), o in zip(items, outs):
                        o["seconds"] = dict(context=sc["seconds"]["context"] + sc["seconds"]["extract"], label=dt)
                        out_q.put(("label", key, o))
                except Exception:  # noqa: BLE001
                    tb = traceback.format_exc()[-2000:]
                    for key, _, _ in items:
                        out_q.put(("error", key, "label: " + tb))
            done_steps = [k[0] for k, _, _ in req]
            if done_steps:  # drop scenes of steps that can no longer be labelled (as labeller.worker_main)
                lo = min(done_steps) - 1
                for k in [k for k in scenes if k[0] < lo and k[0] > -10 ** 5]:
                    scenes.pop(k, None)
        if not busy:
            try:
                msg = scene_q.get(timeout=0.002)
                if msg[0] in ("scene", "error"):
                    scenes[msg[1]] = msg
                    asked.discard(msg[1])
                elif msg[0] == "fatal":
                    out_q.put(msg)
            except queue.Empty:
                pass


class GPULabeller:
    """labeller.Labeller's interface (prep / label / collect / close) + train2's (prep_slots / label_slots / collect_keys)."""

    def __init__(self, n_nav=4, n_carla=10, mem_gb=4.0):
        import multiprocessing as mp
        c = mp.get_context("spawn")
        self.out_q, self.req_q, self.scene_q = c.Queue(), c.Queue(), c.Queue()
        self.worker_qs = {"nav": [c.Queue() for _ in range(n_nav)], "carla": [c.Queue() for _ in range(n_carla)]}
        self.procs = []
        for kind, qs in self.worker_qs.items():
            for q in qs:
                p = c.Process(target=prep_main, args=(kind, q, self.scene_q), daemon=True)
                p.start()
                self.procs.append(p)
        # workers report ready on scene_q, which the server owns: wait for them here first
        ready = 0
        while ready < n_nav + n_carla:
            msg = self.scene_q.get(timeout=900)
            if msg[0] == "fatal":
                raise RuntimeError(msg[2])
            ready += 1
        self.server = c.Process(target=server_main, args=(self.req_q, self.scene_q, self.out_q, self.worker_qs, mem_gb), daemon=True)
        self.server.start()
        msg = self.out_q.get(timeout=900)
        if msg[0] == "fatal":
            raise RuntimeError(msg[2])
        self.buf = {}

    @staticmethod
    def kind_of(m):
        return "nav" if m["ds"] == 0 else "carla"

    def prep(self, step, metas):
        self.req_q.put(("prep", [((step, s), m) for s, m in enumerate(metas)]))

    def prep_slots(self, step, slots, metas):
        self.req_q.put(("prep", [((step, s), m) for s, m in zip(slots, metas)]))

    def label(self, step, metas, poses):
        self.label_slots(step, list(range(len(metas))), metas, poses)

    def label_slots(self, step, slots, metas, poses):
        self.req_q.put(("label", [((step, s), m, np.asarray(p, np.float32)) for s, m, p in zip(slots, metas, poses)]))

    def collect_keys(self, keys, timeout=600):
        keys = list(keys)
        t_end = time.time() + timeout
        while any(k not in self.buf for k in keys):
            if time.time() > t_end:
                raise TimeoutError(f"labels missing after {timeout} s: {[k for k in keys if k not in self.buf][:4]}")
            try:
                msg = self.out_q.get(timeout=1.0)
            except queue.Empty:
                if not self.server.is_alive():
                    raise RuntimeError("GPU label server died")
                continue
            if msg[0] in ("label", "error"):
                self.buf[msg[1]] = msg
            elif msg[0] == "fatal":
                raise RuntimeError(msg[2])
        got, errs = {}, {}
        for k in keys:
            msg = self.buf.pop(k)
            (got if msg[0] == "label" else errs)[k[1]] = msg[2]
        return got, errs

    def collect(self, step, n, timeout=600):
        return self.collect_keys([(step, s) for s in range(n)], timeout)

    def close(self):
        self.req_q.put(("stop",))
        self.server.join(timeout=30)
        for p in self.procs:
            p.join(timeout=10)
