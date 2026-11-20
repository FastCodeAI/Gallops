"""run 3B live labelling of the planner's 32 proposals (run3b/CLAUDE.md section 3: "labelled live with the fact code (NAVSIM's
on NAVSIM samples, carla_full_v1's rules on LEAD samples)").

Worker processes (spawned; NAVSIM and CARLA fact code never share a process):
  NAVSIM  facts.scene_facts(mc, poses, sample, frames, today=False) with the slim Phase 0a context (build_nav_ctx.py, checked
          identical) -> the facts; run 3A's get_sub_score(metric cache, poses) -> run 3A's auxiliary labels (key agents,
          ego areas, map aux) for the planner's auxiliary heads (Navvrat 18 Nov: NAVSIM keeps them)
  CARLA   label_carla.context / label (carla_full_v1's steps, checked identical on stored pool rows)
Protocol: prep(step, slot, meta) loads the scene context ahead (the next batch, while the GPU works); label(step, slot, poses)
labels and returns. A worker always serves label requests before prep requests. Each (step, slot) goes to one fixed worker.
Returned per sample: facts (32, 8) arrays + seconds; NAVSIM also the aux labels."""
import lzma
import os
import pickle
import queue
import sys
import time
import traceback
import zlib

import numpy as np

DEVKIT = "{data.drivejepa_tree}/navsim_v2"
KEYS = ("hit", "offroad", "left_route", "red", "stop", "wrong_way", "metres", "lane_off", "ttc", "cutin_brake")


def _nav_env():
    os.chdir(DEVKIT)
    for p in (DEVKIT, "{runs}/run3a/facts"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import facts as F
    from navsim.agents.drive_jepa_run3a.score_module import compute_navsim_score as CNS
    meta = pickle.load(open("{data.paddock_data}/run3b/nav_ctx_meta.pkl", "rb"))
    pos = {m["token"]: (m["offset"], m["nbytes"]) for m in meta}
    fh = open("{data.paddock_data}/run3b/nav_ctx.bin", "rb")

    def context(m):
        o, n = pos[m["token"]]
        fh.seek(o)
        c = pickle.loads(zlib.decompress(fh.read(n)))
        with lzma.open(m["mc"], "rb") as f:
            c["mc"] = pickle.load(f)
        return c

    def label(c, poses, m):
        fa = F.scene_facts(c["mc"], poses, c["smp"], c["frames"], today=False)
        out = {k: np.asarray(fa[k])[1:] for k in KEYS}
        out["cutin_brake"] = np.nan_to_num(out["cutin_brake"].astype(np.float32), nan=0.0)
        out["comfort_step"] = np.asarray(fa["comfort"], np.float32)[1:].max(-1) >= 1.0
        res = CNS.get_sub_score(m["mc"], poses.astype(np.float32), False)
        out["aux"] = dict(key_corners=np.asarray(res[1], np.float32), key_valid=np.asarray(res[2], bool),
                          ego_areas=np.asarray(res[3], bool), map_aux=np.asarray(res[5], np.float32))
        return out
    return context, label


def _carla_env():
    os.chdir(DEVKIT)
    for p in (DEVKIT, "{runs}/run3b/r3b"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import label_carla as LC
    LC.FU.setup()

    def context(m):
        return LC.context(m["zrel"], int(m["frame"]), bool(m["shifted"]))

    def label(c, poses, m):
        out = LC.label(c, poses)
        out["comfort_step"] = np.zeros(out["hit"].shape, bool)
        return out
    return context, label


def worker_main(kind, prep_q, label_q, out_q):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        context, label = _nav_env() if kind == "nav" else _carla_env()
    except Exception:  # noqa: BLE001
        out_q.put(("fatal", kind, traceback.format_exc()))
        return
    out_q.put(("ready", kind, os.getpid()))
    ctx = {}
    while True:
        job = None
        try:
            job = label_q.get_nowait()
        except queue.Empty:
            try:
                job = prep_q.get_nowait()
            except queue.Empty:
                try:
                    job = label_q.get(timeout=0.02)
                except queue.Empty:
                    continue
        if job is None or job[0] == "stop":
            return
        what, key, payload = job
        try:
            if what == "prep":
                if key not in ctx:
                    t0 = time.time()
                    ctx[key] = (context(payload), payload, time.time() - t0)
            else:
                poses, m = payload
                t0 = time.time()
                if key not in ctx:
                    ctx[key] = (context(m), m, time.time() - t0)
                c, _, t_ctx = ctx.pop(key)
                t1 = time.time()
                out = label(c, np.asarray(poses, np.float64), m)
                out["seconds"] = dict(context=t_ctx, label=time.time() - t1)
                out_q.put(("label", key, out))
                for k in [k for k in ctx if k[0] < key[0] - 1]:  # drop contexts of steps that will never be labelled
                    ctx.pop(k, None)
        except Exception:  # noqa: BLE001
            ctx.pop(key, None)
            out_q.put(("error", key, traceback.format_exc()[-2000:]))


class Labeller:
    def __init__(self, n_nav, n_carla):
        import multiprocessing as mp
        c = mp.get_context("spawn")
        self.out_q = c.Queue()
        self.w = {"nav": [], "carla": []}
        for kind, n in (("nav", n_nav), ("carla", n_carla)):
            for _ in range(n):
                pq, lq = c.Queue(), c.Queue()
                p = c.Process(target=worker_main, args=(kind, pq, lq, self.out_q), daemon=True)
                p.start()
                self.w[kind].append((p, pq, lq))
        ready = 0
        while ready < n_nav + n_carla:
            msg = self.out_q.get(timeout=900)
            if msg[0] == "fatal":
                raise RuntimeError(msg[2])
            ready += 1
        self.pending = {}

    def _worker(self, kind, slot):
        ws = self.w[kind]
        return ws[slot % len(ws)]

    @staticmethod
    def kind_of(m):
        return "nav" if m["ds"] == 0 else "carla"

    def prep(self, step, metas):
        for slot, m in enumerate(metas):
            self._worker(self.kind_of(m), slot)[1].put(("prep", (step, slot), m))

    def label(self, step, metas, poses):
        for slot, (m, p) in enumerate(zip(metas, poses)):
            self._worker(self.kind_of(m), slot)[2].put(("label", (step, slot), (np.asarray(p, np.float32), m)))
            self.pending[(step, slot)] = True

    def collect(self, step, n, timeout=600):
        got, errors = {}, {}
        t_end = time.time() + timeout
        while len(got) + len(errors) < n:
            msg = self.out_q.get(timeout=max(1.0, t_end - time.time()))
            if msg[0] == "label" and msg[1][0] == step:
                got[msg[1][1]] = msg[2]
            elif msg[0] == "error" and msg[1][0] == step:
                errors[msg[1][1]] = msg[2]
            self.pending.pop(msg[1], None) if msg[0] in ("label", "error") else None
        return got, errors

    def close(self):
        for ws in self.w.values():
            for p, pq, lq in ws:
                lq.put(("stop", None, None))
        for ws in self.w.values():
            for p, _, _ in ws:
                p.join(timeout=10)
