"""run 3B v2 held-out evaluation (run3b/CLAUDE.md section 6 "Report at step 2,000 and after every pass", items 2-7, and section 3's
last bullet: every scorer / pick number twice, with true agents and with the decoder's). Item 1 (step time, losses) is the
trainer's. Numbers only, every number with its n.

Held-out samples: N_NAV NAVSIM (the 51 held-out logs) and N_LEAD LEAD (the 416 held-out routes), evenly spaced over the data
object's held-out lists (np.linspace(0, len - 1, n).round(), as r3b/evaluate.py), the same draw every time (data.deterministic).

  2  agents     run 3A's agent-decoder evaluation rule, reused by import:
                  match   agent-decoder/common.py match(): detection = 1 - P(none) >= 0.5 (THR), class = argmax of the 3 agent
                          classes, greedy by confidence (stable), a hit when the nearest still-free true agent of the same class
                          is within 2 m (HIT_M), x-y centre.  Last decoder layer, all slots.
                  rings   lead_inputs_v1/part_c_eval.py RINGS / ring_of: 0-10, 10-20, 20-32 (32 included), 32-46 m (corners of
                          the |x|, |y| <= 32 m box) by the true agent's x-y range (recall) / the detection's own (precision)
                  dirs    common.region() via part_c_eval.dir_of: bearing |b| <= 45 deg front, >= 135 behind, beside between, beside
                          also split left (y > 0) / right
                  keys    part_c_eval.keys() (all, class, dir, ring, ring x dir, ring x class), with one fix: part_c_eval adds
                          "dir front" / "dir behind" twice per agent (its 'dir {d}' and 'dir {beside or d}' keys coincide), so its
                          n_true / n_det for front and behind (and ring x front / behind) are doubled; recall and precision are
                          not affected. Here each key is counted once per agent.
                  truth   NAVSIM: common.gt_of() on navtrain_agents.pkl (the decoder's own targets, all agents with name vehicle /
                          pedestrian / bicycle and |x|, |y| <= 32 m, no 64 cut) = run3a/proof/p6_navsim_det.py (0.62 at 10-20 m).
                          LEAD: lead_inputs_v2 boxes, cls3 (vehicle incl. emergency and, in v2, parked cars; pedestrian; bicycle),
                          'as Bench2Drive' = in_range (|x|, |y| <= 32) & not underground & cls3 >= 0 (part_c_eval.py, run 3A's
                          0.303 / 0.406 / 0.443 key); 'camera-visible' = also in_views & vis_px > 0. LEAD also split normal /
                          shifted. No class merge (run 3A's 3 classes; Phase 1's emergency merge does not apply).
                Computed once, on the decoder's pass (the decoder does not read the agents the looks are fed); the largest
                difference between the two passes' decoder logits is reported.
  3  facts      r3b/evaluate.py item 4, per pass: within-scene pairwise accuracy per yes/no head (path label = the fact at some
                step; prediction = max over steps of the probability; red: paths of unknown colour left out; comfort NAVSIM only)
                on all scored paths (the batch's expert + draw + the planner's 32, labelled live) and on the planner's 32 alone;
                number heads: absolute error per waypoint (ttc prediction clipped to [0, 4] s, cut-in braking to [0, 8]).
  4  metres     r3b/evaluate.py item 5, per pass: |head - true expert metres so far| at 2 s (step 4) and 4 s (step 8); LEAD split by
                'expert stops' (the expert gains < 0.25 m in its last 0.5 s step) against the rest.
  5  pick       r3b/evaluate.py item 6, per pass and per expert-metres source (true / head): pick = argmax over the 32 of the
                score from predicted facts (r3b/score.py score_torch, T4, P); its true score (live facts, true expert metres)
                against the best true score of the 32; share whose pick is a true top (>= best - 1e-6); share of picks that stop
                (< 0.5 m over 4 s) among samples whose expert moves (>= 0.5 m).
  6  planner    r3b/evaluate.py item 7, per pass: nearest proposal's distance to the expert path (mean over the 8 waypoints of the
                x-y distance); spread of the 32 (mean pairwise distance, same measure). NAVSIM: EPDMS (epdms() below) of the pick
                (expert metres true / head) and of the best of the 32, next to run 3A's 0.920 / 0.961 (run3a/eval/pick_ep13/
                compare_heldout.json "all 51", 4,457 samples; recomputed from pick_tune's heads_labels.npz) and run 3A on the same
                samples from that file (run3a_navsim_same_samples).
  7  teachers   LEAD frames with 4 / 1-3 / 0 teachers (len(batch["teachers"][i])), teachers' distance to the expert (p50 / p90).

EPDMS = run 3A's held-out label (run3a/build/pick_tune.py, p2_heldout.py, compare_heldout.py: get_scores -> get_sub_score
targets[:, 12], "today's final label, unmasked"): navsim_v2 navsim/agents/drive_jepa_run3a/score_module/compute_navsim_score.py,
before_score_proposals (PDM-Closed as row 0 + the poses, 0.5 s -> 0.1 s, PDMSimulator) then its module-level PDMScorer
(train_pdm_scorer.PDMScorerConfig()).score_proposals(..., simulated_agent_detections_tracks=None, past human trajectory) on
train_metric_cache_v2_pruned/<log>/unknown/<token>/metric_cache.pkl; the value for each pose row is final_scores[1:] (= get_sub_score
scores[:, 8]). Progress is normalised per row against PDM-Closed only (_aggregate_pdm_scores), so a row's score does not depend on
the other rows scored with it.

API
  epdms(tokens, trajectories, workers=8) -> (N,) or (N, K) float64
  run_eval_v2(model_fn, lab, data, n_nav=1008, n_lead=1008, batch=32, teacher_forcing=True, ...) -> dict (and JSON if out=)
  model_fn(batch, teacher_forced: bool) -> dict (called under torch.no_grad(); eval mode, autocast and device are model_fn's;
  outputs may be CUDA / bf16 tensors or numpy):
      agent_outs     list over decoder layers of dicts: "logits" (or "cls") (B, Q, 4) in order vehicle, pedestrian, bicycle, none;
                     "box" (B, Q, 8) = x, y, z, log l, log w, log h, sin yaw, cos yaw; "vel" (B, Q, 2) (run 3A agent_decoder.py)
      proposals      (B, 32, 8, 3) x, y, heading, rear axle, ego frame at t, 0.5 .. 4.0 s
      path_logits    (B, Pd + 32, 8, 7) yes/no logits in YESNO order, for batch["paths"] (Pd) then the 32 proposals
      path_numbers   (B, Pd + 32, 8, 4) NUMBERS order
      expert_metres  (B, 8) the scene head
  batch (collate(samples)) fields read: ds (B,) 0 = NAVSIM else LEAD; paths (B, Pd, 8, 3); valid (B, Pd); facts dict of (B, Pd, 8)
      (YESNO + NUMBERS + red_known); R (B, 8) true expert metres so far; ntg (B,); expert (B, 8, 3); teachers list of (K, 8, 3);
      meta list of dicts: NAVSIM token (+ log), LEAD token, shifted (+ route) -- r3b/data.py Run3BData / collate.
  data: .nav, .lead (held-out lists; global index [0, n_nav) NAVSIM then LEAD), data[g], data.label_meta(g), .deterministic,
      collate = argument or data.collate.
  lab: gpulab/service.GPULabeller (or r3b labeller.Labeller): prep(step, metas), label(step, metas, poses), collect(step, n).
      Both passes' proposals go in one label request per sample (64 paths), so each scene is prepared once. lab=None: no live
      labels (items 3 on the batch's paths only; 5 needs them and is left out).
"""
import importlib.util
import json
import os
import pickle
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEVKIT = "{data.drivejepa_tree}/navsim_v2"
R3B = Path("{runs}/run3b/r3b")
AGDEC_COMMON = Path("{runs}/agent-decoder/common.py")
PARTC_EVAL = Path("{runs}/run3a/lead_inputs_v1/part_c_eval.py")
MC_ROOT = "{data.p3_renesas}/train_metric_cache_v2_pruned"
NAV_META = "{data.paddock_data}/run3b/pack/navsim_meta.pkl"
LEAD_V2 = Path("{data.paddock_data}/lead_inputs_v2/routes")
NAVTRAIN_AGENTS = "{data.p3_renesas}/bev-e1/navtrain_agents.pkl"
HELDOUT_LOGS = "{data.wajepa_work}/step2/preflight/heldout_logs.txt"
TRUTH_CACHE = HERE / "cache" / "navsim_heldout_agents.pkl"
P2_DUMP = "{runs}/run3a/proof/p2_dump.npz"
PICK_GRID = "{runs}/run3a/eval/pick_ep13/grid.json"
HEADS_LABELS = "{runs}/run3a/eval/pick_ep13/heads_labels.npz"  # pick_tune.py's dump: the 0.920 / 0.961 source
RUN3A_LEAD_REF = HERE / "run3a_reference.json"

CLS3 = ("vehicle", "pedestrian", "bicycle")
YESNO = ("hit", "offroad", "left_route", "red", "stop", "wrong_way", "comfort")      # = r3b/losses.py YESNO
NUMBERS = ("metres_sofar", "lane_off", "ttc", "cutin_brake")                         # = r3b/losses.py NUMBERS
N_PROP = 32
PASSES = (("decoded agents", False), ("true agents", True))
QUOTED = {  # run3b/CLAUDE.md section 6, with the files they come from
    "agents LEAD run 3A epoch 13 (part_c.json 'run 3A epoch 13 | LEAD trial | as Bench2Drive', 4,990 trial frames, lead_inputs_v1 boxes)":
        {"all recall": 0.303, "all precision": 0.4055, "ring 10-20 m recall": 0.4433, "n_true": 34276},
    "agents NAVSIM run 3A epoch 13 (run3a/proof/p6_navsim_det.json, 4,457 held-out samples)": {"ring 10-20 m recall": 0.6223, "n_true": 25820},
    "EPDMS run 3A epoch 13 tuned pick (run3a/eval/pick_ep13/compare_heldout.json 'all 51', 4,457 samples)":
        {"pick": 0.9204, "best of 32": 0.9611, "default pick": 0.9174},
    "facts run 3A heads NAVSIM (spec quote)": {"collision": 0.972, "drivable": 0.959, "direction": 0.967, "light": 0.959, "progress": 0.816},
}


# ------------------------------------------------------------------------------------------------ reused modules, by file path
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOD = {}


def common():
    """agent-decoder/common.py (match, region, gt_of, THR, HIT_M), loaded under its own name (no sys.path change)."""
    if "common" not in _MOD:
        _MOD["common"] = _load("agdec_common", AGDEC_COMMON)
    return _MOD["common"]


def partc():
    """lead_inputs_v1/part_c_eval.py (RINGS, ring_of, dir_of, keys, Tally); its `from common import ...` is pointed at
    agent-decoder/common.py, and sys.path / sys.modules['common'] are put back afterwards."""
    if "partc" not in _MOD:
        cm = common()
        path0, old = list(sys.path), sys.modules.get("common")
        sys.modules["common"] = cm
        try:
            _MOD["partc"] = _load("run3a_part_c_eval", PARTC_EVAL)
        finally:
            sys.path[:] = path0
            if old is None:
                sys.modules.pop("common", None)
            else:
                sys.modules["common"] = old
    return _MOD["partc"]


def score_mod():
    """r3b/score.py (T4 score: parts_np for true facts, score_torch for predicted)."""
    if "score" not in _MOD:
        _MOD["score"] = _load("r3b_score", R3B / "score.py")
    return _MOD["score"]


# ------------------------------------------------------------------------------------------------ item 2: agents
def agent_keys(cls, x, y, names, extra):
    """part_c_eval.keys() with each key once per agent (see the module doc)."""
    return [list(dict.fromkeys(k)) for k in partc().keys(cls, x, y, names, extra)]


def score_agents(tal, conf, cls, xy, gxy, gcls, names=CLS3, extra=("",)):
    """part_c_eval.score() (no merge) with agent_keys(): one frame's slots against its true agents into tally tal."""
    det, tp, hit = common().match(np.asarray(conf, np.float64), np.asarray(cls), np.asarray(xy, np.float64),
                                  np.asarray(gxy, np.float64).reshape(-1, 2), np.asarray(gcls, np.int64))
    gxy = np.asarray(gxy, np.float64).reshape(-1, 2)
    xy = np.asarray(xy, np.float64)
    tal.add(agent_keys(np.asarray(gcls, np.int64), gxy[:, 0], gxy[:, 1], names, list(extra)), (hit >= 0).astype(float),
            agent_keys(np.asarray(cls)[det], xy[det, 0], xy[det, 1], names, list(extra)), tp.astype(float))


def new_tally(names=CLS3):
    return partc().Tally(names)


def decode_slots(out):
    """run 3A's read of a decoder layer (part_c_run3a.py, p2_heldout.py): p = softmax(logits); conf = 1 - p[none];
    class = argmax of the 3 agent classes; centre = box[..., :2]. -> numpy conf (B, Q), cls (B, Q), xy (B, Q, 2)."""
    import torch
    lg = out["logits"] if "logits" in out else out["cls"]
    p = torch.as_tensor(lg).detach().float().cpu().softmax(-1)
    return ((1 - p[..., -1]).cpu().numpy(), p[..., :-1].argmax(-1).cpu().numpy(),
            torch.as_tensor(out["box"])[..., :2].detach().float().cpu().numpy())


def build_navsim_truth_cache():
    """navtrain_agents.pkl (224 MB) -> the 51 held-out logs only, pickled to cache/ (new file, never overwritten)."""
    held = {l.strip() for l in open(HELDOUT_LOGS) if l.strip()}
    with open(NAVTRAIN_AGENTS, "rb") as f:
        agents = pickle.load(f)
    sub = {lg: agents[lg] for lg in agents if lg in held}
    del agents
    TRUTH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = TRUTH_CACHE.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "xb") as f:
        pickle.dump(sub, f, protocol=4)
    if TRUTH_CACHE.exists():
        tmp.unlink()
    else:
        tmp.rename(TRUTH_CACHE)
    return sub


class NavsimTruth:
    """(log, token) -> common.gt_of: x-y centres (N, 2), classes (N,) in CLS3."""

    def __init__(self):
        if TRUTH_CACHE.exists():
            with open(TRUTH_CACHE, "rb") as f:
                self.agents = pickle.load(f)
        else:
            self.agents = build_navsim_truth_cache()
        self.tok2log = {t: lg for lg in self.agents for t in self.agents[lg]}

    def __call__(self, token, log=None):
        log = log or self.tok2log[token]
        gxyz, gcls, _ = common().gt_of(self.agents, f"{log}/{token}")
        return gxyz[:, :2], gcls


class LeadTruth:
    """(token, shifted) -> lead_inputs_v2 record boxes: xy (N, 2), cls3 (N,), base ('as Bench2Drive'), vis ('camera-visible')."""

    def __init__(self, keep=16):
        self.cache, self.keep = OrderedDict(), keep

    def record(self, token, shifted, route=None):
        route = route or token.split("#")[0]
        if route not in self.cache:
            import gzip
            recs = pickle.load(gzip.open(LEAD_V2 / f"{route}.pkl.gz"))["records"]
            self.cache[route] = {(r["token"], bool(r["shifted"])): r for r in recs}
            if len(self.cache) > self.keep:
                self.cache.popitem(last=False)
        return self.cache[route][(token, bool(shifted))]

    def __call__(self, token, shifted, route=None):
        B = self.record(token, shifted, route)["boxes"]
        base = B["in_range"] & ~B["underground"] & (B["cls3"] >= 0)
        vis = base & B["in_views"] & (B["vis_px"] > 0)
        return B["box"][:, :2], B["cls3"].astype(np.int64), base, vis


# ------------------------------------------------------------------------------------------------ EPDMS (run 3A's held-out label)
_CNS = None


def _epdms_init():
    global _CNS
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    try:
        os.nice(10 - os.nice(0)) if os.nice(0) < 10 else None
    except OSError:
        pass
    if DEVKIT not in sys.path:
        sys.path.insert(0, DEVKIT)
    os.chdir(DEVKIT)  # as labeller._nav_env
    from navsim.agents.drive_jepa_run3a.score_module import compute_navsim_score as CNS
    _CNS = CNS


def _epdms_one(job):
    """(metric cache path, poses (K, 8, 3) float32) -> (K,) final scores = get_sub_score(...)[0][:, 12]."""
    path, poses = job
    CNS = _CNS
    mc = CNS.load_metric_cache(path)
    states = CNS.before_score_proposals(mc, poses, mc.ego_state)
    final = CNS.scorer.score_proposals(states, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
                                       mc.map_parameters, None, mc.past_human_trajectory)
    return np.asarray(final[1:], np.float64)


_TOK2LOG = None


def mc_path(t):
    """token | (log, token) | metric cache path -> metric cache path (logs from run3b pack/navsim_meta.pkl)."""
    global _TOK2LOG
    if isinstance(t, (tuple, list)):
        return f"{MC_ROOT}/{t[0]}/unknown/{t[1]}/metric_cache.pkl"
    t = str(t)
    if t.endswith(".pkl"):
        return t
    if _TOK2LOG is None:
        with open(NAV_META, "rb") as f:
            _TOK2LOG = {m["token"]: m["log"] for m in pickle.load(f)["meta"]}
    return f"{MC_ROOT}/{_TOK2LOG[t]}/unknown/{t}/metric_cache.pkl"


def epdms(tokens, trajectories, workers=8, chunksize=2):
    """run 3A's held-out EPDMS label for trajectories (N, 8, 3) -> (N,), or (N, K, 8, 3) -> (N, K) (one metric-cache load per
    sample). Poses: x, y, heading, rear axle, ego frame at t, 0.5 .. 4.0 s (passed as float32, as run 3A's proposals were).
    CPU, `workers` spawned processes (<= 8), nice 10; workers=0 runs in this process."""
    tr = np.asarray(trajectories, np.float32)
    single = tr.ndim == 3
    if single:
        tr = tr[:, None]
    assert tr.ndim == 4 and tr.shape[2:] == (8, 3) and len(tr) == len(tokens), tr.shape
    jobs = [(mc_path(t), tr[i]) for i, t in enumerate(tokens)]
    workers = min(int(workers), 8, len(jobs))
    if workers <= 0:
        cwd = os.getcwd()
        if _CNS is None:
            _epdms_init()
        out = [_epdms_one(j) for j in jobs]
        os.chdir(cwd)
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers, initializer=_epdms_init) as pool:
            out = pool.map(_epdms_one, jobs, chunksize=chunksize)
    out = np.stack(out)
    return out[:, 0] if single else out


# ------------------------------------------------------------------------------------------------ run 3A on the same samples
def run3a_pick(z, gates, route, alpha):
    """run3a/build/compare_heldout.py pick() (copied: that file is a script): run 3A's pick from its 12 head logits."""
    import torch
    z = torch.tensor(z)
    ls = (torch.nn.functional.logsigmoid(z[..., :5]) * torch.tensor(gates)).sum(-1)
    u = torch.tensor((5.0, 5.0, 2.0, 2.0, route))
    ls = ls + torch.log((torch.sigmoid(z[..., 5:10]) * u).sum(-1) / u.sum() + 1e-6)
    if alpha:
        ls = (1 - alpha) * ls + alpha * torch.nn.functional.logsigmoid(z[..., 11])
    return ls.argmax(1).numpy()


def run3a_navsim_same_samples(tokens, nav_truth=None):
    """run 3A epoch 13 on the given NAVSIM held-out tokens from its stored dumps; nothing is run.
    EPDMS: run3a/eval/pick_ep13/heads_labels.npz (pick_tune.py: head logits + training-scorer labels of the 32; the file
    compare_heldout.json's 0.9204 / 0.9611 come from) -> tuned pick (grid.json best), default pick, best of 32.
    Agents: run3a/proof/p2_dump.npz decoder last layer (p6_navsim_det.json's source; p2 re-ran the model at batch 32, so its
    proposals / labels differ slightly from heads_labels': on all 4,457 its tuned pick gives 0.9200, default 0.9175, best 0.9611)."""
    out = {"n_requested": int(len(tokens))}
    h = np.load(HEADS_LABELS)
    pos = {t: i for i, t in enumerate(h["tokens"])}
    rows = np.array([pos[t] for t in tokens if t in pos], np.int64)
    if len(rows):
        best = json.load(open(PICK_GRID))["best"]
        lab, z, r = h["labels"][rows, :, -1], h["logits"][rows], np.arange(len(rows))
        out["epdms"] = {"pick (tuned, locked)": float(lab[r, run3a_pick(z, tuple(best["gates"]), best["route"], best["alpha"])].mean()),
                        "pick (default)": float(lab[r, run3a_pick(z, (1.0,) * 5, 2.0, 0.0)].mean()),
                        "best of 32": float(lab.max(1).mean()), "n": int(len(rows)), "source": HEADS_LABELS}
    d = np.load(P2_DUMP)
    pos = {t: i for i, t in enumerate(d["tokens"])}
    rows = np.array([pos[t] for t in tokens if t in pos], np.int64)
    if len(rows):
        nav_truth = nav_truth or NavsimTruth()
        tal = new_tally()
        for i in rows:
            gxy, gcls = nav_truth(str(d["tokens"][i]))
            score_agents(tal, d["conf"][i], d["cls"][i], d["xy"][i], gxy, gcls)
        out["agents"] = tal.out()
        out["agents_n_samples"], out["agents_source"] = int(len(rows)), P2_DUMP
    return out


# ------------------------------------------------------------------------------------------------ helpers
def pairwise(lab, pred, groups):
    """r3b/evaluate.py pairwise(): lab bool (n,), pred (n,), groups (n,) -> (mean over scenes with contrast, scenes, pairs)."""
    accs, npairs = [], 0
    for g in np.unique(groups):
        m = groups == g
        l, p = lab[m], pred[m]
        pos, neg = p[l], p[~l]
        if len(pos) == 0 or len(neg) == 0:
            continue
        cmp = (pos[:, None] > neg[None]).astype(float) + 0.5 * (pos[:, None] == neg[None])
        accs.append(cmp.mean())
        npairs += cmp.size
    return (float(np.mean(accs)) if accs else None), len(accs), npairs


def live_targets(got, B, R, ntg, is_nav, P, lo, hi):
    """labeller output, columns lo:hi (one pass's 32) -> (B, 32, 8) targets + true score parts (r3b/train2.py live_targets with
    a column slice)."""
    S = score_mod()
    n = hi - lo
    t = {k: np.zeros((B, n, 8), np.float32) for k in YESNO + NUMBERS}
    t["red_known"] = np.zeros((B, n, 8), bool)
    ok = np.zeros(B, bool)
    a, b = np.zeros((B, n), np.float32), np.zeros((B, n), np.float32)
    for i in range(B):
        g = got.get(i)
        if g is None:
            continue
        g = {k: (np.asarray(v)[lo:hi] if k != "aux" and k != "seconds" else v) for k, v in g.items()}
        ok[i] = True
        for k in ("hit", "offroad", "left_route", "stop", "wrong_way"):
            t[k][i] = g[k]
        t["red"][i] = g["red"] == 1
        t["red_known"][i] = g["red"] >= 0
        t["comfort"][i] = g["comfort_step"]
        t["metres_sofar"][i] = np.cumsum(g["metres"].astype(np.float64), -1)
        t["ttc"][i] = np.minimum(g["ttc"].astype(np.float32), 4.0)
        t["cutin_brake"][i] = np.minimum(np.nan_to_num(g["cutin_brake"].astype(np.float32), nan=0.0), 8.0)
        t["lane_off"][i] = g["lane_off"]
        a[i], b[i] = S.parts_np(g, R[i], bool(ntg[i]), bool(is_nav[i]))
    return t, ok, a, b


def scores_from_pred(logits, numbers, R, ntg, is_nav, P):
    """r3b/losses.py predicted_facts + scores_from_pred: the T4 score from predicted facts (torch)."""
    import torch
    p = torch.sigmoid(logits.float())
    d = {k: p[..., i] for i, k in enumerate(YESNO)}
    for i, k in enumerate(NUMBERS):
        d[k] = numbers[..., i].float()
    d["ttc"] = d["ttc"].clamp(0.0, 4.0)
    d["cutin_brake"] = d["cutin_brake"].clamp(0.0, 8.0)
    sc, _, _ = score_mod().score_torch(d, R.float(), ntg, is_nav, P=P)
    return sc


def _np(x):
    import torch
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, np.float32)


def _stats(x):
    x = np.asarray(x, np.float64)
    if not len(x):
        return {"n": 0}
    return dict(mean=float(x.mean()), p50=float(np.median(x)), p90=float(np.percentile(x, 90)), n=int(len(x)))


# ------------------------------------------------------------------------------------------------ the driver
def run_eval_v2(model_fn, lab, data, n_nav=1008, n_lead=1008, batch=32, teacher_forcing=True, P=0.108, collate=None,
                epdms_workers=8, step_base=-10 ** 6, out=None, extra=None, log=print, label_timeout=900, run3a_same=True):
    """the held-out evaluation (module doc). Returns the result dict; writes it to `out` (JSON, never overwritten) if given."""
    import torch
    collate = collate or getattr(data, "collate")
    t0 = time.time()
    passes = PASSES if teacher_forcing else PASSES[:1]
    pnames = [p for p, _ in passes]
    was_det = getattr(data, "deterministic", False)
    data.deterministic = True
    nn_, nl_ = len(data.nav), len(data.lead)
    nav_i = np.linspace(0, nn_ - 1, min(n_nav, nn_)).round().astype(int) if n_nav > 0 and nn_ else np.zeros(0, int)
    lead_i = (np.linspace(0, nl_ - 1, min(n_lead, nl_)).round().astype(int) if n_lead > 0 and nl_ else np.zeros(0, int)) + nn_
    order = np.r_[nav_i, lead_i]
    chunks = [[int(g) for g in order[s:s + batch]] for s in range(0, len(order), batch)]
    nav_truth, lead_truth = NavsimTruth() if len(nav_i) else None, LeadTruth()
    tal = {"NAVSIM": new_tally(), "LEAD as Bench2Drive": new_tally(), "LEAD camera-visible": new_tally()}
    R = defaultdict(list)
    E = []  # NAVSIM: (token or path, proposals per pass, picks per pass)
    dec_diff = 0.0
    key = lambda s: step_base - s  # noqa: E731  (negative label keys, never the trainer's)
    metas_of = lambda idx: [data.label_meta(g) for g in idx]  # noqa: E731
    if lab is not None and chunks:
        lab.prep(key(0), metas_of(chunks[0]))
    for step, idx in enumerate(chunks):
        metas = metas_of(idx)
        if lab is not None and step + 1 < len(chunks):
            lab.prep(key(step + 1), metas_of(chunks[step + 1]))
        b = collate([data[g] for g in idx])
        Bn = len(idx)
        ds = np.asarray(_np(b["ds"])).astype(int)
        is_nav = ds == 0
        dsn = ["NAVSIM" if x else "LEAD" for x in is_nav]
        with torch.no_grad():
            outs = {pn: model_fn(b, teacher_forced=tf) for pn, tf in passes}
        props = {pn: _np(o["proposals"])[:, :N_PROP] for pn, o in outs.items()}
        for pn in pnames:
            assert props[pn].shape == (Bn, N_PROP, 8, 3), (pn, props[pn].shape)
        # live labels: both passes' 32 in one request per sample
        lt = {}
        if lab is not None:
            lab.label(key(step), metas, np.concatenate([props[pn] for pn in pnames], 1).astype(np.float32))
            got, errs = lab.collect(key(step), Bn, timeout=label_timeout)
            R["label_errors"].append(len(errs))
            for k_, e_ in list(errs.items())[:max(0, 3 - len(R["label_error_text"]))]:
                R["label_error_text"].append(f"{dsn[k_] if k_ < Bn else '?'} {metas[k_].get('token') if k_ < Bn else k_}: {str(e_)[-600:]}")
            for k, pn in enumerate(pnames):
                lt[pn] = live_targets(got, Bn, _np(b["R"]), _np(b["ntg"]), is_nav, P, N_PROP * k, N_PROP * (k + 1))
        # 2 agents (decoder's pass)
        ao = outs[pnames[0]]["agent_outs"][-1]
        if len(pnames) > 1:
            lg = lambda o: torch.as_tensor(o["agent_outs"][-1].get("logits", o["agent_outs"][-1].get("cls"))).detach().float().cpu()  # noqa: E731
            dec_diff = max(dec_diff, float((lg(outs[pnames[0]]) - lg(outs[pnames[1]])).abs().max()))
        conf, cls, xy = decode_slots(ao)
        for i in range(Bn):
            m = b["meta"][i]
            if is_nav[i]:
                gxy, gcls = nav_truth(m["token"], m.get("log"))
                score_agents(tal["NAVSIM"], conf[i], cls[i], xy[i], gxy, gcls)
            else:
                gxy, gcls, base, vis = lead_truth(m["token"], m["shifted"], m.get("route"))
                ex = ["", "shifted" if m["shifted"] else "normal"]
                score_agents(tal["LEAD as Bench2Drive"], conf[i], cls[i], xy[i], gxy[base], gcls[base], extra=ex)
                score_agents(tal["LEAD camera-visible"], conf[i], cls[i], xy[i], gxy[vis], gcls[vis], extra=ex)
        Rt, ex_ = _np(b["R"]), _np(b["expert"])
        bf = {k: _np(v) for k, v in b["facts"].items()}
        bvalid = _np(b["valid"]).astype(bool)
        Pd = bvalid.shape[1]
        # 7 LEAD teachers
        for i in range(Bn):
            if not is_nav[i]:
                te = _np(b["teachers"][i]).reshape(-1, 8, 3)
                R["teach_n"].append(len(te))
                if len(te):
                    R["teach_dist"] += list(np.linalg.norm(te[:, :, :2] - ex_[i, None, :, :2], axis=-1).mean(-1))
        picks = {}
        for pn in pnames:
            o = outs[pn]
            logits = torch.as_tensor(o["path_logits"]).detach().float().cpu()
            numbers = torch.as_tensor(o["path_numbers"]).detach().float().cpu()
            assert logits.shape[:3] == (Bn, Pd + N_PROP, 8) and logits.shape[-1] == len(YESNO), logits.shape
            assert numbers.shape[:3] == (Bn, Pd + N_PROP, 8) and numbers.shape[-1] == len(NUMBERS), numbers.shape
            have = pn in lt
            if have:
                t_, ok, la, lb = lt[pn]
                tgt = {k: np.concatenate([bf[k], t_[k]], 1) for k in t_}
                valid = np.concatenate([bvalid, np.repeat(ok[:, None], N_PROP, 1)], 1)
            else:
                tgt, valid = bf, bvalid
            prob = torch.sigmoid(logits).numpy()[:, :valid.shape[1]]
            num = numbers.numpy()[:, :valid.shape[1]]
            plc = np.zeros(valid.shape, bool)
            plc[:, Pd:] = True
            # 3 facts
            for i in range(Bn):
                v = valid[i]
                for j, k in enumerate(YESNO):
                    if k == "comfort" and not is_nav[i]:
                        continue
                    labp = tgt[k][i].astype(bool).any(-1)
                    keep = v.copy()
                    if k == "red":
                        keep &= tgt["red_known"][i].all(-1)
                    pre = f"{pn}|{dsn[i]}|{k}"
                    R[pre + "|lab"].append(labp[keep])
                    R[pre + "|pred"].append(prob[i, :, :, j].max(-1)[keep])
                    R[pre + "|grp"].append(np.full(int(keep.sum()), idx[i]))
                    R[pre + "|planner"].append(plc[i][keep])
                for j, k in enumerate(NUMBERS):
                    pv = num[i, :, :, j]
                    if k == "ttc":
                        pv = np.clip(pv, 0, 4)
                    if k == "cutin_brake":
                        pv = np.clip(pv, 0, 8)
                    R[f"{pn}|{dsn[i]}|{k}|abs_err"].append(np.abs(pv - tgt[k][i])[v].reshape(-1))
            # 4 expert metres head
            em = _np(o["expert_metres"]).reshape(Bn, 8)
            for i in range(Bn):
                R[f"{pn}|{dsn[i]}|metres|err2"].append(abs(em[i, 3] - Rt[i, 3]))
                R[f"{pn}|{dsn[i]}|metres|err4"].append(abs(em[i, 7] - Rt[i, 7]))
                R[f"{pn}|{dsn[i]}|metres|stops"].append(bool(Rt[i, 7] - Rt[i, 6] < 0.25))
            # 5 pick (needs live labels for the true score)
            nav_t = torch.as_tensor(is_nav)
            for which, Ruse in (("true", torch.as_tensor(Rt)), ("head", torch.as_tensor(em))):
                ntg = torch.as_tensor(_np(b["ntg"]).astype(bool)) if which == "true" else (Ruse[:, -1] < 0.5)
                sc = scores_from_pred(logits[:, -N_PROP:], numbers[:, -N_PROP:], Ruse, ntg, nav_t, P).numpy()
                pk = sc.argmax(1)
                picks[(pn, which)] = pk
                if not have:
                    continue
                true = la - P * lb
                for i in range(Bn):
                    if not ok[i]:
                        continue
                    best = true[i].max()
                    pre = f"{pn}|{dsn[i]}|pick_{which}"
                    R[pre + "|gap"].append(best - true[i, pk[i]])
                    R[pre + "|top"].append(true[i, pk[i]] >= best - 1e-6)
                    pm = t_["metres_sofar"][i, pk[i], -1]
                    R[pre + "|stops_expert_moves"].append(bool(pm < 0.5 and Rt[i, -1] >= 0.5))
                    R[pre + "|expert_moves"].append(bool(Rt[i, -1] >= 0.5))
            # 6 planner
            pr = props[pn]
            for i in range(Bn):
                d = np.linalg.norm(pr[i, :, :, :2] - ex_[i, None, :, :2], axis=-1).mean(-1)
                R[f"{pn}|{dsn[i]}|planner|min_dist"].append(d.min())
                pd = np.linalg.norm(pr[i, :, None, :, :2] - pr[i, None, :, :, :2], axis=-1).mean(-1)
                R[f"{pn}|{dsn[i]}|planner|spread"].append(pd[np.triu_indices(N_PROP, 1)].mean())
        for i in range(Bn):
            if is_nav[i]:
                m, lm = b["meta"][i], metas[i]
                tok = lm.get("mc") or ((m["log"], m["token"]) if m.get("log") else m["token"])
                E.append((m["token"], tok, np.concatenate([props[pn][i] for pn in pnames]),
                          {k: int(v[i]) for k, v in picks.items()}))
        if (step + 1) % 10 == 0 or step + 1 == len(chunks):
            log(f"eval {sum(len(c) for c in chunks[:step + 1])}/{len(order)} samples, {(time.time() - t0) / 60:.1f} min")
    data.deterministic = was_det
    res = dict(spec="run3b/CLAUDE.md section 6 items 2-7; section 3 last bullet (true agents / decoded agents)",
               code=str(Path(__file__).resolve()), n_nav=int(len(nav_i)), n_lead=int(len(lead_i)), batch=batch, P=P,
               passes=pnames, live_labels=lab is not None, label_errors=int(sum(R["label_errors"])),
               label_error_examples=R["label_error_text"], quoted_run3a=QUOTED)
    if extra:
        res.update(extra)
    # 2
    res["agents"] = {k: v.out() for k, v in tal.items()}
    res["agents_rule"] = dict(conf=float(common().THR), hit_m=float(common().HIT_M), rings=[list(r) for r in partc().RINGS],
                              classes=list(CLS3), layer="last", decoder_logits_max_abs_diff_between_passes=dec_diff)
    # 3-6 per pass
    facts, met, pick, plan = {}, {}, {}, {}
    for pn in pnames:
        F_ = {}
        for d_ in ("NAVSIM", "LEAD"):
            for k in YESNO:
                if f"{pn}|{d_}|{k}|lab" not in R:
                    continue
                lab_, pred, grp, plc = (np.concatenate(R[f"{pn}|{d_}|{k}|{x}"]) for x in ("lab", "pred", "grp", "planner"))
                a_all, a_32 = pairwise(lab_, pred, grp), pairwise(lab_[plc], pred[plc], grp[plc])
                F_[f"{d_} | {k}"] = dict(pairwise_all=a_all[0], scenes_all=a_all[1], pairs_all=a_all[2], pairwise_planner32=a_32[0],
                                         scenes_planner32=a_32[1], pairs_planner32=a_32[2],
                                         base_rate=float(lab_.mean()) if len(lab_) else None, n_paths=int(len(lab_)))
            for k in NUMBERS:
                if R.get(f"{pn}|{d_}|{k}|abs_err"):
                    F_[f"{d_} | {k} (abs err per waypoint)"] = _stats(np.concatenate(R[f"{pn}|{d_}|{k}|abs_err"]))
        facts[pn] = F_
        M_ = {}
        for d_ in ("NAVSIM", "LEAD"):
            e2, e4, st = (np.asarray(R[f"{pn}|{d_}|metres|{x}"]) for x in ("err2", "err4", "stops"))
            if not len(e2):
                continue
            M_[d_] = dict(err_2s=_stats(e2), err_4s=_stats(e4))
            if d_ == "LEAD":
                st = st.astype(bool)
                for nm, m in (("expert stops", st), ("rest", ~st)):
                    M_[f"LEAD {nm}"] = dict(err_2s=_stats(e2[m]), err_4s=_stats(e4[m]))
        met[pn] = M_
        K_ = {}
        for d_ in ("NAVSIM", "LEAD"):
            for which in ("true", "head"):
                gap, top, se, mv = (np.asarray(R[f"{pn}|{d_}|pick_{which}|{x}"]) for x in ("gap", "top", "stops_expert_moves", "expert_moves"))
                if not len(gap):
                    continue
                K_[f"{d_} | expert metres {which}"] = dict(
                    gap_to_best_of_32=_stats(gap), share_true_top=float(top.mean()), n=int(len(gap)),
                    share_stops_while_expert_moves=float(se.sum() / max(mv.sum(), 1)), n_expert_moves=int(mv.sum()))
        pick[pn] = K_
        plan[pn] = {d_: dict(min_dist=_stats(R[f"{pn}|{d_}|planner|min_dist"]), spread=_stats(R[f"{pn}|{d_}|planner|spread"]))
                    for d_ in ("NAVSIM", "LEAD") if R[f"{pn}|{d_}|planner|min_dist"]}
    res["facts"], res["expert_metres_head"], res["pick"], res["planner"] = facts, met, pick, plan
    # 6 EPDMS (NAVSIM)
    if E and epdms_workers is not None:
        te = time.time()
        sc = epdms([e[1] for e in E], np.stack([e[2] for e in E]), workers=epdms_workers)
        ep = {}
        r = np.arange(len(E))
        for k, pn in enumerate(pnames):
            blk = sc[:, N_PROP * k:N_PROP * (k + 1)]
            ep[pn] = {f"pick (expert metres {w})": dict(mean=float(blk[r, [e[3][(pn, w)] for e in E]].mean()), n=len(E))
                      for w in ("true", "head")}
            ep[pn]["best of 32"] = dict(mean=float(blk.max(1).mean()), n=len(E))
            ep[pn]["mean of 32"] = dict(mean=float(blk.mean()), n=len(E))
        ep["minutes"] = round((time.time() - te) / 60, 2)
        res["epdms_navsim"] = ep
    if run3a_same and E:
        try:
            res["run3a_same_navsim_samples"] = run3a_navsim_same_samples([e[0] for e in E], nav_truth)
        except Exception as ex:  # noqa: BLE001
            res["run3a_same_navsim_samples"] = {"error": repr(ex)}
    if RUN3A_LEAD_REF.exists():
        res["run3a_lead_reference"] = json.load(open(RUN3A_LEAD_REF))
    # 7
    tn = np.asarray(R["teach_n"])
    res["lead_teachers"] = dict(frames=int(len(tn)), teachers_4=int((tn == 4).sum()), teachers_1_3=int(((tn > 0) & (tn < 4)).sum()),
                                teachers_0=int((tn == 0).sum()), teacher_to_expert_dist=_stats(R["teach_dist"]))
    res["minutes"] = round((time.time() - t0) / 60, 2)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "x") as f:
            json.dump(res, f, indent=1, default=float)
    return res
