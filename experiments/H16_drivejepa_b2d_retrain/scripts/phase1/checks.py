"""Phase 1 checks 1-3 (b2d_drivejepa/phase1/CLAUDE.md). Each prints numbers and writes logs/check<N>.json.   cd phase1/src; $PY checks.py c1|c2|c3"""
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wiring as W  # noqa: E402

L, P, M = W.setup()
import torch  # noqa: E402

import data as D  # noqa: E402
import train_b2d as T  # noqa: E402

OUT = W.ROOT / "logs"
torch.cuda.set_per_process_memory_fraction(0.15)


def batch(st, n=8, seed=1):
    g = np.random.default_rng(seed)
    ds = st.train_ds
    return torch.stack([ds[int(i)][0] for i in g.choice(len(ds), n, replace=False)]).cuda()


def c1():
    st = T.Setup(hash_files=True)
    plain = W.build_agent().get_submodule(W.ENC_PATH)
    x = batch(st)
    st.learner.eval(); plain.eval()
    with torch.no_grad():
        a32, b32 = P.encode(st.learner, x), P.encode(plain, x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a16, b16 = P.encode(st.learner, x), P.encode(plain, x)
    B0 = max(float(p.abs().max()) for n, p in st.learner.named_parameters() if n.endswith("lora_B"))
    r = dict(sha256=st.hashes, recorded=dict(agent=W.AGENT_SHA, e50=W.E50_SHA), lora_layers=len(L.lora_modules(st.learner)), lora_params=sum(p.numel() for p in L.lora_parameters(st.learner)), lora_B_max=B0,
             tokens=list(a32.shape), fp32_bit_equal=L.bit_equal(a32, b32), fp32_max_abs_diff=float((a32 - b32).abs().max()), bf16_bit_equal=L.bit_equal(a16, b16), bf16_max_abs_diff=float((a16.float() - b16.float()).abs().max()),
             guesser_params=sum(p.numel() for p in st.guesser.parameters()))
    print(json.dumps(r, indent=1)); json.dump(r, open(OUT / "check1.json", "w"), indent=1)


def snap(mod):
    return {n: p.detach().clone() for n, p in mod.named_parameters()}


def changed(a, mod, keep):
    d = dict(mod.named_parameters())
    return sum(int(not torch.equal(d[n].detach(), v)) for n, v in a.items() if keep(n)), sum(1 for n in a if keep(n))


def c2():
    st = T.Setup(hash_files=False)
    is_l = lambda n: ".lora_" in n  # noqa: E731
    is_b = lambda n: ".lora_" not in n  # noqa: E731
    base0, g0, k0 = snap(st.learner), snap(st.guesser), snap(st.key)
    plan = {}
    L.set_lora_trainable(st.learner, False)
    for p in st.guesser.parameters():
        p.requires_grad_(True)
    opt = st.opt_warmup()
    st.learner.train(); st.guesser.train()
    it = iter(st.loader(1000, 0, workers=2))
    for s in range(10):
        T.set_lr(opt, s)
        st.step(opt, next(it)[0].cuda(), "warmup", s, False)
    plan["after_10_warmup"] = dict(base=changed(base0, st.learner, is_b), lora=changed(base0, st.learner, is_l), guesser=changed(g0, st.guesser, lambda n: True), slow_copy_lora=changed(k0, st.key, is_l), slow_copy_base=changed(k0, st.key, is_b))
    g1, l1, k1 = snap(st.guesser), snap(st.learner), snap(st.key)
    L.set_lora_trainable(st.learner, True)
    opt = st.opt_main()
    for s in range(10):
        T.set_lr(opt, s)
        st.step(opt, next(it)[0].cuda(), "main", s, True)
    plan["after_10_main"] = dict(base=changed(base0, st.learner, is_b), lora=changed(l1, st.learner, is_l), lora_A=changed(l1, st.learner, lambda n: n.endswith("lora_A")), lora_B=changed(l1, st.learner, lambda n: n.endswith("lora_B")),
                                 guesser=changed(g1, st.guesser, lambda n: True), slow_copy_lora=changed(k1, st.key, is_l), slow_copy_base=changed(k1, st.key, is_b),
                                 rest_of_agent_requires_grad=[n for n, p in st.agent.named_parameters() if p.requires_grad and ".lora_" not in n][:3])
    print(json.dumps(plan, indent=1)); json.dump(plan, open(OUT / "check2.json", "w"), indent=1)


class Obs:
    """What the runner's Observation gives the policy for one frame."""

    def __init__(self, t_us, img):
        self.timestamp_us, self._img = t_us, img
        self.ego2world = np.eye(4)
        cam = np.eye(4)
        self.camera2ego = {c: cam for c in ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK")}
        Kn = np.array([[571.2592, 0, 400], [0, 571.2592, 225], [0, 0, 1]])
        self.camera2image = {c: Kn for c in self.camera2ego}
        self.speed, self.accel, self.command = 5.0, 0.0, 2

    def image(self, camera):
        return self._img


def c3():
    sys.path.insert(0, "{ops}/worktrees/rerank-native/src")
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    from navsim.agents.drive_jepa_perception_based.drive_jepa_features import DriveJEPAFeatureBuilder
    from navsim.agents.drive_jepa_perception_based.drive_jepa_model import DriveJEPAModel
    from paddock.adapters.models.drivejepa.policy import DriveJepaPolicy
    import build_cache as BC
    sp = json.load(open(W.ROOT / "splits/clips.json"))
    fc = D.FrameCache()
    ds = D.CachedPairs(sp["train"])
    g = np.random.default_rng(3)
    norm = DriveJEPAModel.make_transform(None)
    worst, rows = 0.0, []
    for k in g.choice(len(ds), 20, replace=False):
        c, i = ds.items[int(k)]
        x, _ = ds[int(k)]
        d = BC.clip_dir(c)
        imgs = []
        for j in (i - 1, i):
            bgr = cv2.imread(str(d / fc.frames[c][j]), cv2.IMREAD_COLOR)
            assert bgr.shape == (900, 1600, 3)
            imgs.append(cv2.resize(bgr[:, :, ::-1].copy(), (800, 450), interpolation=cv2.INTER_AREA))
        pol = DriveJepaPolicy(checkpoint="unused")
        assert pol.image_geometry == "native"
        pol.feature_builder = DriveJEPAFeatureBuilder(DriveJEPAConfig())
        pol.reset()
        for t, im in zip((0, 500000), imgs):
            pol._record(Obs(t, im))
        f = pol.feature_builder.compute_features(pol._agent_input(np.eye(4)))
        r = torch.stack([norm(f["camera_feature_2"]), norm(f["camera_feature_1"])], 0)   # the model's order: older, then current
        dmax = float((r - x).abs().max())
        worst = max(worst, dmax)
        rows.append(dict(clip=c, frame=fc.frames[c][i], max_abs_diff=dmax, bit_equal=bool(torch.equal(r, x))))
    fx, fy = 571.2592 * 512 / 800, 571.2592 * 256 / (450 - 56)
    out = dict(pairs=len(rows), all_bit_equal=all(r["bit_equal"] for r in rows), largest_difference=worst, shape=list(x.shape), dtype=str(x.dtype), fx_at_512=round(fx, 1), fy_at_256=round(fy, 1), runner_path="rerank-native policy.py _agent_input + feature builder compute_features, image_geometry native", resize_800x450="cv2.INTER_AREA on the RGB original", rows=rows[:3])
    print(json.dumps(out, indent=1)); json.dump(out, open(OUT / "check3.json", "w"), indent=1)


if __name__ == "__main__":
    t0 = time.time()
    dict(c1=c1, c2=c2, c3=c3)[sys.argv[1]]()
    print("DONE", sys.argv[1], round(time.time() - t0), "s")
