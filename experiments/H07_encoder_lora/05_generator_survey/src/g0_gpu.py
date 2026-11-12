"""Read-only GPU checks for the generator-on-KIT plan.  No training, nothing saved
except JSON / small pickles in ../out.

    ./run.sh src/g0_gpu.py parts     # C: parameter counts, checkpoint metadata; E: determinism + cache size
    ./run.sh src/g0_gpu.py menu      # G: drift check, then menu metrics 1-4 on the 103 picking moments
    ./run.sh src/g0_gpu.py memory    # F: one-step peak memory, LoRA + generator + hide-and-guess

Only the 50 picking scenes (103 moments) and train frames are read.  Val, overlap
and test are never read.  Weights: the released agent checkpoint (sha256-checked)
and the s2 pass-3 LoRA state, loaded exactly as s3 close.py does.
"""

from __future__ import annotations

import copy
import json
import math
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import common as C          # s1
import drift as D           # s1
import evalsets as E        # s2
import f_final as FF        # s2 (merge)
import lora as L            # s1
import players as P         # s1
import splits as S          # s1
from s2common import CKPT, OUT as S2OUT, wait_for_gpu

RUN = C.kit_runner_module()
ADP = C.kit_adapter_module()
OUT = Path(__file__).resolve().parent.parent / "out"
GEN_PREFIXES = ("hist_encoding.", "init_feature.", "_trajectory_head.")


def wj(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=1, default=str) + "\n")
    print(json.dumps(obj, indent=1, default=str))


def build(pass3: bool, merged: bool = False):
    assert C.sha256_file(C.AGENT_CKPT) == C.AGENT_CKPT_SHA256
    agent = RUN.build_agent(C.AGENT_CKPT, "drive_jepa_perception_based_agent")
    for p in agent.parameters():
        p.requires_grad_(False)
    if not pass3:
        return agent
    L.wrap(agent.get_submodule(E.ENC_PATH))
    sd = torch.load(CKPT / "pass_03" / "state.pt", map_location="cpu", weights_only=False)["lora"]
    enc = agent.get_submodule(E.ENC_PATH)
    params = dict(enc.named_parameters())
    assert set(sd) == {n for n in params if ".lora_" in n}
    with torch.no_grad():
        for n, t in sd.items():
            params[n].copy_(t.to(params[n].device, params[n].dtype))
    agent.eval()
    return FF.merge(agent) if merged else agent


def count(mod) -> int:
    return sum(p.numel() for p in mod.parameters())


# --------------------------------------------------------------------------- #
# C + E
# --------------------------------------------------------------------------- #


def parts() -> None:
    out = {"gpu_wait_s": wait_for_gpu(8192, "parts")}
    agent = build(pass3=False)
    m = agent._pad_model
    named = dict(m.named_parameters())
    uniq = {id(p) for p in m.parameters()}
    head = m._trajectory_head
    ref = head[0]
    br = ref.Bev_refiner
    enc_layer = br.bev_decoder.layers[0] if hasattr(br.bev_decoder, "layers") else None
    c = {
        "encoder_vit (_backbone.img_backbone)": count(m._backbone.img_backbone),
        "projector (_backbone.projector, 1024->256)": count(m._backbone.projector),
        "generator.hist_encoding (ego status 11->256)": count(m.hist_encoding),
        "generator.init_feature (learned queries, 32x8 rows x 256)": count(m.init_feature),
        "generator.refiner.traj_decoder (MLP 256->1024->3, output head)": count(ref.traj_decoder),
        "generator.refiner.Bev_refiner.positional_encoding": count(br.positional_encoding),
        "generator.refiner.Bev_refiner.bev_decoder (1 BEVFormer layer)": count(br.bev_decoder),
        "generator total (unique)": count(m.hist_encoding) + count(m.init_feature) + count(ref),
        "trajectory_head modules / distinct objects": [len(head), len({id(h) for h in head})],
        "scorer": count(m.scorer),
        "scorer children": {n: count(ch) for n, ch in m.scorer.named_children()},
        "unique total": sum(p.numel() for p in m.parameters()),
        "named tensors (state_dict keys under _trajectory_head)": sum(1 for k in m.state_dict() if k.startswith("_trajectory_head.")),
    }
    if enc_layer is not None:
        c["bev_decoder layer children"] = {n: count(ch) for n, ch in enc_layer.named_children()}
    out["C_params"] = c
    out["init_feature_shape"] = list(m.init_feature.weight.shape)
    out["projector_in_backbone"] = "_backbone.projector.weight" in named
    out["optimizer_groups_upstream"] = {
        "backbone_group (0.1*lr) contains projector": any(p is m._backbone.projector.weight for p in m._backbone.parameters()),
    }

    # checkpoint metadata (what the release says about its own training)
    ck = torch.load(C.AGENT_CKPT, map_location="cpu", mmap=True, weights_only=False)
    meta = {k: (v if isinstance(v, (int, float, str)) else type(v).__name__) for k, v in ck.items() if k != "state_dict"}
    if "optimizer_states" in ck:
        og = ck["optimizer_states"][0]["param_groups"]
        meta["optimizer_param_groups"] = [{k: v for k, v in g.items() if k != "params"} | {"n_tensors": len(g["params"])} for g in og]
    for k in ("hyper_parameters", "hparams_name", "loops", "callbacks"):
        if k in ck:
            meta[k] = str(ck[k])[:600]
    out["ckpt_meta"] = meta
    del ck

    # E: determinism of the frozen encoder + projector (pass-3 merged, eval mode)
    del agent
    torch.cuda.empty_cache()
    ag = build(pass3=True, merged=True)
    ag.eval()
    bk = ag._pad_model._backbone
    out["E_modules_with_randomness"] = sorted({
        f"{n}:{type(mod).__name__}(p={getattr(mod, 'p', getattr(mod, 'drop_prob', None))})"
        for n, mod in ag._pad_model.named_modules()
        if ("Drop" in type(mod).__name__ and (getattr(mod, "p", 0) or getattr(mod, "drop_prob", 0) or 0) > 0)
        or "GridMask" in type(mod).__name__})
    out["E_grid_mask_training_flag_in_eval"] = bool(bk.grid_mask.training)
    sel = E.load_selection_features()
    out["selection_feature_keys"] = {k: [list(v.shape), str(v.dtype)] for k, v in sel[0]["features"].items()}
    out["selection_entry_keys"] = sorted(sel[0].keys())

    def enc_out(idx, bs):
        res = []
        for i in range(0, len(idx), bs):
            fs = [sel[j]["features"] for j in idx[i:i + bs]]
            c1 = torch.stack([f["camera_feature_1"] for f in fs]).cuda()
            c2 = torch.stack([f["camera_feature_2"] for f in fs]).cuda()
            tr = ag._pad_model.transform
            x = torch.cat([tr(c2)[:, None], tr(c1)[:, None]], dim=1)
            with torch.no_grad():
                feat, *_ = bk(x, img_metas={})
                raw = bk.img_backbone(torch.einsum("bnchw->bcnhw", x))
            res.append((feat.permute(2, 0, 1, 3).reshape(len(fs), -1, feat.shape[-1]).cpu(), raw.cpu()))
        return torch.cat([r[0] for r in res]), torch.cat([r[1] for r in res])

    idx = list(range(16))
    a1, r1 = enc_out(idx, 1)
    a2, r2 = enc_out(idx, 1)
    b8, rb = enc_out(idx, 8)
    out["E_determinism"] = {
        "moments": len(idx),
        "projector_out_shape_per_moment": list(a1.shape[1:]),
        "encoder_out_shape_per_moment": list(r1.shape[1:]),
        "repeat_batch1_bit_identical": bool(torch.equal(a1, a2) and torch.equal(r1, r2)),
        "batch8_vs_batch1_max_abs_diff_projector": float((a1 - b8).abs().max()),
        "batch8_vs_batch1_max_abs_diff_encoder": float((r1 - rb).abs().max()),
        "projector_out_abs_max": float(a1.abs().max()),
    }
    wj("ce_parts.json", out)


# --------------------------------------------------------------------------- #
# G
# --------------------------------------------------------------------------- #


def menu_metrics(dump: dict, human: dict, base_best: dict | None = None) -> tuple[dict, dict]:
    per = {}
    for k in sorted(dump):
        p = np.asarray(dump[k]["proposals"], dtype=np.float64)[:, :6, :2]   # 32 x 6 (0.5..3 s) x xy
        h = np.asarray(human[k], dtype=np.float64)[:6, :2]
        err = np.linalg.norm(p - h[None], axis=-1)                         # 32 x 6
        ade = err.mean(1)
        i = int(ade.argmin())
        pair = np.linalg.norm(p[:, None] - p[None], axis=-1).mean(-1)     # 32 x 32, mean over 6 waypoints
        iu = np.triu_indices(32, 1)
        per[k] = {"best_ade": float(ade[i]), "best_ade_fde3": float(err[i, 5]), "min_fde3": float(err[:, 5].min()),
                  "n_below_1m": int((ade < 1).sum()), "n_below_2m": int((ade < 2).sum()),
                  "variety_m": float(pair[iu].mean()), "chosen_ade": float(ade[int(dump[k]["chosen"])])}
    v = list(per.values())
    f = lambda key: {"mean": float(np.mean([r[key] for r in v])), "median": float(np.median([r[key] for r in v]))}
    agg = {"n": len(v), **{key: f(key) for key in v[0]}}
    if base_best is not None:
        n_w = math.ceil(0.2 * len(base_best))
        worst = sorted(base_best, key=lambda k: -base_best[k])[:n_w]
        agg["worst20"] = {"n": n_w, "rule": "top ceil(0.2 n) moments by the original model's closest-of-32 ADE",
                          "best_ade": float(np.mean([per[k]["best_ade"] for k in worst])),
                          "best_ade_fde3": float(np.mean([per[k]["best_ade_fde3"] for k in worst])),
                          "keys": [list(k) for k in worst]}
    return agg, per


def bit_eq(a: dict, b: dict) -> int:
    return sum(all(np.asarray(a[k][f]).tobytes() == np.asarray(b[k][f]).tobytes()
                   for f in ("proposals", "pred_logit", "pdm_score", "trajectory")) and a[k]["chosen"] == b[k]["chosen"]
               for k in a)


def menu() -> None:
    out = {"gpu_wait_s": wait_for_gpu(8192, "menu")}
    t0 = time.time()
    data = S.load_splits()
    sel = E.load_selection_features()
    human = {(e["scene_id"], e["keyframe"]): e["future_poses"] for e in sel}
    assert len(sel) == 103 and {e["scene_id"] for e in sel} <= set(data["selection"])
    with open(S2OUT / "c_selection_zero_dump.pkl", "rb") as fh:
        z_saved = pickle.load(fh)
    with open(S2OUT / "e_dump_pass_03.pkl", "rb") as fh:
        p3_saved = pickle.load(fh)

    plain = build(pass3=False)
    z_now, _ = E.selection_eval(plain, sel, RUN)
    p3 = build(pass3=True)
    p3_now, rows3 = E.selection_eval(p3, sel, RUN)
    p3m = FF.merge(p3)
    p3m_now, _ = E.selection_eval(p3m, sel, RUN)

    # The evaluation wrapper's own path, rebuilt from the KIT files (s1 f_baseline / s2 f_final:
    # adapter.build_sample -> feature builders -> agent.forward; compute_trajectory for the pick).
    n_future = plain._trajectory_sampling.num_poses

    def prepare(sid):
        ad = ADP.KITScenesSceneAdapter(Path(data["scene_paths"][sid]))
        kfs = ad.valid_keyframes(num_future_frames=10, stride=90, first_frame=40)
        return sid, [(kf, ad.build_sample(kf, num_future_frames=n_future, command_mode="gt")) for kf in kfs]

    import f_baseline as FB
    feat_eq, human_eq, traj_eq = 0, 0, 0
    w_z, w_p3 = {}, {}
    cached = {(e["scene_id"], e["keyframe"]): e for e in sel}
    builders = plain.get_feature_builders()
    with ThreadPoolExecutor(8) as pool:
        for sid, samples in pool.map(prepare, sorted(data["selection"])):
            for kf, smp in samples:
                k = (sid, kf)
                f = {}
                for b in builders:
                    f.update(b.compute_features(smp.agent_input))
                cf = cached[k]["features"]
                feat_eq += int(f.keys() == cf.keys() and all(torch.equal(f[x], cf[x]) for x in f))
                human_eq += int(np.array_equal(smp.future_poses, cached[k]["future_poses"]))
                w_z[k] = FB.forward_dump(plain, smp.agent_input)
                w_p3[k] = FB.forward_dump(p3, smp.agent_input)
                traj_eq += int(np.array_equal(plain.compute_trajectory(smp.agent_input).poses, w_z[k]["trajectory"]))
    out["drift_check"] = {
        "moments": len(sel),
        "wrapper_features_bit_equal_to_cached": feat_eq,
        "wrapper_human_path_bit_equal_to_cached": human_eq,
        "wrapper_compute_trajectory_eq_dump_pick (original)": traj_eq,
        "original: wrapper dump == cached-feature dump (bit)": bit_eq(w_z, z_now),
        "original: cached-feature dump now == s2 saved zero dump (bit)": bit_eq(z_now, z_saved),
        "pass3 unmerged: wrapper dump == cached-feature dump (bit)": bit_eq(w_p3, p3_now),
        "pass3 unmerged: now == s2 saved pass-3 dump (bit)": bit_eq(p3_now, p3_saved),
        "pass3 unmerged now vs saved": D.drift(p3_saved, p3_now)["summary"],
        "pass3 merged vs unmerged": D.drift(p3_now, p3m_now)["summary"],
        "pass3 ADE mean now (s2 recorded 1.8732948781716308)": float(np.mean([r["ade"] for r in rows3])),
    }

    z_agg, z_per = menu_metrics(z_now, human)
    base_best = {k: z_per[k]["best_ade"] for k in z_per}
    z_agg, z_per = menu_metrics(z_now, human, base_best)
    p_agg, p_per = menu_metrics(p3_now, human, base_best)
    m_agg, _ = menu_metrics(p3m_now, human, base_best)
    out["menu"] = {"original": z_agg, "pass3_unmerged": p_agg, "pass3_merged": m_agg}
    out["wall_s"] = round(time.time() - t0, 1)
    with open(OUT / "g_menu_per_moment.pkl", "wb") as fh:
        pickle.dump({"original": z_per, "pass3": p_per}, fh)
    wj("g_menu.json", out)


# --------------------------------------------------------------------------- #
# F
# --------------------------------------------------------------------------- #


def memory() -> None:
    import cache as K
    import masks2 as M
    out = {"gpu_wait_s": wait_for_gpu(20000, "memory"), "free_mib_before": int(C.gpu_free_mib())}
    data = S.load_splits()
    agent = build(pass3=True)                     # LoRA modules present, pass-3 values
    m = agent._pad_model
    enc = m._backbone.img_backbone
    for p in L.lora_parameters(enc):
        p.requires_grad_(True)
    gen = [p for n, p in m.named_parameters() if n.startswith(GEN_PREFIXES)]
    for p in gen:
        p.requires_grad_(True)
    key = copy.deepcopy(enc).eval()
    for p in key.parameters():
        p.requires_grad_(False)
    guesser = P.build_guesser(torch.load(C.E50, map_location="cpu", mmap=True)).cuda().train()
    trainable = L.lora_parameters(enc) + gen + [p for p in guesser.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-6)
    out["trainable"] = {"lora": sum(p.numel() for p in L.lora_parameters(enc)), "generator": sum(p.numel() for p in gen),
                        "guesser": sum(p.numel() for p in guesser.parameters() if p.requires_grad)}
    sel = E.load_selection_features()             # picking-set features as stand-in planner batches
    fc = K.FrameCache()
    nf = json.loads((K.CACHE / "index.json").read_text())["scene_frames"]
    items = [(s, t) for s in sorted(data["train"])[:40] for t in K.pair_ends(nf[s])][:256]
    gens = M.make()
    cfg = agent._config

    def one(bs: int, steps: int = 3) -> dict:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for s in range(steps):
            fs = [sel[(s * bs + j) % len(sel)] for j in range(bs)]
            feats = {k: torch.stack([e["features"][k] for e in fs]).cuda() for k in fs[0]["features"]}
            target = torch.stack([torch.as_tensor(e["future_poses"], dtype=torch.float32) for e in fs]).cuda()
            x = torch.stack([fc.pair(*items[(s * bs + j) % len(items)]) for j in range(bs)]).cuda()
            masks = M.sample(gens, bs, M.step_rng(1, 9, s), "cuda")
            torch.cuda.synchronize(); t = time.time()
            m.train(); m.scorer.eval()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = m(feats)
                jl, _ = P.game_loss(enc, key, guesser, x, masks)
            pl = [q.float() for q in pred["proposal_list"]]
            gl = agent.trajectory_loss_anchors(pl, target, cfg, [np.zeros(0, dtype=np.int64)] * bs)[0]
            loss = gl + jl
            loss.backward()
            g = {"lora_grad_nonzero_layers": sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0)) for p in L.lora_parameters(enc)),
                 "generator_grad_nonzero_tensors": sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0)) for p in gen),
                 "scorer_grads": sum(int(p.grad is not None) for p in m.scorer.parameters()),
                 "projector_grads": sum(int(p.grad is not None) for p in m._backbone.projector.parameters())}
            opt.zero_grad(set_to_none=True)       # no optimiser step: weights are not changed
            torch.cuda.synchronize(); times.append(time.time() - t)
        return {"batch": bs, "peak_allocated_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                "s_per_step": [round(v, 3) for v in times], "gen_loss": float(gl), "jepa_loss": float(jl), **g}

    out["steps"] = []
    for bs in (8, 16, 32):
        try:
            out["steps"].append(one(bs))
        except torch.cuda.OutOfMemoryError as exc:
            out["steps"].append({"batch": bs, "OOM": str(exc)[:200]})
            break
        print(out["steps"][-1], flush=True)
    out["free_mib_after"] = int(C.gpu_free_mib())
    wj("f_memory.json", out)


if __name__ == "__main__":
    {"parts": parts, "menu": menu, "memory": memory}[sys.argv[1]]()
