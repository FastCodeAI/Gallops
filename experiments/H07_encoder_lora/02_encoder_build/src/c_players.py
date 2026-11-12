"""Part C + D.2 on CPU: who the players are, does the guesser run on our grid,
what the masks hide, how many LoRA parameters.

    ./run.sh src/c_players.py          # writes out/c_players.json, out/masks/*.png
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

import common as C
import lora as L
import players as P
import splits as S


def main() -> None:
    out: dict = {}
    t0 = time.time()
    out["agent_ckpt_sha256"] = C.sha256_file(C.AGENT_CKPT)
    out["e50_sha256"] = C.sha256_file(C.E50)
    assert out["agent_ckpt_sha256"] == C.AGENT_CKPT_SHA256
    assert out["e50_sha256"] == C.E50_SHA256
    print(f"file hashes ok ({time.time() - t0:.0f}s)")

    ckpt = torch.load(C.AGENT_CKPT, map_location="cpu", mmap=True)["state_dict"]
    e50 = torch.load(C.E50, map_location="cpu", mmap=True)

    # ---- C.1 learner = the agent's fine-tuned encoder ------------------------
    agent_enc = P.agent_encoder_state(ckpt)
    enc, tgt = P.e50_state(e50, "encoder"), P.e50_state(e50, "target_encoder")
    plain = P.build_vit()
    res = plain.load_state_dict(agent_enc, strict=True)
    learner_sd = plain.state_dict()
    out["C1"] = {
        "tensors": len(learner_sd),
        "params": sum(v.numel() for v in learner_sd.values()),
        "hash_learner_state": P.state_hash(learner_sd),
        "hash_agent_ckpt_encoder": P.state_hash(agent_enc),
        "hash_e50_encoder": P.state_hash(enc),
        "hash_e50_target_encoder": P.state_hash(tgt),
        "rel_diff_vs_e50_target_encoder": P.relative_diff(agent_enc, tgt),
        "rel_diff_vs_e50_encoder": P.relative_diff(agent_enc, enc),
        "keys_match_e50": sorted(agent_enc) == sorted(tgt),
    }
    c1 = out["C1"]
    c1["learner_is_agent_encoder"] = c1["hash_learner_state"] == c1["hash_agent_ckpt_encoder"]
    c1["learner_differs_from_e50"] = c1["hash_learner_state"] not in (c1["hash_e50_encoder"], c1["hash_e50_target_encoder"])
    print(json.dumps(c1, indent=1))

    # ---- C.2 guesser on our grid ---------------------------------------------
    guesser = P.build_guesser(e50)
    out["C2"] = {"predictor_params": sum(p.numel() for p in guesser.parameters()),
                 "num_patches": guesser.num_patches, "grid_depth": guesser.grid_depth,
                 "mask_tokens": guesser.num_mask_tokens}
    data = S.load_splits()
    from pairs import build_pair

    sel0 = data["selection"][0]
    pair = build_pair(data["scene_paths"][sel0], 40)[None]  # (1, 2, 3, 256, 512)
    gens = P.mask_generators()
    torch.manual_seed(0)
    masks = P.sample_masks(gens, 1, "cpu")
    plain.eval(); guesser.eval()
    with torch.no_grad():
        full = P.encode(plain, pair)
        enc_m, pred_m = masks[0]
        z = P.encode(plain, pair, masks=[enc_m])
        g = guesser(z, [enc_m], [pred_m], mask_index=P.MASK_TOKEN_INDEX)
        # the stock forward (no grid passed to the blocks), for the record
        g_stock = P.VisionTransformerPredictor.forward(guesser, z, [enc_m], [pred_m], mask_index=P.MASK_TOKEN_INDEX)
    attn = guesser.predictor_blocks[0].attn
    ids = torch.tensor([300])
    stock_pos = [int(t) for t in attn.separate_positions(ids)]
    grid_pos = [int(t) for t in attn.separate_positions(ids, P.GRID_H, P.GRID_W)]
    out["C2"].update({
        "encoder_full_tokens": list(full.shape),
        "context_tokens": list(z.shape), "hidden_tokens": int(pred_m.shape[1]),
        "guesser_out": list(g.shape),
        "stock_vs_grid_forward_max_abs_diff": float((g - g_stock).abs().max()),
        "token300_positions_stock_(frame,row,col)": stock_pos,
        "token300_positions_with_grid": grid_pos,
        "pair_from": [sel0, 40],
    })
    print(json.dumps(out["C2"], indent=1))

    # ---- C.3 masks: 1,000 draws per config -----------------------------------
    torch.manual_seed(1)
    fr = {i: [] for i in range(len(gens))}
    union = []
    for _ in range(1000):
        ms = P.sample_masks(gens, 1, "cpu")
        for i, (_, pm) in enumerate(ms):
            fr[i].append(pm.shape[1] / 512)
        union.append(len(set().union(*(set(pm[0].tolist()) for _, pm in ms))) / 512)

    def stats(v):
        v = np.asarray(v)
        return {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
                "p5": round(float(np.percentile(v, 5)), 4), "p50": round(float(np.median(v)), 4),
                "p95": round(float(np.percentile(v, 95)), 4), "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}

    out["C3"] = {"configs": [{k: list(v) if isinstance(v, tuple) else v for k, v in c.items()} for c in P.MASK_CFGS],
                 "hidden_fraction": {f"cfg{i}": stats(v) for i, v in fr.items()},
                 "union_of_both_hidden_fraction": stats(union)}
    # batch truncation: upstream trims every sample to the batch minimum (multiseq_multiblock3d.py:212-213)
    torch.manual_seed(2)
    trunc = {i: [] for i in range(len(gens))}
    for _ in range(100):
        for i, (_, pm) in enumerate(P.sample_masks(gens, 16, "cpu")):
            trunc[i].append(pm.shape[1] / 512)
    out["C3"]["hidden_fraction_after_batch16_trim"] = {f"cfg{i}": stats(v) for i, v in trunc.items()}
    print(json.dumps(out["C3"], indent=1))
    save_mask_images(gens)

    # ---- D.2 LoRA parameter count ---------------------------------------------
    learner, _ = P.make_learner_and_key(plain)
    out["D2"] = {"rank": L.RANK, "alpha": L.ALPHA, **L.count(learner),
                 "trainable_in_learner": sum(p.numel() for p in learner.parameters() if p.requires_grad)}
    print(json.dumps(out["D2"], indent=1))
    (C.OUT / "c_players.json").write_text(json.dumps(out, indent=1) + "\n")


def save_mask_images(gens) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = C.OUT / "masks"
    folder.mkdir(exist_ok=True)
    torch.manual_seed(3)
    for k in range(4):
        ms = P.sample_masks(gens, 1, "cpu")
        fig, axes = plt.subplots(1, len(ms), figsize=(8, 2.4))
        for i, (_, pm) in enumerate(ms):
            grid = np.zeros(512)
            grid[pm[0].numpy()] = 1
            axes[i].imshow(grid.reshape(16, 32), cmap="Greys", vmin=0, vmax=1)
            axes[i].set_title(f"cfg{i}: {P.MASK_CFGS[i]['num_blocks']} blocks, hidden {grid.mean():.0%}", fontsize=8)
            axes[i].set_xticks([]); axes[i].set_yticks([])
        fig.suptitle("16x32 token grid, black = hidden", fontsize=9)
        fig.tight_layout()
        fig.savefig(folder / f"mask_example_{k}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
