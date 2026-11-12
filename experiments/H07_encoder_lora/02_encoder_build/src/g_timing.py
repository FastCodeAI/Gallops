"""Part G: main-mode step time on the shared GPU.  Nothing is saved.

    ./run.sh src/g_timing.py

Batch 8 / 16 / 32 (stop at the first that does not fit), bf16 autocast,
3 warm-up steps then 10 timed steps.  One step = masks -> answer key (no grad)
-> learner on context -> guesser -> L1 -> backward -> AdamW on LoRA + guesser ->
momentum update of the answer key's LoRA.  The batch is real train pairs read
once and reused, so the timing is GPU-only; the reader is measured in Part B.
"""

from __future__ import annotations

import json
import random
import subprocess
import time

import torch

import common as C
import lora as L
import players as P
import splits as S
from pairs import PairDataset

RUN = C.kit_runner_module()


def others_on_gpu() -> list[dict]:
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    tot = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    apps = [dict(zip(("pid", "mib"), map(int, line.split(",")))) for line in q.strip().splitlines() if line.strip()]
    return {"apps": apps, "used_total_util": tot}


def main() -> None:
    out: dict = {"gpu": torch.cuda.get_device_name(0), "before": others_on_gpu(), "free_mib_start": C.gpu_free_mib()}
    if not C.require_gpu(10000, "part G"):
        out["status"] = "BLOCKED-GPU"
        (C.OUT / "g_timing.json").write_text(json.dumps(out, indent=1))
        return
    data = S.load_splits()
    ckpt = torch.load(C.AGENT_CKPT, map_location="cpu", mmap=True)["state_dict"]
    plain = P.build_vit()
    plain.load_state_dict(P.agent_encoder_state(ckpt), strict=True)
    del ckpt
    plain = plain.cuda()
    e50 = torch.load(C.E50, map_location="cpu", mmap=True)
    guesser = P.build_guesser(e50).cuda()
    del e50

    ds = PairDataset(data["train"][:60], data["scene_paths"])
    idx = random.Random(1).sample(range(len(ds)), 32)
    t0 = time.time()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=32, num_workers=8)
    pool, _, _ = next(iter(loader))
    out["read_32_pairs_s"] = round(time.time() - t0, 1)
    pool = pool.cuda()
    gens = P.mask_generators()
    m = 0.99925
    results = {}
    for bs in (8, 16, 32):
        learner, key = P.make_learner_and_key(plain)
        for p in guesser.parameters():
            p.requires_grad_(True)
        learner.train(); guesser.train(); key.eval()
        opt = torch.optim.AdamW([p for p in learner.parameters() if p.requires_grad] + list(guesser.parameters()),
                                lr=1e-4, weight_decay=0.04)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free_before = C.gpu_free_mib()
        try:
            times = []
            for step in range(13):
                xb = pool[:bs]
                torch.cuda.synchronize()
                t = time.perf_counter()
                masks = P.sample_masks(gens, bs, "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = P.game_loss(learner, key, guesser, xb, masks)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                L.momentum_update(key, learner, m)
                torch.cuda.synchronize()
                if step >= 3:
                    times.append(time.perf_counter() - t)
            results[bs] = {"s_per_step_mean": round(sum(times) / len(times), 3),
                           "s_per_step_min": round(min(times), 3), "s_per_step_max": round(max(times), 3),
                           "pairs_per_s": round(bs / (sum(times) / len(times)), 2),
                           "max_memory_allocated_mib": torch.cuda.max_memory_allocated() // (1 << 20),
                           "max_memory_reserved_mib": torch.cuda.max_memory_reserved() // (1 << 20),
                           "free_mib_before": free_before, "free_mib_after": C.gpu_free_mib(),
                           "last_loss": float(loss), "others_during": others_on_gpu()}
            print(bs, results[bs], flush=True)
        except torch.cuda.OutOfMemoryError as exc:
            results[bs] = {"status": "does not fit", "error": str(exc).splitlines()[0],
                           "free_mib_before": free_before}
            print(bs, results[bs], flush=True)
            break
        finally:
            del learner, key, opt
            torch.cuda.empty_cache()
    out["timing"] = results
    out["after"] = others_on_gpu()

    # wall time per pass over all train pairs
    n_pairs = 93784
    reader = json.loads((C.OUT / "pairs_report.json").read_text())["pairs_per_second_by_workers"]
    est = {}
    for bs, r in results.items():
        if "pairs_per_s" not in r:
            continue
        gpu_h = n_pairs / r["pairs_per_s"] / 3600
        est[bs] = {"gpu_only_h": round(gpu_h, 2),
                   **{f"with_{w}_workers_from_NAS_h": round(max(gpu_h, n_pairs / float(v) / 3600), 1) for w, v in reader.items()}}
    out["pass_estimate"] = {"train_pairs": n_pairs, "reader_pairs_per_s": reader, "by_batch": est}
    (C.OUT / "g_timing.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["pass_estimate"], indent=1))


if __name__ == "__main__":
    main()
