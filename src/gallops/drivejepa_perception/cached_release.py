"""R-tilde: the release through the cached path every trained Part B arm runs, adapters at zero (B-0, Q6).

    python -m gallops.drivejepa_perception.cached_release --tokens {out}/tokens-navtest-v1 \
        --labels {out}/labels-navtest-v1 --out {out}/rtilde-navtest-v1

The refiner runs from the token pass's cached fp16 tokens, batched with `batch_fix`, in fp32
(`train_refiner.build` and `forward`, as the first pass's refiner arms ran it), with the adapter whose
output layer starts at zero; the release's own scorer head (`drivejepa_b2d.head.Head`, its tensors from
the checkpoint) reads the pass-4 queries max-pooled over each proposal's 8 poses, and the pick is the
argmax of the sigmoid of its last logit, as the vendor picks. R-tilde is where every trained arm starts,
so F - C never depends on R-tilde - R; R-tilde - R is measured on every navtest scene (B0-G10).

Before any row is written: the adapter's output is exactly zero, so this is the release's refiner and
nothing else. After: proposals, logits, scores and pick, with their gap to the token pass in the manifest.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

BATCH = 64


def run(cfg, tokens_dir: Path, labels_dir: Path, out: Path, batch: int) -> dict:
    from gallops.drivejepa_b2d.head import Head, head_tensors_from_checkpoint
    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env, train_refiner

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    started, sha = time.strftime("%Y-%m-%dT%H:%M:%S"), git_sha()
    entered = env.enter(cfg)
    tmanifest = json.loads((Path(tokens_dir) / "manifest.json").read_text())
    if Path(tmanifest["labels"]).resolve() != Path(labels_dir).resolve():
        raise SystemExit(f"{tokens_dir} was not built from {labels_dir}")
    n = tmanifest["rows"]
    data = {
        "tokens": np.load(Path(tokens_dir) / "tokens.npy", mmap_mode="r"),
        "lidar2img": np.load(Path(tokens_dir) / "lidar2img.npy"),
        "img_shape": np.load(Path(tokens_dir) / "img_shape.npy"),
        "ego": np.load(Path(tokens_dir) / "ego_status.npy")[:, -1].astype(np.float32),
        "spatial": torch.tensor([[16, 32]], dtype=torch.long),
        "level": torch.tensor([0], dtype=torch.long),
    }
    device = torch.device("cuda")
    _, model, adapter = train_refiner.build(entered, "frozen", agents=False)
    model, adapter = model.to(device).eval(), adapter.to(device).eval()
    head = Head()
    head.load_state_dict(head_tensors_from_checkpoint(Path(entered["release"])))
    head = head.to(device).eval()

    out_arrays = {
        "proposals": np.zeros((n, 32, 8, 3), np.float32),
        "logits": np.zeros((n, 32, 6), np.float32),
        "scores": np.zeros((n, 32), np.float32),
        "pick": np.zeros(n, np.int16),
    }
    adapter_max = 0.0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, batch):
            rows = np.arange(i, min(n, i + batch))
            props, bev = train_refiner.forward(model, data, rows, device)
            if i == 0:
                # The adapter's contribution on real queries: exactly zero, or this is not the release.
                queries = bev.reshape(len(rows), 32 * 8, -1)
                adapter_max = float(adapter(props[-1], queries).abs().max())
                if adapter_max != 0.0:
                    raise SystemExit(f"the untrained adapter adds up to {adapter_max:g}: not the release")
            pooled = bev.reshape(len(rows), 32, 8, -1).amax(2)
            logits = head(pooled)
            scores = torch.sigmoid(logits)[..., -1]
            out_arrays["proposals"][rows] = props[-1].float().cpu().numpy()
            out_arrays["logits"][rows] = logits.float().cpu().numpy()
            out_arrays["scores"][rows] = scores.float().cpu().numpy()
            out_arrays["pick"][rows] = scores.argmax(1).cpu().numpy()
            if (i // batch) % 40 == 0:
                print(f"{rows[-1] + 1}/{n} rows, {time.time() - t0:.0f} s", flush=True)

    release = {k: np.load(Path(tokens_dir) / f"{k}.npy") for k in ("proposals", "pick")}
    gap = np.abs(out_arrays["proposals"] - release["proposals"]).max(axis=(2, 3))
    staging = out.with_name(out.name + ".staging")
    staging.mkdir(parents=True)
    for key, value in out_arrays.items():
        np.save(staging / f"{key}.npy", value)
    manifest = {
        "pass": "rtilde",
        "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gallops_sha": sha,
        "entered": entered,
        "tokens": str(tokens_dir),
        "labels": str(labels_dir),
        "rows": n,
        "batch": batch,
        "path": "train_refiner.build('frozen', agents=False) + forward (fp32 from fp16 tokens, batch_fix); "
        "release head on amax-pooled pass-4 queries; argmax of sigmoid(last logit)",
        "adapter_output_max": adapter_max,
        "against_token_pass": {
            "proposals_max_m": float(gap.max()),
            "proposals_p99_m": float(np.quantile(gap.max(axis=1), 0.99)),
            "picks_equal": float((out_arrays["pick"] == release["pick"]).mean()),
        },
        "seconds": round(time.time() - t0),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    staging.rename(out)
    print(json.dumps({k: v for k, v in manifest.items() if k != "entered"}, indent=1))
    return manifest


def main(argv=None) -> None:
    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.cached_release")
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=BATCH)
    args = parser.parse_args(argv)
    run(SiteConfig.load(), Path(args.tokens), Path(args.labels), Path(args.out), args.batch)


if __name__ == "__main__":
    main()
