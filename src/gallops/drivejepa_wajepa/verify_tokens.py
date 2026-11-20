"""Stage verify-tokens: the token cache holds what the trainer would compute live.

For the dumped logs: the scenes match NAVSIM's navtrain filter (none missing, none extra, no .tmp left);
every scene has its 3 files; tokens are float16 (4, 512, 1024) on the CPU; and for K scenes drawn with seed
0, the light features, the target and the tokens recomputed live (the same builders, a live encode_wajepa)
equal the cached ones -- features and target exactly, tokens to a cosine of at least 0.9999 (bf16 autocast).
Exits non-zero on any failure. Run 2's trial dump (3 logs, 86 scenes) passed it with token cosines >=
0.999999.

Runs in Drive-JEPA's interpreter with cwd = the run's tree navsim_v2/, WAJEPA_CODE and WAJEPA_WEIGHTS set.
Ported from paddock-ops wajepa-runs/verify_dump.py (arguments only).

    python -m gallops.drivejepa_wajepa.verify_tokens --navtrain <navtrain.yaml> --data <navsim root>
        --current1 <current_split_1> --cache <cache> --k 16
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--navtrain", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--current1", required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--k", type=int, default=16, help="scenes recomputed live")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.getcwd())

    import torch
    from hydra.utils import instantiate
    from navsim.agents.drive_jepa_perception_based.bevformer.wajepa import build_wajepa_encoder, encode_wajepa
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    from navsim.agents.drive_jepa_perception_based.drive_jepa_features import (
        DriveJEPA4VFeatureBuilder,
        DriveJEPATargetBuilder,
    )
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.training.dataset import load_feature_target_from_pickle
    from omegaconf import OmegaConf

    from gallops.drivejepa_wajepa.dump_tokens import NAMES, SENSOR_CAMERAS, OverlayRoot

    dump = a.cache
    cfg = OmegaConf.load(a.navtrain)
    logs = sorted(os.listdir(dump))
    sf = instantiate(cfg)
    sf.log_names = logs
    loader = SceneLoader(
        data_path=Path(a.data) / "navsim_logs/trainval",
        sensor_blobs_path=OverlayRoot(Path(a.data) / "sensor_blobs/trainval", a.current1),
        synthetic_scenes_path=None,
        scene_filter=sf,
        sensor_config=SensorConfig(**SENSOR_CAMERAS),
    )
    want = set(loader.tokens)
    have = {
        t
        for log in logs
        for t in os.listdir(dump / log)
        if all((dump / log / t / n).is_file() for n in NAMES)
    }
    tmp = [str(p) for p in dump.rglob("*.tmp")]
    print(
        f"logs {len(logs)} | navtrain-filter scenes {len(want)} | complete dumped {len(have)} | "
        f"missing {len(want - have)} | extra {len(have - want)} | .tmp left {len(tmp)}"
    )
    c = DriveJEPAConfig()
    light, target = DriveJEPA4VFeatureBuilder(c), DriveJEPATargetBuilder(c)
    enc, _ = build_wajepa_encoder("cuda")
    bad = []
    for tok in random.Random(0).sample(sorted(have), min(a.k, len(have))):
        scene = loader.get_scene_from_token(tok)
        ai = scene.get_agent_input()
        d = dump / scene.scene_metadata.log_name / tok
        lf, tg, tk = (load_feature_target_from_pickle(d / n) for n in NAMES)
        t = tk["wajepa_tokens"]
        ok_t = t.dtype == torch.float16 and tuple(t.shape) == (4, 512, 1024) and not t.is_cuda
        live = encode_wajepa(ai, enc)
        cos = torch.nn.functional.cosine_similarity(t.float(), live.float(), dim=-1).min().item()
        lref, tref = light.compute_features(ai), target.compute_targets(scene)
        ok_l = all(torch.equal(lf[k], lref[k]) for k in lref) and set(lf) == set(lref)
        ok_g = torch.equal(tg["trajectory"], tref["trajectory"]) and tg["token"] == tok
        print(
            f"{tok}: tokens ok {ok_t} cos_min vs live {cos:.6f} | light identical {ok_l} | "
            f"target identical {ok_g}"
        )
        if not (ok_t and cos >= 0.9999 and ok_l and ok_g):
            bad.append(tok)
    ok = not bad and want == have and not tmp
    print("VERIFY", "PASS" if ok else f"FAIL {bad}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
