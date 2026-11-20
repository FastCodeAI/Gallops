"""Stage agent-targets (agent decoder step 1, 22 Oct): agent_targets.gz next to the tokens, for every sample
of run 2's token cache.

Annotations at time t come from E1's extraction (bev-e1/extract_agents.py: gt_boxes / gt_names /
gt_velocity_3d of the token's own frame in the navtrain logs, which is what navsim's Annotations holds;
vehicles, pedestrians, bicycles only), so no log is read again. Targets:
DriveJEPAAgentTargetBuilder.agent_targets (drive_jepa_features.py). Written like the rest of the cache
(navsim's dump_feature_target_to_pickle: gzip level 1 pickle), to a .tmp then renamed; an existing
agent_targets.gz is never overwritten (skipped and counted). Ported from paddock-ops
agent-decoder/make_agent_targets.py; only this docstring changed.

    python -m gallops.drivejepa_wajepa.agent_targets <navtrain_agents.pkl> <token cache root> <summary.json>
(Drive-JEPA's interpreter, in the agent decoder's tree; NWORK workers, default 6.)
"""

import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from navsim.agents.drive_jepa_perception_based.drive_jepa_features import (
    AGENT_CLASSES,
    DriveJEPAAgentTargetBuilder,
)
from navsim.common.dataclasses import Annotations
from navsim.planning.training.dataset import dump_feature_target_to_pickle

AG, CACHE, SUMMARY = sys.argv[1:4]
NAME = "agent_targets.gz"


def one_log(job):
    log, samples = job
    n_new = n_skip = n_missing_dir = 0
    counts, classes = [], np.zeros(len(AGENT_CLASSES), np.int64)
    for tok, s in samples.items():
        d = Path(CACHE) / log / tok
        if not (d / "wajepa_tokens.gz").exists():
            n_missing_dir += 1
            continue
        out = d / NAME
        if out.exists():
            n_skip += 1
            continue
        k = len(s["names"])
        t = DriveJEPAAgentTargetBuilder.agent_targets(
            Annotations(
                boxes=s["boxes"],
                names=list(s["names"]),
                velocity_3d=s["vel"],
                instance_tokens=[""] * k,
                track_tokens=[""] * k,
            )
        )
        tmp = d / (NAME + ".tmp")
        dump_feature_target_to_pickle(tmp, t)
        os.rename(tmp, out)
        n_new += 1
        counts.append(int(t["agent_count"]))
        m = t["agent_mask"].numpy()
        classes += np.bincount(t["agent_class"].numpy()[m], minlength=len(AGENT_CLASSES))
    return log, n_new, n_skip, n_missing_dir, counts, classes


if __name__ == "__main__":
    t0 = time.time()
    A = pickle.load(open(AG, "rb"))
    jobs = sorted(A.items())
    tot = dict(new=0, skipped_existing=0, missing_sample_dir=0)
    counts, classes = [], np.zeros(len(AGENT_CLASSES), np.int64)
    with Pool(int(os.environ.get("NWORK", 6))) as pool:
        for k, (_log, a, b, c, cn, cl) in enumerate(pool.imap_unordered(one_log, jobs)):
            tot["new"] += a
            tot["skipped_existing"] += b
            tot["missing_sample_dir"] += c
            counts += cn
            classes += cl
            if k % 100 == 0:
                print(k, "/", len(jobs), tot, int(time.time() - t0), "s", flush=True)
    counts = np.array(counts)
    summary = dict(
        tot,
        logs=len(jobs),
        seconds=int(time.time() - t0),
        agents_in_range=dict(
            mean=round(float(counts.mean()), 2),
            median=float(np.median(counts)),
            p95=float(np.percentile(counts, 95)),
            max=int(counts.max()),
            samples_over_64=int((counts > 64).sum()),
        )
        if len(counts)
        else None,
        kept_per_class=dict(zip(AGENT_CLASSES, classes.tolist())),  # noqa: B905 -- Python 3.9
    )
    json.dump(summary, open(SUMMARY, "w"), indent=1)
    print(json.dumps(summary), "AGENT_TARGETS_DONE", flush=True)
