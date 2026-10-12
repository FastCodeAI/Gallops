"""Score 5-seed fold runs: held-out rendered scenes (rescue/kept/frontal) and held-out real scenes (normal driving)."""
import sys, numpy as np
C = "{dj_cache}/"
v2 = np.load(C + "v2/probe_input.npz"); v3 = np.load(C + "v3_nusc/probe_input.npz")
keys = ("hit5_at", "off_road_at", "pdm", "in_frame", "shape", "max_deviation", "travel", "human_travel", "ttc_at", "plans")
d = {k: np.concatenate([v2[k], v3[k]]) for k in keys}
NR = len(v2["scene"])
src = np.array([e.split("/")[0] for e in np.concatenate([v2["episode"], v3["episode"]])])
safe = np.isnan(d["hit5_at"]) & np.isnan(d["off_road_at"]); sig = lambda v: 1 / (1 + np.exp(-v)); SC = ("0053", "0058", "0064", "0070", "0075")
def formula(s): return s[..., 0] ** 4 * s[..., 1] * (5 * s[..., 4] + 5 * s[..., 2] + 2 * s[..., 3]) / 12
def pick_of(s, te):
    rows = np.arange(len(te)); rel = d["pdm"][te].argmax(1); veto = s[rows, rel, 0] < 0.5
    return np.where(veto, formula(s).argmax(1), rel), rel, veto
def rendered(s, te):
    rows = np.arange(len(te)); pick, rel, veto = pick_of(s, te); sft = safe[te]; th = d["hit5_at"][te][rows, rel]
    resc = ~np.isnan(th) & d["in_frame"][te]; calm = sft[rows, rel]; fr2 = resc & (d["shape"][te] == "frontal") & (th > 2)
    return int(sft[rows, pick][resc].sum() - (~sft[rows, pick][calm]).sum()), float(sft[rows, pick][fr2].mean()), float(sft[rows, pick][calm].mean())
def auc(score, label):
    pos, neg = score[label], score[~label]
    allv = np.concatenate([pos, neg]); r = np.empty(len(allv)); r[np.argsort(allv, kind="mergesort")] = np.arange(1, len(allv) + 1)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def real(s, te):
    rows = np.arange(len(te)); pick, rel, veto = pick_of(s, te); sft = safe[te]
    human = d["human_travel"][te]
    out = {"ticks": len(te), "released safe": sft[rows, rel].mean(), "gated pick safe": sft[rows, pick].mean(),
           "override rate": veto.mean(), "picks a swerve": (pick >= 32).mean(),
           "kept safe (released safe)": sft[rows, pick][sft[rows, rel]].mean(),
           "rescued (released hits)": sft[rows, pick][~np.isnan(d["hit5_at"][te][rows, rel])].mean(),
           "travel/human median released": np.median(d["travel"][te][rows, rel] / np.maximum(human, 1e-3)),
           "travel/human median pick": np.median(d["travel"][te][rows, pick] / np.maximum(human, 1e-3)),
           "AUC no-collision": auc(1 - s[..., 0].ravel(), ~np.isnan(d["hit5_at"][te]).ravel()),
           "AUC on-road": auc(1 - s[..., 1].ravel(), ~np.isnan(d["off_road_at"][te]).ravel()),
           "AUC in-lane": auc(1 - s[..., 3].ravel(), (d["max_deviation"][te] > 1.75).ravel()),
           "AUC released score vs collision (natives)": auc(-d["pdm"][te].ravel(), ~np.isnan(d["hit5_at"][te][:, :32]).ravel())}
    return {k: (round(float(v), 3) if not isinstance(v, int) else v) for k, v in out.items()}
for tag in sys.argv[1:]:
    L = [np.load(C + f"v2/probe_{tag}_s{s}.logits.npz") for s in range(5)]
    single = [sum(rendered(sig(l[f"logits_{sc}"]), l[f"test_idx_{sc}"])[0] for sc in SC) for l in L]
    ens = [rendered(np.mean([sig(l[f"logits_{sc}"]) for l in L], 0), L[0][f"test_idx_{sc}"]) for sc in SC]
    print(f"== {tag}\n rendered held-out: single-seed nets {single} mean {np.mean(single):.0f} sd {np.std(single, ddof=1):.0f} | avg-of-5 net {sum(e[0] for e in ens)} {[e[0] for e in ens]}")
    print(f"   frontal >2 s rescued {[round(e[1], 2) for e in ens]}, kept safe {[round(e[2], 3) for e in ens]}")
    if "logits_nusc_holdout" in L[0]:
        te = L[0]["test_idx_nusc_holdout"]
        print(f" real held-out (66 scenes, avg of 5 seeds x 5 folds): {real(np.mean([sig(l['logits_nusc_holdout']) for l in L], 0), te)}")
