import sys
from collections import defaultdict
import numpy as np
d = dict(np.load(sys.argv[1], allow_pickle=False))
n = len(d["ts"]); groups = defaultdict(list)
for i in range(n): groups[(str(d["scene"][i]), int(d["ts"][i]))].append(i)
print(f"{n} ticks, {len(groups)} (scene, timestamp) groups, median {int(np.median([len(g) for g in groups.values()]))} episodes per group")

# which candidates are geometrically identical across a group?
same = np.zeros(64); tot = 0
for g in groups.values():
    if len(g) < 2: continue
    P = d["plans"][g][..., :2]; same += (np.abs(P - P[0]).max(axis=(0, 2, 3)) < 0.01); tot += 1
ident = np.where(same / tot > 0.99)[0]
print("candidates identical across a group in >99% of groups:", ident.tolist())

def pairs(cands):
    out = []
    for (sc, t), g in groups.items():
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                i, j = g[a], g[b]
                for c in cands:
                    if np.abs(d["plans"][i, c, :, :2] - d["plans"][j, c, :, :2]).max() >= 0.01: continue
                    out.append((i, j, c))
    return np.array(out)

P = pairs(ident)
i, j, c = P.T
li, lj = d["label"][i, c], d["label"][j, c]
diff = li != lj
print(f"{len(P)} same-plan pairs; the car changes the outcome in {diff.mean():.1%} of them ({diff.sum()})")

def paired_acc(pred, mask):
    ii, jj, cc = i[mask], j[mask], c[mask]
    clear_is_i = d["label"][ii, cc] > d["label"][jj, cc]
    pc = np.where(clear_is_i, pred[ii, cc], pred[jj, cc]); ph = np.where(clear_is_i, pred[jj, cc], pred[ii, cc])
    return ((pc > ph) + 0.5 * (pc == ph)).mean(), np.abs(pc - ph).mean(), (pc - ph).mean()

def auc(score, y):
    o = np.argsort(score); r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

rows_all = np.unique(np.concatenate([i, j])); 
for name in ("held", "final"):
    pr = d[name]
    acc, gap, signed = paired_acc(pr, diff)
    same_gap = np.abs(pr[i[~diff], c[~diff]] - pr[j[~diff], c[~diff]]).mean()
    # unpaired AUC over the same plans, for scale
    yy = d["label"][rows_all][:, ident].ravel(); ss = pr[rows_all][:, ident].ravel()
    print(f"\n[{name}] unpaired AUC on these plans {auc(ss, yy):.3f}  |  PAIRED accuracy where the car changes the outcome: {acc:.3f}"
          f"  (0.5 = blind to the car)\n         mean answer gap clear-vs-hit {signed:+.3f} (abs {gap:.3f}); gap when the outcome is the same {same_gap:.3f}")
    # variance: how much of the answer is fixed by (place, moment, plan)?
    cell = defaultdict(list)
    for (sc, t), g in groups.items():
        for cc in ident: cell[(sc, t, cc)].append(g)
    for nm, arr in ((name, pr), ("label", d["label"])):
        within = []; vals = []
        for (sc, t, cc), gs in cell.items():
            v = arr[gs[0], cc]; vals.append(v); within.append(((v - v.mean()) ** 2).sum())
        allv = np.concatenate(vals); print(f"         share of {nm:5s} variance that is within (place, moment, plan): {sum(within) / ((allv - allv.mean()) ** 2).sum():.1%}")

# breakdowns, held-out only
pr = d["held"]
print("\nheld-out paired accuracy by breakdown (n = label-differing pairs):")
shi, shj = d["shape"][i], d["shape"][j]
hitter = np.where(li[:] < lj[:], i, j)  # the episode whose plan hits
hit_t = d["hit5"][hitter, c]
infr = d["in_frame"][hitter]
for label, m in [
    ("same scenario, different run", diff & (shi == shj)),
    ("different scenario", diff & (shi != shj)),
    ("frontal vs frontal", diff & (shi == "frontal") & (shj == "frontal")),
    ("hazard in camera view", diff & infr),
    ("hazard not in view", diff & ~infr),
    ("hit within 1 s", diff & (hit_t < 1)), ("hit at 1-2 s", diff & (hit_t >= 1) & (hit_t < 2)),
    ("hit at 2-3 s", diff & (hit_t >= 2) & (hit_t < 3)), ("hit at 3-5 s", diff & (hit_t >= 3)),
] + [(f"scene {s}", diff & (d["scene"][i] == s)) for s in ("0053", "0058", "0064", "0070", "0075")]:
    if m.sum() == 0: print(f"  {label:30s} n=0"); continue
    a, g, sg = paired_acc(pr, m); a2, _, sg2 = paired_acc(d["final"], m)
    print(f"  {label:30s} n={m.sum():6d}  held-out {a:.3f} (gap {sg:+.3f})   trained-on {a2:.3f} (gap {sg2:+.3f})")
