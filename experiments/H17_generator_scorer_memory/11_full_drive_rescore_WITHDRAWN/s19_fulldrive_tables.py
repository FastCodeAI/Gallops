"""Tables for reports/FULLDRIVE_RESCORE.md from out/fulldrive_rescore.json (no other input). Writes out/fulldrive_rescore.csv and out/fulldrive_tables.md."""
import csv, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
R = json.load(open(ROOT / "out/fulldrive_rescore.json")); by = {(r["arm"], r["scene"]): r for r in R}
S12 = [r["scene"] for r in R if r["arm"] == "carry" and r["block"] == "12"]; S18 = [r["scene"] for r in R if r["arm"] == "carry" and r["block"] == "18"]
f = lambda x: f"{x:.3f}"
def ev(r):
    return "; ".join(f"{e['label']} {e['rel']:+.2f}{'' if e['in_old_window'] else '*'}" for e in r["events"]) or "none"
with open(ROOT / "out/fulldrive_rescore.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["arm", "scene", "set", "old", "full", "reproduces", "full_failure", "old_window_end_s", "first_4m_s", "last_tick_s", "events"])
    for r in R:
        w.writerow([r["arm"], r["scene"], "hard" if r["block"] == "12" else "easy", r["old_score_file"], r["full_score"], r["reproduces"], r["full_failure"],
                    round(r["old_win_end_rel"], 2), None if r["first_dist4_rel"] is None else round(r["first_dist4_rel"], 2), round(r["last_tick_rel"], 2), ev(r)])
L = ["| set | scene | carry old → full | 0.1.8b old → full | gap old → full | full result |", "|---|---|---|---|---|---|"]
for s in S12 + S18:
    c, b = by[("carry", s)], by[("018b", s)]; do, df = c["old_score_file"] - b["old_score_file"], c["full_score"] - b["full_score"]
    res = "win" if df > 0.1 else "loss" if df < -0.1 else "tie"
    L.append(f"| {'hard' if s in S12 else 'easy'} | `{s}` | {f(c['old_score_file'])} → **{f(c['full_score'])}** | {f(b['old_score_file'])} → **{f(b['full_score'])}** | {do:+.3f} → **{df:+.3f}** | {res} |")
L += ["", "| arm | scene | old | full | window end (s) | first ≥ 4 m (s) | last tick (s) | events over the whole drive (s after handover; * = after the old window) |", "|---|---|---|---|---|---|---|---|"]
for arm in ("carry", "018b", "reset", "017"):
    for s in S12 + S18:
        r = by.get((arm, s))
        if not r: continue
        d4 = "—" if r["first_dist4_rel"] is None else f"{r['first_dist4_rel']:+.2f}"
        L.append(f"| {arm} | `{s}` | {f(r['old_score_file'])} | {f(r['full_score'])} | {r['old_win_end_rel']:+.2f} | {d4} | {r['last_tick_rel']:+.2f} | {ev(r)} |")
(ROOT / "out/fulldrive_tables.md").write_text("\n".join(L) + "\n"); print("\n".join(L[:34]))
