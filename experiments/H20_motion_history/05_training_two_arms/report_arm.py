"""TRAIN step 4: after an arm finishes, append its check table, the selected checkpoint and why, and the kill switch to
MH/reports/TRAIN_ARMS.md. Reads only runs/arm_<X>/run.json. No closed-loop result is read anywhere.

    report_arm.py A
"""
import json
import sys
from pathlib import Path

import numpy as np

import a4_config as CFG

GO_PULL_MIN = 0.66
REPORT = Path(__file__).resolve().parents[1] / "reports" / "TRAIN_ARMS.md"


def f(x, n=4):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def main(arm: str) -> None:
    rj = json.loads((CFG.RUNS / f"arm_{arm}" / "run.json").read_text())
    ref = rj["reference_step0"]
    lim = rj.get("prereg_stop_at")
    L = [f"\n## Arm {arm} (hide_prob {rj['settings']['hide_prob']})\n",
         f"Started {rj['start']['started']}, ended {rj.get('ended')}; {len(rj['losses']['loss'])} steps; stopped: {rj.get('stopped')}. "
         f"Mean step {np.mean(rj['step_s'][1:]):.2f} s (grading wait {np.mean(rj['grade_wait_s'][1:]):.2f} s). "
         f"Resumes: {len(rj.get('resumes', []))}" + (f" ({'; '.join(r['at'] + ' from ' + Path(str(r.get('from', '?'))).name for r in rj['resumes'])})" if rj.get('resumes') else "") + ".",
         "",
         f"Step 0 (p06 through the wrapper): pick grade {ref['pick_grade_mean']:.6f} (p06's recorded value 0.768489), hard picks above "
         f"10 m/s {ref['hard_picks_above10']} — the reference for the hard-brake gate.",
         "",
         "| check (passes) | step | pick_grade_mean | hard_picks_above10 | go_rate_pull_away | waits_correctly | hard gate | go gate (≥ 0.66) | choosable |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in rj["checks"]:
        s = c["safety"]
        inside = c["pass"] > 0 and (lim is None or c["pass"] <= lim)
        L.append(f"| {c['pass']:.1f} | {c['step']} | {f(s['pick_grade_mean'])} | {s['hard_picks_above10']} | {f(s['go_rate_pull_away'], 3)} | "
                 f"{f(s['waits_correctly'], 3)} | {'pass' if c.get('gate_hard_ok') else 'FAIL'} | {'pass' if c.get('gate_go_ok') else 'FAIL'} | "
                 f"{'step 0 (reference)' if c['pass'] == 0 else ('yes' if inside and c.get('gate_ok') else 'no')} |")
    L.append("")
    if lim is not None:
        L.append(f"0.1.8b's stopping rule would have ended the run at check {lim:.1f}: {rj['prereg_stop_reason']}. The run continued to the "
                 f"fixed span (brief), but later checks are not candidates, as they would not exist under 0.1.8b's rule.")
    else:
        L.append("0.1.8b's stopping rules (2 hard-gate failures in a row, 6 checks without a better choosable pick) did not fire within the span.")
    ch = rj.get("chosen")
    if ch is None:
        L += ["", f"**Selected: none.** No trained check passes both gates. **Kill switch: no checkpoint → arm {arm} not for simulation.**"]
    else:
        c = next(c for c in rj["checks"] if c["pass"] == ch)
        s = c["safety"]
        go = s["go_rate_pull_away"]
        L += ["", f"**Selected: check {ch:.1f}** (`runs/arm_{arm}/ckpt/p{ch:04.1f}.pt`, step {c['step']}): the highest pick grade "
              f"({s['pick_grade_mean']:.4f}) among trained checks with hard picks above 10 m/s ≤ step 0's ({ref['hard_picks_above10']}) and "
              f"pull-away go rate ≥ 0.66 (PREREG §1, 0.1.8b's rule; ties to the earlier check). "
              f"{'Beats' if rj.get('chosen_beats_step0_pick_grade') else 'Does not beat'} step 0's pick grade {ref['pick_grade_mean']:.4f}.",
              "",
              f"**Kill switch (PREREG §3): go_rate_pull_away {go:.3f} vs 0.66 → "
              + ("passes; eligible for simulation (not run).**" if go >= GO_PULL_MIN else f"below; arm {arm} not for simulation.**")]
    with open(REPORT, "a") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
