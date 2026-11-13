"""H15 Part B, B-2's plan: the numbers behind it, drawn for the owner's review.

    <[python] drivejepa> experiments/H15_perception_sidecar/scripts/part_b2_plan_figures.py

For docs/drivejepa-perception-sidecar/part-b/B-2-hidden-refiner.md. Nothing trains and no model runs: it reads
what B-0's and B-1's passes wrote under {runs}/drivejepa_perception, and draws
- b2x-slots.png: how many of the side-car's 128 slots clear each confidence, against the decoded list's cut;
- b2x-room.png: where the room is, navtrain's train rows against navtest, and B-0-3's navtest funnel;
- b2x-scene.jpg: two navtest scenes, each the extreme of a rule in RULES: what arm D reads against what the
  hidden arms can also read, with the 32 proposals, the release's pick and the human's plan;
- b2x-predictions.png: the plan's draft predictions (*argued*) beside B-1's navtest results, copied from its
  records;
- b2x-grid.png: the run grid, arms by recipes, with phases, seeds and estimated cost.

Every number behind a chart is written to ../figures/part-b2-plan-numbers.json.

**Rows.** On navtrain only the train rows are read (splits.json, re-derived by
`navsim_data.load_splits`): never a select or report row. On navtest only the release's labels and the
side-car's agents are read, as B-0-3 read them; nothing is chosen on them.

**The scene pictures** reuse part_b_explained_figures' loader and drawing (its Split, road, the grader's
agents, the plans and camera boxes), so B-2's pictures follow the same rules as B-0's and B-1's.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
os.environ.setdefault("GALLOPS_CONFIG", str(REPO / "gallops.toml"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import part_b_explained_figures as px  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402

from gallops.drivejepa_perception import evaluate, headroom, navsim_data, views  # noqa: E402

OUT, FIGURES, B1 = px.OUT, px.FIGURES, px.B1
NAVTRAIN = {"labels": OUT / "labels-v1", "tokens": OUT / "tokens-v1", "pdm": OUT / "pdm-v1"}
CROSSFIT = OUT / "agents-train-crossfit-v3"
#: Confidence cuts for b2x-slots; the decoded list keeps a slot at evaluate.THR (0.5).
CUTS = (0.1, 0.2, 0.3, 0.5, 0.7)
#: The lowest confidence the hidden arms' evidence is drawn and counted from.
SUB = 0.1
#: "Near": a slot's centre within this of a labelled centre or of a plan pose (the readout's 2 m).
NEAR_M = evaluate.HIT_M
#: Train rows and navtest, as a pair: the reference palette's slots 2 and 1, validated together (the dataviz
#: skill's validator; violet against blue failed it).
TRAIN, NAVTEST = views.CLASS_COLOURS[1], views.CLASS_COLOURS[0]
NUMBERS: dict = {}

#: The plan's draft predictions (*argued*), navtest EPDMS with the human filter, points, per seed.
#: B-2's document copies them; PREDICTIONS.md takes them only once the owner approves.
#: Order: the decision chain, then the rest.
PREDICTIONS = (
    ("0 · the gate: O − C", 0.0, 0.4),
    ("1 · the primary: HD − D", -0.1, 0.2),
    ("2 · HD − C", -0.1, 0.3),
    ("3 · HD − R (meets: low end ≥ −0.4)", -0.4, 0.3),
    ("4 · H6 − HD", -0.3, 0.1),
    ("5 · H3 − H6", -0.1, 0.1),
    ("D − C", -0.1, 0.3),
    ("C − R: the adaptation alone", -0.5, 0.3),
    ("HD, B-2-2 − B-2-1 (descriptive)", -0.3, 0.2),
    ("HD, B-2-3 − B-2-1 (descriptive)", -0.4, 0.2),
)
#: The first pass's F − C across seeds, the scale of seed noise (H15.b1, full mode): −0.21 to +0.24 points.
SEED_SPREAD = (-0.21, 0.24)
MEETS = -0.4

#: The run grid: (arm, recipe) -> (phase, seeds). Hours a run (*estimate*): joint 1–1.5, two stages about
#: 0.75–1.2.
RECIPES = ("B-2-1 · joint", "B-2-2 · two stages, LoRA", "B-2-3 · two stages, full")
HOURS = {RECIPES[0]: (1.0, 1.5), RECIPES[1]: (0.75, 1.2), RECIPES[2]: (0.75, 1.2)}
ARMS = (
    ("C", "nothing from the side-car"),
    ("O", "labelled agents, as D's tokens"),
    ("D", "the decoded list"),
    ("HD", "D + each agent's slot a₆"),
    ("H6", "all 128 slots a₆ + centre"),
    ("H3", "as H6, layer 3"),
)
GRID = {
    ("C", RECIPES[0]): ("1", 3),
    ("O", RECIPES[0]): ("1", 3),
    ("D", RECIPES[0]): ("2a", 3),
    ("HD", RECIPES[0]): ("2a", 3),
    ("H6", RECIPES[0]): ("2a", 3),
    ("H3", RECIPES[0]): ("2a", 1),
    ("C", RECIPES[1]): ("2b", 3),
    ("D", RECIPES[1]): ("2b", 3),
    ("HD", RECIPES[1]): ("2b", 3),
    ("C", RECIPES[2]): ("2b", 3),
    ("D", RECIPES[2]): ("2b", 3),
    ("HD", RECIPES[2]): ("2b", 3),
}
#: The phases, in order: one hue, light to dark (the reference blue ramp's steps 250, 400 and 550,
#: validated as an ordinal ramp).
PHASE_FILL = {"1": "#86b6ef", "2a": "#3987e5", "2b": "#1c5cab"}
PHASE_TEXT = {"1": views.INK, "2a": views.INK, "2b": views.SURFACE}


# ------------------------------------------------------------------------------------------ the data


def navtrain_train(code: Path) -> dict:
    """navtrain's train rows and what the charts read on them: the cross-fit side-cars' confidences, the
    vendored labels of the release's 32 and its pick. Only train rows leave this function. The splits are
    re-derived from their rule against the vendor's navtrain filter, as the preflight reads them."""
    labels = np.load(NAVTRAIN["labels"] / "labels.npz")
    logs = json.loads((NAVTRAIN["labels"] / "manifest.json").read_text())["logs"]
    token_logs = np.array([logs[i] for i in labels["log_id"]])
    splits = navsim_data.load_splits(
        REPO / "recipes/drivejepa_perception/splits.json", navsim_data.navtrain_filter(code)["log_names"]
    )
    cand = np.load(NAVTRAIN["pdm"] / "pdm.npz")["candidates"]
    rows = np.flatnonzero(np.isin(token_logs, splits["train"]) & np.isfinite(cand[:, 0, -1]))
    crossfit = json.loads((CROSSFIT / "manifest.json").read_text())
    if crossfit["rows_decoded"] != len(rows):
        raise SystemExit(f"{CROSSFIT}: {crossfit['rows_decoded']} rows decoded, {len(rows)} train rows")
    return {
        "rows": rows,
        "conf": np.load(CROSSFIT / "agents.npz")["conf"][rows],
        "cand": cand[rows],
        "pick": np.load(NAVTRAIN["tokens"] / "pick.npy").astype(np.int64)[rows],
    }


def fails_nc_or_ttc(cand: np.ndarray, pick: np.ndarray) -> np.ndarray:
    """The pick's vendored verdict: NC below 1 (0 or 0.5) or TTC below 1, B-1-1's labels."""
    n = np.arange(len(pick))
    return (cand[n, pick, headroom.NC] < 1) | (cand[n, pick, headroom.TTC] < 1)


def best_minus_pick(cand: np.ndarray, pick: np.ndarray) -> float:
    """Best of 32 minus the pick, the vendored final, points (headroom.count's own arithmetic)."""
    n = np.arange(len(pick))
    return float((cand[..., headroom.FINAL].max(1) - cand[n, pick, headroom.FINAL]).mean() * 100)


# ---------------------------------------------------------------------------------------- the charts


def fig_slots(train: dict, nav: px.Split) -> None:
    per_row = {
        "train": [float((train["conf"] >= c).sum(1).mean()) for c in CUTS],
        "navtest": [float((nav.agents[0]["conf"] >= c).sum(1).mean()) for c in CUTS],
    }
    NUMBERS["slots_per_row"] = {
        "cuts": list(CUTS),
        "train_rows_crossfit": per_row["train"],
        "navtest_seed0": per_row["navtest"],
        "train_rows": int(len(train["rows"])),
        "navtest_rows": int(len(nav.tokens)),
        "slots": int(train["conf"].shape[1]),
        "decoded_cut": evaluate.THR,
    }
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    x = np.arange(len(CUTS))
    width = 0.36
    for i, (key, colour, label) in enumerate(
        (
            ("train", TRAIN, f"navtrain train rows ({len(train['rows']):,}), the cross-fit side-cars"),
            ("navtest", NAVTEST, f"navtest ({len(nav.tokens):,}), side-car seed 0"),
        )
    ):
        values = per_row[key]
        ax.bar(x + (i - 0.5) * (width + 0.02), values, width, color=colour, label=label, zorder=3)
        for j in (0, CUTS.index(evaluate.THR)):  # selective labels: the widest cut and the decoded list's
            ax.text(
                x[j] + (i - 0.5) * (width + 0.02),
                values[j] + 0.6,
                f"{values[j]:.2f}" if values[j] < 10 else f"{values[j]:.1f}",
                ha="center",
                fontsize=8,
                color=views.INK,
            )
    d = CUTS.index(evaluate.THR)
    ax.axvline(d - 0.5, color=views.INK_2, lw=1, ls=(0, (3, 2)))
    ax.annotate(
        "the decoded list (arm D) keeps\nonly the slots at or above 0.5",
        (d - 0.5, max(per_row["train"]) * 0.82),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=8,
        color=views.INK,
    )
    ax.annotate(
        "what the hidden arms can also read\n(a slot at 0.55 is right about 40% of\n"
        "the time: Part A, measured)",
        (0.62, max(per_row["train"]) * 0.7),
        fontsize=8,
        color=views.INK_2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"≥ {c:g}" for c in CUTS])
    ax.set_xlabel("the slot's confidence, 1 − p(none)")
    ax.set_ylabel("slots a row, of 128 (mean)")
    ax.legend(loc="upper right")
    ax.set_title(
        "What decoding discards: of the side-car's 128 slots a row, the decoded list keeps under five",
        loc="left",
    )
    fig.tight_layout()
    px.save(fig, "b2x-slots.png")


def fig_room(train: dict, nav: px.Split) -> None:
    h = json.loads((OUT / "part-b/b0/headroom-navtest.json").read_text())
    nav_cand, nav_pick = nav.pdm, nav.pick
    fail = {
        "train": float(fails_nc_or_ttc(train["cand"], train["pick"]).mean() * 100),
        "navtest": float(fails_nc_or_ttc(nav_cand, nav_pick).mean() * 100),
    }
    gap = {
        "train": best_minus_pick(train["cand"], train["pick"]),
        "navtest": best_minus_pick(nav_cand, nav_pick),
    }
    if abs(gap["navtest"] - h["best_of_32_minus_pick_vendored_final"]) > 1e-6:
        raise SystemExit("navtest's best-of-32 gap, recomputed, differs from B-0-3's record")
    found = [h["in_view_found_by_seed"][f"s{k}"] for k in (0, 1, 2)]
    funnel = [
        ("the release's pick fails NC or TTC\n(official)", h["pick_fails_nc_or_ttc"]),
        ("… with a safe proposal among its 32", h["with_a_safe_proposal"]),
        (
            "… the agent behind it in CAM_F0's\nview within 50 m, labelled",
            h["by_object_behind_the_failure"]["in_view_50m"],
        ),
        ("… and the side-car found it\n(seeds 0, 1, 2)", int(np.median(found))),
    ]
    NUMBERS["room"] = {
        "pick_fails_nc_or_ttc_vendored_percent": fail,
        "best_of_32_minus_pick_vendored_points": gap,
        "navtest_funnel_official": {
            "failing": funnel[0][1],
            "with_safe_proposal": funnel[1][1],
            "in_view_50m": funnel[2][1],
            "found_by_seed": found,
        },
        "source": "train rows: pdm-v1 and tokens-v1 (counted here); navtest: headroom-navtest.json "
        "(B-0-3's record)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1, 1, 2.1]})
    for ax, values, title, unit in (
        (axes[0], fail, "the release's pick fails NC or TTC", "%"),
        (axes[1], gap, "best of 32 minus the pick", " pts"),
    ):
        keys = ("train", "navtest")
        ax.bar([0, 1], [values[k] for k in keys], 0.55, color=[TRAIN, NAVTEST], zorder=3)
        for i, k in enumerate(keys):
            ax.text(i, values[k] * 1.02, f"{values[k]:.2f}{unit}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["train rows\n(in-sample)", "navtest"])
        ax.set_ylim(0, max(values.values()) * 1.25)
        ax.set_title(title + "\n(vendored labels)", loc="left")
    axes[0].set_ylabel("share of rows (%)")
    axes[1].set_ylabel("points of the vendored final")
    ax = axes[2]
    y = np.arange(len(funnel))[::-1]
    ax.barh(y, [v for _, v in funnel], 0.55, color=NAVTEST, zorder=3)
    lo, hi = min(found), max(found)
    for yy, (_, v) in zip(y[:-1], funnel[:-1]):  # noqa: B905 -- Python 3.9
        ax.text(v + 3, yy, f"{v}", va="center", fontsize=8.5)
    # The last step differs by seed: the bar is the median seed, the whisker the three seeds' range.
    ax.plot([lo, hi], [y[-1], y[-1]], color=views.INK, lw=1.5, zorder=4)
    ax.text(
        hi + 3,
        y[-1],
        f"{funnel[-1][1]} (seeds: {', '.join(str(f) for f in found)})",
        va="center",
        fontsize=8.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _ in funnel])
    ax.set_xlim(0, funnel[0][1] * 1.2)
    ax.set_xlabel("navtest scenes, of 12,146")
    ax.set_title("navtest's room for an agent input (B-0-3, recorded)", loc="left")
    fig.suptitle(
        "Where the room is: the release rarely fails where it trained; on navtest the better plan is usually "
        "already among its 32",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    px.save(fig, "b2x-room.png")


#: The two scenes' rules (`pick_scenes`); ties go to the first token in sorted order.
RULES = {
    "evidence": "among navtest scenes whose release pick fails NC or TTC (official) with a safe proposal "
    "among its 32, behind which is a labelled agent in CAM_F0's view within 50 m that side-car seed 0 "
    "missed at 0.5 (B-0-3's own test): the one with a slot at confidence 0.1 to 0.5 nearest that "
    "agent's centre",
    "noise": "among navtest scenes whose release pick passes NC and TTC (official), the one with the most "
    "slots at confidence 0.1 to 0.5 within 2 m of the picked plan's poses and 2 m or more from every "
    "labelled object",
}


def pick_scenes(nav: px.Split) -> dict:
    """The two scenes b2x-scene draws, each the extreme of its rule in RULES: what decoding drops where it
    mattered (`evidence`), and what a hidden arm must learn to discount where the release was already right
    (`noise`)."""
    official = px.official(OUT / "part-b/b0/navtest-R")
    fails = np.array([official[t]["NC"] < 1 or official[t]["TTC"] < 1 for t in nav.tokens])
    safe = ((nav.pdm[..., headroom.NC] == 1) & (nav.pdm[..., headroom.TTC] == 1)).any(1)
    a0 = nav.agents[0]
    sub = (a0["conf"] >= SUB) & (a0["conf"] < evaluate.THR)
    order = np.argsort(np.array(nav.tokens), kind="stable")
    evidence, noise = None, None
    for r in order:
        r = int(r)
        index = np.flatnonzero(sub[r])
        slots = a0["xyz"][r][index, :2]
        if fails[r] and safe[r]:
            track, step = int(nav.collision_agent[r, nav.pick[r]]), int(nav.collision_step[r, nav.pick[r]])
            if track == headroom.NO_AGENT:
                track, step = int(nav.ttc_agent[r, nav.pick[r]]), int(nav.ttc_step[r, nav.pick[r]])
            kind, slot = headroom.failure_kind(nav.labelled(r), nav.projection(r), track)
            if kind != "in_view_50m" or not len(slots):
                continue
            target = nav.targets(r)
            if headroom.found_by(a0, r, target, slot):
                continue
            dist = np.linalg.norm(slots - target["box"][slot, :2], axis=1)
            k = int(np.argmin(dist))
            if evidence is None or dist[k] < evidence["nearest_sub_slot_m"]:
                q = int(index[k])
                evidence = {
                    "row": r,
                    "token": nav.tokens[r],
                    "track": track,
                    "step": max(step, 0),
                    "nearest_sub_slot_m": float(dist[k]),
                    "sub_slot": q,
                    "sub_slot_conf": float(a0["conf"][r, q]),
                    "sub_slot_class": views.CLASS_NAMES[int(a0["cls"][r, q])],
                }
        elif not fails[r] and len(slots):
            path = np.asarray(nav.proposals[r, nav.pick[r], :, :2])
            on_path = np.linalg.norm(slots[:, None] - path[None], axis=-1).min(1) < NEAR_M
            labelled = nav.labelled(r)["box"][:, :2]
            alone = (
                np.linalg.norm(slots[:, None] - labelled[None], axis=-1).min(1) >= NEAR_M
                if len(labelled)
                else np.ones(len(slots), bool)
            )
            false = index[on_path & alone]
            if len(false) and (noise is None or len(false) > noise["false_slots_on_path"]):
                noise = {
                    "row": r,
                    "token": nav.tokens[r],
                    "false_slots_on_path": int(len(false)),
                    "false_slots": false.tolist(),
                    "false_slot_conf": [round(float(a0["conf"][r, q]), 3) for q in false],
                }
    for example in (evidence, noise):
        r = example["row"]
        example["log"] = str(nav.logs[r])
        example["slots_at_0.1_to_0.5"] = int(sub[r].sum())
        example["decoded"] = int((a0["conf"][r] >= evaluate.THR).sum())
    evidence["rule"], noise["rule"] = RULES["evidence"], RULES["noise"]
    return {"evidence": evidence, "noise": noise}


def outlines(ax, boxes: np.ndarray, colours) -> None:
    """Thin dashed footprints: slots below the decoded list's cut."""
    for box, colour in zip(np.asarray(boxes).reshape(-1, 7), colours):  # noqa: B905 -- Python 3.9
        u, v = px.xy_to_plot(views.footprint(box[None])[0])
        ax.add_patch(
            Polygon(np.c_[u, v], closed=True, fc="none", ec=colour, lw=0.9, ls=(0, (2, 1.5)), zorder=4)
        )


def fig_scene(nav: px.Split, scenes: dict) -> None:
    """Each scene: the camera at t with the side-car's slots, and from above the grader's road, agents, the
    32 proposals and the pick. The rule's own finding is drawn heavier: the missed agent and the slot on
    it, or the slots on the release's safe plan that sit on no labelled object."""
    import textwrap

    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5), gridspec_kw={"width_ratios": [2.0, 1.45]})
    heads = {
        "evidence": "What decoding drops, where it mattered",
        "noise": "What a hidden arm must learn to discount",
    }
    for i, key in enumerate(("evidence", "noise")):
        ex = scenes[key]
        r = ex["row"]
        raw = px.sidecar_raw(nav, r, 0)
        decoded = np.flatnonzero(raw["conf"] >= evaluate.THR)
        sub = np.flatnonzero((raw["conf"] >= SUB) & (raw["conf"] < evaluate.THR))
        marked = [ex["sub_slot"]] if key == "evidence" else list(ex["false_slots"])
        colour = {q: views.CLASS_COLOURS[int(raw["cls"][q])] for q in np.r_[decoded, sub]}
        for q in marked:
            colour[int(q)] = views.INK if key == "evidence" else px.FAIL
        boxes = [px.det_box(raw, q) for q in decoded] + [px.det_box(raw, q) for q in sub]
        colours = [colour[q] for q in decoded] + [colour[q] for q in sub]
        lws = [1.8] * len(decoded) + [2.0 if q in marked else 0.8 for q in sub]
        styles = ["-"] * len(decoded) + [(0, (2, 2))] * len(sub)
        if key == "evidence":
            labelled = nav.labelled(r)
            a = int(np.flatnonzero(labelled["track"] == ex["track"])[0])
            boxes.append(np.asarray(labelled["box"][a], np.float64))
            colours.append(px.FAIL)
            lws.append(2.6)
            styles.append("-")
        cam = axes[i, 0]
        px.image_boxes(cam, views.model_view(nav.image(r)), nav.projection(r), boxes, colours, lws, styles)
        cam.set_title(
            f"{heads[key]} · navtest {ex['token']}\nthe front camera at t: {len(decoded)} slots at ≥ 0.5 "
            f"(solid, arm D's list), {len(sub)} at 0.1–0.5 (dashed)",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
        )
        if key == "evidence":
            detail = (
                "The release's pick fails on the agent outlined in red (labelled, in view): side-car seed 0 "
                "has no slot on it at 0.5, so arm D cannot see it. "
                f"A {ex['sub_slot_class']} slot at confidence {ex['sub_slot_conf']:.2f} (black, dashed) lies "
                f"{ex['nearest_sub_slot_m']:.2f} m from its centre: evidence only the hidden arms receive. "
                "Whether a learned reader can use such a slot without being misled by the others is what HD "
                "and H6 test."
            )
        else:
            detail = (
                f"The release's pick passes NC and TTC here. {ex['false_slots_on_path']} slots at 0.1–0.5 "
                "(red, dashed) lie within 2 m of its plan with no labelled centre within 2 m: false by the "
                "readout's own rule. In the picture most sit around the car ahead at other depths, Part A's "
                "commonest false alarm (45% are 2–4 m misplacements). The hidden arms receive them and D "
                "does not; a reader that trusted them would leave a safe plan."
            )
        cam.text(
            0,
            -0.04,
            textwrap.fill(detail, 120),
            transform=cam.transAxes,
            fontsize=8.5,
            color=views.INK_2,
            va="top",
        )
        ax = axes[i, 1]
        cache = nav.cache(r)
        px.draw_road(ax, cache)
        px.grader_agents(ax, cache, ex.get("step", 0), ex.get("track"))
        px.footprints(
            ax,
            np.array([px.det_box(raw, q) for q in decoded]).reshape(-1, 7),
            [colour[q] for q in decoded],
            fill=True,
            alpha=0.6,
        )
        outlines(ax, np.array([px.det_box(raw, q) for q in sub]).reshape(-1, 7), [colour[q] for q in sub])
        px.plans(ax, nav, r, {"release": (int(nav.pick[r]), "s")})
        extra = [
            Patch(color=views.INK_2, alpha=0.25, label="agents at t, as the grader reads them"),
            Patch(color=views.GREY, alpha=0.6, label="slot ≥ 0.5, filled: arm D's list"),
            Line2D(
                [],
                [],
                color=views.GREY,
                lw=0.9,
                ls=(0, (2, 1.5)),
                label="slot at 0.1–0.5, dashed: hidden arms only",
            ),
            *[
                Patch(color=c, alpha=0.6, label=f"a slot's class: {n}")
                for c, n in zip(views.CLASS_COLOURS, views.CLASS_NAMES)  # noqa: B905 -- Python 3.9
            ],
        ]
        if key == "evidence":
            extra += [
                Patch(fc="none", ec=px.FAIL, label="the agent the pick fails on, at t"),
                Patch(color=px.FAIL, label=f"it, at {ex['step'] / 10:.1f} s"),
                Line2D([], [], color=views.INK, lw=1.6, ls=(0, (2, 1.5)), label="the slot on it, below 0.5"),
            ]
        else:
            extra += [
                Line2D(
                    [],
                    [],
                    color=px.FAIL,
                    lw=1.6,
                    ls=(0, (2, 1.5)),
                    label="a slot on the plan, false by the readout",
                )
            ]
        px.plan_legend(ax, extra, outside=True)
        px.bev(ax, (-22, 22), (-8, 48), "from above")
    fig.suptitle(
        "The same scene, two inputs: the decoded list (arm D) and the slots beneath it (arms H6, H3, and "
        "HD's content) · side-car seed 0, as B-1 read it; B-2 reads the fold side-cars",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    px.save(fig, "b2x-scene.jpg")


def fig_predictions() -> None:
    r = json.loads((B1 / "navtest/navtest.json").read_text())
    r4 = json.loads((B1 / "navtest-b14/navtest.json").read_text())
    measured = [
        ("B-1-1, the ceiling (reads the labels)", r["pairs"]["B-1-1"]),
        ("B-1-2, labelled agents, a rule", r["pairs"]["B-1-2"]),
    ]
    measured += [(f"B-1-3, side-car, a rule, seed {k}", r["pairs"][f"B-1-3-s{k}"]) for k in (0, 1, 2)]
    measured += [(f"B-1-4, side-car, a penalty, seed {k}", r4["pairs"][f"B-1-4-s{k}"]) for k in (0, 1, 2)]
    NUMBERS["predictions_argued"] = [{"contrast": c, "low": lo, "high": hi} for c, lo, hi in PREDICTIONS]
    NUMBERS["b1_measured"] = {
        name: {"minus_R": p["epdms_hf"]["a_minus_b"], "se": p["epdms_hf"]["se"]} for name, p in measured
    }
    rows = len(PREDICTIONS) + len(measured) + 1
    fig, ax = plt.subplots(figsize=(9.6, 0.36 * rows + 1.6))
    ax.axvspan(*SEED_SPREAD, color=views.GRID, alpha=0.7, zorder=0, lw=0)
    ax.axvline(0, color=views.INK_2, lw=1, zorder=1)
    ax.axvline(MEETS, color=views.INK_2, lw=1, ls=(0, (3, 2)), zorder=1)
    labels = []
    y = rows - 1
    for name, lo, hi in PREDICTIONS:
        primary = name.startswith("1 ·")
        ax.plot([lo, hi], [y, y], color=NAVTEST, lw=6 if primary else 4, solid_capstyle="round", zorder=3)
        labels.append((y, name, primary))
        y -= 1
    y -= 1
    for name, p in measured:
        d, se = p["epdms_hf"]["a_minus_b"], p["epdms_hf"]["se"]
        ax.plot([d - 2 * se, d + 2 * se], [y, y], color=views.GREY, lw=1.5, zorder=3)
        ax.plot(d, y, marker="o", ms=7, color=views.GREY, mec=views.SURFACE, mew=1.5, zorder=4)
        ax.text(d + 2 * se + 0.04, y, f"{d:+.2f}", va="center", fontsize=7.5, color=views.INK_2)
        labels.append((y, name, False))
        y -= 1
    ax.set_yticks([yy for yy, _, _ in labels])
    ax.set_yticklabels([name for _, name, _ in labels])
    for tick, (_, _, primary) in zip(ax.get_yticklabels(), labels):  # noqa: B905 -- Python 3.9
        tick.set_fontweight("bold" if primary else "normal")
    ax.set_ylim(-1, rows)
    ax.set_xlim(-0.7, 2.25)
    ax.text(
        SEED_SPREAD[1] + 0.02,
        rows - 0.6,
        "the first pass's seed spread (F − C)",
        fontsize=7.5,
        color=views.INK_2,
        va="top",
    )
    ax.text(MEETS - 0.02, rows - 0.6, "meets R", fontsize=7.5, color=views.INK_2, ha="right", va="top")
    ax.legend(
        handles=[
            Line2D([], [], color=NAVTEST, lw=4, label="B-2: draft prediction, a range (argued)"),
            Line2D(
                [],
                [],
                color=views.GREY,
                lw=1.5,
                marker="o",
                ms=6,
                label="B-1: measured on navtest, ± 2 SE (recorded)",
            ),
        ],
        loc="lower right",
    )
    ax.set_xlabel("navtest EPDMS (human filter), points: the named difference, per seed")
    ax.set_title(
        "B-2's draft predictions, in the decision chain's order, beside what B-1 measured", loc="left"
    )
    fig.tight_layout()
    px.save(fig, "b2x-predictions.png")


def fig_grid() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    cell_w, cell_h = 1.0, 1.0
    totals = {p: [0, 0.0, 0.0] for p in PHASE_FILL}
    for i, (arm, what) in enumerate(ARMS):
        y = len(ARMS) - 1 - i
        ax.text(-0.08, y + 0.5, f"{arm}", ha="right", va="center", fontsize=10, fontweight="bold")
        ax.text(-0.08, y + 0.2, what, ha="right", va="center", fontsize=7.5, color=views.INK_2)
        for j, recipe in enumerate(RECIPES):
            cell = GRID.get((arm, recipe))
            if cell is None:
                ax.add_patch(
                    Rectangle(
                        (j + 0.03, y + 0.05),
                        cell_w - 0.06,
                        cell_h - 0.1,
                        fc=views.SURFACE,
                        ec=views.GRID,
                        lw=1,
                    )
                )
                ax.text(j + 0.5, y + 0.5, "not run", ha="center", va="center", fontsize=8, color=views.GREY)
                continue
            phase, seeds = cell
            lo, hi = (h * seeds for h in HOURS[recipe])
            totals[phase][0] += seeds
            totals[phase][1] += lo
            totals[phase][2] += hi
            ax.add_patch(
                Rectangle((j + 0.03, y + 0.05), cell_w - 0.06, cell_h - 0.1, fc=PHASE_FILL[phase], ec="none")
            )
            seeds_text = "seed 0 (pilot)" if seeds == 1 else f"seeds 0–{seeds - 1}"
            ax.text(
                j + 0.5,
                y + 0.62,
                f"phase {phase}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=PHASE_TEXT[phase],
            )
            ax.text(
                j + 0.5,
                y + 0.33,
                f"{seeds_text} · {lo:.1f}–{hi:.1f} h",
                ha="center",
                va="center",
                fontsize=8,
                color=PHASE_TEXT[phase],
            )
    for j, recipe in enumerate(RECIPES):
        name, how = recipe.split(" · ")
        ax.text(
            j + 0.5,
            len(ARMS) + 0.12,
            f"{name}\n{how}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    NUMBERS["grid"] = {
        "cells": {f"{a} | {r}": {"phase": p, "seeds": s} for (a, r), (p, s) in GRID.items()},
        "hours_per_run_estimate": {r: list(h) for r, h in HOURS.items()},
        "totals_by_phase": {
            p: {"runs": t[0], "hours_low": round(t[1], 1), "hours_high": round(t[2], 1)}
            for p, t in totals.items()
        },
    }
    summary = " · ".join(f"phase {p}: {t[0]} runs, {t[1]:.0f}–{t[2]:.0f} h" for p, t in totals.items())
    ax.text(
        0,
        -0.2,
        summary
        + "\nevery estimate is wall time on one shared GPU; official reads add about 20 min of CPU a run",
        fontsize=8,
        color=views.INK_2,
        va="top",
    )
    ax.set_xlim(-1.9, len(RECIPES) + 0.05)
    ax.set_ylim(-1.1, len(ARMS) + 1.0)
    ax.axis("off")
    ax.set_title(
        "The run grid: which arm runs under which recipe, in which phase (H3 as a pilot, by decision)",
        loc="left",
    )
    fig.tight_layout()
    px.save(fig, "b2x-grid.png")


# ---------------------------------------------------------------------------------------------- main


def main() -> None:
    from gallops.drivejepa_perception import env

    entered = env.enter(
        px.CFG
    )  # navsim's own classes, for unpickling the metric caches the scene pictures draw
    train = navtrain_train(Path(entered["code"]))
    nav = px.Split("navtest")
    fig_slots(train, nav)
    fig_room(train, nav)
    scenes = pick_scenes(nav)
    NUMBERS["scenes"] = scenes
    fig_scene(nav, scenes)
    fig_predictions()
    fig_grid()
    (FIGURES / "part-b2-plan-numbers.json").write_text(json.dumps(NUMBERS, indent=1, default=str))
    print(json.dumps({k: v for k, v in NUMBERS.items() if k != "grid"}, indent=1, default=str))


if __name__ == "__main__":
    main()
