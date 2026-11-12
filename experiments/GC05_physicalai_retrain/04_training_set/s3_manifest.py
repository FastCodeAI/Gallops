"""Part D + checks Z1-Z6: manifest.parquet, tables.md. conda drive-jepa python."""
import glob, json, hashlib, subprocess, numpy as np, pandas as pd, pathlib
import s3_common as C
HERE = C.HERE; out = []; W = lambda s="": (out.append(s), print(s))
def tbl(df):
    f = lambda v: "" if (isinstance(v, float) and np.isnan(v)) or v is None else (f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v))
    W("| " + " | ".join(map(str, df.columns)) + " |"); W("|" + "---|" * len(df.columns))
    for r in df.itertuples(index=False): W("| " + " | ".join(f(v) for v in r) + " |")
    W()
pct = lambda x: f"{100*x:.1f} %"
clips = pd.read_csv(HERE / "clips.csv"); V = C.views(); ci = V["ci"]; dc = pd.read_parquet(C.S0C.RAW_SNAP / "metadata/data_collection.parquet")
cv = pd.concat([pd.read_csv(p) for p in glob.glob(str(HERE / "convert_rows/*.csv"))], ignore_index=True); gr = pd.concat([pd.read_parquet(p) for p in glob.glob(str(HERE / "grade_rows/*.parquet"))], ignore_index=True)
# ---------- download
dl = pd.read_csv(HERE / "download_manifest.csv"); mp4 = dl[dl.file.str.endswith(".mp4")]; W("### Download"); W()
t = pd.to_datetime(dl[dl.file.str.contains("camera_front_wide")].ts); wall = (t.max() - t.min()).total_seconds() / 3600
tbl(pd.DataFrame([dict(item="clips with front-wide video", value=f"{mp4.uuid.nunique()} of {len(clips)}"), dict(item="front-wide video bytes", value=f"{mp4.bytes.sum()/1e9:.1f} GB ({mp4.bytes.mean()/1e6:.1f} MB per clip)"),
                  dict(item="wall time, 6 workers", value=f"{wall:.2f} h"), dict(item="S1 projection for this many clips (13.4 MB, 8.8 s serial / 6)", value=f"{len(clips)*13.4/1e3:.1f} GB, {len(clips)*8.8/3600/6:.2f} h"),
                  dict(item="calibration parquets (chunks)", value=int(dl.file.str.startswith("calibration/").sum())), dict(item="NuRec map members (scenes)", value=f"{dl[dl.file.str.startswith('clipgt/')].uuid.nunique()} fetched + 40 copied from S2"),
                  dict(item="manifest rows / total bytes / rows without sha256", value=f"{len(dl)} / {dl.bytes.sum()/1e9:.1f} GB / {int(dl.sha256.isna().sum())}")]))
# ---------- samples
ok = cv[cv.status == "ok"].copy(); bad = cv[cv.status != "ok"]; ok = ok.merge(clips[["uuid", "source", "type", "chunk", "split", "country"]], on="uuid")
g = gr[gr.sample_id != "-"].set_index("sample_id"); ok["grade_status"] = ok.sample_id.map(g.status) if len(g) else None
ok["has_label2"] = (ok.source == "nurec") & (ok.grade_status == "ok"); ok["dropped_d2"] = ok.grade_status == "dropped_d2"
ok["direction_exempt"] = ok.sample_id.map(g.human_exempt).fillna(False).astype(bool) & ok.has_label2
REPO = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
sh = lambda *ps: hashlib.sha256(b"".join(open(p, "rb").read() for p in ps)).hexdigest()[:12]
conv_id = f"{REPO}+s1_convert:{sh(C.S1 / 's1_convert.py')}"; grad_id = f"{REPO}+grader:{sh(C.S2 / 's2_scene.py', C.S2 / 's2_grade.py', HERE / 's3_scorer.py', HERE / 's3_grade.py')}"; vsha = C.sha256_file(HERE / "vocab_10240.npy")
man = ok[~ok.dropped_d2].copy(); man = pd.DataFrame({"sample_id": man.sample_id, "clip_uuid": man.uuid, "source": man.source, "type": man.type, "split": man.split, "has_label2": man.has_label2, "direction_exempt": man.direction_exempt,
                    "anchor_t": man.t_us / 1e6, "converter_commit": conv_id, "grader_commit": np.where(man.has_label2, grad_id, ""), "vocab_sha256": vsha, "group": "chunk_" + man.chunk.astype(str).str.zfill(4), "country": man.country, "speed": man.speed, "command": man.command})
assert man.sample_id.is_unique; man.to_parquet(HERE / "manifest.parquet", index=False)
# ---------- Z1 (hard)
ids = set(man.clip_uuid); disk = {p.name for p in C.CACHE.iterdir() if p.name != "metadata"} | {p.name for p in (C.S1ROOT / "clips").iterdir()} | {p.name for p in C.SCENES.iterdir()}
z1 = dict(manifest_exam=len(ids & V["exam"]), manifest_test=len(ids & V["test"]), manifest_val=len(ids & V["val"]), disk_exam=len(disk & V["exam"]), disk_test=len(disk & V["test"]), disk_val=len(disk & V["val"]), not_raw_train=len((ids | disk) - V["V"]["raw_train"]))
assert not any(z1.values()), z1; C.assert_train(ids); C.assert_train(set(man[man.source == "nurec"].clip_uuid), nurec=True)
grp = man.groupby("group").split.nunique(); assert (grp == 1).all() and set(man.split) == {"train"} and (ci.groupby("chunk").split.nunique() == 1).all()
W("### Samples"); W()
c = ok.groupby(["source", "type"]).agg(clips=("uuid", "nunique"), anchors_converted=("sample_id", "size"), dropped_d2=("dropped_d2", "sum")).reset_index(); m2 = man.groupby(["source", "type"]).agg(samples_in_manifest=("sample_id", "size"), has_label2=("has_label2", "sum"), direction_exempt=("direction_exempt", "sum")).reset_index()
tbl(c.merge(m2, on=["source", "type"]))
W(f"Clips converted {ok.uuid.nunique()} of {len(clips)} (failed: {len(bad)}{' — ' + '; '.join(bad.status.str[:60].value_counts().head(3).index) if len(bad) else ''}); anchors {len(ok)}; in manifest {len(man)}. Bonnet mask present on {int(ok.masked_ok.sum())} of {len(ok)} samples (N = 37). Frame timing: > 20 ms from target on {int(((ok.dt_older_us.abs() > 20000) | (ok.dt_newer_us.abs() > 20000)).sum())} anchors. Fewer than 4 ego_status rows: {int((ok.n_status_rows < 4).sum())}. Command L / S / R: {ok.command.value_counts().to_dict()}."); W()
# ---------- grading
ng = ok[ok.source == "nurec"]; st = ng.grade_status.fillna("not graded").value_counts()
W("### Grading (nurec_train)"); W(); tbl(pd.DataFrame([dict(status=k, anchors=v, share=pct(v / len(ng))) for k, v in st.items()]))
gk = gr[gr.status == "ok"].copy(); gd = gr[gr.status == "dropped_d2"]; nall = len(ng)
z3 = len(gd) / nall; ex_h = int(gk.human_exempt.sum()); z4a = ex_h / nall
W(f"**D2** real vehicle box (length {gr.veh_len.min():.2f}–{gr.veh_len.max():.2f} m, width {gr.veh_wid.min():.2f}–{gr.veh_wid.max():.2f} m, {gr[['veh_len','veh_wid']].drop_duplicates().dropna().shape[0]} variants). Anchors dropped because the human path fails the drivable-area gate: **{len(gd)} of {nall} = {pct(z3)}**, in {gd.uuid.nunique()} scenes ({(gd.groupby('uuid').size() >= 16).sum()} scenes lose half or more of their anchors). By type: {ng[ng.dropped_d2].type.value_counts().to_dict()}."); W()
W(f"**D3** (declared departure from standard EPDMS): direction-gate exemption applied to the human path on **{ex_h} anchors = {pct(z4a)}** of nurec_train anchors, in {gk[gk.human_exempt].uuid.nunique()} scenes; to the stay-still plan on {int(gk.stay_exempt.sum())}; to vocabulary shapes {int(gk.vocab_exemptions.sum())} times across {int((gk.vocab_exemptions > 0).sum())} anchors. Anchors with a stationary actor ahead (condition b): {int(gk.stationary_ahead.sum())}. Human anchors whose gate stays below 1 after D3: {int((gk.human_DDC < 1).sum())}. Log: `grade_rows/*.parquet` (`human_exempt`, `human_ddc_standard`, `human_score_standard`, `human_opp_s`, `vocab_exemptions`)."); W()
W(f"Human path on graded anchors: mean score {gk.human_score.mean():.3f}, ≥ 0.8 on {pct((gk.human_score >= 0.8).mean())}; terms below 1: " + ", ".join(f"{t} {pct((gk[f'human_{t}'] < 1).mean())}" for t in ["NC", "DAC", "TLC", "DDC", "EP", "TTC", "LK", "HC"]) + f". Clock offset raw→usdz {gr.clock_offset_s.min():.2f}–{gr.clock_offset_s.max():.2f} s; position residual median {gr.groupby('uuid').align_res_m.first().median():.3f} m, scenes above 0.2 m: {int((gr.groupby('uuid').align_res_m.first() > 0.2).sum())}."); W()
# tight passes after D2 + D3
tp = pd.read_csv(C.S2 / "b_tight_events.csv"); tp["t_anchor"] = np.maximum(0.5, np.floor((tp.t_pass - 0.5) / 0.5) * 0.5).clip(upper=16.0); gi = gr[gr.sample_id != "-"].copy(); gi["ta"] = (gi.t_us / 1e6 + gi.clock_offset_s).round(1)
tpm = tp.merge(gi, left_on=["uuid", "t_anchor"], right_on=["uuid", "ta"], how="left"); tok = tpm[tpm.status == "ok"]
z4b = (tok.human_score >= 0.8).mean() if len(tok) else float("nan")
W(f"Tight passes (the 70 of S0 Run 3, same anchors as S2): graded {len(tok)}, dropped by D2 {int((tpm.status == 'dropped_d2').sum())}, not graded/unmatched {int((~tpm.status.isin(['ok', 'dropped_d2'])).sum())}. After D2 + D3: human ≥ 0.8 on **{int((tok.human_score >= 0.8).sum())} of {len(tok)} = {pct(z4b)}**; human > stay on {pct((tok.human_score > tok.stay_score).mean())}; exempted {int(tok.human_exempt.sum())}. Still below 0.8: " + ", ".join(f"{t} {int((tok[tok.human_score < 0.8][f'human_{t}'] < 1).sum())}" for t in ["NC", "DAC", "DDC", "TTC", "LK", "HC"]) + "."); W()
# anchor index emptiness
gk["sbin"] = pd.cut(gk.speed, [0, 2, 8, 16, 24, 32, 99], labels=["0–2", "2–8", "8–16", "16–24", "24–32", "≥32"], right=False)
e = gk.groupby("sbin", observed=True).agg(anchors=("sample_id", "size"), empty_orig=("n_ok_orig", lambda x: (x == 0).mean()), empty_ext=("n_ok_ext", lambda x: (x == 0).mean()), median_n_orig=("n_ok_orig", "median"), median_n_ext=("n_ok_ext", "median"), median_added=("n_ok_added", "median"), best_orig=("best_orig", "median"), best_ext=("best_ext", "median")).reset_index()
e["empty_orig"] = e.empty_orig.map(pct); e["empty_ext"] = e.empty_ext.map(pct); W("Anchor index (shapes graded ≥ 0.95, `convert_to_index.py:17-22`), original 8,192 vs extended 10,240:"); W(); tbl(e.rename(columns={"sbin": "human speed m/s"}))
hw = gk[gk.speed >= 16]; z2 = (hw.n_ok_ext > 0).mean() if len(hw) else float("nan")
W(f"Z2: extended index non-empty on {int((hw.n_ok_ext > 0).sum())} of {len(hw)} highway anchors (≥ 16 m/s) = **{pct(z2)}**; original index non-empty on {pct((hw.n_ok_orig > 0).mean())}. Shapes simulated per anchor: median {gk.n_scored.median():.0f} of 10,240 (exact prefilter, validated on 117 anchors against full scoring: identical index sizes)."); W()
# ---------- mix
W("### Split and mix"); W()
man["T"] = man.type.replace({"B_raw": "B", "B_pass": "B"}); mix = man.groupby(["split", "T"]).agg(samples=("sample_id", "size"), clips=("clip_uuid", "nunique")).reset_index(); mix["share_of_samples"] = mix.samples / len(man)
tgt = {"ordinary": 0.70, "A": 0.10, "B": 0.05}; mix["target"] = mix["T"].map(tgt); mix["relative_gap"] = (mix.share_of_samples - mix.target) / mix.target; mix["within_20pct"] = mix.relative_gap.abs() <= 0.20
z5 = bool(mix.within_20pct.all()); mm = mix.copy(); mm["share_of_samples"] = mm.share_of_samples.map(pct); mm["target"] = mm.target.map(lambda x: "≥ " + pct(x)); mm["relative_gap"] = mm.relative_gap.map(lambda x: f"{100*x:+.0f} %"); tbl(mm.rename(columns={"T": "type"}))
tbl(man.groupby(["split", "source", "type"]).agg(samples=("sample_id", "size"), clips=("clip_uuid", "nunique"), has_label2=("has_label2", "sum")).reset_index())
W(f"Groups (chunks): {man.group.nunique()}, every one in exactly one split (asserted; all {len(ci.groupby('chunk'))} raw chunks are single-split). Countries: {man.groupby('country').clip_uuid.nunique().sort_values(ascending=False).head(5).to_dict()} … ({man.country.nunique()} in total). Dedupe: 0 raw clips share a UUID with a graded NuRec scene (the 2 raw pull-outs that are NuRec scenes are carried once, as NuRec)."); W()
res = dict(Z1=z1, Z2=float(z2), Z3=float(z3), Z4_exempt=float(z4a), Z4_tight=float(z4b), Z5=z5, Z6=True, samples=len(man), clips=int(man.clip_uuid.nunique()), has_label2=int(man.has_label2.sum()))
json.dump(res, open(HERE / "checks.json", "w"), indent=1)
v = [("Z1", "zero exam / test UUIDs in any output", f"manifest and NAS folders: {z1}", "holds"), ("Z2", "extended index non-empty on ≥ 80 % of highway anchors", pct(z2), "holds" if z2 >= 0.8 else "fails"), ("Z3", "D2 drops ≤ 5 % of nurec_train anchors", pct(z3), "holds" if z3 <= 0.05 else "fails"),
     ("Z4", "D3 exemptions ≤ 3 % AND tight-pass human ≥ 0.8 on ≥ 80 %", f"{pct(z4a)}; {pct(z4b)}", "holds" if (z4a <= 0.03 and z4b >= 0.8) else "fails"), ("Z5", "mix within ±20 % of each target share", "; ".join(f"{r['T']} {pct(r.share_of_samples)}" for _, r in mix.iterrows()), "holds" if z5 else "fails"), ("Z6", "original 8,192 rows bit-identical", "verified at write time (s3_vocab.py)", "holds")]
W("### Pre-registered"); W(); tbl(pd.DataFrame(v, columns=["id", "statement", "result", "verdict"]))
open(HERE / "tables.md", "w").write("\n".join(out))
