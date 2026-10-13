"""S4 Part A2: full validation manifest (same schema as train) + asserts + counts. Run with S3_SPLIT=val_full through run_s3_grade.sh
(SCRIPT=s4a_manifest.py). Same rules as s31_val_manifest.py: D2-dropped moments are left out; converted NuRec moments that were not
graded (scene error, ego off every lane, no 4 s future) stay as generator-only rows (has_label2 = false). Writes A2_DONE.json LAST."""
import glob, json, hashlib, subprocess, numpy as np, pandas as pd
import s3_common as C
assert C.SPLIT == "val" and C.VAL_DIR == "val_full"; V = C.views(); ci = V["ci"]; clips = pd.read_csv(C.HERE / "clips.csv")
cv = pd.concat([pd.read_csv(p) for p in glob.glob(str(C.HERE / "convert_rows/*.csv"))], ignore_index=True); gr = pd.concat([pd.read_parquet(p) for p in glob.glob(str(C.HERE / "grade_rows/*.parquet"))], ignore_index=True)
cv = cv[cv.uuid.isin(clips.uuid)]; gr = gr[gr.uuid.isin(clips.uuid)]
ok = cv[cv.status == "ok"].merge(clips[["uuid", "source", "type", "chunk", "split", "country"]], on="uuid"); bad = cv[cv.status != "ok"]; g = gr[gr.sample_id != "-"].set_index("sample_id")
ok["gs"] = ok.sample_id.map(g.status); ok["has_label2"] = (ok.source == "nurec") & (ok.gs == "ok"); ok["direction_exempt"] = ok.sample_id.map(g.human_exempt).fillna(False).astype(bool) & ok.has_label2
REPO = subprocess.run(["git", "-C", str(C.CODE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(); sh = lambda *ps: hashlib.sha256(b"".join(open(p, "rb").read() for p in ps)).hexdigest()[:12]
conv = f"{REPO}+s1_convert:{sh(C.S1 / 's1_convert.py')}"; grad = f"{REPO}+grader:{sh(C.S2 / 's2_scene.py', C.S2 / 's2_grade.py', C.CODE / 's3_scorer.py', C.CODE / 's3_grade.py')}"
k = ok[ok.gs != "dropped_d2"]; man = pd.DataFrame({"sample_id": k.sample_id, "clip_uuid": k.uuid, "source": k.source, "type": k.type, "split": k.split, "has_label2": k.has_label2, "direction_exempt": k.direction_exempt, "anchor_t": k.t_us / 1e6,
        "converter_commit": conv, "grader_commit": np.where(k.has_label2, grad, ""), "vocab_sha256": C.sha256_file(C.CODE / "vocab_10240.npy"), "group": "chunk_" + k.chunk.astype(str).str.zfill(4), "country": k.country, "speed": k.speed, "command": k.command})
tr = pd.read_parquet(C.CODE / "manifest.parquet"); assert list(man.columns) == list(tr.columns) and list(man.dtypes) == list(tr.dtypes) and man.sample_id.is_unique and set(man.split) == {"val"}
ids = set(man.clip_uuid); C.assert_train(ids); C.assert_train(set(man[man.source == "nurec"].clip_uuid), nurec=True)          # val mode: raw_val / nurec_val, never exam / test / train
assert not (ids & V["exam"]) and not (ids & V["test"]) and not (ids & set(tr.clip_uuid)); assert not (set(man.group) & set(tr.group)), "val chunk also in train"
assert not (set(man.sample_id) & set(tr.sample_id)); assert (ci.loc[sorted(ids), "split"] == "val").all()
for sid, u in zip(man.sample_id[man.has_label2], man.clip_uuid[man.has_label2]):                                               # every graded row has its files on the NAS
    d = C.CACHE / u / sid; assert (d / "metric_cache.pkl").exists() and (d / "anchors_scores_index_ext.npy").exists() and (d / "pad_label2.gz").exists(), sid
man.to_parquet(C.HERE / "manifest_val_full.parquet", index=False)
ok["d2"] = ok.gs == "dropped_d2"; ok["in_manifest"] = ~ok.d2
t = ok.groupby(["source", "type"]).agg(clips=("uuid", "nunique"), anchors_converted=("sample_id", "size"), d2_drops=("d2", "sum"), samples=("in_manifest", "sum"), has_label2=("has_label2", "sum"), direction_exempt=("direction_exempt", "sum")).reset_index()
t["graded_share"] = (t.has_label2 / t.samples).round(3); print(t.to_string(index=False)); t.to_csv(C.HERE / "counts.csv", index=False)
sc = gr[gr.sample_id != "-"].groupby("uuid").status.value_counts().unstack(fill_value=0); nn = clips[clips.source == "nurec"]; graded = set(sc.index[sc.get("ok", 0) > 0]) if len(sc) else set()
serr = gr[gr.status.str.startswith("scene error")]; nomap = [u for u in nn.uuid if not (C.SCENES / u / "DONE").exists()]; norow = sorted(set(nn.uuid) - set(gr.uuid))
print("clips", len(clips), "converted", ok.uuid.nunique(), "| conversion errors", len(bad), list(bad.status.str[:70].value_counts().head(4).items()), "| clips without convert row", len(set(clips.uuid) - set(cv.uuid)))
print("anchors", len(ok), "masked", int(ok.masked_ok.sum()), "| manifest rows", len(man), "has_label2", int(man.has_label2.sum()), f"({man.has_label2.mean():.3f})", "direction_exempt", int(man.direction_exempt.sum()))
print("nurec scenes", len(nn), "| graded (>= 1 ok anchor)", len(graded), "| scene errors", len(serr), list(serr.status.str[:90].value_counts().head(4).items()), "| no map", len(nomap), "| no grade row file", len(norow))
print("anchor status totals", gr.status.str[:40].value_counts().to_dict()); gk = gr[gr.status == "ok"]; nu = ok[ok.source == "nurec"]
print("nurec moments kept ungraded (has_label2 false):", int((~nu.has_label2 & ~nu.d2).sum()), "| D2 drops", int(nu.d2.sum()), f"({nu.d2.mean():.3f} of nurec anchors) | human >= 0.8:", round(float((gk.human_score >= 0.8).mean()), 3), "| exempt", int(gk.human_exempt.sum()))
hw = gk[gk.speed >= 16]; print("highway anchors", len(hw), "ext index non-empty", round(float((hw.n_ok_ext > 0).mean()), 3) if len(hw) else None, "orig", round(float((hw.n_ok_orig > 0).mean()), 3) if len(hw) else None)
print("asserts passed: schema + dtypes == train, split == val, 0 exam, 0 test, 0 train uuid, 0 shared sample ids, 0 shared chunks with train (val chunks", man.group.nunique(), ", train chunks", tr.group.nunique(), "), graded rows have their NAS files")
done = dict(rows=len(man), clips=int(man.clip_uuid.nunique()), has_label2=int(man.has_label2.sum()), graded_share=float(man.has_label2.mean()), direction_exempt=int(man.direction_exempt.sum()), d2_drops=int(nu.d2.sum()),
            scenes=len(nn), scenes_graded=len(graded), scene_errors=len(serr), scenes_no_map=len(nomap), conversion_errors=len(bad), chunks=int(man.group.nunique()),
            per_type=[{kk: (vv.item() if hasattr(vv, "item") else vv) for kk, vv in r.items()} for r in t.to_dict("records")], manifest=str(C.HERE / "manifest_val_full.parquet"), complete=bool(not norow and not len(set(clips.uuid) - set(cv.uuid))))
json.dump(done, open(C.HERE / ("A2_DONE.json" if done["complete"] else "A2_PARTIAL.json"), "w"), indent=1); print("complete:", done["complete"])
