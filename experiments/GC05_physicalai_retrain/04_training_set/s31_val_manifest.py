"""S3.1 V3: validation manifest (same schema as train) + asserts. Run with S3_SPLIT=val through run_s3_grade.sh (SCRIPT=s31_val_manifest.py)."""
import glob, json, hashlib, subprocess, numpy as np, pandas as pd
import s3_common as C
assert C.SPLIT == "val"; V = C.views(); ci = V["ci"]; clips = pd.read_csv(C.HERE / "clips.csv")
cv = pd.concat([pd.read_csv(p) for p in glob.glob(str(C.HERE / "convert_rows/*.csv"))], ignore_index=True); gr = pd.concat([pd.read_parquet(p) for p in glob.glob(str(C.HERE / "grade_rows/*.parquet"))], ignore_index=True)
ok = cv[cv.status == "ok"].merge(clips[["uuid", "source", "type", "chunk", "split", "country"]], on="uuid"); g = gr[gr.sample_id != "-"].set_index("sample_id")
ok["gs"] = ok.sample_id.map(g.status); ok["has_label2"] = (ok.source == "nurec") & (ok.gs == "ok"); ok["direction_exempt"] = ok.sample_id.map(g.human_exempt).fillna(False).astype(bool) & ok.has_label2
REPO = subprocess.run(["git", "-C", str(C.CODE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(); sh = lambda *ps: hashlib.sha256(b"".join(open(p, "rb").read() for p in ps)).hexdigest()[:12]
conv = f"{REPO}+s1_convert:{sh(C.S1 / 's1_convert.py')}"; grad = f"{REPO}+grader:{sh(C.S2 / 's2_scene.py', C.S2 / 's2_grade.py', C.CODE / 's3_scorer.py', C.CODE / 's3_grade.py')}"
k = ok[ok.gs != "dropped_d2"]; man = pd.DataFrame({"sample_id": k.sample_id, "clip_uuid": k.uuid, "source": k.source, "type": k.type, "split": k.split, "has_label2": k.has_label2, "direction_exempt": k.direction_exempt, "anchor_t": k.t_us / 1e6,
        "converter_commit": conv, "grader_commit": np.where(k.has_label2, grad, ""), "vocab_sha256": C.sha256_file(C.CODE / "vocab_10240.npy"), "group": "chunk_" + k.chunk.astype(str).str.zfill(4), "country": k.country, "speed": k.speed, "command": k.command})
tr = pd.read_parquet(C.CODE / "manifest.parquet"); assert list(man.columns) == list(tr.columns) and man.sample_id.is_unique and set(man.split) == {"val"}
ids = set(man.clip_uuid); C.assert_train(ids); C.assert_train(set(man[man.source == "nurec"].clip_uuid), nurec=True)          # val mode: raw_val / nurec_val, never exam / test / train
assert not (ids & V["exam"]) and not (ids & V["test"]) and not (ids & set(tr.clip_uuid)); assert not (set(man.group) & set(tr.group)), "val chunk also in train"
assert not (set(man.sample_id) & set(tr.sample_id)); assert (ci.loc[sorted(ids), "split"] == "val").all()
man.to_parquet(C.HERE / "manifest_val.parquet", index=False)
gk = gr[gr.status == "ok"]; sc = gr[gr.sample_id != "-"].groupby("uuid").status.value_counts().unstack(fill_value=0)
print("clips", len(clips), "converted", ok.uuid.nunique(), "anchors", len(ok), "masked", int(ok.masked_ok.sum()), "| manifest rows", len(man), "has_label2", int(man.has_label2.sum()), "direction_exempt", int(man.direction_exempt.sum()))
print(man.groupby(["source", "type"]).agg(samples=("sample_id", "size"), clips=("clip_uuid", "nunique"), has_label2=("has_label2", "sum")).reset_index().to_string(index=False))
print("grade status per scene:"); print(sc.to_string()); print("scenes graded (>= 1 ok anchor):", int((sc.get("ok", 0) > 0).sum()), "of", int((clips.source == "nurec").sum()), "| scene errors:", int(gr.status.str.startswith("scene error").sum()))
print("status totals", gr.status.str[:40].value_counts().to_dict()); print("human >= 0.8:", round(float((gk.human_score >= 0.8).mean()), 3), "| D2 drops", int((gr.status == "dropped_d2").sum()), "| exempt", int(gk.human_exempt.sum()))
hw = gk[gk.speed >= 16]; print("highway anchors", len(hw), "ext index non-empty", round(float((hw.n_ok_ext > 0).mean()), 3) if len(hw) else None, "orig", round(float((hw.n_ok_orig > 0).mean()), 3) if len(hw) else None, "| all anchors empty ext", round(float((gk.n_ok_ext == 0).mean()), 3))
print("asserts passed: schema == train, split == val, 0 exam, 0 test, 0 train uuid, 0 shared chunks with train (val chunks", man.group.nunique(), ", train chunks", tr.group.nunique(), ")")
json.dump(dict(rows=len(man), clips=int(man.clip_uuid.nunique()), has_label2=int(man.has_label2.sum()), scenes_graded=int((sc.get("ok", 0) > 0).sum())), open(C.HERE / "checks_val.json", "w"))
