"""Scene sets: pilot 10 (4 A, 4 tight-pass, 2 ordinary; seed 20261005) + Step-5 set (all nurec_train tight passes, 2 nurec pull-outs)."""
import random, pandas as pd, s2_common as C
S0 = C.S0; V = C.S0C.load_split_views(); _, exam, _ = C.S0C.load_views()
ci = pd.read_parquet(C.S0C.RAW_SNAP / "clip_index.parquet"); fp = pd.read_parquet(C.S0C.RAW_SNAP / "metadata/feature_presence.parquet")
need = ["camera_front_wide_120fov", "camera_intrinsics.offline", "sensor_extrinsics.offline", "egomotion.offline", "vehicle_dimensions"]
ok = set(fp.index[fp[need].all(axis=1)]); exam_chunks = set(ci.loc[[u for u in exam if u in ci.index]].chunk)
elig = lambda us: sorted(u for u in us if u in V["nurec_train"] and u in ok and ci.loc[u, "chunk"] not in exam_chunks)
sc = pd.read_csv(S0 / "d2_scenes.csv"); an = pd.read_csv(S0 / "d3_anchors.csv", low_memory=False); mo = pd.read_csv(S0 / "d3_moveoffs.csv"); ar = pd.read_csv(S0 / "d3_around.csv")
tight = ar[(ar.split == "train") & (ar.clearance <= 1.0)].copy(); assert len(tight) == 70 and tight.uuid.nunique() == 33
A = elig(set(sc[sc.vheld >= 20.0].uuid)); T = elig(set(tight.uuid))
has_A = set(an[an.A == True].uuid); has_B = set(mo.uuid); mid = set(an[(an.v >= 5) & (an.v <= 15)].uuid)
O = elig((V["nurec_train"] - has_A - has_B - set(ar.uuid)) & mid)
rng = random.Random(20261005); pick = [("A", u) for u in rng.sample(A, 4)] + [("T", u) for u in rng.sample(T, 4)] + [("O", u) for u in rng.sample(O, 2)]
df = pd.DataFrame(pick, columns=["type", "uuid"]); df["chunk"] = df.uuid.map(ci.chunk); df["split"] = df.uuid.map(ci.split); C.assert_nurec_train(df.uuid); assert df.uuid.nunique() == 10
df.to_csv(C.HERE / "pilot_clips.csv", index=False)
lat = mo[(mo.split == "train") & (mo.lat4 >= 1.0) & (mo.actor_bum == True)]; assert len(lat) == 2
tight.to_csv(C.HERE / "b_tight_events.csv", index=False); lat.to_csv(C.HERE / "b_pullout_events.csv", index=False)
allsc = sorted(set(df.uuid) | set(tight.uuid) | set(lat.uuid)); C.assert_nurec_train(allsc)
pd.DataFrame({"uuid": allsc, "pilot": [u in set(df.uuid) for u in allsc], "tight": [u in set(tight.uuid) for u in allsc], "pullout": [u in set(lat.uuid) for u in allsc]}).to_csv(C.HERE / "scenes.csv", index=False)
print("eligible A", len(A), "T", len(T), "O", len(O)); print(df.to_string()); print("scenes total", len(allsc), "tight events", len(tight), "in scenes", tight.uuid.nunique(), "pull-outs", lat.uuid.tolist())
