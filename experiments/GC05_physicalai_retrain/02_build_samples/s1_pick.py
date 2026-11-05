"""STEP 2a: seeded pick of 20 raw_train pilot clips: 5 A (>= 20 m/s held 4.5 s), 5 raw pull-outs, 10 ordinary."""
import random, pandas as pd, s1_common as C
S0 = C.S0; V = C.S0C.load_split_views(); _, exam, _ = C.S0C.load_views()
ci = pd.read_parquet(C.S0C.RAW_SNAP / "clip_index.parquet"); fp = pd.read_parquet(C.S0C.RAW_SNAP / "metadata/feature_presence.parquet")
need = [C.FRONT, "camera_intrinsics.offline", "sensor_extrinsics.offline", "egomotion.offline", "vehicle_dimensions"]
ok = set(fp.index[fp[need].all(axis=1)]); exam_chunks = set(ci.loc[[u for u in exam if u in ci.index]].chunk)
elig = lambda us: sorted(u for u in us if u in V["raw_train"] and u in ok and ci.loc[u, "chunk"] not in exam_chunks)
sc = pd.read_csv(S0 / "d2_scenes.csv"); an = pd.read_csv(S0 / "d2_anchors.csv", low_memory=False); mo = pd.read_csv(S0 / "d2_moveoffs.csv")
A = elig(set(sc[sc.vheld >= 20.0].uuid) & V["nurec_train"])
r = pd.read_csv(S0 / "r3_raw_moveoffs.csv"); B = elig(set(r[(r.actor_bum == True) & (r.dyaw4_deg.abs() <= 45)].uuid))
has_A = set(an[an.A == True].uuid); has_B = set(mo.uuid); mid = set(an[(an.v >= 5) & (an.v <= 15)].uuid)
O = elig((V["nurec_train"] - has_A - has_B) & mid)
rng = random.Random(20261004); pick = [("A", u) for u in rng.sample(A, 5)] + [("B", u) for u in rng.sample(B, 5)] + [("O", u) for u in rng.sample(O, 10)]
df = pd.DataFrame(pick, columns=["type", "uuid"]); df["chunk"] = df.uuid.map(ci.chunk); df["split"] = df.uuid.map(ci.split)
C.assert_train(df.uuid); assert df.uuid.nunique() == 20
df.to_csv(C.HERE / "pilot_clips.csv", index=False)
print("eligible: A", len(A), "B", len(B), "ordinary", len(O), "| seed 20261004"); print(df.to_string())
