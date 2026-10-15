#!/usr/bin/env python
"""B3: 20 spot-check PNGs (5 per type). Label-1 path (white) and, where graded, the 3 best (green) and 3 worst (red) vocabulary shapes,
drawn on the masked 512x256 model input. Run: cd paddock-sid/Alpasim && alpasim/.venv/bin/python analysis_3.0/Retrain-data/04_training_set/s3_spotcheck.py"""
import sys, gzip, pickle, random, pathlib, numpy as np, pandas as pd, cv2
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import s3_common as C; import s1_common as C1; C1.NAS = C.S1ROOT; import s1_convert as S
K = S.K; OUTD = C.HERE; man = pd.read_parquet(OUTD / ("manifest_val.parquet" if C.SPLIT == "val" else "manifest.parquet")); C.assert_train(man.clip_uuid.unique()); clips = pd.read_csv(OUTD / "clips.csv").set_index("uuid")
VOC = np.load(C.CODE / "vocab_10240.npy")[:, 4::5]; rng = random.Random(20261006); (OUTD / "spot_png").mkdir(exist_ok=True)
mo = pd.read_csv(C.S0 / "r3_raw_moveoffs.csv").drop_duplicates("uuid").set_index("uuid")
def choose(df, n): return df.loc[rng.sample(sorted(df.index), min(n, len(df)))]
m = man.set_index("sample_id")
braw = m[m.type == "B_raw"].copy(); braw["d"] = (braw.anchor_t - braw.clip_uuid.map(mo.t_move)).abs(); braw = braw.sort_values("d").groupby("clip_uuid").head(1)
picks = pd.concat([choose(m[(m.type == "A") & m.has_label2 & (m.speed >= 16)], 5), choose(braw, 5), choose(m[(m.type == "B_pass") & m.has_label2 & (m.speed > 2)], 5), choose(m[(m.type == "ordinary") & m.has_label2 & (m.speed > 3)], 3), choose(m[(m.type == "ordinary") & (m.source == "raw") & (m.speed > 3)], 2)])
def proj(pts, t_cam):
    P3 = np.c_[pts[:, :2], np.zeros(len(pts))]; pc = (S.R_V.T @ (P3 - t_cam).T).T; ok = pc[:, 2] > 1.0; uv = np.full((len(pc), 2), np.nan)
    if ok.any():
        n_ = pc[ok, :2] / pc[ok, 2:3]; good = (n_ ** 2).sum(1) < 0.81; q = cv2.projectPoints(pc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), S.K_V, S.DIST_V)[0].reshape(-1, 2); q[~good] = np.nan; uv[ok] = q
    return np.c_[uv[:, 0] * 512 / 1920, (uv[:, 1] - K.CROP_ROWS) * 256 / (1080 - 2 * K.CROP_ROWS)]
def draw(img, pts, col, th):
    q = pts[np.isfinite(pts).all(1)] * 3
    for a, b in zip(q[:-1], q[1:]): cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), col, th, cv2.LINE_AA)
    for a in q: cv2.circle(img, tuple(a.astype(int)), th + 1, col, -1, cv2.LINE_AA)
if C.SPLIT == "val":      # S3.1 V4: 4 PNGs
    picks = pd.concat([choose(m[(m.type == "A") & m.has_label2 & (m.speed >= 16)], 1), choose(m[(m.type == "B_pass") & m.has_label2 & (m.speed > 2)], 1), choose(m[(m.type == "ordinary") & m.has_label2 & (m.speed > 3)], 1), choose(m[(m.source == "raw") & (m.speed > 3)], 1)])
rows = []; tiles = []
for sid, r in picks.iterrows():
    d = C.CACHE / r.clip_uuid / sid
    with gzip.open(d / "pad_feature.gz", "rb") as f: feat = pickle.load(f)
    with gzip.open(d / "pad_target.gz", "rb") as f: lab = pickle.load(f)["trajectory"].numpy()
    img = (feat["camera_feature_1"].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8); img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (1536, 768), interpolation=cv2.INTER_LINEAR)
    _, _, _, _, _, t_N = S.load_clip(r.clip_uuid, int(clips.loc[r.clip_uuid, "chunk"])); note = "label 1 only"
    if r.has_label2:
        fin = np.load(d / "anchors_scores_final.npy").astype(np.float32); sc = np.where(np.isfinite(fin))[0]; o = sc[np.argsort(fin[sc], kind="stable")]
        for j in o[:3]: draw(img, proj(VOC[j], t_N), (60, 60, 230), 1)
        for j in o[::-1][:3]: draw(img, proj(VOC[j], t_N), (80, 200, 80), 2)
        note = f"best {fin[o[-1]]:.2f} (rows {', '.join(str(int(j)) for j in o[::-1][:3])}) worst {fin[o[0]]:.2f}; scored {len(sc)}"
    draw(img, proj(lab, t_N), (255, 255, 255), 2)
    cv2.putText(img, f"{r.type} {r.source} {sid} v={r.speed:.1f} m/s cmd={r.command} | {note}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    fn = f"{r.type}_{sid}.png"; cv2.imwrite(str(OUTD / "spot_png" / fn), img); tiles.append(cv2.resize(img, (512, 256))); rows.append(dict(type=r.type, source=r.source, sample_id=sid, speed=round(r.speed, 1), has_label2=bool(r.has_label2), direction_exempt=bool(r.direction_exempt), png=f"spot_png/{fn}", note=note))
pd.DataFrame(rows).to_csv(OUTD / "spot_index.csv", index=False)
while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(OUTD / "spot_png" / "_contact_sheet.png"), np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])); print(pd.DataFrame(rows).to_string())
