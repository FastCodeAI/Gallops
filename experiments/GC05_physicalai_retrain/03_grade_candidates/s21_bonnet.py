#!/usr/bin/env python
"""S2.1 D1: bonnet top edge in the 512x256 model input for every S1 and S2 pilot clip.
Primary: the dataset's own ego mask (calibration/camera_intrinsics.offline, column ego_mask_image_png, native f-theta image), pushed
through the SAME rectification maps and the adapter crop/resize. Cross-check: rows that stay static across moving frames.
Run: cd paddock-sid/Alpasim && CUDA_VISIBLE_DEVICES="" alpasim/.venv/bin/python analysis_3.0/Retrain-data/03_grade_candidates/s21_bonnet.py"""
import sys, pathlib, numpy as np, pandas as pd, cv2
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import s2_common as C2; import s1_common as C1; import s1_convert as S
K = S.K; rows = []; TILES = []
def to_input(img): return cv2.resize(img[K.CROP_ROWS:-K.CROP_ROWS], K.RESIZE_WH, interpolation=cv2.INTER_LINEAR)
for stage, root, csvp in [("S1", pathlib.Path("{runs}/drivejepa_retrain/02_build_samples"), C2.S1 / "pilot_clips.csv"), ("S2", C2.NAS / "s1", HERE / "pilot_clips.csv")]:
    C1.NAS = root; clips = pd.read_csv(csvp); C1.assert_train(clips.uuid)
    for r in clips.itertuples():
        d, ts, ego, ft, R_N, t_N = S.load_clip(r.uuid, r.chunk); maps = S.build_maps(ft, R_N)
        ir = pd.read_parquet(root / f"calibration/camera_intrinsics.offline/camera_intrinsics.offline.chunk_{r.chunk:04d}.parquet").loc[(r.uuid, C1.FRONT)]
        edge_mask = np.nan; frac_bottom = np.nan
        if ir.ego_mask_image_png is not None:
            m = cv2.imdecode(np.frombuffer(ir.ego_mask_image_png, np.uint8), cv2.IMREAD_GRAYSCALE)
            egoval = int(np.median(m[-20:, 860:1060])); egom = (m == egoval).astype(np.uint8) * 255                 # ego = the value under the bottom-centre pixels
            rm = to_input(cv2.remap(egom, maps[0], maps[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)) > 127
            rr = np.where(rm[128:].mean(axis=1) >= 0.01)[0]; edge_mask = int(128 + rr.min()) if len(rr) else 256; frac_bottom = float(rm[-1].mean()); native_share = float((m == egoval).mean())
        # static cross-check on frames where the car moves
        spd = np.hypot(np.interp(ts, ego.ts, ego.v_rig[:, 0]), np.interp(ts, ego.ts, ego.v_rig[:, 1]))
        mov = np.where(spd > 3.0)[0]; edge_static = np.nan; tile = None
        if len(mov) >= 40:
            pick = sorted(set(mov[np.linspace(0, len(mov) - 1, 24).astype(int)].tolist())); fr = S.decode(d / f"{r.uuid}.{C1.FRONT}.mp4", pick)
            inp = [to_input(S.rectify(fr[j], maps)) for j in pick]; X = np.array([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(float) for x in inp])
            # The bonnet's upper edge is the one edge that stays put while the scene moves: average |d/dy| over the moving frames
            # (scene edges smear out, the bonnet edge adds up). Per 16-column bin, the edge row = strongest averaged gradient in
            # rows 176-253; the bonnet top = the highest such point across bins that carry a clear edge (>= 2.5 x the bin's median).
            Gm = np.mean([np.abs(cv2.Sobel(cv2.GaussianBlur(x, (3, 3), 0), cv2.CV_64F, 0, 1, ksize=3)) for x in X], axis=0); pts = []
            for c0 in range(0, 512, 16):
                prof = Gm[176:254, c0:c0 + 16].mean(axis=1); k = int(prof.argmax())
                if prof[k] >= 2.5 * np.median(prof) and prof[k] >= 6.0: pts.append((c0 + 8, 176 + k))
            cen = [p for p in pts if 96 <= p[0] <= 416]; med = float(np.median([p[1] for p in cen])) if len(cen) >= 6 else np.nan
            good = [p for p in cen if abs(p[1] - med) <= 14]                      # the bonnet outline is smooth; isolated points far from it are scene edges
            e = int(min(p[1] for p in good)) if len(good) >= 6 else 256; n_bins = len(good); pts = good
            edge_static = e; tile = inp[len(inp) // 2].copy(); cv2.line(tile, (0, e), (511, e), (0, 255, 255), 1)
            for p in pts: cv2.circle(tile, p, 2, (0, 0, 255), -1)
            if edge_mask == edge_mask and edge_mask < 256: cv2.line(tile, (0, int(edge_mask)), (511, int(edge_mask)), (255, 0, 255), 1)
            cv2.putText(tile, f"{stage} {r.type} {r.uuid[:8]} rows {256 - e}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1); TILES.append(tile[128:])
        rows.append(dict(stage=stage, type=r.type, uuid=r.uuid, top_edge_mask_row=edge_mask, bonnet_rows_mask=256 - edge_mask if edge_mask == edge_mask else np.nan, mask_share_last_row=frac_bottom, top_edge_static_row=edge_static, bonnet_rows_static=256 - edge_static if edge_static == edge_static else np.nan, n_moving_frames=len(mov)))
        print(rows[-1], flush=True)
o = pd.DataFrame(rows); o.to_csv(HERE / "s21_bonnet.csv", index=False)
while len(TILES) % 3: TILES.append(np.zeros_like(TILES[0]))
(HERE / "s21_png").mkdir(exist_ok=True); cv2.imwrite(str(HERE / "s21_png" / "bonnet_edges.png"), np.vstack([np.hstack(TILES[k:k + 3]) for k in range(0, len(TILES), 3)]))
print("mask: rows min/med/max", o.bonnet_rows_mask.min(), o.bonnet_rows_mask.median(), o.bonnet_rows_mask.max(), "| static:", o.bonnet_rows_static.min(), o.bonnet_rows_static.median(), o.bonnet_rows_static.max())
