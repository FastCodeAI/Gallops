"""run 3B v2 camera augmentation, on the GPU, applied by the trainer after loading and before normalisation (run3b/CLAUDE.md
section 1 'Augmentation'). Phase 1's rules (run3a-adapted/phase1/data/augment.py, its values as the defaults) + a field-of-view
jitter, with lidar2img recomputed in run 3A's convention (D derives its rays from to_512 @ lidar2img):

  angle jitter   per camera a rotation R = Rz Ry Rx (camera axes), each angle uniform in [-angle_deg, angle_deg]; both frames
                 of the camera are warped by K' R K^-1; ego2cam' = R ego2cam.                         (Phase 1: 1.0 deg)
  position jitter  ego2cam' translation += uniform [-shift_m, shift_m] per camera axis; calibration only, image not warped.
                                                                                                       (Phase 1: 0.05 m)
  FOV jitter     per camera focal scale s uniform in [1 - fov, 1 + fov]: K' = K with fx, fy x s (principal point kept), image
                 warped by K' K^-1 (one warp with the rotation: K' R K^-1).                          (no value given: 0.0)
  zoom (FOV)     Navvrat 19 Nov: "zoom each image by a random factor between 1.00 and 1.10 (crop the centre, resize back to
                 512 x 256), and update the intrinsics, lidar2img and rays to match. One factor per sample, the same for all four
                 cameras. Applied to 50% of samples." K' = Z K with Z = zoom z about the image centre ((W-1)/2, (H-1)/2) in the
                 pixel convention below; image warped with the rotation in one pass (K' R K^-1). Knobs zoom_max 0.10, zoom_p 0.5.
  camera dropout whole cameras black (0) in both frames, rear included. drop_mode 'sample' (spec v2 wording, default): in a
                 share drop_p of samples one camera, uniform over the 4; 'camera' (Phase 1's rule): every camera with p drop_p.
                                                                                                       (0.1)
  lidar2img      run 3A convention: inv(TO_512) @ K4' @ ego2cam'. When the batch's lidar2img is passed, the change is applied to
                 it (inv(TO_512) K4' D R K4^-1 TO_512 @ lidar2img, float64), so an unchanged camera keeps its matrix.

Pixel convention as Phase 1 (pixel (i, j) has its centre at x = j, y = i; grid_sample align_corners=False), bilinear, zeros
outside. With every knob 0 the images and lidar2img are returned unchanged (bit for bit)."""
import math

import torch
import torch.nn.functional as F

TO_512 = [[2 / 3, 0, 0, 0], [0, 0.625, -7, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
KNOBS = dict(angle_deg=1.0, shift_m=0.05, fov=0.0, drop_p=0.1, drop_mode="sample", zoom_max=0.10, zoom_p=0.5)  # Phase 1 / Navvrat


def _euler_to_R(rx, ry, rz):
    """Phase 1's augment._euler_to_R (Rz @ Ry @ Rx), float64."""
    cx, sx, cy, sy, cz, sz = rx.cos(), rx.sin(), ry.cos(), ry.sin(), rz.cos(), rz.sin()
    one, zero = torch.ones_like(rx), torch.zeros_like(rx)
    Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], -1).view(-1, 3, 3)
    Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], -1).view(-1, 3, 3)
    Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], -1).view(-1, 3, 3)
    return Rz @ Ry @ Rx


def warp(images, H_src_from_dst, chunk=64):
    """images (N, C, H, W) float; H (N, 3, 3): source pixel = H @ destination pixel (homogeneous). Phase 1's warp_rotation."""
    N, C, Hh, W = images.shape
    dev = images.device
    ys, xs = torch.meshgrid(torch.arange(Hh, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    out = torch.empty_like(images)
    for a in range(0, N, chunk):
        h = H_src_from_dst[a:a + chunk].float()[:, :, :, None, None]           # (n, 3, 3, 1, 1)
        den = h[:, 2, 0] * xs + h[:, 2, 1] * ys + h[:, 2, 2]
        sx = (h[:, 0, 0] * xs + h[:, 0, 1] * ys + h[:, 0, 2]) / den
        sy = (h[:, 1, 0] * xs + h[:, 1, 1] * ys + h[:, 1, 2]) / den
        grid = torch.stack([(sx + 0.5) / W * 2 - 1, (sy + 0.5) / Hh * 2 - 1], -1)
        out[a:a + chunk] = F.grid_sample(images[a:a + chunk], grid.to(images.dtype), mode="bilinear", padding_mode="zeros",
                                         align_corners=False)
    return out


def lidar2img_3a(K, ego2cam):
    """K (..., 3, 3), ego2cam (..., 4, 4) of the 512 x 256 image -> run 3A's lidar2img (..., 4, 4) float32 (computed in float64)."""
    K4 = torch.zeros(*K.shape[:-2], 4, 4, dtype=torch.float64, device=K.device)
    K4[..., :3, :3] = K.double()
    K4[..., 3, 3] = 1
    T = torch.tensor(TO_512, dtype=torch.float64, device=K.device)
    return (torch.linalg.inv(T) @ K4 @ ego2cam.double()).float()


def augment(images, K, ego2cam, generator=None, knobs=None, lidar2img=None, out_float=False, chunk=64):
    """images uint8 (B, 4, T, 3, H, W) on the GPU; K (B, 4, 3, 3); ego2cam (B, 4, 4, 4); generator: torch.Generator on the
    images' device (or None); knobs: dict with KNOBS' keys (missing keys take KNOBS' values); lidar2img (B, 4, 4, 4) run 3A
    convention, optional (see module doc).
    -> images (uint8, rounded; float32 [0, 255] if out_float), lidar2img (B, 4, 4, 4) float32 run 3A convention, drop (B, 4) bool,
       info dict (per camera: angles_rad (B, 4, 3), shift_m (B, 4, 3), focal_scale (B, 4); K_new, ego2cam_new)."""
    kn = dict(KNOBS, **(knobs or {}))
    B, V, T, C, H, W = images.shape
    N, dev = B * V, images.device
    f64 = dict(device=dev, dtype=torch.float64)
    # random numbers: always drawn in this order and amount, whatever the knobs (the stream does not depend on them)
    u_ang = torch.rand(N, 3, generator=generator, device=dev).double() * 2 - 1
    u_sh = torch.rand(N, 3, generator=generator, device=dev).double() * 2 - 1
    u_fov = torch.rand(N, generator=generator, device=dev).double() * 2 - 1
    u_drop = torch.rand(B, V, generator=generator, device=dev)
    u_cam = torch.randint(0, V, (B,), generator=generator, device=dev)
    u_zoom = torch.rand(B, generator=generator, device=dev).double()     # drawn after the others: their stream is unchanged
    u_zoom_on = torch.rand(B, generator=generator, device=dev)
    ang = u_ang * math.radians(kn["angle_deg"])
    sh = u_sh * kn["shift_m"]
    s = 1 + u_fov * kn["fov"]
    if kn["drop_mode"] == "sample":
        drop = torch.zeros(B, V, dtype=torch.bool, device=dev)
        drop[torch.arange(B, device=dev), u_cam] = u_drop[:, 0] < kn["drop_p"]
    elif kn["drop_mode"] == "camera":
        drop = u_drop < kn["drop_p"]
    else:
        raise ValueError(kn["drop_mode"])

    Kf = K.reshape(N, 3, 3).double()
    E = ego2cam.reshape(N, 4, 4).double()
    R = _euler_to_R(ang[:, 0], ang[:, 1], ang[:, 2])
    Kn = Kf.clone()
    Kn[:, 0, 0] *= s
    Kn[:, 1, 1] *= s
    Kn[:, 0, 1] *= s
    z = torch.where(u_zoom_on < kn["zoom_p"], 1 + u_zoom * kn["zoom_max"], torch.ones_like(u_zoom))   # (B,) one per sample
    zN = z.repeat_interleave(V)
    Z = torch.eye(3, **f64).repeat(N, 1, 1)
    Z[:, 0, 0], Z[:, 1, 1] = zN, zN
    Z[:, 0, 2], Z[:, 1, 2] = (1 - zN) * (W - 1) / 2, (1 - zN) * (H - 1) / 2
    Kn = Z @ Kn
    R4 = torch.eye(4, **f64).repeat(N, 1, 1)
    R4[:, :3, :3] = R
    D4 = torch.eye(4, **f64).repeat(N, 1, 1)
    D4[:, :3, 3] = sh
    En = D4 @ R4 @ E
    zoom_on = kn["zoom_max"] != 0 and kn["zoom_p"] > 0
    geo = kn["angle_deg"] != 0 or kn["shift_m"] != 0 or kn["fov"] != 0 or zoom_on
    warp_on = kn["angle_deg"] != 0 or kn["fov"] != 0 or zoom_on

    if warp_on:
        Hs = Kf @ R.transpose(1, 2) @ torch.linalg.inv(Kn)          # source pixel from destination pixel
        x = warp(images.reshape(N, T * C, H, W).float(), Hs, chunk).view(B, V, T, C, H, W)
        out = x if out_float else x.round_().clamp_(0, 255).to(torch.uint8)
    else:
        out = images.float() if out_float else images
    if drop.any():
        out = out.masked_fill(drop[:, :, None, None, None, None], 0)

    if not geo:
        l2i = lidar2img if lidar2img is not None else lidar2img_3a(K, ego2cam)
    elif lidar2img is not None:
        K4 = torch.eye(4, **f64).repeat(N, 1, 1)
        K4[:, :3, :3] = Kf
        Kn4 = torch.eye(4, **f64).repeat(N, 1, 1)
        Kn4[:, :3, :3] = Kn
        Tm = torch.tensor(TO_512, **f64)
        M = torch.linalg.inv(Tm) @ Kn4 @ D4 @ R4 @ torch.linalg.inv(K4) @ Tm
        l2i = (M @ lidar2img.reshape(N, 4, 4).double()).float().view(B, V, 4, 4)
    else:
        l2i = lidar2img_3a(Kn, En).view(B, V, 4, 4)
    info = dict(angles_rad=ang.view(B, V, 3), shift_m=sh.view(B, V, 3), focal_scale=s.view(B, V), zoom=z.float(),
                K_new=Kn.float().view(B, V, 3, 3), ego2cam_new=En.float().view(B, V, 4, 4))
    return out, l2i, drop, info
