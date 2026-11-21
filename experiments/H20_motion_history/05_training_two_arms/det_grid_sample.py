"""A repeatable backward for F.grid_sample (TRAIN step 1: resume must reproduce a straight run exactly).

Why: the planner's deformable attention runs in its pure-PyTorch form (DRIVE_JEPA_MSDA=pytorch, mmcv's
multi_scale_deformable_attn_pytorch), which calls F.grid_sample. PyTorch's CUDA backward for it (grid_sampler_2d_backward_cuda) adds
into the value gradient with atomics and has no deterministic version, so the same step twice gives gradients differing in the last bits
(out/det_probe_plain.json: 33 of 79 tensors, all upstream of that attention, up to 0.004) and two runs drift apart from step 2.

What: forward = the original F.grid_sample call, untouched (outputs bit-identical, so step 0 against p06 stays exact). Backward = the same
bilinear / zeros-padding / align_corners=False gradient written out: the grid gradient is a pure gather (no accumulation), the value
gradient is summed with index_put_(accumulate=True) under torch's deterministic mode (a sort-based segment sum in a fixed order).
Same maths as PyTorch's kernel; only the summation order of the value gradient differs (checked in det_check()).
Autocast: PyTorch puts grid_sample on its promote list (widest input type); the wrapper applies the same cast, then calls with autocast
off, so the forward kernel call is the one autocast would make (a first version cast to fp32 and moved step 1's loss: caught, fixed).
The backward is worked in fp32 (exact upcast of bf16 inputs) and cast back to the input dtypes.
Only the case the model uses is replaced (4-D, bilinear, zeros, align_corners False, both inputs floating); anything else calls the
original. install() patches torch.nn.functional.grid_sample, which mmcv looks up at call time.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_ORIG = F.grid_sample


class _DetGridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, grid):
        ctx.save_for_backward(inp, grid)
        return _ORIG(inp, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    @staticmethod
    def backward(ctx, g):
        inp0, grid0 = ctx.saved_tensors
        wd = torch.promote_types(torch.promote_types(inp0.dtype, grid0.dtype), torch.float32)   # at least fp32 (exact upcast), cast back
        inp, grid, g = inp0.to(wd), grid0.to(wd), g.to(wd)
        N, C, H, W = inp.shape
        _, Hg, Wg, _ = grid.shape
        g = g.contiguous()
        ix = ((grid[..., 0] + 1) * W - 1) / 2                      # grid_sampler_compute_source_index, align_corners=False
        iy = ((grid[..., 1] + 1) * H - 1) / 2
        x0 = torch.floor(ix); y0 = torch.floor(iy); x1 = x0 + 1; y1 = y0 + 1
        wx1 = ix - x0; wx0 = x1 - ix; wy1 = iy - y0; wy0 = y1 - iy
        corners = ((x0, y0, wx0 * wy0), (x1, y0, wx1 * wy0), (x0, y1, wx0 * wy1), (x1, y1, wx1 * wy1))   # nw, ne, sw, se
        flat = inp.reshape(N, C, H * W)
        vals, idxs, ws = [], [], []
        for cx, cy, w in corners:
            ok = (cx >= 0) & (cx <= W - 1) & (cy >= 0) & (cy <= H - 1)
            li = (cy.clamp(0, H - 1) * W + cx.clamp(0, W - 1)).long()                                  # (N, Hg, Wg)
            v = torch.gather(flat, 2, li.reshape(N, 1, Hg * Wg).expand(N, C, Hg * Wg)).reshape(N, C, Hg, Wg)
            v = v * ok[:, None].to(v.dtype)
            vals.append(v); idxs.append((li, ok)); ws.append(w)
        grad_inp = grad_grid = None
        if ctx.needs_input_grad[1]:
            v_nw, v_ne, v_sw, v_se = vals
            gix = (g * ((v_ne - v_nw) * wy0[:, None] + (v_se - v_sw) * wy1[:, None])).sum(1)
            giy = (g * ((v_sw - v_nw) * wx0[:, None] + (v_se - v_ne) * wx1[:, None])).sum(1)
            grad_grid = torch.stack((gix * (W / 2), giy * (H / 2)), -1).to(grid0.dtype)
        if ctx.needs_input_grad[0]:
            gt = g.permute(0, 2, 3, 1)                                                                  # (N, Hg, Wg, C)
            base = (torch.arange(N, device=g.device) * (H * W)).view(N, 1, 1)
            rows, src = [], []
            for (li, ok), w in zip(idxs, ws):
                rows.append((base + li).reshape(-1))
                src.append((gt * (w * ok.to(w.dtype))[..., None]).reshape(-1, C))
            out = torch.zeros(N * H * W, C, device=g.device, dtype=g.dtype)
            prev, prev_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
            torch.use_deterministic_algorithms(True)
            try:
                out.index_put_((torch.cat(rows),), torch.cat(src), accumulate=True)
            finally:
                torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
            grad_inp = out.view(N, H, W, C).permute(0, 3, 1, 2).contiguous().to(inp0.dtype)
        return grad_inp, grad_grid


def grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
    if (mode == "bilinear" and padding_mode == "zeros" and align_corners is False and input.dim() == 4
            and input.is_floating_point() and grid.is_floating_point() and torch.is_grad_enabled()
            and (input.requires_grad or grid.requires_grad)):
        if torch.is_autocast_enabled("cuda"):
            # autocast puts grid_sample on its promote list (runs in the widest input type): do the same cast here, then call with
            # autocast off, so the forward is the very kernel call autocast would have made
            dt = torch.promote_types(input.dtype, grid.dtype)
            input, grid = input.to(dt), grid.to(dt)
            with torch.autocast("cuda", enabled=False):
                return _DetGridSample.apply(input, grid)
        return _DetGridSample.apply(input, grid)
    return _ORIG(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


def install() -> None:
    F.grid_sample = grid_sample
    torch.nn.functional.grid_sample = grid_sample


def det_check() -> dict:
    """the replacement vs PyTorch's own backward: forward bit-identical; gradients equal to float64 precision in float64, and within fp32
    rounding in fp32; repeatable (two backwards bit-identical)."""
    torch.manual_seed(0)
    res = {}
    for dt in (torch.float64, torch.float32):
        x = torch.randn(64, 32, 50, 50, device="cuda", dtype=dt, requires_grad=True)
        gr = (torch.rand(64, 300, 8, 2, device="cuda", dtype=dt) * 2.4 - 1.2).requires_grad_(True)   # some points outside: zero padding
        g = torch.randn(64, 32, 300, 8, device="cuda", dtype=dt)
        o1 = _ORIG(x, gr, mode="bilinear", padding_mode="zeros", align_corners=False)
        a = torch.autograd.grad(o1, (x, gr), g)
        o2 = _DetGridSample.apply(x, gr)
        b = torch.autograd.grad(o2, (x, gr), g)
        c = torch.autograd.grad(_DetGridSample.apply(x, gr), (x, gr), g)
        rel = lambda u, v: float((u - v).abs().max() / v.abs().max())
        res[str(dt)] = {"forward_identical": bool(torch.equal(o1, o2)), "grad_value_rel": rel(b[0], a[0]), "grad_grid_rel": rel(b[1], a[1]),
                        "repeatable": bool(torch.equal(b[0], c[0]) and torch.equal(b[1], c[1])),
                        "torch_repeatable": bool(all(torch.equal(u, v) for u, v in zip(a, torch.autograd.grad(
                            _ORIG(x, gr, mode="bilinear", padding_mode="zeros", align_corners=False), (x, gr), g))))}
    # autocast path: same forward as autocast's own call, both all-bf16 and mixed inputs
    for name, dx, dg in (("autocast_bf16_bf16", torch.bfloat16, torch.bfloat16), ("autocast_bf16_fp32", torch.bfloat16, torch.float32)):
        x = torch.randn(64, 32, 50, 50, device="cuda", dtype=dx, requires_grad=True)
        gr = (torch.rand(64, 300, 8, 2, device="cuda", dtype=dg) * 2.4 - 1.2).requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o1 = _ORIG(x, gr, mode="bilinear", padding_mode="zeros", align_corners=False)
            o2 = grid_sample(x, gr, mode="bilinear", padding_mode="zeros", align_corners=False)
        g = torch.randn_like(o1)
        b = torch.autograd.grad(o2, (x, gr), g)
        res[name] = {"forward_identical": bool(o1.dtype == o2.dtype and torch.equal(o1, o2)), "out_dtype": str(o2.dtype),
                     "grad_dtypes": [str(b[0].dtype), str(b[1].dtype)]}
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(det_check(), indent=1))
