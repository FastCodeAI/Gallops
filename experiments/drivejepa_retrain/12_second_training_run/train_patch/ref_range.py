"""S10 task 5: runtime override of the forward range of the refiner's self-attention reference points. Nothing under adapters/ is edited.

bevformer/encoder.py:235 builds the points the self-attention block samples at as ref_pos = (ref_2d[:, :, None, :2] + 32) / 64, where
ref_2d is the plan points (bev_refiner.py:113-114: pose (x, y, heading) in metres, car frame). Last-axis index 0 is x = FORWARD, index 1
is y = left (encoder.py:235 keeps the order of ref_2d[..., :2]; temporal_self_attention.py:222-225 divides index 0 by spatial_shapes[..., 1]
= bev_w = the 8 waypoint slots). So -32 m -> 0 and +32 m -> 1 on both axes.

With set_range(x_hi) the forward axis maps -32 m .. x_hi onto 0 .. 1: (x + 32) / (x_hi + 32) = upstream value x 64 / (x_hi + 32).
The lateral axis is untouched. Only the tensor handed to TemporalSelfAttention as `reference_points` is rescaled (a forward pre-hook on
the block); the metric ref_2d that encoder.py:244 gives compute_corners and the camera projection (:273) never pass through here.
x_hi = 32 (default) leaves the tensor object as it was."""
import torch
from navsim.agents.drive_jepa_perception_based.bevformer import temporal_self_attention as _tsa

STATE = dict(x_hi=32.0, calls=0)


def set_range(x_hi=32.0):
    STATE["x_hi"] = float(x_hi)


def _pre(mod, args, kwargs):
    rp = kwargs.get("reference_points")
    if STATE["x_hi"] == 32.0 or rp is None:
        return None
    assert rp.shape[-1] == 2 and len(args) >= 3 and args[2] is None, "only the plan-point path (value = query, 2-d points)"
    kwargs = dict(kwargs); kwargs["reference_points"] = rp * rp.new_tensor([64.0 / (STATE["x_hi"] + 32.0), 1.0]); STATE["calls"] += 1
    return args, kwargs


def install(refiner):
    """Hook every TemporalSelfAttention (or its batch_fix subclass) inside `refiner`. Returns the names found."""
    found = []
    for n, mod in refiner.named_modules():
        if isinstance(mod, _tsa.TemporalSelfAttention):
            mod.register_forward_pre_hook(_pre, with_kwargs=True); found.append(n)
    return found
