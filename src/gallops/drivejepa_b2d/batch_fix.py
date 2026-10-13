"""Make Drive-JEPA's refiner batch-safe. Upstream mixes batch elements in three places; bs = 1 is exact.

Found 5 Oct 2026 while fine-tuning the generator from cached backbone features: a dump at batch 64 was a
median 1.49 m off the same frames at batch 1. Deployment runs batch 1, so closed-loop numbers are unaffected;
anything that trains or evaluates the refiner batched is not. Every generator run behind the Final uses this
fix.

  1. SpatialCrossAttention.forward (BEVFormer heritage) picks the queries that attend to each camera from
     batch
     element 0's visibility mask and applies that to every element. BEVFormer's BEV grid is the same for every
     sample; Drive-JEPA's queries sit on each sample's own candidate trajectories. Fixed by a forward that
         uses each
     element's own mask (`apply_sca`).
  2. TemporalSelfAttention.forward builds value = stack([q, q], 1) -> order [q0, q0, q1, q1, ...] and then
     feeds
     torch.cat([value[:bs], query]): element j gets q_{j // 2}'s half. Fixed to value[0::2] (`apply_tsa`).
  3. BEVFormerEncoder.forward builds the reference points with torch.cat([ref_pos, ref_pos]) (queue-major)
     while
     TemporalSelfAttention flattens element-major. Fixed to ref_pos.repeat_interleave(2, 0) (`apply_enc`).

2 and 3 patch one line of the upstream source each and assert that line is still there, so an upstream change
fails loudly instead of silently un-fixing. Ported from paddock-ops mtd/stage4/sca_fix.py. Valid Python 3.9:
it runs inside Drive-JEPA's interpreter."""

from __future__ import annotations

import inspect
import textwrap

import torch


def _sca_forward(
    self,
    query,
    key,
    value,
    residual=None,
    query_pos=None,
    key_padding_mask=None,
    reference_points=None,
    spatial_shapes=None,
    reference_points_cam=None,
    bev_mask=None,
    level_start_index=None,
    flag="encoder",
    **kwargs,
):
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    reference_points_cam = reference_points_cam.to(torch.float32)
    with torch.autocast(device_type="cuda", enabled=False):
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos
        bs, num_query, _ = query.size()
        D = reference_points_cam.size(3)
        # the fix: indexes[j][i] from element j's own mask (upstream: mask_per_img[0] for every j)
        indexes = [
            [mask_per_img[j].sum(-1).nonzero().squeeze(-1) for mask_per_img in bev_mask] for j in range(bs)
        ]
        max_len = max(len(e) for row in indexes for e in row)
        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                idx = indexes[j][i]
                queries_rebatch[j, i, : len(idx)] = query[j, idx]
                reference_points_rebatch[j, i, : len(idx)] = reference_points_per_img[j, idx]
        num_cams, length, bs, embed_dims = key.shape
        key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, length, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, length, self.embed_dims)
        if self.full_attention:
            queries = self.encoder(
                tgt=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims), memory=key
            ).view(bs, self.num_cams, max_len, self.embed_dims)
        else:
            queries = self.deformable_attention(
                query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
                key=key,
                value=value,
                reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len, D, 2),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            ).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i in range(len(bev_mask)):
                idx = indexes[j][i]
                slots[j, idx] += queries[j, i, : len(idx)]
        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)
        res = self.dropout(slots) + inp_residual
    return res


def _patch(module, cls_name: str, old: str, new: str, label: str) -> None:
    """Recompile `cls.forward` from its own source with one line replaced."""
    cls = getattr(module, cls_name)
    src = textwrap.dedent(inspect.getsource(cls.forward))
    assert src.count(old) == 1, f"upstream {cls_name}.forward changed: {old!r} not found exactly once"
    ns: dict = {}
    exec(compile(src.replace(old, new), f"{module.__file__} ({label})", "exec"), module.__dict__, ns)
    cls.forward = ns["forward"]


def apply_sca() -> None:
    from navsim.agents.drive_jepa_perception_based.bevformer import spatial_cross_attention

    spatial_cross_attention.SpatialCrossAttention.forward = _sca_forward


def apply_tsa() -> None:
    from navsim.agents.drive_jepa_perception_based.bevformer import temporal_self_attention

    old, new = "query = torch.cat([value[:bs], query], -1)", "query = torch.cat([value[0::2], query], -1)"
    _patch(temporal_self_attention, "TemporalSelfAttention", old, new, "value[0::2] patch")


def apply_enc() -> None:
    from navsim.agents.drive_jepa_perception_based.bevformer import encoder

    old, new = (
        "hybird_ref_2d = torch.cat([ref_pos, ref_pos])",
        "hybird_ref_2d = ref_pos.repeat_interleave(2, 0)",
    )
    _patch(encoder, "BEVFormerEncoder", old, new, "ref repeat_interleave patch")


def apply_all() -> None:
    apply_sca()
    apply_tsa()
    apply_enc()
