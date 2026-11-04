"""H15 phase B's scorers and geometry (torch; they skip in gallops' own interpreter).

The claims tested are the ones the plan's decision rules lean on: every residual mode is exactly
the release at step 0, an empty scene does not poison the attention with NaN, the replacement's
final score is NAVSIM's formula, and the clearances put an agent where the ego will meet it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gallops.drivejepa_b2d.head import Head  # noqa: E402
from gallops.drivejepa_perception import geometry, scorers  # noqa: E402

B, K, A = 2, 32, 5


def _batch(agents_present=True):
    torch.manual_seed(0)
    candidates = torch.zeros(B, K, 8, 3)
    candidates[..., 0] = (
        torch.linspace(1, 8, 8) * torch.linspace(0.5, 2.0, K)[:, None]
    )  # straight, varied speed
    agents = {
        "cls": torch.zeros(B, A, dtype=torch.long),
        "conf": torch.ones(B, A),
        "xyz": torch.tensor([[[10.0, 0.0, 0.5]] * A] * B),
        "lwh": torch.tensor([[[4.5, 1.9, 1.5]] * A] * B),
        "yaw": torch.zeros(B, A),
        "vel": torch.zeros(B, A, 2),
        "path": torch.zeros(B, A, 2, 2),
        "mask": torch.ones(B, A, dtype=torch.bool) if agents_present else torch.zeros(B, A, dtype=torch.bool),
    }
    return {
        "pooled": torch.randn(B, K, 256).half(),
        "shape": torch.randn(B, K, 24),
        "ego": torch.randn(B, 11),
        "logits": torch.randn(B, K, 6),
        "candidates": candidates,
        "agents": agents,
        "pdm": torch.rand(B, K, 9),
        "ade": torch.rand(B, K),
    }


@pytest.mark.parametrize("mode", ["frozen", "lora", "full"])
@pytest.mark.parametrize("agents", [False, True])
def test_residual_modes_are_exactly_the_release_at_step_0(mode, agents):
    torch.manual_seed(1)
    model = scorers.B1Model(mode, agents, Head())
    batch = _batch()
    out = model(batch)
    assert torch.equal(out["logits"], batch["logits"])


def test_an_empty_scene_does_not_make_nan():
    model = scorers.B1Model("frozen", True, Head())
    for p in model.adapter.out.parameters():
        torch.nn.init.normal_(p)  # as if trained, so a NaN would show
    out = model(_batch(agents_present=False))
    assert torch.isfinite(out["logits"]).all()


def test_the_replacement_final_is_navsims_formula():
    # From the vendored scorer's own output (docs/navsim-metric-cache.md): EP 0.791, the rest 1 -> 0.925.
    sub = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.791, 1.0, 1.0, 1.0])
    assert abs(float(scorers.final_from_subscores(sub)) - 0.925) < 5e-4
    sub_collision = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    assert float(scorers.final_from_subscores(sub_collision)) == 0.0


def test_clearance_finds_the_parked_car_ahead_when_the_ego_reaches_it():
    batch = _batch()
    agents = batch["agents"]
    features, pairwise, nearest = geometry.terms(
        batch["candidates"],
        agents["xyz"][..., :2],
        agents["lwh"][..., :2],
        agents["vel"],
        agents["path"],
        agents["mask"],
    )
    assert features.shape == (B, K, geometry.N_FEATURES)
    # The fastest candidate reaches 16 m: it drives through the car at 10 m, so its clearance is < 0.
    assert float(pairwise[0, -1].min()) < 0.0
    # The slowest reaches 4 m: it stays clear.
    assert float(pairwise[0, 0].min()) > 0.0
    assert (nearest >= 0).all()


def test_the_loss_runs_in_every_mode():
    for mode in scorers.MODES:
        model = scorers.B1Model(mode, True, Head())
        total, parts = scorers.loss(model(_batch()), _batch())
        assert torch.isfinite(total), mode
        total.backward()
