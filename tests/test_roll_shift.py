from pathlib import Path

import pytest
import torch
import yaml

from cliffordnet_cl20.layers.roll_shift import (
    Cl20RollShiftInteraction,
    roll_multivector_groups,
)
from cliffordnet_cl20.models.registry import build_model, get_model_spec


def test_group_roll_keeps_basis_components_together():
    value = torch.arange(1 * 12 * 1 * 1).reshape(1, 12, 1, 1).float()
    rolled = roll_multivector_groups(value, 1).reshape(1, 3, 4, 1, 1)
    expected = value.reshape(1, 3, 4, 1, 1).roll(1, dims=1)
    torch.testing.assert_close(rolled, expected)


def test_interaction_preserves_shape_and_records_zero_branch():
    layer = Cl20RollShiftInteraction(16, shifts=(0, 1, 2))
    state = torch.randn(2, 16, 5, 5)
    context = torch.randn_like(state)
    assert layer(state, context).shape == state.shape
    assert layer.shifts == (0, 1, 2)
    assert layer.proj.in_channels == 48


def test_invalid_multivector_width_is_rejected():
    with pytest.raises(ValueError, match="divisible by 4"):
        Cl20RollShiftInteraction(10, shifts=(0, 1))


def test_verified_25_3_signature_and_parameter_count():
    model = build_model("roll_shift_d25_s3")
    state = model.state_dict()
    assert len(model.blocks) == 25
    assert model.blocks[0].mustard_conv.shifts == (0, 1, 2)
    assert state["blocks.0.get_context.0.weight"].shape == (128, 1, 3, 3)
    assert state["blocks.0.mustard_conv.proj.weight"].shape == (128, 384, 1, 1)
    assert "blocks.0.mustard_conv.grade_scale" not in state
    assert sum(t.numel() for t in state.values()) == 2_564_350


@pytest.mark.parametrize("name,depth,branch_count", [
    ("roll_shift_d16_s2", 16, 2),
    ("roll_shift_d18_s5", 18, 5),
    ("roll_shift_d25_s3", 25, 3),
    ("roll_shift_d32_s5", 32, 5),
    ("roll_shift_d48_s2", 48, 2),
    ("roll_shift_d52_s3", 52, 3),
])
def test_portfolio_configurations_are_explicit(name, depth, branch_count):
    spec = get_model_spec(name)
    assert spec.family == "roll_shift"
    assert spec.depth == depth
    assert len(spec.shifts) == branch_count
    path = Path("configs/roll_shift") / f"d{depth}_s{branch_count}.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["model"]["name"] == name
    assert config["model"]["depth"] == depth
    assert tuple(config["model"]["shifts"]) == spec.shifts
