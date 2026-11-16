import pytest
import torch

from cliffordnet_cl20.algebra.cl20 import to_complex_pairs
from cliffordnet_cl20.layers.geometric_shuffle import (
    Cl20ShuffleInteraction,
    geometric_shuffle_pairs,
)
from cliffordnet_cl20.models.registry import build_model, get_model_spec


def _labeled_pairs(groups=8):
    labels = torch.arange(groups).reshape(1, groups, 1, 1).float()
    return torch.complex(labels, labels + 100), torch.complex(
        labels + 1000, labels + 1100
    )


def test_comm_and_anti_receive_identical_bijective_permutation():
    comm, anti = _labeled_pairs()
    shuffled_comm, shuffled_anti = geometric_shuffle_pairs(comm, anti, groups=2)
    comm_order = shuffled_comm.real.flatten().long()
    anti_order = (shuffled_anti.real.flatten() - 1000).long()
    assert torch.equal(comm_order, anti_order)
    assert sorted(comm_order.tolist()) == list(range(8))
    assert comm_order.tolist() == [0, 4, 1, 5, 2, 6, 3, 7]


@pytest.mark.parametrize("groups", [1, 3, 9])
def test_invalid_shuffle_group_count_fails(groups):
    comm, anti = _labeled_pairs(8)
    with pytest.raises(ValueError):
        geometric_shuffle_pairs(comm, anti, groups)


def test_shuffle_never_splits_a_multivector_tuple():
    value = torch.arange(32).reshape(1, 32, 1, 1).float()
    comm, anti = to_complex_pairs(value)
    shuffled_comm, shuffled_anti = geometric_shuffle_pairs(comm, anti, 2)
    order_comm = shuffled_comm.real.long().flatten() // 4
    order_anti = shuffled_anti.real.long().flatten() // 4
    assert torch.equal(order_comm, order_anti)


def test_shuffle_is_not_a_cyclic_roll():
    comm, anti = _labeled_pairs(8)
    shuffled, _ = geometric_shuffle_pairs(comm, anti, groups=2)
    assert not any(
        torch.equal(shuffled, comm.roll(shift, dims=1)) for shift in range(8)
    )


def test_interaction_has_original_plus_shuffle_branches():
    layer = Cl20ShuffleInteraction(32, groups=(2, 4, 8))
    value = torch.randn(2, 32, 4, 4)
    assert layer(value, value).shape == value.shape
    assert layer.proj.in_channels == 32 * 4


@pytest.mark.parametrize(
    "name,depth,parameters",
    [
        ("shuffle_d14_g2_4_8", 14, 1_439_716),
        ("shuffle_d26_g2_4_8", 26, 2_659_300),
    ],
)
def test_log_verified_shuffle_configurations_are_registered(name, depth, parameters):
    spec = get_model_spec(name)
    assert spec.family == "shuffle"
    assert spec.depth == depth
    assert spec.shifts == (2, 4, 8)
    model = build_model(name)
    assert len(model.blocks) == depth
    assert sum(parameter.numel() for parameter in model.parameters()) == parameters
