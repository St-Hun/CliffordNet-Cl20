import pytest
import torch

from cliffordnet_cl20.algebra.cl20 import (
    from_complex_pairs,
    geometric_product,
    geometric_product_direct,
    to_complex_pairs,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complex_pair_round_trip(dtype) -> None:
    value = torch.randn(2, 12, 3, 3, dtype=dtype)
    comm, anti = to_complex_pairs(value)
    assert torch.equal(from_complex_pairs(comm, anti), value)


def test_rejects_channels_not_divisible_by_four() -> None:
    with pytest.raises(ValueError, match="divisible by 4"):
        to_complex_pairs(torch.randn(1, 6, 2, 2))


def test_basis_multiplication_table() -> None:
    basis = torch.eye(4).view(4, 4, 1, 1)
    expected = {
        (1, 1): (1, 0),
        (2, 2): (1, 0),
        (3, 3): (-1, 0),
        (1, 2): (1, 3),
        (2, 1): (-1, 3),
        (1, 3): (1, 2),
        (3, 1): (-1, 2),
        (2, 3): (-1, 1),
        (3, 2): (1, 1),
    }
    for (left_index, right_index), (sign, result_index) in expected.items():
        product = geometric_product(
            basis[left_index : left_index + 1], basis[right_index : right_index + 1]
        )
        want = torch.zeros_like(product)
        want[:, result_index] = sign
        assert torch.equal(product, want), (left_index, right_index)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complex_product_matches_direct_values_and_gradients(dtype) -> None:
    left = torch.randn(2, 8, 2, 2, dtype=dtype, requires_grad=True)
    right = torch.randn(2, 8, 2, 2, dtype=dtype, requires_grad=True)
    complex_result = geometric_product(left, right)
    complex_grads = torch.autograd.grad(
        complex_result.square().sum(), (left, right), retain_graph=True
    )
    direct_result = geometric_product_direct(left, right)
    direct_grads = torch.autograd.grad(direct_result.square().sum(), (left, right))
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    assert torch.allclose(complex_result, direct_result, atol=tolerance, rtol=tolerance)
    assert torch.allclose(
        complex_grads[0], direct_grads[0], atol=tolerance, rtol=tolerance
    )
    assert torch.allclose(
        complex_grads[1], direct_grads[1], atol=tolerance, rtol=tolerance
    )
