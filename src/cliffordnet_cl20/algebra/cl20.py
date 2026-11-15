"""Canonical Cl(2,0) multiplication in real and complex-isomorphic forms."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def _as_multivectors(value: Tensor) -> Tensor:
    if value.ndim != 4 or value.shape[1] % 4:
        raise ValueError("Cl(2,0) tensors must be BCHW with channels divisible by 4")
    batch, channels, height, width = value.shape
    return value.view(batch, channels // 4, 4, height, width)


def to_complex_pairs(value: Tensor) -> tuple[Tensor, Tensor]:
    multivectors = _as_multivectors(value)
    comm = torch.complex(multivectors[:, :, 0], multivectors[:, :, 3])
    anti = torch.complex(multivectors[:, :, 1], -multivectors[:, :, 2])
    return comm, anti


def from_complex_pairs(comm: Tensor, anti: Tensor) -> Tensor:
    if comm.shape != anti.shape:
        raise ValueError("comm and anti tensors must have identical shapes")
    batch, groups, height, width = comm.shape
    return torch.stack((comm.real, anti.real, -anti.imag, comm.imag), dim=2).view(batch, groups * 4, height, width)


def geometric_product(left: Tensor, right: Tensor, scalar_activation: Callable[[Tensor], Tensor] | None = None) -> Tensor:
    if left.shape != right.shape:
        raise ValueError("left and right tensors must have identical shapes")
    left_comm, left_anti = to_complex_pairs(left)
    right_comm, right_anti = to_complex_pairs(right)
    out_comm = left_comm * right_comm + left_anti * torch.conj(right_anti)
    out_anti = left_comm * right_anti + left_anti * torch.conj(right_comm)
    if scalar_activation is not None:
        out_comm = torch.complex(scalar_activation(out_comm.real), out_comm.imag)
    return from_complex_pairs(out_comm, out_anti)


def geometric_product_direct(left: Tensor, right: Tensor) -> Tensor:
    if left.shape != right.shape:
        raise ValueError("left and right tensors must have identical shapes")
    a = _as_multivectors(left)
    b = _as_multivectors(right)
    a0, a1, a2, a12 = a.unbind(dim=2)
    b0, b1, b2, b12 = b.unbind(dim=2)
    components = (
        a0 * b0 + a1 * b1 + a2 * b2 - a12 * b12,
        a0 * b1 + a1 * b0 - a2 * b12 + a12 * b2,
        a0 * b2 + a2 * b0 + a1 * b12 - a12 * b1,
        a0 * b12 + a12 * b0 + a1 * b2 - a2 * b1,
    )
    batch, groups, height, width = components[0].shape
    return torch.stack(components, dim=2).view(batch, groups * 4, height, width)
