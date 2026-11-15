"""Numerical validation helpers for Cl(2,0) implementations."""

import torch

from .cl20 import geometric_product, geometric_product_direct


def maximum_product_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((geometric_product(left, right) - geometric_product_direct(left, right)).abs().max())
