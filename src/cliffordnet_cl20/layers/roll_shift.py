"""Full-multivector Cl(2,0) Roll & Shift interaction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..algebra.cl20 import from_complex_pairs, to_complex_pairs


def roll_multivector_groups(value: Tensor, shift: int) -> Tensor:
    """Roll whole ``(e0,e1,e2,e12)`` tuples, never individual basis channels."""
    if value.ndim != 4 or value.shape[1] % 4:
        raise ValueError("Cl(2,0) channel width must be divisible by 4")
    batch, channels, height, width = value.shape
    groups = channels // 4
    return (
        value.reshape(batch, groups, 4, height, width)
        .roll(shift, dims=1)
        .reshape_as(value)
    )


class Cl20RollShiftInteraction(nn.Module):
    """Apply the Cl(2,0) product across shifted multivector groups.

    The same ``+s`` shift is applied to both complex pairs, matching the
    historical experiments and keeping each multivector group together.
    """

    def __init__(self, dim: int, shifts: tuple[int, ...] | list[int]):
        super().__init__()
        if dim % 4:
            raise ValueError("dim must be divisible by 4 for Cl(2,0)")
        if not shifts:
            raise ValueError("at least one explicit shift is required")
        self.groups = dim // 4
        self.shifts = tuple(int(shift) for shift in shifts)
        self.proj = nn.Conv2d(dim * len(self.shifts), dim, kernel_size=1)

    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        if state.shape != context.shape:
            raise ValueError("state and context must have identical shapes")
        state_comm, state_anti = to_complex_pairs(state)
        context_comm, context_anti = to_complex_pairs(context)
        branches = []
        for shift in self.shifts:
            shifted_comm = torch.roll(context_comm, shift, dims=1)
            shifted_anti = torch.roll(context_anti, shift, dims=1)
            out_comm = state_comm * shifted_comm + state_anti * torch.conj(shifted_anti)
            out_anti = state_comm * shifted_anti + state_anti * torch.conj(shifted_comm)
            out_comm = torch.complex(F.silu(out_comm.real), out_comm.imag)
            branches.append(from_complex_pairs(out_comm, out_anti))
        return self.proj(torch.cat(branches, dim=1))
