"""Structure-preserving ShuffleNet-style permutations for Cl(2,0)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..algebra.cl20 import from_complex_pairs, to_complex_pairs


def geometric_shuffle_pairs(comm: Tensor, anti: Tensor, groups: int) -> tuple[Tensor, Tensor]:
    """Apply one identical bijection to the paired complex representations."""
    if comm.shape != anti.shape or comm.ndim != 4:
        raise ValueError("comm and anti must be identically shaped BGHW tensors")
    group_count = comm.shape[1]
    if groups < 2 or groups > group_count or group_count % groups:
        raise ValueError(f"shuffle groups must divide {group_count} and be in [2, {group_count}]")
    per_group = group_count // groups

    def shuffle(value: Tensor) -> Tensor:
        batch, _, height, width = value.shape
        return value.reshape(batch, groups, per_group, height, width).transpose(1, 2).contiguous().reshape_as(value)

    return shuffle(comm), shuffle(anti)


class Cl20ShuffleInteraction(nn.Module):
    """Original product branch plus one branch per geometric shuffle."""

    def __init__(self, dim: int, groups: tuple[int, ...] | list[int]):
        super().__init__()
        if dim % 4:
            raise ValueError("dim must be divisible by 4 for Cl(2,0)")
        multivectors = dim // 4
        self.groups = tuple(int(group) for group in groups)
        for group in self.groups:
            if group < 2 or group > multivectors or multivectors % group:
                raise ValueError(f"invalid shuffle group {group} for {multivectors} multivectors")
        self.proj = nn.Conv2d(dim * (1 + len(self.groups)), dim, kernel_size=1)

    @staticmethod
    def _product(state_comm, state_anti, context_comm, context_anti):
        out_comm = state_comm * context_comm + state_anti * torch.conj(context_anti)
        out_anti = state_comm * context_anti + state_anti * torch.conj(context_comm)
        return from_complex_pairs(out_comm, out_anti)

    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        if state.shape != context.shape:
            raise ValueError("state and context must have identical shapes")
        state_comm, state_anti = to_complex_pairs(state)
        context_comm, context_anti = to_complex_pairs(context)
        branches = [self._product(state_comm, state_anti, context_comm, context_anti)]
        for groups in self.groups:
            shuffled_comm, shuffled_anti = geometric_shuffle_pairs(context_comm, context_anti, groups)
            branches.append(self._product(state_comm, state_anti, shuffled_comm, shuffled_anti))
        return self.proj(torch.cat(branches, dim=1))
