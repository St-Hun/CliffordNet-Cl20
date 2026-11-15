"""Pinned pure-PyTorch CliffordNet interaction."""

import torch
from torch import nn


class OfficialCliffordInteraction(nn.Module):
    def __init__(self, dim: int, cli_mode: str = "full", ctx_mode: str = "diff", shifts=(1, 2)):
        super().__init__()
        self.dim = dim
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.act = nn.SiLU()
        self.shifts = [s for s in shifts if s < dim]
        self.branch_dim = dim * len(self.shifts)
        if cli_mode == "full":
            cat_dim = self.branch_dim * 2
        elif cli_mode in {"wedge", "inner"}:
            cat_dim = self.branch_dim
        else:
            raise ValueError(f"Invalid cli_mode: {cli_mode}")
        self.proj_ = nn.Conv2d(cat_dim, dim, kernel_size=1)

    def forward(self, z1, z2):
        if self.ctx_mode == "diff":
            context = z2 - z1
        elif self.ctx_mode == "abs":
            context = z2
        else:
            raise ValueError(f"Invalid ctx_mode: {self.ctx_mode}")
        features = []
        for shift in self.shifts:
            shifted_context = torch.roll(context, shifts=shift, dims=1)
            if self.cli_mode in {"wedge", "full"}:
                shifted_state = torch.roll(z1, shifts=shift, dims=1)
                features.append(z1 * shifted_context - context * shifted_state)
            if self.cli_mode in {"inner", "full"}:
                features.append(self.act(z1 * shifted_context))
        return self.proj_(torch.cat(features, dim=1))
