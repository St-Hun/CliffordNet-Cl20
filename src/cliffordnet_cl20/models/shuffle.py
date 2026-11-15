"""Later experimental Geometric Shuffle variant recovered from source."""

from __future__ import annotations

import torch
from torch import nn

from ..layers.geometric_shuffle import Cl20ShuffleInteraction
from .common import DropPath, LayerNorm2d


class Cl20ShuffleBlock(nn.Module):
    def __init__(self, dim: int, groups: tuple[int, ...], drop_path: float = 0.1, init_values: float = 1e-5):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.get_state = nn.Conv2d(dim, dim, kernel_size=1)
        self.get_context = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        self.mustard_conv = Cl20ShuffleInteraction(dim, groups)
        self.gate_fc = nn.Conv2d(dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), init_values))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, value):
        shortcut = value
        normalized = self.norm(value)
        state = self.get_state(normalized)
        context = self.get_context(normalized) - state
        geometric = self.mustard_conv(state, context)
        mixed = normalized + torch.sigmoid(self.gate_fc(normalized)) * geometric
        return shortcut + self.drop_path(self.gamma * mixed)


class Cl20ShuffleNet(nn.Module):
    def __init__(self, num_classes: int = 100, patch_size: int = 2, embed_dim: int = 128,
                 groups: tuple[int, ...] = (2, 4, 8, 16, 32), depth: int = 24,
                 drop_path_rate: float = 0.3):
        super().__init__()
        if embed_dim % 4:
            raise ValueError("embed_dim must be divisible by 4")
        if patch_size != 2:
            raise ValueError("source-recovered shuffle model requires patch_size=2")
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([Cl20ShuffleBlock(embed_dim, groups, rates[index]) for index in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, value):
        value = self.patch_embed(value)
        for block in self.blocks:
            value = block(value)
        return value

    def forward(self, value):
        value = self.forward_features(value).mean(dim=(-2, -1))
        return self.head(self.norm(value))
