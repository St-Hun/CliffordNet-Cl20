"""Official CliffordNet baseline pinned to commit 28119c5."""

import torch
import torch.nn.functional as F
from torch import nn

from ..layers.official_interaction import OfficialCliffordInteraction
from .common import DropPath, LayerNorm2d


class CliffordAlgebraBlock(nn.Module):
    def __init__(self, dim, cli_mode="full", ctx_mode="diff", shifts=(1, 2), drop_path=0.1):
        super().__init__()
        self.get_state = nn.Conv2d(dim, dim, kernel_size=1)
        self.get_context_local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        self.enable_gFFNG = False
        self.norm = LayerNorm2d(dim)
        self.clifford_interaction_local = OfficialCliffordInteraction(dim, cli_mode, ctx_mode, shifts)
        self.act = nn.SiLU()
        self.gate_fc = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), 1e-5))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x_ln = self.norm(x)
        state = self.get_state(x_ln)
        context = self.get_context_local(x_ln)
        geometric = self.clifford_interaction_local(state, context)
        combined = torch.cat([x_ln, geometric], dim=1)
        gate = torch.sigmoid(self.gate_fc(combined))
        mixed = F.silu(x_ln) + gate * geometric
        return shortcut + self.drop_path(self.gamma * mixed)


class GeometricStem(nn.Module):
    def __init__(self, in_chans=3, embed_dim=128, patch_size=2):
        super().__init__()
        if patch_size == 1:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.SiLU(),
                nn.Conv2d(embed_dim // 2, embed_dim, 3, padding=1, bias=False),
            )
        elif patch_size == 2:
            self.proj = nn.Conv2d(in_chans, embed_dim, 3, stride=2, padding=1)
        elif patch_size == 4:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.SiLU(),
                nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=2, padding=1, bias=False),
            )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x))


class OfficialCliffordNet(nn.Module):
    def __init__(self, num_classes=100, patch_size=2, embed_dim=128, cli_mode="full", ctx_mode="diff", shifts=(1, 2), depth=12, drop_path_rate=0.3):
        super().__init__()
        self.patch_embed = GeometricStem(3, embed_dim, patch_size)
        rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [CliffordAlgebraBlock(embed_dim, cli_mode, ctx_mode, shifts, rates[index]) for index in range(depth)]
        )
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

    def forward_features(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_features(x).mean(dim=(-2, -1))
        return self.head(self.norm(x))
