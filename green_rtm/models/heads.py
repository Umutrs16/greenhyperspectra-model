"""Masked multi-trait head and spectral reconstruction head."""

from __future__ import annotations

import torch
from torch import nn


class MultiTraitHead(nn.Module):
    def __init__(self, dim: int, n_traits: int = 5, dropout: float = 0.08):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, n_traits),
        )
        # Heteroscedastic log-variance, one per trait.
        self.log_var = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_traits))

    def forward(self, cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(cls), self.log_var(cls)


class ReconstructHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens).squeeze(-1)
