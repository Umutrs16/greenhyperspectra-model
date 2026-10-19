"""Low-rank real–RTM domain residual: ΔR = B z, rank r = 8–16."""

from __future__ import annotations

import torch
from torch import nn


class LowRankResidual(nn.Module):
    def __init__(self, dim: int, n_bands: int = 1721, rank: int = 8):
        super().__init__()
        self.rank = rank
        self.to_z = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, rank))
        # B ∈ R^{L×r}
        self.basis = nn.Parameter(torch.randn(n_bands, rank) * 0.02)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        z = self.to_z(cls)
        return torch.matmul(z, self.basis.t())

    def penalty(self, delta: torch.Tensor) -> torch.Tensor:
        amp = delta.pow(2).mean()
        smooth = (delta[:, 1:] - delta[:, :-1]).pow(2).mean()
        return amp, smooth
