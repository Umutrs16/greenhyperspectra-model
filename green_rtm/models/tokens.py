"""Wavelength / FWHM-aware spectral tokens.

z_i = MLP([R_i, PE(λ_i), log(FWHM_i), σ_i, mask_i])
"""

from __future__ import annotations

import torch
from torch import nn


class WavelengthPE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = (2.0 ** torch.arange(half, dtype=torch.float32)) * torch.pi
        self.register_buffer("freqs", freqs)

    def forward(self, wavelength_nm: torch.Tensor) -> torch.Tensor:
        wl = wavelength_nm / 1000.0
        ang = wl.unsqueeze(-1) * self.freqs
        enc = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if enc.shape[-1] < self.dim:
            enc = torch.nn.functional.pad(enc, (0, self.dim - enc.shape[-1]))
        return enc[..., : self.dim]


class SpectralTokenizer(nn.Module):
    def __init__(self, dim: int = 192, pe_dim: int = 32):
        super().__init__()
        self.pe = WavelengthPE(pe_dim)
        # R, logFWHM, sigma, valid  + PE
        self.proj = nn.Sequential(
            nn.Linear(4 + pe_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        reflectance: torch.Tensor,
        wavelength: torch.Tensor,
        fwhm: torch.Tensor,
        sigma: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        pe = self.pe(wavelength)
        log_fwhm = torch.log(fwhm.clamp_min(0.25))
        feats = torch.stack([reflectance, log_fwhm, sigma, valid], dim=-1)
        return self.proj(torch.cat([feats, pe], dim=-1))
