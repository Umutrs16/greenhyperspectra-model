"""Monte Carlo nuisance sampling. Default T2 returns the mean spectrum E[F]; optional samples support a Laplace likelihood."""

from __future__ import annotations

import torch
from torch import nn

from ..rtm.emulator import PARAM_ORDER, ForwardEmulator

THETA = ["LAI", "Cab", "Cw", "Cm", "Cp"]
NUISANCE = ["Car", "Anth", "N", "Cbrown", "SZA", "VZA", "RAA", "ALA", "psoil", "rsoil", "hotspot"]


class RTMMarginalizer(nn.Module):
    def __init__(self, emulator: ForwardEmulator | None, M: int = 8, n_bands: int = 1721):
        super().__init__()
        self.emulator = emulator
        self.M = M
        self.n_bands = n_bands
        # Learned prior over nuisance when no emulator is attached (unit-test / smoke).
        self.nuisance_mean = nn.Parameter(torch.tensor(
            [8.0, 0.5, 1.6, 0.1, 30.0, 8.0, 90.0, 45.0, 0.5, 1.0, 0.02], dtype=torch.float32
        ))
        self.nuisance_log_std = nn.Parameter(torch.zeros(len(NUISANCE)))
        self.theta_from_pred = nn.Identity()
        self.bound_nuisance = False

    def enable_library_bounds(self, lo: list[float], hi: list[float], init_physical: list[float], latent_std: float = 0.5) -> None:
        """Map the trainable Gaussian through a logistic onto the emulator library.

        The stored mean and log-std are in unconstrained coordinates. Every draw is inside
        [lo, hi]. This is a separate experiment from the reported unbounded sampler.
        """
        lo_t = torch.tensor(lo, dtype=torch.float32)
        hi_t = torch.tensor(hi, dtype=torch.float32)
        if "nuisance_lo" not in self._buffers:
            self.register_buffer("nuisance_lo", lo_t)
            self.register_buffer("nuisance_hi", hi_t)
        else:
            self.nuisance_lo.copy_(lo_t)
            self.nuisance_hi.copy_(hi_t)
        phys = torch.tensor(init_physical, dtype=torch.float32)
        unit = ((phys - lo_t) / (hi_t - lo_t)).clamp(1e-4, 1.0 - 1e-4)
        latent = torch.log(unit / (1.0 - unit))
        with torch.no_grad():
            self.nuisance_mean.copy_(latent)
            self.nuisance_log_std.fill_(float(torch.log(torch.tensor(latent_std))))
        self.bound_nuisance = True

    def _to_physical(self, latent: torch.Tensor) -> torch.Tensor:
        if not self.bound_nuisance:
            return latent
        unit = torch.sigmoid(latent)
        return self.nuisance_lo + (self.nuisance_hi - self.nuisance_lo) * unit

    def sample_nuisance(
        self,
        batch: int,
        M: int | None = None,
        known: dict[str, torch.Tensor] | None = None,
        fixed: bool = False,
    ) -> torch.Tensor:
        M = M or self.M
        if fixed:
            latent = self.nuisance_mean.view(1, 1, -1).expand(batch, M, -1)
        else:
            std = self.nuisance_log_std.exp()
            eps = torch.randn(batch, M, len(NUISANCE), device=self.nuisance_mean.device)
            latent = self.nuisance_mean + std * eps
        eta = self._to_physical(latent)
        if known:
            eta = eta.clone()
            for name, value in known.items():
                if name in NUISANCE:
                    j = NUISANCE.index(name)
                    eta[:, :, j] = value.view(-1, 1)
        return eta.contiguous()

    def forward(
        self,
        theta: torch.Tensor,
        M: int | None = None,
        known_nuisance: dict[str, torch.Tensor] | None = None,
        return_samples: bool = False,
    ) -> torch.Tensor:
        """theta: (B, 5) in physical units. Mean spectrum (B, L) or samples (B, M, L)."""
        M = M or self.M
        bsz = theta.shape[0]
        fixed = bool(getattr(self, "fixed_nuisance", False))
        eta = self.sample_nuisance(bsz, M, known_nuisance, fixed=fixed)
        # Assemble full parameter table (B, M, P)
        full = theta.new_zeros(bsz, M, len(PARAM_ORDER))
        for i, name in enumerate(THETA):
            if name in PARAM_ORDER:
                full[:, :, PARAM_ORDER.index(name)] = theta[:, i].unsqueeze(1)
        for i, name in enumerate(NUISANCE):
            full[:, :, PARAM_ORDER.index(name)] = eta[:, :, i]
        flat = full.reshape(bsz * M, len(PARAM_ORDER))
        if self.emulator is None:
            # Tests only: still consume η so M=1 and M>1 are not identical.
            wl = torch.linspace(400, 2450, self.n_bands, device=theta.device)
            lai = theta[:, 0].view(bsz, 1, 1)
            cab = theta[:, 1].view(bsz, 1, 1)
            sza = eta[:, :, NUISANCE.index("SZA")].unsqueeze(-1)
            psoil = eta[:, :, NUISANCE.index("psoil")].unsqueeze(-1)
            veg = torch.sigmoid((wl - 700.0) / 40.0) * (1 - torch.exp(-0.35 * lai))
            pig = torch.exp(-((wl - 670.0) ** 2) / (2 * 35.0**2)) * (cab / 80.0)
            shade = 0.15 * (sza / 70.0)
            soil = 0.04 + 0.08 * psoil
            r = (soil + veg * 0.45 - 0.15 * pig - shade).clamp(0.0, 1.0)
            return r if return_samples else r.mean(dim=1)
        r = self.emulator(flat).reshape(bsz, M, -1)
        return r if return_samples else r.mean(dim=1)
