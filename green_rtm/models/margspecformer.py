"""RTM-MargSpecFormer: variable-band encoder + marginalized RTM + low-rank residual."""

from __future__ import annotations

import torch
from torch import nn

from .encoder import VariableBandEncoder
from .heads import MultiTraitHead, ReconstructHead
from .residual import LowRankResidual
from .rtm_marginalizer import RTMMarginalizer


class RTMMargSpecFormer(nn.Module):
    def __init__(self, cfg: dict, emulator=None, physics_weight: torch.Tensor | None = None):
        super().__init__()
        m = cfg["model"]
        n_traits = len(cfg["traits"]["primary"])
        n_bands = int(cfg["spectra"]["n_bands"])
        self.cfg = cfg
        self.encoder = VariableBandEncoder(
            dim=int(m["dim"]),
            depth=int(m["depth"]),
            heads=int(m["heads"]),
            mlp_ratio=float(m["mlp_ratio"]),
            dropout=float(m["dropout"]),
            patch_size=int(m.get("patch_size", 10)),
        )
        self.trait_head = MultiTraitHead(int(m["dim"]), n_traits=n_traits, dropout=float(m["dropout"]))
        self.recon_head = ReconstructHead(int(m["dim"]))
        self.marginalizer = RTMMarginalizer(
            emulator=emulator,
            M=int(cfg["rtm"]["M_train"]),
            n_bands=n_bands,
        )
        if bool(m.get("bound_nuisance", False)):
            from .rtm_marginalizer import NUISANCE
            lo, hi, init = [], [], []
            theta = cfg["rtm"]["theta"]
            nuisance = cfg["rtm"]["nuisance"]
            physical_init = {
                "Car": 8.0, "Anth": 0.5, "N": 1.6, "Cbrown": 0.1,
                "SZA": 30.0, "VZA": 8.0, "RAA": 90.0, "ALA": 45.0,
                "psoil": 0.5, "rsoil": 1.0, "hotspot": 0.02,
            }
            for name in NUISANCE:
                bounds = nuisance.get(name, theta.get(name))
                lo.append(float(bounds[0]))
                hi.append(float(bounds[1]))
                init.append(float(physical_init[name]))
            self.marginalizer.enable_library_bounds(lo, hi, init, latent_std=0.5)
        self.residual = LowRankResidual(int(m["dim"]), n_bands=n_bands, rank=int(m["residual_rank"]))
        if physics_weight is None:
            physics_weight = torch.ones(n_bands)
        self.register_buffer("physics_weight", physics_weight.float())
        self.register_buffer("trait_mean", torch.zeros(n_traits))
        self.register_buffer("trait_std", torch.ones(n_traits))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, int(m["dim"])))
        nn.init.normal_(self.mask_token, std=0.02)
        if emulator is not None:
            for p in self.marginalizer.emulator.parameters():
                p.requires_grad = False
        self.marginalizer.fixed_nuisance = not bool(m.get("use_marginalization", True))
        mins = torch.tensor([0.05, 1.0, 0.001, 0.001, 0.0002], dtype=torch.float32)
        maxs = torch.tensor([9.0, 90.0, 0.07, 0.04, 0.012], dtype=torch.float32)
        self.register_buffer("theta_min", mins)
        self.register_buffer("theta_max", maxs)

    def set_trait_stats(self, stats: dict[str, tuple[float, float]], order: list[str] | None = None) -> None:
        keys = order or ["LAI", "Cab", "Cw", "Cm", "Cp"]
        mean = torch.tensor([stats[k][0] for k in keys], dtype=torch.float32)
        std = torch.tensor([stats[k][1] for k in keys], dtype=torch.float32)
        self.trait_mean.copy_(mean)
        self.trait_std.copy_(std)

    def make_mask(self, valid: torch.Tensor) -> torch.Tensor:
        """60% random + 40% Jacobian-guided masking among valid bands."""
        bsz, n = valid.shape
        ratio = float(self.cfg["model"]["mask_ratio"])
        jac_frac = float(self.cfg["model"]["jacobian_mask_frac"]) if self.cfg["model"].get("use_jacobian_mask", True) else 0.0
        n_mask = max(1, int(n * ratio))
        n_jac = int(n_mask * jac_frac)
        n_rand = n_mask - n_jac
        scores = torch.rand(bsz, n, device=valid.device)
        if jac_frac > 0:
            w = self.physics_weight.to(valid.device)
            if w.numel() == n:
                scores = scores * (1.0 - jac_frac) + (1.0 - w / (w.max() + 1e-6)) * jac_frac
        scores = scores.masked_fill(valid < 0.5, 1e9)
        # Lowest scores are masked (high physics weight preferred when jac_frac>0 after inversion)
        if jac_frac > 0 and w.numel() == n:
            phys = w / (w.max() + 1e-6)
            jac_score = -phys.unsqueeze(0).expand(bsz, -1) + 0.05 * torch.rand_like(scores)
            jac_score = jac_score.masked_fill(valid < 0.5, 1e9)
            _, jac_idx = torch.topk(jac_score, k=max(n_jac, 1), dim=-1, largest=False)
        else:
            jac_idx = scores.new_zeros(bsz, 0, dtype=torch.long)
        _, rand_idx = torch.topk(scores, k=max(n_rand, 1), dim=-1, largest=False)
        mask = torch.zeros(bsz, n, dtype=torch.bool, device=valid.device)
        if jac_idx.numel():
            mask.scatter_(1, jac_idx, True)
        mask.scatter_(1, rand_idx, True)
        mask = mask & (valid > 0.5)
        return mask

    def encode(self, batch: dict[str, torch.Tensor], mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch = dict(batch)
        if mask is not None:
            r = batch["reflectance"].clone()
            r = r.masked_fill(mask, 0.0)
            batch["reflectance"] = r
        return self.encoder(batch)

    def _pool_like_encoder(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder._pool(x, self.encoder.patch_size)

    def forward(self, batch: dict[str, torch.Tensor], pretrain: bool = False) -> dict[str, torch.Tensor]:
        r = torch.nan_to_num(batch["reflectance"], nan=0.0, posinf=1.2, neginf=0.0)
        batch = {**batch, "reflectance": r}
        mask = self.make_mask(batch["valid"]) if pretrain else None
        enc = self.encode(batch, mask=mask)
        traits, log_var = self.trait_head(enc["cls"])
        recon = self.recon_head(enc["tokens"])
        out = {
            "traits": traits,
            "log_var": log_var,
            "recon": recon,
            "cls": enc["cls"],
            "mask": mask if mask is not None else torch.zeros_like(batch["valid"], dtype=torch.bool),
            "recon_target": self._pool_like_encoder(r),
            "recon_mask": self._pool_like_encoder((mask if mask is not None else torch.zeros_like(batch["valid"])).float()) > 0.25,
        }
        theta = None
        if not pretrain:
            need_phys = bool(self.cfg["model"].get("use_protein_fraction", False)) or bool(
                self.cfg["model"].get("use_rtm", True)
            )
            if need_phys:
                theta = self._traits_to_physical(traits)
            if bool(self.cfg["model"].get("use_protein_fraction", False)):
                traits = traits.clone()
                traits[:, 4] = (theta[:, 4] - self.trait_mean[4]) / self.trait_std[4].clamp_min(1e-8)
                out["traits"] = traits
        if pretrain:
            out["delta"] = recon.new_zeros(recon.shape[0], 1)
            return out
        if not self.cfg["model"].get("use_rtm", True):
            out["delta"] = recon.new_zeros(recon.shape[0], 1)
            return out
        out["theta_phys"] = theta
        known = batch.get("known_nuisance")
        fix = self.cfg["model"].get("fix_geometry") or {}
        if fix:
            known = dict(known or {})
            bsz = theta.shape[0]
            for name, value in fix.items():
                known[name] = torch.full((bsz,), float(value), device=theta.device, dtype=theta.dtype)
        if bool(self.cfg["model"].get("use_likelihood_marginal", False)):
            samples = self.marginalizer(theta, known_nuisance=known, return_samples=True)
            out["rtm_samples"] = samples
            out["rtm"] = samples.mean(dim=1)
        else:
            out["rtm"] = self.marginalizer(theta, known_nuisance=known)
        if self.cfg["model"].get("use_low_rank_residual", False):
            out["delta"] = self.residual(enc["cls"])
            out["rtm_plus_res"] = out["rtm"] + out["delta"]
        else:
            out["delta"] = out["rtm"].new_zeros(out["rtm"].shape[0], 1)
            out["rtm_plus_res"] = out["rtm"]
        return out

    def _traits_to_physical(self, z: torch.Tensor) -> torch.Tensor:
        """Map the trait head (z-scored in the training loop) back to RTM units."""
        phys = torch.nan_to_num(z * self.trait_std + self.trait_mean, nan=0.0)
        phys = phys.clamp(self.theta_min, self.theta_max)
        if bool(self.cfg["model"].get("use_protein_fraction", False)):
            cm = phys[:, 3]
            fp = torch.sigmoid(z[:, 4])
            cp = (fp * cm).clamp(self.theta_min[4], self.theta_max[4])
            phys = phys.clone()
            phys[:, 4] = cp
        return phys


def build_model(cfg: dict, emulator=None, physics_weight=None) -> RTMMargSpecFormer:
    return RTMMargSpecFormer(cfg, emulator=emulator, physics_weight=physics_weight)
