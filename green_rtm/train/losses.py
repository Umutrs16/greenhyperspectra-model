"""Masked multi-trait Huber, Jacobian-weighted reconstruction, residual, RTM."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    delta: float = 1.0,
) -> torch.Tensor:
    """L_trait = Σ_k m_k ω_k Huber(ŷ,y) / (Σ m_k + ε)."""
    w = mask if weights is None else mask * weights
    if float(w.sum().detach()) <= 0:
        # Stay on the graph so empty-label batches do not break backward().
        return pred.sum() * 0.0
    err = F.huber_loss(pred, torch.nan_to_num(target, nan=0.0), delta=delta, reduction="none")
    denom = w.sum() + 1e-6
    return (err * w).sum() / denom


def jacobian_weighted_recon(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    physics_weight: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """L_mask = Σ_i [α + (1-α) w_phys(λ_i)] |R_i − R̂_i| on masked tokens."""
    w = alpha + (1.0 - alpha) * physics_weight.to(pred.device)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    err = (pred - target).abs() * w
    m = mask.float()
    return (err * m).sum() / (m.sum() + 1e-6)


def residual_penalty(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amp = delta.pow(2).mean()
    smooth = (delta[:, 1:] - delta[:, :-1]).pow(2).mean()
    return amp, smooth


def rtm_marginal_nll(
    samples: torch.Tensor,
    obs: torch.Tensor,
    valid: torch.Tensor,
    scale: float = 0.02,
    match_mae: bool = False,
) -> torch.Tensor:
    """-log (1/M Σ_m exp(-L_m / b)) with L_m = MAE on valid bands (Laplace spectral likelihood).

    match_mae multiplies by b so the large-b limit is the average per-draw residual,
    on a scale comparable to the mean-spectrum MAE, rather than that average divided by b.
    """
    m = valid.unsqueeze(1).float()
    err = (samples - torch.nan_to_num(obs, nan=0.0).unsqueeze(1)).abs()
    per = (err * m).sum(dim=-1) / (m.sum(dim=-1) + 1e-6)
    b = max(float(scale), 1e-6)
    nll = -(torch.logsumexp(-per / b, dim=1) - math.log(float(samples.shape[1]))).mean()
    if match_mae:
        nll = nll * b
    return nll


def rtm_match(pred_spec: torch.Tensor, rtm_spec: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    err = (pred_spec - rtm_spec).abs()
    m = valid.float()
    return (err * m).sum() / (m.sum() + 1e-6)


def inverse_frequency_weights(mask: torch.Tensor) -> torch.Tensor:
    freq = mask.sum(dim=0).clamp_min(1.0)
    w = (freq.sum() / freq)
    return (w / w.mean()).to(mask.dtype)


def combined_loss(out: dict, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    lam = cfg["train"]["lambdas"]
    delta = float(cfg["train"]["huber_delta"])
    trait_w = inverse_frequency_weights(batch["trait_mask"])
    l_trait = masked_huber(out["traits"], batch["traits"], batch["trait_mask"], trait_w, delta=delta)
    target = out.get("recon_target", batch["reflectance"])
    mask = out.get("recon_mask", out["mask"])
    phys = out.get(
        "physics_weight",
        batch.get("physics_weight", torch.ones(out["recon"].shape[-1], device=out["recon"].device)),
    )
    if phys.numel() != out["recon"].shape[-1]:
        # Pool 1 nm weights onto the encoder patch grid.
        p = max(int(round(phys.numel() / max(out["recon"].shape[-1], 1))), 1)
        pad = (p - phys.numel() % p) % p
        w = torch.nn.functional.pad(phys.reshape(-1), (0, pad))
        phys = w.view(-1, p).mean(1)[: out["recon"].shape[-1]]
    if target.shape[-1] != out["recon"].shape[-1]:
        p = max(int(round(target.shape[-1] / max(out["recon"].shape[-1], 1))), 1)
        pad = (p - target.shape[-1] % p) % p
        t = torch.nn.functional.pad(target, (0, pad))
        target = t.view(t.shape[0], -1, p).mean(-1)[:, : out["recon"].shape[-1]]
        m = torch.nn.functional.pad(mask.float(), (0, pad))
        mask = m.view(m.shape[0], -1, p).mean(-1)[:, : out["recon"].shape[-1]] > 0.5
    target = torch.nan_to_num(target, nan=0.0)
    l_mask = jacobian_weighted_recon(out["recon"], target, mask, phys, alpha=0.5)
    l_rtm = torch.zeros((), device=l_trait.device)
    if "rtm_plus_res" in out:
        l_rtm = rtm_match(
            torch.nan_to_num(out["rtm_plus_res"], nan=0.0),
            torch.nan_to_num(batch["reflectance"], nan=0.0),
            batch["valid"],
        )
    amp, sm = residual_penalty(out["delta"])
    l_res = float(cfg["model"].get("residual_amp", 0.05)) * amp + float(cfg["model"].get("residual_smooth", 0.20)) * sm
    l_trait = torch.nan_to_num(l_trait, nan=0.0)
    l_mask = torch.nan_to_num(l_mask, nan=0.0)
    l_rtm = torch.nan_to_num(l_rtm, nan=0.0)
    l_res = torch.nan_to_num(l_res, nan=0.0)
    total = (
        float(lam["trait"]) * l_trait
        + float(lam["mask"]) * l_mask
        + float(lam["rtm"]) * l_rtm
        + float(lam["residual"]) * l_res
    )
    logs = {
        "loss": float(total.detach()),
        "l_trait": float(l_trait.detach()),
        "l_mask": float(l_mask.detach()),
        "l_rtm": float(l_rtm.detach()),
        "l_res": float(l_res.detach()),
    }
    return total, logs
