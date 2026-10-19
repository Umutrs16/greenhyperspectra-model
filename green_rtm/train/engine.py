"""Shared pretrain / finetune loops."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import (
    combined_loss,
    masked_huber,
    inverse_frequency_weights,
    rtm_match,
    rtm_marginal_nll,
    residual_penalty,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save(model, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def train_pretrain(model, loader: DataLoader, cfg: dict, out_dir: Path) -> Path:
    device = _device()
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    epochs = int(cfg["train"]["pretrain_epochs"])
    clip = float(cfg["train"]["grad_clip"])
    model.train()
    for epoch in range(epochs):
        running = 0.0
        n = 0
        for batch in tqdm(loader, desc=f"pretrain {epoch+1}/{epochs}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, pretrain=True)
            target = torch.nan_to_num(out.get("recon_target", batch["reflectance"]), nan=0.0)
            pred = out["recon"]
            if pred.shape[-1] != target.shape[-1]:
                target = target[:, : pred.shape[-1]]
            m = out.get("recon_mask")
            if m is None or m.shape[-1] != pred.shape[-1]:
                m = torch.ones_like(pred, dtype=torch.bool)
            else:
                m = m.bool()
            if m.any():
                loss = (pred - target).abs()[m].mean()
            else:
                loss = (pred - target).abs().mean()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if any(p.grad is not None and (not torch.isfinite(p.grad).all()) for p in model.parameters()):
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            running += float(loss.detach()) * len(batch["reflectance"])
            n += len(batch["reflectance"])
        print(f"[pretrain] epoch {epoch+1} loss={running / max(n,1):.4f}")
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("NaN weights during pretrain; refuse to save checkpoint")
    path = out_dir / "pretrain_last.pt"
    _save(model, path, {"epoch": epochs})
    return path


def _finetune_batch_loss(model, batch: dict, cfg: dict) -> torch.Tensor:
    out = model(batch, pretrain=False)
    w = inverse_frequency_weights(batch["trait_mask"])
    loss = masked_huber(
        out["traits"],
        batch["traits"],
        batch["trait_mask"],
        w,
        delta=float(cfg["train"]["huber_delta"]),
    )
    if float(cfg["train"]["lambdas"]["rtm"]) > 0:
        if "rtm_samples" in out:
            loss = loss + float(cfg["train"]["lambdas"]["rtm"]) * rtm_marginal_nll(
                out["rtm_samples"],
                batch["reflectance"],
                batch["valid"],
                scale=float(cfg["train"].get("rtm_laplace_scale", 0.02)),
                match_mae=bool(cfg["train"].get("rtm_nll_match_mae", False)),
            )
        elif "rtm" in out:
            loss = loss + float(cfg["train"]["lambdas"]["rtm"]) * rtm_match(
                torch.nan_to_num(out["rtm"], nan=0.0),
                torch.nan_to_num(batch["reflectance"], nan=0.0),
                batch["valid"],
            )
    if out.get("delta") is not None and out["delta"].ndim == 2 and out["delta"].shape[-1] > 1:
        amp, sm = residual_penalty(out["delta"])
        loss = loss + float(cfg["train"]["lambdas"]["residual"]) * (
            float(cfg["model"].get("residual_amp", 0.05)) * amp
            + float(cfg["model"].get("residual_smooth", 0.20)) * sm
        )
    return loss


def train_finetune(
    model,
    loader: DataLoader,
    cfg: dict,
    out_dir: Path,
    tag: str = "finetune",
    history_path: Path | None = None,
    val_loader: DataLoader | None = None,
) -> Path:
    device = _device()
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]) * 0.5,
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    epochs = int(cfg["train"]["finetune_epochs"])
    clip = float(cfg["train"]["grad_clip"])
    model.train()
    history: list[dict] = []
    for epoch in range(epochs):
        running = 0.0
        n = 0
        for batch in tqdm(loader, desc=f"{tag} {epoch+1}/{epochs}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = _finetune_batch_loss(model, batch, cfg)
            if not torch.isfinite(loss) or not loss.requires_grad:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            running += float(loss.detach()) * len(batch["reflectance"])
            n += len(batch["reflectance"])
        rec = {"epoch": epoch + 1, "tag": tag, "train_loss": running / max(n, 1)}
        if val_loader is not None:
            model.eval()
            v_run = 0.0
            v_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    v_loss = _finetune_batch_loss(model, batch, cfg)
                    if not torch.isfinite(v_loss):
                        continue
                    v_run += float(v_loss) * len(batch["reflectance"])
                    v_n += len(batch["reflectance"])
            rec["val_loss"] = v_run / max(v_n, 1)
            model.train()
        history.append(rec)
        extra = f" val={rec['val_loss']:.4f}" if "val_loss" in rec else ""
        print(f"[{tag}] epoch {epoch+1} loss={rec['train_loss']:.4f}{extra}")
        if history_path is not None:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)
    safe_tag = str(tag).replace("/", "_")
    path = out_dir / f"{safe_tag}_last.pt"
    _save(model, path, {"epoch": epochs, "tag": tag})
    return path


def audit_rtm_pathways(model, batch: dict, cfg: dict) -> dict[str, float]:
    """Gradient norms that show whether L_spectral actually reaches the encoder."""
    device = next(model.parameters()).device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    model.zero_grad(set_to_none=True)
    model.train()
    out = model(batch, pretrain=False)
    from .losses import masked_huber, rtm_match, inverse_frequency_weights

    w = inverse_frequency_weights(batch["trait_mask"])
    l_trait = masked_huber(
        out["traits"], batch["traits"], batch["trait_mask"], w, delta=float(cfg["train"]["huber_delta"])
    )
    rec = {"has_rtm": float("rtm" in out), "l_trait": float(l_trait.detach())}
    cls = out["cls"]
    g_t = torch.autograd.grad(l_trait, cls, retain_graph=True, allow_unused=True)[0]
    rec["g_trait_cls"] = float(g_t.norm().detach()) if g_t is not None else 0.0
    if "rtm" in out:
        l_rtm = rtm_match(
            torch.nan_to_num(out["rtm"], nan=0.0),
            torch.nan_to_num(batch["reflectance"], nan=0.0),
            batch["valid"],
        )
        rec["l_rtm"] = float(l_rtm.detach())
        g_r = torch.autograd.grad(l_rtm, cls, retain_graph=True, allow_unused=True)[0]
        rec["g_rtm_cls"] = float(g_r.norm().detach()) if g_r is not None else 0.0
        if "theta_phys" in out:
            g_th = torch.autograd.grad(l_rtm, out["theta_phys"], retain_graph=True, allow_unused=True)[0]
            rec["g_rtm_theta"] = float(g_th.norm().detach()) if g_th is not None else 0.0
        rec["ratio_rtm_over_trait"] = rec["g_rtm_cls"] / max(rec["g_trait_cls"], 1e-12)
    return rec
