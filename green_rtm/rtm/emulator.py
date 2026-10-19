"""Differentiable MLP forward emulator F_φ(θ, η) → R(λ)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PARAM_ORDER = [
    "LAI",
    "Cab",
    "Cw",
    "Cm",
    "Cp",
    "Car",
    "Anth",
    "N",
    "Cbrown",
    "SZA",
    "VZA",
    "RAA",
    "ALA",
    "psoil",
    "rsoil",
    "hotspot",
]


class ForwardEmulator(nn.Module):
    def __init__(self, n_params: int = 16, n_bands: int = 1721, hidden: list[int] | None = None):
        super().__init__()
        hidden = hidden or [256, 256, 256]
        layers: list[nn.Module] = []
        d = n_params
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers.append(nn.Linear(d, n_bands))
        self.net = nn.Sequential(*layers)
        self.register_buffer("p_mean", torch.zeros(n_params))
        self.register_buffer("p_std", torch.ones(n_params))
        self.register_buffer("r_mean", torch.zeros(n_bands))
        self.register_buffer("r_std", torch.ones(n_bands))

    def set_stats(self, params: np.ndarray, spectra: np.ndarray) -> None:
        self.p_mean.copy_(torch.tensor(params.mean(0), dtype=torch.float32))
        self.p_std.copy_(torch.tensor(params.std(0) + 1e-6, dtype=torch.float32))
        self.r_mean.copy_(torch.tensor(spectra.mean(0), dtype=torch.float32))
        self.r_std.copy_(torch.tensor(spectra.std(0) + 1e-6, dtype=torch.float32))

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        z = (params - self.p_mean) / self.p_std
        return self.net(z) * self.r_std + self.r_mean

    def pack_params(self, table) -> np.ndarray:
        import pandas as pd

        if isinstance(table, pd.DataFrame):
            return table[PARAM_ORDER].to_numpy(np.float32)
        return np.asarray(table, dtype=np.float32)


def train_emulator(
    params: np.ndarray,
    spectra: np.ndarray,
    hidden: list[int],
    epochs: int = 40,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: str | None = None,
    out_path: Path | None = None,
) -> ForwardEmulator:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardEmulator(n_params=params.shape[1], n_bands=spectra.shape[1], hidden=hidden)
    model.set_stats(params, spectra)
    model.to(device)
    ds = TensorDataset(torch.from_numpy(params), torch.from_numpy(spectra))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(xb)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[emulator] epoch {epoch+1}/{epochs}  mae~{running / len(ds):.5f}")
    model.eval()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "hidden": hidden, "param_order": PARAM_ORDER}, out_path)
    return model


def load_emulator(path: Path, device: str | None = None) -> ForwardEmulator:
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model = ForwardEmulator(hidden=ckpt.get("hidden"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
