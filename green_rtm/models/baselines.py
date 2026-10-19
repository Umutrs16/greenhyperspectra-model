"""Paper baselines B0–B2 style models (not copies of HyspecSSL weights)."""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
import torch
from torch import nn


class PLSRBaseline:
    """Trait-specific PLSR: each output is fit only on rows that have that label."""

    def __init__(self, n_components: int = 20):
        self.n_components = n_components
        self.models: list[PLSRegression] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PLSRBaseline":
        self.models = []
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        for k in range(y.shape[1]):
            m = np.isfinite(y[:, k])
            n = int(m.sum())
            if n < 8:
                self.models.append(None)
                continue
            nc = int(min(self.n_components, n - 1, x.shape[1]))
            pls = PLSRegression(n_components=max(nc, 1))
            pls.fit(x[m], y[m, k])
            self.models.append(pls)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.full((len(x), len(self.models)), np.nan, dtype=np.float32)
        for k, pls in enumerate(self.models):
            if pls is None:
                continue
            out[:, k] = np.asarray(pls.predict(x), dtype=np.float32).reshape(-1)
        return out


class RFBaseline:
    def __init__(self, n_estimators: int = 200, seed: int = 42):
        self.imputer = SimpleImputer(strategy="median")
        self.model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RFBaseline":
        self.model.fit(x, self.imputer.fit_transform(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


class CNN1D(nn.Module):
    def __init__(self, n_bands: int = 1721, n_traits: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 32, 128), nn.GELU(), nn.Linear(128, n_traits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.head(self.net(x))
