"""Trait-wise metrics and source-blocked comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _finite_pair(y: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(y) & np.isfinite(yhat)
    return y[m], yhat[m]


def trait_metrics(y: np.ndarray, yhat: np.ndarray, trait: str) -> dict:
    yt, yp = _finite_pair(np.asarray(y, dtype=np.float64), np.asarray(yhat, dtype=np.float64))
    n = int(len(yt))
    if n < 3:
        return {"trait": trait, "n": n, "r2": np.nan, "rmse": np.nan, "mae": np.nan, "nrmse": np.nan, "bias": np.nan, "spearman": np.nan}
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    rng = float(np.ptp(yt)) if np.ptp(yt) > 0 else np.nan
    rho = float(stats.spearmanr(yt, yp).statistic)
    return {
        "trait": trait,
        "n": n,
        "r2": float(r2),
        "rmse": rmse,
        "mae": mae,
        "nrmse": float(rmse / rng) if rng == rng else np.nan,
        "bias": float(np.mean(yp - yt)),
        "spearman": rho,
    }


def table_from_predictions(df: pd.DataFrame, traits: list[str]) -> pd.DataFrame:
    rows = []
    for t in traits:
        rows.append(trait_metrics(df[t].to_numpy(), df[f"{t}_pred"].to_numpy(), t))
    return pd.DataFrame(rows)


def paired_delta(err_a: np.ndarray, err_b: np.ndarray) -> dict:
    """Paired Wilcoxon on |error| per held-out source, plus mean ΔRMSE."""
    a = np.asarray(err_a, dtype=np.float64)
    b = np.asarray(err_b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 6:
        return {"delta_rmse": np.nan, "wilcoxon_p": np.nan, "n": int(len(a))}
    try:
        p = float(stats.wilcoxon(a, b, alternative="two-sided").pvalue)
    except ValueError:
        p = np.nan
    return {
        "delta_rmse": float(np.sqrt(np.mean(b**2)) - np.sqrt(np.mean(a**2))),
        "wilcoxon_p": p,
        "n": int(len(a)),
    }


def pooled_oof_r2(frame: pd.DataFrame, trait: str) -> float:
    """Concatenate all OOF (y, ŷ) pairs for one trait and compute a single R²."""
    y = pd.to_numeric(frame[trait], errors="coerce").to_numpy()
    yp = pd.to_numeric(frame[f"{trait}_pred"], errors="coerce").to_numpy()
    return trait_metrics(y, yp, trait)["r2"]
