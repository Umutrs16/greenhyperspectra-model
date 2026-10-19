"""Latin-hypercube PROSPECT-PRO + 4SAIL library with Sobol sanity checks."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .prospect_pro import simulate_row

PARAM_COLS = [
    "LAI",
    "Cab",
    "Cw",
    "Cm",
    "Cp",
    "Car",
    "Anth",
    "Cbc",
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


def _scale(u: np.ndarray, lo: float, hi: float, log: bool = False) -> np.ndarray:
    if log:
        return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
    return lo + u * (hi - lo)


def draw_parameters(n: int, cfg: dict, seed: int = 42) -> pd.DataFrame:
    th = cfg["rtm"]["theta"]
    nu = cfg["rtm"]["nuisance"]
    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=16, seed=seed)
    u = sampler.random(n)
    lai = _scale(u[:, 0], *th["LAI"])
    cab = _scale(u[:, 1], *th["Cab"])
    cw = _scale(u[:, 2], *th["Cw"], log=True)
    cm = _scale(u[:, 3], *th["Cm"], log=True)
    # Cp is a fraction of Cm, then expanded to the configured range.
    cp_frac = _scale(u[:, 4], 0.05, 0.45)
    cp = np.clip(cm * cp_frac, th["Cp"][0], th["Cp"][1])
    cbc = np.clip(cm - cp, 1e-5, None)
    car = _scale(u[:, 5], *th["Car"])
    anth = _scale(u[:, 6], *th["Anth"])
    nleaf = _scale(u[:, 7], *nu["N"])
    brown = _scale(u[:, 8], *nu["Cbrown"])
    sza = _scale(u[:, 9], *nu["SZA"])
    vza = _scale(u[:, 10], *nu["VZA"])
    raa = _scale(u[:, 11], *nu["RAA"])
    ala = _scale(u[:, 12], *nu["ALA"])
    psoil = _scale(u[:, 13], *nu["psoil"])
    rsoil = _scale(u[:, 14], *nu["rsoil"])
    if cfg["rtm"].get("hotspot_log_uniform", True):
        hspot = _scale(u[:, 15], *nu["hotspot"], log=True)
    else:
        hspot = _scale(u[:, 15], *nu["hotspot"])
    return pd.DataFrame(
        {
            "LAI": lai,
            "Cab": cab,
            "Cw": cw,
            "Cm": cm,
            "Cp": cp,
            "Car": car,
            "Anth": anth,
            "Cbc": cbc,
            "N": nleaf,
            "Cbrown": brown,
            "SZA": sza,
            "VZA": vza,
            "RAA": raa,
            "ALA": ala,
            "psoil": psoil,
            "rsoil": rsoil,
            "hotspot": hspot,
        }
    )


def _one(row_dict: dict) -> np.ndarray:
    return simulate_row(row_dict, ghs_grid=True)


def simulate_library(params: pd.DataFrame, n_jobs: int = 1) -> np.ndarray:
    records = params.to_dict("records")
    if n_jobs == 1:
        specs = [_one(r) for r in records]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            specs = list(ex.map(_one, records, chunksize=16))
    return np.stack(specs, axis=0).astype(np.float32)


def sobol_sanity(params: pd.DataFrame, spectra: np.ndarray, traits=("LAI", "Cab", "Cw", "Cm", "Cp")) -> pd.DataFrame:
    """Cheap first-order correlation sanity check (not a full Sobol index)."""
    rows = []
    for t in traits:
        x = params[t].to_numpy()
        # Use NIR / red-edge / SWIR summaries that should correlate with each trait.
        if t == "LAI":
            y = spectra[:, 500:700].mean(axis=1)  # ~NIR on GHS grid
        elif t == "Cab":
            y = spectra[:, 80:180].mean(axis=1)  # red / red-edge
        elif t == "Cw":
            y = spectra[:, 900:1100].mean(axis=1)
        else:
            y = spectra[:, 1400:].mean(axis=1)
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            r = 0.0
        else:
            r = float(np.corrcoef(x, y)[0, 1])
        rows.append({"trait": t, "proxy_corr": r, "n": len(params)})
    return pd.DataFrame(rows)


def write_library(params: pd.DataFrame, spectra: np.ndarray, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p_par = out_dir / "rtm_params.parquet"
    p_spec = out_dir / "rtm_spectra.npy"
    params.to_parquet(p_par, index=False)
    np.save(p_spec, spectra)
    return {"params": p_par, "spectra": p_spec}
