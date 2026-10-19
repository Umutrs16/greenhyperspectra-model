"""PROSPECT-PRO leaf optics + 4SAIL canopy using official Féret coefficients.

Coefficients: `spec_prospect_full_range` from jbferet/prospect
(Féret, Berger, de Boissieu & Malenovský, RSE 2021, 252:112173).
4SAIL is executed via the installed `prosail` package (Verhoef).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import prosail
from prosail.prospect_d import prospect_d

from ..spectra import FULL_LAMBDA, to_ghs_grid

COEFF_PATH = Path(__file__).with_name("spec_prospect_full_range.csv")


@lru_cache(maxsize=1)
def load_prospect_pro_coeffs() -> dict[str, np.ndarray]:
    df = pd.read_csv(COEFF_PATH)
    if len(df) != 2101:
        raise RuntimeError(f"Expected 2101 PROSPECT wavelengths, got {len(df)}")
    return {
        "lambda": df["lambda"].to_numpy(np.float64),
        "nr": df["nrefrac"].to_numpy(np.float64),
        "kab": df["sac_chl"].to_numpy(np.float64),
        "kcar": df["sac_car"].to_numpy(np.float64),
        "kant": df["sac_ant"].to_numpy(np.float64),
        "kbrown": df["sac_brown"].to_numpy(np.float64),
        "kw": df["sac_ewt"].to_numpy(np.float64),
        "kprot": df["sac_prot"].to_numpy(np.float64),
        "kcbc": df["sac_cbc"].to_numpy(np.float64),
        "klma": df["sac_lma"].to_numpy(np.float64),
    }


def run_prospect_pro(
    N: float,
    cab: float,
    car: float,
    cbrown: float,
    cw: float,
    prot: float,
    cbc: float,
    ant: float = 0.0,
    alpha: float = 40.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leaf directional-hemispherical reflectance / transmittance, 400–2500 nm."""
    c = load_prospect_pro_coeffs()
    # Reuse the validated Stokes plate solver. Protein + CBC replace LMA.
    km = c["kprot"] * float(prot) + c["kcbc"] * float(cbc)
    wv, refl, trans = prospect_d(
        float(N),
        float(cab),
        float(car),
        float(cbrown),
        float(cw),
        1.0,
        float(ant),
        c["nr"],
        c["kab"],
        c["kcar"],
        c["kbrown"],
        c["kw"],
        km,
        c["kant"],
        alpha=alpha,
    )
    return wv.astype(np.float32), refl.astype(np.float32), trans.astype(np.float32)


def run_prospectpro_4sail(
    N: float,
    cab: float,
    car: float,
    cbrown: float,
    cw: float,
    prot: float,
    cbc: float,
    lai: float,
    lidfa: float,
    hspot: float,
    tts: float,
    tto: float,
    psi: float,
    ant: float = 0.0,
    psoil: float = 0.5,
    rsoil: float = 1.0,
    typelidf: int = 2,
    lidfb: float = 0.0,
    ghs_grid: bool = True,
) -> np.ndarray:
    """Canopy directional reflectance (SDR). Default 1721-band GHS grid."""
    _wv, refl, trans = run_prospect_pro(N, cab, car, cbrown, cw, prot, cbc, ant=ant)
    rho = prosail.run_sail(
        refl,
        trans,
        lai=float(lai),
        lidfa=float(lidfa),
        hspot=float(hspot),
        tts=float(tts),
        tto=float(tto),
        psi=float(psi),
        typelidf=typelidf,
        lidfb=lidfb,
        factor="SDR",
        rsoil=float(rsoil),
        psoil=float(psoil),
    )
    rho = np.clip(np.asarray(rho, dtype=np.float32), 0.0, 1.2)
    if ghs_grid:
        return to_ghs_grid(rho)
    return rho


DEFAULT_FIXED_NUISANCE = dict(
    N=1.5,
    Cbrown=0.0,
    ALA=57.3,
    psoil=0.8,
    rsoil=1.0,
    hotspot=0.01,
    SZA=30.0,
    VZA=0.0,
    RAA=0.0,
    Car=8.0,
    Anth=0.0,
)


def simulate_row(params: dict, ghs_grid: bool = True) -> np.ndarray:
    cm = float(params.get("Cm", params.get("prot", 0.0) + params.get("cbc", 0.008)))
    cp = float(params.get("Cp", params.get("prot", 0.2 * cm)))
    cbc = float(params.get("Cbc", max(cm - cp, 1e-5)))
    return run_prospectpro_4sail(
        N=float(params.get("N", 1.5)),
        cab=float(params["Cab"]),
        car=float(params.get("Car", 8.0)),
        cbrown=float(params.get("Cbrown", 0.0)),
        cw=float(params["Cw"]),
        prot=cp,
        cbc=cbc,
        lai=float(params["LAI"]),
        lidfa=float(params.get("ALA", params.get("lidfa", 57.3))),
        hspot=float(params.get("hotspot", params.get("hspot", 0.01))),
        tts=float(params.get("SZA", 30.0)),
        tto=float(params.get("VZA", 0.0)),
        psi=float(params.get("RAA", 0.0)),
        ant=float(params.get("Anth", 0.0)),
        psoil=float(params.get("psoil", 0.5)),
        rsoil=float(params.get("rsoil", 1.0)),
        ghs_grid=ghs_grid,
    )
