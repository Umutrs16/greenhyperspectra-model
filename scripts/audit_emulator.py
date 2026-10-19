#!/usr/bin/env python
"""Hold-out audit of the 4,000-sample PROSPECT-PRO/4SAIL emulator.

Reports reflectance fidelity on N=1000 unused Latin-hypercube states and
compares emulator Jacobians with finite-difference RTM Jacobians on a subset.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from green_rtm.config import load_config, resolve_path
from green_rtm.eval.figures import fig_emulator_band_error
from green_rtm.rtm.emulator import PARAM_ORDER, load_emulator
from green_rtm.rtm.generate_library import draw_parameters, simulate_library
from green_rtm.spectra import GHS_WAVELENGTHS, PRIMARY_TRAITS


TRAITS = list(PRIMARY_TRAITS)
# Relative steps keep finite differences on the scale of each trait.
REL_STEP = {
    "LAI": 0.02,
    "Cab": 0.02,
    "Cw": 0.03,
    "Cm": 0.03,
    "Cp": 0.05,
}
ABS_FLOOR = {
    "LAI": 0.02,
    "Cab": 0.20,
    "Cw": 2e-4,
    "Cm": 1e-4,
    "Cp": 3e-5,
}


def _pack(params: pd.DataFrame) -> np.ndarray:
    return params[list(PARAM_ORDER)].to_numpy(np.float32)


def spectral_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0)))


def reflectance_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    sam = spectral_angle(true, pred)
    band_rmse = np.sqrt(np.mean(err**2, axis=0))
    band_bias = err.mean(axis=0)
    return {
        "n": int(len(true)),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r2_forward": r2,
        "sam_mean_deg": float(sam.mean()),
        "sam_median_deg": float(np.median(sam)),
        "band_rmse_mean": float(band_rmse.mean()),
        "band_rmse_p95": float(np.quantile(band_rmse, 0.95)),
        "band_rmse": band_rmse.astype(np.float32),
        "band_bias": band_bias.astype(np.float32),
    }


def jacobian_pair(emu, row: dict, traits=TRAITS) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    from green_rtm.rtm.prospect_pro import simulate_row

    base = simulate_row(row)
    x0 = np.array([[row[k] for k in PARAM_ORDER]], dtype=np.float32)
    with torch.no_grad():
        r0 = emu(torch.from_numpy(x0)).numpy()[0]
    j_rtm, j_emu = {}, {}
    for t in traits:
        step = max(ABS_FLOOR[t], REL_STEP[t] * abs(float(row[t])))
        pert = dict(row)
        pert[t] = float(row[t]) + step
        # PROSPECT-PRO sees protein and CBC, not dry mass. Keep Cm = Cp + Cbc.
        if t == "Cm":
            pert["Cbc"] = max(float(pert["Cm"]) - float(pert.get("Cp", 0.0)), 1e-5)
        if t == "Cp":
            pert["Cbc"] = max(float(pert.get("Cm", pert["Cp"])) - float(pert["Cp"]), 1e-5)
        r_rtm = simulate_row(pert)
        j_rtm[t] = (r_rtm - base) / step
        xp = x0.copy()
        xp[0, PARAM_ORDER.index(t)] += step
        with torch.no_grad():
            rp = emu(torch.from_numpy(xp)).numpy()[0]
        j_emu[t] = (rp - r0) / step
    return j_rtm, j_emu


def main() -> None:
    cfg = load_config("emit_free")
    tables = resolve_path(cfg, "tables")
    figs = resolve_path(cfg, "figures")
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    print("drawing unused Latin-hypercube hold-out (seed=2026, n=1000)", flush=True)
    params = draw_parameters(1000, cfg, seed=2026)
    true = simulate_library(params, n_jobs=max(1, (os.cpu_count() or 4) // 2))
    emu = load_emulator(resolve_path(cfg, "rtm_library") / "forward_emulator.pt")
    emu.cpu().eval()
    with torch.no_grad():
        pred = emu(torch.from_numpy(_pack(params))).numpy()
    rec = reflectance_metrics(true, pred)
    band = pd.DataFrame(
        {
            "wavelength_nm": GHS_WAVELENGTHS[: rec["band_rmse"].size],
            "rmse": rec["band_rmse"],
            "bias": rec["band_bias"],
        }
    )
    band.to_csv(tables / "emulator_band_error.csv", index=False)
    summary = {k: v for k, v in rec.items() if k not in {"band_rmse", "band_bias"}}
    pd.DataFrame([summary]).to_csv(tables / "emulator_holdout_metrics.csv", index=False)
    fig_emulator_band_error(band, figs / "figS_emulator_band_error.png")
    print("hold-out reflectance", summary, flush=True)

    print("Jacobian finite-difference comparison n=150", flush=True)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(params), size=150, replace=False)
    rows = []
    for i in idx:
        j_rtm, j_emu = jacobian_pair(emu, params.iloc[int(i)].to_dict())
        for t in TRAITS:
            a = j_rtm[t]
            b = j_emu[t]
            if (not np.isfinite(a).all()) or (not np.isfinite(b).all()) or np.linalg.norm(a) < 1e-8:
                corr = np.nan
                rel = np.nan
            elif np.std(a) < 1e-12 or np.std(b) < 1e-12:
                corr = np.nan
                rel = float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a**2)) + 1e-12))
            else:
                corr = float(np.corrcoef(a, b)[0, 1])
                rel = float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a**2)) + 1e-12))
            rows.append(
                {
                    "trait": t,
                    "corr": corr,
                    "rel_rmse": rel,
                    "rtm_l2": float(np.linalg.norm(a)),
                    "emu_l2": float(np.linalg.norm(b)),
                }
            )
    jac = pd.DataFrame(rows)
    jac.to_csv(tables / "emulator_jacobian_pairs.csv", index=False)
    summ = jac.groupby("trait").agg(
        n=("corr", "size"),
        corr_mean=("corr", "mean"),
        corr_median=("corr", "median"),
        rel_rmse_mean=("rel_rmse", "mean"),
        rel_rmse_median=("rel_rmse", "median"),
    ).reset_index()
    summ.to_csv(tables / "emulator_jacobian_audit.csv", index=False)
    print(summ.to_string(index=False), flush=True)
    print("wrote emulator audit tables", flush=True)


if __name__ == "__main__":
    main()
