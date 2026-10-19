#!/usr/bin/env python
"""Redraw Fig. 7 and Fig. S1 from the reported 50k emulator (and 4k overlay on S1)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch

from green_rtm.config import load_config, resolve_path
from green_rtm.eval.figures import fig_emulator_band_compare, fig_jacobian
from green_rtm.rtm.emulator import PARAM_ORDER, load_emulator
from green_rtm.rtm.jacobian import bandwise_jacobian, save_jacobian_library
from green_rtm.spectra import GHS_WAVELENGTHS
from audit_emulator import _pack, reflectance_metrics


def main() -> None:
    cfg = load_config("emit_free")
    lib = resolve_path(cfg, "rtm_library")
    tables = resolve_path(cfg, "tables")
    figs = resolve_path(cfg, "figures")
    hold_p = pd.read_parquet(lib / "learning_curve" / "holdout_params.parquet")
    hold_s = np.load(lib / "learning_curve" / "holdout_spectra.npy")
    frames = []
    for n, path in (
        (4000, lib / "forward_emulator.pt"),
        (50000, lib / "learning_curve" / "forward_emulator_50000.pt"),
    ):
        emu = load_emulator(path)
        emu.cpu().eval()
        with torch.no_grad():
            pred = emu(torch.from_numpy(_pack(hold_p))).numpy()
        rec = reflectance_metrics(hold_s, pred)
        band = pd.DataFrame(
            {
                "wavelength_nm": GHS_WAVELENGTHS[: rec["band_rmse"].size],
                "rmse": rec["band_rmse"],
                "bias": rec["band_bias"],
                "n_train": n,
            }
        )
        band.to_csv(tables / f"emulator_band_error_{n}.csv", index=False)
        frames.append(band)
        print(f"N={n} rmse={rec['rmse']:.4f}", flush=True)
    both = pd.concat(frames, ignore_index=True)
    both.to_csv(tables / "emulator_band_error_compare.csv", index=False)
    fig_emulator_band_compare(both, figs / "figS_emulator_band_error.png")

    emu50 = load_emulator(lib / "learning_curve" / "forward_emulator_50000.pt")
    params = pd.read_parquet(lib / "rtm_params.parquet")
    jac = bandwise_jacobian(emu50, params[list(PARAM_ORDER)].to_numpy(np.float32), max_n=128)
    save_jacobian_library(jac, lib / "jacobian_library_50000.npz")
    fig_jacobian(jac, figs / "fig7_jacobian.png")
    print("wrote Fig. 7 and Fig. S1", flush=True)


if __name__ == "__main__":
    main()
