#!/usr/bin/env python
"""Emulator learning curve: N_RTM in {4k, 10k, 25k, 50k} on a fixed hold-out.

The reported 4,000-sample library is the prefix. Extra states use seed=43 so they
do not overlap the seed=2026 hold-out used in the published audit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch

from audit_emulator import TRAITS, _pack, jacobian_pair, reflectance_metrics
from green_rtm.config import load_config, resolve_path
from green_rtm.eval.figures import fig_emulator_learning_curve
from green_rtm.rtm.emulator import PARAM_ORDER, load_emulator, train_emulator
from green_rtm.rtm.generate_library import draw_parameters, simulate_library


SIZES = (4000, 10000, 25000, 50000)
HOLD_SEED = 2026
HOLD_N = 1000
JAC_N = 150
EXTRA_SEED = 43
CHUNK = 4000


def _simulate(params: pd.DataFrame) -> np.ndarray:
    return simulate_library(params, n_jobs=1)


def _cache_holdout(cfg: dict, cache: Path) -> tuple[pd.DataFrame, np.ndarray]:
    p_path = cache / "holdout_params.parquet"
    s_path = cache / "holdout_spectra.npy"
    if p_path.exists() and s_path.exists() and np.load(s_path).shape[0] == HOLD_N:
        print("reuse hold-out cache", flush=True)
        return pd.read_parquet(p_path), np.load(s_path)
    print(f"simulating hold-out n={HOLD_N} seed={HOLD_SEED}", flush=True)
    params = draw_parameters(HOLD_N, cfg, seed=HOLD_SEED)
    spectra = _simulate(params)
    cache.mkdir(parents=True, exist_ok=True)
    params.to_parquet(p_path, index=False)
    np.save(s_path, spectra)
    return params, spectra


def _extend_library(cfg: dict, lib: Path, cache: Path, n_target: int) -> tuple[pd.DataFrame, np.ndarray]:
    base_p = pd.read_parquet(lib / "rtm_params.parquet")
    base_s = np.load(lib / "rtm_spectra.npy")
    if len(base_p) < 4000:
        raise RuntimeError(f"reported library has {len(base_p)} rows; expected >= 4000")
    base_p = base_p.iloc[:4000].reset_index(drop=True)
    base_s = base_s[:4000]
    extra_n = n_target - 4000
    extra_p_path = cache / "extra_params.parquet"
    extra_s_path = cache / "extra_spectra.npy"
    if extra_p_path.exists() and extra_s_path.exists():
        extra_p = pd.read_parquet(extra_p_path)
        extra_s = np.load(extra_s_path)
    else:
        extra_p = pd.DataFrame()
        extra_s = np.zeros((0, base_s.shape[1]), dtype=np.float32)
    if len(extra_p) < extra_n:
        need = extra_n - len(extra_p)
        print(f"simulating extra library rows={need} (have {len(extra_p)}/{extra_n})", flush=True)
        drawn = draw_parameters(extra_n, cfg, seed=EXTRA_SEED).iloc[len(extra_p) : extra_n]
        chunks = []
        for start in range(0, len(drawn), CHUNK):
            part = drawn.iloc[start : start + CHUNK]
            print(f"  extra {len(extra_p) + start + len(part)}/{extra_n}", flush=True)
            chunks.append(_simulate(part))
        extra_p = pd.concat([extra_p, drawn.reset_index(drop=True)], ignore_index=True)
        extra_s = np.concatenate([extra_s, *chunks], axis=0).astype(np.float32)
        cache.mkdir(parents=True, exist_ok=True)
        extra_p.to_parquet(extra_p_path, index=False)
        np.save(extra_s_path, extra_s)
    params = pd.concat([base_p, extra_p.iloc[:extra_n]], ignore_index=True)
    spectra = np.concatenate([base_s, extra_s[:extra_n]], axis=0)
    return params, spectra


def _eval_emu(emu, hold_p: pd.DataFrame, hold_s: np.ndarray, jac_idx: np.ndarray) -> tuple[dict, pd.DataFrame]:
    emu.cpu().eval()
    with torch.no_grad():
        pred = emu(torch.from_numpy(_pack(hold_p))).numpy()
    rec = reflectance_metrics(hold_s, pred)
    rec = {k: v for k, v in rec.items() if k not in {"band_rmse", "band_bias"}}
    rows = []
    for i in jac_idx:
        j_rtm, j_emu = jacobian_pair(emu, hold_p.iloc[int(i)].to_dict())
        for t in TRAITS:
            a, b = j_rtm[t], j_emu[t]
            if (not np.isfinite(a).all()) or (not np.isfinite(b).all()) or np.linalg.norm(a) < 1e-8:
                corr, rel = np.nan, np.nan
            elif np.std(a) < 1e-12 or np.std(b) < 1e-12:
                corr = np.nan
                rel = float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a**2)) + 1e-12))
            else:
                corr = float(np.corrcoef(a, b)[0, 1])
                rel = float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a**2)) + 1e-12))
            rows.append({"trait": t, "corr": corr, "rel_rmse": rel})
    return rec, pd.DataFrame(rows)


def main() -> None:
    cfg = load_config("emit_free")
    lib = resolve_path(cfg, "rtm_library")
    tables = resolve_path(cfg, "tables")
    figs = resolve_path(cfg, "figures")
    cache = lib / "learning_curve"
    cache.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    hold_p, hold_s = _cache_holdout(cfg, cache)
    rng = np.random.default_rng(7)
    jac_idx = rng.choice(len(hold_p), size=JAC_N, replace=False)
    full_p, full_s = _extend_library(cfg, lib, cache, max(SIZES))

    hold_rows, jac_rows = [], []
    for n in SIZES:
        tag = f"N{n}"
        emu_path = cache / f"forward_emulator_{n}.pt"
        if n == 4000 and (lib / "forward_emulator.pt").exists() and not emu_path.exists():
            print("using reported 4k emulator", flush=True)
            emu = load_emulator(lib / "forward_emulator.pt")
        elif emu_path.exists():
            print(f"reuse emulator n={n}", flush=True)
            emu = load_emulator(emu_path)
        else:
            print(f"training emulator n={n}", flush=True)
            emu = train_emulator(
                full_p.iloc[:n][list(PARAM_ORDER)].to_numpy(np.float32),
                full_s[:n],
                hidden=list(cfg["rtm"]["emulator_hidden"]),
                epochs=25,
                batch_size=1024,
                out_path=emu_path,
            )
        rec, jac = _eval_emu(emu, hold_p, hold_s, jac_idx)
        rec["n_train"] = n
        rec["tag"] = tag
        hold_rows.append(rec)
        summ = jac.groupby("trait").agg(corr_median=("corr", "median"), corr_mean=("corr", "mean"), rel_rmse_median=("rel_rmse", "median")).reset_index()
        summ["n_train"] = n
        jac_rows.append(summ)
        print(f"n={n} RMSE={rec['rmse']:.4f} SAM={rec['sam_mean_deg']:.3f} Cab={float(summ.loc[summ.trait=='Cab','corr_median'].iloc[0]):.3f} Cp={float(summ.loc[summ.trait=='Cp','corr_median'].iloc[0]):.3f}", flush=True)

    hold_df = pd.DataFrame(hold_rows)
    jac_df = pd.concat(jac_rows, ignore_index=True)
    hold_df.to_csv(tables / "emulator_learning_curve_holdout.csv", index=False)
    jac_df.to_csv(tables / "emulator_learning_curve_jacobian.csv", index=False)
    fig_emulator_learning_curve(hold_df, jac_df, figs / "figS_emulator_learning_curve.png")
    print(hold_df.to_string(index=False), flush=True)
    print(jac_df.to_string(index=False), flush=True)
    print("wrote emulator learning curve", flush=True)


if __name__ == "__main__":
    main()
