#!/usr/bin/env python
"""End-to-end smoke test on synthetic spectra (no HF / GEE required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from green_rtm.config import load_config, set_seed
from green_rtm.data.datasets import VariableBandDataset, collate_variable
from green_rtm.data.metadata import infer_source_fields
from green_rtm.data.splits import assert_no_emit_leak, emit_free_split
from green_rtm.models import build_model
from green_rtm.spectra import GHS_WAVELENGTHS, N_GHS_BANDS
from green_rtm.train.losses import combined_loss


def _synthetic_frame(n: int = 32) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {str(int(w)): rng.uniform(0.02, 0.55, n).astype(np.float32) for w in GHS_WAVELENGTHS}
    df = pd.DataFrame(data)
    df["sample_id"] = [f"s{i}" for i in range(n)]
    df["dataset"] = ["AVIRIS-NG_forest"] * (n // 2) + ["ASD_crop"] * (n - n // 2)
    df["LAI"] = rng.uniform(0.2, 6.0, n)
    df["Cab"] = rng.uniform(10, 70, n)
    df["Cw"] = rng.uniform(0.004, 0.04, n)
    df["Cm"] = rng.uniform(0.004, 0.02, n)
    df["Cp"] = rng.uniform(0.001, 0.006, n)
    # punch missing labels
    df.loc[0:3, "LAI"] = np.nan
    return df


def main() -> None:
    cfg = load_config()
    cfg["train"]["pretrain_epochs"] = 1
    cfg["train"]["finetune_epochs"] = 1
    cfg["model"]["depth"] = 2
    cfg["model"]["dim"] = 64
    cfg["model"]["heads"] = 4
    set_seed(0)

    assert infer_source_fields("EMIT_L2A_africa")["is_emit"]
    assert infer_source_fields("AVIRIS-NG_2020")["sensor"] == "AVIRIS-NG"
    assert N_GHS_BANDS == 1721

    df = _synthetic_frame()
    from green_rtm.data.metadata import attach_source_fields

    df = attach_source_fields(df)
    rec = emit_free_split(df)
    assert_no_emit_leak(df.loc[~df["is_emit"]], "smoke")

    ds = VariableBandDataset(df, degrade=True, seed=0)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_variable)
    model = build_model(cfg)
    batch = next(iter(loader))
    out = model(batch, pretrain=True)
    loss, logs = combined_loss(out, batch, cfg)
    loss.backward()
    print("smoke ok", {k: round(v, 4) for k, v in logs.items()}, "n_bands", batch["reflectance"].shape[-1])
    print("emit_free held-out sources", rec["held_out_sources"])


if __name__ == "__main__":
    main()
