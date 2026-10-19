#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from green_rtm.config import load_config, resolve_path
from green_rtm.rtm.emulator import PARAM_ORDER, train_emulator


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()
    cfg = load_config()
    lib = resolve_path(cfg, "rtm_library")
    params = pd.read_parquet(lib / "rtm_params.parquet")[PARAM_ORDER].to_numpy(np.float32)
    spectra = np.load(lib / "rtm_spectra.npy")
    model = train_emulator(
        params,
        spectra,
        hidden=list(cfg["rtm"]["emulator_hidden"]),
        epochs=args.epochs or int(cfg["rtm"]["emulator_epochs"]),
        batch_size=int(cfg["rtm"]["emulator_batch"]),
        out_path=lib / "forward_emulator.pt",
    )
    print(model)


if __name__ == "__main__":
    main()
