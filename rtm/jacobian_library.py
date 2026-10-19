#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from green_rtm.config import load_config, resolve_path
from green_rtm.rtm.emulator import PARAM_ORDER, load_emulator
from green_rtm.rtm.jacobian import bandwise_jacobian, save_jacobian_library


def main() -> None:
    cfg = load_config()
    lib = resolve_path(cfg, "rtm_library")
    model = load_emulator(lib / "forward_emulator.pt")
    params = pd.read_parquet(lib / "rtm_params.parquet")[PARAM_ORDER].to_numpy(np.float32)
    jac = bandwise_jacobian(model, params, max_n=256)
    path = save_jacobian_library(jac, lib / "jacobian_library.npz")
    print("wrote", path, "traits", list(jac))


if __name__ == "__main__":
    main()
