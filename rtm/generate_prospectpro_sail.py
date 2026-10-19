#!/usr/bin/env python
"""Generate the PROSPECT-PRO + 4SAIL physics library."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from green_rtm.config import load_config, resolve_path, set_seed
from green_rtm.rtm.generate_library import draw_parameters, simulate_library, sobol_sanity, write_library


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None, help="Override library size (default from config)")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--sanity-only", action="store_true")
    args = p.parse_args()
    cfg = load_config()
    set_seed(int(cfg["project"]["seed"]))
    n = args.n or (int(cfg["rtm"]["n_sanity"]) if args.sanity_only else int(cfg["rtm"]["n_samples"]))
    params = draw_parameters(n, cfg, seed=int(cfg["project"]["seed"]))
    spectra = simulate_library(params, n_jobs=args.jobs)
    out = resolve_path(cfg, "rtm_library")
    paths = write_library(params, spectra, out)
    qa = sobol_sanity(params, spectra)
    qa.to_csv(out / "sobol_sanity.csv", index=False)
    print(qa.to_string(index=False))
    print("wrote", paths)


if __name__ == "__main__":
    main()
