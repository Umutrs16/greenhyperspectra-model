#!/usr/bin/env python
"""Second-review experiments. Does not overwrite the reported ladder.

T1bound/T2bound: logistic nuisance draws inside the emulator library, 48 epochs.
T0cp625/T2cp625: the same 12-epoch protocol after multiplying the Cp column by 6.25.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import torch

from green_rtm.config import load_config, resolve_path, set_seed
from green_rtm.data.splits import apply_split, load_split
from green_rtm.models.rtm_marginalizer import NUISANCE
from run_real_experiments import prepare_frames
from run_v2_ladder import _variant_cfg, run_transformer

FOLDS = tuple(f"cross_dataset_fold{i:02d}" for i in range(10))


def _base_cfg() -> dict:
    cfg = load_config("emit_free")
    cfg["train"]["batch_size"] = 32
    cfg["model"].update({"depth": 4, "dim": 128, "heads": 4, "patch_size": 16, "use_low_rank_residual": False})
    cfg["project"]["seed"] = 42
    return cfg


def _check_bounds(cfg: dict) -> None:
    from green_rtm.models import build_model

    model = build_model(cfg, emulator=None)
    eta = model.marginalizer.sample_nuisance(256, 64, fixed=False)
    lo = model.marginalizer.nuisance_lo
    hi = model.marginalizer.nuisance_hi
    if not torch.all(eta >= lo - 1e-5) or not torch.all(eta <= hi + 1e-5):
        raise RuntimeError("bounded nuisance draw left the library")
    outside = 0
    for j, name in enumerate(NUISANCE):
        span = float(hi[j] - lo[j])
        print(
            f"bound {name}: mean {float(eta[:, :, j].mean()):.4f} "
            f"sd {float(eta[:, :, j].std()):.4f} span {span:.4f}",
            flush=True,
        )
    print("bounded draws stayed inside the library", flush=True)


def _run(cfg: dict, labeled: pd.DataFrame, split_dir: Path, emu: Path, jobs: list[tuple[str, str, int]]) -> None:
    out = resolve_path(cfg, "tables")
    summary = out / "round3_metrics.csv"
    rows = [pd.read_csv(summary)] if summary.exists() else []

    def done(tag: str):
        path = out / f"{tag.replace('/', '_')}_metrics.csv"
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
        return None

    for stem in FOLDS:
        spath = split_dir / f"{stem}.json"
        split = load_split(spath)
        tr, te = apply_split(labeled, split, "train"), apply_split(labeled, split, "test")
        if len(tr) < 20 or len(te) < 8:
            print(f"skip {stem}: {len(tr)}/{len(te)}", flush=True)
            continue
        for name, prefix, epochs in jobs:
            tag = f"{prefix}/{stem}"
            have = done(tag)
            if have is not None:
                print(f"skip {tag}", flush=True)
                rows.append(have)
                continue
            started = time.time()
            print(f"--- {tag} epochs={epochs} n={len(tr)}/{len(te)} ---", flush=True)
            vcfg = _variant_cfg(cfg, name)
            if prefix.endswith("bound"):
                vcfg["model"]["bound_nuisance"] = True
            vcfg["project"]["seed"] = 42
            tab, _ = run_transformer(vcfg, tr, te, out, tag, emu, epochs=epochs)
            rows.append(tab)
            print(f"done {tag} in {time.time() - started:.0f}s", flush=True)
            pd.concat(rows, ignore_index=True).drop_duplicates(subset=["tag", "trait"], keep="last").to_csv(summary, index=False)
    if rows:
        extra = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["tag", "trait"], keep="last")
        extra.to_csv(summary, index=False)
        print(extra.groupby(extra["tag"].str.split("/").str[0])["r2"].median(), flush=True)


def main() -> None:
    cfg = _base_cfg()
    set_seed(42)
    labeled, split_dir = prepare_frames(cfg)
    emu = resolve_path(cfg, "rtm_library") / "learning_curve" / "forward_emulator_50000.pt"
    if not emu.exists():
        raise FileNotFoundError(emu)
    probe = _variant_cfg(cfg, "T2")
    probe["model"]["bound_nuisance"] = True
    _check_bounds(probe)
    print("=== bounded 48-epoch ladder ===", flush=True)
    _run(cfg, labeled, split_dir, emu, [("T1", "T1bound", 48), ("T2", "T2bound", 48)])
    scaled = labeled.copy()
    scaled["Cp"] = pd.to_numeric(scaled["Cp"], errors="coerce") * 6.25
    print("=== Cp x 6.25, 12-epoch T0/T2 ===", flush=True)
    _run(cfg, scaled, split_dir, emu, [("T0", "T0cp625", 12), ("T2", "T2cp625", 12)])
    print("round3 complete", flush=True)


if __name__ == "__main__":
    main()
