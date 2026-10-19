#!/usr/bin/env python
"""Identity-split T0/T1/T2 train/hold-out loss curves for Supplementary Figure S3."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from green_rtm.config import load_config, resolve_path, set_seed
from green_rtm.data.splits import apply_split, load_split
from green_rtm.eval.figures import fig_training_curves
from green_rtm.models import build_model
from green_rtm.rtm.emulator import load_emulator
from green_rtm.train.engine import train_finetune
from run_real_experiments import _apply_zscore, _fit_zscore, _loader, prepare_frames
from run_v2_ladder import _variant_cfg


def main() -> None:
    cfg = load_config("emit_free")
    cfg["train"]["batch_size"] = 32
    cfg["model"].update({"depth": 4, "dim": 128, "heads": 4, "patch_size": 16, "use_low_rank_residual": False})
    set_seed(int(cfg["project"]["seed"]))
    labeled, split_dir = prepare_frames(cfg)
    split = load_split(split_dir / "id_random.json")
    train_df, test_df = apply_split(labeled, split, "train"), apply_split(labeled, split, "test")
    emu_path = resolve_path(cfg, "rtm_library") / "forward_emulator.pt"
    tables = resolve_path(cfg, "tables")
    figs = resolve_path(cfg, "figures")
    ckpt = resolve_path(cfg, "checkpoints") / "training_curves"
    tables.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("T0", "T1", "T2"):
        hist_path = tables / f"training_curve_{name}.csv"
        if hist_path.exists():
            print(f"reuse {hist_path.name}", flush=True)
            part = pd.read_csv(hist_path)
            part["method"] = name
            rows.append(part)
            continue
        vcfg = _variant_cfg(cfg, name)
        vcfg["train"]["finetune_epochs"] = 12
        vcfg["train"]["num_workers"] = 0
        stats = _fit_zscore(train_df)
        train_z = _apply_zscore(train_df, stats)
        test_z = _apply_zscore(test_df, stats)
        emu = None
        if vcfg["model"].get("use_rtm", True):
            emu = load_emulator(emu_path)
            emu.eval()
            for p in emu.parameters():
                p.requires_grad = False
        model = build_model(vcfg, emulator=emu)
        model.set_trait_stats(stats)
        print(f"training identity-split {name}", flush=True)
        train_finetune(
            model,
            _loader(train_z, vcfg, shuffle=True, degrade=True),
            vcfg,
            ckpt,
            tag=f"curve_{name}",
            history_path=hist_path,
            val_loader=_loader(test_z, vcfg, shuffle=False, degrade=False),
        )
        part = pd.read_csv(hist_path)
        part["method"] = name
        rows.append(part)
    hist = pd.concat(rows, ignore_index=True)
    hist.to_csv(tables / "training_curves.csv", index=False)
    fig_training_curves(hist, figs / "figS_training_curves.png")
    print(hist.to_string(index=False), flush=True)
    print("wrote training curves", flush=True)


if __name__ == "__main__":
    main()
