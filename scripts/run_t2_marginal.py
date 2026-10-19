#!/usr/bin/env python
"""T2m: Laplace log-sum-exp nuisance likelihood on the reported 4k emulator."""

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
from run_real_experiments import prepare_frames
from run_v2_ladder import _variant_cfg, run_transformer

PRIORITY = (
    "cross_dataset_fold01",
    "cross_dataset_fold03",
    "cross_dataset_fold07",
    "cross_dataset_fold00",
    "cross_dataset_fold02",
    "cross_dataset_fold04",
    "cross_dataset_fold05",
    "cross_dataset_fold06",
    "cross_dataset_fold08",
    "cross_dataset_fold09",
)


def main() -> None:
    cfg = load_config("emit_free")
    cfg["train"]["batch_size"] = 32
    cfg["model"].update({"depth": 4, "dim": 128, "heads": 4, "patch_size": 16, "use_low_rank_residual": False})
    set_seed(int(cfg["project"]["seed"]))
    labeled, split_dir = prepare_frames(cfg)
    emu_path = resolve_path(cfg, "rtm_library") / "forward_emulator.pt"
    out = resolve_path(cfg, "tables")
    summary = out / "v5_t2m_metrics.csv"
    rows = [pd.read_csv(summary)] if summary.exists() else []

    def done(tag: str):
        p = out / f"{tag.replace('/', '_')}_metrics.csv"
        return pd.read_csv(p) if p.exists() and p.stat().st_size > 0 else None

    for stem in PRIORITY:
        spath = split_dir / f"{stem}.json"
        if not spath.exists():
            continue
        split = load_split(spath)
        tr, te = apply_split(labeled, split, "train"), apply_split(labeled, split, "test")
        tag = f"T2m/{stem}"
        have = done(tag)
        if have is not None:
            print(f"skip {tag}", flush=True)
            rows.append(have)
            continue
        print(f"=== {tag} {len(tr)}/{len(te)} ===", flush=True)
        tab, _ = run_transformer(_variant_cfg(cfg, "T2m"), tr, te, out, tag, emu_path, epochs=12)
        rows.append(tab)
        pd.concat(rows, ignore_index=True).drop_duplicates(subset=["tag", "trait"], keep="last").to_csv(summary, index=False)
    extra = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["tag", "trait"], keep="last")
    extra.to_csv(summary, index=False)
    print(extra.groupby("trait")["r2"].median() if not extra.empty else "empty", flush=True)


if __name__ == "__main__":
    main()
