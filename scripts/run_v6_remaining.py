#!/usr/bin/env python
"""Finish leftover experiments: 50k identity T1/T2, 50k M=16, and fp-parameterized T0/T1/T2.

Does not overwrite the 4k T0–T2 ladder files.
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
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

OOD = (
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
ALL = ("id_random",) + OOD


def _done(out: Path, tag: str):
    p = out / f"{tag.replace('/', '_')}_metrics.csv"
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 0 else None


def _append(rows: list, summary: Path, tab: pd.DataFrame) -> None:
    rows.append(tab)
    pd.concat(rows, ignore_index=True).drop_duplicates(subset=["tag", "trait"], keep="last").to_csv(summary, index=False)


def main() -> None:
    cfg = load_config("emit_free")
    cfg["train"]["batch_size"] = 32
    cfg["model"].update({"depth": 4, "dim": 128, "heads": 4, "patch_size": 16, "use_low_rank_residual": False})
    set_seed(int(cfg["project"]["seed"]))
    labeled, split_dir = prepare_frames(cfg)
    emu50 = resolve_path(cfg, "rtm_library") / "learning_curve" / "forward_emulator_50000.pt"
    if not emu50.exists():
        raise FileNotFoundError(emu50)
    out = resolve_path(cfg, "tables")
    summary = out / "v6_remaining_metrics.csv"
    rows = [pd.read_csv(summary)] if summary.exists() else []

    def run_one(stem: str, variant: str, tag: str, protein: bool = False) -> None:
        have = _done(out, tag)
        if have is not None:
            print(f"skip {tag}", flush=True)
            _append(rows, summary, have)
            return
        spath = split_dir / f"{stem}.json"
        if not spath.exists():
            print(f"missing split {stem}", flush=True)
            return
        split = load_split(spath)
        tr, te = apply_split(labeled, split, "train"), apply_split(labeled, split, "test")
        if len(tr) < 20 or len(te) < 8:
            return
        vcfg = _variant_cfg(cfg, variant)
        if protein:
            vcfg = deepcopy(vcfg)
            vcfg["model"]["use_protein_fraction"] = True
        print(f"=== {tag} {len(tr)}/{len(te)} ===", flush=True)
        tab, _ = run_transformer(vcfg, tr, te, out, tag, emu50, epochs=12)
        _append(rows, summary, tab)

    for name in ("T1", "T2"):
        run_one("id_random", name, f"{name}e50/id_random")
    for stem in OOD:
        run_one(stem, "T2M16", f"T2e50M16/{stem}")
    for stem in ALL:
        run_one(stem, "T0", f"T0p/{stem}", protein=True)
        run_one(stem, "T1", f"T1p50/{stem}", protein=True)
        run_one(stem, "T2", f"T2p50/{stem}", protein=True)

    extra = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["tag", "trait"], keep="last")
    extra.to_csv(summary, index=False)
    print(extra.groupby(extra["tag"].str.split("/").str[0])["r2"].median(), flush=True)
    print("wrote", summary, flush=True)


if __name__ == "__main__":
    main()
