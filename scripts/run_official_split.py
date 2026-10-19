#!/usr/bin/env python
"""Official GreenHyperSpectra train/test split with trait-column aliases. Scratch M3 only."""

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
from green_rtm.data.metadata import attach_source_fields
from green_rtm.data.frame import normalize_trait_columns
from run_real_experiments import _log, run_transformer, TRAITS


def main() -> None:
    cfg = load_config("emit_free")
    cfg["train"]["batch_size"] = 32
    cfg["model"].update({"depth": 4, "dim": 128, "heads": 4, "patch_size": 16})
    set_seed(int(cfg["project"]["seed"]))
    raw = resolve_path(cfg, "raw_hf")
    tr = normalize_trait_columns(pd.read_parquet(raw / "labeled_official_train.parquet"))
    te = normalize_trait_columns(pd.read_parquet(raw / "labeled_official_test.parquet"))
    tr = attach_source_fields(tr)
    te = attach_source_fields(te)
    if "sample_id" not in tr.columns:
        tr = tr.reset_index(drop=True)
        tr["sample_id"] = [f"offtr_{i}" for i in range(len(tr))]
    if "sample_id" not in te.columns:
        te = te.reset_index(drop=True)
        te["sample_id"] = [f"offte_{i}" for i in range(len(te))]
    avail = {t: int(pd.to_numeric(tr[t], errors="coerce").notna().sum()) for t in TRAITS if t in tr.columns}
    _log(f"official train={len(tr)} test={len(te)} traits={avail}")
    if max(avail.values(), default=0) < 20:
        raise RuntimeError(f"official split still has no traits: {avail}")
    out = resolve_path(cfg, "tables")
    tab = run_transformer(cfg, tr, te, out, "OFF_M3/official_split", epochs=12)
    if int(tab["n"].sum()) == 0:
        raise RuntimeError("official metrics empty")
    allp = out / "all_metrics.csv"
    old = pd.read_csv(allp) if allp.exists() else pd.DataFrame()
    if len(old):
        old = old[~old["tag"].astype(str).str.contains("official", na=False)]
    pd.concat([old, tab], ignore_index=True).to_csv(allp, index=False)
    tab.to_csv(out / "official_metrics.csv", index=False)
    _log("official split metrics written")


if __name__ == "__main__":
    main()
