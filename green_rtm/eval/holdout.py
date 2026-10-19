"""Leave-sensor / leave-platform evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.splits import load_split
from .bootstrap import bootstrap_table
from .metrics import table_from_predictions


def evaluate_predictions(
    pred_df: pd.DataFrame,
    traits: list[str],
    out_dir: Path,
    tag: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    point = table_from_predictions(pred_df, traits)
    boot = bootstrap_table(pred_df, traits, n_boot=n_boot, seed=seed)
    p1 = out_dir / f"{tag}_point_metrics.csv"
    p2 = out_dir / f"{tag}_source_bootstrap.csv"
    point.to_csv(p1, index=False)
    boot.to_csv(p2, index=False)
    return {"point": p1, "bootstrap": p2}


def summarize_split_file(path: Path) -> dict:
    rec = load_split(path)
    return {
        "file": path.name,
        "protocol": rec.get("protocol"),
        "fold": rec.get("fold"),
        "n_train": len(rec.get("train_ids", [])),
        "n_test": len(rec.get("test_ids", [])),
        "held_out": rec.get("held_out_sources", rec.get("held_out_group")),
    }
