#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from green_rtm.config import load_config, resolve_path, set_seed
from green_rtm.data.datasets import VariableBandDataset, collate_variable
from green_rtm.data.metadata import attach_source_fields
from green_rtm.data.splits import assert_no_emit_leak, load_split
from green_rtm.models import build_model
from green_rtm.train import train_pretrain


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="*", default=["emit_free"])
    p.add_argument("--split", default="emit_free.json")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(*args.config)
    set_seed(int(cfg["project"]["seed"]))
    raw = resolve_path(cfg, "raw_hf") / "unlabeled.parquet"
    df = attach_source_fields(pd.read_parquet(raw))
    split = load_split(resolve_path(cfg, "splits") / args.split)
    if "unlabeled_keep_ids" in split:
        keep = set(split["unlabeled_keep_ids"])
        if "sample_id" in df.columns:
            df = df[df["sample_id"].astype(str).isin(keep)]
        elif "unlabeled_exclude_ids" in split:
            ex = set(split["unlabeled_exclude_ids"])
            df = df[~df.get("sample_id", pd.Series(df.index.astype(str))).astype(str).isin(ex)]
    if cfg["data"].get("emit_free", True):
        df = df.loc[~df["is_emit"].astype(bool)].copy()
        assert_no_emit_leak(df, "pretrain")
    if args.max_samples:
        df = df.sample(min(args.max_samples, len(df)), random_state=int(cfg["project"]["seed"]))
    ds = VariableBandDataset(df, degrade=bool(cfg["model"]["use_sensor_degradation"]), seed=int(cfg["project"]["seed"]))
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_variable,
    )
    model = build_model(cfg)
    train_pretrain(model, loader, cfg, resolve_path(cfg, "checkpoints"))


if __name__ == "__main__":
    main()
