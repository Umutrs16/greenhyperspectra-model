#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from green_rtm.config import load_config, resolve_path, set_seed
from green_rtm.data.datasets import VariableBandDataset, collate_variable
from green_rtm.data.metadata import attach_source_fields
from green_rtm.data.splits import apply_split, assert_no_emit_leak, load_split
from green_rtm.models import build_model
from green_rtm.train import train_finetune


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="*", default=["emit_free"])
    p.add_argument("--split", default="id_random.json")
    p.add_argument("--pretrain", type=Path, default=None)
    args = p.parse_args()
    cfg = load_config(*args.config)
    set_seed(int(cfg["project"]["seed"]))
    df = attach_source_fields(pd.read_parquet(resolve_path(cfg, "raw_hf") / "labeled_all.parquet"))
    split = load_split(resolve_path(cfg, "splits") / args.split)
    train_df = apply_split(df, split, "train")
    if cfg["data"].get("emit_free", True):
        train_df = train_df.loc[~train_df["is_emit"].astype(bool)].copy()
        assert_no_emit_leak(train_df, "finetune")
    ds = VariableBandDataset(train_df, traits=list(cfg["traits"]["primary"]), seed=int(cfg["project"]["seed"]))
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, collate_fn=collate_variable)
    model = build_model(cfg)
    if args.pretrain and args.pretrain.exists():
        ckpt = torch.load(args.pretrain, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
    train_finetune(model, loader, cfg, resolve_path(cfg, "checkpoints"), tag=Path(args.split).stem)


if __name__ == "__main__":
    main()
