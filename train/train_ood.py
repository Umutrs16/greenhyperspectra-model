#!/usr/bin/env python
"""Run source / platform / sensor OOD protocols from frozen split manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from green_rtm.config import load_config, resolve_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", default="cross_dataset", choices=["cross_dataset", "leave_platform_out", "leave_sensor_out", "emit_free"])
    args = p.parse_args()
    cfg = load_config("emit_free")
    split_dir = resolve_path(cfg, "splits")
    files = sorted(split_dir.glob(f"{args.protocol}*.json"))
    if not files:
        raise SystemExit(f"No split files matching {args.protocol}*.json in {split_dir}. Run scripts/build_splits.py first.")
    print(f"{len(files)} frozen manifests for {args.protocol}:")
    for f in files:
        print(" ", f.name)
    print("Train each fold with: python train/finetune_traits.py --split <file>")


if __name__ == "__main__":
    main()
