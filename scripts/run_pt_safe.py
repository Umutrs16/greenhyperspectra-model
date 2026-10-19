#!/usr/bin/env python
"""Patch-MAE pretrain (recon-only) + ID/OOD finetune. Abort if weights are non-finite."""

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
import torch

from green_rtm.config import load_config, resolve_path, set_seed
from green_rtm.data.metadata import attach_source_fields
from green_rtm.data.splits import apply_split, load_split
from green_rtm.models import build_model
from green_rtm.train.engine import train_pretrain
from run_real_experiments import _log, _loader, collect_protocol_rows, prepare_frames, run_transformer


def main() -> None:
    cfg = load_config("emit_free")
    cfg["train"]["batch_size"] = 32
    cfg["model"].update({"depth": 4, "dim": 128, "heads": 4, "patch_size": 16, "mask_ratio": 0.45})
    set_seed(int(cfg["project"]["seed"]))
    labeled, split_dir = prepare_frames(cfg)

    ckpt = resolve_path(cfg, "checkpoints") / "pretrain_safe.pt"
    if ckpt.exists():
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        if any(torch.is_floating_point(v) and (not torch.isfinite(v).all()) for v in sd.values()):
            ckpt.unlink()
    if not ckpt.exists():
        path = resolve_path(cfg, "raw_hf") / "unlabeled.parquet"
        df = attach_source_fields(pd.read_parquet(path))
        if "is_emit" in df.columns:
            df = df.loc[~df["is_emit"].astype(bool)]
        df = df.sample(min(25000, len(df)), random_state=int(cfg["project"]["seed"]))
        _log(f"safe patch-MAE n={len(df)}")
        pcfg = deepcopy(cfg)
        pcfg["train"]["pretrain_epochs"] = 5
        pcfg["train"]["lr"] = 1e-4
        loader = _loader(df, pcfg, shuffle=True, degrade=False)
        model = build_model(pcfg)
        last = train_pretrain(model, loader, pcfg, resolve_path(cfg, "checkpoints"))
        last.replace(ckpt)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    if any(torch.is_floating_point(v) and (not torch.isfinite(v).all()) for v in sd.values()):
        raise RuntimeError("safe pretrain still has NaN weights")
    _log(f"safe checkpoint OK {ckpt}")

    out = resolve_path(cfg, "tables")
    rows = []
    for proto, spath in collect_protocol_rows(split_dir):
        split = load_split(spath)
        tr, te = apply_split(labeled, split, "train"), apply_split(labeled, split, "test")
        if len(tr) < 20 or len(te) < 8:
            continue
        tag = f"PTSAFE_M3/{spath.stem}"
        _log(f"=== {tag} {len(tr)}/{len(te)} ===")
        tab = run_transformer(cfg, tr, te, out, tag, pretrain_path=ckpt, epochs=12)
        if int(tab["n"].sum()) == 0:
            raise RuntimeError(f"{tag} empty metrics")
        rows.append(tab)
        if proto == "id_random":
            fixed = deepcopy(cfg)
            fixed["model"]["use_marginalization"] = False
            fixed["rtm"]["M_train"] = 1
            rows.append(run_transformer(fixed, tr, te, out, f"PTSAFE_B3/{spath.stem}", pretrain_path=ckpt, epochs=12))

    extra = pd.concat(rows, ignore_index=True)
    allp = out / "all_metrics.csv"
    old = pd.read_csv(allp) if allp.exists() else pd.DataFrame()
    if len(old):
        old = old[~old["tag"].astype(str).str.contains("PT_|official|PTSAFE", na=False)]
        old = old[old["n"].fillna(0) > 0]
    pd.concat([old, extra], ignore_index=True).to_csv(allp, index=False)
    extra.to_csv(out / "ptsafe_metrics.csv", index=False)
    _log("safe pretrained metrics written")


if __name__ == "__main__":
    main()
