#!/usr/bin/env python
"""λ_RTM screening on three source-blocked folds. T0 (λ=0) and λ=0.3 already exist."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from copy import deepcopy

import pandas as pd

from green_rtm.config import load_config, resolve_path, set_seed
from green_rtm.data.splits import apply_split, load_split
from green_rtm.train.engine import audit_rtm_pathways
from run_real_experiments import _apply_zscore, _fit_zscore, _loader, _log, prepare_frames
from run_v2_ladder import _variant_cfg, run_transformer

FOLDS = ("cross_dataset_fold01", "cross_dataset_fold03", "cross_dataset_fold07")
LAMBDAS = (0.1, 1.0, 3.0)
VARIANTS = ("T1", "T2")


def main() -> None:
    cfg = load_config("emit_free")
    cfg["train"]["batch_size"] = 32
    cfg["model"].update({"depth": 4, "dim": 128, "heads": 4, "patch_size": 16, "use_low_rank_residual": False})
    set_seed(int(cfg["project"]["seed"]))
    labeled, split_dir = prepare_frames(cfg)
    emu_path = resolve_path(cfg, "rtm_library") / "forward_emulator.pt"
    out = resolve_path(cfg, "tables")
    rows, audits = [], []
    for fold in FOLDS:
        split = load_split(split_dir / f"{fold}.json")
        tr, te = apply_split(labeled, split, "train"), apply_split(labeled, split, "test")
        for name in VARIANTS:
            for lam in LAMBDAS:
                tag = f"{name}_L{lam:g}/{fold}"
                mp = out / f"{tag.replace('/', '_')}_metrics.csv"
                if mp.exists() and mp.stat().st_size > 0:
                    _log(f"skip {tag}")
                    rows.append(pd.read_csv(mp))
                    continue
                vcfg = _variant_cfg(cfg, name)
                vcfg["train"]["lambdas"]["rtm"] = float(lam)
                _log(f"--- {tag} ---")
                tab, model = run_transformer(vcfg, tr, te, out, tag, emu_path, epochs=12)
                rows.append(tab)
                batch = next(iter(_loader(_apply_zscore(tr, _fit_zscore(tr)), vcfg, shuffle=True)))
                rec = audit_rtm_pathways(model, batch, vcfg)
                rec["tag"] = tag
                rec["lambda_rtm"] = float(lam)
                audits.append(rec)
                pd.concat(rows, ignore_index=True).to_csv(out / "v4_lambda_screen.csv", index=False)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out / "v4_lambda_screen.csv", index=False)
    if audits:
        pd.DataFrame(audits).to_csv(out / "v4_lambda_audit.csv", index=False)
    _log("lambda screen written")


if __name__ == "__main__":
    main()
