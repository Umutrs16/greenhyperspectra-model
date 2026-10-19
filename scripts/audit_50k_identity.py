#!/usr/bin/env python
"""Identity-split RTM pathway audit for the reported 50k T1/T2 models."""

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
from green_rtm.train.engine import audit_rtm_pathways
from run_real_experiments import _apply_zscore, _fit_zscore, _loader, prepare_frames
from run_v2_ladder import _variant_cfg, run_transformer


def main() -> None:
    cfg = load_config("emit_free")
    cfg["train"]["batch_size"] = 32
    cfg["model"].update({"depth": 4, "dim": 128, "heads": 4, "patch_size": 16, "use_low_rank_residual": False})
    set_seed(int(cfg["project"]["seed"]))
    labeled, split_dir = prepare_frames(cfg)
    emu50 = resolve_path(cfg, "rtm_library") / "learning_curve" / "forward_emulator_50000.pt"
    out = resolve_path(cfg, "tables")
    split = load_split(split_dir / "id_random.json")
    tr, te = apply_split(labeled, split, "train"), apply_split(labeled, split, "test")
    audits = []
    for name in ("T1", "T2"):
        tag = f"{name}e50/id_random_audit"
        print(f"=== audit {name} 50k identity ===", flush=True)
        vcfg = _variant_cfg(cfg, name)
        _, model = run_transformer(vcfg, tr, te, out, tag, emu50, epochs=12)
        batch = next(iter(_loader(_apply_zscore(tr, _fit_zscore(tr)), vcfg, shuffle=True)))
        rec = audit_rtm_pathways(model, batch, vcfg)
        rec["tag"] = f"{name}/id_random"
        rec["emulator"] = "50000"
        audits.append(rec)
        print(rec, flush=True)
    pd.DataFrame(audits).to_csv(out / "rtm_pathway_audit_50k.csv", index=False)
    print("wrote 50k pathway audit", flush=True)


if __name__ == "__main__":
    main()
