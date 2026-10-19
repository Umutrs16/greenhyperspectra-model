#!/usr/bin/env python
"""P0 reconstruction: T0/T1/T2 ladder, trait-wise PLSR, RTM pathway audit.

T0 = same Transformer, no RTM
T1 = RTM with fixed nuisance (M=1)
T2 = RTM with marginalized nuisance (M=8)
Spectral consistency is L(R, F(θ̂, η)), so gradients must return through θ̂.
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch

from green_rtm.config import load_config, resolve_path, set_seed
from green_rtm.data.splits import apply_split, load_split
from green_rtm.eval.metrics import trait_metrics
from green_rtm.models import CNN1D, PLSRBaseline, build_model
from green_rtm.rtm.emulator import load_emulator
from green_rtm.train.engine import audit_rtm_pathways, train_finetune
from green_rtm.train.losses import masked_huber
from run_real_experiments import (
    TRAITS,
    _apply_zscore,
    _arrays,
    _device,
    _fit_zscore,
    _invert_zscore,
    _loader,
    _log,
    collect_protocol_rows,
    eval_predictions,
    prepare_frames,
)


def _variant_cfg(cfg: dict, name: str) -> dict:
    out = deepcopy(cfg)
    out["model"]["use_low_rank_residual"] = False
    if name == "T0":
        out["model"]["use_rtm"] = False
        out["model"]["use_marginalization"] = False
        out["train"]["lambdas"]["rtm"] = 0.0
    elif name == "T1":
        out["model"]["use_rtm"] = True
        out["model"]["use_marginalization"] = False
        out["rtm"]["M_train"] = 1
        out["train"]["lambdas"]["rtm"] = 0.3
    elif name == "T2":
        out["model"]["use_rtm"] = True
        out["model"]["use_marginalization"] = True
        out["model"]["use_likelihood_marginal"] = False
        out["rtm"]["M_train"] = 8
        out["train"]["lambdas"]["rtm"] = 0.3
    elif name == "T2m":
        out["model"]["use_rtm"] = True
        out["model"]["use_marginalization"] = True
        out["model"]["use_likelihood_marginal"] = True
        out["rtm"]["M_train"] = 8
        out["train"]["lambdas"]["rtm"] = 0.3
        out["train"]["rtm_laplace_scale"] = 0.02
        out["train"]["rtm_nll_match_mae"] = False
    elif name == "T2mN":
        # Same Laplace likelihood as T2m, multiplied by b so λ_RTM matches mean-spectrum MAE scale.
        out["model"]["use_rtm"] = True
        out["model"]["use_marginalization"] = True
        out["model"]["use_likelihood_marginal"] = True
        out["rtm"]["M_train"] = 8
        out["train"]["lambdas"]["rtm"] = 0.3
        out["train"]["rtm_laplace_scale"] = 0.02
        out["train"]["rtm_nll_match_mae"] = True
    elif name == "T2geo":
        # Clamp sun-view angles. GreenHyperSpectra has no per-spectrum geometry, so these
        # are the training-code initial angles, not campaign metadata.
        out["model"]["use_rtm"] = True
        out["model"]["use_marginalization"] = True
        out["model"]["use_likelihood_marginal"] = False
        out["rtm"]["M_train"] = 8
        out["train"]["lambdas"]["rtm"] = 0.3
        out["model"]["fix_geometry"] = {"SZA": 30.0, "VZA": 8.0, "RAA": 90.0}
    elif name == "T2M16":
        out["model"]["use_rtm"] = True
        out["model"]["use_marginalization"] = True
        out["model"]["use_likelihood_marginal"] = False
        out["rtm"]["M_train"] = 16
        out["train"]["lambdas"]["rtm"] = 0.3
    else:
        raise ValueError(name)
    return out


def run_plsr(train_df, test_df, out_dir: Path, tag: str) -> pd.DataFrame:
    xtr, ytr, _ = _arrays(train_df)
    xte, _, _ = _arrays(test_df)
    pred = PLSRBaseline(n_components=20).fit(xtr, ytr).predict(xte)
    return eval_predictions(test_df, pred, tag, out_dir)


def run_cnn(train_df, test_df, out_dir: Path, tag: str) -> pd.DataFrame:
    device = _device()
    stats = _fit_zscore(train_df)
    xtr, ytr, mtr = _arrays(_apply_zscore(train_df, stats))
    xte, _, _ = _arrays(test_df)
    model = CNN1D(n_bands=xtr.shape[1], n_traits=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.from_numpy(xtr)
    yt = torch.from_numpy(np.nan_to_num(ytr, nan=0.0))
    mt = torch.from_numpy(mtr)
    model.train()
    for _ in range(25):
        perm = torch.randperm(len(xt))
        for s in range(0, len(xt), 64):
            idx = perm[s : s + 64]
            pred = model(xt[idx].to(device))
            loss = masked_huber(pred, yt[idx].to(device), mt[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(xte), 256):
            preds.append(model(torch.from_numpy(xte[s : s + 256]).to(device)).cpu().numpy())
    return eval_predictions(test_df, _invert_zscore(np.concatenate(preds, 0), stats), tag, out_dir)


def run_transformer(cfg, train_df, test_df, out_dir, tag, emu_path: Path, epochs: int) -> tuple[pd.DataFrame, object]:
    set_seed(int(cfg["project"]["seed"]))
    stats = _fit_zscore(train_df)
    train_z = _apply_zscore(train_df, stats)
    emu = None
    if cfg["model"].get("use_rtm", True):
        emu = load_emulator(emu_path)
        emu.eval()
        for p in emu.parameters():
            p.requires_grad = False
    model = build_model(cfg, emulator=emu)
    model.set_trait_stats(stats)
    cfg = deepcopy(cfg)
    cfg["train"]["finetune_epochs"] = epochs
    cfg["train"]["num_workers"] = 0
    loader = _loader(train_z, cfg, shuffle=True, degrade=True)
    train_finetune(model, loader, cfg, resolve_path(cfg, "checkpoints"), tag=tag)
    device = _device()
    model.to(device).eval()
    preds = []
    with torch.no_grad():
        for batch in _loader(test_df, cfg, shuffle=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, pretrain=False)
            preds.append(out["traits"].cpu().numpy())
    tab = eval_predictions(test_df, _invert_zscore(np.concatenate(preds, 0), stats), tag, out_dir)
    return tab, model


def label_shift_table(train_df, test_df, fold: str) -> pd.DataFrame:
    rows = []
    for t in TRAITS:
        a = pd.to_numeric(train_df[t], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(test_df[t], errors="coerce").dropna().to_numpy()
        if len(a) < 5 or len(b) < 5:
            continue
        # 1-D Wasserstein via sorted interpolation
        qs = np.linspace(0.05, 0.95, 19)
        w = float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))
        overlap = float(min(a.max(), b.max()) - max(a.min(), b.min()))
        span = float(max(a.max(), b.max()) - min(a.min(), b.min()) + 1e-9)
        extra = float(np.mean((b < a.min()) | (b > a.max())))
        rows.append(
            {
                "fold": fold,
                "trait": t,
                "wasserstein_q": w,
                "range_overlap_frac": max(overlap, 0.0) / span,
                "extrapolation_rate": extra,
                "n_train": int(len(a)),
                "n_test": int(len(b)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config("emit_free")
    cfg["train"]["batch_size"] = 32
    cfg["model"].update({"depth": 4, "dim": 128, "heads": 4, "patch_size": 16, "use_low_rank_residual": False})
    set_seed(int(cfg["project"]["seed"]))
    labeled, split_dir = prepare_frames(cfg)
    emu_path = resolve_path(cfg, "rtm_library") / "forward_emulator.pt"
    if not emu_path.exists():
        raise FileNotFoundError(emu_path)
    out = resolve_path(cfg, "tables")
    rows = []
    shifts = []
    audits = []

    def _metrics_path(tag: str) -> Path:
        return out / f"{tag.replace('/', '_')}_metrics.csv"

    def _load_done(tag: str) -> pd.DataFrame | None:
        p = _metrics_path(tag)
        if p.exists() and p.stat().st_size > 0:
            return pd.read_csv(p)
        return None

    protocols = collect_protocol_rows(split_dir)
    for proto, spath in protocols:
        split = load_split(spath)
        tr, te = apply_split(labeled, split, "train"), apply_split(labeled, split, "test")
        if len(tr) < 20 or len(te) < 8:
            continue
        stem = spath.stem
        _log(f"=== {stem} {len(tr)}/{len(te)} ===")
        shifts.append(label_shift_table(tr, te, stem))
        plsr_tag = f"PLSR/{stem}"
        done = _load_done(plsr_tag)
        rows.append(done if done is not None else run_plsr(tr, te, out, plsr_tag))
        variants = ("T0", "T1", "T2")
        if proto == "id_random":
            cnn_tag = f"CNN/{stem}"
            done = _load_done(cnn_tag)
            rows.append(done if done is not None else run_cnn(tr, te, out, cnn_tag))
        for name in variants:
            tag = f"{name}/{stem}"
            done = _load_done(tag)
            if done is not None:
                _log(f"skip {tag} (already written)")
                rows.append(done)
                continue
            vcfg = _variant_cfg(cfg, name)
            _log(f"--- {name} {stem} ---")
            tab, model = run_transformer(vcfg, tr, te, out, tag, emu_path, epochs=12)
            rows.append(tab)
            if proto == "id_random" and name in {"T1", "T2"}:
                batch = next(iter(_loader(_apply_zscore(tr, _fit_zscore(tr)), vcfg, shuffle=True)))
                rec = audit_rtm_pathways(model, batch, vcfg)
                rec["tag"] = tag
                audits.append(rec)
                _log(f"audit {name}: {rec}")
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(out / "v2_ladder_metrics.csv", index=False)

    extra = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    extra.to_csv(out / "v2_ladder_metrics.csv", index=False)
    if shifts:
        pd.concat(shifts, ignore_index=True).to_csv(out / "label_shift.csv", index=False)
    if audits:
        audit_path = out / "rtm_pathway_audit.csv"
        new_aud = pd.DataFrame(audits)
        if audit_path.exists():
            old_aud = pd.read_csv(audit_path)
            keep = ~old_aud["tag"].astype(str).isin(new_aud["tag"].astype(str))
            new_aud = pd.concat([old_aud[keep], new_aud], ignore_index=True)
        new_aud.to_csv(audit_path, index=False)
    allp = out / "all_metrics.csv"
    old = pd.read_csv(allp) if allp.exists() else pd.DataFrame()
    if len(old):
        old = old[~old["tag"].astype(str).str.contains(r"^T0/|^T1/|^T2/|^PLSR/|^CNN/", na=False)]
    pd.concat([old, extra], ignore_index=True).to_csv(allp, index=False)
    _log("v2 ladder written")


if __name__ == "__main__":
    main()
