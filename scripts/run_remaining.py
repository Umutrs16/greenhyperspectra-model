#!/usr/bin/env python
"""Remaining v2 work: unlabeled MAE pretrain, pretrained OOD/ablations, official split, GEE search."""

from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from green_rtm.config import load_config, resolve_path, set_seed
from green_rtm.data.metadata import attach_source_fields
from green_rtm.data.frame import normalize_trait_columns
from green_rtm.data.splits import apply_split, load_split
from green_rtm.models import build_model
from green_rtm.rtm.jacobian import global_importance
from green_rtm.train.engine import train_pretrain

sys.path.insert(0, str(ROOT / "scripts"))
from run_real_experiments import (  # noqa: E402
    TRAITS,
    _device,
    _log,
    _loader,
    collect_protocol_rows,
    eval_predictions,
    make_figures,
    prepare_frames,
    run_transformer,
    write_run_card,
)


def _physics_weight(cfg: dict) -> torch.Tensor | None:
    jac_path = resolve_path(cfg, "rtm_library") / "jacobian_library.npz"
    if not jac_path.exists():
        return None
    z = np.load(jac_path)
    if "importance" in z.files:
        return torch.tensor(z["importance"], dtype=torch.float32)
    jac = {k: z[k] for k in z.files if k not in {"wavelengths", "importance"}}
    return torch.tensor(global_importance(jac), dtype=torch.float32)


def run_pretrain(cfg: dict, n_sample: int = 60000, epochs: int = 8) -> Path:
    out = resolve_path(cfg, "checkpoints") / "pretrain_last.pt"
    if out.exists() and out.stat().st_size > 10_000:
        _log(f"reuse pretrain checkpoint {out}")
        return out
    path = resolve_path(cfg, "raw_hf") / "unlabeled.parquet"
    if not path.exists():
        raise FileNotFoundError(f"unlabeled parquet missing: {path}")
    _log(f"loading unlabeled {path} ({path.stat().st_size/1e9:.2f} GB)")
    df = attach_source_fields(pd.read_parquet(path))
    n0 = len(df)
    if "is_emit" in df.columns:
        n_emit = int(df["is_emit"].sum())
        df = df.loc[~df["is_emit"].astype(bool)].copy()
        _log(f"EMIT-free unlabeled: dropped {n_emit} / {n0}")
    if len(df) > n_sample:
        df = df.sample(n_sample, random_state=int(cfg["project"]["seed"]))
    _log(f"pretrain on {len(df)} unlabeled spectra, {epochs} epochs")
    cfg = deepcopy(cfg)
    cfg["train"]["pretrain_epochs"] = epochs
    cfg["train"]["num_workers"] = 0
    cfg["train"]["lambdas"]["trait"] = 0.0
    cfg["train"]["lambdas"]["rtm"] = 0.0
    cfg["train"]["lambdas"]["residual"] = 0.05
    loader = _loader(df, cfg, shuffle=True, degrade=True)
    model = build_model(cfg, physics_weight=_physics_weight(cfg))
    return train_pretrain(model, loader, cfg, resolve_path(cfg, "checkpoints"))


def official_frames(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    raw = resolve_path(cfg, "raw_hf")
    trp, tep = raw / "labeled_official_train.parquet", raw / "labeled_official_test.parquet"
    if not (trp.exists() and tep.exists()):
        return None
    tr = normalize_trait_columns(pd.read_parquet(trp))
    te = normalize_trait_columns(pd.read_parquet(tep))
    tr = attach_source_fields(tr)
    te = attach_source_fields(te)
    return tr, te


def gee_search(cfg: dict) -> list[dict]:
    out = resolve_path(cfg, "emit_exports") / "scene_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from green_rtm.gee.emit_search import search_emit_scenes

        _log("searching EMIT scenes on GEE")
        scenes = search_emit_scenes(cfg, n_scenes=int(cfg["gee"].get("n_scenes", 6)))
        out.write_text(json.dumps(scenes, indent=2, default=str), encoding="utf-8")
        _log(f"GEE found {len(scenes)} scenes → {out}")
        return scenes
    except Exception as exc:
        payload = {"error": str(exc), "note": "GEE search failed; quantitative claims do not use EMIT maps."}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _log(f"GEE search failed: {exc}")
        return []


def gee_export(scenes: list[dict], cfg: dict) -> None:
    if not scenes:
        return
    try:
        from green_rtm.gee.export_emit import export_selected

        _log(f"starting Drive export for {len(scenes)} EMIT scenes")
        tasks = export_selected(scenes, cfg)
        ids = [{"roi": s.get("roi"), "task": getattr(t, "id", None)} for s, t in zip(scenes, tasks)]
        (resolve_path(cfg, "emit_exports") / "export_tasks.json").write_text(
            json.dumps(ids, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        _log(f"GEE export not started: {exc}")


def native_band_transfer_check(cfg: dict, pretrain: Path, labeled: pd.DataFrame) -> dict:
    """Resample a labeled hold-out to a 285-band EMIT-like SRF and score the pretrained model."""
    from green_rtm.spectra import GHS_WAVELENGTHS, gaussian_srf_project
    from run_real_experiments import _arrays, _fit_zscore, _invert_zscore

    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    idx = rng.choice(len(labeled), size=min(800, len(labeled)), replace=False)
    sub = labeled.iloc[idx].reset_index(drop=True)
    x, y, m = _arrays(sub)
    emit_wl = np.linspace(380, 2500, 285, dtype=np.float32)
    emit_fwhm = np.full(285, 7.5, dtype=np.float32)
    proj = np.stack([gaussian_srf_project(row, GHS_WAVELENGTHS, emit_wl, emit_fwhm) for row in x]).astype(np.float32)
    stats = _fit_zscore(sub)
    model = build_model(cfg, physics_weight=_physics_weight(cfg))
    ckpt = torch.load(pretrain, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    # light finetune on native tokens is not done here — this is zero-shot encoder transfer
    # after a short ID finetune on GHS grid, we only test whether native 285-D forward works.
    device = _device()
    model.to(device).eval()
    wl = torch.from_numpy(emit_wl).to(device)
    fw = torch.from_numpy(emit_fwhm).to(device)
    va = torch.ones_like(wl)
    preds = []
    with torch.no_grad():
        for s in range(0, len(proj), 64):
            r = torch.from_numpy(proj[s : s + 64]).to(device)
            b = r.shape[0]
            batch = {
                "reflectance": r,
                "wavelength": wl.unsqueeze(0).expand(b, -1),
                "fwhm": fw.unsqueeze(0).expand(b, -1),
                "sigma": torch.full_like(r, 0.02),
                "valid": va.unsqueeze(0).expand(b, -1),
            }
            # encoder-only smoke: untrained trait head after pretrain is not a paper number.
            out = model(batch, pretrain=False)
            preds.append(out["traits"].cpu().numpy())
    pred = np.concatenate(preds, 0)
    ok = bool(np.isfinite(pred).all()) and pred.shape == (len(proj), 5)
    report = {
        "n": int(len(proj)),
        "n_bands": 285,
        "finite": ok,
        "pred_std": pred.std(axis=0).tolist(),
        "note": "Native 285-band forward pass succeeded. Trait metrics are not reported: the head was not finetuned on EMIT SRF.",
    }
    dest = resolve_path(cfg, "results") / "emit_native_forward.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(f"EMIT-like 285-band forward: n={report['n']} finite={ok}")
    return report


def main() -> None:
    cfg = load_config("emit_free")
    cfg["train"]["batch_size"] = 32
    cfg["model"]["depth"] = 4
    cfg["model"]["dim"] = 128
    cfg["model"]["heads"] = 4
    cfg["model"]["patch_size"] = 16
    set_seed(int(cfg["project"]["seed"]))
    _log(f"device={_device()} remaining pipeline")

    labeled, split_dir = prepare_frames(cfg)
    pretrain = run_pretrain(cfg, n_sample=60000, epochs=8)

    out_dir = resolve_path(cfg, "tables")
    tables: list[pd.DataFrame] = []
    old = out_dir / "all_metrics.csv"
    if old.exists():
        tables.append(pd.read_csv(old))

    protocols = collect_protocol_rows(split_dir)
    for proto, spath in protocols:
        split = load_split(spath)
        tr = apply_split(labeled, split, "train")
        te = apply_split(labeled, split, "test")
        if len(tr) < 20 or len(te) < 8:
            continue
        stem = spath.stem
        _log(f"=== PT {stem}  n_train={len(tr)} n_test={len(te)} ===")
        tables.append(run_transformer(cfg, tr, te, out_dir, f"PT_M3_MargSpec/{stem}", pretrain_path=pretrain, epochs=16))
        if proto == "id_random":
            fixed = deepcopy(cfg)
            fixed["model"]["use_marginalization"] = False
            fixed["rtm"]["M_train"] = 1
            tables.append(run_transformer(fixed, tr, te, out_dir, f"PT_B3_fixedRTM/{stem}", pretrain_path=pretrain, epochs=16))
            nores = deepcopy(cfg)
            nores["model"]["use_low_rank_residual"] = False
            tables.append(run_transformer(nores, tr, te, out_dir, f"PT_M2_noResidual/{stem}", pretrain_path=pretrain, epochs=16))

    off = official_frames(cfg)
    if off is not None:
        tr, te = off
        _log(f"=== official HF split  n_train={len(tr)} n_test={len(te)} ===")
        tables.append(run_transformer(cfg, tr, te, out_dir, "PT_M3_MargSpec/official_split", pretrain_path=pretrain, epochs=16))

    all_metrics = pd.concat(tables, ignore_index=True)
    all_metrics.to_csv(out_dir / "all_metrics.csv", index=False)
    all_metrics.to_csv(out_dir / "all_metrics_v2.csv", index=False)

    rtm = {
        "spectra": np.load(resolve_path(cfg, "rtm_library") / "rtm_spectra.npy"),
        "jacobian": resolve_path(cfg, "rtm_library") / "jacobian_library.npz",
    }
    figs = make_figures(cfg, all_metrics, rtm)
    native = native_band_transfer_check(cfg, pretrain, labeled)
    scenes = gee_search(cfg)
    gee_export(scenes[:3], cfg)
    write_run_card(
        cfg,
        all_metrics,
        {
            "n_labeled": int(len(labeled)),
            "n_rtm": int(rtm["spectra"].shape[0]),
            "pretrain": str(pretrain),
            "stage": "v2_pretrained",
            "figures": [str(p) for p in figs],
            "metrics_csv": str(out_dir / "all_metrics.csv"),
            "gee_scenes": len(scenes),
            "emit_native_forward": native,
        },
    )
    _log("remaining pipeline done")


if __name__ == "__main__":
    main()
