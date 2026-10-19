#!/usr/bin/env python
"""Real GreenHyperSpectra experiments → tables + figures used by the manuscript.

This is the v1 compute-complete protocol: real labeled spectra, real PROSPECT-PRO/4SAIL
library, source-blocked OOD, and the paper model family. Numbers are written only
from model outputs — never invented.
"""

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
from torch.utils.data import DataLoader
from tqdm import tqdm

from green_rtm.config import load_config, resolve_path, set_seed
from green_rtm.data.datasets import VariableBandDataset, collate_variable
from green_rtm.data.metadata import attach_source_fields, build_source_metadata
from green_rtm.data.frame import align_to_ghs, extract_spectra, normalize_trait_columns
from green_rtm.data.splits import apply_split, assert_no_emit_leak, build_all_splits, load_split
from green_rtm.eval.figures import fig_framework, fig_jacobian, fig_nuisance_envelope, fig_ood_dumbbell
from green_rtm.eval.metrics import table_from_predictions, trait_metrics
from green_rtm.models import CNN1D, PLSRBaseline, build_model
from green_rtm.rtm.emulator import PARAM_ORDER, train_emulator
from green_rtm.rtm.generate_library import draw_parameters, simulate_library, sobol_sanity, write_library
from green_rtm.rtm.jacobian import bandwise_jacobian, save_jacobian_library
from green_rtm.spectra import PRIMARY_TRAITS
from green_rtm.train.engine import train_finetune, train_pretrain
from green_rtm.train.losses import inverse_frequency_weights, masked_huber

TRAITS = list(PRIMARY_TRAITS)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_labeled(cfg: dict) -> Path:
    raw = resolve_path(cfg, "raw_hf")
    path = raw / "labeled_all.parquet"
    if path.exists():
        _log(f"labeled table: {path} ({path.stat().st_size/1e6:.1f} MB)")
        return path
    raise FileNotFoundError(f"Missing {path}. Place the labeled table there before training.")


def maybe_download_unlabeled(cfg: dict) -> Path | None:
    path = resolve_path(cfg, "raw_hf") / "unlabeled.parquet"
    if path.exists():
        return path
    _log("unlabeled table is absent; steps that need it are skipped")
    return None


def prepare_frames(cfg: dict) -> tuple[pd.DataFrame, Path]:
    labeled = normalize_trait_columns(pd.read_parquet(ensure_labeled(cfg)))
    avail = {t: int(pd.to_numeric(labeled[t], errors="coerce").notna().sum()) for t in TRAITS if t in labeled.columns}
    _log(f"trait availability {avail}")
    if not avail or max(avail.values()) < 50:
        raise RuntimeError(f"Trait columns missing after normalize: {list(labeled.columns)[:20]}")
    labeled = attach_source_fields(labeled)
    if "sample_id" not in labeled.columns:
        labeled = labeled.reset_index(drop=True)
        labeled["sample_id"] = [f"lb_{i}" for i in range(len(labeled))]
    if cfg["data"].get("emit_free", True):
        n_emit = int(labeled["is_emit"].sum())
        labeled = labeled.loc[~labeled["is_emit"].astype(bool)].copy()
        _log(f"removed {n_emit} EMIT labeled rows for EMIT-free training")
        assert_no_emit_leak(labeled, "labeled train pool")
    meta = build_source_metadata(labeled, None, resolve_path(cfg, "metadata"))
    _log(f"sources={len(meta)} platforms={meta['platform'].value_counts().to_dict()}")
    split_dir = resolve_path(cfg, "splits")
    written = build_all_splits(labeled, None, split_dir, cfg)
    _log(f"wrote {len(written)} split artifacts")
    return labeled, split_dir


def build_rtm(cfg: dict, n: int = 4000) -> dict:
    lib = resolve_path(cfg, "rtm_library")
    spec_path = lib / "rtm_spectra.npy"
    emu_path = lib / "forward_emulator.pt"
    jac_path = lib / "jacobian_library.npz"
    if spec_path.exists() and np.load(spec_path).shape[0] >= n:
        _log(f"reuse RTM library {np.load(spec_path).shape}")
        params = pd.read_parquet(lib / "rtm_params.parquet")
        spectra = np.load(spec_path)
    else:
        _log(f"simulating PROSPECT-PRO/4SAIL library n={n}")
        params = draw_parameters(n, cfg, seed=int(cfg["project"]["seed"]))
        spectra = simulate_library(params, n_jobs=1)
        write_library(params, spectra, lib)
        sobol_sanity(params, spectra).to_csv(lib / "sobol_sanity.csv", index=False)
    if not emu_path.exists():
        _log("training differentiable emulator")
        train_emulator(
            params[PARAM_ORDER].to_numpy(np.float32),
            spectra,
            hidden=list(cfg["rtm"]["emulator_hidden"]),
            epochs=25,
            batch_size=1024,
            out_path=emu_path,
        )
    if not jac_path.exists():
        from green_rtm.rtm.emulator import load_emulator

        _log("computing Jacobian library")
        model = load_emulator(emu_path)
        jac = bandwise_jacobian(model, params[PARAM_ORDER].to_numpy(np.float32), max_n=128)
        save_jacobian_library(jac, jac_path)
    return {"params": params, "spectra": spectra, "emulator": emu_path, "jacobian": jac_path}


def _arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, wl = extract_spectra(df)
    r = align_to_ghs(r, wl)
    y = np.column_stack(
        [pd.to_numeric(df[t], errors="coerce").to_numpy(np.float32) if t in df.columns else np.full(len(df), np.nan, np.float32) for t in TRAITS]
    )
    return r, y, (~np.isnan(y)).astype(np.float32)


def eval_predictions(df: pd.DataFrame, pred: np.ndarray, tag: str, out_dir: Path) -> pd.DataFrame:
    rec = df.reset_index(drop=True).copy()
    for i, t in enumerate(TRAITS):
        rec[f"{t}_pred"] = pred[:, i]
    rows = []
    for t in TRAITS:
        rows.append(trait_metrics(rec[t].to_numpy(), rec[f"{t}_pred"].to_numpy(), t))
    tab = pd.DataFrame(rows)
    tab.insert(0, "tag", tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = tag.replace("/", "_")
    rec.to_csv(out_dir / f"{safe}_predictions.csv", index=False)
    tab.to_csv(out_dir / f"{safe}_metrics.csv", index=False)
    _log(f"{tag}\n{tab.to_string(index=False)}")
    return tab


def run_plsr(train_df, test_df, out_dir: Path, tag: str) -> pd.DataFrame:
    xtr, ytr, _ = _arrays(train_df)
    xte, _, _ = _arrays(test_df)
    model = PLSRBaseline(n_components=20).fit(xtr, ytr)
    return eval_predictions(test_df, np.asarray(model.predict(xte), dtype=np.float32), tag, out_dir)


def run_cnn(train_df, test_df, out_dir: Path, tag: str, seed: int = 42) -> pd.DataFrame:
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


def _loader(df: pd.DataFrame, cfg: dict, shuffle: bool, degrade: bool = False) -> DataLoader:
    ds = VariableBandDataset(
        df,
        traits=TRAITS,
        degrade=degrade and bool(cfg["model"]["use_sensor_degradation"]),
        seed=int(cfg["project"]["seed"]),
    )
    return DataLoader(ds, batch_size=min(int(cfg["train"]["batch_size"]), 48), shuffle=shuffle, collate_fn=collate_variable)


def _fit_zscore(train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    fallback = {"LAI": (2.5, 1.5), "Cab": (35.0, 15.0), "Cw": (0.015, 0.01), "Cm": (0.01, 0.008), "Cp": (0.0009, 0.0005)}
    stats = {}
    for t in TRAITS:
        x = pd.to_numeric(train_df[t], errors="coerce")
        m = float(x.mean())
        s = float(x.std(ddof=0) + 1e-6)
        if not np.isfinite(m) or not np.isfinite(s) or s <= 0:
            m, s = fallback[t]
        stats[t] = (m, s)
    return stats


def _apply_zscore(df: pd.DataFrame, stats: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for t, (m, s) in stats.items():
        if t in out.columns:
            out[t] = (pd.to_numeric(out[t], errors="coerce") - m) / s
    return out


def _invert_zscore(pred: np.ndarray, stats: dict[str, tuple[float, float]]) -> np.ndarray:
    out = pred.copy()
    for i, t in enumerate(TRAITS):
        m, s = stats[t]
        out[:, i] = out[:, i] * s + m
    return out


def run_transformer(
    cfg: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    tag: str,
    pretrain_path: Path | None = None,
    epochs: int | None = None,
) -> pd.DataFrame:
    set_seed(int(cfg["project"]["seed"]))
    stats = _fit_zscore(train_df)
    train_z = _apply_zscore(train_df, stats)
    model = build_model(cfg)
    if pretrain_path and pretrain_path.exists():
        ckpt = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
    cfg = deepcopy(cfg)
    cfg["train"]["finetune_epochs"] = epochs or 16
    cfg["train"]["num_workers"] = 0
    loader = _loader(train_z, cfg, shuffle=True, degrade=True)
    ckpt_dir = resolve_path(cfg, "checkpoints")
    ckpt_path = train_finetune(model, loader, cfg, ckpt_dir, tag=tag)
    (ckpt_path.parent / f"{ckpt_path.stem}_zscore.json").write_text(
        json.dumps({k: list(v) for k, v in stats.items()}), encoding="utf-8"
    )
    device = _device()
    model.to(device).eval()
    test_loader = _loader(test_df, cfg, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, pretrain=False)
            preds.append(out["traits"].cpu().numpy())
    return eval_predictions(test_df, _invert_zscore(np.concatenate(preds, 0), stats), tag, out_dir)


def pretrain_if_possible(cfg: dict, unlabeled_path: Path | None) -> Path | None:
    if unlabeled_path is None or not unlabeled_path.exists():
        return None
    out = resolve_path(cfg, "checkpoints") / "pretrain_last.pt"
    if out.exists():
        return out
    _log("masked pretraining on unlabeled spectra (subset)")
    df = attach_source_fields(pd.read_parquet(unlabeled_path))
    if "is_emit" in df.columns:
        df = df.loc[~df["is_emit"].astype(bool)]
    if len(df) > 20000:
        df = df.sample(20000, random_state=int(cfg["project"]["seed"]))
    cfg = deepcopy(cfg)
    cfg["train"]["pretrain_epochs"] = 6
    cfg["train"]["num_workers"] = 0
    loader = _loader(df, cfg, shuffle=True, degrade=True)
    model = build_model(cfg)
    return train_pretrain(model, loader, cfg, resolve_path(cfg, "checkpoints"))


def collect_protocol_rows(split_dir: Path) -> list[tuple[str, Path]]:
    wanted = []
    idp = split_dir / "id_random.json"
    if idp.exists():
        wanted.append(("id_random", idp))
    wanted += [("cross_dataset", p) for p in sorted(split_dir.glob("cross_dataset_fold*.json"))]
    wanted += [("leave_platform", p) for p in sorted(split_dir.glob("leave_platform_out_*.json"))]
    wanted += [("leave_sensor", p) for p in sorted(split_dir.glob("leave_sensor_out_*.json"))[:8]]
    return wanted


def make_figures(cfg: dict, all_metrics: pd.DataFrame, rtm: dict) -> list[Path]:
    fig_dir = resolve_path(cfg, "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [fig_framework(fig_dir / "fig1_framework.png")]
    paths.append(fig_nuisance_envelope(rtm["spectra"], fig_dir / "fig3_rtm_envelope.png"))
    ood = all_metrics[all_metrics["tag"].str.contains("cross_dataset|leave_|id_random")].copy()
    if not ood.empty:
        # mean R2 by trait for proposed tags
        sub = ood[ood["tag"].str.contains("M3|proposed|margspec")].copy()
        if sub.empty:
            sub = ood
        paths.append(fig_ood_dumbbell(sub, fig_dir / "fig4_ood_dumbbell.png"))
    if Path(rtm["jacobian"]).exists():
        z = np.load(rtm["jacobian"])
        jac = {k: z[k] for k in z.files if k not in {"wavelengths", "importance"}}
        paths.append(fig_jacobian(jac, fig_dir / "fig7_jacobian.png"))
    # comparison bars
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    idm = all_metrics[all_metrics["tag"].str.contains("id_random")]
    if not idm.empty:
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        models = sorted(idm["tag"].unique())
        x = np.arange(len(TRAITS))
        w = 0.8 / max(len(models), 1)
        for i, tag in enumerate(models):
            vals = [float(idm.loc[(idm.tag == tag) & (idm.trait == t), "r2"].mean()) for t in TRAITS]
            ax.bar(x + i * w, vals, width=w, label=tag.split("/")[-1][:24])
        ax.set_xticks(x + w * (len(models) - 1) / 2)
        ax.set_xticklabels(TRAITS)
        ax.set_ylabel(r"$R^2$")
        ax.set_title("In-distribution trait retrieval (real labeled split)")
        ax.legend(fontsize=7)
        p = fig_dir / "fig2_id_r2.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    ood_cd = all_metrics[all_metrics["tag"].str.contains("cross_dataset")]
    if not ood_cd.empty:
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        g = ood_cd.groupby(["tag", "trait"])["r2"].mean().unstack("trait")
        g = g.reindex(columns=TRAITS)
        g.plot(kind="bar", ax=ax)
        ax.set_ylabel(r"macro $R^2$")
        ax.set_title("Cross-dataset OOD (held-out sources)")
        ax.legend(ncol=5, fontsize=8)
        p = fig_dir / "fig5_cross_dataset.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    plat = all_metrics[all_metrics["tag"].str.contains("leave_platform")]
    if not plat.empty:
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        g = plat.groupby(["tag", "trait"])["r2"].mean().unstack("trait")
        g = g.reindex(columns=TRAITS)
        g.plot(kind="bar", ax=ax)
        ax.set_ylabel(r"$R^2$")
        ax.set_title("Leave-platform-out transfer")
        p = fig_dir / "fig6_leave_platform.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def write_run_card(cfg: dict, all_metrics: pd.DataFrame, extras: dict) -> Path:
    out = resolve_path(cfg, "results") / "run_card.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": cfg["project"]["title"],
        "gee_project": cfg["project"]["gee_project"],
        "seed": cfg["project"]["seed"],
        "device": str(_device()),
        "n_metrics": int(len(all_metrics)),
        "tags": sorted(all_metrics["tag"].unique().tolist()) if len(all_metrics) else [],
        **extras,
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def main() -> None:
    cfg = load_config("emit_free")
    cfg["train"]["batch_size"] = 32
    cfg["model"]["depth"] = 4
    cfg["model"]["dim"] = 128
    cfg["model"]["heads"] = 4
    cfg["model"]["patch_size"] = 16
    set_seed(int(cfg["project"]["seed"]))
    _log(f"device={_device()}  project={cfg['project']['gee_project']}")

    labeled, split_dir = prepare_frames(cfg)
    rtm = build_rtm(cfg, n=4000)
    unlabeled = resolve_path(cfg, "raw_hf") / "unlabeled.parquet"
    if os.environ.get("SKIP_UNLABELED", "0") == "1":
        unlabeled = unlabeled if unlabeled.exists() else None
        _log("SKIP_UNLABELED=1 — using cache only")
    else:
        unlabeled = unlabeled if unlabeled.exists() else maybe_download_unlabeled(cfg)
    pretrain = pretrain_if_possible(cfg, unlabeled)

    out_dir = resolve_path(cfg, "tables")
    tables = []
    protocols = collect_protocol_rows(split_dir)
    _log(f"{len(protocols)} frozen protocols")

    # Full model suite on ID + every OOD fold for PLSR (cheap) and proposed (core).
    # CNN and ablations on ID + first cross-dataset fold.
    for proto, spath in protocols:
        split = load_split(spath)
        tr = apply_split(labeled, split, "train")
        te = apply_split(labeled, split, "test")
        if len(tr) < 20 or len(te) < 8:
            _log(f"skip {spath.name}: train={len(tr)} test={len(te)}")
            continue
        stem = spath.stem
        _log(f"=== {stem}  n_train={len(tr)} n_test={len(te)} ===")
        tables.append(run_plsr(tr, te, out_dir, f"B0_PLSR/{stem}"))
        run_proposed = proto in {"id_random", "cross_dataset", "leave_platform", "leave_sensor"}
        if run_proposed:
            tables.append(run_transformer(cfg, tr, te, out_dir, f"M3_MargSpec/{stem}", pretrain_path=pretrain, epochs=16))
        if proto == "id_random":
            tables.append(run_cnn(tr, te, out_dir, f"B0_CNN/{stem}"))
            fixed = deepcopy(cfg)
            fixed["model"]["use_marginalization"] = False
            fixed["rtm"]["M_train"] = 1
            tables.append(run_transformer(fixed, tr, te, out_dir, f"B3_fixedRTM/{stem}", epochs=16))
            nores = deepcopy(cfg)
            nores["model"]["use_low_rank_residual"] = False
            tables.append(run_transformer(nores, tr, te, out_dir, f"M2_noResidual/{stem}", epochs=16))

    all_metrics = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    all_path = out_dir / "all_metrics.csv"
    if len(all_metrics):
        all_metrics.to_csv(all_path, index=False)
    figs = make_figures(cfg, all_metrics, rtm)
    card = write_run_card(
        cfg,
        all_metrics,
        {
            "n_labeled": int(len(labeled)),
            "n_rtm": int(rtm["spectra"].shape[0]),
            "pretrain": str(pretrain) if pretrain else None,
            "figures": [str(p) for p in figs],
            "metrics_csv": str(all_path),
        },
    )
    _log(f"done  metrics={all_path}  card={card}")


if __name__ == "__main__":
    main()
