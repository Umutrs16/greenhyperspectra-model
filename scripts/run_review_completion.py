#!/usr/bin/env python
"""Experiments requested by the first-round reviews and not previously run.

Does not overwrite the reported T0/T1/T2 metric files.
GreenHyperSpectra has no sun-view angles and no EMIT in-situ labels, so those
two requests are not fabricated: geometry is a controlled clamp at the
training-code initial angles, and EMIT in-situ is skipped.
"""

from __future__ import annotations

import os
import sys
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import torch

from green_rtm.config import load_config, resolve_path, set_seed
from green_rtm.data.splits import apply_split, load_split
from green_rtm.models import build_model
from green_rtm.rtm.emulator import load_emulator
from green_rtm.train.engine import audit_rtm_pathways
from run_real_experiments import _apply_zscore, _fit_zscore, _loader, prepare_frames
from run_v2_ladder import _variant_cfg, run_cnn, run_transformer

OOD = tuple(f"cross_dataset_fold{i:02d}" for i in range(10))
FOLDS3 = ("cross_dataset_fold01", "cross_dataset_fold03", "cross_dataset_fold07")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _done(out: Path, tag: str) -> bool:
    p = out / f"{tag.replace('/', '_')}_metrics.csv"
    return p.exists() and p.stat().st_size > 0


def _base():
    cfg = load_config("emit_free")
    cfg["train"]["batch_size"] = 32
    cfg["model"].update({"depth": 4, "dim": 128, "heads": 4, "patch_size": 16, "use_low_rank_residual": False})
    return cfg


def _run(cfg, labeled, split_dir, emu, out, variant, stem, tag, epochs, seed, model_over=None, lr=None):
    if _done(out, tag):
        _log(f"skip {tag}")
        return
    cfg = deepcopy(cfg)
    cfg["project"]["seed"] = int(seed)
    if model_over:
        cfg["model"].update(model_over)
    if lr is not None:
        cfg["train"]["lr"] = float(lr)
    set_seed(int(seed))
    split = load_split(split_dir / f"{stem}.json")
    tr, te = apply_split(labeled, split, "train"), apply_split(labeled, split, "test")
    vcfg = _variant_cfg(cfg, variant)
    if model_over:
        vcfg["model"].update(model_over)
    if lr is not None:
        vcfg["train"]["lr"] = float(lr)
    t0 = time.perf_counter()
    _log(f"=== {tag} n={len(tr)}/{len(te)} epochs={epochs} seed={seed} ===")
    run_transformer(vcfg, tr, te, out, tag, emu, epochs=epochs)
    _log(f"done {tag} {time.perf_counter() - t0:.1f}s")


def audit_50k(cfg, labeled, split_dir, emu_path, out: Path) -> None:
    dest = out / "rtm_pathway_audit_50k.csv"
    if dest.exists() and dest.stat().st_size > 0:
        _log(f"skip audit {dest.name}")
        return
    ckpt = resolve_path(cfg, "checkpoints")
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for variant, stem in [
        ("T1", "id_random"),
        ("T2", "id_random"),
        *[(v, s) for v in ("T1", "T2") for s in OOD],
    ]:
        tag = f"{variant}e50/{stem}"
        path = ckpt / f"{tag.replace('/', '_')}_last.pt"
        if not path.exists():
            _log(f"missing checkpoint {path.name}")
            continue
        split = load_split(split_dir / f"{stem}.json")
        tr = apply_split(labeled, split, "train")
        stats = _fit_zscore(tr)
        vcfg = _variant_cfg(cfg, variant)
        emu = load_emulator(emu_path)
        emu.eval()
        for p in emu.parameters():
            p.requires_grad = False
        model = build_model(vcfg, emulator=emu)
        model.set_trait_stats(stats)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        batch = next(iter(_loader(_apply_zscore(tr, stats), vcfg, shuffle=True)))
        rec = audit_rtm_pathways(model, batch, vcfg)
        rec["tag"] = tag
        rows.append(rec)
        _log(f"audit {tag} ratio={rec.get('ratio_rtm_over_trait')}")
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(dest, index=False)
    _log(f"wrote {dest}")


def cnn_ood(cfg, labeled, split_dir, out: Path) -> None:
    for stem in OOD:
        tag = f"CNN/{stem}"
        if _done(out, tag):
            _log(f"skip {tag}")
            continue
        split = load_split(split_dir / f"{stem}.json")
        tr, te = apply_split(labeled, split, "train"), apply_split(labeled, split, "test")
        t0 = time.perf_counter()
        _log(f"=== {tag} ===")
        run_cnn(tr, te, out, tag)
        _log(f"done {tag} {time.perf_counter() - t0:.1f}s")


def identity_cab(out: Path, tag: str) -> float:
    p = out / f"{tag.replace('/', '_')}_metrics.csv"
    if not p.exists():
        return float("nan")
    tab = pd.read_csv(p)
    hit = tab[tab["trait"] == "Cab"]
    return float(hit["r2"].iloc[0]) if len(hit) else float("nan")


def summarize(out: Path) -> None:
    rows = []
    for p in out.glob("*_metrics.csv"):
        name = p.name
        if not any(k in name for k in ("CNN_cross", "T2mN50", "T2geo50", "T0s", "T1s", "T2s", "T0tune", "T1tune", "T2tune", "e50_L")):
            continue
        tab = pd.read_csv(p)
        rows.append(tab)
    if not rows:
        return
    all_tab = pd.concat(rows, ignore_index=True)
    all_tab.to_csv(out / "review_round2_metrics.csv", index=False)
    all_tab["model"] = all_tab["tag"].str.split("/").str[0]
    med = all_tab.groupby(["model", "trait"])["r2"].median().unstack("trait")
    med.to_csv(out / "review_round2_median_r2.csv")
    _log(med.to_string())


def main() -> None:
    bench = "--bench" in sys.argv
    cfg = _base()
    labeled, split_dir = prepare_frames(cfg)
    emu = resolve_path(cfg, "rtm_library") / "learning_curve" / "forward_emulator_50000.pt"
    if not emu.exists():
        raise FileNotFoundError(emu)
    out = resolve_path(cfg, "tables")
    if bench:
        _run(cfg, labeled, split_dir, emu, out, "T2", "cross_dataset_fold01", "BENCH/t2", 1, 42)
        return
    t_all = time.perf_counter()
    audit_50k(cfg, labeled, split_dir, emu, out)
    cnn_ood(cfg, labeled, split_dir, out)

    # Identity-only search. The promoted setting is the higher Cab R² of the two.
    _run(cfg, labeled, split_dir, emu, out, "T0", "id_random", "T0e48/id_random", 48, 42)
    wide = {"depth": 6, "dim": 256, "heads": 8}
    _run(cfg, labeled, split_dir, emu, out, "T0", "id_random", "T0w256/id_random", 48, 42, model_over=wide, lr=3e-4)
    cab_long = identity_cab(out, "T0e48/id_random")
    cab_wide = identity_cab(out, "T0w256/id_random")
    _log(f"identity Cab T0e48={cab_long:.3f} T0w256={cab_wide:.3f}")
    if cab_wide > cab_long:
        tune_over, tune_lr, tune_name = wide, 3e-4, "w256"
    else:
        tune_over, tune_lr, tune_name = None, None, "e48"
    for variant in ("T0", "T1", "T2"):
        for stem in OOD:
            _run(
                cfg, labeled, split_dir, emu, out, variant, stem,
                f"{variant}tune{tune_name}/{stem}", 48, 42, model_over=tune_over, lr=tune_lr,
            )

    for stem in OOD:
        _run(cfg, labeled, split_dir, emu, out, "T2mN", stem, f"T2mN50/{stem}", 12, 42)
        _run(cfg, labeled, split_dir, emu, out, "T2geo", stem, f"T2geo50/{stem}", 12, 42)
    _run_lambda(cfg, labeled, split_dir, emu, out)
    for seed in (43, 44):
        for variant in ("T0", "T1", "T2"):
            for stem in OOD:
                _run(cfg, labeled, split_dir, emu, out, variant, stem, f"{variant}s{seed}/{stem}", 12, seed)
    summarize(out)
    _log(f"ALL DONE {time.perf_counter() - t_all:.1f}s")


def _run_lambda(cfg, labeled, split_dir, emu, out: Path) -> None:
    for lam in (0.1, 1.0, 3.0):
        for variant in ("T1", "T2"):
            for stem in FOLDS3:
                tag = f"{variant}e50_L{lam:g}/{stem}"
                if _done(out, tag):
                    _log(f"skip {tag}")
                    continue
                local = deepcopy(cfg)
                local["project"]["seed"] = 42
                local["train"]["lambdas"]["rtm"] = float(lam)
                set_seed(42)
                split = load_split(split_dir / f"{stem}.json")
                tr, te = apply_split(labeled, split, "train"), apply_split(labeled, split, "test")
                vcfg = _variant_cfg(local, variant)
                vcfg["train"]["lambdas"]["rtm"] = float(lam)
                t0 = time.perf_counter()
                _log(f"=== {tag} ===")
                run_transformer(vcfg, tr, te, out, tag, emu, epochs=12)
                _log(f"done {tag} {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
