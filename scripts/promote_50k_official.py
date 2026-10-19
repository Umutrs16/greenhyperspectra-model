#!/usr/bin/env python
"""Build official 50k T1/T2 tables, effects, pooled OOF, bootstrap, and OOD figures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from green_rtm.config import load_config, resolve_path
from green_rtm.eval.figures import fig_id_bars, fig_ood_ablation, fig_ood_median_bars, fig_ood_violin
from green_rtm.eval.metrics import trait_metrics
from effect_size_t2_t1 import paired
from finish_v2_draft import TRAITS, _load_preds
from recompute_bootstrap import N_BOOT
import numpy as np


def _rename_e50(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tag"] = (
        out["tag"]
        .astype(str)
        .str.replace("T1e50/", "T1/", regex=False)
        .str.replace("T2e50/", "T2/", regex=False)
    )
    return out


def official_ladder(tables: Path) -> pd.DataFrame:
    ladder = pd.read_csv(tables / "v2_ladder_metrics.csv")
    improved = pd.read_csv(tables / "v5_improved_emu_metrics.csv")
    remaining = pd.read_csv(tables / "v6_remaining_metrics.csv") if (tables / "v6_remaining_metrics.csv").exists() else pd.DataFrame()
    keep = ladder[~ladder["tag"].astype(str).str.match(r"^(T1|T2)/")]
    parts = [keep, _rename_e50(improved)]
    if not remaining.empty:
        id50 = remaining[remaining["tag"].astype(str).isin(("T1e50/id_random", "T2e50/id_random"))]
        if not id50.empty:
            parts.append(_rename_e50(id50))
    official = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["tag", "trait"], keep="last")
    official.to_csv(tables / "v6_official_ladder.csv", index=False)
    return official


def official_effects(tables: Path, official: pd.DataFrame) -> pd.DataFrame:
    raw = pd.concat(
        [
            pd.read_csv(tables / "v2_ladder_metrics.csv"),
            pd.read_csv(tables / "v5_improved_emu_metrics.csv"),
        ],
        ignore_index=True,
    )
    out = pd.concat(
        [paired(raw, "T2e50", "T1e50"), paired(raw, "T2e50", "T0")],
        ignore_index=True,
    )
    out["contrast"] = out["contrast"].replace({"T2e50−T1e50": "T2−T1", "T2e50−T0": "T2−T0"})
    out.to_csv(tables / "v6_effect_size.csv", index=False)
    return out


def official_pooled_boot(tables: Path) -> None:
    rows, boots = [], []
    mapping = {"PLSR": "PLSR", "T0": "T0", "T1": "T1e50", "T2": "T2e50"}
    for label, file_method in mapping.items():
        frame = _load_preds(tables, file_method)
        if frame.empty:
            continue
        src = "dataset" if "dataset" in frame.columns else "source_id"
        frame[src] = frame[src].astype(str)
        for t in TRAITS:
            rec = trait_metrics(frame[t].to_numpy(), frame[f"{t}_pred"].to_numpy(), t)
            rec["method"] = label
            rec["metric"] = "pooled_oof"
            rows.append(rec)
        groups = {
            s: g[[t for t in TRAITS] + [f"{t}_pred" for t in TRAITS]].to_numpy(dtype=np.float64)
            for s, g in frame.groupby(src, sort=False)
        }
        rng = np.random.default_rng(42)
        for t_i, t in enumerate(TRAITS):
            pred_i = t_i + len(TRAITS)
            labeled_keys = [
                s
                for s, arr in groups.items()
                if np.isfinite(arr[:, t_i]).any() and np.isfinite(arr[:, pred_i]).any()
            ]
            n_obs = int(
                sum((np.isfinite(groups[s][:, t_i]) & np.isfinite(groups[s][:, pred_i])).sum() for s in labeled_keys)
            )
            recs = []
            for _ in range(N_BOOT):
                ys, yps = [], []
                for s in rng.choice(labeled_keys, size=len(labeled_keys), replace=True):
                    arr = groups[s]
                    m = np.isfinite(arr[:, t_i]) & np.isfinite(arr[:, pred_i])
                    if not m.any():
                        continue
                    take = arr[m]
                    if len(take) > 30:
                        take = take[rng.choice(len(take), 30, replace=False)]
                    ys.append(take[:, t_i])
                    yps.append(take[:, pred_i])
                if ys:
                    recs.append(trait_metrics(np.concatenate(ys), np.concatenate(yps), t))
            tab = pd.DataFrame(recs)
            boot = {
                "trait": t,
                "method": label,
                "n_boot": int(len(tab)),
                "n_obs": n_obs,
                "n_sources": int(len(labeled_keys)),
            }
            for col in ("r2", "rmse", "spearman"):
                boot[f"{col}_mean"] = float(tab[col].mean())
                boot[f"{col}_lo"] = float(tab[col].quantile(0.025))
                boot[f"{col}_hi"] = float(tab[col].quantile(0.975))
            boots.append(boot)
            print(f"{label} {t} r2={boot['r2_mean']:.3f} [{boot['r2_lo']:.3f}, {boot['r2_hi']:.3f}]", flush=True)
    pd.DataFrame(rows).to_csv(tables / "v6_pooled_oof.csv", index=False)
    pd.DataFrame(boots).to_csv(tables / "v6_source_bootstrap.csv", index=False)


def official_figures(tables: Path, figs: Path, official: pd.DataFrame) -> None:
    fig_id_bars(official, figs / "fig2_id_r2.png")
    ood = official[official["tag"].astype(str).str.contains("cross_dataset", na=False)].copy()
    fig_ood_median_bars(ood, figs / "fig4_ood_median.png")
    fig_ood_violin(ood, figs / "fig5_ood_violin.png")
    (figs / "fig4_ood_dumbbell.png").write_bytes((figs / "fig4_ood_median.png").read_bytes())
    (figs / "fig5_cross_dataset.png").write_bytes((figs / "fig5_ood_violin.png").read_bytes())
    remaining = pd.read_csv(tables / "v6_remaining_metrics.csv") if (tables / "v6_remaining_metrics.csv").exists() else pd.DataFrame()
    t2m = pd.read_csv(tables / "v5_t2m_metrics.csv") if (tables / "v5_t2m_metrics.csv").exists() else pd.DataFrame()
    t2m50 = pd.read_csv(tables / "v6_t2m50_metrics.csv") if (tables / "v6_t2m50_metrics.csv").exists() else pd.DataFrame()
    extra = [ood[ood["tag"].astype(str).str.match(r"^(T0|T2)/")]]
    if not remaining.empty:
        extra.append(remaining[remaining["tag"].astype(str).str.contains("cross_dataset", na=False)])
    if not t2m.empty:
        extra.append(t2m)
    if not t2m50.empty:
        extra.append(t2m50)
    abl = pd.concat(extra, ignore_index=True)
    fig_ood_ablation(abl, figs / "figS_ablation_ood.png")


def main() -> None:
    cfg = load_config("emit_free")
    tables = resolve_path(cfg, "tables")
    figs = resolve_path(cfg, "figures")
    official = official_ladder(tables)
    official_effects(tables, official)
    official_figures(tables, figs, official)
    print("wrote official 50k ladder, effects, and figures", flush=True)


if __name__ == "__main__":
    main()
