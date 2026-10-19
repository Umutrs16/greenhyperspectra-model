"""Journal figures generated from frozen metric tables and physics libraries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..spectra import GHS_WAVELENGTHS, PRIMARY_TRAITS


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _method(tag: str) -> str:
    return str(tag).split("/")[0]


def _map_background(ax, ccrs, cfeature):
    ax.add_feature(cfeature.OCEAN, facecolor="#dce8f2", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f3efe4", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.35", zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="0.55", zorder=2)


def _plot_sites(ax, ccrs, campaigns, emit, label_emit: bool):
    if campaigns is not None and not campaigns.empty:
        ax.scatter(
            campaigns["lon"].to_numpy(),
            campaigns["lat"].to_numpy(),
            transform=ccrs.PlateCarree(),
            s=28,
            c="#4C78A8",
            edgecolors="0.15",
            linewidths=0.35,
            zorder=4,
            label="Named campaigns (approx.)",
            clip_on=True,
        )
    if emit is None or emit.empty:
        return
    aliases = {
        "ca_central_valley": "1",
        "us_midwest": "2",
        "sahel_savanna": "3",
        "iberia_dehesa": "4",
        "australia_wheatbelt": "5",
        "amazon_tapajos": "6",
    }
    first = True
    for _, row in emit.iterrows():
        xs = [float(row["lon_min"]), float(row["lon_max"]), float(row["lon_max"]), float(row["lon_min"]), float(row["lon_min"])]
        ys = [float(row["lat_min"]), float(row["lat_min"]), float(row["lat_max"]), float(row["lat_max"]), float(row["lat_min"])]
        ax.plot(
            xs,
            ys,
            transform=ccrs.PlateCarree(),
            color="#E45756",
            lw=0.9,
            zorder=5,
            label="EMIT footprint" if first else None,
            clip_on=True,
        )
        first = False
        if not label_emit:
            continue
        lab = aliases.get(str(row.get("region", "")), "")
        if not lab:
            continue
        ax.annotate(
            lab,
            xy=(0.5 * (xs[0] + xs[1]), 0.5 * (ys[0] + ys[2])),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            color="#8b1e1e",
            zorder=6,
            clip_on=True,
        )


def fig_data_map(emit: pd.DataFrame, campaigns: pd.DataFrame, out: Path) -> Path:
    """World map with EMIT footprints and two regional insets."""
    _style()
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(8.4, 6.6))
    ax = fig.add_subplot(2, 1, 1, projection=ccrs.Robinson(central_longitude=0))
    ax.set_global()
    _map_background(ax, ccrs, cfeature)
    gl = ax.gridlines(draw_labels=False, linewidth=0.25, color="0.7", alpha=0.5, linestyle=":")
    gl.xlocator = plt.FixedLocator(range(-180, 181, 60))
    gl.ylocator = plt.FixedLocator(range(-60, 61, 30))
    _plot_sites(ax, ccrs, campaigns, emit, label_emit=True)
    ax.legend(loc="lower left", frameon=True, fancybox=False, edgecolor="0.6", fontsize=7)
    ax.set_title("Approximate campaign centroids and EMIT granule footprints", fontsize=10)

    panels = (
        (fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree()), [-128, -114, 32, 42], "Western North America"),
        (fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree()), [-12, 16, 36, 52], "Western Europe"),
    )
    for axi, extent, title in panels:
        axi.set_extent(extent, crs=ccrs.PlateCarree())
        _map_background(axi, ccrs, cfeature)
        _plot_sites(axi, ccrs, campaigns, emit, label_emit=True)
        axi.set_title(title, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def fig_framework(out: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    data = "#d9e8f5"
    phys = "#f3e0c8"
    boxes = [
        (0.02, 0.58, 0.20, 0.30, "Labeled spectra\nand incomplete traits", data),
        (0.02, 0.12, 0.20, 0.30, "Unlabeled spectra", data),
        (0.28, 0.58, 0.22, 0.30, "Variable-band\nTransformer → θ̂", data),
        (0.28, 0.08, 0.22, 0.38, "PROSPECT-PRO/4SAIL\nηₘ ∼ p(η), m = 1…M\n→ mean spectrum", phys),
        (0.56, 0.35, 0.20, 0.30, "L_trait(θ̂, y)\n+ L(R, E[F])", "#efe7d8"),
        (0.80, 0.35, 0.18, 0.30, "External EMIT\nnative-band\ndemonstration", data),
    ]
    for x, y, w, h, title, fc in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor=fc, edgecolor="0.2", lw=1.2))
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(0.28, 0.73), xytext=(0.22, 0.73), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(0.28, 0.68), xytext=(0.22, 0.27), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.45"))
    ax.annotate("", xy=(0.39, 0.46), xytext=(0.39, 0.58), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(0.56, 0.55), xytext=(0.50, 0.70), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(0.56, 0.45), xytext=(0.50, 0.27), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(0.80, 0.50), xytext=(0.76, 0.50), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(0.12, 0.93, "Observed data", color="#2a4d6e", fontsize=8, fontweight="bold")
    ax.text(0.34, 0.02, "Physics branch  (unconditional p(η))", color="#7a4a12", fontsize=8, fontweight="bold")
    ax.set_title(r"Trait estimate $\hat\theta$ enters a nuisance-averaged RTM constraint")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_nuisance_envelope(spectra: np.ndarray, out: Path, title: str = "Marginalized PROSPECT-PRO/4SAIL envelope") -> Path:
    _style()
    q = np.quantile(spectra, [0.05, 0.5, 0.95], axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    wl = GHS_WAVELENGTHS[: q.shape[1]]
    ax.fill_between(wl, q[0], q[2], alpha=0.28, color="#4C78A8", label="5–95%")
    ax.plot(wl, q[1], lw=1.2, color="#1f3b5c", label="median")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title(title)
    ax.legend(frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_nuisance_only_envelope(curves: dict[str, dict[str, np.ndarray]], out: Path) -> Path:
    """Fixed-θ, sampled-η envelopes. Each panel is one canopy state."""
    _style()
    names = list(curves)
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * max(len(names), 1), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    wl = GHS_WAVELENGTHS
    for ax, name in zip(axes, names):
        rec = curves[name]
        q05, q50, q95 = rec["q05"], rec["q50"], rec["q95"]
        ax.fill_between(wl[: len(q05)], q05, q95, alpha=0.28, color="#4C78A8", label="5–95% nuisance")
        ax.plot(wl[: len(q50)], q50, color="#1f3b5c", lw=1.2, label="nuisance median")
        if "fixed" in rec:
            ax.plot(wl[: len(rec["fixed"])], rec["fixed"], color="#E45756", lw=1.1, ls="--", label="fixed nuisance")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Reflectance")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(r"Nuisance-only envelope at fixed $\theta$", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_emulator_band_error(table: pd.DataFrame, out: Path) -> Path:
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    axes[0].plot(table["wavelength_nm"], table["rmse"], color="#4C78A8", lw=1.1)
    axes[0].set_ylabel(r"RMSE$_\lambda$")
    axes[0].set_title("Hold-out emulator reflectance error")
    axes[1].axhline(0, color="0.7", lw=0.8)
    axes[1].plot(table["wavelength_nm"], table["bias"], color="#E45756", lw=1.1)
    axes[1].set_ylabel(r"Bias$_\lambda$")
    axes[1].set_xlabel("Wavelength (nm)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_emulator_band_compare(table: pd.DataFrame, out: Path) -> Path:
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    colors = {4000: "#4C78A8", 50000: "#E45756"}
    labels = {4000: r"$N=4{,}000$ (development)", 50000: r"$N=50{,}000$ (reported)"}
    for n, sub in table.groupby("n_train"):
        n = int(n)
        axes[0].plot(sub["wavelength_nm"], sub["rmse"], color=colors.get(n, "0.4"), lw=1.1, label=labels.get(n, str(n)))
        axes[1].plot(sub["wavelength_nm"], sub["bias"], color=colors.get(n, "0.4"), lw=1.1, label=labels.get(n, str(n)))
    axes[0].set_ylabel(r"RMSE$_\lambda$")
    axes[0].set_title("Hold-out emulator reflectance error")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0, color="0.7", lw=0.8)
    axes[1].set_ylabel(r"Bias$_\lambda$")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_id_bars(table: pd.DataFrame, out: Path) -> Path:
    _style()
    frame = table[table["tag"].astype(str).str.endswith("/id_random")].copy()
    frame["method"] = frame["tag"].map(_method)
    order = [m for m in ("PLSR", "CNN", "T0", "T1", "T2") if m in set(frame["method"])]
    traits = [t for t in PRIMARY_TRAITS if t in set(frame["trait"])]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = np.arange(len(traits))
    w = 0.78 / max(len(order), 1)
    colors = {"PLSR": "#4C78A8", "CNN": "#F58518", "T0": "#54A24B", "T1": "#E45756", "T2": "#B279A2"}
    for i, m in enumerate(order):
        vals = [float(frame.loc[(frame.method == m) & (frame.trait == t), "r2"].mean()) for t in traits]
        ax.bar(x + i * w, vals, width=w, label=m, color=colors.get(m, "0.5"))
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xticks(x + w * (len(order) - 1) / 2)
    ax.set_xticklabels(traits)
    ax.set_ylabel(r"$R^2$")
    ax.set_title("In-distribution identity split")
    ax.legend(ncol=len(order), frameon=False, fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_ood_median_bars(table: pd.DataFrame, out: Path) -> Path:
    _style()
    frame = table[table["tag"].astype(str).str.contains("cross_dataset")].copy()
    frame["method"] = frame["tag"].map(_method)
    order = [m for m in ("PLSR", "T0", "T1", "T2") if m in set(frame["method"])]
    traits = [t for t in PRIMARY_TRAITS if t in set(frame["trait"])]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = np.arange(len(traits))
    w = 0.78 / max(len(order), 1)
    colors = {"PLSR": "#4C78A8", "T0": "#54A24B", "T1": "#E45756", "T2": "#B279A2"}
    for i, m in enumerate(order):
        vals = [float(frame.loc[(frame.method == m) & (frame.trait == t), "r2"].median()) for t in traits]
        ax.bar(x + i * w, vals, width=w, label=m, color=colors.get(m, "0.5"))
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xticks(x + w * (len(order) - 1) / 2)
    ax.set_xticklabels(traits)
    ax.set_ylabel(r"median fold $R^2$")
    ax.set_title("Source-blocked cross-source median $R^2$")
    ax.legend(frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_ood_ablation(table: pd.DataFrame, out: Path) -> Path:
    _style()
    frame = table[table["tag"].astype(str).str.contains("cross_dataset")].copy()
    frame["method"] = frame["tag"].map(_method)
    order = [m for m in ("T0", "T2", "T2m", "T2m50", "T2e50M16", "T2p50") if m in set(frame["method"])]
    traits = [t for t in PRIMARY_TRAITS if t in set(frame["trait"])]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    x = np.arange(len(traits))
    w = 0.84 / max(len(order), 1)
    colors = {"T0": "#54A24B", "T2": "#B279A2", "T2m": "#4C78A8", "T2m50": "#72B7B2", "T2e50M16": "#E45756", "T2p50": "#F58518"}
    labels = {"T0": "T0", "T2": "T2 (M=8)", "T2m": "T2-LM-4k", "T2m50": "T2-LM", "T2e50M16": "T2-M16", "T2p50": "T2-PF"}
    for i, m in enumerate(order):
        vals = [float(frame.loc[(frame.method == m) & (frame.trait == t), "r2"].median()) for t in traits]
        ax.bar(x + i * w, vals, width=w, label=labels.get(m, m), color=colors.get(m, "0.5"))
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xticks(x + w * (len(order) - 1) / 2)
    ax.set_xticklabels(traits)
    ax.set_ylabel(r"median fold $R^2$")
    ax.set_title("Physics ablations on source-blocked folds")
    ax.legend(frameon=False, fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_ood_dumbbell(table: pd.DataFrame, out: Path) -> Path:
    return fig_ood_median_bars(table, out)


def fig_ood_violin(table: pd.DataFrame, out: Path) -> Path:
    _style()
    frame = table[table["tag"].astype(str).str.contains("cross_dataset")].copy()
    frame["method"] = frame["tag"].map(_method)
    methods = [m for m in ("PLSR", "T0", "T1", "T2") if m in set(frame["method"])]
    traits = [t for t in PRIMARY_TRAITS if t in set(frame["trait"])]
    fig, axes = plt.subplots(2, len(traits), figsize=(2.45 * max(len(traits), 1), 7.2), sharex="col")
    axes = np.atleast_2d(axes)
    rng = np.random.default_rng(0)
    colors = {"PLSR": "#4C78A8", "T0": "#54A24B", "T1": "#E45756", "T2": "#B279A2"}
    for row, ylim, ylab in (
        (0, (-6.2, 1.15), r"fold $R^2$"),
        (1, (-1.0, 1.0), r"fold $R^2$ in $[-1,1]$"),
    ):
        for ax, t in zip(axes[row], traits):
            sub = frame[frame["trait"] == t]
            for i, m in enumerate(methods):
                vals = pd.to_numeric(sub.loc[sub["method"] == m, "r2"], errors="coerce").dropna().to_numpy()
                x = np.full(len(vals), i, dtype=float) + rng.normal(0, 0.055, size=len(vals))
                ax.scatter(x, vals, s=22, alpha=0.78, color=colors.get(m, "0.4"), zorder=2)
                if len(vals):
                    ax.hlines(np.median(vals), i - 0.22, i + 0.22, color="0.1", lw=1.6, zorder=3)
            ax.axhline(0, color="0.7", lw=0.8)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=10)
            if row == 0:
                ax.set_title(t, fontsize=12)
            ax.tick_params(axis="y", labelsize=10)
            if t == traits[0]:
                ax.set_ylabel(ylab, fontsize=10)
            ax.set_ylim(*ylim)
    fig.suptitle("Source-blocked fold $R^2$ (points) and median (bar). Bottom row zooms to $[-1,1]$.", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_label_shift(table: pd.DataFrame, out: Path) -> Path:
    _style()
    frame = table[table["fold"].astype(str).str.startswith("cross_dataset")].copy()
    traits = [t for t in PRIMARY_TRAITS if t in set(frame["trait"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, col, ylab, title in (
        (axes[0], "extrapolation_rate", "Extrapolation rate", "Test labels outside the training range"),
        (axes[1], "range_overlap_frac", "Range-overlap fraction", "Train–test trait-range overlap"),
    ):
        for t in traits:
            vals = pd.to_numeric(frame.loc[frame["trait"] == t, col], errors="coerce").dropna().to_numpy()
            ax.scatter(np.full(len(vals), traits.index(t)) + np.random.default_rng(1).normal(0, 0.06, len(vals)), vals, s=16, alpha=0.75)
            if len(vals):
                ax.hlines(np.median(vals), traits.index(t) - 0.22, traits.index(t) + 0.22, color="0.15", lw=1.4)
        ax.set_xticks(range(len(traits)))
        ax.set_xticklabels(traits)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_m_convergence(table: pd.DataFrame, out: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(table["M"], table["rmse_to_M64"], "o-", color="#4C78A8", lw=1.4)
    if (table["M"] == 8).any():
        y8 = float(table.loc[table["M"] == 8, "rmse_to_M64"].iloc[0])
        ax.axvline(8, color="0.55", ls=":", lw=1.0)
        ax.scatter([8], [y8], s=46, zorder=3, color="#4C78A8")
        ax.text(8.3, y8, r"reported $M=8$", fontsize=8, color="0.25", va="bottom")
    if (table["M"] == 16).any():
        ax.axvline(16, color="#E45756", ls="--", lw=1.0, label=r"tighter $M=16$")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(table["M"])
    ax.set_xticklabels([str(int(m)) for m in table["M"]])
    ax.set_xlabel("Monte Carlo samples $M$")
    ax.set_ylabel(r"RMSE$(R_M, R_{64})$")
    ax.set_title("Nuisance-average spectral convergence")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_jacobian(jac: dict[str, np.ndarray], out: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    for t, w in jac.items():
        ax.plot(GHS_WAVELENGTHS[: len(w)], w / (np.max(w) + 1e-8), label=t, lw=1.1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Normalized $|\partial R/\partial\theta|$")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.set_title("Emulator Jacobian (each trait scaled independently)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_emulator_learning_curve(hold: pd.DataFrame, jac: pd.DataFrame, out: Path) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9))
    hold = hold.sort_values("n_train")
    axes[0].plot(hold["n_train"], hold["rmse"], "o-", color="#4C78A8", label="RMSE")
    axes[0].plot(hold["n_train"], hold["sam_mean_deg"] / 100.0, "s--", color="#F58518", label=r"SAM$^\circ$/100")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$N_{\mathrm{RTM}}$")
    axes[0].set_ylabel("Hold-out spectral error")
    axes[0].set_title("Forward reflectance fidelity")
    axes[0].legend(frameon=False, fontsize=8)
    colors = {"LAI": "#4C78A8", "Cab": "#E45756", "Cw": "#54A24B", "Cm": "#F58518", "Cp": "#B279A2"}
    for t in PRIMARY_TRAITS:
        sub = jac[jac["trait"] == t].sort_values("n_train")
        if sub.empty:
            continue
        lw = 2.0 if t in {"Cab", "Cp"} else 1.1
        axes[1].plot(sub["n_train"], sub["corr_median"], "o-", color=colors.get(t, "0.4"), label=t, lw=lw)
    axes[1].axhline(0.8, color="0.65", ls=":", lw=1.0)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$N_{\mathrm{RTM}}$")
    axes[1].set_ylabel(r"median corr$(J_{\mathrm{RTM}}, J_{\mathrm{emu}})$")
    axes[1].set_title("Local Jacobian agreement")
    axes[1].set_ylim(0.4, 1.02)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_training_curves(hist: pd.DataFrame, out: Path) -> Path:
    _style()
    methods = [m for m in ("T0", "T1", "T2") if m in set(hist["method"])]
    fig, axes = plt.subplots(1, max(len(methods), 1), figsize=(3.1 * max(len(methods), 1), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, methods):
        sub = hist[hist["method"] == m].sort_values("epoch")
        ax.plot(sub["epoch"], sub["train_loss"], "o-", color="#4C78A8", label="train")
        if "val_loss" in sub.columns:
            ax.plot(sub["epoch"], sub["val_loss"], "s--", color="#E45756", label="identity hold-out")
        ax.set_title(m)
        ax.set_xlabel("Epoch")
        ax.set_xticks(sorted(sub["epoch"].unique()))
    if len(axes):
        axes[0].set_ylabel("Loss")
        axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Identity-split fine-tune loss (12 epochs, no early stopping)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def make_all_figures(out_dir: Path, tables: dict[str, pd.DataFrame] | None = None, jac: dict | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [fig_framework(out_dir / "fig1_framework.png")]
    tables = tables or {}
    if "id" in tables:
        paths.append(fig_id_bars(tables["id"], out_dir / "fig2_id_r2.png"))
    if "ood" in tables:
        paths.append(fig_ood_median_bars(tables["ood"], out_dir / "fig4_ood_median.png"))
        paths.append(fig_ood_violin(tables["ood"], out_dir / "fig5_ood_violin.png"))
    if "shift" in tables:
        paths.append(fig_label_shift(tables["shift"], out_dir / "fig6_label_shift.png"))
    if "mconv" in tables:
        paths.append(fig_m_convergence(tables["mconv"], out_dir / "fig3b_m_convergence.png"))
    if jac:
        paths.append(fig_jacobian(jac, out_dir / "fig7_jacobian.png"))
    return paths
