"""Source-level bootstrap 95% CIs. Pixels/spectra inside a source are not treated as independent."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import trait_metrics


def source_bootstrap(
    df: pd.DataFrame,
    trait: str,
    pred_col: str,
    source_col: str = "source_id",
    n_boot: int = 1000,
    seed: int = 42,
    max_per_source: int = 30,
) -> dict:
    rng = np.random.default_rng(seed)
    sources = df[source_col].astype(str).unique()
    recs = []
    for _ in range(n_boot):
        pick = rng.choice(sources, size=len(sources), replace=True)
        ys, yps = [], []
        for s in pick:
            g = df.loc[df[source_col].astype(str) == s, [trait, pred_col]].dropna()
            if g.empty:
                continue
            if len(g) > max_per_source:
                g = g.sample(max_per_source, random_state=int(rng.integers(0, 1_000_000)))
            ys.append(g[trait].to_numpy())
            yps.append(g[pred_col].to_numpy())
        if not ys:
            continue
        m = trait_metrics(np.concatenate(ys), np.concatenate(yps), trait)
        recs.append(m)
    if not recs:
        return {"trait": trait, "n_boot": 0}
    tab = pd.DataFrame(recs)
    out = {"trait": trait, "n_boot": len(tab), "n_sources": int(len(sources))}
    for col in ("r2", "rmse", "mae", "nrmse", "bias", "spearman"):
        out[f"{col}_mean"] = float(tab[col].mean())
        out[f"{col}_lo"] = float(tab[col].quantile(0.025))
        out[f"{col}_hi"] = float(tab[col].quantile(0.975))
    return out


def bootstrap_table(df: pd.DataFrame, traits: list[str], **kwargs) -> pd.DataFrame:
    return pd.DataFrame([source_bootstrap(df, t, f"{t}_pred", **kwargs) for t in traits])
