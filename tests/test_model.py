import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from green_rtm.config import load_config
from green_rtm.data.datasets import VariableBandDataset, collate_variable
from green_rtm.models import build_model
from green_rtm.spectra import GHS_WAVELENGTHS
from green_rtm.train.losses import combined_loss


def _tiny_cfg():
    cfg = load_config()
    cfg["model"]["depth"] = 1
    cfg["model"]["dim"] = 32
    cfg["model"]["heads"] = 4
    return cfg


def _frame(n=8, bands=None):
    rng = np.random.default_rng(1)
    wl = GHS_WAVELENGTHS if bands is None else bands
    data = {str(int(w)): rng.random(n).astype(np.float32) * 0.4 + 0.05 for w in wl}
    df = pd.DataFrame(data)
    df["LAI"] = rng.random(n) * 4
    df["Cab"] = rng.random(n) * 40 + 10
    df["Cw"] = rng.random(n) * 0.02 + 0.002
    df["Cm"] = rng.random(n) * 0.01 + 0.002
    df["Cp"] = rng.random(n) * 0.004 + 0.001
    return df


def test_forward_and_loss():
    cfg = _tiny_cfg()
    ds = VariableBandDataset(_frame(), degrade=True)
    batch = next(iter(DataLoader(ds, batch_size=4, collate_fn=collate_variable)))
    model = build_model(cfg)
    out = model(batch, pretrain=True)
    assert out["traits"].shape == (4, 5)
    assert out["recon"].shape[0] == 4
    loss, logs = combined_loss(out, batch, cfg)
    loss.backward()
    assert logs["loss"] == logs["loss"]


def test_protein_fraction_enforces_cp_le_cm():
    cfg = _tiny_cfg()
    cfg["model"]["use_protein_fraction"] = True
    cfg["model"]["use_rtm"] = False
    model = build_model(cfg)
    model.trait_mean.copy_(torch.tensor([2.0, 30.0, 0.01, 0.01, 0.004]))
    model.trait_std.copy_(torch.tensor([1.0, 10.0, 0.005, 0.005, 0.002]))
    z = torch.tensor([[0.0, 0.0, 0.0, 0.0, 8.0], [0.0, 0.0, 0.0, 1.0, -8.0]])
    phys = model._traits_to_physical(z)
    assert torch.all(phys[:, 4] <= phys[:, 3] + 1e-6)
    assert float(phys[0, 4]) > float(phys[1, 4])


def test_emit_native_tokens():
    cfg = _tiny_cfg()
    # 285 dummy EMIT-like bands
    wl = np.linspace(381, 2493, 285).astype(np.float32)
    fwhm = np.full(285, 7.4, np.float32)
    rng = np.random.default_rng(2)
    data = {str(i): rng.random(6).astype(np.float32) * 0.3 for i in range(285)}
    df = pd.DataFrame(data)
    for t in ("LAI", "Cab", "Cw", "Cm", "Cp"):
        df[t] = 1.0
    # overwrite spectral cols to actual wavelengths as names for extractor
    df = pd.DataFrame({str(int(w)): rng.random(6).astype(np.float32) * 0.3 for w in wl})
    for t in ("LAI", "Cab", "Cw", "Cm", "Cp"):
        df[t] = 1.0
    ds = VariableBandDataset(df, wavelengths=wl, fwhm=fwhm)
    batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=collate_variable)))
    model = build_model(cfg)
    # Encoder accepts any length; residual/RTM stay on 1721 so only encode here
    enc = model.encoder(batch)
    assert enc["cls"].shape == (2, cfg["model"]["dim"])
    assert enc["tokens"].shape[1] == 285
