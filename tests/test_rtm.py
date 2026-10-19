import numpy as np

from green_rtm.rtm.prospect_pro import run_prospect_pro, run_prospectpro_4sail
from green_rtm.spectra import N_GHS_BANDS


def test_prospect_pro_leaf():
    wv, r, t = run_prospect_pro(N=1.5, cab=40, car=8, cbrown=0.0, cw=0.01, prot=0.002, cbc=0.008)
    assert len(wv) == 2101
    assert r.min() >= 0
    assert t.min() >= 0
    assert (r + t).max() <= 1.05


def test_canopy_ghs_grid():
    rho = run_prospectpro_4sail(
        N=1.4,
        cab=35,
        car=8,
        cbrown=0.0,
        cw=0.012,
        prot=0.002,
        cbc=0.007,
        lai=2.5,
        lidfa=45,
        hspot=0.02,
        tts=30,
        tto=0,
        psi=0,
        ghs_grid=True,
    )
    assert rho.shape == (N_GHS_BANDS,)
    assert np.isfinite(rho).all()
    assert 0 <= rho.min() and rho.max() <= 1.2
