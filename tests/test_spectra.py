from green_rtm.spectra import GHS_WAVELENGTHS, N_GHS_BANDS, FULL_LAMBDA, valid_band_mask, assert_ghs_grid


def test_ghs_band_count():
    assert_ghs_grid()
    assert N_GHS_BANDS == 1721
    assert len(GHS_WAVELENGTHS) == 1721
    assert len(FULL_LAMBDA) == 2101


def test_water_vapor_dropped():
    mask = valid_band_mask()
    dropped = FULL_LAMBDA[~mask]
    assert dropped.min() >= 1351
    assert 1400 in set(dropped.astype(int))
    assert 1900 in set(dropped.astype(int))
    assert 400 in set(GHS_WAVELENGTHS.astype(int))
    assert 800 in set(GHS_WAVELENGTHS.astype(int))
