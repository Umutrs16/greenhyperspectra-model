"""GreenHyperSpectra 1 nm grid, water-vapor masking, and sensor projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FULL_LAMBDA = np.arange(400, 2501, dtype=np.float32)  # 2101 bands
DROP_RANGES = ((1351, 1430), (1801, 2050), (2451, 2500))
PRIMARY_TRAITS = ("LAI", "Cab", "Cw", "Cm", "Cp")
ALL_TRAITS = PRIMARY_TRAITS + ("Car", "Anth")


def valid_band_mask(
    wavelengths: np.ndarray | None = None,
    drop_ranges: tuple[tuple[int, int], ...] = DROP_RANGES,
) -> np.ndarray:
    wl = FULL_LAMBDA if wavelengths is None else np.asarray(wavelengths)
    keep = np.ones(len(wl), dtype=bool)
    for lo, hi in drop_ranges:
        keep &= ~((wl >= lo) & (wl <= hi))
    return keep


GHS_MASK = valid_band_mask()
GHS_WAVELENGTHS = FULL_LAMBDA[GHS_MASK]
N_GHS_BANDS = int(GHS_MASK.sum())  # 1721
GHS_FWHM = np.full(N_GHS_BANDS, 1.0, dtype=np.float32)


def assert_ghs_grid() -> None:
    if N_GHS_BANDS != 1721:
        raise RuntimeError(f"Expected 1721 GreenHyperSpectra bands, got {N_GHS_BANDS}")


@dataclass(frozen=True)
class SpectralGrid:
    wavelengths: np.ndarray
    fwhm: np.ndarray
    valid: np.ndarray

    @classmethod
    def greenhyperspectra(cls) -> "SpectralGrid":
        assert_ghs_grid()
        return cls(GHS_WAVELENGTHS.copy(), GHS_FWHM.copy(), np.ones(N_GHS_BANDS, dtype=bool))

    @classmethod
    def emit_native(cls, wavelengths: np.ndarray, fwhm: np.ndarray, valid: np.ndarray | None = None) -> "SpectralGrid":
        wl = np.asarray(wavelengths, dtype=np.float32)
        fw = np.asarray(fwhm, dtype=np.float32)
        if valid is None:
            valid = valid_band_mask(wl)
        return cls(wl, fw, np.asarray(valid, dtype=bool))

    def __len__(self) -> int:
        return int(len(self.wavelengths))


def to_ghs_grid(spectrum_2101: np.ndarray) -> np.ndarray:
    spec = np.asarray(spectrum_2101)
    if spec.shape[-1] != 2101:
        raise ValueError(f"Expected last dim 2101, got {spec.shape}")
    return spec[..., GHS_MASK]


def gaussian_srf_project(
    highres_r: np.ndarray,
    highres_wl: np.ndarray,
    centers: np.ndarray,
    fwhm: np.ndarray,
) -> np.ndarray:
    """Bandwidth-aware projection of a 1 nm spectrum onto sensor bands."""
    r = np.asarray(highres_r, dtype=np.float64)
    wl = np.asarray(highres_wl, dtype=np.float64)
    out = np.zeros(len(centers), dtype=np.float64)
    for i, (c, w) in enumerate(zip(centers, fwhm)):
        sigma = max(float(w) / (2.0 * np.sqrt(2.0 * np.log(2.0))), 0.4)
        kernel = np.exp(-0.5 * ((wl - c) / sigma) ** 2)
        if kernel.sum() <= 0:
            out[i] = np.interp(c, wl, r)
        else:
            out[i] = np.dot(kernel, r) / kernel.sum()
    return out.astype(np.float32)


def wavelength_pe(wavelengths: np.ndarray, dim: int) -> np.ndarray:
    """Fourier positional encoding of wavelength in micrometres."""
    wl = np.asarray(wavelengths, dtype=np.float32) / 1000.0
    half = dim // 2
    freqs = (2.0 ** np.arange(half, dtype=np.float32)) * np.pi
    angles = wl[:, None] * freqs[None, :]
    enc = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
    if enc.shape[-1] < dim:
        enc = np.pad(enc, ((0, 0), (0, dim - enc.shape[-1])))
    return enc[:, :dim].astype(np.float32)

