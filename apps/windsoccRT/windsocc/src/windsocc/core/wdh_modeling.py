"""
Wind-driven halo (WDH) analytic model and a small demo against raw + PSF science.

The fit uses **raw** science ``raw``, subtracts the median PSF estimate ``psf`` to form
``T = raw - psf``, then finds scalar ``s`` minimizing ``||T - s * WDH||^2``. That is the
same as minimizing the PSF-subtracted image ``(raw - s * WDH) - psf = T - s * WDH``.

This module is experimental; placeholder geometry parameters are meant to be tuned by hand.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter


def load_psf_estimate_fits(path: Path | str) -> np.ndarray:
    """Load 2D PSF data from ``psf_estimate_med.fits`` (or any single-HDU FITS)."""
    path = Path(path)
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D PSF array in {path}, got shape {data.shape}.")
    return data


def match_array_shape_to_reference(
    arr: np.ndarray,
    ref_shape: tuple[int, int],
) -> np.ndarray:
    """Center-crop ``arr`` to ``ref_shape`` if needed (same ndim)."""
    if arr.shape == ref_shape:
        return arr
    ny, nx = ref_shape
    sy, sx = arr.shape
    if sy < ny or sx < nx:
        raise ValueError(
            f"Array shape {arr.shape} is smaller than reference {ref_shape}; cannot crop."
        )
    cy = (sy - ny) // 2
    cx = (sx - nx) // 2
    return np.asarray(arr[cy : cy + ny, cx : cx + nx], dtype=arr.dtype)


def gen_wdh_image(
    x: np.ndarray,
    y: np.ndarray,
    beta: float,
    h0: float,
    sigma: float,
    PA_deg: float,
    x0: float,
    fwhm: float | None,
    gamma: float,
) -> np.ndarray:
    """
    Generate a single WDH-like intensity map on the same grid as ``x`` and ``y``.

    ``PA_deg`` is the position angle (degrees) used to rotate the halo axis; ``x0`` is an
    offset along the rotated radial coordinate (pixels, same units as ``x``, ``y``).
    """
    # At pixel scale ~0.012"/pixel this is the IWA at g', about 4 lambda/D
    r1 = 10.0
    pa_rad = -np.deg2rad(PA_deg)
    x_rot = np.sin(pa_rad) * x + np.cos(pa_rad) * y
    y_rot = np.cos(pa_rad) * x - np.sin(pa_rad) * y

    dx = x_rot - x0
    r = np.sqrt(dx**2 + y_rot**2)
    r_safe = np.maximum(r, 1e-6)

    power_law = (1.0 / r_safe) ** beta
    denom = h0 * (dx**2)
    denom = np.where(np.abs(denom) < 1e-12, np.copysign(1e-12, denom + 1e-30), denom)
    radial_term = (r**2 / denom) ** gamma
    exp_term = np.exp(-0.5 * (radial_term + (x_rot / sigma) ** 2))
    i_map = power_law * exp_term
    i_map = np.nan_to_num(i_map, nan=0.0, posinf=0.0, neginf=0.0)
    i_map = np.where(r < r1, 0.0, i_map)

    if fwhm is not None and fwhm > 0:
        # Blur in pixels (FWHM -> sigma); use separable Gaussian (fast vs full-image kernel).
        sigma_pix = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        i_map = gaussian_filter(i_map, sigma=sigma_pix, mode="constant")
    return i_map.astype(np.float32, copy=False)


def _parse_json_constant(x: str) -> float:
    if x == "NaN":
        return float("nan")
    if x == "Infinity":
        return float("inf")
    if x == "-Infinity":
        return float("-inf")
    raise ValueError(f"Unexpected constant {x!r}")


def load_wind_tracks_json(path: Path | str) -> list[dict[str, Any]]:
    """Load a windsocc ``*_wind_attributes.json`` file (may contain non-standard ``NaN``)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text, parse_constant=_parse_json_constant)
    tracks = data.get("tracks")
    if tracks is None:
        raise KeyError(f"No 'tracks' key in {path}")
    return list(tracks)


def aggregate_tracks_for_model(
    tracks: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect ``corrected_direction`` and flux per track; normalize flux by max (largest = 1).

    If all flux values are non-positive or non-finite, weights are uniform.
    """
    if not tracks:
        raise ValueError("No tracks to aggregate.")

    pa_deg = np.array([float(t["corrected_direction"]) for t in tracks], dtype=np.float64)
    fluxes = np.array([float(t["flux"]) for t in tracks], dtype=np.float64)

    max_flux = np.nanmax(fluxes)
    if not np.isfinite(max_flux) or max_flux <= 0:
        flux_norm = np.ones_like(fluxes) / len(fluxes)
    else:
        flux_norm = fluxes / max_flux

    return pa_deg, flux_norm


def fit_wdh_scale_least_squares(
    science: np.ndarray,
    wdh_model: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """
    Find scalar ``s`` that minimizes ``sum (science - s * wdh_model)^2`` over finite pixels.

    Returns ``(scale, residual, rms)`` where ``residual = science - scale * wdh_model`` and
    ``rms`` is the root-mean-square of the residual over the same pixels.

    For raw FITS + a PSF map, use :func:`fit_wdh_scale_from_raw_and_psf` so ``science`` is
    ``raw - psf`` implicitly.
    """
    s = np.asarray(science, dtype=np.float64)
    m = np.asarray(wdh_model, dtype=np.float64)
    valid = np.isfinite(s) & np.isfinite(m)
    num = float(np.sum(s[valid] * m[valid]))
    den = float(np.sum(m[valid] ** 2))
    if not np.isfinite(den) or den <= 0.0:
        scale = 0.0
    else:
        scale = num / den
    residual = s - scale * m
    residual = np.where(np.isfinite(residual), residual, np.nan)
    rms = float(np.sqrt(np.nanmean(residual**2)))
    return scale, residual.astype(np.float32), rms


def fit_wdh_scale_from_raw_and_psf(
    raw_science: np.ndarray,
    wdh_model: np.ndarray,
    psf_estimate: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> tuple[float, np.ndarray, float, list[dict[str, float]]]:
    """
    Find ``s`` minimizing ``||(raw - psf) - s * wdh_model||^2``.

    Algebraically ``(raw - s * wdh) - psf`` equals ``(raw - psf) - s * wdh``, so the
    PSF-subtracted residual after subtracting the scaled WDH from raw and then the PSF is
    ``T - s * wdh`` with ``T = raw - psf``.

    Newton iterations minimize the quadratic in ``s`` until ``|Δs| < tol``.

    Returns ``(scale, residual, rms, history)`` with
    ``residual = (raw - psf) - s * wdh_model``.
    """
    raw = np.asarray(raw_science, dtype=np.float64)
    m = np.asarray(wdh_model, dtype=np.float64)
    p = np.asarray(psf_estimate, dtype=np.float64)
    if raw.shape != m.shape or raw.shape != p.shape:
        raise ValueError(
            f"Shape mismatch: raw {raw.shape}, wdh_model {m.shape}, psf {p.shape}."
        )
    t = raw - p
    valid = np.isfinite(t) & np.isfinite(m)
    den = float(np.sum(m[valid] ** 2))
    history: list[dict[str, float]] = []
    if not np.isfinite(den) or den <= 0.0:
        scale = 0.0
        residual = t - scale * m
        rms = float(np.sqrt(np.nanmean(residual**2)))
        history.append(
            {
                "iter": 0.0,
                "scale": scale,
                "rss": float(np.nansum(residual**2)),
                "rms": rms,
            }
        )
        return scale, residual.astype(np.float32), rms, history

    max_iter = max(1, max_iter)
    scale = 1e5
    residual = t - scale * m
    for k in range(max_iter):
        r = t[valid] - scale * m[valid]
        grad = -2.0 * float(np.sum(m[valid] * r))
        hess = 2.0 * den
        scale_new = scale - grad / hess
        residual = t - scale_new * m
        rss = float(np.sum(residual[valid] ** 2))
        rms = float(np.sqrt(np.mean(residual[valid] ** 2)))
        history.append(
            {"iter": float(k), "scale": float(scale_new), "rss": rss, "rms": rms}
        )
        if abs(scale_new - scale) < tol:
            scale = float(scale_new)
            break
        scale = float(scale_new)

    residual = np.where(np.isfinite(residual), residual, np.nan)
    rms = float(np.sqrt(np.nanmean(residual**2)))
    return scale, residual.astype(np.float32), rms, history


def _save_or_show_figure(fig, basename: str) -> None:
    plt.tight_layout()
    if matplotlib.get_backend().lower() == "agg":
        out_path = Path(tempfile.gettempdir()) / basename
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure to {out_path} (non-interactive backend).")
    else:
        plt.show()


def _demo_plot(
    science: np.ndarray,
    science_psf_sub: np.ndarray,
    wdh_model: np.ndarray,
    title_left: str = "Raw science",
    title_middle: str = "PSF-subtracted science",
    title_right: str = "WDH model (sum)",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin = float(np.nanpercentile(science, 0.1))
    vmax = float(np.nanpercentile(science, 99.9))
    im0 = axes[0].imshow(science, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(title_left)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    smin = float(np.nanpercentile(science_psf_sub, 0.1))
    smax = float(np.nanpercentile(science_psf_sub, 99.9))
    im1 = axes[1].imshow(science_psf_sub, origin="lower", cmap="magma", vmin=smin, vmax=smax)
    axes[1].set_title(title_middle)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    mmin = float(np.nanpercentile(wdh_model, 0.1))
    mmax = float(np.nanpercentile(wdh_model, 99.9))
    im2 = axes[2].imshow(wdh_model, origin="lower", cmap="magma", vmin=mmin, vmax=mmax)
    axes[2].set_title(title_right)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    _save_or_show_figure(fig, "wdh_model_demo.png")


def _plot_scaled_fit(
    science_psf_sub: np.ndarray,
    scaled_wdh: np.ndarray,
    scale: float,
    residual: np.ndarray,
    rms: float,
) -> None:
    """``raw - psf``, ``s * WDH``, and residual ``(raw - psf) - s * WDH``."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    vmin = float(np.nanpercentile(science_psf_sub, 5))
    vmax = float(np.nanpercentile(science_psf_sub, 99.5))
    im0 = axes[0].imshow(science_psf_sub, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("PSF-subtracted (raw − PSF)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(scaled_wdh, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Scaled WDH (s = {scale:.6g})")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    lim = float(np.nanpercentile(np.abs(residual), 99.0))
    if not np.isfinite(lim) or lim <= 0:
        lim = float(np.nanmax(np.abs(residual)) or 1.0)
    im2 = axes[2].imshow(
        residual,
        origin="lower",
        cmap="coolwarm",
        vmin=-lim,
        vmax=lim,
    )
    axes[2].set_title(f"Residual (RMS = {rms:.6g})")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    _save_or_show_figure(fig, "wdh_model_fit_residual.png")


if __name__ == "__main__":
    # Default test paths (HR 4796 run); override by editing or importing.
    _DEFAULT_RAW = Path(
        "/Users/jkueny/data/HR4796_rg_smlyot_20230312_13/raws_rg_smlyot_20230313T043914/"
        "camsci1/lite_psflib/camsci1_2x2bin_60.5_parang.fits"
    )
    _DEFAULT_JSON = Path(
        "/Users/jkueny/data/HR4796_rg_smlyot_20230312_13/raws_rg_smlyot_20230313T043914/"
        "camwfs/measure_results/wind_data/"
        "camwfs_20230313071325997167000_mf_response_unsharp_wind_attributes.json"
    )

    # Placeholder geometry shared across layers (hand-tune).
    _BETA = 1.0
    _H0 = 15.0
    _SIGMA = 30.0
    _X0 = 0.0
    _FWHM = 3.0
    _GAMMA = 1.0

    tracks = load_wind_tracks_json(_DEFAULT_JSON)
    pa_deg, flux_norm = aggregate_tracks_for_model(tracks)

    with fits.open(_DEFAULT_RAW) as hdul:
        raw = np.asarray(hdul[0].data, dtype=np.float32)
    ny, nx = raw.shape
    # Pixel coordinates centered on the image (same units as x0 in gen_wdh_image).
    y_pix, x_pix = np.indices((ny, nx), dtype=np.float32)
    cx = (nx - 1) * 0.5
    cy = (ny - 1) * 0.5
    x = x_pix - cx
    y = y_pix - cy

    
    for w, pa in zip(flux_norm, pa_deg):
        print(f"Using flux_norm: {w} and pa_deg: {pa}")

    wdh_model = np.zeros_like(raw, dtype=np.float32)
    for pa, w in zip(pa_deg, flux_norm):
        wdh_model += w * gen_wdh_image(
            x,
            y,
            beta=_BETA,
            h0=_H0,
            sigma=_SIGMA,
            PA_deg=float(pa),
            x0=_X0,
            fwhm=_FWHM,
            gamma=_GAMMA,
        )

    _psf_path = _DEFAULT_RAW.parent / "psf_subtracted" / "psf_estimate_med.fits"
    if not _psf_path.is_file():
        raise FileNotFoundError(
            f"PSF estimate not found: {_psf_path}. Run do_crude_psf_subtraction.py first "
            "to write psf_subtracted/psf_estimate_med.fits."
        )
    psf_estimate = load_psf_estimate_fits(_psf_path)
    psf_estimate = match_array_shape_to_reference(psf_estimate, raw.shape)

    _demo_plot(raw, raw - psf_estimate, wdh_model, title_left="Raw science", title_right="WDH model (sum)")
    exit()
    scale, residual, rms, history = fit_wdh_scale_from_raw_and_psf(
        raw, wdh_model, psf_estimate
    )
    print(
        "Newton iterations (minimize ||(raw − PSF) − s·WDH||_2, same as "
        "||(raw − s·WDH) − PSF||_2):"
    )
    for row in history:
        print(
            f"  iter {int(row['iter'])}: scale={row['scale']:.8g} "
            f"RSS={row['rss']:.6g} RMS={row['rms']:.6g}"
        )
    print(f"Final scale: {scale:.8g}")
    print(f"Final RMS residual (PSF-subtracted): {rms:.8g}")

    science_psf_sub = raw - psf_estimate
    scaled_wdh = scale * wdh_model

    _plot_scaled_fit(science_psf_sub, scaled_wdh, scale, residual, rms)
