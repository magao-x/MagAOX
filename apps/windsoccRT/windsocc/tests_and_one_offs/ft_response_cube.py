"""Display the Fourier transform of the first slice of a FITS cube."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from skimage.feature import peak_local_max


DEFAULT_FITS_PATH = Path(
    "/Users/jkueny/source/MagAOX/apps/windsoccRT/exao2/"
    "camwfs_20260503T033720867231/distill_results/"
    "camwfs_20260503T033720867231_00000_unsharp.fits"
)
SIGMA_HIGH = 3.0
PEAK_MIN_DISTANCE = 8
PEAK_ZERO_RADIUS = 3


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Display the Fourier transform magnitude of the first FITS cube slice."
    )
    parser.add_argument(
        "fits_path",
        nargs="?",
        type=Path,
        default=DEFAULT_FITS_PATH,
        help=f"Path to the FITS cube. Default: {DEFAULT_FITS_PATH}",
    )
    parser.add_argument(
        "--sigma-high",
        type=float,
        default=SIGMA_HIGH,
        help="High-side sigma threshold for identifying FFT peaks.",
    )
    parser.add_argument(
        "--peak-min-distance",
        type=int,
        default=PEAK_MIN_DISTANCE,
        help="Minimum pixel spacing for local FFT peak detection.",
    )
    parser.add_argument(
        "--peak-zero-radius",
        type=int,
        default=PEAK_ZERO_RADIUS,
        help="Radius, in pixels, to zero around each detected local FFT peak.",
    )
    return parser.parse_args()


def high_sigma_mask(values: np.ndarray, sigma_high: float) -> np.ndarray:
    """Return a mask for values above the high-side sigma threshold."""
    finite_values = values[np.isfinite(values)]
    threshold = np.mean(finite_values) + sigma_high * np.std(finite_values)
    return values > threshold


def disk_mask(shape: tuple[int, int], centers: np.ndarray, radius: int) -> np.ndarray:
    """Return a mask with circular regions around each center."""
    mask = np.zeros(shape, dtype=bool)
    yy, xx = np.ogrid[: shape[0], : shape[1]]

    for center_y, center_x in centers:
        mask |= (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2

    return mask


def peak_finder_mask(
    values: np.ndarray,
    sigma_high: float,
    min_distance: int,
    zero_radius: int,
) -> np.ndarray:
    """Return a mask around local maxima above a high-side sigma threshold."""
    finite_values = values[np.isfinite(values)]
    threshold = np.mean(finite_values) + sigma_high * np.std(finite_values)
    coordinates = peak_local_max(
        values,
        min_distance=min_distance,
        threshold_abs=threshold,
        exclude_border=False,
    )

    return disk_mask(values.shape, coordinates, zero_radius)


def zero_masked_fft(shifted_fft: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a copy of a shifted FFT with masked coefficients zeroed."""
    filtered_fft = shifted_fft.copy()
    filtered_fft[mask] = 0
    return filtered_fft


def display_magnitude(shifted_fft: np.ndarray) -> np.ndarray:
    """Return a display-scaled Fourier magnitude image."""
    return np.log10(1.0 + np.abs(shifted_fft))


def inverse_shifted_fft(shifted_fft: np.ndarray) -> np.ndarray:
    """Return the real image reconstructed from a shifted FFT."""
    return np.fft.ifft2(np.fft.ifftshift(shifted_fft)).real


def main() -> None:
    """Load the first FITS cube slice and display its Fourier magnitude."""
    args = parse_args()

    with fits.open(args.fits_path, memmap=True) as hdul:
        cube = hdul[0].data
        if cube is None:
            raise ValueError(f"No data found in primary HDU: {args.fits_path}")
        if cube.ndim < 2:
            raise ValueError(f"Expected at least 2D FITS data, got shape {cube.shape}")

        image = np.asarray(cube[0] if cube.ndim > 2 else cube, dtype=float)

    shifted_fft = np.fft.fftshift(np.fft.fft2(np.nan_to_num(image)))
    ft_magnitude = np.abs(shifted_fft)
    sigma_mask = high_sigma_mask(ft_magnitude, args.sigma_high)
    peak_mask = peak_finder_mask(
        ft_magnitude,
        args.sigma_high,
        args.peak_min_distance,
        args.peak_zero_radius,
    )

    sigma_zeroed_fft = zero_masked_fft(shifted_fft, sigma_mask)
    peak_zeroed_fft = zero_masked_fft(shifted_fft, peak_mask)

    print(f"FT magnitude shape: {ft_magnitude.shape}")
    print(f"Sigma mask zeroed coefficients: {np.count_nonzero(sigma_mask)}")
    print(f"Peak finder zeroed coefficients: {np.count_nonzero(peak_mask)}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    vmin = np.percentile(image, 1)
    vmax = np.percentile(image, 99)

    original_display = display_magnitude(shifted_fft)
    peak_display = display_magnitude(peak_zeroed_fft)
    peak_filtered_image = inverse_shifted_fft(peak_zeroed_fft)
    ft_vmin = np.percentile(original_display, 1)
    ft_vmax = np.percentile(original_display, 99)
    filtered_vmin = np.percentile(peak_filtered_image, 1)
    filtered_vmax = np.percentile(peak_filtered_image, 99)

    image_plot = axes[0, 0].imshow(
        image,
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 0].set_title("First FITS Slice")
    axes[0, 0].set_xlabel("x [px]")
    axes[0, 0].set_ylabel("y [px]")
    fig.colorbar(image_plot, ax=axes[0, 0], fraction=0.046, pad=0.04)

    ft_plot = axes[0, 1].imshow(
        original_display,
        origin="lower",
        cmap="magma",
        vmin=ft_vmin,
        vmax=ft_vmax,
    )
    axes[0, 1].set_title("Original log10(1 + |FFT|)")
    axes[0, 1].set_xlabel("kx [cycles/image]")
    axes[0, 1].set_ylabel("ky [cycles/image]")
    fig.colorbar(ft_plot, ax=axes[0, 1], fraction=0.046, pad=0.04)

    peak_plot = axes[1, 0].imshow(
        peak_display,
        origin="lower",
        cmap="magma",
        vmin=ft_vmin,
        vmax=ft_vmax,
    )
    axes[1, 0].set_title("Peak Finder Zeroed Spectrum")
    axes[1, 0].set_xlabel("kx [cycles/image]")
    axes[1, 0].set_ylabel("ky [cycles/image]")
    fig.colorbar(peak_plot, ax=axes[1, 0], fraction=0.046, pad=0.04)

    filtered_plot = axes[1, 1].imshow(
        peak_filtered_image,
        origin="lower",
        cmap="gray",
        vmin=filtered_vmin,
        vmax=filtered_vmax,
    )
    axes[1, 1].set_title("Inverse FFT After Peak Zeroing")
    axes[1, 1].set_xlabel("x [px]")
    axes[1, 1].set_ylabel("y [px]")
    fig.colorbar(filtered_plot, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(args.fits_path.name)
    plt.show()


if __name__ == "__main__":
    main()
