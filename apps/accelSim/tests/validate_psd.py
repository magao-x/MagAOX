#!/usr/bin/env python3
r"""\file validate_psd.py
\brief Offline Welch/AR-spectrum comparison for the accelSim test export.

This is a validation tool, never part of the acquisition loop. Short-record RMS
of the very high-r modes is deliberately not forced to its ensemble target.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch


def analyze(directory, fps, zero, sensitivity, volts_per_bit, noise_psd):
    """Compare physical and raw-ADC sample spectra against the resolved mode table."""
    data = np.genfromtxt(directory / "samples.csv", delimiter=",", names=True)
    modes = np.genfromtxt(directory / "modes.csv", delimiter=",", names=True)
    if data.size < 2 or not all(np.isfinite(data[key]).all() for key in data.dtype.names):
        raise ValueError("Sample export is empty or contains nonfinite values")
    if not np.allclose(np.diff(data["time"]), 1 / fps, rtol=1e-6, atol=1e-12):
        raise ValueError("Export timestamps do not match the configured nominal FPS")
    nperseg = min(int(10 * fps), data.size)
    settings = dict(fs=fps, window="hann", nperseg=nperseg,
                    detrend="constant", scaling="density")
    frequency, px = welch(data["accel_x"], **settings)
    _, py = welch(data["accel_y"], **settings)
    adc = np.column_stack((data["adc_x"], data["adc_y"]))
    if not ((adc >= 0) & (adc <= 4095) & (adc == np.round(adc))).all():
        raise ValueError("Raw ADC samples violate the UINT16/12-bit contract")
    # Offline inverse calibration only; the production consumer receives raw counts.
    calibrated = (adc - zero) * volts_per_bit / sensitivity
    _, raw_x = welch(calibrated[:, 0], **settings)
    _, raw_y = welch(calibrated[:, 1], **settings)
    expected = np.full(frequency.shape, noise_psd)
    z = np.exp(-2j * np.pi * frequency / fps)
    for mode in np.atleast_1d(modes):
        r, f = mode["r"], mode["frequency"]
        omega = 2 * np.pi * f / fps
        d = (1 - r) * (1 + r)
        b = d * d + 4 * r * r * np.sin(omega) ** 2
        q = mode["accel_rms"] ** 2 * d * b / (1 + r * r)
        denominator = np.abs(1 - 2 * r * np.cos(omega) * z + r * r * z * z) ** 2
        expected += 2 * q / fps / denominator
    expected[0] /= 2
    if frequency[-1] == fps / 2:
        expected[-1] /= 2
    results = {
        "samples": int(data.size),
        "mode_count": int(modes.size),
        "physical_acceleration_rms": [float(np.std(data[key])) for key in ("accel_x", "accel_y")],
        "stationary_acceleration_rms": float(np.sqrt(np.sum(modes["accel_rms"] ** 2) + noise_psd * fps / 2)),
        "rail_samples_xy": np.sum((adc == 0) | (adc == 4095), axis=0).tolist(),
        "band_power_ratios_xy": {},
        "note": "High-r short-record RMS and individual narrow peaks are not pass/fail criteria.",
    }
    # Broad bands avoid demanding a particular random amplitude from high-r low modes.
    for low, high in ((100, 200), (200, 400), (400, 900), (1200, min(2800, fps / 2))):
        selected = (frequency >= low) & (frequency < high)
        if selected.any():
            denominator = np.sum(expected[selected])
            results["band_power_ratios_xy"][f"{low}-{high}Hz"] = [
                float(np.sum(power[selected]) / denominator) for power in (px, py)
            ]
    np.savetxt(directory / "psd.csv", np.column_stack((frequency, px, py, raw_x, raw_y, expected)),
               delimiter=",", header="Hz,physical_x,physical_y,calibrated_adc_x,calibrated_adc_y,stationary_theory",
               comments="")
    (directory / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def main():
    """Read a numerical export and write reproducible offline validation artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--fps", type=float, default=6000)
    parser.add_argument("--zero", type=float, default=2048)
    parser.add_argument("--sensitivity", type=float, default=1.02)
    parser.add_argument("--volts-per-bit", type=float, default=0.0083799)
    parser.add_argument("--noise-psd", type=float, default=1e-8)
    args = parser.parse_args()
    analyze(args.directory, args.fps, args.zero, args.sensitivity, args.volts_per_bit, args.noise_psd)


if __name__ == "__main__":
    main()
