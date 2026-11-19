#!/usr/bin/env python3
r"""\file validate_psd.py
\brief Offline time-domain and Welch/AR-spectrum plots of the accelSim C++ export.

All analysis stays outside the acquisition loop. Theory is the stationary
ensemble expectation of the same Hann-windowed, constant-detrended estimator;
a high-r mode's individual 60-second realization need not match that expectation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import get_window, welch


PROMINENT_HZ = (11, 14, 17, 20, 25, 28, 54, 56, 58, 75, 85, 120, 220, 300)
DEFAULTS = dict(fps=6000, zero=2048, sensitivity=1.02, volts_per_bit=0.0083799,
                noise_psd=1e-8, lever_arm=10)
ARCSEC_TO_RAD = np.pi / (180 * 3600)


def model_covariances(modes, fps, lever_arm, noise_psd, count):
    """Return stationary lag covariances: pointing [mas], WFS [mas], acceleration [m/s²]."""
    lag = np.arange(count)
    covariance = {name: np.zeros((count, 2)) for name in ("pointing", "wfs", "accel")}
    for mode in np.atleast_1d(modes):
        f, r = mode["frequency"], mode["r"]
        if not (0 < f < fps / 2 and 0 <= r < 1 and mode["accel_rms"] >= 0):
            raise ValueError("Invalid mode frequency, pole radius, or RMS")
        omega = 2 * np.pi * f / fps
        # Yule-Walker: gamma[1]/gamma[0] = 2*r*cos(omega)/(1+r*r).
        correlation = r**lag * (np.cos(lag * omega) +
                               (1 - r) * (1 + r) / (1 + r*r) /
                               np.tan(omega) * np.sin(lag * omega))
        acceleration = mode["accel_rms"]**2 * correlation[:, None]
        mas_factor = (2 * np.pi * f)**2 * lever_arm * ARCSEC_TO_RAD / 1000
        pointing = acceleration / mas_factor**2
        covariance["accel"] += acceleration
        covariance["pointing"] += pointing
        covariance["wfs"] += pointing * np.array([mode["coupling_x"], mode["coupling_y"]])**2
    covariance["accel"][0] += noise_psd * fps / 2
    return covariance


def expected_welch(covariance, fps):
    """Compute the exact ensemble mean Hann/detrended periodogram from stationary lags.

    Window autocorrelation weights each covariance lag. Subtracting the segment
    mean adds the two cross terms and the mean-variance term below. Averaging
    overlapping stationary segments changes uncertainty, not this expectation.
    No dense covariance matrix or numerical simulation is needed.
    """
    count = covariance.shape[0]
    window = get_window("hann", count, fftbins=True)
    window_correlation = np.fft.irfft(abs(np.fft.rfft(window, n=2*count))**2,
                                      n=2*count)[:count]
    weighted = covariance * window_correlation[:, None]
    power = 2 * np.fft.rfft(weighted, axis=0).real - weighted[0]
    prefix = np.cumsum(covariance, axis=0)
    row_sums = prefix + prefix[::-1] - covariance[0]  # Covariance matrix times the all-ones vector.
    window_fft = np.fft.rfft(window)[:, None]
    cross = np.fft.rfft(window[:, None] * row_sums, axis=0) / count
    power -= 2 * (window_fft.conj() * cross).real
    power += abs(window_fft)**2 * np.sum(row_sums, axis=0) / count**2
    power /= fps * np.sum(window**2)
    if count % 2 == 0:
        power[1:-1] *= 2  # Do not double DC or Nyquist.
    else:
        power[1:] *= 2
    return np.maximum(power, 0)  # Remove tiny negative roundoff in otherwise nonnegative spectra.


def rms(values):
    """Return per-axis root mean square without removing the record mean."""
    return np.sqrt(np.mean(values**2, axis=0)).tolist()


def plot_record(directory, time, signals, frequency, spectra, theory, start, seconds, show):
    """Save five diagnostic figures; optionally display the same figures interactively."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    colors = ("tab:blue", "tab:orange")
    selected = (time >= start) & (time < start + seconds)
    if np.count_nonzero(selected) < 2:
        raise ValueError("Requested time window contains fewer than two samples")
    paths = []
    with plt.rc_context({"font.size": 10, "axes.grid": True, "grid.alpha": 0.18}):
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
        for axis, label in enumerate(("X", "Y")):
            axes[axis].plot(time[selected], signals["pointing"][selected, axis], color=colors[axis], lw=0.8)
            axes[axis].set_ylabel(f"{label} pointing [mas]")
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle("Physical pointing — same C++ model as accelSim")
        paths.append(directory / "pointing_timeseries.png")
        fig.savefig(paths[-1], dpi=160)

        kinds = (("pointing", "Physical pointing", "mas²/Hz"),
                 ("wfs", "WFS-coupled pointing", "mas²/Hz"),
                 ("accel", "Acceleration including sensor noise", "(m/s²)²/Hz"),
                 ("adc", "ADC-derived acceleration vs. pre-ADC acceleration", "(m/s²)²/Hz"))
        positive = frequency > 0
        for name, title, units in kinds:
            fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
            for axis, label in enumerate(("X", "Y")):
                ax = axes[axis]
                ax.loglog(frequency[positive], spectra[name][positive, axis], color=colors[axis],
                          lw=0.9, label=f"{label} realization (Welch)")
                if name == "adc":
                    ax.loglog(frequency[positive], spectra["accel"][positive, axis], color="0.45",
                              lw=0.7, alpha=0.7, label="Pre-ADC realization")
                prediction = theory["accel" if name == "adc" else name]
                ax.loglog(frequency[positive], prediction[positive, axis], "k--", lw=1,
                          label="Pre-ADC ensemble theory" if name == "adc" else "Ensemble theory (same estimator)")
                for index, f in enumerate(PROMINENT_HZ):
                    if f < frequency[-1]:
                        ax.axvline(f, color="0.5", lw=0.5, ls=":", alpha=0.65)
                        ax.text(f, 0.98 - (index % 3) * 0.085, str(f), transform=ax.get_xaxis_transform(),
                                rotation=90, va="top", ha="right", fontsize=7, color="0.35")
                ax.set_xlim(max(1, frequency[1]), frequency[-1])
                ax.set_ylabel(f"{label} PSD [{units}]")
                ax.legend(loc="lower left", fontsize=8)
            axes[-1].set_xlabel("Frequency [Hz] · dotted lines mark the 14 prominent modes")
            fig.suptitle(title)
            paths.append(directory / f"{name}_psd.png")
            fig.savefig(paths[-1], dpi=160)
        if not show:
            plt.close("all")
    return [str(path) for path in paths]


def analyze(directory, overrides, start=0, seconds=1, show=False):
    """Analyze a C++ export, print RMS/clipping, and save numerical results and plots."""
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    config = {key: metadata.get(key, value) for key, value in DEFAULTS.items()}
    config.update({key: value for key, value in overrides.items() if value is not None})
    if not all(np.isfinite(value) for value in config.values()):
        raise ValueError("Nonfinite analysis configuration")
    if (min(config[key] for key in ("fps", "sensitivity", "volts_per_bit", "lever_arm")) <= 0 or
            config["noise_psd"] < 0 or not np.isfinite([start, seconds]).all() or start < 0 or seconds <= 0):
        raise ValueError("Invalid rate, calibration, noise PSD, or time window")
    fps = config["fps"]
    data = np.atleast_1d(np.genfromtxt(directory / "samples.csv", delimiter=",", names=True))
    modes = np.atleast_1d(np.genfromtxt(directory / "modes.csv", delimiter=",", names=True))
    if data.size < 4 or not all(np.isfinite(data[key]).all() for key in data.dtype.names):
        raise ValueError("Sample export is too short or contains nonfinite values")
    if not modes.size or not all(np.isfinite(modes[key]).all() for key in modes.dtype.names):
        raise ValueError("Mode export is empty or contains nonfinite values")
    if not np.allclose(np.diff(data["time"]), 1 / fps, rtol=1e-6, atol=1e-12):
        raise ValueError("Export timestamps do not match the configured nominal FPS")
    adc = np.column_stack((data["adc_x"], data["adc_y"]))
    if not ((adc >= 0) & (adc <= 4095) & (adc == np.round(adc))).all():
        raise ValueError("Raw ADC samples violate the UINT16/12-bit contract")
    signals = {
        "pointing": np.column_stack((data["pointing_x"], data["pointing_y"])) * 1000,
        "wfs": np.column_stack((data["wfs_x"], data["wfs_y"])) * 1000,
        "accel": np.column_stack((data["accel_x"], data["accel_y"])),
        "adc": (adc - config["zero"]) * config["volts_per_bit"] / config["sensitivity"],
    }
    nperseg = min(int(10 * fps), data.size)
    if nperseg < 4:
        raise ValueError("Welch segment must contain at least four samples")
    spectra = {}
    for name, values in signals.items():
        frequency, spectra[name] = welch(values, fs=fps, window="hann", nperseg=nperseg,
                                         noverlap=nperseg//2, detrend="constant", scaling="density", axis=0)
    covariances = model_covariances(modes, fps, config["lever_arm"], config["noise_psd"], nperseg)
    theory = {name: expected_welch(value, fps) for name, value in covariances.items()}
    # Legacy exports lack exact clamp totals; reconstruct them with C++ half-away-from-zero rounding.
    generation = {key: metadata.get(key, config[key]) for key in ("zero", "sensitivity", "volts_per_bit")}
    counts = generation["zero"] + signals["accel"] * generation["sensitivity"] / generation["volts_per_bit"]
    rounded = np.copysign(np.floor(abs(counts) + 0.5), counts)
    clipping = {"low_xy": metadata.get("clip_low_xy", np.sum(rounded < 0, axis=0).tolist()),
                "high_xy": metadata.get("clip_high_xy", np.sum(rounded > 4095, axis=0).tolist()),
                "source": "C++ model counters" if "clip_low_xy" in metadata else "reconstructed from exported acceleration"}
    results = {
        "samples": int(data.size), "duration_seconds": data.size / fps, "mode_count": int(modes.size),
        "configuration": config, "seed": metadata.get("seed"), "mode_seed": metadata.get("mode_seed"),
        "welch": dict(window="hann", nperseg=int(nperseg), noverlap=int(nperseg//2),
                      detrend="constant", scaling="density"),
        "physical_pointing_rms_mas": rms(signals["pointing"]),
        "wfs_pointing_rms_mas": rms(signals["wfs"]),
        "physical_acceleration_rms": rms(signals["accel"]),
        "adc_acceleration_rms": rms(signals["adc"]),
        "rms_units": {"pointing": "mas", "wfs": "mas", "accel": "m/s^2", "adc": "m/s^2"},
        "stationary_rms_xy": {name: np.sqrt(value[0]).tolist() for name, value in covariances.items()},
        "adc_clipping_counts": clipping,
        "rail_samples_xy": np.sum((adc == 0) | (adc == 4095), axis=0).tolist(),
        "band_power_ratios_xy": {},
        "note": "RMS is sqrt(mean(x²)). High-r finite-record amplitudes need not match ensemble RMS. "
                "ADC theory is pre-ADC; quantization/clipping are not included in that overlay.",
    }
    for low, high in ((100, 200), (200, 400), (400, 900), (1200, min(2800, fps / 2))):
        selected = (frequency >= low) & (frequency < high)
        if selected.any() and np.all(np.sum(theory["accel"][selected], axis=0) > 0):
            results["band_power_ratios_xy"][f"{low}-{high}Hz"] = (
                np.sum(spectra["accel"][selected], axis=0) / np.sum(theory["accel"][selected], axis=0)).tolist()
    columns, names = [frequency], ["Hz"]
    for name in signals:
        for axis, label in enumerate(("x", "y")):
            columns.append(spectra[name][:, axis])
            names.append(f"{name}_{label}")
    for name in theory:
        for axis, label in enumerate(("x", "y")):
            columns.append(theory[name][:, axis])
            names.append(f"{name}_theory_{label}")
    np.savetxt(directory / "psd.csv", np.column_stack(columns), delimiter=",", header=",".join(names), comments="")
    results["plots"] = plot_record(directory / "plots", data["time"], signals, frequency, spectra,
                                   theory, start, seconds, show)
    (directory / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    if show:
        import matplotlib.pyplot as plt
        plt.show()
        plt.close("all")
    return results


def main():
    """Read a numerical export and save plots, optionally opening interactive windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    for key in DEFAULTS:
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=None,
                            help=f"Override export metadata (legacy default: {DEFAULTS[key]})")
    parser.add_argument("--time-start", type=float, default=0, help="Time trace start [s], default 0")
    parser.add_argument("--time-seconds", type=float, default=1, help="Time trace length [s], default 1")
    parser.add_argument("--show", action="store_true", help="Also display figures interactively after saving")
    args = parser.parse_args()
    try:
        analyze(args.directory, {key: getattr(args, key) for key in DEFAULTS}, args.time_start, args.time_seconds, args.show)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Validation failed: {error}\n")


if __name__ == "__main__":
    main()
