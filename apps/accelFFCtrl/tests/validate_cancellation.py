#!/usr/bin/env python3
r"""\file validate_cancellation.py
\brief Score and plot offline synthetic DM cancellation; no native stream or application access.
\ingroup accelFFCtrl_unit_test

Trace float64 columns: relative time, truth[2], DM effect[2], residual[2], modal[2],
prediction ready, proposal state, predicted target[2], target error[2], query flag.
All traces are uniform 8 kHz; unavailable predictions remain explicit zeros at
the command input, with delayed/decaying physical response retained in residuals.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from validate_predictions import RAD_TO_MAS

RATE = 8000
SETTLE = 4


def coherent(values, times, frequency):
    """Return peak cos(omega*t+phase) phasor; records cover integer 28/56-Hz cycles."""
    values, times = np.asarray(values), np.asarray(times)
    return 2*np.mean(values*np.exp(-2j*np.pi*frequency*times))


def spectral_suppression(before, after, frequency, rate=RATE):
    """Compare uniform same-time PSDs, reporting both the exact carrier bin and its +/-1-Hz band."""
    options = dict(fs=rate, nperseg=min(8*rate, len(before)), detrend="constant")
    f, original = signal.welch(before, **options)
    _, residual = signal.welch(after, **options)
    nearest = np.argmin(abs(f-frequency))
    band = abs(f-frequency) <= 1
    with np.errstate(divide="ignore", invalid="ignore"):
        peak = 10*np.log10(original[nearest]/residual[nearest])
        integrated = 10*np.log10(np.sum(original[band])/np.sum(residual[band]))
    return dict(bin_frequency_hz=f[nearest], bin_suppression_db=peak, band_suppression_db=integrated), (f, original, residual)


def present(population, frequency):
    """Distinguish requested modes from unrelated tails/leakage of the isolated source."""
    mode = int(population.split("_")[1])
    return mode == 2 or frequency == (28 if mode == 0 else 56)


def analyze(directory, plots=True):
    """Cross-check C++ full-rate metrics, export PSDs and compare causal runs with explicitly separate oracles."""
    root = Path(directory)
    report = root/"report"
    report.mkdir(exist_ok=True)
    summary = pd.read_csv(root/"summary.csv")
    if summary.empty or not np.all(np.isfinite(summary.before_rms_rad)):
        raise ValueError("incomplete cancellation summary")
    zeros = summary[summary.gain == 0]
    np.testing.assert_array_equal(zeros.before_rms_rad, zeros.after_rms_rad)
    np.testing.assert_array_equal(zeros.modal_peak, np.zeros(len(zeros)))
    if not np.all(summary.modal_peak <= summary.modal_bound*(1+1e-12)):
        raise AssertionError("modal proposal exceeded its synthetic bound")
    summary["before_rms_mas"] = summary.before_rms_rad*RAD_TO_MAS
    summary["after_rms_mas"] = summary.after_rms_rad*RAD_TO_MAS
    summary["prediction_rmse_mas"] = summary.prediction_rmse_rad*RAD_TO_MAS
    summary = summary[[present(p, f) for p, f in zip(summary.population, summary.frequency)]]
    summary.to_csv(report/"metrics.csv", index=False)
    psd_rows = []
    traces = {}
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    for name, run in summary.groupby("run", sort=False):
        filename = run.iloc[0].trace
        if filename == "-":
            continue
        data = np.memmap(root/filename, dtype="<f8", mode="r")
        if data.size % 16:
            raise ValueError(f"partial trace: {filename}")
        data = data.reshape(-1, 16)
        np.testing.assert_allclose(np.diff(data[:, 0]), 1/RATE, atol=1e-14, rtol=0)
        np.testing.assert_allclose(data[:, 5:7], data[:, 1:3]+data[:, 3:5], atol=1e-22, rtol=1e-12)
        inhibited = np.isin(data[:, 10], (0, 1))
        np.testing.assert_array_equal(data[inhibited, 7:9], np.zeros((np.sum(inhibited), 2)))
        scored = data[:, 0] >= SETTLE
        queries = scored & (data[:, 15] == 1) & (data[:, 9] == 1)
        traces[name] = data
        for axis in (0, 1):
            rows = run[run.axis == axis]
            before, after = data[scored, 1+axis], data[scored, 5+axis]
            np.testing.assert_allclose(np.sqrt(np.mean(after**2)), rows.iloc[0].after_rms_rad, rtol=2e-10)
            np.testing.assert_allclose(np.sqrt(np.mean(before**2)), rows.iloc[0].before_rms_rad, rtol=2e-10)
            if np.any(queries):
                np.testing.assert_allclose(np.sqrt(np.mean(data[queries, 13+axis]**2)),
                                           rows.iloc[0].prediction_rmse_rad, rtol=2e-10, atol=1e-24)
            for row in rows.itertuples():
                values, spectrum = spectral_suppression(before, after, row.frequency)
                psd_rows.append(dict(run=name, axis=axis, frequency=row.frequency, **values))
                original = coherent(before, data[scored, 0], row.frequency)
                residual = coherent(after, data[scored, 0], row.frequency)
                ratio = residual/original
                np.testing.assert_allclose(abs(ratio), row.residual_amplitude_ratio, rtol=1e-8, atol=1e-12)
                phase_error = np.remainder(np.angle(ratio, deg=True)-row.residual_phase_deg+180, 360)-180
                if abs(ratio) > 1e-10:
                    assert abs(phase_error) < 1e-6
            f, original_psd, residual_psd = spectrum
            pd.DataFrame(dict(hz=f, before_mas2_per_hz=original_psd*RAD_TO_MAS**2,
                              after_mas2_per_hz=residual_psd*RAD_TO_MAS**2)).to_csv(
                                  report/f"{name}_axis{axis}_psd.csv", index=False)
        if plots:
            figure, axes = plt.subplots(4, 1, figsize=(11, 12))
            selected = (data[:, 0] >= 4) & (data[:, 0] < 4.3)
            for axis, color in ((0, "tab:blue"), (1, "tab:orange")):
                axes[0].plot(data[selected, 0], data[selected, 1+axis]*RAD_TO_MAS, color=color,
                             linewidth=.8, label=f"Original {'XY'[axis]}")
                axes[0].plot(data[selected, 0], data[selected, 5+axis]*RAD_TO_MAS, color=color,
                             linestyle="--", linewidth=.8, label=f"Residual {'XY'[axis]}")
                _, (f, before, after) = spectral_suppression(data[scored, 1+axis], data[scored, 5+axis], 28)
                axes[1].semilogy(f, before*RAD_TO_MAS**2, color=color, linewidth=.8)
                axes[1].semilogy(f, after*RAD_TO_MAS**2, color=color, linestyle="--", linewidth=.8)
                axes[2].plot(data[selected, 0], data[selected, 7+axis], label=f"Mode {axis}")
            axes[0].set_ylabel("Pointing [mas]")
            axes[0].legend(ncol=4)
            axes[1].set(xlim=(10, 100), ylabel="PSD [mas²/Hz]", xlabel="Frequency [Hz]")
            axes[2].set(ylabel="Modal coefficient", xlabel="Time [s]")
            axes[2].legend()
            view = (data[:, 0] >= 4.9) & (data[:, 0] < 5.6)
            axes[3].plot(data[view, 0], data[view, 9], label="Prediction ready")
            axes[3].plot(data[view, 0], data[view, 10]/3, label="Proposal state / 3", linestyle="--")
            axes[3].set(xlabel="Time [s]", ylabel="Readiness/state")
            axes[3].legend()
            figure.suptitle(f"OFFLINE SYNTHETIC DM — {name}")
            figure.tight_layout()
            figure.savefig(report/f"{name}.png")
            plt.close(figure)
    pd.DataFrame(psd_rows).to_csv(report/"spectral_suppression.csv", index=False)
    if plots:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for population, axis in enumerate(axes):
            label = f"ar_{population}"
            rows = summary[(summary.population == label) & summary.run.str.match(label+r"_gain\d+$")]
            oracle = traces[f"{label}_oracle"]
            scored = oracle[:, 0] >= SETTLE
            for channel in (0, 1):
                points = rows[rows.axis == channel].drop_duplicates("gain").sort_values("gain")
                axis.plot(points.gain, points.variance_reduction_pct, marker=".", label=f"Predictor {'XY'[channel]}")
                gains = points.gain.to_numpy()
                truth, effect = oracle[scored, 1+channel], oracle[scored, 3+channel]
                ideal = [100*(1-np.var(truth+gain*effect)/np.var(truth)) for gain in gains]
                axis.plot(gains, ideal, linestyle="--", label=f"Perfect prediction {'XY'[channel]}")
            axis.set(title=("28 Hz", "56 Hz", "28 + 56 Hz")[population], xlabel="Feed-forward gain")
            axis.grid(alpha=.3)
        axes[0].set_ylabel("Variance reduction [%]")
        axes[-1].legend(fontsize=8)
        figure.suptitle("OFFLINE SYNTHETIC DM — same latency/bandwidth; perfect-prediction oracle shown separately")
        figure.tight_layout()
        figure.savefig(report/"gain_sweep.png")
        plt.close(figure)
    with (report/"versions.json").open("w") as stream:
        import scipy
        json.dump(dict(numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__), stream, indent=2)
    primary = summary[summary.run.str.match(r"ar_[012]_gain10$")]
    print(primary[["run", "axis", "frequency", "before_rms_mas", "after_rms_mas", "variance_reduction_pct",
                   "prediction_nrmse", "opposing_phase_error_deg"]].to_string(index=False))
    print(f"Checked {summary.run.nunique()} runs; reports: {report}")
    return summary, pd.DataFrame(psd_rows)


def main():
    """Analyze only the caller's existing local offline export directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    analyze(args.directory, not args.no_plots)


if __name__ == "__main__":
    main()
