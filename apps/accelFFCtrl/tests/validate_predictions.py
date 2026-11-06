#!/usr/bin/env python3
r"""\file validate_predictions.py
\brief Score offline accelSim truth against raw-only inputProcessor exports; never attach a stream.
\ingroup accelFFCtrl_unit_test

Binary layout v1: little-endian float64, row-major, no headers/padding. Integer
monotonic nanoseconds/counters are exactly representable for these recordings.
Truth columns: time, adc[2], measured_accel[2], clean_accel[2], full_theta[2],
target_accel[2], target_theta[2]. Prediction columns: nominal_time (-1 for a late
query), now, sample_time, requested_target, ready, produced, state, dropped,
resets, processed, accel[2], theta[2], peak_accel[2], phase_at_sample[2], rcond,
sequence, producer_delay_us. Invalid numerical fields must be masked by ready.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

BASE = 1_000_000_000
PERIOD = 125_000
RATE = 2000
OMEGA = 2 * np.pi * 28
RAD_TO_MAS = 180 * 3600 * 1000 / np.pi
SETTLE = 4


def truth_indices(target_ns, count):
    """Require exact acquisition-grid targets; never silently floor or shift truth."""
    ns = np.asarray(target_ns)
    if np.any(ns != np.rint(ns)) or np.any((ns - BASE) % PERIOD):
        raise ValueError("target is not exactly on the acquisition grid")
    indices = ((ns - BASE) / PERIOD).astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= count):
        raise ValueError("target outside truth recording")
    return indices


def metrics(predicted, truth, times):
    """RMSE/RMS(truth), signed bias, Pearson correlation and coherent 28-Hz phasors.

    Amplitude is the peak of the finite-record least-squares harmonic projection,
    not RMS or the instantaneous stochastic envelope. Positive phase means lead.
    """
    p, y, t = map(np.asarray, (predicted, truth, times))
    if len(y) < 3:
        return {key: np.nan for key in (
            "rmse", "nrmse", "bias", "correlation", "truth_rms", "prediction_rms",
            "coherent_peak_truth", "coherent_peak_prediction", "amplitude_error_pct", "phase_error_deg")}
    error = p - y
    rms = np.sqrt(np.mean(y**2))
    design = np.column_stack((np.ones(len(t)), np.cos(OMEGA * t), np.sin(OMEGA * t)))
    coef = np.linalg.lstsq(design, np.column_stack((y, p)), rcond=None)[0]
    phasor = coef[1] - 1j * coef[2]
    ratio = phasor[1] / phasor[0] if abs(phasor[0]) > 1e-30 else np.nan + 1j * np.nan
    rmse = np.sqrt(np.mean(error**2))
    return dict(rmse=rmse, nrmse=rmse / rms if rms else np.nan,
                bias=np.mean(error), correlation=np.corrcoef(p, y)[0, 1] if np.std(p) * np.std(y) else np.nan,
                truth_rms=rms, prediction_rms=np.sqrt(np.mean(p**2)),
                coherent_peak_truth=abs(phasor[0]), coherent_peak_prediction=abs(phasor[1]),
                amplitude_error_pct=100 * (abs(ratio) - 1), phase_error_deg=np.angle(ratio, deg=True))


def spectra(predicted, truth):
    """Uniform 2-kHz, 8-second Hann segments; 28 Hz lies exactly on bin 224."""
    options = dict(fs=RATE, nperseg=min(16000, len(truth)), detrend="constant")
    f, yy = signal.welch(truth, **options)
    _, pp = signal.welch(predicted, **options)
    _, ee = signal.welch(predicted - truth, **options)
    _, yp = signal.csd(truth, predicted, **options)
    index = np.argmin(abs(f - 28))
    transfer = yp[index] / yy[index]
    band = (f >= 27) & (f <= 30)
    result = dict(spectral_frequency_hz=f[index], spectral_gain_28=abs(transfer),
                  spectral_phase_28_deg=np.angle(transfer, deg=True),
                  band_residual_reduction_db=10 * np.log10(np.sum(yy[band]) / np.sum(ee[band])))
    return result, (f, yy, pp, ee)


def availability(data, events, settle=SETTLE):
    """Count missing deadline slots as unavailable and censor interrupted recoveries."""
    nominal = data[:, 0] >= 0
    valid = data[:, 4] == 1
    scored = nominal & ((data[:, 1] - BASE) * 1e-9 >= settle)
    ready_times = data[valid, 1]
    recoveries = []
    recovered = np.zeros(len(events), dtype=bool)
    for i, event in enumerate(events):
        candidate = np.searchsorted(ready_times, event)
        end = events[i + 1] if i + 1 < len(events) else np.inf
        if candidate < len(ready_times) and ready_times[candidate] < end:
            recoveries.append((ready_times[candidate] - event) * 1e-6)
            recovered[i] = True
    return dict(availability_pct=100 * np.mean(valid[nominal]),
                scored_availability_pct=100 * np.mean(valid[scored]),
                recovered_events=int(np.sum(recovered)), censored_events=int(np.sum(~recovered)),
                recovery_mean_ms=np.mean(recoveries) if recoveries else np.nan,
                recovery_max_ms=max(recoveries) if recoveries else np.nan), recovered


def timing_us(data):
    """Report measurement age and sample-to-target horizon in microseconds, only for ready predictions."""
    valid = data[:, 4] == 1
    return (data[valid, 1] - data[valid, 2]) * 1e-3, (data[valid, 3] - data[valid, 2]) * 1e-3


def weighted_fit(times, acceleration, target):
    """Independent batch QR oracle for the two-stage 0.1-s window, including irregular intervals.

    This analysis can examine short histories that production correctly refuses
    to publish. It does not call or modify the estimator or its readiness gate.
    For stage weights b_j and decay l_j, the weight at endpoint k is
    b_j * product(l_[j+1:k]) * sum(b_[j:k]); uniform intervals give
    b**2 * (k-j+1) * l**(k-j).
    """
    times = np.asarray(times)
    dt = np.r_[1 / 8000, np.diff(times)]
    if np.any(dt <= 0):
        raise ValueError("batch acquisitions must be strictly ordered")
    beta = -np.expm1(-2 * dt / .1)
    decay = np.r_[np.cumprod((1 - beta[:0:-1]))[::-1], 1.]
    weight = beta * decay * np.cumsum(beta[::-1])[::-1]
    phase = OMEGA * (times - times[0])
    design = np.column_stack((np.ones(len(times)), np.cos(phase), np.sin(phase)))
    gram = design.T @ (weight[:, None] * design)
    np.linalg.cholesky(gram)  # Explicit SPD check, separate from statistical accuracy.
    moments = design.T @ (weight[:, None] * acceleration)
    coef = np.linalg.lstsq(design * np.sqrt(weight[:, None]),
                           acceleration * np.sqrt(weight[:, None]), rcond=None)[0]
    phase = OMEGA * (target - times[0])
    predicted = coef[1] * np.cos(phase) + coef[2] * np.sin(phase)
    residual = np.max(abs(gram @ coef - moments)) / max(np.max(abs(moments)), np.finfo(float).tiny)
    return predicted, 1 / np.linalg.cond(gram, 1), residual


def recovery_study(truth, population):
    """Probe cold-start history and hypothetical retention across one gap; change no production policy.

    Repeat at the 59 one-second loss epochs in a 60-second recording. Scores
    here are one probe per epoch/age, not a continuous controller trace. A
    retained fit is an offline candidate requiring separate future qualification.
    """
    rows = []
    seconds = int((len(truth) - 162) / 8000)
    decoded = (truth[:, 1:3] - 2048) * .0083799 / 1.02
    ages = (5, 10, 20, 35.75, 50, 75, 100, 150, 200, 300)
    for age in ages:
        predictions, references, conditions, residuals = [], [], [], []
        for second in range(1, seconds):
            first = second * 8000 + 1  # The sample at the blackout start is overwritten.
            last = first + round(age * 8)
            target = last + 9
            p, condition, residual = weighted_fit(np.arange(last-first+1) / 8000,
                decoded[first:last+1], (last-first+9) / 8000)
            predictions.append(p)
            references.append(truth[target, 9:11])
            conditions.append(condition)
            residuals.append(residual)
        for axis in range(2):
            p, y = np.array(predictions)[:, axis], np.array(references)[:, axis]
            error = p-y
            rows.append(dict(population=population, policy="cold_start_probe", age_ms=age, axis="XY"[axis],
                probes=len(p), nrmse=np.sqrt(np.mean(error**2) / np.mean(y**2)),
                rmse_ms2=np.sqrt(np.mean(error**2)), bias_ms2=np.mean(error),
                max_abs_error_ms2=max(abs(error)), rcond_min=min(conditions),
                normal_residual_max=max(residuals), max_delta_from_uninterrupted_ms2=np.nan))
    for age in (0, 1, 10, 50, 100, 300):
        predictions, references, differences, conditions, residuals = [], [], [], [], []
        for second in range(1, seconds):
            missing = second * 8000
            last = missing + 1 + round(age * 8)
            indices = np.arange(max(0, missing - 8000), last + 1)
            times = (indices - indices[0]) / 8000
            target = (last + 9 - indices[0]) / 8000
            uninterrupted, _, _ = weighted_fit(times, decoded[indices], target)
            retained = indices != missing
            p, condition, residual = weighted_fit(times[retained], decoded[indices[retained]], target)
            predictions.append(p)
            references.append(truth[last + 9, 9:11])
            differences.append(p - uninterrupted)
            conditions.append(condition)
            residuals.append(residual)
        for axis in range(2):
            p, y = np.array(predictions)[:, axis], np.array(references)[:, axis]
            error = p-y
            rows.append(dict(population=population, policy="retain_one_gap_candidate", age_ms=age, axis="XY"[axis],
                probes=len(p), nrmse=np.sqrt(np.mean(error**2) / np.mean(y**2)),
                rmse_ms2=np.sqrt(np.mean(error**2)), bias_ms2=np.mean(error),
                max_abs_error_ms2=max(abs(error)), rcond_min=min(conditions),
                normal_residual_max=max(residuals),
                max_delta_from_uninterrupted_ms2=max(abs(np.array(differences)[:, axis]))))
    return rows


def analyze(directory, plots=True):
    """Generate compact CSV reports, full-resolution PSDs and optional shareable PNG figures."""
    directory = Path(directory)
    report = directory / "report"
    report.mkdir(exist_ok=True)
    runs = pd.read_csv(directory / "runs.csv")
    summary, health, calibration, recovery, baseline, traces = [], [], [], [], {}, {}
    for population, group in runs.groupby("population", sort=False):
        truth = np.memmap(directory / f"{population}_truth.bin", dtype="<f8", mode="r").reshape(-1, 13)
        recovery.extend(recovery_study(truth, population))
        for axis in range(2):
            decoded = (truth[:, 1 + axis] - 2048) * .0083799 / 1.02
            quantization = decoded - truth[:, 3 + axis]
            noise = truth[:, 3 + axis] - truth[:, 5 + axis]
            calibration.append(dict(population=population, axis="XY"[axis], count_ms2=.0083799 / 1.02,
                quantization_rmse=np.sqrt(np.mean(quantization**2)), quantization_max_abs=max(abs(quantization)),
                sensor_noise_rms=np.sqrt(np.mean(noise**2)), expected_noise_rms=np.sqrt(1e-8 * 8000 / 2),
                target_accel_rms=np.sqrt(np.mean(truth[:, 9 + axis]**2)),
                target_pointing_rms_mas=np.sqrt(np.mean(truth[:, 11 + axis]**2)) * RAD_TO_MAS,
                angular_identity_max_abs=np.max(abs(truth[:, 11 + axis] + truth[:, 9 + axis] / (OMEGA**2 * 10)))))
            if np.max(abs(quantization)) > .5 * .0083799 / 1.02 + 1e-12:
                raise AssertionError("ADC rounding or clipping violates half-count bound")
        for run in group.itertuples():
            data = np.memmap(directory / f"{run.file}.bin", dtype="<f8", mode="r").reshape(-1, 21)
            event_table = pd.read_csv(directory / f"{run.file}_events.csv")
            events = event_table.time_ns.to_numpy()
            health_values, recovered = availability(data, events)
            valid = data[:, 4] == 1
            score = valid & ((data[:, 1] - BASE) * 1e-9 >= SETTLE)
            ix = truth_indices(data[:, 3], len(truth))
            age, effective = timing_us(data)
            recovery_window = np.zeros(len(data), dtype=bool)
            for event in events[recovered]:
                start = np.searchsorted(data[:, 1], event)
                ready_after = np.flatnonzero(valid[start:])
                if len(ready_after):
                    time = data[start + ready_after[0], 1]
                    recovery_window |= (data[:, 1] >= time) & (data[:, 1] < time + 500_000_000)
            identity = dict(population=population, scenario=run.scenario, horizon_ms=run.horizon_ns * 1e-6)
            health.append(dict(**identity, **health_values, published=run.published, processed=run.processed,
                dropped=run.dropped, resets=run.resets, overrun_events=len(events),
                overruns_per_s=len(events) / (run.published / 8000),
                first_ready_ms=(run.first_ready_ns - BASE) * 1e-6 if run.first_ready_ns >= 0 else np.nan,
                age_min_us=min(age) if len(age) else np.nan, age_max_us=max(age) if len(age) else np.nan,
                effective_horizon_us=np.mean(effective) if len(effective) else np.nan))
            for axis in range(2):
                for quantity, predcol, targetcol, fullcol, scale in (
                    ("acceleration_ms2", 10, 9, 5, 1), ("pointing_mas", 12, 11, 7, RAD_TO_MAS)):
                    for reference, col in (("target28", targetcol), ("full_clean", fullcol)):
                        p, y = data[:, predcol + axis] * scale, truth[ix, col + axis] * scale
                        values = metrics(p[score], y[score], (data[score, 3] - BASE) * 1e-9)
                        if run.scenario == "ideal":
                            spectral, psd = spectra(p[score], y[score])
                            values.update(spectral)
                            if run.horizon_ns == 1000000 and reference == "target28":
                                baseline[population, axis, quantity] = (data[score, 3].copy(), p[score].copy(),
                                    y[score].copy(), truth[ix[score], fullcol + axis] * scale, psd)
                        recovery_score = score & recovery_window
                        values["recovery_nrmse"] = metrics(p[recovery_score], y[recovery_score],
                            (data[recovery_score, 3] - BASE) * 1e-9)["nrmse"]
                        summary.append(dict(**identity, axis="XY"[axis], quantity=quantity, reference=reference,
                                            valid_samples=int(np.sum(score)), **values))
            if run.horizon_ns == 1000000:
                # CSV at 500 Hz; PSDs above always use the unthinned, continuous ideal record.
                keep = np.flatnonzero(data[:, 0] >= 0)[::4]
                trace = dict(query_s=(data[keep, 1] - BASE) * 1e-9,
                    target_s=(data[keep, 3] - BASE) * 1e-9, ready=valid[keep].astype(int),
                    dropped=data[keep, 7], resets=data[keep, 8])
                for axis in range(2):
                    for label, predcol, truthcol, scale in (("accel", 10, 9, 1), ("pointing_mas", 12, 11, RAD_TO_MAS)):
                        key = f"{label}_{'xy'[axis]}"
                        trace[f"true_{key}"] = truth[ix[keep], truthcol + axis] * scale
                        trace[f"pred_{key}"] = np.where(valid[keep], data[keep, predcol + axis] * scale, np.nan)
                        trace[f"residual_{key}"] = trace[f"pred_{key}"] - trace[f"true_{key}"]
                        fullcol = 5 if label == "accel" else 7
                        trace[f"full_true_{key}"] = truth[ix[keep], fullcol + axis] * scale
                frame = pd.DataFrame(trace)
                frame.to_csv(report / f"{population}_{run.scenario}_trace.csv", index=False)
                traces[population, run.scenario] = frame
    summary = pd.DataFrame(summary)
    summary.to_csv(report / "accuracy.csv", index=False)
    pd.DataFrame(health).to_csv(report / "availability.csv", index=False)
    pd.DataFrame(calibration).to_csv(report / "calibration.csv", index=False)
    pd.DataFrame(recovery).to_csv(report / "recovery_study.csv", index=False)
    (report / "analysis_versions.json").write_text(json.dumps(dict(numpy=np.__version__, pandas=pd.__version__,
        scipy=__import__("scipy").__version__), indent=2) + "\n")
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for population in runs.population.unique():
            fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True)
            psdfig, psdaxes = plt.subplots(2, 2, figsize=(13, 7))
            for axis in range(2):
                for row, quantity in enumerate(("acceleration_ms2", "pointing_mas")):
                    ns, p, y, full, psd = baseline[population, axis, quantity]
                    t = (ns - BASE) * 1e-9
                    select = (t >= 5) & (t < 7)
                    ax = axes[2 * row, axis]
                    if row == 1:
                        ax.plot(t[select], full[select], alpha=.4, label="all modes")
                    ax.plot(t[select], y[select], label="28-Hz truth")
                    ax.plot(t[select], p[select], label="prediction")
                    ax.set_ylabel(f"{'XY'[axis]} {quantity}")
                    ax.legend(fontsize=8)
                    axes[2 * row + 1, axis].plot(t[select], (p-y)[select])
                    axes[2 * row + 1, axis].set_ylabel("pred − target28")
                    axes[-1, axis].set_xlabel("target time [s]")
                    f, yy, pp, ee = psd
                    ax = psdaxes[row, axis]
                    _, full_psd = signal.welch(full, fs=RATE, nperseg=min(16000, len(full)), detrend="constant")
                    for power, label in ((yy, "28-Hz truth"), (full_psd, "all modes"), (pp, "prediction"), (ee, "residual")):
                        ax.semilogy(f, power, label=label)
                    ax.set(xlim=(0, 150), xlabel="Hz", ylabel=f"{quantity}²/Hz", title=f"{'XY'[axis]}")
                    ax.legend()
                    pd.DataFrame(dict(frequency_hz=f, truth_psd=yy, full_truth_psd=full_psd, prediction_psd=pp,
                        residual_psd=ee)).to_csv(report / f"{population}_{quantity}_{axis}_psd.csv", index=False)
            fig.suptitle(f"{population}: query + 1 ms; sample + 1.125 ms")
            fig.tight_layout()
            fig.savefig(report / f"{population}_predictions.png", dpi=140)
            psdfig.tight_layout()
            psdfig.savefig(report / f"{population}_psd.png", dpi=140)
            fig, axes = plt.subplots(2, 1, figsize=(12, 6))
            for i, scenario in enumerate(runs.scenario.unique()):
                frame = traces[population, scenario]
                for ax in axes:
                    ax.step(frame.query_s, i + frame.ready * .7, label=scenario)
            axes[0].set(xlabel="query time [s]", ylabel="ready (offset per scenario)")
            axes[1].set(xlim=(4.9, 5.7), xlabel="query time [s]", ylabel="overrun recovery")
            axes[0].legend(fontsize=8, loc="upper right")
            fig.tight_layout()
            fig.savefig(report / f"{population}_availability.png", dpi=140)
            plt.close("all")
        fig, ax = plt.subplots(figsize=(8, 5))
        subset = summary[(summary.scenario == "ideal") & (summary.quantity == "pointing_mas") &
                         (summary.reference == "target28")]
        for (population, axis), group in subset.groupby(["population", "axis"]):
            ax.plot(group.horizon_ms, group.nrmse, "o-", label=f"{population} {axis}")
        ax.set(xlabel="horizon beyond query [ms]", ylabel="RMSE / RMS(target28)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(report / "horizons.png", dpi=140)
        plt.close(fig)
    print(summary[(summary.scenario == "ideal") & (summary.horizon_ms == 1) &
                  (summary.reference == "target28")].to_string(index=False))
    print(pd.DataFrame(health).query("horizon_ms == 1").to_string(index=False))
    print(f"Reports: {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    arguments = parser.parse_args()
    analyze(arguments.directory, not arguments.no_plots)
