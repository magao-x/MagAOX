#!/usr/bin/env python3
r"""\file validate_real_harmonic.py
\brief Real-data qualification of the unchanged C++ harmonic estimator and normalized file-only shadow proposals.
\ingroup accelFFCtrl_unit_test

This is an experiment driver, not a new controller. Physical sensor/WFS/DM
calibration is absent; count-normalized model coordinates are explicitly used.
"""
import argparse
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy import signal

from analyze_timed_telemetry import timed_stream, spectral_grid, features, ridge_models, refit
from inspect_reduced_telemetry import sha256
from qualify_telemetry import replay

HORIZON = 1_000_000
RATE = 2000
BANDS = ((27, 30), (54, 58))


def target_labels(wfs, wt, targets, epoch, center_offset_ns=0):
    """Interpolate future WFS labels for scoring/calibration only; return exact right-bracket times for partition checks."""
    if np.any(targets < wt[0] + center_offset_ns) or np.any(targets > wt[-1] + center_offset_ns):
        raise ValueError("target outside observed WFS coverage")
    position = np.searchsorted(wt, targets - center_offset_ns, side="right")
    position = np.clip(position, 1, len(wt) - 1)
    t = (wt - epoch) * 1e-9
    label = np.column_stack(
        [np.interp((targets - center_offset_ns - epoch) * 1e-9, t, wfs[:, axis]) for axis in range(2)]
    )
    return label, wt[position] + center_offset_ns


def fit_map(x, y, kind):
    """Fit a real origin-preserving 2x2 numerical calibration: direct axes, swapped axes or cross-axis registration."""
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("invalid calibration rows")
    mapping = np.zeros((2, 2))
    if kind == "full":
        mapping = np.linalg.lstsq(x, y, rcond=None)[0].T
    elif kind in ("direct", "swapped"):
        for axis in range(2):
            sensor = axis if kind == "direct" else 1 - axis
            energy = x[:, sensor] @ x[:, sensor]
            if energy <= 0:
                raise ValueError("unexcited calibration axis")
            mapping[axis, sensor] = x[:, sensor] @ y[:, axis] / energy
    else:
        raise ValueError("unknown mapping")
    return mapping


def refined_carriers(acc, at, wfs, wt, start, end, epoch):
    """Estimate two existing frequency controls from training ADC spectra only; do not track or tune on held-out data."""
    _, x, _ = spectral_grid(acc, at, wfs, wt, start, end, epoch)
    f, power = signal.welch(x[:, :2], fs=RATE, nperseg=8000, noverlap=0, axis=0)
    result = []
    for channel, (lo, hi) in enumerate(((28, 29), (54, 58))):
        bins = np.flatnonzero((f >= lo) & (f <= hi))
        k = bins[np.argmax(power[bins, channel])]
        p = np.log(power[k - 1 : k + 2, channel])
        delta = np.clip(0.5 * (p[0] - p[2]) / (p[0] - 2 * p[1] + p[2]), -0.5, 0.5)
        result.append(float(f[k] + delta * (f[1] - f[0])))
    return result


def normalized_manifest(queries, frequencies):
    """Declare count-unit arithmetic normalizers and a synthetic numerical basis, without claiming SI or DM calibration."""
    return dict(
        accel=dict(rate_hz=RATE),
        estimator=dict(memory_s=0.1, lever_m=1, rotation_deg=0, frequencies_hz=list(frequencies)),
        adc=dict(zero=[0, 0], volts_per_bit=[1, 1], sensitivity=[1, 1]),
        query=dict(file=str(queries)),
        shadow=dict(
            gain=0,
            bounds=[3, 3],
            calibration_id=14,
            evidence=0,
            coordinates=np.eye(2).tolist(),
            modal_map=np.eye(2).tolist(),
            provenance="COUNT-NORMALIZED experiment only. Unit-valued numerical normalizers are not volts/count, V/(m/s²), metres or a physical DM map.",
        ),
    )


def phase_diagnostic_features(data, frequencies):
    """Expose existing Phase 12 target quadratures for a calibration diagnostic, not the real static controller map."""
    columns = []
    for band, axis in ((0, "ax"), (1, "ay")):
        factor = -1 / (2 * np.pi * frequencies[band]) ** 2
        columns.extend([factor * data[f"{axis}_re{band}"].to_numpy(), factor * data[f"{axis}_im{band}"].to_numpy()])
    return np.column_stack(columns)


def run_cpp(executable, manifest, adc, epoch, output):
    """Use the existing Phase 12 C++ file replay, retaining only experiment products outside transient work files."""
    with tempfile.TemporaryDirectory(prefix="cpp-", dir=output) as work:
        return replay(executable, manifest, adc, epoch, work, HORIZON)


def baselines(acc, at, wfs, wt, indices, truth, train, validation, earlier, test):
    """Refit existing fixed-target baselines using supplied ADC0/1 only, or the historical three-channel comparison."""
    tr, va, pre = [np.flatnonzero(mask)[::4] for mask in (train, validation, earlier)]
    history = 64
    tx = features(acc, at, wfs, wt, indices[tr], history)
    vx = features(acc, at, wfs, wt, indices[va], history)
    px = features(acc, at, wfs, wt, indices[pre], history)
    results = {}
    channels = min(3, acc.shape[1])
    cases = [("phase13_regression_3ch", np.arange(tx.shape[1]))] if channels == 3 else []
    cases.append(
        ("regression_01", np.r_[np.arange(128), 128 + (channels * np.arange(64)[:, None] + np.arange(2)).ravel()])
    )
    for name, columns in cases:
        alpha = ridge_models(tx[:, columns], truth[tr], 128, vx[:, columns], truth[va])
        coef, intercept = refit(px[:, columns], truth[pre], 128, alpha)
        results[name] = (columns, coef["wfs_accel"], intercept["wfs_accel"])
        if "wfs_only" not in results:
            results["wfs_only"] = (columns, coef["wfs_only"], intercept["wfs_only"])
    predictions = {name: np.zeros_like(truth) for name in results}
    test = np.flatnonzero(test)
    # Caller scores only its held-out partition; actual timestamps determine all feature availability.
    for first in range(0, len(test), 10000):
        rows = test[first : first + 10000]
        x = features(acc, at, wfs, wt, indices[rows], history)
        for name, (columns, coef, intercept) in results.items():
            predictions[name][rows] = x[:, columns] @ coef + intercept
    return predictions


def scores(prediction, truth, target_ns, epoch, name):
    """Score raw prediction error and post-prediction residual PSD, preserving a fixed common test cohort."""
    t = (target_ns - epoch) * 1e-9
    grid = np.arange(np.ceil(t[0] * RATE), np.floor(t[-1] * RATE) + 1) / RATE
    y = np.column_stack([np.interp(grid, t, truth[:, a]) for a in range(2)])
    error = prediction - truth
    e = np.column_stack([np.interp(grid, t, error[:, a]) for a in range(2)])
    f, py = signal.welch(y, fs=RATE, nperseg=8000, noverlap=0, axis=0)
    _, pe = signal.welch(e, fs=RATE, nperseg=8000, noverlap=0, axis=0)
    rows = []
    for axis in range(2):
        row = dict(
            model=name,
            axis=axis,
            count=len(truth),
            rmse=float(np.sqrt(np.mean(error[:, axis] ** 2))),
            nrmse=float(np.sqrt(np.mean(error[:, axis] ** 2) / np.mean(truth[:, axis] ** 2))),
            bias=float(np.mean(error[:, axis])),
            correlation=float(np.corrcoef(prediction[:, axis], truth[:, axis])[0, 1]),
        )
        for band, (lo, hi) in enumerate(BANDS):
            keep = (f >= lo) & (f <= hi)
            before, after = np.sum(py[keep, axis]), np.sum(pe[keep, axis])
            row[f"band{band}_truth_rms"] = float(np.sqrt(before * (f[1] - f[0])))
            row[f"band{band}_error_rms"] = float(np.sqrt(after * (f[1] - f[0])))
            row[f"band{band}_ideal_suppression_db"] = float(10 * np.log10(before / after))
        rows.append(row)
    return rows, (f, py, pe)


def analyze(dataset, executable, output, plots=True):
    """Evaluate existing harmonic outputs and normalized proposals; all calibration choices precede held-out scoring."""
    root, output = Path(dataset).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("output must be outside original data")
    before = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
    output.mkdir(parents=True, exist_ok=True)
    acc, at, aw, _ = timed_stream(root, "acc000")
    wfs, wt, _, _ = timed_stream(root, "wfs_tiptilt")
    epoch = int(at[0])
    meta = json.loads((root / "metadata.json").read_text())
    start, stop = [int(pd.Timestamp(meta[k]).value) for k in ("start_utc", "end_utc")]
    train_end, test_start = start + 120_000_000_000, start + 150_000_000_000
    indices = np.flatnonzero((wt >= start) & (wt + 10_000_000 < stop))
    query = wt[indices]
    target = query + HORIZON
    truth, right = target_labels(wfs, wt, target, epoch)
    train = right < train_end
    validation = (query >= train_end) & (right < test_start)
    earlier = right < test_start
    test = query >= test_start
    sigma = np.std(truth[earlier], axis=0)
    selection_sigma = np.std(truth[train], axis=0)
    query_path = output / "queries_absolute.csv"
    pd.DataFrame(dict(query_ns=query)).to_csv(query_path, index=False)
    raw_timing = np.load(root / "acc000_timings.npy", mmap_mode="r", allow_pickle=False)
    adc = pd.DataFrame(
        dict(
            acquired_ns=at,
            written_ns=aw,
            receipt_ns=aw,
            counter=raw_timing[:, 0],
            generation=1,
            adc_x=acc[:, 0].astype(np.uint16),
            adc_y=acc[:, 1].astype(np.uint16),
        )
    )
    frequency_sets = {
        "configured": [28.5, 56.0],
        "training_peak": refined_carriers(acc, at, wfs, wt, start, train_end, epoch),
    }
    candidates, forecasts, diagnostic = [], {}, {}
    for family, frequencies in frequency_sets.items():
        manifest = normalized_manifest(query_path, frequencies)
        d = run_cpp(executable, manifest, adc, epoch, output)
        valid = d.valid.to_numpy().astype(bool)
        quadratures = phase_diagnostic_features(d, frequencies)
        coefficient = np.linalg.lstsq(quadratures[train & valid], truth[train & valid], rcond=None)[0]
        loss = float(
            np.mean(
                ((quadratures[validation & valid] @ coefficient - truth[validation & valid]) / selection_sigma) ** 2
            )
        )
        diagnostic[family] = (loss, quadratures, valid)
        for variant, bands in (("28_only", (0,)), ("56_only", (1,)), ("dual_shared", (0, 1))):
            x = sum(d[[f"px{band}", f"py{band}"]].to_numpy() for band in bands)
            forecasts[family, variant] = (x, valid)
            for kind in ("direct", "swapped", "full"):
                mapping = fit_map(x[train & valid], truth[train & valid], kind)
                loss = float(
                    np.mean(((x[validation & valid] @ mapping.T - truth[validation & valid]) / selection_sigma) ** 2)
                )
                candidates.append(dict(family=family, variant=variant, mapping=kind, validation_loss=loss))
    choices = pd.DataFrame(candidates)
    chosen = choices.loc[choices.groupby("variant").validation_loss.idxmin()].copy()
    choices.to_csv(output / "mapping_validation.csv", index=False)
    predictions = baselines(acc, at, wfs, wt, indices, truth, train, validation, earlier, test)
    predictions["persistence"] = wfs[indices]
    calibrated = {}
    for _, choice in chosen.iterrows():
        x, valid = forecasts[choice.family, choice.variant]
        mapping = fit_map(x[earlier & valid], truth[earlier & valid], choice.mapping)
        p = x @ mapping.T
        p[~valid] = 0
        name = "harmonic_" + choice.variant
        predictions[name] = p
        calibrated[choice.variant] = dict(
            family=choice.family,
            mapping_type=choice.mapping,
            map_to_wfs=mapping.tolist(),
            frequencies_hz=[
                frequency_sets[choice.family][i]
                for i in ((0,) if choice.variant == "28_only" else (1,) if choice.variant == "56_only" else (0, 1))
            ],
            heldout_valid_fraction=float(np.mean(valid[test])),
        )
    diagnostic_family = min(diagnostic, key=lambda family: diagnostic[family][0])
    _, quadratures, valid = diagnostic[diagnostic_family]
    diagnostic_map = np.linalg.lstsq(quadratures[earlier & valid], truth[earlier & valid], rcond=None)[0]
    diagnostic_prediction = quadratures @ diagnostic_map
    diagnostic_prediction[~valid] = 0
    predictions["phase_calibration_diagnostic_ONLY"] = diagnostic_prediction
    all_rows, spectra_by_model = [], {}
    for name, p in predictions.items():
        rows, spectrum = scores(p[test], truth[test], target[test], epoch, name)
        all_rows.extend(rows)
        spectra_by_model[name] = spectrum
    pd.DataFrame(all_rows).to_csv(output / "metrics.csv", index=False)
    # Retain the existing shared-map dual-bank correction convention, separately from single-bank diagnostics.
    config = calibrated["dual_shared"]
    mapping = np.asarray(config["map_to_wfs"])
    manifest = normalized_manifest(query_path, config["frequencies_hz"])
    manifest["shadow"]["modal_map"] = (mapping / sigma[:, None]).tolist()
    disabled = run_cpp(executable, manifest, adc, epoch, output)
    disabled_manifest = copy.deepcopy(manifest)
    manifest["shadow"]["gain"] = 1
    shadow = run_cpp(executable, manifest, adc, epoch, output)
    proposed = shadow[["mode0", "mode1"]].to_numpy()
    expected_prediction = shadow[["theta_x", "theta_y"]].to_numpy() @ mapping.T / sigma
    ratios = np.divide(
        3.0, np.abs(expected_prediction), out=np.ones_like(expected_prediction), where=expected_prediction != 0
    )
    scale = np.minimum(1, np.min(ratios, axis=1))
    active = shadow.valid.to_numpy().astype(bool)
    expected = -expected_prediction * scale[:, None]
    expected[~active] = 0
    np.testing.assert_allclose(proposed, expected, rtol=2e-12, atol=1e-12)
    np.testing.assert_array_equal(disabled[["mode0", "mode1"]], np.zeros_like(proposed))
    if np.any(np.abs(proposed) > 3 * (1 + 1e-12)) or np.any(np.sum(proposed * expected_prediction, axis=1) > 1e-12):
        raise ValueError("normalized amplitude or opposing-sign invariant failed")
    log = pd.DataFrame(
        dict(
            query_ns=query,
            target_ns=target,
            sample_ns=shadow.sample_ns.to_numpy() - 1_000_000_000 + epoch,
            source_counter=shadow.counter,
            valid=shadow.valid,
            state=shadow.state,
            scale=shadow.scale,
            predicted_tip=predictions["harmonic_dual_shared"][:, 0],
            predicted_tilt=predictions["harmonic_dual_shared"][:, 1],
            normalized_tip=proposed[:, 0],
            normalized_tilt=proposed[:, 1],
        )
    )
    log.to_csv(output / "normalized_shadow.csv", index=False)
    # Frozen calibration/forecaster, additional assumed ADC availability delays; no held-out refit.
    sensitivity = []
    for latency in (500000, 1000000):
        delayed = adc.copy()
        delayed.receipt_ns += latency
        d = run_cpp(executable, manifest, delayed, epoch, output)
        p = d[["theta_x", "theta_y"]].to_numpy() @ mapping.T
        p[d.valid == 0] = 0
        rows, _ = scores(p[test], truth[test], target[test], epoch, f"adc_latency_{latency}")
        sensitivity.extend([dict(row, latency_ns=latency, valid_fraction=float(d.valid[test].mean())) for row in rows])
    pd.DataFrame(sensitivity).to_csv(output / "availability_sensitivity.csv", index=False)
    if plots:
        make_plots(output, predictions, truth, target, test, epoch, spectra_by_model, log)
    after = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
    if before != after:
        raise ValueError("original inputs changed")
    report = dict(
        input_files_unchanged=len(before),
        actual_cpp_model="harmonicEstimator + inputProcessor + shadowCommand",
        model_changed=False,
        physical_calibration=False,
        live_application_dual_bank_support_claimed=False,
        heldout_wfs_used_by_harmonic_predictions=False,
        model_units="raw-count acceleration normalizer; pointing=-harmonic_counts/omega², NOT SI radians",
        adc_channels=[0, 1],
        physical_adc_pair_justified=False,
        exposure_center_offset_ns_assumed=0,
        adc_availability_latency_ns_assumed=0,
        wfs_availability_latency_ns_assumed=0,
        horizon_ns=HORIZON,
        common_epoch_ns=epoch,
        train_end_ns=train_end,
        test_start_ns=test_start,
        maximum_calibration_label_right_ns=int(right[earlier].max()),
        count_test=int(np.sum(test)),
        frequency_sets=frequency_sets,
        selected=calibrated,
        training_wfs_sigma=sigma.tolist(),
        phase_calibration_diagnostic=dict(
            family=diagnostic_family,
            coefficients=diagnostic_map.tolist(),
            status="Existing causal quadratures + fixed earlier-fit phase calibration; NOT current real static pointing/correction map; no shadow commands emitted from it",
        ),
        shadow=dict(
            gain_default=0,
            counterfactual_gain=1,
            normalized_bound=3,
            units="training WFS standard deviations; not DM modal/surface/voltage units",
            clipped=int(np.sum(shadow.state == 3)),
            inhibited=int(np.sum(shadow.state == 1)),
            heldout_valid_fraction=float(np.mean(active[test])),
            maximum_absolute_normalized_command=np.max(np.abs(proposed), axis=0).tolist(),
            disabled_nonzero_values=int(np.count_nonzero(disabled[["mode0", "mode1"]].to_numpy())),
            invalid_nonzero_values=int(np.count_nonzero(proposed[~active])),
            maximum_target_error_ns=int(np.max(np.abs(shadow.target_ns - shadow.query_ns - HORIZON))),
            processed=int(shadow.processed.iloc[-1]),
            dropped=int(shadow.dropped.iloc[-1]),
            resets=int(shadow.resets.iloc[-1]),
        ),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "shadow_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "disabled_manifest.json").write_text(json.dumps(disabled_manifest, indent=2) + "\n")
    (output / "input_sha256.json").write_text(json.dumps(before, indent=2) + "\n")
    return report


def make_plots(output, predictions, truth, target, test, epoch, spectra_by_model, shadow, names=None):
    """Save actual-model comparisons and explicitly normalized host-only correction proposals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if names is None:
        names = ("wfs_only", "phase13_regression_3ch", "harmonic_dual_shared", "phase_calibration_diagnostic_ONLY")
    time = (target - epoch) * 1e-9
    first = time[test][0]
    show = test & (time >= first + 1) & (time < first + 1.3)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for a in range(2):
        axes[a].plot(time[show], truth[show, a], color="black", label="Observed WFS interpolated at target", lw=1)
        for name in names:
            axes[a].plot(time[show], predictions[name][show, a], label=name, alpha=0.7)
        axes[a].set_ylabel("Recorded WFS modal units")
        axes[a].set_title(("Tip", "Tilt")[a])
        axes[a].legend(fontsize=8)
    axes[-1].set_xlabel("Target acquisition-field time [s from common epoch]")
    fig.tight_layout()
    fig.savefig(output / "prediction_comparison.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for a in range(2):
        for name in names:
            axes[a].plot(time[show], truth[show, a] - predictions[name][show, a], label=name, alpha=0.7)
        axes[a].set(ylabel="Residual [recorded WFS units]", title=("Tip", "Tilt")[a])
        axes[a].legend(fontsize=8)
    axes[-1].set_xlabel("Target acquisition-field time [s from common epoch]")
    fig.tight_layout()
    fig.savefig(output / "residual_timeseries.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for a in range(2):
        f, py, _ = spectra_by_model["wfs_only"]
        axes[a].semilogy(f, py[:, a], color="black", label="Observed WFS")
        for name in names:
            f, _, pe = spectra_by_model[name]
            axes[a].semilogy(f, pe[:, a], label=name + " error")
        axes[a].set(xlim=(20, 65), ylabel="PSD [recorded modal²/Hz]")
        axes[a].set_title(("Tip", "Tilt")[a])
        axes[a].legend(fontsize=8)
    axes[-1].set_xlabel("Frequency [Hz, recorded clock]")
    fig.tight_layout()
    fig.savefig(output / "residual_psds.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    q = (shadow.query_ns.to_numpy() - epoch) * 1e-9
    show = (q >= first + 1) & (q < first + 1.3)
    for a, field in enumerate(("normalized_tip", "normalized_tilt")):
        axes[a].plot(q[show], shadow.loc[show, field])
        axes[a].set(ylabel="Training WFS SD", title=("Tip", "Tilt")[a] + " contribution; numerical bound ±3 SD")
    axes[-1].set_xlabel("Query time [s from common epoch]")
    fig.suptitle("Offline counterfactual gain 1; NOT physical DM commands")
    fig.tight_layout()
    fig.savefig(output / "normalized_shadow.png")
    plt.close(fig)


def main():
    """Replay only local files through the existing numerical controller; never attach a command stream."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(analyze(args.dataset, args.replay, args.output, not args.no_plots), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
