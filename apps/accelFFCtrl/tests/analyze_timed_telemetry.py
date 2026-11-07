#!/usr/bin/env python3
r"""\file analyze_timed_telemetry.py
\brief Recorded-clock spectra and chronological causal AR/ARX comparisons, without physical timing certification.
\ingroup accelFFCtrl_unit_test

Companion to Phase 13 inspection. Raw values retain counts/modal units; no
sensor or DM calibration, clock correction, application or stream API is used.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, signal

from inspect_reduced_telemetry import filename_ns, sha256, utc
from qualify_telemetry import spectra

RATE = 2000
STEP_NS = 500000
BANDS = ((27, 30), (54, 58))
MODELS = ("persistence", "wfs_only", "accel_only", "wfs_accel")


def timed_stream(root, stream):
    """Verify array/timing counts and every filename boundary; preserve row order and exact integer ns."""
    root = Path(root)
    entries = json.loads((root / stream / "manifest.json").read_text())
    if not entries or len({e["output_file"] for e in entries}) != len(entries):
        raise ValueError("empty/duplicate array manifest")
    if {e["output_file"] for e in entries} != {p.name for p in (root / stream).glob("*.npy")}:
        raise ValueError("manifest/array inventory mismatch")
    timing = np.load(root / f"{stream}_timings.npy", mmap_mode="r", allow_pickle=False)
    if timing.ndim != 2 or timing.shape[1] != 5 or timing.dtype != np.uint64:
        raise ValueError("timing must be original UINT64 N-by-5 records")
    if np.any(timing[:, [2, 4]] >= 1000000000) or np.any(timing[:, [1, 3]] == 0):
        raise ValueError("invalid timing components")
    if np.any(timing[:, [1, 3]] > (2**63 - 1) // 1000000000):
        raise ValueError("timing exceeds signed nanosecond domain")
    acquired = timing[:, 1].astype(np.int64) * 1000000000 + timing[:, 2].astype(np.int64)
    written = timing[:, 3].astype(np.int64) * 1000000000 + timing[:, 4].astype(np.int64)
    counter = timing[:, 0]
    if np.any(np.diff(acquired) <= 0) or np.any(written < acquired) or np.any(counter == 0):
        raise ValueError("invalid acquisition/publication ordering")
    if np.any(counter[1:] <= counter[:-1]) or np.any(counter[1:] - counter[:-1] != 1):
        raise ValueError("split discontinuous source runs before this gapless analysis")
    arrays, boundaries, offset = [], [], 0
    previous = -1
    for entry in entries:
        name = entry["output_file"]
        if Path(name).name != name or name != Path(entry["source_file"]).with_suffix(".npy").name:
            raise ValueError("invalid source/output association")
        a = np.load(root / stream / name, mmap_mode="r", allow_pickle=False)
        expected = np.uint16 if stream == "acc000" else np.float32
        if (
            a.ndim != 2
            or a.dtype != expected
            or list(a.shape) != entry["shape"]
            or str(a.dtype) != entry["dtype"]
            or not np.all(np.isfinite(a))
        ):
            raise ValueError("invalid array shape/dtype/data")
        label = filename_ns(entry["source_file"])
        if offset + len(a) > len(acquired) or label <= previous or acquired[offset] != label:
            raise ValueError("timing row/file boundary association mismatch")
        previous = label
        boundaries.append(
            dict(
                stream=stream,
                source_file=entry["source_file"],
                first_row=offset,
                frames=len(a),
                first_counter=int(counter[offset]),
                first_acquired_ns=int(acquired[offset]),
                filename_ns=label,
                difference_ns=int(acquired[offset]) - label,
            )
        )
        arrays.append(np.asarray(a, dtype=float))
        offset += len(a)
    values = np.concatenate(arrays)
    if (
        len(values) != len(timing)
        or (stream == "acc000" and values.shape[1] < 3)
        or (stream == "wfs_tiptilt" and values.shape[1] != 2)
    ):
        raise ValueError("data/timing count or channel mismatch")
    return values, acquired, written, boundaries


def cadence(acquired, written):
    """Describe interval asymmetry, compensation and nominal-counter drift; these are not calibrated clock offsets."""
    dt = np.diff(acquired)
    elapsed = (acquired - acquired[0]).astype(float)
    index = np.arange(len(acquired), dtype=float)
    center = index - index.mean()
    slope = np.dot(center, elapsed - elapsed.mean()) / np.dot(center, center)
    residual = elapsed - (elapsed.mean() - slope * index.mean() + slope * index)
    short = dt < STEP_NS
    return dict(
        first_ns=int(acquired[0]),
        last_ns=int(acquired[-1]),
        first_utc=utc(acquired[0]),
        last_utc=utc(acquired[-1]),
        records=len(acquired),
        counter_gaps=0,
        acquisition_monotonic=True,
        average_rate_hz=float((len(acquired) - 1) * 1e9 / (int(acquired[-1]) - int(acquired[0]))),
        interval_quantiles_ns={
            str(q): float(np.quantile(dt, q)) for q in (0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1)
        },
        interval_mean_ns=float(dt.mean()),
        interval_std_ns=float(dt.std()),
        lag1_interval_correlation=float(np.corrcoef(dt[1:], dt[:-1])[0, 1]),
        fraction_below_500us=float(short.mean()),
        below_500us_mean_ns=float(dt[short].mean()) if np.any(short) else None,
        above_500us_mean_ns=float(dt[~short].mean()) if np.any(~short) else None,
        nominal_counter_slope_ppm=float((slope / STEP_NS - 1) * 1e6),
        detrended_counter_time_rms_ns=float(np.std(residual)),
        per_interval_average_std_ns={
            str(k): float(np.std((acquired[k:] - acquired[:-k]) / k)) for k in (2, 4, 10, 100)
        },
        write_equals_acquisition=bool(np.array_equal(acquired, written)),
        publication_latency_identified=False,
        physical_clock_registration_verified=False,
    )


def spectral_grid(acc, at, wfs, wt, start, stop, epoch):
    """Register up to three ADC channels on a spectral grid, never supplying it to causal features."""
    if np.max(np.diff(at)) > 1000000 or np.max(np.diff(wt)) > 1000000:
        raise ValueError("large acquisition hole: split runs instead of interpolating across it")
    grid_ns = start + np.arange((stop - start) // STEP_NS, dtype=np.int64) * STEP_NS
    if grid_ns[0] < max(at[0], wt[0]) or grid_ns[-1] > min(at[-1], wt[-1]):
        raise ValueError("analysis grid exceeds measured overlap")
    t = (grid_ns - epoch) * 1e-9
    x = np.column_stack([np.interp(t, (at - epoch) * 1e-9, acc[:, c]) for c in range(min(3, acc.shape[1]))])
    y = np.column_stack([np.interp(t, (wt - epoch) * 1e-9, wfs[:, c]) for c in range(2)])
    return grid_ns, x, y


def spectral_report(grid, x, y, start, stop):
    """Return full/train/test pairwise coherence and recorded-clock CSD phase at training-selected WFS peaks."""
    duration = stop - start
    groups = {
        "full": np.ones(len(grid), dtype=bool),
        "train": grid < start + duration * 2 // 5,
        "test": grid >= start + duration // 2,
    }
    results = {}
    for name, keep in groups.items():
        results[name] = [
            spectra(
                [(np.arange(np.sum(keep)) / RATE, np.repeat(x[keep, c : c + 1], 2, axis=1), y[keep])], RATE, seconds=4
            )
            for c in range(3)
        ]
    reference = results["train"][0]
    peaks = {}
    for axis in range(2):
        for band, (lo, hi) in enumerate(BANDS):
            bins = np.flatnonzero((reference["f"] >= lo) & (reference["f"] <= hi))
            peaks[axis, band] = int(bins[np.argmax(reference["pyy"][bins, axis])])
    rows = []
    for name, channels in results.items():
        for channel, spec in enumerate(channels):
            for (axis, band), k in peaks.items():
                rows.append(
                    dict(
                        interval=name,
                        accel_channel=channel,
                        wfs_axis=axis,
                        band=band,
                        frequency_hz=float(spec["f"][k]),
                        coherence=float(spec["coherence"][k, axis]),
                        recorded_clock_phase_deg=float(np.angle(spec["h"][k, axis], deg=True)),
                        approximate_phase_halfwidth_deg=float(np.degrees(spec["phase_ci95_rad"][k, axis])),
                        transfer_accepted=bool(spec["accepted"][k, axis]),
                        blocks=spec["segments"],
                    )
                )
    return pd.DataFrame(rows), results


def features(acc, at, wfs, wt, indices, history=64, guard_ns=0):
    """Use only current/past WFS frames and ADC acquisitions no later than query minus guard/lag."""
    if history < 1 or guard_ns < 0 or np.any(indices < history - 1):
        raise ValueError("invalid causal history/guard")
    past = indices[:, None] - np.arange(history)
    wf = wfs[past].reshape(len(indices), -1)
    deadlines = wt[indices, None] - guard_ns - np.arange(history, dtype=np.int64) * STEP_NS
    ai = np.searchsorted(at, deadlines, side="right") - 1
    if np.any(ai < 0) or np.any(at[ai] > deadlines):
        raise ValueError("unavailable or future ADC history")
    af = acc[ai, :3].reshape(len(indices), -1)
    return np.column_stack((wf, af))


def ridge_models(x, targets, n_wfs, validate_x, validate_y):
    """Select ridge strengths on an earlier validation interval, with training-only scaling."""
    splits = {"wfs_only": slice(None, n_wfs), "accel_only": slice(n_wfs, None), "wfs_accel": slice(None)}
    means, scales = x.mean(axis=0), x.std(axis=0)
    if np.any(scales <= 1e-12):
        raise ValueError("constant or degenerate training feature")
    z = (x - means) / scales
    vz = (validate_x - means) / scales
    ym = targets.mean(axis=0)
    result = {}
    for name, cols in splits.items():
        a, va = z[:, cols], vz[:, cols]
        gram = a.T @ a / len(a)
        rhs = a.T @ (targets - ym) / len(a)
        choices = []
        for alpha in (1e-4, 1e-3, 1e-2, 0.1, 1.0):
            coef = linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs, assume_a="pos")
            error = np.mean((va @ coef + ym - validate_y) ** 2, axis=0)
            choices.append((alpha, coef, error))
        selected = np.argmin(np.stack([c[2] for c in choices]), axis=0)
        result[name] = np.array([choices[i][0] for i in selected])
    return result


def refit(x, targets, n_wfs, alphas):
    """Refit on the complete pre-test interval, using only already selected regularization strengths."""
    mean, scale, ym = x.mean(axis=0), x.std(axis=0), targets.mean(axis=0)
    z = (x - mean) / scale
    coefficients = {}
    for name, cols in {
        "wfs_only": slice(None, n_wfs),
        "accel_only": slice(n_wfs, None),
        "wfs_accel": slice(None),
    }.items():
        a = z[:, cols]
        gram = a.T @ a / len(a)
        rhs = a.T @ (targets - ym) / len(a)
        c = np.zeros((x.shape[1], targets.shape[1]))
        for output, alpha in enumerate(alphas[name]):
            c[cols, output] = linalg.solve(gram + alpha * np.eye(len(gram)), rhs[:, output], assume_a="pos")
        coefficients[name] = c / scale[:, None]
    intercept = {name: ym - mean @ coef for name, coef in coefficients.items()}
    return coefficients, intercept


def predict_record(
    acc, at, wfs, wt, start, stop, steps=(1, 2, 4, 10, 20), history=64, guards=(0, 500000, 1000000, 2000000)
):
    """Chronological frame-ahead AR/ARX comparison; future frame values/times are labels, never input features."""
    duration = stop - start
    train_end = start + duration * 2 // 5
    test_start = start + duration // 2
    largest = max(steps)
    index = np.arange(history - 1, len(wfs) - largest)
    index = index[(wt[index] >= start) & (wt[index + largest] < stop)]
    train = index[wt[index + largest] < train_end][::4]
    validation = index[(wt[index] >= train_end) & (wt[index + largest] < test_start)][::4]
    earlier = index[wt[index + largest] < test_start][::4]
    test = index[wt[index] >= test_start]
    if min(len(train), len(validation), len(test)) < 100:
        raise ValueError("insufficient chronological coverage")
    rows, traces, selected = [], {}, []
    for guard in guards:
        used_steps = steps if guard == 0 else (2,)
        tr_x = features(acc, at, wfs, wt, train, history, guard)
        va_x = features(acc, at, wfs, wt, validation, history, guard)
        re_x = features(acc, at, wfs, wt, earlier, history, guard)
        tr_y = np.column_stack([wfs[train + k] for k in used_steps])
        va_y = np.column_stack([wfs[validation + k] for k in used_steps])
        re_y = np.column_stack([wfs[earlier + k] for k in used_steps])
        alpha = ridge_models(tr_x, tr_y, 2 * history, va_x, va_y)
        coefficients, intercept = refit(re_x, re_y, 2 * history, alpha)
        predictions = {name: np.empty((len(test), len(used_steps) * 2)) for name in coefficients}
        for i in range(0, len(test), 10000):
            chunk = test[i : i + 10000]
            fx = features(acc, at, wfs, wt, chunk, history, guard)
            for name in coefficients:
                predictions[name][i : i + len(chunk)] = fx @ coefficients[name] + intercept[name]
        for h, k in enumerate(used_steps):
            truth = wfs[test + k]
            for name in MODELS:
                pred = wfs[test] if name == "persistence" else predictions[name][:, 2 * h : 2 * h + 2]
                error = pred - truth
                for axis in range(2):
                    rows.append(
                        dict(
                            guard_ns=guard,
                            frames_ahead=k,
                            expected_horizon_ms=k / 2,
                            axis=axis,
                            model=name,
                            count=len(test),
                            rmse=float(np.sqrt(np.mean(error[:, axis] ** 2))),
                            nrmse=float(np.sqrt(np.mean(error[:, axis] ** 2) / np.mean(truth[:, axis] ** 2))),
                            correlation=float(np.corrcoef(pred[:, axis], truth[:, axis])[0, 1]),
                            bias=float(np.mean(error[:, axis])),
                            actual_horizon_median_ns=float(np.median(wt[test + k] - wt[test])),
                            actual_horizon_p01_ns=float(np.quantile(wt[test + k] - wt[test], 0.01)),
                            actual_horizon_p99_ns=float(np.quantile(wt[test + k] - wt[test], 0.99)),
                        )
                    )
                if guard == 0:
                    traces[k, name] = (test, truth, pred)
            for name, values in alpha.items():
                for axis in range(2):
                    selected.append(
                        dict(guard_ns=guard, frames_ahead=k, model=name, axis=axis, alpha=float(values[2 * h + axis]))
                    )
    return (
        pd.DataFrame(rows),
        traces,
        pd.DataFrame(selected),
        dict(
            train_samples=len(train),
            validation_samples=len(validation),
            refit_samples=len(earlier),
            test_samples=len(test),
            train_end_ns=int(train_end),
            test_start_ns=int(test_start),
            history_frames=history,
            fit_stride_frames=4,
            last_refit_target_ns=int(wt[earlier[-1] + largest]),
            first_test_query_ns=int(wt[test[0]]),
            last_test_target_ns=int(wt[test[-1] + largest]),
        ),
    )


def prediction_bands(traces, wt, epoch):
    """Score band-limited held-out errors by post-prediction spectral registration, never filtering features."""
    rows = []
    for (ahead, name), (indices, truth, pred) in traces.items():
        time = (wt[indices + ahead] - epoch) * 1e-9
        grid = np.arange(np.ceil(time[0] * RATE), np.floor(time[-1] * RATE) + 1) / RATE
        y = np.column_stack([np.interp(grid, time, truth[:, a]) for a in range(2)])
        e = np.column_stack([np.interp(grid, time, pred[:, a] - truth[:, a]) for a in range(2)])
        f, py = signal.welch(y, fs=RATE, nperseg=8000, noverlap=0, axis=0)
        _, pe = signal.welch(e, fs=RATE, nperseg=8000, noverlap=0, axis=0)
        for axis in range(2):
            for band, (lo, hi) in enumerate(BANDS):
                keep = (f >= lo) & (f <= hi)
                rows.append(
                    dict(
                        frames_ahead=ahead,
                        model=name,
                        axis=axis,
                        band=band,
                        truth_band_rms=float(np.sqrt(np.sum(py[keep, axis]) * (f[1] - f[0]))),
                        error_band_rms=float(np.sqrt(np.sum(pe[keep, axis]) * (f[1] - f[0]))),
                    )
                )
    return pd.DataFrame(rows)


def block_improvement(traces, wt):
    """Bootstrap paired five-second held-out error blocks, conditional on these fitted models and this record."""
    rows = []
    rng = np.random.default_rng(13013)
    for ahead in sorted({key[0] for key in traces}):
        indices, truth, baseline = traces[ahead, "wfs_only"]
        assisted = traces[ahead, "wfs_accel"][2]
        group = (wt[indices] - wt[indices[0]]) // 5_000_000_000
        block = []
        for k in np.unique(group):
            keep = group == k
            if np.sum(keep) >= 9800:
                block.append(np.stack([np.mean((v[keep] - truth[keep]) ** 2, axis=0) for v in (baseline, assisted)]))
        b = np.asarray(block)
        if len(b) < 8:
            raise ValueError("insufficient held-out blocks for error uncertainty")
        draws = rng.integers(0, len(b), (4000, len(b)))
        mean = b[draws].mean(axis=1)
        gain = 100 * (1 - np.sqrt(mean[:, 1, :] / mean[:, 0, :]))
        for axis in range(2):
            rows.append(
                dict(
                    frames_ahead=ahead,
                    axis=axis,
                    blocks=len(b),
                    block_seconds=5,
                    rms_improvement_percent=float(
                        100
                        * (
                            1
                            - np.sqrt(
                                np.mean((assisted[:, axis] - truth[:, axis]) ** 2)
                                / np.mean((baseline[:, axis] - truth[:, axis]) ** 2)
                            )
                        )
                    ),
                    conditional_ci95_low=float(np.quantile(gain[:, axis], 0.025)),
                    conditional_ci95_high=float(np.quantile(gain[:, axis], 0.975)),
                )
            )
    return pd.DataFrame(rows)


def analyze(dataset, output, plots=True):
    """Audit local files and compare metadata-clock predictors; never certify physical exposure/clock semantics."""
    root, output = Path(dataset).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("output must be outside original dataset")
    before = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
    acc, at, aw, ab = timed_stream(root, "acc000")
    wfs, wt, ww, wb = timed_stream(root, "wfs_tiptilt")
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("wfs_modes") != dict(tip=0, tilt=1):
        raise ValueError("WFS tip/tilt mode identity required")
    overlap_start, overlap_stop = max(int(at[0]), int(wt[0])), min(int(at[-1]), int(wt[-1]))
    start = max(overlap_start, int(pd.Timestamp(metadata["start_utc"]).value))
    stop = min(overlap_stop, int(pd.Timestamp(metadata["end_utc"]).value))
    if stop - start < 100_000_000_000:
        raise ValueError("need >=100 s chronological overlap for this study")
    epoch = overlap_start
    grid, x, y = spectral_grid(acc, at, wfs, wt, start, stop, epoch)
    pairs, spec = spectral_report(grid, x, y, start, stop)
    metrics, traces, alphas, split = predict_record(acc, at, wfs, wt, start, stop)
    band_scores = prediction_bands(traces, wt, epoch)
    uncertainty = block_improvement(traces, wt)
    output.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output / "coherence_phase.csv", index=False)
    metrics.to_csv(output / "prediction_metrics.csv", index=False)
    band_scores.to_csv(output / "prediction_bands.csv", index=False)
    alphas.to_csv(output / "selected_ridge.csv", index=False)
    uncertainty.to_csv(output / "prediction_improvement.csv", index=False)
    pd.DataFrame(ab + wb).to_csv(output / "timing_boundaries.csv", index=False)
    for c, s in enumerate(spec["full"]):
        columns = dict(frequency_hz=s["f"], accel_psd=s["pxx"][:, 0])
        for a in range(2):
            columns.update(
                {
                    f"wfs_psd_{a}": s["pyy"][:, a],
                    f"coherence_{a}": s["coherence"][:, a],
                    f"csd_real_{a}": s["pxy"][:, a].real,
                    f"csd_imag_{a}": s["pxy"][:, a].imag,
                    f"phase_deg_{a}": np.where(s["accepted"][:, a], np.angle(s["h"][:, a], deg=True), np.nan),
                }
            )
        pd.DataFrame(columns).to_csv(output / f"spectrum_accel{c}.csv", index=False)
    # Keep exact query/target/acquisition evidence for the nominal two-frame comparison.
    indices, truth, _ = traces[2, "wfs_only"]
    ai = np.searchsorted(at, wt[indices], side="right") - 1
    log = dict(
        query_ns=wt[indices],
        target_ns=wt[indices + 2],
        latest_accel_ns=at[ai],
        tip_truth=truth[:, 0],
        tilt_truth=truth[:, 1],
    )
    for name in MODELS:
        log[name + "_tip"] = traces[2, name][2][:, 0]
        log[name + "_tilt"] = traces[2, name][2][:, 1]
    pd.DataFrame(log).to_csv(output / "heldout_predictions.csv", index=False)
    if plots:
        make_plots(output, spec, pairs, metrics, band_scores, traces, at, wt, epoch)
    after = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
    if before != after:
        raise ValueError("input changed during analysis")
    report = dict(
        evidence="real recorded ADC/WFS, conditional recorded-clock alignment",
        input_files_unchanged=len(before),
        physical_clock_and_exposure_semantics_verified=False,
        timestamps_fabricated=False,
        overlap_start_ns=overlap_start,
        overlap_end_ns=overlap_stop,
        overlap_start_utc=utc(overlap_start),
        overlap_end_utc=utc(overlap_stop),
        overlap_span_ns=overlap_stop - overlap_start,
        analysis_start_ns=start,
        analysis_end_exclusive_ns=stop,
        common_epoch_ns=epoch,
        accel=cadence(at, aw),
        wfs=cadence(wt, ww),
        split=split,
        scope="Acquisition-field alignment, not absolute optical phase/DM calibration. Causal features assume recorded-field clock comparability and availability; unknown exposure/processing offsets remain.",
        method="64-frame standardized ridge AR/ARX; fixed strengths selected on earlier validation, refit pre-test, no recursive feedback or production controller change",
    )
    age = wt[indices] - at[ai]
    report["latest_accel_age_at_heldout_query_ns"] = {
        str(q): float(np.quantile(age, q)) for q in (0, 0.01, 0.5, 0.99, 1)
    }
    report["coherence_interpretation"] = (
        "Recorded-field clock comparability is assumed, not independently established; constant clock/exposure offsets alter phase and causal availability. No signal-derived clock correction applied."
    )
    report["absolute_phase_sensitivity_per_1ms_offset_deg"] = {
        str(f): float(360 * f * 0.001) for f in sorted(pairs[pairs.interval == "train"].frequency_hz.unique())
    }
    report["query_availability_assumption"] = (
        "Latest measurements are available at their recorded acquisition/write fields; actual receipt/readout latency is unrecorded."
    )
    (output / "input_sha256.json").write_text(json.dumps(before, indent=2) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def make_plots(output, spectra_by_interval, pairs, metrics, band_scores, traces, at, wt, epoch):
    """Save acquisition-clock scientific diagnostics without claiming exposure-center phase calibration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for c, s in enumerate(spectra_by_interval["full"]):
        for a in range(2):
            axes[0, a].semilogy(s["f"], s["pxx"][:, a], label=f"ADC {c} [counts²/Hz]")
            axes[1, a].plot(s["f"], s["coherence"][:, a], label=f"ADC {c}")
            axes[2, a].plot(
                s["f"], np.where(s["accepted"][:, a], np.angle(s["h"][:, a], deg=True), np.nan), label=f"ADC {c}"
            )
            if c == 0:
                optical = axes[0, a].twinx()
                optical.semilogy(s["f"], s["pyy"][:, a], color="black", alpha=0.5)
                optical.set_ylabel("WFS [recorded modal²/Hz]")
    for a in range(2):
        axes[0, a].set_ylabel("ADC [counts²/Hz]")
        axes[0, a].set_title("Tip" if a == 0 else "Tilt")
        axes[1, a].set_ylabel("Magnitude-squared coherence")
        axes[1, a].legend()
        axes[2, a].set(
            xlabel="Frequency [Hz, recorded acquisition clock]", ylabel="CSD phase [deg, coherence ≥0.8]", xlim=(20, 65)
        )
    fig.suptitle("REAL telemetry; physical clock/exposure-center registration UNVERIFIED")
    fig.tight_layout()
    fig.savefig(output / "spectra_coherence_phase.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for t, label in ((at, "ADC"), (wt, "WFS")):
        d = np.diff(t) * 1e-3
        axes[0].hist(d, bins=100, histtype="step", label=label)
        subset = t[::1000]
        n = np.arange(len(t))[::1000]
        axes[1].plot((subset - epoch) * 1e-9, (subset - t[0] - n * STEP_NS) * 1e-3, label=label)
    axes[0].set(xlabel="Acquisition interval [us]", ylabel="Count")
    axes[0].legend()
    axes[1].set(xlabel="Recorded time [s]", ylabel="Residual from nominal counter grid [us]")
    axes[1].legend()
    fig.suptitle("Cadence variation is not an independently calibrated clock difference")
    fig.tight_layout()
    fig.savefig(output / "timing.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for a in range(2):
        for name, rows in metrics[(metrics.guard_ns == 0) & (metrics.axis == a)].groupby("model"):
            axes[a].plot(rows.expected_horizon_ms, rows.rmse, "o-", label=name)
        axes[a].set(
            xlabel="Expected frames-ahead horizon [ms; actual times retained]",
            ylabel="Held-out RMS error [modal units]",
        )
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "horizon_errors.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for a in range(2):
        for j, name in enumerate(("persistence", "wfs_only", "wfs_accel")):
            rows = band_scores[(band_scores.frames_ahead == 2) & (band_scores.axis == a) & (band_scores.model == name)]
            axes[a].bar(
                np.arange(2) + (j - 1) * 0.25, rows.error_band_rms / rows.truth_band_rms, width=0.25, label=name
            )
        axes[a].set(xticks=[0, 1], xticklabels=["27–30 Hz", "54–58 Hz"], ylabel="Error RMS / measured band RMS")
        axes[a].set_title("Tip" if a == 0 else "Tilt")
    axes[0].legend()
    fig.suptitle("Held-out two-frame forecast; broadband model need not beat persistence in every band")
    fig.tight_layout()
    fig.savefig(output / "band_errors.png")
    plt.close(fig)
    indices, truth, _ = traces[2, "wfs_only"]
    time = (wt[indices + 2] - epoch) * 1e-9
    show = (time >= time[0] + 1) & (time < time[0] + 1.3)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for a in range(2):
        axes[a].plot(time[show], truth[show, a], label="Measured target frame", color="black", lw=1)
        for name in ("wfs_only", "wfs_accel"):
            axes[a].plot(time[show], traces[2, name][2][show, a], label=name, alpha=0.7)
        axes[a].set_ylabel("Recorded modal units")
        axes[a].legend()
    axes[-1].set_xlabel("Target acquisition-field time [s from common epoch]")
    fig.tight_layout()
    fig.savefig(output / "heldout_prediction.png")
    plt.close(fig)


def main():
    """Analyze only local data and original timing sidecars, preserving all input bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(analyze(args.dataset, args.output, not args.no_plots), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
