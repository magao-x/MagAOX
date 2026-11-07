#!/usr/bin/env python3
r"""\file validate_phase_calibration.py
\brief Earlier-fit per-frequency calibration of existing causal harmonic quadratures and normalized shadow replay.
\ingroup accelFFCtrl_unit_test

Only recorded ADC0/1 are used. Coefficients describe recorded WFS units, not
physical acceleration, pointing radians or a DM calibration. No estimator changes.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_timed_telemetry import block_improvement, timed_stream
from inspect_reduced_telemetry import sha256
from validate_real_harmonic import (
    HORIZON,
    baselines,
    fit_map,
    make_plots,
    normalized_manifest,
    phase_diagnostic_features,
    refined_carriers,
    run_cpp,
    scores,
    target_labels,
)


def fit_calibration(quadratures, truth, selected):
    """Fit eight finite origin-preserving coefficients on explicitly selected earlier rows, rejecting rank loss."""
    x, y = quadratures[selected], truth[selected]
    if x.ndim != 2 or x.shape[1] != 4 or y.shape != (len(x), 2) or not len(x):
        raise ValueError("invalid paired-band calibration rows")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("nonfinite calibration data")
    coefficient, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    if rank != 4 or singular[-1] / singular[0] < 1e-8 or not np.all(np.isfinite(coefficient)):
        raise ValueError("unconditioned phase calibration")
    return coefficient


def shadow_checks(data, forecast, sigma, gain=1, bounds=(3, 3)):
    """Audit the actual C++ calibrated shadow against exact opposing gain, common clipping, timing and inhibition."""
    valid = data.valid.to_numpy().astype(bool)
    proposed = data[["mode0", "mode1"]].to_numpy()
    numerical = -gain * forecast / sigma
    ratios = np.divide(bounds, np.abs(numerical), out=np.ones_like(numerical), where=numerical != 0)
    scale = np.minimum(1, np.min(ratios, axis=1))
    expected = scale[:, None] * numerical
    expected[~valid] = 0
    np.testing.assert_allclose(data.loc[valid, ["theta_x", "theta_y"]], forecast[valid], rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(proposed, expected, rtol=1e-11, atol=1e-13)
    np.testing.assert_array_equal(data.target_ns - data.query_ns, np.full(len(data), HORIZON))
    if np.any(np.abs(proposed) > np.asarray(bounds) * (1 + 1e-12)):
        raise ValueError("normalized bound violation")
    if np.any(np.sum(proposed * forecast / sigma, axis=1) > 1e-12):
        raise ValueError("incorrect normalized correction sign")
    return dict(
        gain=gain,
        maximum_absolute=np.max(np.abs(proposed), axis=0).tolist(),
        clipped=int(np.sum(data.state == 3)),
        invalid_nonzero=int(np.count_nonzero(proposed[~valid])),
        inhibited=int(np.sum(data.state == 1)),
    )


def analyze(dataset, executable, output, plots=True, export_response=False):
    """Freeze an initial two-minute calibration, score later time blocks and repeat predeclared forward-only fits."""
    root, output = Path(dataset).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("output must be outside original data")
    before = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
    output.mkdir(parents=True, exist_ok=True)
    acc, at, written, _ = timed_stream(root, "acc000")
    acc = acc[:, :2]  # Every subsequent spectral, baseline, calibration and controller calculation uses ADC0/1 only.
    wfs, wt, _, _ = timed_stream(root, "wfs_tiptilt")
    metadata = json.loads((root / "metadata.json").read_text())
    start, stop = [int(pd.Timestamp(metadata[k]).value) for k in ("start_utc", "end_utc")]
    epoch = int(at[0])
    fit_end, test_start = start + 120_000_000_000, start + 150_000_000_000
    indices = np.flatnonzero((wt >= start) & (wt + 10_000_000 < stop))
    query, target = wt[indices], wt[indices] + HORIZON
    truth, right = target_labels(wfs, wt, target, epoch)
    train = right < fit_end
    test = query >= test_start
    queries = output / "queries_absolute.csv"
    pd.DataFrame(dict(query_ns=query)).to_csv(queries, index=False)
    counters = np.load(root / "acc000_timings.npy", mmap_mode="r", allow_pickle=False)[:, 0]
    adc = pd.DataFrame(
        dict(
            acquired_ns=at,
            written_ns=written,
            receipt_ns=written,
            counter=counters,
            generation=1,
            adc_x=acc[:, 0].astype(np.uint16),
            adc_y=acc[:, 1].astype(np.uint16),
        )
    )
    # Carrier selection uses initial training ADC only; the same carriers stay fixed for every later fold.
    frequencies = refined_carriers(acc, at, wfs, wt, start, fit_end, epoch)
    manifest = normalized_manifest(queries, frequencies)
    raw = run_cpp(executable, manifest, adc, epoch, output)
    valid = raw.valid.to_numpy().astype(bool)
    quadratures = phase_diagnostic_features(raw, frequencies)
    coefficient = fit_calibration(quadratures, truth, train & valid)
    forecast = quadratures @ coefficient
    forecast[~valid] = 0
    sigma = np.std(truth[train], axis=0)
    # The same total training interval is available to baseline models, with internal 90/30-s ridge selection.
    initial = right < start + 90_000_000_000
    validation = (query >= start + 90_000_000_000) & train
    evaluate = np.ones(len(query), dtype=bool) if export_response else test
    predictions = baselines(acc, at, wfs, wt, indices, truth, initial, validation, train, evaluate)
    original = raw[["theta_x", "theta_y"]].to_numpy()
    real_map = fit_map(original[train & valid], truth[train & valid], "direct")
    predictions["original_real_map"] = original @ real_map.T
    predictions["phase_calibrated"] = forecast
    metrics, spectra_by_model = [], {}
    for name, prediction in predictions.items():
        rows, spectrum = scores(prediction[test], truth[test], target[test], epoch, name)
        metrics.extend(rows)
        spectra_by_model[name] = spectrum
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    cohort = np.flatnonzero(test)
    uncertainty = block_improvement(
        {
            (HORIZON, "wfs_only"): (cohort, truth[test], predictions["original_real_map"][test]),
            (HORIZON, "wfs_accel"): (cohort, truth[test], forecast[test]),
        },
        query,
    ).rename(columns={"frames_ahead": "horizon_ns"})
    uncertainty.to_csv(output / "rms_improvement_bootstrap.csv", index=False)

    # Report every predeclared later portion; none can select or modify the main calibration.
    chronological = []
    for first, last in ((120, 150), (150, 210), (210, 270), (270, 300)):
        keep = (query >= start + first * 1_000_000_000) & (query < start + last * 1_000_000_000)
        for name in ("original_real_map", "phase_calibrated"):
            rows, _ = scores(predictions[name][keep], truth[keep], target[keep], epoch, name)
            chronological.extend([dict(row, first_s=first, last_s=last) for row in rows])
    pd.DataFrame(chronological).to_csv(output / "frozen_chronological.csv", index=False)
    forward, fold_coefficients = [], []
    for first in (0, 60, 120):
        end = first + 120
        fit = (query >= start + first * 1_000_000_000) & (right < start + end * 1_000_000_000) & valid
        keep = (query >= start + end * 1_000_000_000) & (query < start + (end + 60) * 1_000_000_000)
        c = fit_calibration(quadratures, truth, fit)
        p = quadratures @ c
        p[~valid] = 0
        m = fit_map(original[fit], truth[fit], "direct")
        for name, prediction in (("original_real_map", original @ m.T), ("phase_calibrated", p)):
            rows, _ = scores(prediction[keep], truth[keep], target[keep], epoch, name)
            forward.extend([dict(row, train_first_s=first, train_last_s=end, test_last_s=end + 60) for row in rows])
        fold_coefficients.append(
            dict(train_first_s=first, train_last_s=end, weights=c.tolist(), last_label_right_ns=int(right[fit].max()))
        )
    pd.DataFrame(forward).to_csv(output / "forward_folds.csv", index=False)

    manifest["phase_calibration"] = coefficient.tolist()
    manifest["shadow"]["modal_map"] = np.diag(1 / sigma).tolist()
    manifest["shadow"][
        "provenance"
    ] += " Earlier-fit paired-band phase coefficients yield recorded WFS units; normalized virtual identity response only."
    disabled = run_cpp(executable, manifest, adc, epoch, output)
    shadow_checks(disabled, forecast, sigma, gain=0)
    disabled_manifest = copy.deepcopy(manifest)
    manifest["shadow"]["gain"] = 1
    shadow = run_cpp(executable, manifest, adc, epoch, output)
    audit = shadow_checks(shadow, forecast, sigma)
    active = shadow.valid.to_numpy().astype(bool)
    sample = shadow.sample_ns.to_numpy() - 1_000_000_000 + epoch
    log = pd.DataFrame(
        dict(
            query_ns=query,
            target_ns=target,
            sample_ns=sample,
            counter=shadow.counter,
            valid=shadow.valid,
            state=shadow.state,
            scale=shadow.scale,
            predicted_tip=forecast[:, 0],
            predicted_tilt=forecast[:, 1],
            observed_tip=truth[:, 0],
            observed_tilt=truth[:, 1],
            normalized_tip=shadow.mode0,
            normalized_tilt=shadow.mode1,
        )
    )
    log.to_csv(output / "normalized_shadow.csv", index=False)
    if export_response:
        # Serialized qualified numerical outputs only; no new predictor and no physical DM calibration.
        wfs_counters = np.load(root / "wfs_tiptilt_timings.npy", mmap_mode="r", allow_pickle=False)[indices, 0]
        np.savez(
            output / "response_inputs.npz",
            query_ns=query,
            target_ns=target,
            sample_ns=sample,
            counter=shadow.counter.to_numpy(),
            generation=shadow.generation.to_numpy(),
            valid=shadow.valid.to_numpy(),
            reference=wfs[indices],
            wfs_counter=wfs_counters,
            train=train,
            test=test,
            sigma=sigma,
            wfs_times_ns=wt,
            wfs_values=wfs,
            **{name: predictions[name] for name in ("wfs_only", "regression_01", "phase_calibrated")},
        )
    if plots:
        make_plots(
            output,
            predictions,
            truth,
            target,
            test,
            epoch,
            spectra_by_model,
            log,
            names=("wfs_only", "regression_01", "original_real_map", "phase_calibrated"),
        )
    after = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
    if before != after:
        raise ValueError("original inputs changed")
    age = query[active & test] - sample[active & test]
    report = dict(
        input_files_unchanged=len(before),
        adc_channels=[0, 1],
        physical_calibration=False,
        production_model_changed=False,
        horizon_ns=HORIZON,
        common_epoch_ns=epoch,
        adc_availability_latency_ns_assumed=0,
        wfs_availability_latency_ns_assumed=0,
        exposure_center_offset_ns_assumed=0,
        physical_clock_registration_verified=False,
        fit_end_ns=fit_end,
        last_calibration_label_right_ns=int(right[train & valid].max()),
        test_start_ns=test_start,
        count_test=int(test.sum()),
        frequencies_hz=frequencies,
        weights=coefficient.tolist(),
        weights_basis="[-A0*cos(psi0)/omega0²,-A0*sin(psi0)/omega0²,-A1*cos(psi1)/omega1²,-A1*sin(psi1)/omega1²]",
        effective_complex_gain=[
            dict(
                frequency_hz=f,
                channel=i,
                magnitude=np.hypot(coefficient[2 * i], coefficient[2 * i + 1]).tolist(),
                phase_deg=np.degrees(np.arctan2(-coefficient[2 * i + 1], coefficient[2 * i])).tolist(),
            )
            for i, f in enumerate(frequencies)
        ],
        original_real_map=real_map.tolist(),
        training_wfs_sigma=sigma.tolist(),
        forward_fold_calibrations=fold_coefficients,
        shadow=dict(
            audit,
            gain_default=0,
            heldout_valid_fraction=float(active[test].mean()),
            processed=int(shadow.processed.iloc[-1]),
            dropped=int(shadow.dropped.iloc[-1]),
            resets=int(shadow.resets.iloc[-1]),
            heldout_age_ns_quantiles=np.quantile(age, [0, 0.5, 0.99, 1]).tolist(),
        ),
    )
    for name, value in (
        ("report", report),
        ("disabled_manifest", disabled_manifest),
        ("shadow_manifest", manifest),
        ("input_sha256", before),
    ):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return report


def main():
    """Run only local file replay and chronological analysis with the unchanged harmonic estimator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(analyze(args.dataset, args.replay, args.output, not args.no_plots), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
