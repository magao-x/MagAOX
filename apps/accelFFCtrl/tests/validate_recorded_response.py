#!/usr/bin/env python3
r"""\file validate_recorded_response.py
\brief Frozen recorded-data predictors through existing normalized shadow and DM response; additive replay, not feedback.
\ingroup accelFFCtrl_unit_test
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd

from inspect_reduced_telemetry import sha256
from validate_phase_calibration import analyze as phase_replay
from validate_real_harmonic import HORIZON, scores

METHODS = ("wfs_only", "phase_calibrated", "regression_01")
GAINS = (0, 0.5, 1)


def response(executable, forecasts, output, sigma, gain, steps=2, bandwidth=0, sign=1):
    """Run the existing C++ shadow/proposer and dmResponse with explicit normalized identity response settings."""
    output = Path(output)
    parameters = output.with_suffix(".txt")
    parameters.write_text(" ".join(map(str, [gain, *sigma, 3, 3, HORIZON, steps, bandwidth, sign])) + "\n")
    subprocess.run([str(executable), str(parameters), str(forecasts), str(output)], check=True)
    d = pd.read_csv(output, dtype={"query_ns": "int64", "target_ns": "int64", "scale": "float64"})
    if not np.all(np.isfinite(d.to_numpy())):
        raise ValueError("nonfinite response")
    np.testing.assert_array_equal(d.target_ns - d.query_ns, np.full(len(d), HORIZON))
    if np.any(np.abs(d[["mode0", "mode1"]].to_numpy()) > 3 * (1 + 1e-12)):
        raise ValueError("normalized bound violation")
    if np.any(d.loc[d.state.isin([0, 1]), ["mode0", "mode1"]].to_numpy() != 0):
        raise ValueError("nonzero disabled/inhibited contribution")
    return d


def metrics(reference, effect, times, epoch, name):
    """Reuse target-aligned PSD scoring for an additive residual; quantify outside-band WFS power separately."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rows, spectra = scores(-effect, reference, times, epoch, name)
    f, before, after = spectra
    outside = ~(((f >= 27) & (f <= 30)) | ((f >= 54) & (f <= 58)))
    for axis, row in enumerate(rows):
        row["residual_bias"] = -row.pop("bias")
        row.pop("correlation")  # Undefined for zero correction; not a residual-performance metric.
        for band in (0, 1):
            row[f"band{band}_suppression_db"] = row.pop(f"band{band}_ideal_suppression_db")
        row["outside_power_change_percent"] = float(
            100 * (after[outside, axis].sum() / before[outside, axis].sum() - 1)
        )
        for lo, hi in ((0, 20), (20, 27), (30, 54), (58, 200), (200, 1000)):
            keep = (f >= lo) & (f < hi) & outside
            row[f"reference_rms_{lo}_{hi}"] = float(np.sqrt(before[keep, axis].sum() * (f[1] - f[0])))
            row[f"residual_rms_{lo}_{hi}"] = float(np.sqrt(after[keep, axis].sum() * (f[1] - f[0])))
            row[f"power_change_{lo}_{hi}_percent"] = float(
                100 * (after[keep, axis].sum() / before[keep, axis].sum() - 1)
            )
    return rows, spectra


def choose_gain(training):
    """Select only training rows: minimum normalized MSE, <=1% RMS increase and <=5% outside-band power increase."""
    choices = []
    for gain, rows in training.groupby("gain", sort=True):
        if len(rows) != 2 or np.any(rows.nrmse > 1.01) or np.any(rows.outside_power_change_percent > 5):
            continue
        choices.append((float(np.mean(rows.nrmse**2)), float(gain)))
    if not choices:
        raise ValueError("no eligible training gain, including zero")
    return min(choices)[1]


def analyze(dataset, replay_executable, response_executable, output):
    """Replay the unchanged earlier-fit predictors, select small training-only gains and score frozen additive responses."""
    root, output = Path(dataset).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("output must be outside original data")
    output.mkdir(parents=True, exist_ok=True)
    predictor = output / "predictor"
    provenance = phase_replay(dataset, replay_executable, predictor, plots=False, export_response=True)
    with np.load(predictor / "response_inputs.npz", allow_pickle=False) as exported:
        data = {key: exported[key] for key in exported.files}
    query, reference, sigma = data["query_ns"], data["reference"], data["sigma"]
    epoch, train, test = provenance["common_epoch_ns"], data["train"], data["test"]
    selected, results, training, spectra, selected_traces = {}, [], [], {}, {}
    rows, spec = metrics(reference[test], np.zeros_like(reference[test]), query[test], epoch, "no_correction")
    results.extend([dict(row, scenario="aoSim_lag1", gain=0) for row in rows])
    spectra["no_correction"] = spec
    with tempfile.TemporaryDirectory(prefix="response-", dir=output) as work:
        work = Path(work)
        sources = {}
        for name in METHODS:
            harmonic = name == "phase_calibrated"
            frame = pd.DataFrame(
                dict(
                    query_ns=query,
                    sample_ns=data["sample_ns"] if harmonic else query,
                    target_ns=data["target_ns"],
                    counter=data["counter"] if harmonic else data["wfs_counter"],
                    generation=data["generation"] if harmonic else 1,
                    valid=data["valid"] if harmonic else 1,
                    tip=data[name][:, 0],
                    tilt=data[name][:, 1],
                    reference_tip=reference[:, 0],
                    reference_tilt=reference[:, 1],
                )
            )
            sources[name] = work / f"{name}.csv"
            frame.to_csv(sources[name], index=False)
            runs, tuning = {}, []
            for gain in GAINS:
                d = response(response_executable, sources[name], work / f"response_{name}_{gain}.csv", sigma, gain)
                effect = d[["effect_tip", "effect_tilt"]].to_numpy()
                rows, _ = metrics(reference[train], effect[train], query[train], epoch, name)
                tuning.extend([dict(row, gain=gain) for row in rows])
                runs[gain] = d
            tuning = pd.DataFrame(tuning)
            gain = choose_gain(tuning)
            selected[name] = gain
            training.append(tuning)
            d = runs[gain]
            selected_traces[name] = d
            effect = d[["effect_tip", "effect_tilt"]].to_numpy()
            rows, spec = metrics(reference[test], effect[test], query[test], epoch, name)
            results.extend([dict(row, scenario="aoSim_lag1", gain=gain) for row in rows])
            spectra[name] = spec
            d.to_csv(output / f"{name}_selected_response.csv", index=False)
        # No sensitivity case may change the trained gain or prediction target.
        for scenario, steps, bandwidth, sign in (
            ("aoSim_lag0", 1, 0, 1),
            ("aoSim_lag2", 3, 0, 1),
            ("nominal_pole300Hz", 1, 300, 1),
            ("wrong_sign", 2, 0, -1),
        ):
            for name in METHODS:
                if sign == -1 and name != "phase_calibrated":
                    continue
                d = response(
                    response_executable,
                    sources[name],
                    work / "sensitivity.csv",
                    sigma,
                    selected[name],
                    steps,
                    bandwidth,
                    sign,
                )
                effect = d[["effect_tip", "effect_tilt"]].to_numpy()
                rows, _ = metrics(reference[test], effect[test], query[test], epoch, name)
                results.extend([dict(row, scenario=scenario, gain=selected[name]) for row in rows])
    for offset in (-1000000, -500000, 500000, 1000000):
        labels = np.column_stack(
            [
                np.interp(
                    (query[test] + offset - epoch) * 1e-9,
                    (data["wfs_times_ns"] - epoch) * 1e-9,
                    data["wfs_values"][:, a],
                )
                for a in range(2)
            ]
        )
        for name in METHODS:
            effect = selected_traces[name][["effect_tip", "effect_tilt"]].to_numpy()[test]
            rows, _ = metrics(labels, effect, query[test], epoch, name)
            results.extend([dict(row, scenario=f"reference_offset_{offset}ns", gain=selected[name]) for row in rows])
    ideal_reference = np.column_stack(
        [
            np.interp(
                (data["target_ns"][test] - epoch) * 1e-9,
                (data["wfs_times_ns"] - epoch) * 1e-9,
                data["wfs_values"][:, a],
            )
            for a in range(2)
        ]
    )
    for name in METHODS:
        effect = selected_traces[name][["mode0", "mode1"]].to_numpy()[test] * sigma
        rows, _ = metrics(ideal_reference, effect, data["target_ns"][test], epoch, name)
        results.extend([dict(row, scenario="ideal_target_addition", gain=selected[name]) for row in rows])
    table = pd.DataFrame(results)
    table.to_csv(output / "metrics.csv", index=False)
    pd.concat(training, ignore_index=True).to_csv(output / "training_gains.csv", index=False)
    plot_results(output, query, reference, test, epoch, selected_traces, spectra)
    nominal = table[table.scenario == "aoSim_lag1"]
    target_error = query[2:] - data["target_ns"][:-2]
    audit = {}
    for name, d in selected_traces.items():
        commands = d[["mode0", "mode1"]].to_numpy()
        # Independent exact delay and virtual normalization audit, retaining original irregular query times.
        np.testing.assert_allclose(
            d[["effect_tip", "effect_tilt"]].to_numpy()[2:], commands[:-2] * sigma, rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            d[["residual_tip", "residual_tilt"]], reference + d[["effect_tip", "effect_tilt"]], rtol=1e-12, atol=1e-14
        )
        audit[name] = dict(
            maximum_command=np.max(np.abs(commands), axis=0).tolist(),
            clipped=int(np.sum(d.state == 3)),
            inhibited=int(np.sum(d.state == 1)),
        )
    repo = Path(__file__).resolve().parents[3]
    report = dict(
        scope="Recorded WFS plus synthetic DM response; NOT a feedback simulation or physical suppression",
        adc_channels=[0, 1],
        gain_candidates=GAINS,
        selected_training_gains=selected,
        training_gain_guard="<=1% RMS increase per axis and <=5% total outside-band WFS power increase",
        gain_selection="Minimum mean squared per-axis NRMSE on initial 120 s only; in-sample calibration/gain fit",
        horizon_ns=HORIZON,
        aoSim_lag=1,
        delay_steps=2,
        nominal_period_ns=500000,
        actual_target_error_ns_quantiles=np.quantile(target_error, [0, 0.01, 0.5, 0.99, 1]).tolist(),
        physical_calibration=False,
        actual_feedback_simulation=False,
        timing_assumptions="Comparable recorded clocks, availability at acquisition fields, zero exposure-center offset; unverified",
        finite_bandwidth_case="Existing dmResponse with nominal 0.5-ms step, 0.5-ms pure delay and 300-Hz pole; synthetic only",
        reference_offsets="Frozen-input mismatch sensitivity: positive means later WFS recording labels; not inferred physical delay",
        count_test=int(test.sum()),
        shadow_audits=audit,
        nominal_metrics=nominal.to_dict(orient="records"),
        source_sha256={
            str(p.relative_to(repo)): sha256(p)
            for p in (
                repo / "apps/aoSim/xapp/aoSim/__init__.py",
                repo / "apps/accelFFCtrl/offline/accelFFCtrlCancellation.hpp",
            )
        },
        predictor_report=provenance,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def plot_results(output, query, reference, test, epoch, traces, spectra):
    """Export recorded-reference/residual PSDs, time series and normalized absolute commands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = (query - epoch) * 1e-9
    first = time[test][0]
    show = test & (time >= first + 1) & (time < first + 1.3)
    for kind in ("residual", "commands"):
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for a in range(2):
            if kind == "residual":
                axes[a].plot(
                    time[show], reference[show, a], color="black", alpha=0.5, label="Recorded WFS, no correction"
                )
            else:
                axes[a].axhline(3, color="gray", ls=":")
                axes[a].axhline(-3, color="gray", ls=":")
            for name, d in traces.items():
                field = ("residual_tip", "residual_tilt")[a] if kind == "residual" else f"mode{a}"
                axes[a].plot(time[show], d.loc[show, field], label=name, alpha=0.8)
            axes[a].set(
                title=("Tip", "Tilt")[a], ylabel="Recorded WFS units" if kind == "residual" else "Training WFS SD"
            )
            axes[a].legend(fontsize=8)
        axes[-1].set_xlabel("Response query time [s from common epoch]")
        fig.suptitle("Additive normalized simulation, no feedback")
        fig.tight_layout()
        fig.savefig(output / f"{kind}_timeseries.png")
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for a in range(2):
        for name, (f, _, residual) in spectra.items():
            axes[a].semilogy(f, residual[:, a], label=name)
        axes[a].set(xlim=(10, 200), ylabel="PSD [recorded WFS units²/Hz]", title=("Tip", "Tilt")[a])
        axes[a].legend(fontsize=8)
    axes[-1].set_xlabel("Frequency [Hz, recorded clock]")
    fig.tight_layout()
    fig.savefig(output / "residual_psds.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for a in range(2):
        for band, bounds in enumerate(((20, 35), (48, 63))):
            for name, (f, _, residual) in spectra.items():
                axes[a, band].semilogy(f, residual[:, a], label=name)
            axes[a, band].set(
                xlim=bounds,
                title=("Tip", "Tilt")[a],
                xlabel="Frequency [Hz, recorded clock]",
                ylabel="PSD [recorded WFS units²/Hz]",
            )
            axes[a, band].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "target_psds.png")
    plt.close(fig)


def main():
    """Run only local numerical file tools; no shared-memory or instrument access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.dataset, args.replay, args.response, args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
