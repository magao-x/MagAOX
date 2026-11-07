#!/usr/bin/env python3
r"""\file validate_closed_loop.py
\brief Frozen simulation calibration and genuine corrected-WFS feedback using existing numerical components.
\ingroup accelFFCtrl_unit_test
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from inspect_reduced_telemetry import sha256
from validate_phase_calibration import fit_calibration
from validate_predictions import RAD_TO_MAS
from validate_recorded_response import metrics

ORIGIN = 1_000_000_000
PERIOD = 500_000
CASES = (
    "none",
    "wfs_only",
    "harmonic_only",
    "wfs_harmonic",
    "shadow_disabled",
    "ff_off",
    "wrong_sign",
    "wfs_only_gain03",
    "wfs_harmonic_gain03",
    "high_gain08",
)
AO_GAINS = dict(zip(CASES, (0, 0.1, 0, 0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.8)))


def prepare_calibration(executable, directory, seconds, population, seed):
    """Fit the existing eight-coefficient paired map on completed early simulator labels only; never load real-data gains."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(executable), "prepare", str(directory), str(seconds), population, str(seed)], check=True)
    training = pd.read_csv(
        directory / "training.csv", dtype={k: "int64" for k in ("query_ns", "sample_ns", "target_ns")}
    )
    boundary = ORIGIN + seconds * 8_000 // 3 * 125_000
    if training.empty or np.any(training.target_ns >= boundary) or np.any(training.sample_ns >= training.query_ns):
        raise ValueError("training labels or sample availability crossed a causal boundary")
    x = training[["q0_cos", "q0_sin", "q1_cos", "q1_sin"]].to_numpy()
    y = training[["tip_rad", "tilt_rad"]].to_numpy()
    weights = fit_calibration(x, y, np.ones(len(x), dtype=bool))
    sigma = np.std(y, axis=0)
    lower = 28 if population == "full" else 28.5
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("invalid training WFS normalization")
    values = ["accelFFCtrl-sim-phase-v1", 1000000, 10, lower, 56, *sigma]
    for band in range(2):
        values.extend(weights[2 * band : 2 * band + 2].T.ravel())
    (directory / "calibration.txt").write_text(" ".join(map(str, values)) + "\n")
    result = dict(
        kind="Simulation-only, radians-to-radians calibration; not real telemetry or physical DM calibration",
        policy="ADC0 at lower band, ADC1 at 56 Hz; eight coefficients unchanged from paired replay topology",
        seconds=seconds,
        population=population,
        seed=seed,
        fit_end_ns=boundary,
        last_label_ns=int(training.target_ns.max()),
        lower_frequency_hz=lower,
        weights=weights.tolist(),
        sigma_rad=sigma.tolist(),
        source_sha256=sha256(directory / "source.bin"),
    )
    (directory / "calibration.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def audit_trace(data, sigma, gain, seconds):
    """Prove the actual previous corrected WFS drives the integrator and delayed bounded commands drive later WFS."""
    q = data.query_ns.to_numpy()
    np.testing.assert_array_equal(np.diff(q), np.full(len(q) - 1, PERIOD))
    np.testing.assert_array_equal(data.target_ns - q, np.full(len(q), 1000000))
    np.testing.assert_array_equal(data.prior_wfs_ns.iloc[1:], q[:-1])
    residual = data[["residual_tip", "residual_tilt"]].to_numpy()
    previous = data[["prior_tip", "prior_tilt"]].to_numpy()
    np.testing.assert_array_equal(previous[0], [0, 0])
    np.testing.assert_allclose(previous[1:], residual[:-1], rtol=1e-13, atol=1e-23)
    control = q >= ORIGIN + int(seconds * 8000 // 3) * 125000
    ao = data[["ao0", "ao1"]].to_numpy()
    previous_ao = np.vstack((np.zeros(2), ao[:-1]))
    np.testing.assert_allclose(ao[control], (previous_ao - gain * previous / sigma)[control], rtol=1e-11, atol=1e-13)
    command = data[["total0", "total1"]].to_numpy()
    effect = data[["effect_tip", "effect_tilt"]].to_numpy()
    np.testing.assert_array_equal(effect[:2], np.zeros((2, 2)))
    np.testing.assert_allclose(effect[2:], command[:-2] * sigma, rtol=1e-12, atol=1e-23)
    disturbance = data[["disturbance_tip", "disturbance_tilt"]].to_numpy()
    np.testing.assert_allclose(residual, disturbance + effect, rtol=1e-12, atol=1e-23)
    ff = data[["ff0", "ff1"]].to_numpy()
    if np.any(np.abs(command) > 3 * (1 + 1e-12)) or np.any(np.abs(ff) > 3 * (1 + 1e-12)):
        raise ValueError("normalized command bound violated")
    np.testing.assert_array_equal(ff[data.ff_enabled == 0], np.zeros((np.sum(data.ff_enabled == 0), 2)))
    ready = data.ready.to_numpy().astype(bool)
    np.testing.assert_array_equal(data.sample_ns[ready], q[ready] - 125000)
    np.testing.assert_array_equal(data.counter[ready], (q[ready] - ORIGIN) // 125000)
    return dict(
        maximum_ao=float(np.max(np.abs(ao))),
        maximum_total=float(np.max(np.abs(command))),
        maximum_ff=float(np.max(np.abs(ff))),
        total_saturated=int(np.sum(data.total_scale < 1)),
        ff_limited=int(np.sum(data.ff_state == 3)),
        causal_feedback=True,
        two_frame_delay_verified=True,
    )


def analyze_run(directory, metadata, plots=False):
    """Score every predeclared scenario and preserve failures, including saturation-bounded linear instability."""
    directory = Path(directory)
    sigma = np.asarray(metadata["sigma_rad"])
    seconds = metadata["seconds"]
    traces = {
        name: pd.read_csv(
            directory / f"{name}.csv",
            dtype={k: "int64" for k in ("query_ns", "prior_wfs_ns", "sample_ns", "target_ns", "counter")},
        )
        for name in CASES
    }
    baseline = traces["wfs_only"]
    pd.testing.assert_frame_equal(baseline, traces["shadow_disabled"], check_exact=True)
    reference = traces["none"][["disturbance_tip", "disturbance_tilt"]].to_numpy()
    q = baseline.query_ns.to_numpy()
    test = q >= ORIGIN + int(seconds * 1e9 / 2)
    rows, spectrum, audits, blocks = [], {}, {}, []
    for name, data in traces.items():
        np.testing.assert_array_equal(data[["disturbance_tip", "disturbance_tilt"]], reference)
        audits[name] = audit_trace(data, sigma, AO_GAINS[name], seconds)
        y = data[["residual_tip", "residual_tilt"]].to_numpy()
        for comparison, base in (
            ("no_correction", reference),
            ("wfs_only", baseline[["residual_tip", "residual_tilt"]].to_numpy()),
        ):
            measured, spec = metrics(base[test] * RAD_TO_MAS, (y - base)[test] * RAD_TO_MAS, q[test], ORIGIN, name)
            f, before, after = spec
            outside = ~(((f >= 27) & (f <= 30)) | ((f >= 54) & (f <= 58)))
            for axis, row in enumerate(measured):
                row["outside_reference_rms_mas"] = float(np.sqrt(before[outside, axis].sum() * (f[1] - f[0])))
                row["outside_residual_rms_mas"] = float(np.sqrt(after[outside, axis].sum() * (f[1] - f[0])))
                for label, frequency in (("lower", metadata["lower_frequency_hz"]), ("56", 56)):
                    k = np.argmin(np.abs(f - frequency))
                    row[f"carrier_{label}_suppression_db"] = float(10 * np.log10(before[k, axis] / after[k, axis]))
            rows.extend([dict(row, comparison=comparison) for row in measured])
            if comparison == "no_correction":
                spectrum[name] = spec
        for second in range(seconds):
            selected = (q >= ORIGIN + second * 1_000_000_000) & (q < ORIGIN + (second + 1) * 1_000_000_000)
            if np.any(selected):
                rms = np.sqrt(np.mean(y[selected] ** 2, axis=0)) * RAD_TO_MAS
                blocks.append(dict(case=name, second=second, tip_mas=rms[0], tilt_mas=rms[1]))
        current = data.loc[test]
        audits[name].update(
            test_saturation_percent=float(100 * np.mean(current.total_scale < 1)),
            test_maximum_ao=float(np.max(np.abs(current[["ao0", "ao1"]].to_numpy()))),
        )
    # The actual feedback commands must change when FF changes the measured residual; FF itself stays ADC-only.
    active = traces["wfs_harmonic"]
    if np.max(np.abs(active[["ao0", "ao1"]].to_numpy() - baseline[["ao0", "ao1"]].to_numpy())) < 1e-8:
        raise ValueError("WFS feedback did not react to the feed-forward correction")
    np.testing.assert_array_equal(active[["ff0", "ff1"]], traces["harmonic_only"][["ff0", "ff1"]])
    off = traces["ff_off"]
    off_time = ORIGIN + int(seconds * 0.75 * 1e9)
    post = q >= off_time
    np.testing.assert_array_equal(off.loc[post, ["ff0", "ff1"]], np.zeros((np.sum(post), 2)))
    direct = np.zeros((len(q), 2))
    direct[2:] = off[["ff0", "ff1"]].to_numpy()[:-2] * off.total_scale.to_numpy()[:-2, None] * sigma
    np.testing.assert_array_equal(direct[q >= off_time + 2 * PERIOD], np.zeros((np.sum(q >= off_time + 2 * PERIOD), 2)))
    difference = np.max(
        np.abs(
            off[["residual_tip", "residual_tilt"]].to_numpy() - baseline[["residual_tip", "residual_tilt"]].to_numpy()
        )
        / sigma,
        axis=1,
    )
    future_max = np.maximum.accumulate(difference[::-1])[::-1]
    settled = np.flatnonzero(post & (future_max <= 1e-8))
    recovery = float((q[settled[0]] - off_time) * 1e-9) if len(settled) else None
    table = pd.DataFrame(rows)
    table.to_csv(directory / "metrics.csv", index=False)
    pd.DataFrame(blocks).to_csv(directory / "rms_by_time.csv", index=False)
    if sha256(directory / "source.bin") != metadata["source_sha256"]:
        raise ValueError("forcing changed between scenarios")
    report = dict(
        metadata,
        feedback="Existing syntheticIntegrator receives previous corrected WFS, not disturbance truth",
        wfs_only_baseline="Existing synthetic AO integrator, not the recorded-data AR baseline",
        paired_calibration_limits="Independent simulator axes need not be predictable from one channel per band",
        controls_start_s=seconds / 3,
        score_start_s=seconds / 2,
        disable_s=seconds * 0.75,
        ao_pole_radius={str(g): float(np.max(np.abs(np.roots([1, -1, 0, g])))) for g in (0.1, 0.3, 0.8)},
        audits=audits,
        disabled_shadow_exact_parity=True,
        ff_disable_recovery_s=recovery,
        ff_disable_tolerance_training_sd=1e-8,
        ff_disable_post1s_maximum_error_sd=float(np.max(difference[q >= off_time + 1_000_000_000])),
    )
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if plots:
        plot_run(directory, traces, spectrum, pd.DataFrame(blocks), metadata)
    return table, report


def plot_run(directory, traces, spectrum, blocks, metadata):
    """Export actual feedback residuals, RMS evolution, target PSDs and command/disable diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("none", "wfs_only", "harmonic_only", "wfs_harmonic")
    first = metadata["seconds"] / 2 + 1
    time = (traces["none"].query_ns.to_numpy() - ORIGIN) * 1e-9
    show = (time >= first) & (time < first + 0.3)
    for kind in ("residual", "command"):
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for axis in range(2):
            for name in names:
                data = traces[name]
                field = ("residual_tip", "residual_tilt")[axis] if kind == "residual" else f"total{axis}"
                axes[axis].plot(
                    time[show], data[field].to_numpy()[show] * (RAD_TO_MAS if kind == "residual" else 1), label=name
                )
            axes[axis].set(
                ylabel="WFS [mas, simulated]" if kind == "residual" else "Training WFS SD", title=("Tip", "Tilt")[axis]
            )
            axes[axis].legend(fontsize=8)
        axes[-1].set_xlabel("Simulation time [s]")
        fig.tight_layout()
        fig.savefig(directory / f"{kind}_timeseries.png")
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for axis in range(2):
        for band, limits in enumerate(((20, 35), (48, 63))):
            for name in names:
                f, _, after = spectrum[name]
                axes[axis, band].semilogy(f, after[:, axis], label=name)
            axes[axis, band].set(
                xlim=limits, ylabel="WFS PSD [mas²/Hz]", xlabel="Frequency [Hz]", title=("Tip", "Tilt")[axis]
            )
            axes[axis, band].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "target_psds.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for axis, field in enumerate(("tip_mas", "tilt_mas")):
        for name in (*names, "ff_off", "high_gain08"):
            b = blocks[blocks["case"] == name]
            axes[axis].semilogy(b.second + 0.5, b[field], label=name)
        axes[axis].axvline(metadata["seconds"] / 3, color="gray", ls=":")
        axes[axis].axvline(metadata["seconds"] * 0.75, color="gray", ls="--")
        axes[axis].set(ylabel="One-second RMS [mas]", title=("Tip", "Tilt")[axis])
        axes[axis].legend(fontsize=8)
    axes[-1].set_xlabel("Simulation time [s]; dotted = enable, dashed = FF-off test")
    fig.tight_layout()
    fig.savefig(directory / "rms_time.png")
    plt.close(fig)


def main():
    """Run fixed simulation populations/seeds without instrument, shared-memory or production application access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 701])
    parser.add_argument("--populations", nargs="+", choices=("dual", "full"), default=["dual", "full"])
    args = parser.parse_args()
    tables, reports = [], []
    for population in args.populations:
        for seed in args.seeds:
            directory = args.output / f"{population}_{seed}"
            metadata = prepare_calibration(args.executable, directory, args.seconds, population, seed)
            subprocess.run([str(args.executable), "run", str(directory)], check=True)
            table, report = analyze_run(directory, metadata, plots=seed == args.seeds[0])
            tables.append(table.assign(population=population, seed=seed))
            reports.append(report)
    pd.concat(tables, ignore_index=True).to_csv(args.output / "metrics.csv", index=False)
    (args.output / "report.json").write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")
    print(
        pd.concat(tables)
        .loc[
            lambda d: (d.comparison == "wfs_only") & (d.model == "wfs_harmonic"),
            [
                "population",
                "seed",
                "axis",
                "rmse",
                "nrmse",
                "band0_suppression_db",
                "band1_suppression_db",
                "outside_power_change_percent",
            ],
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
