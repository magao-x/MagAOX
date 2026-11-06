#!/usr/bin/env python3
r"""\file validate_gaps.py
\brief Paired offline 8/4-kHz loss, availability and bounded-gap numerical analysis.
\ingroup accelFFCtrl_unit_test

Uses the existing float64 truth/prediction schema. Accepted-record binaries add
two columns: physical zero-based sample index and actual monotonic receipt ns.
No native stream, instrument, INDI or production application is accessed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate_predictions import (BASE, OMEGA, RAD_TO_MAS, SETTLE, availability,
                                  metrics, timing_us, truth_indices, weighted_fit)


def fit_quality(times, acceleration, target, rate):
    """Independent QR/window diagnostics and a white-noise coefficient-uncertainty proxy.

    Noise covariance uses the squared weights (sandwich covariance), not the
    inverse Gram alone. It excludes colored vibration/model error and therefore
    is a diagnostic, not a calibrated confidence bound or an admission rule.
    """
    dt = np.r_[1/rate, np.diff(times)]
    beta = -np.expm1(-2*dt/.1)
    weight = beta * np.r_[np.cumprod((1-beta[:0:-1]))[::-1], 1.] * np.cumsum(beta[::-1])[::-1]
    phase = OMEGA*(times-times[0])
    design = np.column_stack((np.ones(len(times)), np.cos(phase), np.sin(phase)))
    gram = design.T @ (weight[:, None]*design)
    coef = np.linalg.solve(gram, design.T @ (weight[:, None]*acceleration))
    inverse = np.linalg.inv(gram)
    variance = 1e-8*rate/2 + (.0083799/1.02)**2/12
    covariance = variance * inverse @ (design.T @ ((weight**2)[:, None]*design)) @ inverse
    predicted, condition, residual = weighted_fit(times, acceleration, target, sample_rate=rate)
    expected = coef[1]*np.cos(OMEGA*(target-times[0])) + coef[2]*np.sin(OMEGA*(target-times[0]))
    np.testing.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-14)
    weighted_rms = np.sqrt(np.sum(weight[:, None]*(acceleration-design@coef)**2, axis=0)/np.sum(weight))
    peak = np.hypot(coef[1], coef[2])
    amplitude_se, phase_se = [], []
    for axis in range(2):
        gradient = coef[1:, axis] / max(peak[axis], np.finfo(float).tiny)
        phase_gradient = np.array([coef[2, axis], -coef[1, axis]]) / max(peak[axis]**2, np.finfo(float).tiny)
        amplitude_se.append(np.sqrt(gradient @ covariance[1:, 1:] @ gradient))
        phase_se.append(np.degrees(np.sqrt(phase_gradient @ covariance[1:, 1:] @ phase_gradient)))
    return dict(prediction=predicted, rcond=condition, residual=residual, weighted_rms=weighted_rms,
                amplitude_se=amplitude_se, phase_se=phase_se)


def load_predictions(path):
    """Read a complete existing-schema binary recording, rejecting partial rows."""
    return np.memmap(path, dtype="<f8", mode="r").reshape(-1, 21)


def analyze(root, plots=True):
    """Produce paired accuracy/availability, horizon checks, recovery and fit-quality tables plus plots."""
    root = Path(root)
    report = root / "report"
    report.mkdir(exist_ok=True)
    accuracy, health, quality, runs = [], [], [], {}
    for manifest in sorted(root.glob("*/*/*_runs.csv")):
        rate, policy = int(manifest.parent.parent.name), manifest.parent.name
        for run in pd.read_csv(manifest).itertuples():
            key = rate, run.population, run.scenario, policy
            runs[key] = run, manifest.parent, load_predictions(manifest.parent / f"{run.file}.bin")
    if not runs:
        raise ValueError("no completed gap-study manifests")
    for (rate, population, scenario, policy), (run, directory, data) in runs.items():
        truth = np.memmap(directory.parent / f"{population}_truth.bin", dtype="<f8", mode="r").reshape(-1, 13)
        if len(truth) != run.published + 162:
            raise ValueError("incomplete truth recording")
        nominal = data[:, 0] >= 0
        valid = data[:, 4] == 1
        if np.sum(nominal) != run.scheduled or np.sum(nominal & valid) != run.scheduled_ready:
            raise ValueError("prediction artifact disagrees with manifest")
        events = pd.read_csv(directory / f"{run.file}_events.csv")
        values, _ = availability(data, events.time_ns.to_numpy())
        age, horizon = timing_us(data)
        if len(age):
            if np.max(age) > 2000 or np.min(age) < 0:
                raise AssertionError("ready prediction violated sample freshness")
            np.testing.assert_array_equal(data[valid, 3]-data[valid, 1], np.full(np.sum(valid), 1_000_000))
            np.testing.assert_allclose(horizon, age+1000, rtol=0, atol=1e-9)
        identity = dict(rate=rate, population=population, scenario=scenario, policy=policy)
        produced = data[:, 5] == 1
        health.append(dict(**identity, **values, dropped=run.dropped, resets=run.resets, retained=run.retained,
            published=run.published, dropped_per_s=run.dropped/(run.published/rate),
            overrun_events=len(events), overrun_events_per_s=len(events)/(run.published/rate),
            first_ready_ms=(run.first_ready_ns-BASE)*1e-6 if run.first_ready_ns >= 0 else np.nan,
            stale_queries=int(np.sum(produced & (data[:, 6] == 5))),
            fault_queries=int(np.sum(produced & np.isin(data[:, 6], [2, 3, 4, 6, 7]))),
            warming_queries=int(np.sum(produced & (data[:, 6] == 0))),
            blocked_queries=int(np.sum(nominal & ~produced)),
            rcond_min=np.min(data[valid, 18]) if np.any(valid) else np.nan,
            age_max_us=max(age) if len(age) else np.nan,
            effective_horizon_max_us=max(horizon) if len(horizon) else np.nan))
        current = data[nominal]
        reference = runs[rate, population, scenario, "reset"][2]
        reference = reference[reference[:, 0] >= 0]
        ideal = runs[rate, population, "ideal", "reset"][2]
        ideal = ideal[ideal[:, 0] >= 0]
        np.testing.assert_array_equal(current[:, 1], reference[:, 1])
        np.testing.assert_array_equal(current[:, 1], ideal[:, 1])
        if scenario == "ideal":
            np.testing.assert_array_equal(current, reference)  # Prototype has exactly the same lossless numerics.
        accepted_path = directory / f"{run.file}_accepted.bin"
        reset_path = directory.parent / "reset" / f"{run.file}_accepted.bin"
        np.testing.assert_array_equal(np.fromfile(accepted_path, dtype="<f8"),
                                      np.fromfile(reset_path, dtype="<f8"))
        indices = truth_indices(current[:, 3], len(truth), period=1_000_000_000//rate)
        ready = current[:, 4] == 1
        settled = (current[:, 1]-BASE)*1e-9 >= SETTLE
        for axis in range(2):
            for mask_name, selection in (("own_valid", ready & settled),
                ("common_with_reset", ready & (reference[:, 4] == 1) & settled)):
                p = current[:, 10+axis]
                y = truth[indices, 9+axis]
                result = metrics(p[selection], y[selection], (current[selection, 3]-BASE)*1e-9)
                delta = p[selection]-ideal[selection, 10+axis]
                incremental = metrics(p[selection], ideal[selection, 10+axis], (current[selection, 3]-BASE)*1e-9)
                accuracy.append(dict(**identity, axis="XY"[axis], mask=mask_name,
                    valid_samples=int(np.sum(selection)), **result, pointing_rmse_mas=result["rmse"]*RAD_TO_MAS/(OMEGA**2*10),
                    gap_gain_change_pct=incremental["amplitude_error_pct"],
                    gap_phase_change_deg=incremental["phase_error_deg"],
                    delta_from_ideal_rmse=np.sqrt(np.mean(delta**2)) if len(delta) else np.nan))
        # Independently inspect at most 12 actually retained gaps, at receipt and 50 ms later.
        if policy != "reset" and len(events):
            retained_events = events[events.resets.diff().fillna(events.resets-1) == 0].head(12)
            accepted = np.fromfile(directory / f"{run.file}_accepted.bin", dtype="<f8").reshape(-1, 2)
            accepted_indices = accepted[:, 0].astype(np.int64)
            decoded = (truth[:, 1:3]-2048)*.0083799/1.02
            for event in retained_events.itertuples():
                event_sample = accepted_indices[np.searchsorted(accepted[:, 1], event.time_ns)]
                for after_ms in (0, 50):
                    last = event_sample + round(after_ms*rate/1000)
                    inspection = np.searchsorted(data[:, 1], BASE+(last+1)*(1_000_000_000//rate), side="right")-1
                    if not data[inspection, 4] or data[inspection, 8] != event.resets:
                        continue  # A later cluster reset invalidates the mature-window covariance probe.
                    start = max(0, last-rate)
                    actual = accepted_indices[(accepted_indices >= start) & (accepted_indices <= last)]
                    uninterrupted = np.arange(start, last+1)
                    # Compare at the same sample+1ms+one-period target, including measurement age.
                    target_s = (last + 1 + rate//1000)/rate
                    q = fit_quality(actual/rate, decoded[actual], target_s, rate)
                    off = fit_quality(uninterrupted/rate, decoded[uninterrupted], target_s, rate)
                    for axis in range(2):
                        quality.append(dict(**identity, event_ns=event.time_ns, after_ms=after_ms, axis="XY"[axis],
                            rcond=q["rcond"], normal_residual=q["residual"],
                            weighted_residual_ms2=q["weighted_rms"][axis],
                            residual_ratio=q["weighted_rms"][axis]/off["weighted_rms"][axis],
                            amplitude_se_ms2=q["amplitude_se"][axis],
                            amplitude_se_ratio=q["amplitude_se"][axis]/off["amplitude_se"][axis],
                            phase_se_deg=q["phase_se"][axis],
                            delta_from_uninterrupted_ms2=q["prediction"][axis]-off["prediction"][axis]))
    accuracy = pd.DataFrame(accuracy)
    health = pd.DataFrame(health)
    accuracy.to_csv(report / "accuracy.csv", index=False)
    health.to_csv(report / "availability.csv", index=False)
    pd.DataFrame(quality).to_csv(report / "fit_quality.csv", index=False)
    (report / "analysis_versions.json").write_text(json.dumps(dict(numpy=np.__version__, pandas=pd.__version__,
        scipy=__import__("scipy").__version__), indent=2) + "\n")
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for rate in (8000, 4000):
            for population in ("single28", "full214"):
                fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex="col")
                truth = np.memmap(root / str(rate) / f"{population}_truth.bin", dtype="<f8", mode="r").reshape(-1,13)
                for column, scenario in enumerate(("loss1_4Hz", "cluster3_5s")):
                    for policy in ("reset", "gap1", "gap7"):
                        _, _, data = runs[rate, population, scenario, policy]
                        data = data[data[:, 0] >= 0]
                        t = (data[:, 1]-BASE)*1e-9
                        valid = data[:, 4] == 1
                        ix = truth_indices(data[:, 3], len(truth), period=1_000_000_000//rate)
                        focus = (t >= (1 if column == 0 else 4.9)) & (t <= (2 if column == 0 else 5.7))
                        theta = np.where(valid, data[:, 12]*RAD_TO_MAS, np.nan)
                        y = truth[ix, 11]*RAD_TO_MAS
                        axes[0,column].plot(t[focus], theta[focus], label=policy)
                        axes[1,column].plot(t[focus], (theta-y)[focus], label=policy)
                        axes[2,column].step(t[focus], valid[focus].astype(int) + .03*(policy == "gap7"), label=policy)
                        if policy == "reset":
                            axes[0,column].plot(t[focus], y[focus], "k--", alpha=.5, label="target28 truth")
                    axes[0,column].set_title(scenario)
                    axes[0,column].legend(fontsize=8)
                    axes[0,column].set_ylabel("X pointing at target [mas]")
                    axes[1,column].set_ylabel("X prediction − truth [mas]")
                    axes[2,column].set(xlabel="query time [s]", ylabel="ready", ylim=(-.1,1.15))
                    axes[2,column].set_xlim((1,2) if column == 0 else (4.9,5.7))
                    axes[2,column].legend(fontsize=8)
                fig.suptitle(f"{rate} Hz / {population}: offline policy comparison")
                fig.tight_layout()
                fig.savefig(report / f"{rate}_{population}_recovery.png", dpi=140)
                plt.close(fig)
        for population in ("single28", "full214"):
            fig, axes = plt.subplots(2, 1, figsize=(13, 7))
            for rate, ax in zip((8000,4000),axes):
                subset = health[(health.rate == rate) & (health.population == population)]
                scenarios = list(subset.scenario.unique())
                x = np.arange(len(scenarios))
                for i, policy in enumerate(("reset", "gap1", "gap7")):
                    values = subset[subset.policy == policy].set_index("scenario").loc[scenarios]
                    ax.bar(x + (i-1)*.25, values.scored_availability_pct, .25, label=policy)
                ax.set(xticks=x, xticklabels=scenarios, ylabel="ready %, 4–60 s", title=f"{rate} Hz")
                ax.tick_params(axis="x", labelrotation=25)
                ax.legend()
            fig.tight_layout()
            fig.savefig(report / f"{population}_availability.png", dpi=140)
            plt.close(fig)
    print(health[health.population == "full214"].to_string(index=False))
    print(accuracy.query("population == 'full214' and mask == 'own_valid'")[
        ["rate","scenario","policy","axis","nrmse","phase_error_deg"]].to_string(index=False))
    print(f"Reports: {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    analyze(args.directory, not args.no_plots)
