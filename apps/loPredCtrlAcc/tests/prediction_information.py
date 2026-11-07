r"""\file prediction_information.py
\brief Score fixed native QR fits on common causal ADC0/1 records, without changing the simulation or controller.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import numpy as np
from scipy import signal

from simulation_demo import APP, TESTS, aosim_methods, bridge, simulate, source, score, full_time_series
from vibration_demo import plots
from vibration_report import FS, pair, spectra, line_power
from matplotlib import pyplot as plt


def states(data, acceleration, mode, origins, accel_history=5, wfs_history=6):
    """Construct newest-first WFS/consumed-command state with explicit WFS and causal ADC history lengths."""
    y, u = pair(data, "yx", "yy"), pair(data, "applied_x", "applied_y")
    return np.concatenate([y[origins-j] for j in range(wfs_history)] + [u[origins-j-1] for j in range(wfs_history-1)] +
                          ([acceleration[origins-j] for j in range(accel_history)] if mode == 4 else []), axis=1).astype(np.float32)


def forecast(data, acceleration, model, mode, origins):
    """Evaluate a frozen native matrix; future consumed inputs appear only in the post-hoc scoring target."""
    wfs_history = int(model.get("history", 6))
    base = (3+2*wfs_history-1)*2
    history = (model["theta"].shape[1]-base)//2 if mode == 4 else 0
    x = states(data, acceleration, mode, origins, accel_history=history, wfs_history=wfs_history)
    predicted = x @ model["theta"][:, :-6].T
    target = np.concatenate([pair(data, "yx", "yy")[origins+h] for h in (3, 2, 1)], axis=1)
    # Remove actual future corrective effects for the zero-future-command counterfactual.
    u = pair(data, "applied_x", "applied_y")
    target[:, :2] -= u[origins+1]
    target[:, 2:4] -= u[origins]
    return predicted, target, x


def rms(value):
    """Two-axis per-component RMS in the existing normalized WFS units."""
    return float(np.sqrt(np.mean(np.asarray(value)**2)))


def peak_coherence(acceleration, wfs):
    """Return all ADC-to-WFS pairs at actual nearest Welch bins; phase is WFS minus ADC, no truth input."""
    result = []
    options = dict(fs=FS, nperseg=2048, noverlap=1024, window="hann", detrend="constant")
    for sensor in range(2):
        for axis in range(2):
            f, c = signal.coherence(acceleration[:, sensor], wfs[:, axis], **options)
            _, cross = signal.csd(acceleration[:, sensor], wfs[:, axis], **options)
            for hz in (28, 56):
                i = int(np.argmin(abs(f-hz)))
                result.append(dict(adc_channel=sensor, wfs_axis=axis, line_hz=hz, bin_hz=float(f[i]),
                                   coherence=float(c[i]), wfs_minus_adc_phase_deg=float(np.angle(cross[i], deg=True))))
    return result


def conditional_information(data, acceleration, origins):
    """Audit incremental linear information; these statistical fits are never controllers or ready snapshots."""
    common = states(data, acceleration, 3, origins).astype(float)
    extra = states(data, acceleration, 4, origins)[:, -10:].astype(float)
    target = pair(data, "yx", "yy")[origins+2] - pair(data, "applied_x", "applied_y")[origins]
    z = extra-common @ np.linalg.lstsq(common, extra, rcond=None)[0]
    e = target-common @ np.linalg.lstsq(common, target, rcond=None)[0]
    remaining = e-z @ np.linalg.lstsq(z, e, rcond=None)[0]
    # Whole-window partial R2 is descriptive, not an out-of-sample improvement claim.
    result = dict(descriptive_partial_r2=(1-np.mean(remaining**2, axis=0)/np.mean(e**2, axis=0)).tolist())
    # One earlier -> later time split, with a three-frame guard: all fit targets precede test inputs.
    split = len(origins)//2
    result["fit_last_target_frame"] = int(origins[split-4]+2)
    result["test_first_origin_frame"] = int(origins[split])
    for mode, x in ((3, common), (4, np.column_stack((common, extra)))):
        coefficient = np.linalg.lstsq(x[:split-3], target[:split-3], rcond=None)[0]
        result[f"blocked_information_rms_{mode}"] = rms(x[split:] @ coefficient-target[split:])
    return result


def line_generalization(data, acceleration, models):
    """Measure actual calibration/late source phasors and learned cross-axis errors, solely after the experiment."""
    disturbance = pair(data, "yx", "yy") - pair(data, "dm_x", "dm_y")
    f, power = signal.welch(disturbance[3400:], fs=FS, nperseg=2048, axis=0)
    _, cross = signal.csd(disturbance[3400:], acceleration[3400:], fs=FS, nperseg=2048, axis=0)
    result = []
    for hz in (28, 56):
        i = np.argmin(abs(f-hz))
        sensor_transfer = cross[i]/power[i]  # Measured ADC/WFS, not ideal acceleration or a control input.
        z = np.exp(-2j*np.pi*hz/FS)
        matrices = {}
        for mode, model in models.items():
            theta = model["theta"][2:4]
            a = sum(theta[:, 2*j:2*j+2] * z**j for j in range(6))
            d = sum(theta[:, 22+2*j:24+2*j] * z**j for j in range(5)) if mode == 4 else np.zeros((2, 2))
            matrices[mode] = (a + d @ np.diag(sensor_transfer))*z**2 - np.eye(2)
        windows = []
        for first, end in ((100, 2300), (3400, 5400), (8400, 10400)):
            n = np.arange(first, end)
            basis = np.column_stack([function(2*np.pi*h*n/FS) for h in (hz, 84-hz)
                                     for function in (np.sin, np.cos)])
            coefficients = np.linalg.lstsq(basis, disturbance[first:end], rcond=None)[0]
            phasor = coefficients[1]-1j*coefficients[0]
            windows.append(dict(first_frame=first, end_frame=end,
                                y_over_x_amplitude=float(abs(phasor[1]/phasor[0])),
                                y_minus_x_phase_deg=float(np.angle(phasor[1]/phasor[0], deg=True)),
                                relative_transfer_error={mode: float(np.linalg.norm(error @ phasor)/np.linalg.norm(phasor))
                                                         for mode, error in matrices.items()}))
        result.append(dict(hz=hz, windows=windows,
                           off_axis_transfer_error={mode: float(np.linalg.norm(error-np.diag(np.diag(error))))
                                                    for mode, error in matrices.items()}))
    return result


def response(model, mode, hz):
    """Audit the fixed gain's linear feedback sensitivity; no alternative controller or future truth is used."""
    gain = model["gain"][-2:].astype(float)
    z = np.exp(-2j*np.pi*hz/FS)
    ky = sum(gain[:, 2*j:2*j+2] * z**j for j in range(6))
    ku = sum(gain[:, 12+2*j:14+2*j] * z**(j+1) for j in range(5))
    ka = sum(gain[:, 22+2*j:24+2*j] * z**j for j in range(5)) if mode == 4 else np.zeros((2, 2))
    feedback = np.linalg.solve(np.eye(2)-ku, ky)
    feedforward = np.linalg.solve(np.eye(2)-ku, ka)
    sensitivity = np.linalg.inv(np.eye(2)-z**2*feedback)
    return dict(hz=hz, wfs_noise_sensitivity_norm=float(np.linalg.norm(sensitivity, 2)),
                adc_to_wfs_norm=float(np.linalg.norm(sensitivity * z**2 @ feedforward, 2)))


def figures(records, errors, out):
    """Expose forecast and closed-loop errors over the same full scored interval, and retain peak PSD plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for row, scale in enumerate((1, 100)):
        for mode, color, label in ((3, "#0072b2", "Standard"), (4, "#009e73", "Assisted ADC0/1")):
            e, origins = errors[scale, mode]
            starts = np.arange(0, len(e)-99, 100)
            axes[row, 0].plot((origins[starts+50]-2400)/FS, [rms(e[i:i+100, 2:4]) for i in starts],
                              color=color, label=label)
            b = records[scale, mode][3400:]
            y = pair(b, "yx", "yy")
            starts = np.arange(0, len(y)-99, 100)
            axes[row, 1].plot((b["frame"][starts+50]-2400)/FS, [rms(y[i:i+100]) for i in starts], color=color, label=label)
        axes[row, 0].set_title(f"{scale}×: common open-loop, k+2 forecast error")
        axes[row, 1].set_title(f"{scale}×: each controller's closed-loop residual")
        for ax in axes[row]:
            ax.set_ylabel("50-ms RMS (normalized)")
            ax.grid(alpha=.25)
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("Time after calibration (s)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out/("prediction_and_residual."+ext), dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for row, scale in enumerate((1, 100)):
        for mode, color, label in ((0, "#555555", "No DDSPC"), (3, "#0072b2", "Standard"),
                                  (4, "#009e73", "Assisted ADC0/1")):
            f, power = spectra(pair(records[scale, mode][3400:], "yx", "yy"))
            for column, hz in enumerate((28, 56)):
                axes[row, column].semilogy(f, power, color=color, label=label)
                axes[row, column].set(xlim=(hz-5, hz+5), title=f"{scale}×: {hz} Hz residual",
                                      ylabel="WFS units²/Hz", xlabel="Frequency (Hz)")
                axes[row, column].grid(alpha=.25)
    axes[0, 0].legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out/("peak_psd."+ext), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """Build only the fixed-fit test library and repeat the two declared amplitudes with post-hoc model audits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magaox-repo", type=Path, default=Path.home()/"source/MagAOX")
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--output", type=Path, default=TESTS/"results/prediction_information")
    args = parser.parse_args()
    if socket.gethostname().lower().startswith("exao"):
        parser.error("Personal workstation only")
    build = APP/"build"/("simulation-"+Path(args.cxx).name+"-train2400")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    with (out/"build.log").open("w") as log:
        subprocess.run(["make", "-C", str(APP), "-f", "tests/Makefile.sim", "-j2", "BUILD_DIR="+str(build),
                        "CXX="+args.cxx, "TRAINING_END=2400", "MAGAOX_REPO="+str(args.magaox_repo.resolve())],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    lib, methods = bridge(build/"libddspc_sim.so"), aosim_methods(args.magaox_repo.resolve())
    findings, records, errors = {}, {}, {}
    for scale in (1, 100):
        raw, pointing, units = source(lib, 10400, scale)
        models = {mode: {} for mode in (3, 4)}
        for mode in (0, 3, 4):
            records[scale, mode] = simulate(lib, methods, raw, pointing, units, mode=mode,
                                            calibration_model=models.get(mode))
        baseline = records[scale, 0]
        acceleration = pair(records[scale, 4], "ax", "ay")  # Decoded native UINT16 features, never source acceleration.
        origins = np.arange(3400, len(baseline)-3)
        result = dict(coherence=peak_coherence(acceleration[3400:], pair(baseline[3400:], "yx", "yy")),
                      closed_loop=score(np.concatenate([records[scale, m] for m in (0, 3, 4)]), 2400),
                      conditional_information=conditional_information(baseline, acceleration, origins),
                      line_generalization=line_generalization(baseline, acceleration, models),
                      prediction={}, response={}, models={})
        for mode, model in models.items():
            np.savez(out/f"model_{scale}_{mode}.npz", **model)
            calibration_origins = np.arange(5, 2397)
            fit, fit_target, fit_state = forecast(baseline, acceleration, model, mode, calibration_origins)
            u = pair(baseline, "applied_x", "applied_y")
            future_u = np.concatenate([u[calibration_origins+j] for j in (2, 1, 0)], axis=1)
            phi = np.column_stack((fit_state, future_u.astype(np.float32))).astype(float)
            target = np.concatenate([pair(baseline, "yx", "yy")[calibration_origins+h] for h in (3, 2, 1)],
                                    axis=1).astype(np.float32).astype(float)
            gamma = float(np.float32(.9995))
            weights = gamma**((2396-calibration_origins)/2)
            # Independent double batch fit includes the same decayed information prior as native QR.
            prior = np.sqrt(float(np.float32(.001))*gamma**len(phi))
            batch = np.linalg.lstsq(np.vstack((phi*weights[:, None], prior*np.eye(phi.shape[1]))),
                                   np.vstack((target*weights[:, None], np.zeros((phi.shape[1], 6)))), rcond=None)[0].T
            result["models"][mode] = dict(version=model["version"], theta_shape=list(model["theta"].shape),
                    gain_norm=float(np.linalg.norm(model["gain"])),
                    calibration_h2_rms=rms(fit[:, 2:4]-fit_target[:, 2:4]),
                    batch_theta_max_error=float(np.max(abs(batch-model["theta"]))),
                    column_scaled_calibration_condition=float(np.linalg.cond(phi/np.linalg.norm(phi, axis=0))),
                    batch_open_loop_h2_rms=rms(states(baseline, acceleration, mode, origins) @ batch[2:4, :-6].T -
                                              pair(baseline, "yx", "yy")[origins+2]))
            result["prediction"][mode] = {}
            for trajectory in (0, 3, 4):
                prediction, target, x = forecast(records[scale, trajectory], acceleration, model, mode, origins)
                error = prediction-target
                if trajectory == 0:
                    errors[scale, mode] = error, origins
                actual = model["gain"][-2:].astype(float) @ x.astype(float).T
                c = model["theta"][:, -6:].astype(float)
                ideal_c = np.zeros((6, 6)); ideal_c[:2, 2:4] = ideal_c[2:4, 4:6] = np.eye(2)
                # Diagnostic command difference from a known C solve, not a simulated replacement controller.
                known_c_command = -prediction[:, 2:4]/1.02
                result["prediction"][mode][trajectory] = dict(rms=rms(error),
                        horizon_rms={h: rms(error[:, 2*(3-h):2*(4-h)]) for h in (1, 2, 3)},
                        zero_future_forecast_rms=rms(prediction),
                        command_difference_from_known_c=rms(actual.T-known_c_command),
                        control_block_error=float(np.linalg.norm(c-ideal_c)))
                if trajectory == 0:
                    f, power = spectra(error[:, 2:4])
                    result["prediction"][mode][trajectory]["h2_error_line_power"] = {
                        hz: line_power(f, power, hz) for hz in (28, 56)}
            result["response"][mode] = [response(model, mode, hz) for hz in (28, 56)]
        destination = out/f"scale_{scale}"
        destination.mkdir(exist_ok=True)
        data = np.concatenate([records[scale, m] for m in (0, 3, 4)])
        plots(data, dict(train=2400), destination)
        full_time_series(data, destination)
        findings[scale] = result
        print(scale, "calibration models", json.dumps(result["models"]), flush=True)
        print(scale, "conditional information", json.dumps(result["conditional_information"]), flush=True)
    figures(records, errors, out)
    files = [Path(__file__), TESTS/"simulation_demo.py", TESTS/"sim_bridge.cpp", TESTS/"async_shadow.cpp",
             APP/"ar_controller.cpp", APP/"qrd_rls.cpp", APP/"shadow_controller.cpp",
             args.magaox_repo/"apps/accelSim/accelSimModel.hpp", args.magaox_repo/"apps/aoSim/xapp/aoSim/__init__.py"]
    manifest = dict(compiler=subprocess.check_output([args.cxx, "--version"]).decode(),
                    binary_sha256=hashlib.sha256((build/"libddspc_sim.so").read_bytes()).hexdigest(),
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    adc_channels=[0, 1], training_end=2400, score_start=3400, frame_count=10400,
                    prediction_horizons=[1, 2, 3], correction_effect_frames=2,
                    timing="ADC frame k: four samples before decision; injection visible in WFS k+2")
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (out/"metrics.json").write_text(json.dumps(findings, indent=2)+"\n")


if __name__ == "__main__":
    main()
