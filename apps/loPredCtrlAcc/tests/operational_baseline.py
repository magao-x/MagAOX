r"""\file operational_baseline.py
\brief Audit user-supplied operational values and replay Legacy algebra on verified causal simulation records.
"""
import argparse
import configparser
import csv
import json
from pathlib import Path
import socket
import subprocess

import numpy as np

from calibration_generalization import CONTROL_START, FINAL, evaluate, training_diagnostics
from independent_realizations import SEEDS, broadband_rms, digest, save_csv, source_bank, summarize
from simulation_demo import APP, TESTS, aosim_methods, bridge, simulate
from vibration_report import FS, pair, spectra
from matplotlib import pyplot as plt

ROLES = {"no_control": 0, "standard": 3, "assisted_h5": 4}
COLORS = {"no_control": "#555555", "standard": "#0072b2", "assisted_h5": "#009e73"}


def parameters():
    """Read only the recorded user excerpt, never /opt/MagAOX or any live application configuration."""
    parser = configparser.ConfigParser()
    parser.read(TESTS/"operational_parameters.ini")
    p = parser["parameters"]
    result = {name: p.getfloat(name) for name in ("gain", "copy_gain", "regularization", "gamma", "covariance")}
    result.update({name: p.getint(name) for name in ("history", "future", "num_modes")})
    result.update(fpsSource=p["fpsSource"], own_shmim=p.getboolean("own_shmim"), qrd=p.getboolean("qrd"))
    if result["future"] != 3 or result["num_modes"] != 2 or not result["qrd"]:
        raise ValueError("This existing modal bridge supports two modes, F=3 and QR only")
    return result


def pole_radius(model):
    """Compute unconstrained always-ready homogeneous poles for this two-frame absolute-input plant, not a safety proof."""
    history = int(model["history"])
    gain = model["gain"][-2:].astype(float)
    coefficients = np.zeros((2, 2*(history+1)))
    for j in range(history):
        coefficients[:, 2*(j+1):2*(j+2)] += gain[:, 2*j:2*j+2]
    for j in range(history-1):
        coefficients[:, 2*j:2*j+2] += gain[:, 2*history+2*j:2*history+2*j+2]
    companion = np.zeros((2*(history+1), 2*(history+1)))
    companion[:2] = coefficients
    companion[2:, :-2] = np.eye(2*history)
    return float(np.max(abs(np.linalg.eigvals(companion))))


def figures(records, old_rows, out):
    """Show unchanged-plant absolute-output results versus the archived short-history experiment without retuning."""
    rms_fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    psd_fig, psd_axes = plt.subplots(2, 2, figsize=(12, 7))
    first, stop = FINAL
    for row, scale in enumerate((1, 100)):
        for role in ROLES:
            arrays = [pair(records[scale, seed, role][first:stop], "yx", "yy") for seed in SEEDS]
            curves = np.array([[float(np.sqrt(np.mean(y[i:i+100]**2))) for i in range(0, stop-first, 100)] for y in arrays])
            t = (first+np.arange(50, stop-first, 100))/FS
            low, median, high = np.percentile(curves, [25, 50, 75], axis=0)
            axes[row].plot(t, median, label=role, color=COLORS[role])
            axes[row].fill_between(t, low, high, color=COLORS[role], alpha=.12)
            f = spectra(arrays[0])[0]
            powers = np.array([spectra(y)[1] for y in arrays])
            for column, hz in enumerate((28, 56)):
                axis = psd_axes[row, column]
                axis.semilogy(f, np.median(powers, axis=0), color=COLORS[role], label=role)
                axis.set(xlim=(hz-5, hz+5), title=f"{scale}×: {hz}Hz", xlabel="Frequency (Hz)", ylabel="WFS units²/Hz")
                axis.grid(alpha=.25)
        axes[row].set(title=f"{scale}×: Legacy algebra, causal records, absolute inputs, synthetic positive plant",
                      ylabel="50-ms WFS RMS")
        axes[row].grid(alpha=.25)
    axes[0].legend()
    axes[-1].set_xlabel("Evaluation time (s); shading is between-seed IQR")
    psd_axes[0, 0].legend(fontsize=8)
    for figure, name in ((rms_fig, "operational_rms"), (psd_fig, "operational_psd")):
        figure.tight_layout()
        for extension in ("png", "pdf"):
            figure.savefig(out/(name+"."+extension), dpi=150, bbox_inches="tight")
        plt.close(figure)


def main():
    """Run one fixed operational-algebra profile; preserve timing and state explicitly what is not reproduced."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--magaox-repo", type=Path, default=Path.home()/"source/MagAOX")
    parser.add_argument("--bank", type=Path, default=APP/"build/independent-realization-source")
    parser.add_argument("--output", type=Path, default=TESTS/"results/operational_baseline")
    args = parser.parse_args()
    if socket.gethostname().lower().startswith("exao"):
        parser.error("Personal development workstation only")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    p = parameters()
    profile = dict(parameters=p, source="User excerpt; deployed revision/runtime INDI state not verified",
                   history_convention="Corrected causal records, not Legacy calculated-delta/indexing semantics",
                   control_law="Legacy maxCoeff(C^T C)*100 scaled inverse, plus .3*current_WFS",
                   predicting=True, learning="0..4s only; frozen afterward", integrating=False,
                   copy_gain="Inactive in predictive mode; checked-out clone loads it from parameters.gain",
                   plant="Unchanged synthetic y=d+delayed_absolute_u; real modal calibration/sign unknown",
                   seeds=SEEDS, source_scales=[1, 100], adc_channels=[0, 1], adc_history=5,
                   calibration_frames=[0, CONTROL_START], evaluation_frames=FINAL,
                   no_operational_fidelity_claim=True)
    (out/"profile.json").write_text(json.dumps(profile, indent=2)+"\n")
    build = APP/"build"/("operational-"+Path(args.cxx).name)
    first_update = p["history"]+p["future"]-1
    with (out/"build.log").open("w") as log:
        subprocess.run(["make", "-C", str(APP), "-f", "tests/Makefile.sim", "-j2", "CXX="+args.cxx,
                        "BUILD_DIR="+str(build), "MAGAOX_REPO="+str(args.magaox_repo.resolve()),
                        "WFS_HISTORY="+str(p["history"]), "QR_GAMMA="+str(p["gamma"]),
                        "QR_PRIOR="+str(p["covariance"]), "CONTROL_PENALTY="+str(p["regularization"]),
                        "FEEDBACK_GAIN="+str(p["gain"]), "LEGACY_LAW=1", "ACCEL_HISTORY=5",
                        "TRAINING_START="+str(first_update), "TRAINING_END=8000"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    lib = bridge(build/"libddspc_sim.so")
    compiler = subprocess.check_output([args.cxx, "--version"]).decode()
    bank = args.bank.resolve()
    recorded = source_bank(bank, lib, args.magaox_repo.resolve(), compiler)
    # Check against published original stimulus bytes, not a newly compiled producer's RNG behavior.
    original = json.loads((TESTS/"results/independent_realizations/source_manifest.json").read_text())
    for actual, archived in zip(recorded["records"], original["records"]):
        for key in ("seed", "scale", "adc_sha256", "pointing_sha256"):
            assert actual[key] == archived[key], key
    (out/"source_manifest.json").write_text(json.dumps(recorded, indent=2)+"\n")
    old_rows = list(csv.DictReader((TESTS/"results/independent_realizations/per_realization.csv").open()))
    old = {(int(r["scale"]), int(r["seed"]), r["controller"]): r for r in old_rows}
    methods, rows, diagnostics, time_rows, records = aosim_methods(args.magaox_repo.resolve()), [], [], [], {}
    for seed in SEEDS:
        for scale in (1, 100):
            sample = np.load(bank/f"seed_{seed}_scale_{scale}.npz")
            baseline = None
            for role, mode in ROLES.items():
                model = {} if mode else None
                data = simulate(lib, methods, sample["adc"], sample["pointing"], sample["units"], train=CONTROL_START,
                                mode=mode, calibration_model=model, calibration_seed=10000+seed, wfs_seed=20000+seed,
                                allow_unready_diagnostic=True)
                records[scale, seed, role] = data
                if mode == 0:
                    baseline = data
                else:
                    np.testing.assert_array_equal(pair(data[:CONTROL_START], "yx", "yy"), pair(baseline[:CONTROL_START], "yx", "yy"))
                    np.testing.assert_array_equal(pair(data, "disturbance_x", "disturbance_y"), pair(baseline, "disturbance_x", "disturbance_y"))
                    assert model["version"] == CONTROL_START-first_update and int(model["history"]) == p["history"]
                b = data[FINAL[0]:FINAL[1]]; y = pair(b, "yx", "yy")
                f, power = spectra(y)
                result = evaluate(data, baseline, model, dict(mode=mode), FINAL) if mode else dict(
                    residual_rms=float(np.sqrt(np.mean(y*y))), prediction_h2_rms=None, availability=0.,
                    peak_command=0., limit_fraction=0., suppression_28_db=0., suppression_56_db=0.)
                row = dict(seed=seed, scale=scale, controller=role, **result, broadband_rms=broadband_rms(f, power),
                           tip_rms=float(np.sqrt(np.mean(y[:, 0]**2))), tilt_rms=float(np.sqrt(np.mean(y[:, 1]**2))),
                           finite_trace=bool(np.isfinite(data.view(np.float64)).all()),
                           peak_command_full_run=float(np.max(abs(pair(data, "applied_x", "applied_y")))))
                reference = old[scale, seed, role]
                row["short_history_residual_rms"] = float(reference["residual_rms"])
                row["residual_change_vs_short_percent"] = 100*(row["residual_rms"]/float(reference["residual_rms"])-1)
                if not mode:
                    assert row["residual_rms"] == float(reference["residual_rms"])
                else:
                    row["linearized_pole_radius"] = pole_radius(model)
                    row["calibration_ready"] = model["calibration_ready"]
                    d = training_diagnostics(data, model, dict(mode=mode, seconds=4, prior=p["covariance"], gamma=p["gamma"]))
                    theta = model["theta"].astype(float); c = theta[:, -6:]
                    literal = np.linalg.solve(c.T @ c+p["regularization"]*np.eye(6), -c.T @ theta[:, :-6])
                    d.update(seed=seed, scale=scale, controller=role,
                             literal_lambda100_gain_norm=float(np.linalg.norm(literal)),
                             actual_gain_norm=float(np.linalg.norm(model["gain"])),
                             legacy_effective_penalty=float(np.max(c.T @ c)*p["regularization"]))
                    diagnostics.append(d)
                    np.savez(out/f"model_{seed}_{scale}_{role}.npz", **model)
                rows.append(row)
                for first in range(CONTROL_START, len(data)-99, 100):
                    ybin = pair(data[first:first+100], "yx", "yy")
                    time_rows.append(dict(seed=seed, scale=scale, controller=role, time_s=(first+50)/FS,
                                          residual_rms=float(np.sqrt(np.mean(ybin*ybin)))))
                save_csv(out/"per_realization.csv", rows)
                print(seed, scale, role, "forecast", row["prediction_h2_rms"], "RMS", row["residual_rms"], flush=True)
    summary = {}
    for scale in (1, 100):
        indexed = {(r["seed"], r["controller"]): r for r in rows if r["scale"]==scale}
        differences = [100*(1-indexed[seed,"assisted_h5"]["residual_rms"]/indexed[seed,"standard"]["residual_rms"]) for seed in SEEDS]
        summary[scale] = dict(assisted_rms_improvement=summarize(differences),
            mean_metrics={role: {name: float(np.mean([indexed[seed,role][name] for seed in SEEDS]))
                                for name in ("residual_rms", "broadband_rms", "suppression_28_db", "suppression_56_db")}
                          for role in ROLES})
    save_csv(out/"diagnostics.csv", diagnostics); save_csv(out/"time_rms.csv", time_rows)
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    figures(records, old_rows, out)
    files = [Path(__file__), TESTS/"operational_parameters.ini", TESTS/"operational_law.cpp", TESTS/"operational_law.hpp",
             TESTS/"simulation_demo.py", TESTS/"sim_bridge.cpp", TESTS/"Makefile.sim", TESTS/"async_shadow.cpp",
             TESTS/"prediction_information.py", TESTS/"calibration_generalization.py", TESTS/"independent_realizations.py",
             APP/"ar_controller.cpp", APP/"qrd_rls.cpp", APP/"shadow_controller.cpp",
             APP/"loPredCtrlAcc.hpp", APP.parent/"loPredCtrl/loPredCtrl.hpp", APP.parent/"loPredCtrl/ar_controller.cpp"]
    manifest = dict(compiler=compiler, binary_sha256=digest(build/"libddspc_sim.so"),
                    source_sha256={str(path): digest(path) for path in files},
                    input_bank_sha256=digest(bank/"manifest.json"), recorded_source_arrays_match_previous=True,
                    operational_binary_or_config_file_accessed=False, runtime_integrating_state_known=False,
                    actual_operational_fidelity_verified=False)
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
