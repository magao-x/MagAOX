r"""\file calibration_generalization.py
\brief Small preregistered QR calibration/history/prior comparison with separate validation and final simulation intervals.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import numpy as np

from prediction_information import forecast, rms
from simulation_demo import APP, TESTS, aosim_methods, bridge, simulate, source
from vibration_report import FS, pair, spectra, line_power
from matplotlib import pyplot as plt

CONTROL_START = 8000
VALIDATION = (9000, 13000)
FINAL = (24000, 32000)


def candidates():
    """Declare three duration pairs plus two shorter histories and one tenfold initial-information adjustment."""
    return [dict(name=f"d{s}_h5_p001", seconds=s, history=5, prior=.001, mode=mode)
            for s in (1, 2, 4) for mode in (3, 4)] + [
        dict(name="d4_h3_p001", seconds=4, history=3, prior=.001, mode=4),
        dict(name="d4_h1_p001", seconds=4, history=1, prior=.001, mode=4),
        dict(name="d4_h5_p010", seconds=4, history=5, prior=.01, mode=4)]


def identity(config):
    """Distinguish the independent standard/assisted fits using the same build configuration."""
    return config["name"] + ("_standard" if config["mode"] == 3 else "_assisted")


def select(validation):
    """Choose only by pooled validation forecast MSE across both declared source amplitudes, with a fixed tie break."""
    scores = {key: float(np.sqrt(np.mean([rows[key]["prediction_h2_rms"]**2 for rows in validation.values()])))
              for key in next(iter(validation.values()))}
    configs = {identity(c): c for c in candidates()}
    groups = dict(standard=[key for key, c in configs.items() if c["mode"] == 3],
                  existing_assisted=[key for key, c in configs.items() if c["mode"] == 4 and c["history"] == 5 and c["prior"] == .001],
                  selected_assisted=[key for key, c in configs.items() if c["mode"] == 4])
    return dict(scores=scores, choices={group: min(keys, key=lambda key: (scores[key], key)) for group, keys in groups.items()})


def training_diagnostics(data, model, config):
    """Score the end-aligned training window and its weighted design without changing native readiness."""
    first, stop = CONTROL_START-int(config["seconds"]*FS), CONTROL_START
    history = int(model.get("history", 6))
    a, b = history*2, (history-1)*2
    origins = np.arange(first+history-1, stop-3)
    acceleration = pair(data, "ax", "ay")
    prediction, target, state = forecast(data, acceleration, model, config["mode"], origins)
    u, y = pair(data, "applied_x", "applied_y"), pair(data, "yx", "yy")
    future = np.concatenate([u[origins+j] for j in (2, 1, 0)], axis=1).astype(np.float32)
    phi = np.column_stack((state, future)).astype(float)
    observed = np.concatenate([y[origins+h] for h in (3, 2, 1)], axis=1)
    gamma = float(np.float32(config.get("gamma", .9995)))
    weighted = phi * gamma**((stop-4-origins)/2)[:, None]
    scaled = weighted/np.linalg.norm(weighted, axis=0)
    singular = np.linalg.svd(scaled, compute_uv=False)
    spectrum = np.linalg.svd(weighted, compute_uv=False)
    theta = model["theta"]
    result = dict(training_h2_rms=rms(prediction[:, 2:4]-target[:, 2:4]),
                  training_conditional_h2_rms=rms((phi @ theta.T-observed)[:, 2:4]),
                  features=theta.shape[1], native_rank=int(data["rank"][stop]),
                  design_rank=int(np.sum(singular > singular[0]*max(scaled.shape)*np.finfo(float).eps)),
                  scaled_design_condition=float(singular[0]/singular[-1]),
                  unscaled_design_condition=float(spectrum[0]/spectrum[-1]),
                  theta_norm=float(np.linalg.norm(theta)), theta_max=float(np.max(abs(theta))),
                  a_norm=float(np.linalg.norm(theta[:, :a])), b_norm=float(np.linalg.norm(theta[:, a:a+b])),
                  d_norm=float(np.linalg.norm(theta[:, a+b:-6])), c_norm=float(np.linalg.norm(theta[:, -6:])),
                  gain_norm=float(np.linalg.norm(model["gain"])), version=model["version"],
                  effective_records=float(model["version"] if gamma == 1 else (1-gamma**model["version"])/(1-gamma)),
                  remaining_initial_information=float(config["prior"]*gamma**model["version"]))
    if config["mode"] == 4:
        result["max_adc_wfs_column_cosine"] = float(np.max(abs(scaled[:, :a].T @ scaled[:, a+b:-6])))
    return result


def evaluate(data, baseline, model, config, interval):
    """Score common open-loop forecasts and every closed-loop frame, including invalid holds and limits."""
    first, stop = interval
    origins = np.arange(first, stop-3)
    acceleration = pair(data, "ax", "ay")
    prediction, target, _ = forecast(baseline, acceleration, model, config["mode"], origins)
    b, original = data[first:stop], pair(baseline[first:stop], "yx", "yy")
    residual = pair(b, "yx", "yy")
    f, p0 = spectra(original)
    _, p = spectra(residual)
    active = b["valid"] != 0
    assert (b["version"][active] == model["version"]).all()
    assert (b["age_ns"][active] <= 2000000).all()
    np.testing.assert_array_equal(pair(data, "applied_x", "applied_y"), pair(data, "request_x", "request_y"))
    np.testing.assert_array_equal(pair(data, "dm_x", "dm_y")[2:], pair(data, "applied_x", "applied_y")[:-2])
    result = dict(prediction_h2_rms=rms(prediction[:, 2:4]-target[:, 2:4]), residual_rms=rms(residual),
                  variance_reduction=float(1-np.var(residual, axis=0).mean()/np.var(original, axis=0).mean()),
                  availability=float(active.mean()), ready_fraction=float(b["ready"].mean()),
                  peak_command=float(np.max(abs(pair(b, "applied_x", "applied_y")))),
                  limit_fraction=float(np.any(abs(pair(b, "applied_x", "applied_y")) >= 4-1e-6, axis=1).mean()))
    for hz in (28, 56):
        result[f"suppression_{hz}_db"] = float(10*np.log10(line_power(f, p0, hz)/line_power(f, p, hz)))
    return result


def figures(records, baselines, choice, out):
    """Plot only the locked three-role comparison on the untouched final interval, with no-control reference."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    psd, spectra_axes = plt.subplots(2, 2, figsize=(12, 7))
    colors = dict(standard="#0072b2", existing_assisted="#e69f00", selected_assisted="#009e73")
    first, stop = FINAL
    for row, scale in enumerate((1, 100)):
        curves = [("No DDSPC", baselines[scale], "#555555")] + [
            (role+": "+key, records[scale, key], colors[role]) for role, key in choice.items()]
        for label, data, color in curves:
            values = pair(data[first:stop], "yx", "yy")
            starts = np.arange(0, len(values)-99, 100)
            axes[row].plot((first+starts+50)/FS, [rms(values[i:i+100]) for i in starts], color=color, label=label)
            f, power = spectra(values)
            for column, hz in enumerate((28, 56)):
                axis = spectra_axes[row, column]
                axis.semilogy(f, power, color=color, label=label)
                axis.set(xlim=(hz-5, hz+5), title=f"{scale}×: {hz} Hz", xlabel="Frequency (Hz)", ylabel="WFS units²/Hz")
                axis.grid(alpha=.25)
        axes[row].set(title=f"{scale}×: reserved final evaluation", ylabel="50-ms WFS RMS (normalized)")
        axes[row].grid(alpha=.25)
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Absolute simulation time (s)")
    spectra_axes[0, 0].legend(fontsize=7)
    for figure, name in ((fig, "final_rms"), (psd, "final_psd")):
        figure.tight_layout()
        for extension in ("png", "pdf"):
            figure.savefig(out/(name+"."+extension), dpi=150, bbox_inches="tight")
        plt.close(figure)


def main():
    """Lock selection before generating final samples; reuse the native source/plant/QR paths for every candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--magaox-repo", type=Path, default=Path.home()/"source/MagAOX")
    parser.add_argument("--output", type=Path, default=TESTS/"results/calibration_generalization")
    args = parser.parse_args()
    if socket.gethostname().lower().startswith("exao"):
        parser.error("Personal development workstation only")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    configs = candidates()
    protocol = dict(candidates=configs, source_amplitudes=[1, 100], adc_channels=[0, 1],
                    common_excitation_and_control_start_frame=CONTROL_START, validation_frames=VALIDATION, final_frames=FINAL,
                    calibration_windows={seconds: [CONTROL_START-seconds*FS, CONTROL_START] for seconds in (1, 2, 4)},
                    selection="Minimum pooled validation two-frame forecast RMS across both amplitudes; lexical ID breaks ties",
                    rank_tolerance=1e-5, gamma=.9995, command_penalty=.02,
                    note="All cases receive identical 0..4s excitation. Training windows end at4s; all fitted examples lie within their window.")
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    libraries, builds = {}, {}
    for c in configs:
        if c["name"] in libraries:
            continue
        build = APP/"build"/("generalization-endaligned-"+Path(args.cxx).name+"-"+c["name"])
        build.mkdir(parents=True, exist_ok=True)
        with (out/(c["name"]+"-build.log")).open("w") as log:
            subprocess.run(["make", "-C", str(APP), "-f", "tests/Makefile.sim", "-j2", "CXX="+args.cxx,
                            "BUILD_DIR="+str(build), "MAGAOX_REPO="+str(args.magaox_repo.resolve()),
                            "TRAINING_START="+str(CONTROL_START-c["seconds"]*FS+8),
                            "TRAINING_END="+str(CONTROL_START), "ACCEL_HISTORY="+str(c["history"]),
                            "QR_PRIOR="+str(c["prior"])], check=True, stdout=log, stderr=subprocess.STDOUT)
        libraries[c["name"]] = bridge(build/"libddspc_sim.so"); builds[c["name"]] = build
    methods, results, validation_prefix, training, validation_models = aosim_methods(args.magaox_repo.resolve()), {}, {}, {}, {}
    source_prefix, full_records, baselines = {}, {}, {}
    for stage, interval in (("validation", VALIDATION), ("final", FINAL)):
        results[stage] = {}
        for scale in (1, 100):
            raw, pointing, units = source(next(iter(libraries.values())), interval[1], scale)
            if stage == "validation":
                source_prefix[scale] = raw.copy(), pointing.copy()
            else:
                np.testing.assert_array_equal(raw[:VALIDATION[1]], source_prefix[scale][0])
                np.testing.assert_array_equal(pointing[:VALIDATION[1]], source_prefix[scale][1])
            baseline = simulate(next(iter(libraries.values())), methods, raw, pointing, units, train=CONTROL_START, mode=0)
            results[stage][scale] = {}
            if stage == "final":
                baselines[scale] = baseline
            for c in configs:
                key, model = identity(c), {}
                data = simulate(libraries[c["name"]], methods, raw, pointing, units, train=CONTROL_START,
                                mode=c["mode"], calibration_model=model)
                np.testing.assert_array_equal(pair(data[:CONTROL_START], "yx", "yy"), pair(baseline[:CONTROL_START], "yx", "yy"))
                result = evaluate(data, baseline, model, c, interval)
                results[stage][scale][key] = result
                if stage == "validation":
                    training[scale, key] = training_diagnostics(data, model, c)
                    validation_prefix[scale, key] = hashlib.sha256(data.tobytes()).hexdigest()
                    validation_models[scale, key] = model
                else:
                    assert hashlib.sha256(data[:VALIDATION[1]].tobytes()).hexdigest() == validation_prefix[scale, key]
                    np.testing.assert_array_equal(model["theta"], validation_models[scale, key]["theta"])
                    np.testing.assert_array_equal(model["gain"], validation_models[scale, key]["gain"])
                    full_records[scale, key] = data
                    np.savez(out/f"model_{scale}_{key}.npz", **model)
                print(stage, scale, key, "forecast", round(result["prediction_h2_rms"], 7),
                      "residual", round(result["residual_rms"], 7), flush=True)
        if stage == "validation":
            (out/"validation_metrics.json").write_text(json.dumps(results[stage], indent=2)+"\n")
            locked = select(results[stage])
            selection_bytes = (json.dumps(locked, indent=2)+"\n").encode()
            (out/"selection.json").write_bytes(selection_bytes)
            print("LOCKED", locked["choices"], flush=True)
        else:
            assert (out/"selection.json").read_bytes() == selection_bytes
    rows = []
    for stage, scales in results.items():
        for scale, records in scales.items():
            for c in configs:
                key = identity(c)
                rows.append(dict(stage=stage, scale=scale, identity=key, **c, **training[scale, key], **records[key]))
    with (out/"metrics.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    (out/"final_metrics.json").write_text(json.dumps(results["final"], indent=2)+"\n")
    figures(full_records, baselines, locked["choices"], out)
    files = [Path(__file__), TESTS/"sim_bridge.cpp", TESTS/"Makefile.sim", TESTS/"async_shadow.cpp",
             TESTS/"simulation_demo.py", TESTS/"prediction_information.py", APP/"qrd_rls.cpp", APP/"ar_controller.cpp",
             APP/"shadow_controller.cpp", args.magaox_repo/"apps/accelSim/accelSimModel.hpp",
             args.magaox_repo/"apps/aoSim/xapp/aoSim/__init__.py"]
    manifest = dict(compiler=subprocess.check_output([args.cxx, "--version"]).decode(),
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    binaries={key: hashlib.sha256((build/"libddspc_sim.so").read_bytes()).hexdigest() for key, build in builds.items()},
                    selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
                    validation_prefixes_exact=True, coefficients_unchanged_in_final=True, final_generated_after_selection=True,
                    final_no_control_rms={scale: rms(pair(data[FINAL[0]:FINAL[1]], "yx", "yy")) for scale, data in baselines.items()})
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
