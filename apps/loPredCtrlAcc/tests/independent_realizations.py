r"""\file independent_realizations.py
\brief Paired fixed-configuration QR DDSPC trials over eight independent seeded accelSim realizations.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import numpy as np

from calibration_generalization import CONTROL_START, FINAL, evaluate, training_diagnostics
from prediction_information import peak_coherence, rms
from simulation_demo import APP, TESTS, aosim_methods, bridge, simulate, source
from vibration_report import FS, pair, spectra, line_power
from matplotlib import pyplot as plt

SEEDS = (101, 211, 307, 401, 503, 601, 701, 809)
ROLES = {"no_control": (0, 0), "standard": (3, 5), "assisted_h5": (4, 5), "assisted_h1": (4, 1)}
COLORS = {"no_control": "#555555", "standard": "#0072b2", "assisted_h5": "#009e73", "assisted_h1": "#e69f00"}


def digest(path):
    """Hash a complete recorded artifact, without modifying it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_phases(pointing):
    """Audit four initial modal phases from the plant-only first250ms; never supply these values to a controller."""
    y = pointing.reshape(-1, 2)[:2000]
    t = np.arange(len(y))/8000
    basis = np.column_stack([function(2*np.pi*hz*t) for hz in (28, 56) for function in (np.sin, np.cos)])
    coefficient = np.linalg.lstsq(basis, y, rcond=None)[0]
    return {f"{hz}_{axis}": float(np.degrees(np.arctan2(coefficient[2*j+1, axis], coefficient[2*j, axis])))
            for j, hz in enumerate((28, 56)) for axis in (0, 1)}


def source_bank(path, lib, repository, compiler):
    """Record actual source pairs once, or verify/replay an existing bank so compiler comparisons share exact inputs."""
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path/"manifest.json"
    expected = dict(seeds=list(SEEDS), scales=[1, 100], frames=FINAL[1], adc_hz=8000, adc_channels=[0, 1],
                    producer_header_sha256=digest(repository/"apps/accelSim/accelSimModel.hpp"))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for key, value in expected.items():
            if manifest[key] != value:
                raise RuntimeError("Source bank protocol/source mismatch: "+key)
        for entry in manifest["records"]:
            if digest(path/entry["file"]) != entry["sha256"]:
                raise RuntimeError("Source bank payload changed")
        return manifest
    records = []
    for seed in SEEDS:
        for scale in (1, 100):
            adc, pointing, units = source(lib, FINAL[1], scale, seed=seed)
            name = f"seed_{seed}_scale_{scale}.npz"
            np.savez(path/name, adc=adc, pointing=pointing, units=units)
            records.append(dict(seed=seed, scale=scale, file=name, sha256=digest(path/name),
                                adc_sha256=hashlib.sha256(adc.tobytes()).hexdigest(),
                                pointing_sha256=hashlib.sha256(pointing.tobytes()).hexdigest(),
                                adc_rail_samples=int(((adc == 0) | (adc == 4095)).sum()),
                                initial_phase_deg=initial_phases(pointing)))
        print("SOURCE", seed, flush=True)
    # The recording is authoritative on replay. Retain the original bridge as provenance;
    # unrelated controller-adapter edits must not change a frozen input bank's identity.
    manifest = dict(**expected, producer_bridge_sha256=digest(TESTS/"sim_bridge.cpp"),
                    producer_compiler=compiler, records=records)
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def broadband_rms(frequency, power):
    """Integrate5–1000Hz residual PSD excluding the declared±2Hz vibration-line bands; no signal is filtered."""
    band = (frequency >= 5) & (frequency <= FS/2) & (abs(frequency-28) > 2) & (abs(frequency-56) > 2)
    return float(np.sqrt(power[band].sum()*(frequency[1]-frequency[0])))


def summarize(values):
    """Resample independent paired-trial effects, never correlated WFS frames; intervals are exploratory at n=8."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Complete finite per-realization effects required")
    rng = np.random.default_rng(20261106)
    bootstrap = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                sample_std=float(values.std(ddof=1)), wins=int((values > 0).sum()), ties=int((values == 0).sum()),
                mean_ci95_percentile=np.percentile(bootstrap, [2.5, 97.5]).tolist())


def comparisons(rows):
    """Pair each assisted score with its own seed/amplitude standard baseline; never pool compilers or amplitudes."""
    indexed = {(r["scale"], r["seed"], r["controller"]): r for r in rows}
    paired, aggregate = [], {}
    measures = ("residual_rms", "tip_rms", "tilt_rms", "prediction_h2_rms", "broadband_rms")
    for scale in (1, 100):
        aggregate[scale] = {}
        for role in ("assisted_h5", "assisted_h1"):
            subset = []
            for seed in SEEDS:
                baseline, assisted = indexed[scale, seed, "standard"], indexed[scale, seed, role]
                row = dict(scale=scale, seed=seed, controller=role,
                           **{metric+"_improvement_percent": 100*(1-assisted[metric]/baseline[metric]) for metric in measures},
                           **{f"line_{hz}_advantage_db": assisted[f"suppression_{hz}_db"]-baseline[f"suppression_{hz}_db"]
                              for hz in (28, 56)},
                           residual_rms_difference=baseline["residual_rms"]-assisted["residual_rms"])
                paired.append(row); subset.append(row)
            aggregate[scale][role] = {metric: summarize([r[metric] for r in subset]) for metric in subset[0]
                                      if metric not in ("scale", "seed", "controller")}
    return paired, aggregate


def save_csv(path, rows):
    """Write all cases with missing non-controller predictions represented by empty CSV fields."""
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)


def figures(paired, summary, time_rows, psds, out):
    """Plot per-seed paired effects, whole post-calibration RMS trajectories and median/IQR residual spectra."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis, scale in zip(axes, (1, 100)):
        for offset, role in ((-.1, "assisted_h5"), (.1, "assisted_h1")):
            values = [r["residual_rms_improvement_percent"] for r in paired if r["scale"] == scale and r["controller"] == role]
            axis.plot(np.arange(len(SEEDS))+offset, values, "o-", color=COLORS[role], label=role)
            result = summary[scale][role]["residual_rms_improvement_percent"]
            low, high = result["mean_ci95_percentile"]
            axis.vlines(len(SEEDS)+offset, low, high, color=COLORS[role], linewidth=2)
            axis.plot(len(SEEDS)+offset, result["mean"], "s", color=COLORS[role])
        axis.axhline(0, color="black", linewidth=.8)
        axis.set(title=f"{scale}× ADC source; positive is lower WFS RMS than paired standard", ylabel="RMS improvement (%)")
        axis.grid(alpha=.25)
    axes[0].legend()
    axes[-1].set(xticks=np.arange(len(SEEDS)+1), xticklabels=[str(s) for s in SEEDS]+["Mean /95% CI"], xlabel="Independent realization seed")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(out/("paired_improvement."+extension), dpi=150, bbox_inches="tight")
    plt.close(fig)
    for scale in (1, 100):
        fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
        for axis, seed in zip(axes.flat, SEEDS):
            for role in ROLES:
                subset = [r for r in time_rows if r["scale"] == scale and r["seed"] == seed and r["controller"] == role]
                axis.semilogy([r["time_s"] for r in subset], [r["rms"] for r in subset], color=COLORS[role], label=role)
            axis.axvspan(FINAL[0]/FS, FINAL[1]/FS, color="grey", alpha=.1)
            axis.set(title=f"Seed{seed}", ylabel="50-ms WFS RMS")
            axis.grid(alpha=.25)
        axes[0, 0].legend(fontsize=8, ncol=2)
        for axis in axes[-1]:
            axis.set_xlabel("Simulation time (s); shaded interval is evaluation")
        fig.suptitle(f"Source{scale}×: unchanged4s fits; all held/limited frames included")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(out/(f"rms_scale_{scale}."+extension), dpi=150, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for row, scale in enumerate((1, 100)):
        for role in ROLES:
            f = psds[scale, SEEDS[0], role][0]
            power = np.array([psds[scale, seed, role][1] for seed in SEEDS])
            low, middle, high = np.percentile(power, [25, 50, 75], axis=0)
            for column, hz in enumerate((28, 56)):
                axis = axes[row, column]
                axis.semilogy(f, middle, color=COLORS[role], label=role)
                axis.fill_between(f, low, high, color=COLORS[role], alpha=.1)
                axis.set(xlim=(hz-5, hz+5), title=f"{scale}×: {hz}Hz", xlabel="Frequency (Hz)", ylabel="WFS units²/Hz")
                axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Median residual PSD; shading is between-realization IQR, not a confidence interval")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(out/("residual_psd."+extension), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run eight paired fixed-policy trials per amplitude, with no per-seed or result-dependent parameter choice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--magaox-repo", type=Path, default=Path.home()/"source/MagAOX")
    parser.add_argument("--bank", type=Path, default=APP/"build/independent-realization-source")
    parser.add_argument("--output", type=Path, default=TESTS/"results/independent_realizations")
    args = parser.parse_args()
    if socket.gethostname().lower().startswith("exao"):
        parser.error("Personal development workstation only")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    protocol = dict(seeds=SEEDS, source_scales=[1, 100], adc_channels=[0, 1], calibration_frames=[0, CONTROL_START],
                    evaluation_frames=FINAL, controllers=ROLES, forgetting=.9995, prior=.001, control_penalty=.02,
                    wfs_history=6, command_history=5, horizon=3, correction_delay_frames=2,
                    calibration_seed="10000+vibration_seed", wfs_seed="20000+vibration_seed",
                    broadband="5..1000Hz excluding±2Hz around28/56", bootstrap_replicates=20000, bootstrap_seed=20261106,
                    note="Eight trial units, paired within seed; amplitudes and compiler replays are not independent trials.")
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    libraries, builds = {}, {}
    for history in (5, 1):
        build = APP/"build"/f"generalization-endaligned-{Path(args.cxx).name}-d4_h{history}_p001"
        with (out/f"history_{history}-build.log").open("w") as log:
            subprocess.run(["make", "-C", str(APP), "-f", "tests/Makefile.sim", "-j2", "CXX="+args.cxx,
                            "BUILD_DIR="+str(build), "MAGAOX_REPO="+str(args.magaox_repo.resolve()),
                            "TRAINING_START=8", "TRAINING_END=8000", f"ACCEL_HISTORY={history}", "QR_PRIOR=.001"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        libraries[history] = bridge(build/"libddspc_sim.so"); builds[history] = build
    compiler = subprocess.check_output([args.cxx, "--version"]).decode()
    bank = args.bank.resolve()
    bank_manifest = source_bank(bank, libraries[5], args.magaox_repo.resolve(), compiler)
    (out/"source_manifest.json").write_text(json.dumps(bank_manifest, indent=2)+"\n")
    print("BANK READY", bank, flush=True)
    methods, rows, time_rows, psds, conditions = aosim_methods(args.magaox_repo.resolve()), [], [], {}, []
    for seed in SEEDS:
        for scale in (1, 100):
            recorded = np.load(bank/f"seed_{seed}_scale_{scale}.npz")
            raw, pointing, units = recorded["adc"], recorded["pointing"], recorded["units"]
            baseline = None
            for role, (mode, history) in ROLES.items():
                model = {} if mode else None
                data = simulate(libraries[history or 5], methods, raw, pointing, units, train=CONTROL_START,
                                mode=mode, calibration_model=model, calibration_seed=10000+seed, wfs_seed=20000+seed)
                if mode == 0:
                    baseline = data
                else:
                    np.testing.assert_array_equal(pair(data[:CONTROL_START], "yx", "yy"), pair(baseline[:CONTROL_START], "yx", "yy"))
                    np.testing.assert_array_equal(pair(data, "disturbance_x", "disturbance_y"), pair(baseline, "disturbance_x", "disturbance_y"))
                    assert model["version"] == 7992
                b = data[FINAL[0]:FINAL[1]]
                values = pair(b, "yx", "yy")
                f, power = spectra(values); psds[scale, seed, role] = f, power
                metrics = evaluate(data, baseline, model, dict(mode=mode), FINAL) if mode else dict(
                    prediction_h2_rms=None, residual_rms=rms(values), availability=0., ready_fraction=0., variance_reduction=0.,
                    peak_command=0., limit_fraction=0., suppression_28_db=0., suppression_56_db=0.)
                command = pair(data, "applied_x", "applied_y")
                row = dict(seed=seed, scale=scale, controller=role, **metrics,
                           tip_rms=rms(values[:, 0]), tilt_rms=rms(values[:, 1]), broadband_rms=broadband_rms(f, power),
                           saturated_final_frames=int(np.any(abs(command[FINAL[0]:FINAL[1]]) >= 4-1e-6, axis=1).sum()),
                           peak_command_full_run=float(np.max(abs(command))),
                           saturated_full_frames=int(np.any(abs(command) >= 4-1e-6, axis=1).sum()))
                rows.append(row)
                if mode:
                    config = dict(mode=mode, seconds=4, prior=.001)
                    conditions.append(dict(seed=seed, scale=scale, controller=role, **training_diagnostics(data, model, config)))
                    np.savez(out/f"model_{seed}_{scale}_{role}.npz", **model)
                    if role == "assisted_h5":
                        coherent = peak_coherence(pair(data[:CONTROL_START], "ax", "ay"),
                                                  pair(baseline[:CONTROL_START], "yx", "yy")-pair(baseline[:CONTROL_START], "dm_x", "dm_y"))
                        conditions[-1]["calibration_coherence_28"] = float(np.mean([x["coherence"] for x in coherent if x["line_hz"]==28 and x["adc_channel"]==x["wfs_axis"]]))
                        conditions[-1]["calibration_coherence_56"] = float(np.mean([x["coherence"] for x in coherent if x["line_hz"]==56 and x["adc_channel"]==x["wfs_axis"]]))
                for first in range(CONTROL_START, len(data)-99, 100):
                    y = pair(data[first:first+100], "yx", "yy")
                    time_rows.append(dict(seed=seed, scale=scale, controller=role, time_s=(first+50)/FS,
                                          rms=rms(y), tip_rms=rms(y[:, 0]), tilt_rms=rms(y[:, 1])))
                save_csv(out/"per_realization.csv", rows)
                print(seed, scale, role, "RMS", round(row["residual_rms"], 7), flush=True)
    paired, aggregate = comparisons(rows)
    save_csv(out/"paired.csv", paired); save_csv(out/"time_rms.csv", time_rows); save_csv(out/"conditioning.csv", conditions)
    (out/"summary.json").write_text(json.dumps(aggregate, indent=2)+"\n")
    np.savez(out/"spectra.npz", frequency=next(iter(psds.values()))[0],
             **{f"scale_{scale}_seed_{seed}_{role}": value[1] for (scale, seed, role), value in psds.items()})
    figures(paired, aggregate, time_rows, psds, out)
    files = [Path(__file__), TESTS/"simulation_demo.py", TESTS/"calibration_generalization.py", TESTS/"prediction_information.py",
             TESTS/"sim_bridge.cpp", TESTS/"async_shadow.cpp", TESTS/"Makefile.sim", APP/"ar_controller.cpp", APP/"qrd_rls.cpp",
             APP/"shadow_controller.cpp", args.magaox_repo/"apps/accelSim/accelSimModel.hpp", args.magaox_repo/"apps/aoSim/xapp/aoSim/__init__.py"]
    manifest = dict(compiler=compiler, canonical_bank_sha256=digest(bank/"manifest.json"),
                    source_sha256={str(path): digest(path) for path in files},
                    binary_sha256={str(history): digest(build/"libddspc_sim.so") for history, build in builds.items()},
                    trials_per_amplitude=len(SEEDS), row_count=len(rows), new_controller_or_tuning=False)
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
