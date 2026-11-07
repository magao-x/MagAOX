r"""\file vibration_demo.py
\brief Build and run the existing causal QR harness, reporting three offline 28/56-Hz suppression cases.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ddspc-vibration-demo-matplotlib")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vibration_report import FS, metrics, pair, self_test, spectra

TESTS = Path(__file__).resolve().parent
APP = TESTS.parent
REPO = APP.parents[1]
MODELS = (0, 3, 4)
LABELS = ("No predictive control", "Standard DDSPC", "Accelerometer-assisted DDSPC")
COLORS = ("#555555", "#0072b2", "#009e73")


def sha256(path):
    """Identify an input or executable without modifying it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(data, config):
    """Check common disturbances/calibration, causal evidence, bounded requests and the actual plant response."""
    train = int(config["train"])
    baseline = data[data["controller"] == 0]
    g = np.array([[1, .08], [-.05, .9]], dtype=np.float32).astype(float)
    errors = {}
    for model in MODELS:
        b = data[data["controller"] == model]
        for column in ("dx", "dy", "ax", "ay", "sensor_index", "sensor_time_ns",
                       "sensor_available_ns", "wfs_available_ns"):
            np.testing.assert_array_equal(b[column], baseline[column])
        np.testing.assert_array_equal(pair(b[:train], "applied_x", "applied_y"),
                                      pair(baseline[:train], "applied_x", "applied_y"))
        assert not b["invalid_changes"].any() and not b["overflows"].any()
        requested = pair(b, "request_x", "request_y")
        applied = pair(b, "applied_x", "applied_y")
        assert np.isfinite(requested).all() and np.isfinite(applied).all()
        assert np.max(np.abs(requested)) <= float(config["limit"])
        np.testing.assert_array_equal(requested, applied)  # Declared zero extra transport delay.
        assert np.all(b["sensor_time_ns"] <= 1e9 + (b["frame"] + 1) * 500000)
        assert np.all(b["sensor_available_ns"] <= b["wfs_available_ns"])
        if model != 0:
            valid = (b["valid"] != 0) & (b["frame"] >= train)
            assert valid.any()
            assert np.all(b["model_sequence"][valid] < b["origin"][valid])
            assert np.all(b["origin"][valid] <= b["frame"][valid])
            assert np.all(b["age_ns"][valid] <= int(config["max_age_ns"]))
            invalid = np.flatnonzero((b["valid"] == 0) & (b["frame"] >= train))
            np.testing.assert_array_equal(requested[invalid], requested[invalid - 1])
        # Independent reconstruction from actual applied inputs; no model proposal or disturbance subtraction.
        dm = np.zeros(2)
        residual = np.empty_like(applied)
        for i in range(len(b)):
            residual[i] = [b["dx"][i], b["dy"][i]] + g @ dm
            dm = float(config["dm_pole"]) * dm + (1 - float(config["dm_pole"])) * applied[i]
        observed = pair(b, "rx", "ry")
        error = float(np.max(np.abs(observed - residual)))
        # Float32 Eigen arithmetic and 10-significant-digit CSV serialization differ from this double replay.
        bound = 32 * float(np.finfo(np.float32).eps) * max(1., float(np.max(np.abs(observed))))
        assert error <= bound, (model, error, bound)
        errors[str(model)] = dict(max_plant_reconstruction_error=error, float_replay_bound=bound)
    return errors


def plots(data, config, output):
    """Plot the same measured residual interval and fixed Welch PSD for the three requested trajectories."""
    start = int(config["train"]) + 1000
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    for model, label, color in zip(MODELS, LABELS, COLORS):
        b = data[(data["controller"] == model) & (data["frame"] >= start) & (data["frame"] < start + 400)]
        t = (b["frame"] - start) / FS
        for axis, column in enumerate(("yx", "yy")):
            axes[axis, 0].plot(t, b[column], label=label, color=color, linewidth=1)
            if model:
                axes[axis, 1].plot(t, b[column], label=label, color=color, linewidth=1)
    for axis, name in enumerate(("Tip", "Tilt")):
        axes[axis, 0].set_ylabel(name + " WFS residual (synthetic units)")
        for ax in axes[axis]:
            ax.grid(alpha=.25)
            ax.set_xlabel("Time within scored interval (s)")
    axes[0, 0].set_title("Original and corrected residuals")
    axes[0, 1].set_title("Corrected residuals, expanded scale")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Offline QR DDSPC: identical 28 + 56 Hz disturbance and measurement noise")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / ("time_series." + extension), dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for model, label, color in zip(MODELS, LABELS, COLORS):
        b = data[(data["controller"] == model) & (data["frame"] >= start)]
        hz, psd = spectra(pair(b, "yx", "yy"))
        ax.semilogy(hz, psd, label=label, color=color)
    for hz in (28, 56):
        ax.axvline(hz, color="black", linestyle=":", alpha=.4)
    ax.set(xlim=(5, 150), xlabel="Frequency (Hz)", ylabel="Mean tip/tilt PSD (synthetic units²/Hz)",
           title="Offline WFS residuals; 28 and 56 Hz marked")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / ("residual_psd." + extension), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """Compile only the offline harness, run its unchanged mixed QR case and write a compact auditable report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="c++", help="C++ compiler; use a separate build directory per compiler")
    parser.add_argument("--output", type=Path, default=TESTS / "results/minimal_vibration_demo")
    args = parser.parse_args()
    if socket.gethostname().lower().startswith("exao"):
        parser.error("Run this offline demonstration on a personal development workstation")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    build = APP / "build" / ("minimal-vibration-" + Path(args.cxx).name)
    raw = build / "trace"
    build.mkdir(parents=True, exist_ok=True)
    with (build / "build.log").open("w") as log:
        subprocess.run(["make", "-C", str(APP), "-f", "tests/Makefile.vibration", "-j2",
                        "BUILD_DIR=" + str(build), "CXX=" + args.cxx], check=True, stdout=log, stderr=subprocess.STDOUT)
    binary = build / "ddspc_vibration"
    with (build / "simulation.log").open("w") as log:
        subprocess.run([str(binary), str(raw), "mixed_qr"], check=True, stdout=log, stderr=subprocess.STDOUT)
    with (raw / "config.csv").open() as source:
        config, = list(csv.DictReader(source))
    assert config["qrd"] == "1" and config["strategy"] == "baseline" and config["broadband"] == "0"
    trace = raw / "mixed_qr.csv"
    with trace.open() as source:
        columns = source.readline().strip().split(",")
        data = np.loadtxt(source, delimiter=",", dtype=[(name, np.float64) for name in columns])
    self_test()
    checks = verify(data, config)
    rows = [row for row in metrics(data, config) if row["controller"] in MODELS]
    for row, label in zip(rows, LABELS):
        row["name"] = label
        row["rms_reduction"] = 1 - row["rms"] / rows[0]["rms"]
    with (output / "metrics.csv").open("w") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plots(data, config, output)
    rms_gain = 1 - rows[2]["rms"] / rows[1]["rms"]
    variance_gain = 1 - rows[2]["variance"] / rows[1]["variance"]
    table = ["| Case | WFS RMS | RMS reduction | Variance reduction | 28 Hz dB | 56 Hz dB |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        table.append(f"| {row['name']} | {row['rms']:.5f} | {100*row['rms_reduction']:.2f}% | "
                     f"{100*row['variance_reduction']:.2f}% | {row['suppression_28_db']:.2f} | "
                     f"{row['suppression_56_db']:.2f} |")
    report = ("# Minimal offline vibration suppression demonstration\n\n"
              "Synthetic 28 + 56 Hz, causal QR DDSPC; no instrument or shared-memory I/O.\n\n" +
              "\n".join(table) + f"\n\nAccelerometer assistance {'improves' if rms_gain > 0 else 'does not improve'} "
              f"standard DDSPC: **{100*rms_gain:.2f}% lower RMS**, **{100*variance_gain:.2f}% lower variance**.\n\n"
              "Reductions in the table are relative to no predictive control. PSD suppression integrates ±2 Hz "
              "bands using the existing Hann/Welch metric. RMS is sqrt(mean((tip²+tilt²)/2)).\n\n"
              "![WFS time series](time_series.png)\n\n![Residual PSD](residual_psd.png)\n\n"
              "2 kHz WFS, 1.2 s common initial calibration, 4 s control; score the final 3.5 s, including "
              "invalid/held frames. Both learned controllers retain actual applied-command history; only the "
              "assisted model receives accelerometer histories. The runner also computes auxiliary Legacy/oracle "
              "trajectories; neither enters these three cases or their decision data. No recovery policy is used.\n\n"
              "Synthetic plant: x[i+1]=.1*x[i]+.9*u[i], y[i]=d[i]+G*x[i]+noise, "
              "G=[[1,.08],[-.05,.9]]. Commands are bounded absolute inputs; their nominal effect is one WFS "
              "frame later. Sensor features are created 125 µs before WFS origin and available 1 µs later. "
              "This selects one synchronized feature per WFS frame; it is not a full 8-kHz acquisition or UINT16 "
              "ADC simulation. Noise amplitudes, gains and units are synthetic.\n\n"
              f"See [the demonstration notes]({os.path.relpath(TESTS / 'results/minimal_vibration_demo.md', output)}) "
              "for accelSim inspection, causal "
              "conventions and the minimum later simulation-stream integration work.\n")
    (output / "README.md").write_text(report)
    sources = ["ar_controller.cpp", "recursive_least_squares.cpp", "qrd_rls.cpp", "shadow_controller.cpp",
               "tests/vibration.cpp", "tests/vibration.hpp", "tests/vibration_main.cpp",
               "tests/vibration_report.py", "tests/vibration_demo.py", "tests/Makefile.vibration"]
    manifest = dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
                    host=socket.gethostname(), compiler=subprocess.check_output([args.cxx, "--version"]).decode(),
                    config=config, causal_and_plant_checks=checks, trace_sha256=sha256(trace),
                    binary_sha256=sha256(binary), source_sha256={name: sha256(APP / name) for name in sources},
                    numpy=np.__version__, matplotlib=matplotlib.__version__,
                    assisted_rms_reduction_vs_standard=rms_gain,
                    assisted_variance_reduction_vs_standard=variance_gain)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(table))
    print(f"Assistance: {100*rms_gain:.2f}% lower RMS than standard DDSPC. Results: {output}")


if __name__ == "__main__":
    main()
