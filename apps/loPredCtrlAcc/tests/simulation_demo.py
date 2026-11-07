r"""\file simulation_demo.py
\brief Local connection of actual accelSim physics/ADC, aoSim DM/WFS methods and DDSPC; optional private shmim transport.
"""
import argparse
import ast
import csv
import ctypes as ct
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import socket
import subprocess
from types import MethodType, SimpleNamespace

import numpy as np
from vibration_demo import plots
from vibration_report import FS, pair, spectra, line_power
from matplotlib import pyplot as plt

TESTS = Path(__file__).resolve().parent
APP = TESTS.parent


class Prediction(ct.Structure):
    """Fixed C ABI metadata; version/evidence refer to the snapshot actually read."""
    _fields_ = [(name, ct.c_int) for name in ("valid", "ready", "reason", "rank")] + [
        (name, ct.c_uint64) for name in ("version", "sequence", "age")] + [
        ("ax", ct.c_float), ("ay", ct.c_float)] + [(name, ct.c_double) for name in
            ("fit_rmse", "theta_norm", "control_error", "gain_norm")]


class MemoryImage:
    """Host-memory transport boundary; has no stream name or facility for opening a device."""
    def __init__(self):
        """Start with a zero two-mode image and an independently recorded consumed value."""
        self.data = np.zeros((2, 1), dtype=np.float32)
        self.last_read = self.data.copy()

    def get_data(self, wait=False):
        """Record the exact input returned to aoSim.update_dm, not a proposed correction."""
        assert wait is False
        self.last_read = self.data.copy()
        return self.last_read.copy()

    def write(self, data):
        """Copy one finite modal image without any shared-memory or external output operation."""
        data = np.asarray(data, dtype=np.float32)
        if data.shape != (2, 1) or not np.isfinite(data).all():
            raise ValueError("Expected a finite two-mode simulation image")
        self.data = data.copy()


def aosim_methods(repository):
    """Load only the two existing numerical method bodies; never import/start XDevice or create streams."""
    path = repository / "apps/aoSim/xapp/aoSim/__init__.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    klass, = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "aoSim"]
    methods = [node for node in klass.body if isinstance(node, ast.FunctionDef) and
               node.name in ("update_dm", "update_wfs")]
    assert {m.name for m in methods} == {"update_dm", "update_wfs"}
    namespace = {"np": np}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["update_dm"], namespace["update_wfs"]


def plant(methods, noise=0., lag=0, dm=None, wfs=None):
    """Bind the actual aoSim delay/WFS methods to explicit local images; skip setup and its unrelated sine."""
    p = SimpleNamespace(_dm=dm if dm is not None else MemoryImage(),
                        _wfs=wfs if wfs is not None else MemoryImage(), _nmodes=2, _noise=noise, _lag=lag,
                        _current_dm_state=np.zeros((2, 1), dtype=np.float32),
                        _current_disturbance=np.zeros((2, 1), dtype=np.float32),
                        _dm_command_history=np.zeros((lag + 1, 2, 1), dtype=np.float32))
    p.update_dm = MethodType(methods[0], p)
    p.update_wfs = MethodType(methods[1], p)
    return p


def bridge(path):
    """Declare the limited array ABI; it exposes no hardware/stream names or production application instance."""
    lib = ct.CDLL(str(path))
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, shape=(2,), flags="C_CONTIGUOUS")
    u16 = np.ctypeslib.ndpointer(dtype=np.uint16, shape=(2,), flags="C_CONTIGUOUS")
    raw_frame = np.ctypeslib.ndpointer(dtype=np.uint16, shape=(4, 2), flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, shape=(2,), flags="C_CONTIGUOUS")
    units_type = np.ctypeslib.ndpointer(dtype=np.float64, shape=(5,), flags="C_CONTIGUOUS")
    lib.ddspc_sim_prediction_size.restype = ct.c_size_t
    assert lib.ddspc_sim_prediction_size() == ct.sizeof(Prediction)
    lib.ddspc_sim_source_create.argtypes = [ct.c_double, ct.c_uint64]
    lib.ddspc_sim_source_create.restype = ct.c_void_p
    lib.ddspc_sim_source_destroy.argtypes = [ct.c_void_p]
    lib.ddspc_sim_source_step.argtypes = [ct.c_void_p, u16, f64]
    lib.ddspc_sim_source_units.argtypes = [ct.c_void_p, units_type]
    lib.ddspc_sim_source_audit.argtypes = [ct.c_void_p, f64]
    lib.ddspc_sim_controller_create.argtypes = [ct.c_int, units_type]
    lib.ddspc_sim_controller_create.restype = ct.c_void_p
    lib.ddspc_sim_controller_destroy.argtypes = [ct.c_void_p]
    lib.ddspc_sim_model_features.argtypes = [ct.c_void_p]
    lib.ddspc_sim_model_features.restype = ct.c_int
    lib.ddspc_sim_model_history.argtypes = [ct.c_void_p]
    lib.ddspc_sim_model_history.restype = ct.c_int
    array = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    lib.ddspc_sim_copy_model.argtypes = [ct.c_void_p, array, ct.c_int, array, ct.c_int, ct.POINTER(ct.c_uint64)]
    lib.ddspc_sim_query.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_uint64, ct.c_uint64, f32, raw_frame, f32,
                                    ct.POINTER(Prediction)]
    lib.ddspc_sim_accept.argtypes = [ct.c_void_p, f32]
    return lib


def source(lib, count, scale, seed=42):
    """Advance the existing seeded accelSim source at 8 kHz, keeping simultaneous ADC/plant pairs together."""
    handle = lib.ddspc_sim_source_create(scale, seed)
    if not handle:
        raise RuntimeError("Cannot configure accelSim numerical source")
    adc = np.empty((count, 4, 2), dtype=np.uint16)
    pointing = np.empty((count, 4, 2), dtype=np.float64)
    units = np.empty(5, dtype=np.float64)
    try:
        assert lib.ddspc_sim_source_units(handle, units) == 0
        for k in range(count):
            for j in range(4):
                assert lib.ddspc_sim_source_step(handle, adc[k, j], pointing[k, j]) == 0
    finally:
        lib.ddspc_sim_source_destroy(handle)
    return adc, pointing, units


def copy_model(lib, handle, mode, allow_unready_diagnostic=False):
    """Copy a coherent QR fit for scoring; unready diagnostics require explicit opt-in and never authorize corrections."""
    features = lib.ddspc_sim_model_features(handle)
    history = lib.ddspc_sim_model_history(handle)
    base = (3+2*history-1)*2
    if (mode == 3 and features != base) or (mode == 4 and (features < base+2 or (features-base) % 2)):
        raise RuntimeError("Unexpected canonical simulation dimensions")
    theta, gain, version = np.empty(6*features, np.float32), np.empty(6*(features-6), np.float32), ct.c_uint64()
    status = lib.ddspc_sim_copy_model(handle, theta, theta.size, gain, gain.size, ct.byref(version))
    if status < 0 or (status != 0 and not allow_unready_diagnostic):
        raise RuntimeError("No ready calibration snapshot")
    return dict(theta=theta.reshape((6, features), order="F"), gain=gain.reshape((6, features-6), order="F"),
                version=version.value, history=history, calibration_ready=status == 0)


def simulate(lib, methods, adc, pointing, units, train=2400, mode=0, evaluation_probe=.005, images=None,
             calibration_model=None, calibration_seed=100, wfs_seed=303, allow_unready_diagnostic=False):
    """Run the existing causal loop with paired noise/probe seeds and an optional diagnostic snapshot copy."""
    n = len(adc)
    wfs_unit, accel_unit = units[:2]
    correction = plant(methods, dm=images.images["correction"] if images else None)
    injection = plant(methods, dm=images.images["disturbance"] if images else None)
    sensor = plant(methods, noise=.03 * wfs_unit, wfs=images.images["wfs"] if images else None)
    np.random.seed(wfs_seed)  # Same WFS noise realization in each paired trajectory; no source uses this generator.
    exploration = np.random.default_rng(calibration_seed).uniform(-np.sqrt(3), np.sqrt(3), (n, 2)).astype(np.float32)
    handle = lib.ddspc_sim_controller_create(mode - 3, units) if mode else None
    if mode and not handle:
        raise RuntimeError("Cannot create causal DDSPC pair")
    fields = ("controller", "frame", "yx", "yy", "rx", "ry", "request_x", "request_y", "applied_x",
              "applied_y", "dm_x", "dm_y", "disturbance_x", "disturbance_y", "valid", "ready", "rank",
              "version", "model_sequence", "age_ns", "ax", "ay", "time_ns", "first_sensor_ns",
              "fit_rmse", "theta_norm", "control_error", "gain_norm")
    data = np.zeros(n, dtype=[(field, np.float64) for field in fields])
    held = np.zeros(2, dtype=np.float32)
    try:
        for k in range(n):
            now = 1000000000 + (k + 1) * 500000
            first = now - 500000  # Last ADC pair is 125 us before WFS, available 1 us later.
            cstate = correction._current_dm_state.copy()
            dstate = injection._current_dm_state.copy()
            sensor._current_dm_state = cstate + dstate
            sensor.update_wfs()
            y = sensor._wfs.get_data(wait=False)[:, 0].copy()
            raw, disturbance_input = (images.source_frame(adc[k], pointing[k], k, first) if images else
                                       (adc[k], pointing[k, -1]))
            proposal = np.zeros(2, dtype=np.float32)
            info = Prediction()
            if mode:
                assert lib.ddspc_sim_query(handle, k, now, first, y, raw, proposal, ct.byref(info)) == 0
                if info.valid:
                    assert info.sequence < k and info.age <= 2000000
                decoded = ((raw.astype(float) - units[2]) * units[3] / units[4] / accel_unit).mean(axis=0)
                np.testing.assert_allclose([info.ax, info.ay], decoded, rtol=1e-6, atol=1e-6)
            if k < train:
                request = (.3 * wfs_unit * exploration[k]).astype(np.float32)
            elif mode == 0:
                request = np.zeros(2, dtype=np.float32)
            elif info.valid:
                request = proposal + (evaluation_probe * wfs_unit * exploration[k]).astype(np.float32)
            else:
                request = held.copy()
            limit = np.float32(4 * wfs_unit)
            if float(limit) > 4 * wfs_unit:
                limit = np.nextafter(limit, np.float32(0))
            request = np.clip(request, -limit, limit).astype(np.float32)
            if mode and k >= train and not info.valid:
                np.testing.assert_array_equal(request, held)
            # Independent additive simulated DM channels, each through the actual aoSim delay method.
            correction._dm.write(request.reshape(2, 1))
            injection._dm.write(disturbance_input.astype(np.float32).reshape(2, 1))
            correction.update_dm()
            injection.update_dm()
            consumed = correction._dm.last_read[:, 0].copy()
            np.testing.assert_array_equal(correction._dm_command_history[-1, :, 0], consumed)
            if images:
                consumed = images.consumed(consumed, cstate + dstate, k, now)
            if mode:
                assert lib.ddspc_sim_accept(handle, consumed) == 0
                if calibration_model is not None and k == train-1:
                    calibration_model.update(copy_model(lib, handle, mode, allow_unready_diagnostic))
            normalized = y / wfs_unit
            row = data[k]
            row["controller"], row["frame"] = mode, k
            row["yx"], row["yy"] = normalized
            row["rx"], row["ry"] = ((cstate + dstate) / wfs_unit)[:, 0]
            for prefix, value in (("request", request / wfs_unit), ("applied", consumed / wfs_unit),
                                  ("dm", cstate[:, 0] / wfs_unit), ("disturbance", dstate[:, 0] / wfs_unit)):
                row[prefix + "_x"], row[prefix + "_y"] = value
            for field in ("valid", "ready", "rank", "version", "age"):
                row["age_ns" if field == "age" else field] = getattr(info, field)
            row["model_sequence"] = info.sequence
            row["ax"], row["ay"] = info.ax, info.ay
            row["time_ns"], row["first_sensor_ns"] = now, first
            for field in ("fit_rmse", "theta_norm", "control_error", "gain_norm"):
                row[field] = getattr(info, field)
            held = consumed
    finally:
        if handle:
            lib.ddspc_sim_controller_destroy(handle)
    assert np.isfinite(pair(data, "yx", "yy")).all()
    return data


def score(data, train):
    """Reuse the existing fixed spectral bands and report all held/invalid rows in the same score window."""
    rows = []
    baseline = data[(data["controller"] == 0) & (data["frame"] >= train + 1000)]
    f, original = spectra(pair(baseline, "yx", "yy"))
    for model, name in ((0, "Disturbance only"), (3, "Standard QR DDSPC"), (4, "Assisted QR DDSPC")):
        b = data[(data["controller"] == model) & (data["frame"] >= train + 1000)]
        y = pair(b, "yx", "yy")
        _, psd = spectra(y)
        row = dict(controller=model, name=name, rms=float(np.sqrt(np.mean(y**2))),
                   variance=float(np.var(y, axis=0).mean()), availability=float(b["valid"].mean()),
                   command_rms=float(np.sqrt(np.mean(pair(b, "applied_x", "applied_y")**2))),
                   first_scored_second_rms=float(np.sqrt(np.mean(y[:2000]**2))),
                   last_second_rms=float(np.sqrt(np.mean(y[-2000:]**2))))
        row["variance_reduction"] = 1 - row["variance"] / float(np.var(pair(baseline, "yx", "yy"), axis=0).mean())
        for hz in (28, 56):
            row[f"suppression_{hz}_db"] = float(10 * np.log10(line_power(f, original, hz) / line_power(f, psd, hz)))
        rows.append(row)
    return rows


def full_time_series(data, destination):
    """Expose the whole scored trajectory and its variation, alongside the inherited 200-ms detail plot."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for mode, label, color in ((0, "Disturbance only", "#555555"), (3, "Standard QR", "#0072b2"),
                                (4, "Assisted QR", "#009e73")):
        b = data[(data["controller"] == mode) & (data["frame"] >= 3400)]
        t = (b["frame"] - 2400) / FS
        y = pair(b, "yx", "yy")
        for axis in range(2):
            axes[axis].plot(t, y[:, axis], color=color, linewidth=.5, label=label)
        bins = np.arange(0, len(b) - 99, 100)
        axes[2].plot(t[bins + 50], [np.sqrt(np.mean(y[i:i+100]**2)) for i in bins], color=color, label=label)
    for axis, label in enumerate(("Tip residual", "Tilt residual", "Two-axis RMS, 50-ms bins")):
        axes[axis].set_ylabel(label + " (normalized units)")
        axes[axis].grid(alpha=.25)
    axes[0].legend()
    axes[0].set_title("Complete scored interval: suppression varies over time")
    axes[2].set_xlabel("Time from start of control (s); score starts at 0.5 s")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(destination / ("full_time_series." + ext), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """Build the simulation-only bridge against the other repository and run comparable 28/56-Hz trajectories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magaox-repo", type=Path, default=Path.home() / "source/MagAOX")
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--training-end", type=int, help="Diagnostic: stop coefficient updates at this frame; keep validating fresh records")
    parser.add_argument("--accel-channels", nargs=2, type=int, default=[0, 1], help="Intended raw input pair; this two-channel simulator uses 0 1")
    parser.add_argument("--transport", choices=("memory", "shmim"), default="memory",
                        help="Optional create-only local ImageStreamIO roundtrip; never native instrument applications")
    parser.add_argument("--source-scales", nargs="+", type=float, default=[1., 100.])
    parser.add_argument("--output", type=Path, default=TESTS / "results/simulation_connection")
    args = parser.parse_args()
    if args.accel_channels != [0, 1]:
        parser.error("This simulation validates channels 0 and 1, in that order; no tip/tilt reinterpretation is implied")
    if socket.gethostname().lower().startswith("exao"):
        parser.error("Personal development workstation only")
    repository = args.magaox_repo.resolve()
    if args.training_end is not None and args.training_end < 100:
        parser.error("training-end must allow at least 100 calibration frames")
    build = APP / "build" / ("simulation-" + Path(args.cxx).name +
                             (f"-train{args.training_end}" if args.training_end is not None else ""))
    build.mkdir(parents=True, exist_ok=True)
    with (build / "build.log").open("w") as log:
        subprocess.run(["make", "-C", str(APP), "-f", "tests/Makefile.sim", "-j2", "BUILD_DIR=" + str(build),
                        "MAGAOX_REPO=" + str(repository), "CXX=" + args.cxx] +
                       ([f"TRAINING_END={args.training_end}"] if args.training_end is not None else []), check=True,
                       stdout=log, stderr=subprocess.STDOUT)
    lib = bridge(build / "libddspc_sim.so")
    methods = aosim_methods(repository)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    configurations = []
    namespaces = []
    for scale in args.source_scales:
        adc, pointing, units = source(lib, 10400, scale)
        traces = []
        for mode in (0, 3, 4):
            images = None
            try:
                if args.transport == "shmim":
                    from sim_shmim import SimImages
                    images = SimImages()
                    namespaces.append(dict(scale=scale, mode=mode, prefix=images.prefix))
                traces.append(simulate(lib, methods, adc, pointing, units, mode=mode, images=images))
            finally:
                if images:
                    images.close()
        for other in traces[1:]:
            np.testing.assert_array_equal(other["disturbance_x"], traces[0]["disturbance_x"])
            np.testing.assert_array_equal(other["disturbance_y"], traces[0]["disturbance_y"])
            np.testing.assert_array_equal(pair(other[:2400], "yx", "yy"), pair(traces[0][:2400], "yx", "yy"))
            np.testing.assert_array_equal(pair(other[:2400], "applied_x", "applied_y"),
                                          pair(traces[0][:2400], "applied_x", "applied_y"))
        data = np.concatenate(traces)
        rows = score(data, 2400)
        destination = output / f"scale_{scale:g}"
        destination.mkdir(exist_ok=True)
        plots(data, dict(train=2400), destination)
        full_time_series(data, destination)
        trace = build / f"trace_scale_{scale:g}.csv"
        np.savetxt(trace, data.view(np.float64).reshape(len(data), -1),
                   delimiter=",", header=",".join(data.dtype.names), comments="", fmt="%.12g")
        np.savez(build / f"source_scale_{scale:g}.npz", adc=adc, pointing=pointing, units=units)
        for row in rows:
            row["source_scale"] = scale
        all_rows.extend(rows)
        configurations.append(dict(source_scale=scale, wfs_unit_arcsec=units[0], accel_unit=units[1],
                                   adc_zero=units[2], volts_per_count=units[3], sensitivity=units[4],
                                   rail_samples=int(((adc == 0) | (adc == 4095)).sum()),
                                   adc_sha256=hashlib.sha256(adc.tobytes()).hexdigest(),
                                   pointing_sha256=hashlib.sha256(pointing.tobytes()).hexdigest(),
                                   trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest()))
        print(f"scale={scale:g}: RMS " + ", ".join(f"{r['name']}={r['rms']:.6f}" for r in rows) +
              f"; assistance RMS change={100*(1-rows[2]['rms']/rows[1]['rms']):.2f}%", flush=True)
    with (output / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    files = [repository / "apps/accelSim/accelSimModel.hpp", repository / "apps/aoSim/xapp/aoSim/__init__.py",
             TESTS / "sim_bridge.cpp", Path(__file__), TESTS / "Makefile.sim", TESTS / "async_shadow.cpp",
             TESTS / "sim_shmim.py"]
    files += [APP / name for name in ("ar_controller.cpp", "qrd_rls.cpp", "shadow_controller.cpp")]
    manifest = dict(transport=args.transport, namespaces=namespaces,
                    execution="Existing numerical code; not three native application processes", config=configurations,
                    compiler=subprocess.check_output([args.cxx, "--version"]).decode(),
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    binary_sha256=hashlib.sha256((build / "libddspc_sim.so").read_bytes()).hexdigest(),
                    runtime_modules={name: find_spec(name) is not None for name in
                                     ("ImageStreamIOWrap", "magaox", "purepyindi2", "xconf")},
                    adc_hz=8000, wfs_hz=2000, train=2400, evaluate=8000, aosim_lag=0,
                    history=6, future=3, accel_history=5, noise_fraction=.03,
                    training_end=args.training_end,
                    accel_channels=args.accel_channels,
                    feature_mapping={"ax": "ADC channel 0: simulated-X acceleration, not angular tip",
                                     "ay": "ADC channel 1: simulated-Y acceleration, not angular tilt"},
                    source_seed=42, calibration_seed=100, wfs_noise_seed=303,
                    decision_to_wfs_delay_frames=2, sensor_window_ns=[-500000, -125000],
                    sensor_publication_delay_ns=1000,
                    amplitude_scales="Reference producer mode amplitudes multiplied explicitly; ADC defaults unchanged")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = ["# accelSim → aoSim → DDSPC: simulation connection", "",
              f"**Transport: {args.transport}. Actual numerical implementations; no native application processes run.**", "",
              "| Source amplitude | Case | WFS RMS | Variance reduction | 28 Hz dB | 56 Hz dB |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in all_rows:
        report.append(f"| {row['source_scale']:g}× | {row['name']} | {row['rms']:.5f} | "
                      f"{100*row['variance_reduction']:.2f}% | {row['suppression_28_db']:.2f} | "
                      f"{row['suppression_56_db']:.2f} |")
    report += ["", "Variance reduction and line suppression are relative to disturbance-only within each amplitude.",
               "RMS is normalized to the configured 28-Hz pointing RMS, recorded in manifest.json; amplitudes "
               "and the earlier harness have different units and are not absolute-RMS comparisons.", ""]
    for config in configurations:
        scale = config["source_scale"]
        group = [r for r in all_rows if r["source_scale"] == scale]
        reduction = 100 * (1 - group[2]["rms"] / group[1]["rms"])
        report += [f"At {scale:g}×, RMS reduction from assistance relative to standard is **{reduction:.2f}%** "
                   "(negative means worse).", "",
                   f"![Complete scored interval, {scale:g}×](scale_{scale:g}/full_time_series.png)", "",
                   f"![WFS residuals, {scale:g}×](scale_{scale:g}/time_series.png)", "",
                   f"![Residual PSD, {scale:g}×](scale_{scale:g}/residual_psd.png)", ""]
    report += [("Adaptive training continues throughout control; late degradation is retained." if args.training_end is None else
                f"Coefficient updates stop at source frame {args.training_end}; fresh input, rank and model-age checks continue."),
               " Whole time series and first/last-second RMS are retained. No universal assisted-model benefit is claimed.", "",
               "The 1× case retains reference mode amplitudes and default ADC/noise settings. The 100× case "
               "explicitly increases simulated mode amplitudes; ADC calibration/noise remain unchanged. "
               "WFS noise and input normalization retain the same relative scale. Neither is instrument calibration.",
               "", "No generator pointing reaches DDSPC: only WFS, raw ADC and actually consumed correction "
               "inputs cross its interface. Disturbance and correction traverse separate aoSim DM delay channels.",
               "", "See [interface findings and exact remaining application wiring](../simulation_connection.md)."]
    (output / "README.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
