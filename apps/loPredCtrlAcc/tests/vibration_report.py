r"""\file vibration_report.py
\brief Reproducible SI-free spectral/RMS metrics and publication artifacts from offline traces only.
"""
import argparse
import csv
import json
import hashlib
import os
import platform
import socket
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ddspc-phase10-matplotlib")
import numpy as np
import scipy
from scipy import signal

LABELS = ("No control", "Perfect oracle", "Legacy core", "Standard causal", "Accel assisted")
COLORS = ("#555555", "#999999", "#d28d00", "#0072b2", "#009e73")
FS = 2000


def spectra(values):
    """Return mean two-axis one-sided Welch density; variance units per Hz, fixed Hann/overlap settings."""
    f, p = signal.welch(values, fs=FS, window="hann", nperseg=min(2048, len(values)),
                        noverlap=min(1024, len(values) // 2), detrend="constant", axis=0)
    return f, p.mean(axis=1)


def line_power(f, p, frequency):
    """Integrate a predeclared ±2 Hz line band; absent peaks remain noise-floor comparisons."""
    return float(p[np.abs(f - frequency) <= 2].sum() * (f[1] - f[0]))


def coherence(values, sensor):
    """Mean diagonal MSC in 10–250 Hz excluding ±3 Hz of the peaks; finite-segment bias is retained."""
    values = np.asarray(values)
    if not np.isfinite(sensor).all():
        return float("nan")
    f, c = signal.coherence(values, sensor, fs=FS, window="hann", nperseg=min(1024, len(values)),
                            noverlap=min(512, len(values) // 2), detrend="constant", axis=0)
    band = (f >= 10) & (f <= 250) & (abs(f - 28) > 3) & (abs(f - 56) > 3)
    return float(c[band].mean())


def pair(a, x, y):
    """Collect two explicit modal columns in their declared order."""
    return np.column_stack((a[x], a[y]))


def metrics(a, cfg):
    """Measure all rows, including holds and invalid models; never restrict residual scores to valid queries."""
    train, count = int(cfg["train"]), int(cfg["evaluate"])
    settle = min(1000, count // 4)
    measured = (a["frame"] >= train + settle)
    baseline = a[measured & (a["controller"] == 0)]
    f, original = spectra(pair(baseline, "yx", "yy"))
    source_coherence = coherence(pair(baseline, "yx", "yy"), pair(baseline, "ax", "ay"))
    rows = []
    for model in range(5):
        b = a[measured & (a["controller"] == model)]
        wfs, physical = pair(b, "yx", "yy"), pair(b, "rx", "ry")
        _, psd = spectra(wfs)
        pred_count = b["prediction_count"].sum()
        origin_delay = (b["frame"] - b["origin"] + int(cfg["dm_extra_frames"]) + 1) / FS
        active = b["valid"] != 0
        row = dict(scenario=cfg["scenario"], controller=model, name=LABELS[model], qrd=int(cfg["qrd"]),
                   frames=len(b), settling_frames=settle, measured_coherence=source_coherence,
                   mixture=float(cfg["coherence"]), rms=float(np.sqrt(np.mean(wfs ** 2))),
                   true_rms=float(np.sqrt(np.mean(physical ** 2))), variance=float(np.var(wfs, axis=0).mean()),
                   variance_reduction=1 - float(np.var(wfs, axis=0).mean()) /
                   float(np.var(pair(baseline, "yx", "yy"), axis=0).mean()),
                   ready_fraction=float(np.mean(b["ready"])), availability=float(np.mean(active)),
                   requested_rms=float(np.sqrt(np.mean(pair(b, "request_x", "request_y") ** 2))),
                   applied_rms=float(np.sqrt(np.mean(pair(b, "applied_x", "applied_y") ** 2))),
                   command_max=float(np.max(np.abs(pair(b, "request_x", "request_y")))),
                   prediction_rmse=float(np.sqrt(b["prediction_squared"].sum() / pred_count))
                   if pred_count else float("nan"),
                   origin_to_effect_ms=float(1000 * np.mean(origin_delay[active])) if active.any() else float("nan"),
                   evidence_age_max_ms=float(b["age_ns"].max() / 1e6),
                   overflows=int(b["overflows"].max()), invalid_changes=int(b["invalid_changes"].max()),
                   theta_norm_max=float(b["theta_norm"].max()), d_norm_max=float(b["d_norm"].max()))
        for hz in (28, 56):
            before, after = line_power(f, original, hz), line_power(f, psd, hz)
            row[f"suppression_{hz}_db"] = 10 * np.log10(max(before, 1e-30) / max(after, 1e-30))
        full = a[(a["controller"] == model)]
        ready = np.flatnonzero(full["ready"])
        row["first_ready_ms"] = float(ready[0] / FS * 1000) if len(ready) else float("nan")
        rows.append(row)
    return rows


def self_test():
    """Independent Parseval, line-integration, suppression-sign and coherence checks with analytic signals."""
    t = np.arange(16384) / FS
    x = np.column_stack((np.sin(2 * np.pi * 28 * t), np.sin(2 * np.pi * 56 * t)))
    f, p = spectra(x)
    assert abs(np.sum(p) * (f[1] - f[0]) - .5) < 1e-3
    assert abs(line_power(f, p, 28) - .25) < .003
    _, reduced = spectra(.1 * x)
    assert abs(10 * np.log10(line_power(f, p, 28) / line_power(f, reduced, 28)) - 20) < 1e-10
    rng = np.random.default_rng(19)
    a = rng.normal(size=(16384, 2))
    assert coherence(a, a) > .999999
    assert coherence(a, rng.normal(size=a.shape)) < .12
    print("Spectral power, two-axis RMS scaling, 20 dB suppression and coherence controls pass")


def figures(root, configs, selected, rows):
    """Create six standalone scientific figures, retaining unfavorable RLS and fault outcomes in the tables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120, "savefig.bbox": "tight"})

    def save(fig, name):
        fig.savefig(root / (name + ".png"))
        fig.savefig(root / (name + ".pdf"), metadata={"CreationDate": None})
        plt.close(fig)

    a, cfg = selected["broadband_qr"], configs["broadband_qr"]
    train = int(cfg["train"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, component, name in zip(axes, ("yx", "yy"), ("Tip", "Tilt")):
        for model in range(5):
            b = a[(a["controller"] == model) & (a["frame"] >= train + 2000) & (a["frame"] < train + 2400)]
            ax.plot((b["frame"] - train) / FS, b[component], label=LABELS[model], color=COLORS[model], alpha=.85)
        ax.set_ylabel(name + " residual (synthetic WFS units)")
        ax.grid(alpha=.25)
    axes[0].set_title("Offline broadband + 28/56 Hz: QR models, identical disturbance/noise")
    axes[0].legend(ncol=3)
    axes[1].set_xlabel("Time after calibration (s)")
    save(fig, "phase10-timeseries")

    fig, ax = plt.subplots(figsize=(10, 4))
    for model in range(5):
        b = a[(a["controller"] == model) & (a["frame"] >= train + 1000)]
        f, p = spectra(pair(b, "yx", "yy"))
        ax.semilogy(f, p, color=COLORS[model], label=LABELS[model])
    for hz in (28, 56): ax.axvline(hz, color="black", alpha=.25, linestyle=":")
    ax.set(xlim=(5, 300), xlabel="Frequency (Hz)", ylabel="Two-axis mean PSD (synthetic WFS units²/Hz)",
           title="Offline residual spectra; Hann Welch, 2048 points, 50% overlap")
    ax.legend(ncol=3); ax.grid(alpha=.25); save(fig, "phase10-psd")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, scenario, title in zip(axes, ("mixed_qr", "broadband_qr"), ("28 + 56 Hz", "Broadband + peaks")):
        for j, model in enumerate((2, 3, 4)):
            r = next(x for x in rows if x["scenario"] == scenario and x["controller"] == model)
            ax.bar(np.arange(2) + (j - 1) * .24, [r["suppression_28_db"], r["suppression_56_db"]],
                   width=.24, label=LABELS[model], color=COLORS[model])
        ax.set(xticks=(0, 1), xticklabels=("28 Hz", "56 Hz"), title=title); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("PSD line-band suppression vs no control (dB)")
    axes[1].legend(); fig.suptitle("Offline QR comparison: positive dB means lower residual power")
    save(fig, "phase10-suppression")

    fig, ax = plt.subplots(figsize=(8, 4))
    for model in (2, 3, 4):
        points = [r for r in rows if r["scenario"] in
                  ("coherence_0_qr", "coherence_1_qr", "coherence_2_qr", "coherence_3_qr", "broadband_qr")
                  and r["controller"] == model]
        points.sort(key=lambda r: r["measured_coherence"])
        ax.plot([r["measured_coherence"] for r in points], [r["rms"] for r in points], "o-",
                color=COLORS[model], label=LABELS[model])
    ax.set(xlabel="Mean sensor / no-control WFS MSC, 10–250 Hz excluding lines",
           ylabel="Measured two-axis residual RMS (synthetic WFS units)",
           title="Offline QR coherence sweep; finite-window MSC bias retained")
    ax.legend(); ax.grid(alpha=.25); save(fig, "phase10-coherence")

    fig, ax = plt.subplots(figsize=(8, 4))
    for model in (2, 3, 4):
        cases = ("broadband_qr", "latency_1_qr", "latency_2_qr", "latency_3_qr")
        points = [next(r for r in rows if r["scenario"] == name and r["controller"] == model) for name in cases]
        ax.plot(np.arange(4) * .5, [r["rms"] for r in points], "o-", color=COLORS[model], label=LABELS[model])
    ax.set(xlabel="Uncompensated additional DM transport delay (ms)",
           ylabel="Measured two-axis residual RMS (synthetic WFS units)",
           title="Offline latency mismatch; full reference law is unchanged")
    ax.legend(); ax.grid(alpha=.25); save(fig, "phase10-latency")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for scenario, style in (("broadband_qr", "-"), ("broadband", "--")):
        data = selected[scenario]
        for model in (2, 3, 4):
            b = data[data["controller"] == model]
            bins = np.arange(0, len(b) - 199, 200)
            rms = [np.sqrt(np.mean(pair(b[i:i + 200], "yx", "yy") ** 2)) for i in bins]
            axes[0].semilogy((bins + 100 - train) / FS, rms, style, color=COLORS[model],
                             label=LABELS[model] + (" QR" if style == "-" else " RLS"))
            axes[1].plot((bins + 100 - train) / FS,
                         [b[i:i + 200]["valid"].mean() for i in bins], style, color=COLORS[model])
    axes[0].set(ylabel="0.1-s residual RMS (synthetic units)", title="Offline convergence and availability; all samples retained")
    axes[1].set(xlabel="Time relative to calibration end (s)", ylabel="Valid prediction fraction", ylim=(-.05, 1.05))
    for ax in axes: ax.axvline(0, color="black", alpha=.4); ax.grid(alpha=.25)
    axes[0].legend(ncol=2); save(fig, "phase10-convergence")


def main():
    """Summarize full traces and export CSV/JSON plus six PNG/PDF figures; never run an application."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.input or not args.output:
        parser.error("input and output directories are required")
    args.output.mkdir(parents=True, exist_ok=True)
    configs = {r["scenario"]: r for r in csv.DictReader((args.input / "config.csv").open())}
    rows, selected = [], {}
    for name, cfg in configs.items():
        a = np.genfromtxt(args.input / (name + ".csv"), delimiter=",", names=True)
        assert len(a) == 5 * (int(cfg["train"]) + int(cfg["evaluate"]))
        assert np.isfinite(pair(a, "yx", "yy")).all()
        assert not a["invalid_changes"].any()
        rows.extend(metrics(a, cfg))
        if name in ("broadband", "broadband_qr"): selected[name] = a
    with (args.output / "phase10-metrics.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "phase10-config.json").write_text(json.dumps(configs, indent=2) + "\n")
    repo = Path(__file__).resolve().parents[3]
    paths = [Path(__file__), *(Path(__file__).parent.glob("vibration*.cpp")),
             Path(__file__).parent / "vibration.hpp", Path(__file__).parent / "Makefile.vibration"]
    def digest(path):
        """Hash an artifact in bounded chunks without retaining all raw traces in memory."""
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    manifest = dict(host=socket.gethostname(), platform=platform.platform(), python=platform.python_version(),
                    numpy=np.__version__, scipy=scipy.__version__,
                    head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip(),
                    sources={str(p.relative_to(repo)): digest(p) for p in paths},
                    traces={str(p.name): digest(p) for p in args.input.glob("*.csv")})
    (args.output / "phase10-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.skip_plots:
        figures(args.output, configs, selected, rows)
    print(f"Summarized {len(configs)} cases / {len(rows)} controller rows; plots={not args.skip_plots}")


if __name__ == "__main__":
    main()
